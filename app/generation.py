"""Generation: system prompt + LLM call + grounded-JSON contract + response cache.

Sections:
  1. Prompt        — PROMPT_VERSION, SYSTEM_PROMPT, context formatting.
  2. Backends      — openai-compatible (Ollama / Alem.ai / OpenAI) or offline stub.
  3. Parsing       — model output -> validated AnswerResponse (refuses on garbage).
  4. ResponseCache — full-answer cache with TTL + index_version invalidation.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any, Optional

from app.observability import log
from app.reliability import LLMUnavailableError, call_with_reliability
from app.schemas import AnswerResponse, RetrievedChunk, Source, settings

# ============================================================ 1. prompt

PROMPT_VERSION = "rag_v2"
REFUSAL = "I don't know from the provided context"

SYSTEM_PROMPT = """You are a grounded question-answering assistant for the Labor Code and
the Tax Code of the Republic of Kazakhstan.

Rules:
1. Answer ONLY using facts found in the CONTEXT below. Never use outside knowledge.
2. Answer in the language of the question (Russian questions -> Russian answer).
3. Quote the concrete provision: the figure, the term, the deadline. Do not paraphrase
   away the number. If the article states "40 часов", the answer says "40 часов".
4. In "sources", list the chunk_id of every chunk you used, copied EXACTLY as given in
   the header of that chunk. If you cannot copy it exactly, cite its number ([1], [2]).
5. If the context does not contain the answer, reply with exactly:
   {"answer": "I don't know from the provided context", "sources": [], "confidence": 0.0, "used_context": false}
   Do this even if you know the answer from memory — an ungrounded answer is a failure.
6. "confidence" in [0,1] is how well the context supports your answer.
7. Reply with a SINGLE JSON object and nothing else: no markdown fence, no prose, no
   reasoning outside the JSON.

Response schema:
{"answer": <string>, "sources": [<chunk_id or index>, ...], "confidence": <float 0..1>, "used_context": <bool>}
"""


# Article-aware chunks are whole legal articles: a few are enormous (10k+ chars).
# Five of those blow past the model's context window, and the answer degrades or the
# call times out. Cap each chunk; the answer to a lookup question is near the top of
# the article, and the full text stays available via /ask -> sources -> chunk_id.
MAX_CHUNK_CHARS_IN_PROMPT = 2500


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context retrieved)"
    blocks = []
    for i, rc in enumerate(chunks, 1):
        c = rc.chunk
        header = f"[{i}] chunk_id={c.chunk_id} source={c.source_file}"
        if c.section_title:
            header += f" section={c.section_title}"
        text = c.text
        if len(text) > MAX_CHUNK_CHARS_IN_PROMPT:
            text = text[:MAX_CHUNK_CHARS_IN_PROMPT] + " …[обрезано]"
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    user = f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {question}\n\nJSON answer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ============================================================ 2. backends


def _call_openai(messages: list[dict[str, str]]) -> tuple[str, int]:
    """Any OpenAI-compatible endpoint: base_url + key + model come from settings.

    Two hard-won details:
      - Not every OpenAI-compatible server implements `response_format=json_object`
        (Ollama and some hosted gateways reject it). We ask for it, and on rejection
        retry once without it — the prompt already pins the JSON contract.
      - Any transport/API failure is re-raised as LLMUnavailableError so the retry
        layer and the endpoint can turn it into a clean 502, never a 500 stack trace.
    """
    from openai import OpenAI  # lazy

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0.0,
    }
    try:
        resp = client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}
        )
    except Exception as exc:
        if not _is_response_format_error(exc):
            raise LLMUnavailableError(f"llm call failed: {type(exc).__name__}: {exc}") from exc
        log("llm_response_format_unsupported", model=settings.llm_model)
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc2:
            raise LLMUnavailableError(
                f"llm call failed: {type(exc2).__name__}: {exc2}"
            ) from exc2

    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    total = getattr(usage, "total_tokens", 0) if usage else 0
    return text, int(total or 0)


def _is_response_format_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "response_format" in msg or "json_object" in msg


def _call_stub(messages: list[dict[str, str]], chunks: list[RetrievedChunk]) -> tuple[str, int]:
    """Deterministic offline backend so the prompt regression tests are hermetic:
    grounds on the top chunk if its score is positive, otherwise refuses."""
    if not chunks or chunks[0].score <= 0:
        payload = {"answer": REFUSAL, "sources": [], "confidence": 0.0, "used_context": False}
    else:
        top = chunks[0].chunk
        payload = {
            "answer": " ".join(top.text.split())[:240],
            "sources": [top.chunk_id],
            "confidence": round(min(0.99, 0.5 + chunks[0].score / 2), 2),
            "used_context": True,
        }
    approx_tokens = sum(len(m["content"].split()) for m in messages)
    return json.dumps(payload, ensure_ascii=False), approx_tokens


# ============================================================ 3. parsing

# Reasoning models (qwen3 etc.) emit <think>…</think> before the answer, and many
# models wrap JSON in a markdown fence. A greedy {.*} over that grabs prose braces and
# blows up, which used to degrade a perfectly good answer into a refusal.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _iter_json_candidates(text: str):
    """Yield balanced {...} blocks, last one first (the final answer usually wins)."""
    starts: list[int] = []
    spans: list[tuple[int, int]] = []
    for i, ch in enumerate(text):
        if ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            start = starts.pop()
            if not starts:  # closed a top-level object
                spans.append((start, i + 1))
    for start, end in reversed(spans):
        yield text[start:end]


def _parse(raw: str) -> dict[str, Any]:
    text = _THINK_RE.sub("", raw or "")
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    for candidate in _iter_json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "answer" in data:
            return data
    raise ValueError("no JSON object with an 'answer' field in model output")


def _resolve_source(cite: Any, retrieved: list[RetrievedChunk]) -> RetrievedChunk | None:
    """Map one citation from the model onto a chunk we actually retrieved.

    Models rarely copy a 48-char chunk_id byte-for-byte: they truncate it, cite the
    index "[2]", or cite the article ("Статья 68"). Exact-match-only silently produced
    an empty `sources` list on answers that were in fact correctly grounded, which made
    the whole system look like it had found nothing. We stay strict about grounding —
    a citation must resolve to a retrieved chunk — but tolerant about its format.
    """
    cite = str(cite).strip()
    if not cite:
        return None
    # 1. exact chunk_id
    for rc in retrieved:
        if rc.chunk.chunk_id == cite:
            return rc
    # 2. positional citation: "1", "[2]"
    digits = cite.strip("[]() ")
    if digits.isdigit() and 1 <= int(digits) <= len(retrieved):
        return retrieved[int(digits) - 1]
    # 3. partial chunk_id (model truncated it)
    for rc in retrieved:
        if cite in rc.chunk.chunk_id or rc.chunk.chunk_id in cite:
            return rc
    # 4. section title / article number ("Статья 68", "Статья 68. Нормальная...")
    for rc in retrieved:
        title = rc.chunk.section_title or ""
        if title and (cite in title or title in cite):
            return rc
    return None


def _to_response(
    data: dict[str, Any], retrieved: list[RetrievedChunk], request_id: str
) -> AnswerResponse:
    """Grounding rule: a source must resolve to a chunk we actually retrieved, so the
    model cannot invent a citation. Format of the citation, though, is forgiving."""
    sources: list[Source] = []
    seen: set[str] = set()
    for cite in data.get("sources", []) or []:
        rc = _resolve_source(cite, retrieved)
        if rc and rc.chunk.chunk_id not in seen:
            seen.add(rc.chunk.chunk_id)
            sources.append(
                Source(
                    chunk_id=rc.chunk.chunk_id,
                    source_file=rc.chunk.source_file,
                    section_title=rc.chunk.section_title,
                    page=rc.chunk.page,
                    score=rc.score,
                )
            )

    answer_text = str(data.get("answer", REFUSAL))
    is_refusal = answer_text.strip() == REFUSAL or not answer_text.strip()

    # The model answered from the context but cited nothing resolvable: attribute the
    # answer to the top-ranked chunk rather than shipping a groundless answer with an
    # empty source list. Logged, so an unreliable model shows up in the logs.
    if not sources and not is_refusal and retrieved:
        log("llm_unresolved_citation", cited=data.get("sources"), fallback_to_top1=True)
        top = retrieved[0]
        sources.append(
            Source(
                chunk_id=top.chunk.chunk_id,
                source_file=top.chunk.source_file,
                section_title=top.chunk.section_title,
                page=top.chunk.page,
                score=top.score,
            )
        )
    return AnswerResponse(
        answer=str(data.get("answer", REFUSAL)),
        sources=sources,
        confidence=float(data.get("confidence", 0.0)),
        used_context=bool(sources) and not is_refusal,
        request_id=request_id,
        prompt_version=PROMPT_VERSION,
        model_name=settings.llm_model if settings.llm_backend != "stub" else "stub",
        cached=False,
    )


def generate(
    question: str, retrieved: list[RetrievedChunk], request_id: str
) -> tuple[AnswerResponse, int]:
    """Public entry point. The LLM call is guarded by timeout + retry (reliability.py)."""
    messages = build_messages(question, retrieved)

    def _invoke() -> tuple[str, int]:
        if settings.llm_backend == "stub":
            return _call_stub(messages, retrieved)
        return _call_openai(messages)

    raw, tokens = call_with_reliability(
        _invoke, timeout_s=settings.llm_timeout_s, max_retries=settings.llm_max_retries
    )
    try:
        data = _parse(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        # A malformed model reply degrades to an honest refusal, never a 500.
        log("llm_parse_error", error=str(exc), raw_preview=raw[:200])
        data = {"answer": REFUSAL, "sources": [], "confidence": 0.0, "used_context": False}
    return _to_response(data, retrieved, request_id), tokens


# ============================================================ 4. response cache


class ResponseCache:
    """Key = sha256(question + index_version + top_k).

    Invalidation (documented in ADR + README):
      - index_version salt: re-indexing changes it, so stale answers can never hit.
      - TTL: entries expire after RESPONSE_CACHE_TTL_S.
      - LRU bound: at most RESPONSE_CACHE_MAX entries.
    """

    def __init__(self, index_version: str, ttl_s: int, max_size: int) -> None:
        self.index_version = index_version
        self.ttl_s = ttl_s
        self.max_size = max_size
        self._store: "OrderedDict[str, tuple[float, AnswerResponse]]" = OrderedDict()

    def _key(self, question: str, top_k: int) -> str:
        raw = f"{question.strip().lower()}\x00{self.index_version}\x00{top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, question: str, top_k: int) -> Optional[AnswerResponse]:
        key = self._key(question, top_k)
        item = self._store.get(key)
        if item is None:
            return None
        ts, resp = item
        if time.time() - ts > self.ttl_s:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return resp

    def put(self, question: str, top_k: int, resp: AnswerResponse) -> None:
        key = self._key(question, top_k)
        self._store[key] = (time.time(), resp)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
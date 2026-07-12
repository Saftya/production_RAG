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
from app.reliability import call_with_reliability
from app.schemas import AnswerResponse, RetrievedChunk, Source, settings

# ============================================================ 1. prompt

PROMPT_VERSION = "rag_v1"
REFUSAL = "I don't know from the provided context"

SYSTEM_PROMPT = """You are a grounded question-answering assistant for an internal document base.

Rules:
1. Answer ONLY using facts found in the CONTEXT below. Do not use outside knowledge.
2. If the context does not contain the answer, you MUST reply with exactly:
   {"answer": "I don't know from the provided context", "sources": [], "confidence": 0.0, "used_context": false}
3. Cite the chunk_id of every source you used in "sources".
4. "confidence" is your calibrated certainty in [0,1] that the answer is supported by the context.
5. Reply with a SINGLE JSON object and nothing else. No markdown, no prose outside JSON.

Response schema:
{"answer": <string>, "sources": [<chunk_id string>, ...], "confidence": <float 0..1>, "used_context": <bool>}
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context retrieved)"
    blocks = []
    for rc in chunks:
        c = rc.chunk
        header = f"[chunk_id={c.chunk_id}] source={c.source_file}"
        if c.section_title:
            header += f" section={c.section_title}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    user = f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {question}\n\nJSON answer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ============================================================ 2. backends


def _call_openai(messages: list[dict[str, str]]) -> tuple[str, int]:
    """Any OpenAI-compatible endpoint: base_url + key + model come from settings."""
    from openai import OpenAI  # lazy

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    total = getattr(usage, "total_tokens", 0) if usage else 0
    return text, int(total or 0)


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

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> dict[str, Any]:
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError("no JSON object in model output")
    return json.loads(match.group(0))


def _to_response(
    data: dict[str, Any], retrieved: list[RetrievedChunk], request_id: str
) -> AnswerResponse:
    """Sources are resolved against what we actually retrieved: a chunk_id the model
    invented cannot enter the response."""
    by_id = {rc.chunk.chunk_id: rc for rc in retrieved}
    sources: list[Source] = []
    for cid in data.get("sources", []) or []:
        rc = by_id.get(cid)
        if rc:
            sources.append(
                Source(
                    chunk_id=rc.chunk.chunk_id,
                    source_file=rc.chunk.source_file,
                    section_title=rc.chunk.section_title,
                    page=rc.chunk.page,
                    score=rc.score,
                )
            )
    return AnswerResponse(
        answer=str(data.get("answer", REFUSAL)),
        sources=sources,
        confidence=float(data.get("confidence", 0.0)),
        used_context=bool(data.get("used_context", bool(sources))),
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

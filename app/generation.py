"""Answer generation: wrap an LLM behind a grounded-JSON contract.

Backends:
  - openai: any OpenAI-compatible endpoint (Ollama /v1, Alem.ai, OpenAI). No SDK
    lock-in; base_url + api_key + model come from config.
  - stub:   deterministic, offline. Grounds its answer in the top chunk or refuses.
            Used by the prompt regression tests so CI needs no live model.

`generate()` returns (AnswerResponse, token_usage). The LLM call is guarded by
timeout + retry from reliability.py.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.config import settings
from app.observability import log
from app.prompts import PROMPT_VERSION, build_messages
from app.reliability import call_with_reliability
from app.schemas import AnswerResponse, RetrievedChunk, Source

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
REFUSAL = "I don't know from the provided context"


# ------------------------------------------------------------- backends


def _call_openai(messages: list[dict[str, str]]) -> tuple[str, int]:
    from openai import OpenAI  # lazy

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    tokens = getattr(resp, "usage", None)
    total = getattr(tokens, "total_tokens", 0) if tokens else 0
    return text, int(total or 0)


def _call_stub(messages: list[dict[str, str]], chunks: list[RetrievedChunk]) -> tuple[str, int]:
    """Deterministic offline backend. Answers from the top chunk if it looks
    relevant (positive score), else refuses. Keeps regression tests hermetic."""
    if not chunks or chunks[0].score <= 0:
        payload = {"answer": REFUSAL, "sources": [], "confidence": 0.0, "used_context": False}
    else:
        top = chunks[0].chunk
        snippet = " ".join(top.text.split())[:240]
        payload = {
            "answer": snippet,
            "sources": [top.chunk_id],
            "confidence": round(min(0.99, 0.5 + chunks[0].score / 2), 2),
            "used_context": True,
        }
    approx_tokens = sum(len(m["content"].split()) for m in messages)
    return json.dumps(payload, ensure_ascii=False), approx_tokens


# ------------------------------------------------------------- parsing


def _parse(raw: str) -> dict[str, Any]:
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError("no JSON object in model output")
    return json.loads(match.group(0))


def _to_response(
    data: dict[str, Any],
    retrieved: list[RetrievedChunk],
    request_id: str,
    tokens: int,
) -> AnswerResponse:
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


# ------------------------------------------------------------- public API


def generate(
    question: str, retrieved: list[RetrievedChunk], request_id: str
) -> tuple[AnswerResponse, int]:
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
        log("llm_parse_error", error=str(exc), raw_preview=raw[:200])
        data = {"answer": REFUSAL, "sources": [], "confidence": 0.0, "used_context": False}
    return _to_response(data, retrieved, request_id, tokens), tokens

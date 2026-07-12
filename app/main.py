"""FastAPI service.

Endpoints:
  GET  /health   liveness  — process is up.
  GET  /ready    readiness — index + embedder loaded (and LLM configured).
  POST /ask      grounded RAG answer with sources + confidence.
  GET  /metrics  Prometheus text (or ?format=json for a JSON snapshot).

Per-request: UUID4 request_id (from X-Request-ID header or generated), staged
latency timing, structured JSON logs, response cache, and graceful degradation
if retrieval is unavailable.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from app.generation import PROMPT_VERSION, REFUSAL, ResponseCache, generate
from app.observability import (
    get_request_id,
    log,
    metrics,
    new_request_id,
    set_request_id,
)
from app.reliability import LLMTimeoutError, LLMUnavailableError
from app.retrieval import FaissStore, Retriever, build_embedder
from app.schemas import (
    AnswerResponse,
    AskRequest,
    HealthResponse,
    ReadyResponse,
    settings,
)

state: dict = {"retriever": None, "cache": None, "ready": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("startup_begin", index_dir=settings.index_dir, embedding_backend=settings.embedding_backend)
    try:
        if not FaissStore.exists(settings.index_dir):
            raise FileNotFoundError(f"no index at {settings.index_dir} — run `make index`")
        embedder = build_embedder()
        store = FaissStore.load(settings.index_dir)
        retriever = Retriever(embedder, store, mode=settings.retrieval_mode)
        cache = ResponseCache(
            store.index_version, settings.response_cache_ttl_s, settings.response_cache_max
        )
        state.update(retriever=retriever, cache=cache, ready=True)
        log("startup_ready", chunks=len(store.chunks), index_version=store.index_version)
    except Exception as exc:  # stay alive but not ready
        log("startup_failed", error=str(exc))
        state["ready"] = False
    yield
    log("shutdown")


app = FastAPI(title="Production RAG", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_mw(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or new_request_id()
    set_request_id(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ------------------------------------------------------------- ops endpoints


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready() -> JSONResponse:
    r = state["ready"]
    body = ReadyResponse(
        ready=r,
        index_loaded=state["retriever"] is not None,
        embedder_loaded=state["retriever"] is not None,
        llm_configured=settings.llm_backend in ("openai", "stub"),
    )
    return JSONResponse(status_code=200 if r else 503, content=body.model_dump())


@app.get("/metrics")
def get_metrics(format: str = "prometheus") -> Response:
    if format == "json":
        return JSONResponse(metrics.snapshot())
    return PlainTextResponse(metrics.prometheus())


# ------------------------------------------------------------- main endpoint


@app.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest) -> JSONResponse:
    rid = get_request_id()
    t0 = time.perf_counter()
    top_k = req.top_k or settings.top_k
    stages = {"embedding_retrieval_ms": 0.0, "generation_ms": 0.0}
    retriever: Retriever = state["retriever"]
    cache: ResponseCache = state["cache"]

    # graceful degradation: index not loaded
    if not state["ready"] or retriever is None:
        resp = AnswerResponse(
            answer="I couldn't search the documents right now.",
            sources=[], confidence=0.0, used_context=False, request_id=rid,
            prompt_version=PROMPT_VERSION, model_name="n/a", degraded=True,
        )
        log("ask_degraded", question=req.question, reason="index_not_ready")
        return JSONResponse(status_code=503, content=resp.model_dump())

    # response cache
    cached = cache.get(req.question, top_k)
    if cached is not None:
        total_ms = (time.perf_counter() - t0) * 1000
        metrics.observe(total_ms, error=False, tokens=0, top1_score=None, cache_hit=True)
        out = cached.model_copy(update={"request_id": rid, "cached": True})
        log("ask_cache_hit", question=req.question, latency_ms=round(total_ms, 1))
        return JSONResponse(out.model_dump())

    try:
        tr = time.perf_counter()
        retrieved, retrieval_cache_hit = retriever.retrieve(req.question, top_k, req.filters)
        stages["embedding_retrieval_ms"] = round((time.perf_counter() - tr) * 1000, 1)

        tg = time.perf_counter()
        answer, tokens = generate(req.question, retrieved, rid)
        stages["generation_ms"] = round((time.perf_counter() - tg) * 1000, 1)
    except LLMTimeoutError:
        total_ms = (time.perf_counter() - t0) * 1000
        metrics.observe(total_ms, error=True, tokens=0, top1_score=None, cache_hit=False)
        log("ask_timeout", question=req.question, error="llm_timeout")
        return JSONResponse(
            status_code=504, content={"error": "llm_timeout", "request_id": rid}
        )
    except LLMUnavailableError:
        total_ms = (time.perf_counter() - t0) * 1000
        metrics.observe(total_ms, error=True, tokens=0, top1_score=None, cache_hit=False)
        log("ask_llm_unavailable", question=req.question, error="llm_unavailable")
        return JSONResponse(
            status_code=502, content={"error": "llm_unavailable", "request_id": rid}
        )
    except Exception as exc:  # last line of defence: never leak a 500 + stack trace
        total_ms = (time.perf_counter() - t0) * 1000
        metrics.observe(total_ms, error=True, tokens=0, top1_score=None, cache_hit=False)
        log(
            "ask_internal_error",
            question=req.question,
            error=type(exc).__name__,
            detail=str(exc)[:200],
        )
        return JSONResponse(
            status_code=502, content={"error": "internal_error", "request_id": rid}
        )

    cache.put(req.question, top_k, answer)
    total_ms = (time.perf_counter() - t0) * 1000
    top1 = retrieved[0].score if retrieved else None
    metrics.observe(total_ms, error=False, tokens=tokens, top1_score=top1, cache_hit=False)

    log(
        "ask_ok",
        question=req.question,
        retrieved_chunk_ids=[rc.chunk.chunk_id for rc in retrieved],
        prompt_version=PROMPT_VERSION,
        model_name=answer.model_name,
        latency_ms_by_stage={**stages, "total_ms": round(total_ms, 1)},
        token_usage=tokens,
        answer_length=len(answer.answer),
        cache_hit_bool=False,
        retrieval_cache_hit=retrieval_cache_hit,
        error_bool=False,
        refused=answer.answer == REFUSAL,
    )
    return JSONResponse(answer.model_dump())
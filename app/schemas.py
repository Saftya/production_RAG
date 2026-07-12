"""Pydantic contracts + runtime configuration.

Everything the service passes around is typed here:
  - Settings   : config read from environment (.env), single source of truth.
  - Chunk      : one indexed unit of the corpus + its metadata.
  - AskRequest : validated API input (min 3 / max 500 chars -> HTTP 422).
  - AnswerResponse : the grounded JSON contract the LLM must conform to.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env into the process environment before Settings reads it. Real env
# vars still win (load_dotenv default: override=False) — needed so
# tests/conftest.py's os.environ overrides, set before this module is
# imported, aren't clobbered by .env.
load_dotenv()

# ============================================================ configuration


def _get(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    # --- paths ---
    data_dir: str = _get("DATA_DIR", "data")
    index_dir: str = _get("INDEX_DIR", "data/index")
    chunks_path: str = _get("CHUNKS_PATH", "data/chunks.jsonl")
    embed_cache_dir: str = _get("EMBED_CACHE_DIR", "data/.embed_cache")

    # --- embeddings ---
    # "st" = real sentence-transformers | "hash" = offline deterministic (CI/tests)
    embedding_backend: str = _get("EMBEDDING_BACKEND", "st")
    embedding_model: str = _get(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    preprocessing_version: str = _get("PREPROCESSING_VERSION", "v1")
    embedding_dim: int = int(_get("EMBEDDING_DIM", "384"))

    # --- retrieval ---
    top_k: int = int(_get("TOP_K", "5"))
    retrieval_mode: str = _get("RETRIEVAL_MODE", "hybrid")  # "vector" | "hybrid"
    rrf_k: int = int(_get("RRF_K", "60"))
    # 120: below this, a "Статья N. Title" match is a table-of-contents stub with
    # no body (measured empirically: ~40% of raw chunks on the Labor/Tax Code PDFs),
    # not an answerable article. Filtering them lifted hybrid recall@5 0.88 -> 0.92.
    min_chunk_chars: int = int(_get("MIN_CHUNK_CHARS", "120"))

    # --- generation (OpenAI-compatible: Ollama / Alem.ai / OpenAI) ---
    llm_backend: str = _get("LLM_BACKEND", "openai")  # "openai" | "stub"
    llm_base_url: str = _get("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = _get("LLM_API_KEY", "ollama")
    llm_model: str = _get("LLM_MODEL", "llama3.2:3b")

    # --- reliability ---
    llm_timeout_s: float = float(_get("LLM_TIMEOUT_S", "30"))
    llm_max_retries: int = int(_get("LLM_MAX_RETRIES", "3"))

    # --- caching ---
    response_cache_ttl_s: int = int(_get("RESPONSE_CACHE_TTL_S", "3600"))
    response_cache_max: int = int(_get("RESPONSE_CACHE_MAX", "512"))

    # --- observability ---
    metrics_window: int = int(_get("METRICS_WINDOW", "100"))


settings = Settings()


# ============================================================ internal types


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    page: Optional[int] = None
    section_title: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    document_version: str = "v1"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


# ============================================================ API contracts


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    filters: Optional[dict[str, Any]] = None


class Source(BaseModel):
    chunk_id: str
    source_file: str
    section_title: Optional[str] = None
    page: Optional[int] = None
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    used_context: bool
    request_id: str
    prompt_version: str
    model_name: str
    cached: bool = False
    degraded: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    index_loaded: bool
    embedder_loaded: bool
    llm_configured: bool

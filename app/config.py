"""Central configuration. All values come from environment (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass


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
    # backend: "st" (real sentence-transformers) | "hash" (offline deterministic, for CI/tests)
    embedding_backend: str = _get("EMBEDDING_BACKEND", "st")
    embedding_model: str = _get(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    preprocessing_version: str = _get("PREPROCESSING_VERSION", "v1")
    embedding_dim: int = int(_get("EMBEDDING_DIM", "384"))

    # --- retrieval ---
    top_k: int = int(_get("TOP_K", "5"))
    # mode: "vector" | "hybrid" (BM25 + vector with RRF)
    retrieval_mode: str = _get("RETRIEVAL_MODE", "hybrid")
    rrf_k: int = int(_get("RRF_K", "60"))

    # --- generation ---
    # backend: "openai" (OpenAI-compatible: Ollama/Alem/OpenAI) | "stub" (offline, for tests)
    llm_backend: str = _get("LLM_BACKEND", "openai")
    llm_base_url: str = _get("LLM_BASE_URL", "http://localhost:11434/v1")  # Ollama default
    llm_api_key: str = _get("LLM_API_KEY", "ollama")
    llm_model: str = _get("LLM_MODEL", "llama3.2:3b")
    prompt_version: str = "rag_v1"

    # --- reliability ---
    llm_timeout_s: float = float(_get("LLM_TIMEOUT_S", "30"))
    llm_max_retries: int = int(_get("LLM_MAX_RETRIES", "3"))

    # --- caching ---
    response_cache_ttl_s: int = int(_get("RESPONSE_CACHE_TTL_S", "3600"))
    response_cache_max: int = int(_get("RESPONSE_CACHE_MAX", "512"))

    # --- observability ---
    metrics_window: int = int(_get("METRICS_WINDOW", "100"))


settings = Settings()

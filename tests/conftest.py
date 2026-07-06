"""Shared test fixtures.

Forces the offline backends (hash embedder + stub LLM) and builds a small FAISS
index from the committed sample corpus into a temp dir BEFORE any app module is
imported, so the whole suite runs in CI without network or a live model.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="rag_test_")

# --- must be set before importing app.config ---
os.environ.update(
    EMBEDDING_BACKEND="hash",
    LLM_BACKEND="stub",
    RETRIEVAL_MODE="hybrid",
    INDEX_DIR=str(Path(_TMP) / "index"),
    CHUNKS_PATH=str(Path(_TMP) / "chunks.jsonl"),
    EMBED_CACHE_DIR=str(Path(_TMP) / "embed_cache"),
)


@pytest.fixture(scope="session", autouse=True)
def _build_index():
    from app.config import settings
    from app.embeddings import build_embedder
    from app.ingestion import build_chunks, write_chunks_jsonl
    from app.vectorstore import FaissStore

    chunks = build_chunks(str(ROOT / "data" / "raw"), strategy="section")
    write_chunks_jsonl(chunks, settings.chunks_path)
    embedder = build_embedder()
    vectors = embedder.encode([c.text for c in chunks])
    store = FaissStore.build(vectors, chunks, embedder.model_name, settings.preprocessing_version)
    store.save(settings.index_dir)
    yield


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def retriever():
    from app.config import settings
    from app.embeddings import build_embedder
    from app.retrieval import Retriever
    from app.vectorstore import FaissStore

    store = FaissStore.load(settings.index_dir)
    return Retriever(build_embedder(), store, mode="hybrid")

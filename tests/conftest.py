"""Shared fixtures.

The suite is hermetic: it builds a small FAISS index from `tests/fixtures/corpus`
(committed, tiny) with the offline backends, so CI needs no network, no torch and
no copy of the real corpus. The REAL recall number for the README/defense comes
from `make eval` against `data/index`, not from here.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
_TMP = tempfile.mkdtemp(prefix="rag_test_")

# Must be set before app.schemas (and therefore Settings) is imported.
# setdefault, so `EMBEDDING_BACKEND=st pytest` can still exercise the real model.
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("LLM_BACKEND", "stub")
os.environ.setdefault("RETRIEVAL_MODE", "hybrid")
os.environ.update(
    INDEX_DIR=str(Path(_TMP) / "index"),
    CHUNKS_PATH=str(Path(_TMP) / "chunks.jsonl"),
    EMBED_CACHE_DIR=str(Path(_TMP) / "embed_cache"),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evaluation"))


@pytest.fixture(scope="session")
def gt_path() -> str:
    return str(FIXTURES / "ground_truth.jsonl")


@pytest.fixture(scope="session", autouse=True)
def _build_index():
    from build_index import build_chunks, write_chunks_jsonl  # scripts/build_index.py

    from app.retrieval import FaissStore, build_embedder
    from app.schemas import settings

    chunks = build_chunks(str(FIXTURES / "corpus"), strategy="section")
    write_chunks_jsonl(chunks, settings.chunks_path)

    embedder = build_embedder()
    vectors = embedder.encode([c.text for c in chunks])
    store = FaissStore.build(
        vectors, chunks, embedder.model_name, settings.preprocessing_version
    )
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
    from app.retrieval import FaissStore, Retriever, build_embedder
    from app.schemas import settings

    store = FaissStore.load(settings.index_dir)
    return Retriever(build_embedder(), store, mode="hybrid")

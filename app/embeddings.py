"""Embedding backends behind one interface.

- STEmbedder: real Sentence-BERT (production / real recall numbers).
- HashEmbedder: deterministic, dependency-free, offline. Used in CI/tests so the
  suite runs without downloading a model. NOT for production recall reporting.

Both return L2-normalized float32 vectors so FAISS inner-product == cosine.
`build_embedder()` is the single factory the rest of the app depends on.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from app.config import settings
from app.embedding_cache import EmbeddingCache


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder(Protocol):
    dim: int
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Bag-of-hashed-token vector. Purely deterministic, no network, no torch."""

    def __init__(self, dim: int = 384, model_name: str = "hash-embedder-v1") -> None:
        self.dim = dim
        self.model_name = model_name

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
            v[(h // self.dim) % self.dim] += 0.5
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.vstack([self._vec(t) for t in texts]))


class STEmbedder:
    """sentence-transformers wrapper (lazy import so the hash backend needs no torch)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(mat)


class CachedEmbedder:
    """Wraps any Embedder with an on-disk cache; only cache misses are encoded."""

    def __init__(self, base: Embedder, cache: EmbeddingCache) -> None:
        self.base = base
        self.cache = cache
        self.dim = base.dim
        self.model_name = base.model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        to_encode: list[str] = []
        positions: list[int] = []
        for i, t in enumerate(texts):
            cached = self.cache.get(t)
            if cached is not None:
                out[i] = cached
            else:
                to_encode.append(t)
                positions.append(i)
        if to_encode:
            fresh = self.base.encode(to_encode)
            for pos, text, vec in zip(positions, to_encode, fresh):
                self.cache.put(text, vec)
                out[pos] = vec
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(out).astype(np.float32)

    def cache_stats(self) -> dict[str, int]:
        return self.cache.stats()


def build_embedder(use_cache: bool = True) -> CachedEmbedder:
    if settings.embedding_backend == "hash":
        base: Embedder = HashEmbedder(dim=settings.embedding_dim)
    else:
        base = STEmbedder(settings.embedding_model)
    cache = EmbeddingCache(
        settings.embed_cache_dir, base.model_name, settings.preprocessing_version
    )
    return CachedEmbedder(base, cache)

"""Retrieval stack: embeddings (+ on-disk cache) -> FAISS -> BM25 -> RRF -> Retriever.

Sections in this file:
  1. EmbeddingCache  — on-disk cache keyed by sha256(text + model + preproc_version).
  2. Embedders       — STEmbedder (production) / HashEmbedder (offline CI) / CachedEmbedder.
  3. FaissStore      — persisted vector index (IndexFlatIP over L2-normalized vecs = cosine).
  4. BM25 + RRF      — lexical ranking fused with the vector ranking.
  5. Retriever       — the public entry point, with an in-memory retrieval cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Any, Optional, Protocol

import numpy as np

from app.schemas import Chunk, RetrievedChunk, settings

# ============================================================ 1. embedding cache


class EmbeddingCache:
    """Key = sha256(text + model_name + preprocessing_version); one .npy per key.

    Required by the spec: re-indexing during ablation must not re-embed the corpus.
    """

    def __init__(self, cache_dir: str, model_name: str, preprocessing_version: str) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.preprocessing_version = preprocessing_version
        os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        payload = f"{text}\x00{self.model_name}\x00{self.preprocessing_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        shard = os.path.join(self.cache_dir, key[:2])  # avoid one huge flat dir
        os.makedirs(shard, exist_ok=True)
        return os.path.join(shard, f"{key}.npy")

    def get(self, text: str) -> np.ndarray | None:
        path = self._path(self._key(text))
        if os.path.exists(path):
            self.hits += 1
            return np.load(path)
        self.misses += 1
        return None

    def put(self, text: str, vec: np.ndarray) -> None:
        np.save(self._path(self._key(text)), vec.astype(np.float32))

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


# ============================================================ 2. embedders


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize so FAISS inner product == cosine similarity."""
    mat = mat.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder(Protocol):
    dim: int
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens. No torch, no network -> used in CI.

    NOT for reporting recall: it has no semantics, only lexical overlap.
    """

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
    """Sentence-BERT. Lazy import so the hash backend never needs torch."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(mat)


class CachedEmbedder:
    """Wraps an Embedder with the on-disk cache: only cache misses are encoded."""

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


def build_embedder() -> CachedEmbedder:
    """Single factory used by the service, build_index and the evaluation."""
    if settings.embedding_backend == "hash":
        base: Embedder = HashEmbedder(dim=settings.embedding_dim)
    else:
        base = STEmbedder(settings.embedding_model)
    cache = EmbeddingCache(
        settings.embed_cache_dir, base.model_name, settings.preprocessing_version
    )
    return CachedEmbedder(base, cache)


# ============================================================ 3. FAISS store

META_FILE = "meta.json"
CHUNKS_FILE = "chunks.jsonl"
INDEX_FILE = "index.faiss"


def _index_version(model: str, prep: str, count: int) -> str:
    return hashlib.sha1(f"{model}:{prep}:{count}".encode()).hexdigest()[:12]


class FaissStore:
    """Persisted vector index. The service loads it at startup and NEVER re-embeds."""

    def __init__(self, index: Any, chunks: list[Chunk], meta: dict[str, Any]) -> None:
        self.index = index
        self.chunks = chunks
        self.meta = meta

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        chunks: list[Chunk],
        embedding_model: str,
        preprocessing_version: str,
    ) -> "FaissStore":
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # exact search; corpus is small enough
        index.add(embeddings.astype(np.float32))
        meta = {
            "embedding_model": embedding_model,
            "dim": dim,
            "count": len(chunks),
            "preprocessing_version": preprocessing_version,
            "index_version": _index_version(embedding_model, preprocessing_version, len(chunks)),
        }
        return cls(index, chunks, meta)

    def save(self, index_dir: str) -> None:
        import faiss

        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, INDEX_FILE))
        with open(os.path.join(index_dir, CHUNKS_FILE), "w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(ch.model_dump_json() + "\n")
        with open(os.path.join(index_dir, META_FILE), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "FaissStore":
        import faiss

        index = faiss.read_index(os.path.join(index_dir, INDEX_FILE))
        chunks: list[Chunk] = []
        with open(os.path.join(index_dir, CHUNKS_FILE), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))
        with open(os.path.join(index_dir, META_FILE), encoding="utf-8") as f:
            meta = json.load(f)
        return cls(index, chunks, meta)

    @staticmethod
    def exists(index_dir: str) -> bool:
        return all(
            os.path.exists(os.path.join(index_dir, f))
            for f in (INDEX_FILE, CHUNKS_FILE, META_FILE)
        )

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        out: list[tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx != -1:
                out.append((self.chunks[idx], float(score)))
        return out

    @property
    def index_version(self) -> str:
        return self.meta.get("index_version", "unknown")


# ============================================================ 4. BM25 + RRF


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Lexical ranking. On legal text it catches exact article numbers and terms
    that a dense model blurs together."""

    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[Chunk, float]]], k: int, rrf_k: int = 60
) -> list[tuple[Chunk, float]]:
    """Fuse ranked lists by rank, not by score: sum(1 / (rrf_k + rank)).

    Rank-based fusion avoids having to calibrate cosine scores against BM25 scores,
    which live on completely different scales.
    """
    fused: dict[str, float] = {}
    keep: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            keep[chunk.chunk_id] = chunk
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(keep[cid], score) for cid, score in ordered]


# ============================================================ 5. Retriever


class Retriever:
    """Retrieval cache key = hash(query + index_version + top_k + mode).

    Invalidation is automatic: rebuilding the index changes index_version, so every
    old key stops matching. No stale hits after re-indexing.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: FaissStore,
        mode: str = "hybrid",
        cache_size: int = 256,
        candidate_pool: Optional[int] = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.mode = mode
        self.candidate_pool = candidate_pool or settings.candidate_pool
        self._bm25: Optional[BM25Index] = BM25Index(store.chunks) if mode == "hybrid" else None
        self._cache: "OrderedDict[str, list[RetrievedChunk]]" = OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(self, query: str, top_k: int) -> str:
        raw = f"{query}\x00{self.store.index_version}\x00{top_k}\x00{self.mode}\x00{self.candidate_pool}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _apply_filters(
        results: list[tuple[Chunk, float]], filters: Optional[dict[str, Any]]
    ) -> list[tuple[Chunk, float]]:
        if not filters:
            return results
        out = []
        for chunk, score in results:
            data = chunk.model_dump()
            if all(data.get(key) == val for key, val in filters.items()):
                out.append((chunk, score))
        return out

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> tuple[list[RetrievedChunk], bool]:
        """Returns (results, cache_hit)."""
        k = top_k or settings.top_k
        key = self._cache_key(query, k)
        if filters is None and key in self._cache:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return self._cache[key], True

        self.cache_misses += 1
        # Over-fetch a FIXED candidate pool, never a multiple of the requested k, so
        # the fused ranking (and therefore the top-5) does not change when the caller
        # asks for a different number of results.
        pool = max(self.candidate_pool, k)
        qvec = self.embedder.encode([query])[0]
        vector_hits = self.store.search(qvec, pool)

        if self.mode == "hybrid" and self._bm25 is not None:
            lexical_hits = self._bm25.search(query, pool)
            fused = reciprocal_rank_fusion(
                [vector_hits, lexical_hits], k=pool, rrf_k=settings.rrf_k
            )
        else:
            fused = vector_hits

        fused = self._apply_filters(fused, filters)[:k]
        results = [RetrievedChunk(chunk=c, score=s) for c, s in fused]

        if filters is None:
            self._cache[key] = results
            self._cache.move_to_end(key)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return results, False
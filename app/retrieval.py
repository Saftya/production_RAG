"""Retriever: ties embedder + FAISS + BM25 together, with a retrieval cache.

Retrieval cache key = hash(query + index_version + top_k + mode). A trivial
in-memory LRU is enough — retrieval is deterministic for a fixed index version,
so the cache is invalidated automatically whenever the index is rebuilt (the
index_version changes and old keys stop matching).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Optional

from app.config import settings
from app.embeddings import Embedder
from app.hybrid import BM25Index, reciprocal_rank_fusion
from app.schemas import Chunk, RetrievedChunk
from app.vectorstore import FaissStore


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: FaissStore,
        mode: str = "hybrid",
        cache_size: int = 256,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.mode = mode
        self._bm25: Optional[BM25Index] = None
        if mode == "hybrid":
            self._bm25 = BM25Index(store.chunks)
        self._cache: "OrderedDict[str, list[RetrievedChunk]]" = OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(self, query: str, top_k: int) -> str:
        raw = f"{query}\x00{self.store.index_version}\x00{top_k}\x00{self.mode}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _apply_filters(
        self, results: list[tuple[Chunk, float]], filters: Optional[dict[str, Any]]
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
        k = top_k or settings.top_k
        key = self._cache_key(query, k)
        if filters is None and key in self._cache:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return self._cache[key], True

        self.cache_misses += 1
        qvec = self.embedder.encode([query])[0]
        # over-fetch before fusion/filtering so we still return k good hits
        vector_hits = self.store.search(qvec, k * 4)

        if self.mode == "hybrid" and self._bm25 is not None:
            lexical_hits = self._bm25.search(query, k * 4)
            fused = reciprocal_rank_fusion(
                [vector_hits, lexical_hits], k=k * 4, rrf_k=settings.rrf_k
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

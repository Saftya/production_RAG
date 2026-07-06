"""Lexical retrieval (BM25) and Reciprocal Rank Fusion.

Hybrid retrieval fuses the vector ranking with a BM25 ranking. On structured
legal/financial corpora, lexical signal (exact article numbers, ticker names,
figures) meaningfully lifts recall over pure dense retrieval.
"""
from __future__ import annotations

import re

from app.schemas import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Thin wrapper over rank_bm25.BM25Okapi aligned to a chunk list."""

    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[Chunk, float]]],
    k: int,
    rrf_k: int = 60,
) -> list[tuple[Chunk, float]]:
    """Fuse multiple ranked lists. RRF score = sum(1 / (rrf_k + rank))."""
    fused: dict[str, float] = {}
    keep: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            keep[chunk.chunk_id] = chunk
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(keep[cid], score) for cid, score in ordered]

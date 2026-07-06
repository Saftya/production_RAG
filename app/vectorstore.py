"""FAISS vector store with on-disk persistence.

Layout under `index_dir/`:
  index.faiss   - the FAISS index (IndexFlatIP over normalized vectors = cosine)
  chunks.jsonl  - the chunk store, aligned 1:1 with index row order
  meta.json     - {embedding_model, dim, index_version, count, preprocessing_version}

The service loads all three at startup and NEVER re-embeds the corpus.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from app.schemas import Chunk

META_FILE = "meta.json"
CHUNKS_FILE = "chunks.jsonl"
INDEX_FILE = "index.faiss"


class FaissStore:
    def __init__(self, index: Any, chunks: list[Chunk], meta: dict[str, Any]) -> None:
        self.index = index
        self.chunks = chunks
        self.meta = meta

    # ---------- build / persist ----------
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
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine
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

    # ---------- load ----------
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

    # ---------- query ----------
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        out: list[tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            out.append((self.chunks[idx], float(score)))
        return out

    @property
    def index_version(self) -> str:
        return self.meta.get("index_version", "unknown")


def _index_version(model: str, prep: str, count: int) -> str:
    import hashlib

    return hashlib.sha1(f"{model}:{prep}:{count}".encode()).hexdigest()[:12]

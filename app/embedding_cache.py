"""On-disk embedding cache.

Key = sha256(text + model_name + preprocessing_version). Values are stored as
individual .npy files under a sharded directory so the cache survives re-indexing
and ablation runs without recomputing embeddings.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


class EmbeddingCache:
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
        shard = os.path.join(self.cache_dir, key[:2])
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

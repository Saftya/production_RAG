"""Response cache: full-answer cache keyed by hash(question + index_version).

Invalidation strategy (documented in ADR/README):
  - index_version salt: rebuilding the index changes index_version, so every old
    key misses automatically. No stale answers after re-indexing.
  - TTL: entries expire after RESPONSE_CACHE_TTL_S seconds.
  - LRU bound: at most RESPONSE_CACHE_MAX entries.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Optional

from app.schemas import AnswerResponse


class ResponseCache:
    def __init__(self, index_version: str, ttl_s: int, max_size: int) -> None:
        self.index_version = index_version
        self.ttl_s = ttl_s
        self.max_size = max_size
        self._store: "OrderedDict[str, tuple[float, AnswerResponse]]" = OrderedDict()

    def _key(self, question: str, top_k: int) -> str:
        raw = f"{question.strip().lower()}\x00{self.index_version}\x00{top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, question: str, top_k: int) -> Optional[AnswerResponse]:
        key = self._key(question, top_k)
        item = self._store.get(key)
        if item is None:
            return None
        ts, resp = item
        if time.time() - ts > self.ttl_s:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return resp

    def put(self, question: str, top_k: int, resp: AnswerResponse) -> None:
        key = self._key(question, top_k)
        self._store[key] = (time.time(), resp)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

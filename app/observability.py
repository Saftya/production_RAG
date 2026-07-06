"""Structured JSON logging + request_id propagation + in-memory metrics."""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from collections import deque
from contextvars import ContextVar
from typing import Any

from app.config import settings

# request_id is set per-request by the FastAPI middleware and read by the logger.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    return str(uuid.uuid4())


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


def log(event: str, **fields: Any) -> None:
    """Emit one structured JSON log line to stdout."""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "request_id": get_request_id(),
        "event": event,
        **fields,
    }
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class Metrics:
    """Thread-safe rolling metrics over the last N requests."""

    def __init__(self, window: int) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=window)
        self._errors: deque[int] = deque(maxlen=window)
        self._top1_scores: deque[float] = deque(maxlen=window)
        self.total_requests = 0
        self.total_tokens = 0
        self.cache_hits = 0
        self.cache_lookups = 0

    def observe(
        self,
        latency_ms: float,
        error: bool,
        tokens: int,
        top1_score: float | None,
        cache_hit: bool,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_tokens += tokens
            self._latencies.append(latency_ms)
            self._errors.append(1 if error else 0)
            if top1_score is not None:
                self._top1_scores.append(top1_score)
            self.cache_lookups += 1
            if cache_hit:
                self.cache_hits += 1

    @staticmethod
    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
        return round(s[idx], 2)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = list(self._latencies)
            errs = list(self._errors)
            scores = list(self._top1_scores)
            hit_rate = (self.cache_hits / self.cache_lookups) if self.cache_lookups else 0.0
            return {
                "latency_ms_p50": self._pct(lat, 50),
                "latency_ms_p95": self._pct(lat, 95),
                "error_rate": round(sum(errs) / len(errs), 4) if errs else 0.0,
                "cache_hit_rate": round(hit_rate, 4),
                "total_tokens": self.total_tokens,
                "total_requests": self.total_requests,
                "mean_top1_retrieval_score": round(sum(scores) / len(scores), 4)
                if scores
                else 0.0,
                "window": len(lat),
            }

    def prometheus(self) -> str:
        snap = self.snapshot()
        lines = [
            "# HELP rag_latency_ms Request latency percentiles",
            f'rag_latency_ms{{quantile="0.5"}} {snap["latency_ms_p50"]}',
            f'rag_latency_ms{{quantile="0.95"}} {snap["latency_ms_p95"]}',
            f"rag_error_rate {snap['error_rate']}",
            f"rag_cache_hit_rate {snap['cache_hit_rate']}",
            f"rag_tokens_total {snap['total_tokens']}",
            f"rag_requests_total {snap['total_requests']}",
            f"rag_top1_retrieval_score {snap['mean_top1_retrieval_score']}",
        ]
        return "\n".join(lines) + "\n"


metrics = Metrics(settings.metrics_window)

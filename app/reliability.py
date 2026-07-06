"""Reliability patterns: bounded timeout + retry with exponential backoff.

Both patterns are transport-agnostic: they wrap any callable. `generation.py`
uses `call_with_reliability` to guard the LLM call.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable, TypeVar

from app.observability import log

T = TypeVar("T")

# Single shared executor so we don't spawn a thread pool per request.
_EXECUTOR = ThreadPoolExecutor(max_workers=8)


class LLMTimeoutError(Exception):
    """Raised when the wrapped call exceeds the timeout budget."""


class LLMUnavailableError(Exception):
    """Raised when all retries are exhausted on transient failures."""


def with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    """Run `fn` in a worker thread and abort if it exceeds `timeout_s`."""
    future = _EXECUTOR.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout as exc:
        future.cancel()
        raise LLMTimeoutError(f"call exceeded {timeout_s}s") from exc


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    transient: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Retry `fn` on transient errors: waits base, 2*base, 4*base ..."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except LLMTimeoutError:
            # A timeout is not transient in the retry sense — propagate up.
            raise
        except transient as exc:  # noqa: PERF203
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            log("llm_retry", attempt=attempt, delay_s=delay, error=str(exc))
            time.sleep(delay)
    raise LLMUnavailableError(f"exhausted {max_retries} retries") from last_exc


def call_with_reliability(
    fn: Callable[[], T],
    timeout_s: float,
    max_retries: int,
    base_delay: float = 1.0,
) -> T:
    """Compose retry(timeout(fn)): each attempt is time-bounded, and transient
    failures are retried with exponential backoff. A timeout stops retrying."""

    def guarded() -> T:
        return with_timeout(fn, timeout_s)

    return retry_with_backoff(
        guarded,
        max_retries=max_retries,
        base_delay=base_delay,
        transient=(LLMUnavailableError, ConnectionError, TimeoutError, OSError),
    )

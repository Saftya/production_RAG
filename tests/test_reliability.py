"""Unit tests for the reliability patterns (timeout + retry with backoff).

Proves: a slow call raises LLMTimeoutError; transient failures are retried and
eventually succeed; exhausted retries raise LLMUnavailableError; and a timeout
inside /ask surfaces as HTTP 504.
"""
from __future__ import annotations

import time

import pytest

from app.reliability import (
    LLMTimeoutError,
    LLMUnavailableError,
    call_with_reliability,
    retry_with_backoff,
    with_timeout,
)


def test_with_timeout_raises_on_slow_call():
    def slow():
        time.sleep(2.0)
        return "never"

    with pytest.raises(LLMTimeoutError):
        with_timeout(slow, timeout_s=0.2)


def test_with_timeout_returns_fast_result():
    assert with_timeout(lambda: 42, timeout_s=1.0) == 42


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    out = retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_exhausts_and_raises():
    def always_fail():
        raise ConnectionError("down")

    with pytest.raises(LLMUnavailableError):
        retry_with_backoff(always_fail, max_retries=3, base_delay=0.01)


def test_timeout_is_not_retried():
    calls = {"n": 0}

    def slow():
        calls["n"] += 1
        time.sleep(1.0)

    with pytest.raises(LLMTimeoutError):
        call_with_reliability(slow, timeout_s=0.1, max_retries=3, base_delay=0.01)
    assert calls["n"] == 1  # timeout short-circuits the retry loop


def test_ask_returns_504_on_llm_timeout(client, monkeypatch):
    import app.main as main

    def boom(*_args, **_kwargs):
        raise LLMTimeoutError("llm too slow")

    monkeypatch.setattr(main, "generate", boom)
    r = client.post("/ask", json={"question": "Что такое трудовой договор?"})
    assert r.status_code == 504
    assert r.json()["error"] == "llm_timeout"

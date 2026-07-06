"""Endpoint contract tests: /health, /ready, /ask, /metrics + validation."""
from __future__ import annotations


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_true_after_index_load(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_ask_valid_returns_answer_and_sources(client):
    r = client.post(
        "/ask",
        json={"question": "Какова минимальная продолжительность ежегодного отпуска?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert len(body["sources"]) >= 1
    assert 0.0 <= body["confidence"] <= 1.0
    assert "request_id" in body
    assert body["prompt_version"] == "rag_v1"


def test_ask_empty_question_is_422_not_500(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_too_long_question_is_422(client):
    r = client.post("/ask", json={"question": "a" * 501})
    assert r.status_code == 422


def test_request_id_header_roundtrip(client):
    r = client.post(
        "/ask",
        headers={"X-Request-ID": "test-rid-123"},
        json={"question": "Что такое трудовой договор?"},
    )
    assert r.headers.get("X-Request-ID") == "test-rid-123"
    assert r.json()["request_id"] == "test-rid-123"


def test_metrics_json_snapshot(client):
    client.post("/ask", json={"question": "Как оплачивается сверхурочная работа?"})
    r = client.get("/metrics?format=json")
    assert r.status_code == 200
    snap = r.json()
    for key in ("latency_ms_p50", "latency_ms_p95", "error_rate", "cache_hit_rate", "total_tokens"):
        assert key in snap


def test_response_cache_marks_cached(client):
    q = {"question": "Какие дисциплинарные взыскания может применить работодатель?"}
    assert client.post("/ask", json=q).json()["cached"] is False
    assert client.post("/ask", json=q).json()["cached"] is True

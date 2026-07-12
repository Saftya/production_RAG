"""Prompt regression suite (runs in CI, must stay green).

Drives generation directly with controlled context so the expected behavior is
exact: relevant context -> grounded answer with sources; empty/irrelevant
context -> honest refusal. Uses the stub backend so no live model is needed.
"""
from __future__ import annotations

from app.generation import REFUSAL, generate
from app.schemas import Chunk, RetrievedChunk


def _chunk(cid: str, text: str, section: str) -> Chunk:
    return Chunk(chunk_id=cid, text=text, source_file="labor_code_sample.md", section_title=section)


def _grounded(cid: str, text: str, section: str, score: float = 0.8) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=_chunk(cid, text, section), score=score)]


def test_grounded_answer_cites_sources():
    ctx = _grounded(
        "c-otpusk",
        "Ежегодный оплачиваемый трудовой отпуск не менее 24 календарных дней.",
        "Статья 20",
    )
    resp, _ = generate("Какова минимальная продолжительность отпуска?", ctx, "rid-1")
    assert resp.answer != REFUSAL
    assert resp.used_context is True
    assert len(resp.sources) == 1
    assert resp.sources[0].chunk_id == "c-otpusk"


def test_grounded_numeric_answer():
    ctx = _grounded(
        "c-time",
        "Нормальная продолжительность рабочего времени не должна превышать 40 часов в неделю.",
        "Статья 12",
    )
    resp, _ = generate("Сколько часов в неделю?", ctx, "rid-2")
    assert resp.used_context is True
    assert resp.confidence > 0.0


def test_grounded_definition_answer():
    ctx = _grounded("c-def", "Трудовой договор — письменное соглашение сторон.", "Статья 1")
    resp, _ = generate("Что такое трудовой договор?", ctx, "rid-3")
    assert resp.answer != REFUSAL
    assert resp.used_context is True


def test_refuses_on_empty_context():
    resp, _ = generate("Кто выиграл ЧМ по футболу 1998?", [], "rid-4")
    assert resp.answer == REFUSAL
    assert resp.used_context is False
    assert resp.sources == []
    assert resp.confidence == 0.0


def test_refuses_on_irrelevant_context():
    # score 0.0 signals "nothing relevant retrieved" to the stub contract.
    ctx = [RetrievedChunk(chunk=_chunk("c-x", "нерелевантный текст", "Статья 99"), score=0.0)]
    resp, _ = generate("Столица Франции?", ctx, "rid-5")
    assert resp.answer == REFUSAL
    assert resp.used_context is False


def test_prompt_version_is_stamped():
    resp, _ = generate("вопрос", _grounded("c1", "текст", "Статья 1"), "rid-6")
    assert resp.prompt_version == "rag_v2"
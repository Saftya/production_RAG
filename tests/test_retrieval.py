"""Retrieval-quality gate: recall@5 >= 0.75.

Here the gate runs on the committed fixture corpus so it is deterministic in CI
and fails loudly if a change to chunking / fusion / the retriever regresses.
The number that goes into the README and the defense is produced by
`make eval` (real corpus + real ground truth + the Sentence-BERT embedder).
"""
from __future__ import annotations

from evaluate_retrieval import evaluate, load_gt  # evaluation/ is on sys.path (conftest)

RECALL_GATE = 0.75


def test_recall_at_5_meets_gate(retriever, gt_path):
    metrics = evaluate(retriever, load_gt(gt_path), k=5)
    assert metrics["recall@5"] >= RECALL_GATE, metrics


def test_hybrid_beats_or_matches_vector(retriever, gt_path):
    from app.retrieval import Retriever, build_embedder

    gt = load_gt(gt_path)
    hybrid = evaluate(retriever, gt, k=5)["recall@5"]
    vector = evaluate(
        Retriever(build_embedder(), retriever.store, mode="vector"), gt, k=5
    )["recall@5"]
    assert hybrid >= vector  # the improvement lever must not hurt


def test_label_does_not_match_across_source_files(retriever, gt_path):
    """Article numbers repeat across the Labor and Tax codes: a label for one file
    must never be satisfied by the same article number in the other."""
    from evaluate_retrieval import is_relevant

    from app.schemas import Chunk, RetrievedChunk

    gold = {"source_file": "tk.pdf", "relevant_sections": ["Статья 68"]}
    other = RetrievedChunk(
        chunk=Chunk(
            chunk_id="x", text="...", source_file="nalog.pdf",
            section_title="Статья 68. Совсем другое",
        ),
        score=1.0,
    )
    assert not is_relevant(other, gold)

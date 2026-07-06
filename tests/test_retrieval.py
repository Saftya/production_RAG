"""Retrieval-quality gate: recall@5 on the hand-labeled ground-truth set.

The defense threshold is recall@5 >= 0.75. This test asserts it on the current
index so a regression in chunking/retrieval fails CI. (In CI it runs on the hash
embedder over the sample corpus; on your real corpus, run
`python3 evaluation/evaluate_retrieval.py` with EMBEDDING_BACKEND=st.)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate_retrieval import evaluate, load_gt  # noqa: E402

RECALL_GATE = 0.75


def test_recall_at_5_meets_gate(retriever):
    metrics = evaluate(retriever, load_gt(), k=5)
    assert metrics["recall@5"] >= RECALL_GATE, metrics


def test_hybrid_beats_or_matches_vector(retriever):
    from app.embeddings import build_embedder
    from app.retrieval import Retriever

    gt = load_gt()
    hybrid = evaluate(retriever, gt, k=5)["recall@5"]
    vector_retriever = Retriever(build_embedder(), retriever.store, mode="vector")
    vector = evaluate(vector_retriever, gt, k=5)["recall@5"]
    assert hybrid >= vector  # the improvement lever should not hurt

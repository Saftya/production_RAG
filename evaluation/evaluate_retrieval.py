#!/usr/bin/env python3
"""Evaluate retrieval quality against the hand-labeled ground-truth set.

Metrics reported: recall@5 (primary gate >= 0.75), MRR, precision@5, nDCG@10.

Relevance rule: a retrieved chunk counts as relevant if its `section_title`
contains any string in the gold `relevant_sections`, or its `chunk_id` is in
`relevant_chunk_ids`. This keeps labels stable across re-chunking (we label by
article, not by fragile content hash).

Usage:
    python3 evaluation/evaluate_retrieval.py
    python3 evaluation/evaluate_retrieval.py --k 5 --mode hybrid
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.embeddings import build_embedder  # noqa: E402
from app.retrieval import Retriever  # noqa: E402
from app.schemas import RetrievedChunk  # noqa: E402
from app.vectorstore import FaissStore  # noqa: E402

GT_PATH = Path(__file__).parent / "ground_truth.jsonl"
RESULTS_PATH = Path(__file__).parent / "results" / "retrieval_metrics.json"


def is_relevant(rc: RetrievedChunk, gold: dict) -> bool:
    sec = rc.chunk.section_title or ""
    if any(marker in sec for marker in gold.get("relevant_sections", [])):
        return True
    return rc.chunk.chunk_id in set(gold.get("relevant_chunk_ids", []))


def load_gt() -> list[dict]:
    with open(GT_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(retriever: Retriever, gt: list[dict], k: int) -> dict:
    recalls, precisions, rr, ndcgs = [], [], [], []
    per_q = []
    for item in gt:
        retrieved, _ = retriever.retrieve(item["question"], top_k=max(k, 10))
        flags = [is_relevant(rc, item) for rc in retrieved]
        top_k_flags = flags[:k]

        recall = 1.0 if any(top_k_flags) else 0.0  # >=1 relevant in top-k
        precision = sum(top_k_flags) / k
        first = next((i for i, f in enumerate(flags) if f), None)
        reciprocal = 1.0 / (first + 1) if first is not None else 0.0
        # nDCG@10 over judged (retrieved) chunks: the ideal ranking places every
        # relevant chunk found in the top-10 at the front, so nDCG is bounded by 1.0.
        dcg = sum((1.0 / math.log2(i + 2)) for i, f in enumerate(flags[:10]) if f)
        rel_count = sum(1 for f in flags[:10] if f)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(rel_count))
        ndcg = dcg / idcg if idcg else 0.0

        recalls.append(recall)
        precisions.append(precision)
        rr.append(reciprocal)
        ndcgs.append(ndcg)
        per_q.append({"id": item.get("id"), "recall@k": recall, "rr": round(reciprocal, 3)})

    n = len(gt)
    return {
        "n_questions": n,
        "k": k,
        "mode": retriever.mode,
        f"recall@{k}": round(sum(recalls) / n, 4),
        f"precision@{k}": round(sum(precisions) / n, 4),
        "MRR": round(sum(rr) / n, 4),
        "nDCG@10": round(sum(ndcgs) / n, 4),
        "per_question": per_q,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", choices=["vector", "hybrid"], default=settings.retrieval_mode)
    args = p.parse_args()

    store = FaissStore.load(settings.index_dir)
    retriever = Retriever(build_embedder(), store, mode=args.mode)
    metrics = evaluate(retriever, load_gt(), args.k)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({m: metrics[m] for m in metrics if m != "per_question"}, ensure_ascii=False, indent=2))
    print(f"-> {RESULTS_PATH}")


if __name__ == "__main__":
    main()

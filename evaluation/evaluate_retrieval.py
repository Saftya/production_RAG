#!/usr/bin/env python3
"""Retrieval evaluation against the hand-labeled ground-truth set.

Metrics: recall@5 (defense gate >= 0.75), precision@5, MRR, nDCG@10.

Relevance rule
--------------
A retrieved chunk is relevant iff
  (a) the gold `source_file` matches (when the label specifies one), AND
  (b) its `section_title` contains one of the gold `relevant_sections` strings,
      or its `chunk_id` is listed in `relevant_chunk_ids`.

(a) matters because the Labor Code and the Tax Code both number their articles
from 1: without the file check, "Статья 68" would match in BOTH codes and inflate
recall with false positives. Labels are by article, not by content hash, so they
survive re-chunking.

Usage
-----
    python3 evaluation/evaluate_retrieval.py                 # metrics table
    python3 evaluation/evaluate_retrieval.py --mode vector   # ablation lever
    python3 evaluation/evaluate_retrieval.py --diagnose      # per-question OK/FAIL
    python3 evaluation/evaluate_retrieval.py --show "Статья 68"   # print an article
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval import FaissStore, Retriever, build_embedder  # noqa: E402
from app.schemas import RetrievedChunk, settings  # noqa: E402

GT_PATH = Path(__file__).parent / "ground_truth.jsonl"
RESULTS_PATH = Path(__file__).parent / "results" / "retrieval_metrics.json"


def load_gt(path: str | Path = GT_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(rc: RetrievedChunk, gold: dict) -> bool:
    want_file = gold.get("source_file")
    if want_file and rc.chunk.source_file != want_file:
        return False  # same article number in another code is NOT a hit
    section = rc.chunk.section_title or ""
    if any(marker in section for marker in gold.get("relevant_sections", [])):
        return True
    return rc.chunk.chunk_id in set(gold.get("relevant_chunk_ids", []))


# ---------------------------------------------------------------- metrics


def evaluate(retriever: Retriever, gt: list[dict], k: int = 5) -> dict:
    recalls, precisions, rr, ndcgs, per_q = [], [], [], [], []

    for item in gt:
        retrieved, _ = retriever.retrieve(item["question"], top_k=max(k, 10))
        flags = [is_relevant(rc, item) for rc in retrieved]
        top_k_flags = flags[:k]

        recall = 1.0 if any(top_k_flags) else 0.0  # >=1 relevant chunk in top-k
        precision = sum(top_k_flags) / k
        first = next((i for i, f in enumerate(flags) if f), None)
        reciprocal = 1.0 / (first + 1) if first is not None else 0.0

        # nDCG@10 normalized over the relevant chunks actually retrieved, so the
        # ideal ranking puts them all in front and the score is bounded by 1.0.
        dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags[:10]) if f)
        rel_count = sum(1 for f in flags[:10] if f)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(rel_count))
        ndcg = dcg / idcg if idcg else 0.0

        recalls.append(recall)
        precisions.append(precision)
        rr.append(reciprocal)
        ndcgs.append(ndcg)
        per_q.append({"id": item.get("id"), "hit": bool(recall), "rr": round(reciprocal, 3)})

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


# ---------------------------------------------------------------- dev helpers


def diagnose(retriever: Retriever, gt: list[dict], k: int = 5) -> None:
    """Per-question OK/FAIL with the top-3 actually retrieved — the tool you use
    to check that a label points at the article that really answers the question."""
    hits = 0
    for item in gt:
        retrieved, _ = retriever.retrieve(item["question"], top_k=k)
        hit = any(is_relevant(rc, item) for rc in retrieved)
        hits += hit
        print(f"{'OK  ' if hit else 'FAIL'} [{item.get('id')}] {item['question'][:60]}")
        if not hit:
            print(f"      label: {item.get('relevant_sections')} in {item.get('source_file')}")
            for rc in retrieved[:3]:
                print(f"      top:   {rc.chunk.source_file} | {(rc.chunk.section_title or '')[:45]}")
    print(f"\nrecall@{k} = {hits}/{len(gt)} = {hits / len(gt):.3f}")


def show(store: FaissStore, query: str, source_file: str | None = None) -> None:
    """Print the full text of matching articles, so you can verify a label by eye."""
    for c in store.chunks:
        if source_file and c.source_file != source_file:
            continue
        if c.section_title and query.lower() in c.section_title.lower():
            print(f"=== {c.source_file} | {c.section_title} ===")
            print(c.text[:1500])
            print("-" * 60)


# ---------------------------------------------------------------- cli


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", choices=["vector", "hybrid"], default=settings.retrieval_mode)
    p.add_argument("--gt", default=str(GT_PATH))
    p.add_argument("--diagnose", action="store_true", help="per-question OK/FAIL breakdown")
    p.add_argument("--show", metavar="ARTICLE", help='print an article, e.g. "Статья 68"')
    p.add_argument("--source-file", default=None, help="restrict --show to one file")
    args = p.parse_args()

    store = FaissStore.load(settings.index_dir)

    if args.show:
        show(store, args.show, args.source_file)
        return

    retriever = Retriever(build_embedder(), store, mode=args.mode)
    gt = load_gt(args.gt)

    if args.diagnose:
        diagnose(retriever, gt, args.k)
        return

    metrics = evaluate(retriever, gt, args.k)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    summary = {m: v for m, v in metrics.items() if m != "per_question"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {RESULTS_PATH}")


if __name__ == "__main__":
    main()

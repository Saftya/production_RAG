#!/usr/bin/env python3
"""Ablation harness: chunk_size / retrieval / min_chunk experiments.

Each configuration is indexed into its own scratch directory under
data/.ablation/<config>/ — the production index at data/index/ is never
touched or read. The on-disk embedding cache (data/.embed_cache/, keyed by
hash(text + model + preprocessing_version)) is shared and reused as-is, so
re-running an experiment (or a config whose chunk text overlaps one already
indexed for the service) is cheap.

Each config's build+embed+evaluate runs in its OWN subprocess (see
`run_config` / `_worker_main`), not in one long-lived process. Repeated
heavy torch (OpenMP/MKL) and faiss (also OpenMP) calls in a single process
segfault on shutdown under sustained load on this machine; isolating each
config means that crash, if it happens, kills an already-finished worker
after its result row is safely on disk — it can't lose earlier configs or
take down the whole experiment.

Reuses, does not reimplement:
  build_chunks                        <- scripts/build_index.py
  build_embedder, FaissStore, Retriever <- app/retrieval.py
  evaluate, load_gt                   <- evaluation/evaluate_retrieval.py

Usage:
    python3 scripts/run_ablation.py --experiment chunk_size
    python3 scripts/run_ablation.py --experiment retrieval
    python3 scripts/run_ablation.py --experiment min_chunk
    python3 scripts/run_ablation.py --experiment chunk_size --output evaluation/results
"""
from __future__ import annotations

import os

# Must be set before torch/faiss are ever imported (both lazy-imported, only
# inside the worker subprocess). KMP_DUPLICATE_LIB_OK suppresses the abort
# when two OpenMP runtimes (torch's MKL/OpenMP, faiss's OpenMP) load in the
# same process; capping thread count reduces contention without going fully
# serial. Belt-and-suspenders alongside the subprocess isolation above.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evaluation"))

ABLATION_DIR = ROOT / "data" / ".ablation"
GT_PATH = ROOT / "evaluation" / "ground_truth.jsonl"
RAW_DIR = str(ROOT / "data" / "raw")

# ---------------------------------------------------------------- worker (one config, one process)


def _build_and_evaluate(config: dict[str, Any], top_k: int, gt_path: str) -> dict[str, Any]:
    """Ingest + chunk + embed + index one config, evaluate it, return a result row.
    Runs inside the worker subprocess spawned by run_config()."""
    from build_index import build_chunks  # noqa: E402
    from app.retrieval import FaissStore, Retriever, build_embedder  # noqa: E402
    from app.schemas import settings  # noqa: E402
    from evaluate_retrieval import evaluate, load_gt  # noqa: E402

    embedder = build_embedder()
    t0 = time.perf_counter()
    chunks = build_chunks(
        RAW_DIR,
        strategy=config["strategy"],
        chunk_size=config.get("chunk_size") or 800,
        chunk_overlap=config.get("overlap") or 150,
        min_chunk_chars=config.get("min_chunk_chars") or 0,
    )
    if not chunks:
        raise SystemExit(f"no chunks produced for config {config!r} — check {RAW_DIR}")

    t_embed0 = time.perf_counter()
    vectors = embedder.encode([c.text for c in chunks])
    embed_s = time.perf_counter() - t_embed0

    store = FaissStore.build(vectors, chunks, embedder.model_name, settings.preprocessing_version)
    build_s = time.perf_counter() - t0
    store.save(str(ABLATION_DIR / config["name"]))

    gt = load_gt(gt_path)
    retriever = Retriever(
        embedder, store, mode=config.get("mode", "hybrid"),
        candidate_pool=config.get("candidate_pool"),
    )
    m = evaluate(retriever, gt, k=top_k)

    return {
        "config": config["name"],
        "mode": config.get("mode", "hybrid"),
        "strategy": config["strategy"],
        "chunk_size": config.get("chunk_size"),
        "overlap": config.get("overlap"),
        "min_chunk_chars": config.get("min_chunk_chars") or 0,
        "candidate_pool": retriever.candidate_pool,
        "n_chunks": len(chunks),
        "median_chunk_chars": round(statistics.median(len(c.text) for c in chunks), 1),
        "index_build_seconds": round(build_s, 2),
        "encode_ms_per_chunk": round((embed_s * 1000) / len(chunks), 3),
        f"recall@{top_k}": m[f"recall@{top_k}"],
        f"precision@{top_k}": m[f"precision@{top_k}"],
        "MRR": m["MRR"],
        "nDCG@10": m["nDCG@10"],
    }


def _worker_main(config_path: str, result_path: str, top_k: int, gt_path: str) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    row = _build_and_evaluate(config, top_k, gt_path)
    Path(result_path).write_text(json.dumps(row), encoding="utf-8")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip normal interpreter shutdown — sidesteps the native-lib segfault on exit


def run_config(config: dict[str, Any], top_k: int, gt_path: str = str(GT_PATH)) -> dict[str, Any]:
    """Run one config in an isolated subprocess; return its result row."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "config.json"
        result_path = Path(tmp) / "result.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable, str(SELF),
                "--_worker", "--_config", str(config_path), "--_result", str(result_path),
                "--top-k", str(top_k), "--gt", gt_path,
            ],
            cwd=str(ROOT),
        )
        if not result_path.exists():
            raise RuntimeError(
                f"config {config['name']!r} produced no result (subprocess exit code {proc.returncode})"
            )
        row = json.loads(result_path.read_text(encoding="utf-8"))

    print(
        f"  [{config['name']:>18}] n_chunks={row['n_chunks']:>5} "
        f"recall@{top_k}={row[f'recall@{top_k}']:.3f} MRR={row['MRR']:.3f}"
    )
    return row


# ---------------------------------------------------------------- experiments


def experiment_chunk_size(top_k: int = 5, gt_path: str = str(GT_PATH)) -> list[dict[str, Any]]:
    """A) recursive chunk-size sweep {200,400,800,1600}, overlap=size//5, plus the
    section (article-aware) strategy as a baseline row.

    ONE variable moves: how the text is cut. Everything else is pinned — same
    embedding model, hybrid retrieval, same top_k, and the SAME min_chunk_chars
    across every row (see below). An earlier version gave the section baseline the
    ToC-stub filter (min_chunk_chars=120) while leaving it off for recursive; that
    changes two variables at once and quietly flatters the baseline. The stub filter
    is measured on its own in experiment_min_chunk().
    """
    MIN_CHUNK_CHARS_FIXED = 0  # pinned across all rows: isolate the chunking variable

    configs = [
        {
            "name": f"recursive_{size}", "strategy": "recursive",
            "chunk_size": size, "overlap": size // 5,
            "min_chunk_chars": MIN_CHUNK_CHARS_FIXED, "mode": "hybrid",
        }
        for size in (200, 400, 800, 1600)
    ]
    configs.append(
        {
            "name": "section_baseline", "strategy": "section",
            "chunk_size": None, "overlap": None,
            "min_chunk_chars": MIN_CHUNK_CHARS_FIXED, "mode": "hybrid",
        }
    )
    return [run_config(c, top_k, gt_path) for c in configs]


def experiment_retrieval(top_k: int = 5, gt_path: str = str(GT_PATH)) -> list[dict[str, Any]]:
    """B) strategy=section fixed, mode in {vector, hybrid}."""
    from app.schemas import settings  # noqa: E402

    rows = []
    for mode in ("vector", "hybrid"):
        rows.append(
            run_config(
                {
                    "name": mode, "strategy": "section", "chunk_size": None, "overlap": None,
                    "min_chunk_chars": settings.min_chunk_chars, "mode": mode,
                },
                top_k,
                gt_path,
            )
        )
    return rows


def experiment_min_chunk(top_k: int = 5, gt_path: str = str(GT_PATH)) -> list[dict[str, Any]]:
    """C) strategy=section fixed, min_chunk_chars in {0, 120}: how many
    table-of-contents stub chunks (heading, no body) get filtered, and the
    effect on precision@5."""
    rows = []
    for min_chars in (0, 120):
        rows.append(
            run_config(
                {
                    "name": f"section_min{min_chars}", "strategy": "section",
                    "chunk_size": None, "overlap": None, "min_chunk_chars": min_chars, "mode": "hybrid",
                },
                top_k,
            )
        )
    if len(rows) == 2:
        rows[1]["stub_chunks_filtered"] = rows[0]["n_chunks"] - rows[1]["n_chunks"]
    return rows


def experiment_candidate_pool(top_k: int = 5, gt_path: str = str(GT_PATH)) -> list[dict[str, Any]]:
    """D) How many candidates each retriever hands to RRF before fusion.

    This one exists because of a bug: the pool used to be `top_k * 4`, so asking for
    10 results changed the top-5 (a bigger BM25 pool re-shuffled the fusion). diagnose
    (top_k=5) and evaluate (top_k=10) disagreed on the SAME config. The pool is now
    fixed and independent of top_k; this sweep picks its value on evidence rather than
    on a guess. Chunking, embeddings and mode are pinned — only the pool moves.
    """
    from app.schemas import settings  # noqa: E402

    rows = []
    for pool in (10, 20, 40, 80):
        rows.append(
            run_config(
                {
                    "name": f"pool_{pool}", "strategy": "section",
                    "chunk_size": None, "overlap": None,
                    "min_chunk_chars": settings.min_chunk_chars,
                    "mode": "hybrid", "candidate_pool": pool,
                },
                top_k,
                gt_path,
            )
        )
    return rows


EXPERIMENTS = {
    "chunk_size": experiment_chunk_size,
    "candidate_pool": experiment_candidate_pool,
    "retrieval": experiment_retrieval,
    "min_chunk": experiment_min_chunk,
}

# ---------------------------------------------------------------- output writers


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if v is None:
        return "—"
    return str(v)


_MD_COLUMNS = {
    "chunk_size": [
        ("config", "config"), ("strategy", "strategy"), ("chunk_size", "chunk_size"),
        ("overlap", "overlap"), ("n_chunks", "n_chunks"), ("median_chunk_chars", "median_chars"),
        ("index_build_seconds", "build_s"), ("encode_ms_per_chunk", "encode_ms/chunk"),
    ],
    "retrieval": [
        ("mode", "mode"), ("n_chunks", "n_chunks"), ("index_build_seconds", "build_s"),
        ("encode_ms_per_chunk", "encode_ms/chunk"),
    ],
    "min_chunk": [
        ("min_chunk_chars", "min_chunk_chars"), ("n_chunks", "n_chunks"),
        ("stub_chunks_filtered", "stub_chunks_filtered"), ("median_chunk_chars", "median_chars"),
    ],
    "candidate_pool": [
        ("candidate_pool", "candidate_pool"), ("n_chunks", "n_chunks"),
    ],
}


def write_json(rows: list[dict], path: Path, *, experiment: str) -> None:
    path.write_text(
        json.dumps({"experiment": experiment, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown(rows: list[dict], path: Path, *, experiment: str, title: str, top_k: int) -> None:
    metric_cols = [
        (f"recall@{top_k}", f"recall@{top_k}"), (f"precision@{top_k}", f"precision@{top_k}"),
        ("MRR", "MRR"), ("nDCG@10", "nDCG@10"),
    ]
    columns = _MD_COLUMNS[experiment] + metric_cols
    headers = [h for _, h in columns]
    keys = [k for k, _ in columns]
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(k)) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(rows: list[dict], path: Path, *, experiment: str, top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    recall_key, precision_key = f"recall@{top_k}", f"precision@{top_k}"

    if experiment == "chunk_size":
        recursive = [r for r in rows if r["strategy"] == "recursive"]
        baseline = next((r for r in rows if r["strategy"] == "section"), None)
        sizes = [r["chunk_size"] for r in recursive]
        ax.plot(sizes, [r[recall_key] for r in recursive], marker="o", label="recursive")
        if baseline:
            ax.axhline(
                baseline[recall_key], ls="--", color="green",
                label=f"section baseline ({baseline[recall_key]:.2f})",
            )
        ax.axhline(0.75, ls="--", color="red", alpha=0.6, label="defense gate 0.75")
        ax.set_xlabel("chunk size (chars, recursive)")
        ax.set_ylabel(recall_key)
        ax.set_title("Chunk-size ablation")
        ax.set_ylim(0, 1)
        ax.legend()

    elif experiment == "retrieval":
        metrics = [recall_key, precision_key, "MRR", "nDCG@10"]
        x = list(range(len(metrics)))
        width = 0.35
        for i, r in enumerate(rows):
            offs = [xi + i * width for xi in x]
            ax.bar(offs, [r[m] for m in metrics], width=width, label=r["mode"])
        ax.set_xticks([xi + width / 2 for xi in x])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title("Retrieval strategy: vector vs hybrid")
        ax.legend()

    elif experiment == "candidate_pool":
        pools = [r["candidate_pool"] for r in rows]
        ax.plot(pools, [r[recall_key] for r in rows], marker="o", label=recall_key)
        ax.plot(pools, [r["MRR"] for r in rows], marker="s", label="MRR")
        ax.axhline(0.75, ls="--", color="red", alpha=0.6, label="defense gate 0.75")
        ax.set_xlabel("candidate_pool (candidates fused by RRF)")
        ax.set_ylim(0, 1)
        ax.set_title("Candidate-pool ablation")
        ax.legend()

    else:  # min_chunk
        labels = [str(r["min_chunk_chars"]) for r in rows]
        ax.bar(labels, [r["n_chunks"] for r in rows], color="#999999", alpha=0.5, label="n_chunks")
        ax.set_ylabel("n_chunks")
        ax.set_xlabel("min_chunk_chars")
        ax2 = ax.twinx()
        ax2.plot(labels, [r[precision_key] for r in rows], marker="o", color="C0", label=precision_key)
        ax2.plot(labels, [r[recall_key] for r in rows], marker="s", color="C1", label=recall_key)
        ax2.set_ylim(0, 1)
        ax.set_title("ToC-stub filtering: min_chunk_chars")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- cli

_TITLES = {
    "chunk_size": "Chunk-size sweep (recursive) vs section (article-aware) baseline",
    "retrieval": "Retrieval strategy: vector vs hybrid (section chunking)",
    "min_chunk": "min_chunk_chars: filtering table-of-contents stub chunks",
    "candidate_pool": "candidate_pool: how many candidates RRF fuses before truncating to top_k",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation experiments for retrieval quality")
    p.add_argument("--experiment", choices=list(EXPERIMENTS))
    p.add_argument("--output", default=str(ROOT / "evaluation" / "results"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--gt", default=str(GT_PATH))
    # internal: run a single config in an isolated subprocess (spawned by run_config())
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_config", help=argparse.SUPPRESS)
    p.add_argument("--_result", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._worker:
        _worker_main(args._config, args._result, args.top_k, args.gt)
        return  # unreachable — _worker_main always os._exit()s

    if not args.experiment:
        p.error("--experiment is required")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    from evaluate_retrieval import load_gt  # noqa: E402

    gt = load_gt(args.gt)
    print(f"[{args.experiment}] ground_truth_questions={len(gt)}")

    rows = EXPERIMENTS[args.experiment](args.top_k, args.gt)

    json_path = out_dir / f"ablation_{args.experiment}.json"
    md_path = out_dir / f"ablation_{args.experiment}.md"
    png_path = out_dir / f"ablation_{args.experiment}.png"

    write_json(rows, json_path, experiment=args.experiment)
    write_markdown(rows, md_path, experiment=args.experiment, title=_TITLES[args.experiment], top_k=args.top_k)
    write_plot(rows, png_path, experiment=args.experiment, top_k=args.top_k)

    print(f"[{args.experiment}] wrote {json_path.name}, {md_path.name}, {png_path.name} -> {out_dir}")


if __name__ == "__main__":
    main()
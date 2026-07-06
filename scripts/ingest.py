#!/usr/bin/env python3
"""Ingest raw corpus -> data/chunks.jsonl. Idempotent and reproducible.

Usage:
    python3 scripts/ingest.py --raw data/raw --out data/chunks.jsonl \
        --strategy section
    python3 scripts/ingest.py --raw data/raw --strategy recursive \
        --chunk-size 800 --overlap 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.ingestion import build_chunks, write_chunks_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest corpus to chunks.jsonl")
    p.add_argument("--raw", default=f"{settings.data_dir}/raw")
    p.add_argument("--out", default=settings.chunks_path)
    p.add_argument("--strategy", choices=["section", "recursive"], default="section")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=150)
    args = p.parse_args()

    chunks = build_chunks(
        args.raw,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )
    n = write_chunks_jsonl(chunks, args.out)
    print(f"strategy={args.strategy} chunks={n} -> {args.out}")


if __name__ == "__main__":
    main()

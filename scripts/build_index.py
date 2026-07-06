#!/usr/bin/env python3
"""Build the FAISS index from data/chunks.jsonl.

- Embeds every chunk through the CachedEmbedder (on-disk cache => re-runs are
  near-instant, which matters because ablation re-indexes several times).
- Persists index.faiss + chunks.jsonl + meta.json under INDEX_DIR.
- Idempotent: running twice yields the same index_version for the same inputs.

Usage:
    python3 scripts/build_index.py                 # uses .env / defaults
    EMBEDDING_BACKEND=hash python3 scripts/build_index.py   # offline/CI
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.embeddings import build_embedder  # noqa: E402
from app.ingestion import build_chunks, read_chunks_jsonl, write_chunks_jsonl  # noqa: E402
from app.vectorstore import FaissStore  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()

    chunks_path = Path(settings.chunks_path)
    if not chunks_path.exists():
        print(f"[build_index] {chunks_path} missing — ingesting data/raw first")
        chunks = build_chunks(f"{settings.data_dir}/raw", strategy="section")
        write_chunks_jsonl(chunks, str(chunks_path))
    chunks = read_chunks_jsonl(str(chunks_path))
    if not chunks:
        raise SystemExit("no chunks found — check data/raw and run ingest")

    embedder = build_embedder()
    texts = [c.text for c in chunks]
    print(f"[build_index] embedding {len(texts)} chunks with {embedder.model_name} ...")
    vectors = embedder.encode(texts)

    store = FaissStore.build(
        vectors, chunks, embedder.model_name, settings.preprocessing_version
    )
    store.save(settings.index_dir)

    if hasattr(embedder, "cache_stats"):
        print(f"[build_index] embedding cache: {embedder.cache_stats()}")
    dt = time.perf_counter() - t0
    print(
        f"[build_index] done: {len(chunks)} chunks, dim={store.meta['dim']}, "
        f"index_version={store.index_version}, {dt:.1f}s -> {settings.index_dir}"
    )


if __name__ == "__main__":
    main()

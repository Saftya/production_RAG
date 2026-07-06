# ADR 002 — Vector store: FAISS (flat, on-disk) over ChromaDB

Status: accepted

## Context
We need a vector index that (a) loads from disk at startup without re-embedding,
(b) reaches `/ready` in <30s, (c) runs on a laptop CPU, and (d) is easy to reason
about during a defense. Corpus is small (hundreds of chunks, low thousands at most).

## Options considered
1. **FAISS `IndexFlatIP`** on L2-normalized vectors = exact cosine. Zero-approximation
   (perfect recall ceiling for the index itself), trivial to persist/load, no server.
2. **ChromaDB (persistent)**. Nice metadata filtering and a collection abstraction,
   but adds a dependency/service surface and hides the ANN behavior we want to defend.
3. **FAISS `IndexHNSWFlat`**. Approximate, faster at scale — unnecessary at our size
   and introduces recall/latency knobs we don't need yet.

## Decision
Use **FAISS `IndexFlatIP`**, persisted as `index.faiss` + `chunks.jsonl` + `meta.json`.
Metadata filtering is done in Python post-retrieval (small corpus, cheap).

## Consequences
- (+) Exact search — the index never loses a relevant chunk; recall problems are
  provably a chunking/embedding issue, not an ANN issue.
- (+) Startup just `read_index` + load JSONL; well under the 30s gate.
- (−) `IndexFlatIP` is O(N) per query — fine now, but the documented scale path is
  HNSW/IVF-PQ past ~1M vectors (see architecture.md).
- (−) Post-hoc Python filtering won't scale to huge corpora; would move into the
  index (Chroma-style metadata) at that point.

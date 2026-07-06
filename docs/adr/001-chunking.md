# ADR 001 — Chunking strategy: section-aware over fixed-size

Status: accepted

## Context
Our corpus is structured legal text (Trudovoy Codex): numbered, self-contained
articles. Ground-truth questions are article-lookup style. The chunker decides
what a "unit of retrieval" is, which caps achievable recall before the embedder
or LLM ever runs.

## Options considered
1. **Fixed-size recursive** (`RecursiveCharacterTextSplitter`, 800/150). Simple,
   corpus-agnostic, but splits mid-article and dilutes the article's key facts
   across two chunks — hurts single-article recall.
2. **Section/article-aware** (split on `Статья N` / headings). Each chunk is one
   article: a natural, self-contained answer unit. Risk: very long articles
   exceed the embedder's useful context.
3. **Semantic chunking** (`SemanticChunker`). Powerful for prose, but overkill
   for already-delimited legal text and adds an embedding pass at build time.

## Decision
Use **section-aware chunking as the default**, keep recursive available and use
it as the ablation baseline. Long articles fall back to recursive internally.

## Consequences
- (+) On our ground truth, section beats recursive-800 on recall@5 (see README §5–6).
- (+) `section_title` metadata gives clean, human-readable citations.
- (−) Strategy is corpus-shaped; a corpus without headings degrades to recursive.
- (−) Chunk sizes are uneven, so BM25/vector score scales vary — mitigated by RRF
  fusion, which is rank-based not score-based.

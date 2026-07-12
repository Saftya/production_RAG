#!/usr/bin/env python3
"""Idempotent corpus build: load -> clean -> chunk -> embed -> FAISS index.

This is the single reproducible entry point from raw documents to a served index
(§1 + §2 of the spec). `notebooks/01_ingestion.ipynb` imports the functions below,
so the notebook and the production build can never drift apart.

Two chunking strategies (spec requires >= 2):
  recursive : RecursiveCharacterTextSplitter (LangChain) — size + overlap.
  section   : custom rule splitting on article boundaries ("Статья N", "Глава N",
              markdown headings). Each article becomes one self-contained chunk.

Usage:
    python3 scripts/build_index.py                              # section (default)
    python3 scripts/build_index.py --strategy recursive --chunk-size 800 --overlap 150
    python3 scripts/build_index.py --min-chunk-chars 120        # drop ToC stubs
    EMBEDDING_BACKEND=hash python3 scripts/build_index.py       # offline / CI
"""
from __future__ import annotations

import argparse
import bisect
import glob
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval import FaissStore, build_embedder  # noqa: E402
from app.schemas import Chunk, settings  # noqa: E402

# ---------------------------------------------------------------- loading


def _read_text_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader  # lazy

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_html(path: str) -> str:
    from bs4 import BeautifulSoup  # lazy

    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text("\n")


def load_documents(raw_dir: str) -> list[tuple[str, str]]:
    """Return [(source_file, raw_text)]. Supports md / txt / pdf / html."""
    docs: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)):
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in (".md", ".txt"):
            text = _read_text_file(path)
        elif ext == ".pdf":
            text = _read_pdf(path)
        elif ext in (".html", ".htm"):
            text = _read_html(path)
        else:
            continue
        docs.append((os.path.basename(path), text))
    return docs


# ---------------------------------------------------------------- cleaning

_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")
_MULTI_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)


def clean_text(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # rejoin words split across lines
    text = _PAGE_NUM_LINE.sub("", text)  # drop bare page numbers
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- chunking


def _chunk_id(source: str, char_start: int, text: str) -> str:
    """Content-addressed: same input -> same id, so the build is idempotent."""
    h = hashlib.sha1(f"{source}:{char_start}:{text[:64]}".encode()).hexdigest()[:12]
    return f"{os.path.splitext(source)[0]}::{char_start}::{h}"


_SECTION_HEADER = re.compile(
    r"(?m)^(?:\s*)((?:Статья|Глава|Раздел)\s+[\d\-]+[\.\)]?.*|#{1,6}\s+.+)$"
)


def _section_title_lookup(text: str) -> Callable[[int], Optional[str]]:
    """Build a lookup: char offset -> title of the article/heading it falls under.

    Same headings chunk_section() splits on, so a recursive chunk and a section
    chunk covering the same span agree on section_title — required for
    evaluate_retrieval.py's is_relevant(), which matches ground-truth labels
    against section_title regardless of which chunking strategy produced the chunk.
    """
    headers = [(m.start(), m.group(1).strip().lstrip("#").strip()) for m in _SECTION_HEADER.finditer(text)]
    starts = [h[0] for h in headers]

    def lookup(pos: int) -> Optional[str]:
        idx = bisect.bisect_right(starts, pos) - 1
        return headers[idx][1] if idx >= 0 else None

    return lookup


def chunk_recursive(
    source: str,
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    document_version: str = "v1",
) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # lazy

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    section_title_at = _section_title_lookup(text)
    chunks: list[Chunk] = []
    cursor = 0
    for piece in splitter.split_text(text):
        start = text.find(piece, cursor)
        start = start if start != -1 else cursor
        end = start + len(piece)
        cursor = max(cursor, end - chunk_overlap)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(source, start, piece),
                text=piece,
                source_file=source,
                section_title=section_title_at(start),
                char_start=start,
                char_end=end,
                document_version=document_version,
            )
        )
    return chunks


def chunk_section(source: str, text: str, document_version: str = "v1") -> list[Chunk]:
    """Article-aware chunking: heading -> section_title, body -> chunk text."""
    matches = list(_SECTION_HEADER.finditer(text))
    if not matches:
        return chunk_recursive(source, text, document_version=document_version)
    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        title = m.group(1).strip().lstrip("#").strip()
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(source, start, body),
                text=body,
                source_file=source,
                section_title=title,
                char_start=start,
                char_end=end,
                document_version=document_version,
            )
        )
    return chunks


def build_chunks(
    raw_dir: str,
    strategy: str = "section",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 0,
) -> list[Chunk]:
    """Full ingestion pipeline. `min_chunk_chars` drops table-of-contents stubs:
    a heading with no body can never answer a question, it only pollutes top-k."""
    all_chunks: list[Chunk] = []
    for source, raw in load_documents(raw_dir):
        text = clean_text(raw)
        if strategy == "recursive":
            all_chunks.extend(chunk_recursive(source, text, chunk_size, chunk_overlap))
        else:
            all_chunks.extend(chunk_section(source, text))
    if min_chunk_chars > 0:
        all_chunks = [c for c in all_chunks if len(c.text) >= min_chunk_chars]
    return all_chunks


# ---------------------------------------------------------------- persistence


def write_chunks_jsonl(chunks: Iterable[Chunk], path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(ch.model_dump_json() + "\n")
            n += 1
    return n


def read_chunks_jsonl(path: str) -> list[Chunk]:
    out: list[Chunk] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Chunk.model_validate_json(line))
    return out


# ---------------------------------------------------------------- index build


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest corpus and build the FAISS index")
    p.add_argument("--raw", default=f"{settings.data_dir}/raw")
    p.add_argument("--chunks", default=settings.chunks_path)
    p.add_argument("--strategy", choices=["section", "recursive"], default="section")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--min-chunk-chars", type=int, default=settings.min_chunk_chars)
    p.add_argument(
        "--reuse-chunks",
        action="store_true",
        help="skip ingestion and embed the existing chunks.jsonl",
    )
    args = p.parse_args()
    t0 = time.perf_counter()

    if args.reuse_chunks and Path(args.chunks).exists():
        chunks = read_chunks_jsonl(args.chunks)
        print(f"[build_index] reusing {len(chunks)} chunks from {args.chunks}")
    else:
        chunks = build_chunks(
            args.raw,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            min_chunk_chars=args.min_chunk_chars,
        )
        write_chunks_jsonl(chunks, args.chunks)
        print(f"[build_index] strategy={args.strategy} chunks={len(chunks)} -> {args.chunks}")

    if not chunks:
        raise SystemExit(f"no chunks — is {args.raw} empty?")

    embedder = build_embedder()
    print(f"[build_index] embedding {len(chunks)} chunks with {embedder.model_name} ...")
    vectors = embedder.encode([c.text for c in chunks])

    store = FaissStore.build(vectors, chunks, embedder.model_name, settings.preprocessing_version)
    store.save(settings.index_dir)

    print(f"[build_index] embedding cache: {embedder.cache_stats()}")
    print(
        f"[build_index] done: {len(chunks)} chunks, dim={store.meta['dim']}, "
        f"index_version={store.index_version}, "
        f"{time.perf_counter() - t0:.1f}s -> {settings.index_dir}"
    )


if __name__ == "__main__":
    main()

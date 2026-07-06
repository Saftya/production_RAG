"""Corpus ingestion: load -> clean -> chunk -> Chunk objects with metadata.

Two chunking strategies (assignment requires >= 2):
  1. recursive  -> RecursiveCharacterTextSplitter (LangChain), size+overlap.
  2. section     -> custom rule that splits on article/heading boundaries
                    ("Статья N", markdown "#" headings). Self-contained units.

The pipeline is idempotent: same inputs -> same chunk_ids (content hash based).
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
from typing import Iterable

from app.schemas import Chunk

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
    """Return list of (source_file, raw_text). Supports md/txt/pdf/html."""
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
# crude page-number / running-header lines (a bare number or ALLCAPS short line)
_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)


def clean_text(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # join hyphenated line breaks
    text = _PAGE_NUM_LINE.sub("", text)  # drop standalone page numbers
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- chunking


def _chunk_id(source: str, char_start: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{char_start}:{text[:64]}".encode()).hexdigest()[:12]
    return f"{os.path.splitext(source)[0]}::{char_start}::{h}"


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
                char_start=start,
                char_end=end,
                document_version=document_version,
            )
        )
    return chunks


# Split on legal article headers ("Статья 12."), Chapter headers, or md "#".
_SECTION_HEADER = re.compile(
    r"(?m)^(?:\s*)((?:Статья|Глава|Раздел)\s+\d+[\.\)]?.*|#{1,6}\s+.+)$"
)


def chunk_section(
    source: str, text: str, document_version: str = "v1"
) -> list[Chunk]:
    """Section/article-aware chunking. Each article becomes one self-contained
    chunk with its heading as section_title."""
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
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for source, raw in load_documents(raw_dir):
        text = clean_text(raw)
        if strategy == "recursive":
            all_chunks.extend(
                chunk_recursive(source, text, chunk_size, chunk_overlap)
            )
        else:
            all_chunks.extend(chunk_section(source, text))
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

"""Pydantic models: request/response contracts and internal chunk types."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------- internal types ----------
class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    page: Optional[int] = None
    section_title: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    document_version: str = "v1"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


# ---------- API request ----------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    filters: Optional[dict[str, Any]] = None


# ---------- API response ----------
class Source(BaseModel):
    chunk_id: str
    source_file: str
    section_title: Optional[str] = None
    page: Optional[int] = None
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    used_context: bool
    request_id: str
    prompt_version: str
    model_name: str
    cached: bool = False
    degraded: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    index_loaded: bool
    embedder_loaded: bool
    llm_configured: bool

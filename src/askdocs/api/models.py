"""Pydantic request / response models for the REST API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    filename: str
    source: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    cited_ids: list[str]
    missing_citations: list[str]


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to a file or directory to ingest")
    reset: bool = Field(default=False, description="Wipe existing index before ingesting")


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    vector_store_size: int

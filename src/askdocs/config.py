"""Centralised settings — override any value via environment variable or .env file."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    # pydantic-settings v2 automatically maps field_name → FIELD_NAME env var.
    # No need for the deprecated env= kwarg on Field().
    llm_provider: Literal["openai", "ollama"] = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 64

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # ── Retrieval ─────────────────────────────────────────────────────────────
    bm25_top_k: int = 20
    vector_top_k: int = 20
    hybrid_top_k: int = 10
    rrf_k: int = 60

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    vector_store_path: Path = Path("data/processed/faiss_index")
    bm25_store_path: Path = Path("data/processed/bm25_store.pkl")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Set API_KEY to a secret string to enable bearer-token auth on all endpoints.
    # Leave empty (default) to disable auth for local/private use.
    api_key: str = ""
    # Comma-separated list of allowed CORS origins.
    # Default "*" is fine for local dev; restrict to your domain in production.
    cors_origins: str = "*"
    # Ingestion is only allowed from within this directory tree (path traversal guard).
    # Override with an absolute path in production.
    allowed_ingest_dir: Path = Path("data")
    # Set LOG_FORMAT=json for structured JSON logs (recommended in production).
    log_format: str = "text"

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_dataset_path: Path = Path("data/eval/golden_dataset.jsonl")
    eval_faithfulness_threshold: float = 0.7
    eval_relevance_threshold: float = 0.7
    eval_citation_threshold: float = 0.8

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

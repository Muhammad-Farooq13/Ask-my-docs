"""Orchestrates load → chunk → embed → index into VectorStore and BM25Store."""
from __future__ import annotations

import logging
from pathlib import Path

from askdocs.config import settings
from askdocs.ingestion.chunker import Chunk, chunk_document
from askdocs.ingestion.loader import Document, load_directory, load_document
from askdocs.retrieval.bm25_store import BM25Store
from askdocs.retrieval.embedder import Embedder
from askdocs.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def run_ingestion(
    source: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    reset: bool = False,
) -> tuple[VectorStore, BM25Store]:
    """
    Full ingestion pipeline.

    Args:
        source:        File or directory to ingest.
        chunk_size:    Token/char budget per chunk (defaults to settings).
        chunk_overlap: Overlap between consecutive chunks (defaults to settings).
        reset:         If True, wipe any existing index before writing.

    Returns:
        (VectorStore, BM25Store) — ready for querying.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    source = Path(source)
    docs: list[Document] = []
    if source.is_file():
        docs.append(load_document(source))
    else:
        docs.extend(load_directory(source))

    if not docs:
        raise ValueError(f"No supported documents found at {source!r}")

    logger.info("Loaded %d document(s)", len(docs))

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    logger.info("Produced %d chunks", len(all_chunks))

    texts = [c.text for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    ids = [c.chunk_id for c in all_chunks]

    embedder = Embedder(model_name=settings.embed_model, batch_size=settings.embed_batch_size)

    logger.info("Generating embeddings (model=%s)…", settings.embed_model)
    embeddings = embedder.embed(texts)

    # ── Vector store ─────────────────────────────────────────────────────────
    vs_path = settings.vector_store_path
    vs_path.mkdir(parents=True, exist_ok=True)

    if reset or not (vs_path / "index.faiss").exists():
        vs = VectorStore.build(embeddings=embeddings, texts=texts, metadatas=metadatas, ids=ids)
    else:
        vs = VectorStore.load(vs_path)
        vs.add(embeddings=embeddings, texts=texts, metadatas=metadatas, ids=ids)
    vs.save(vs_path)

    # ── BM25 store ───────────────────────────────────────────────────────────
    bm25_path = settings.bm25_store_path
    bm25_path.parent.mkdir(parents=True, exist_ok=True)

    if reset or not bm25_path.exists():
        bm25 = BM25Store.build(texts=texts, metadatas=metadatas, ids=ids)
    else:
        bm25 = BM25Store.load(bm25_path)
        bm25.add(texts=texts, metadatas=metadatas, ids=ids)
    bm25.save(bm25_path)

    logger.info(
        "Ingestion complete — %d chunks indexed. VS: %s | BM25: %s",
        len(vs._texts),
        vs_path,
        bm25_path,
    )
    return vs, bm25

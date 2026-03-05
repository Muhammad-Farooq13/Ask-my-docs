"""FastAPI route handlers."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import StreamingResponse

from askdocs.api.middleware import require_api_key
from askdocs.api.models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Dependency helpers ────────────────────────────────────────────────────────

def _get_chain(request: Request):
    chain = request.app.state.chain
    if chain is None:
        raise HTTPException(status_code=503, detail="Index not loaded. POST /api/v1/ingest first.")
    return chain


def _get_embedder(request: Request):
    return request.app.state.embedder


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    chain = request.app.state.chain
    loaded = chain is not None
    size = len(chain.retriever.vs._texts) if loaded else 0
    return HealthResponse(status="ok", index_loaded=loaded, vector_store_size=size)


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest, chain=Depends(_get_chain)) -> QueryResponse:
    """Retrieve, rerank, and generate an answer with inline citations."""
    try:
        response = chain.run(req.question)
    except Exception:
        logger.exception("RAG chain error for question: %r", req.question)
        raise HTTPException(status_code=500, detail="Internal generation error.")

    sources = [
        SourceItem(
            chunk_id=s.chunk_id,
            text=s.text,
            score=round(s.score, 4),
            filename=s.metadata.get("filename", ""),
            source=s.metadata.get("source", ""),
        )
        for s in response.sources
    ]
    return QueryResponse(
        question=response.query,
        answer=response.answer,
        sources=sources,
        cited_ids=response.cited_ids,
        missing_citations=response.missing_citations,
    )


@router.post("/query/stream", tags=["rag"])
def query_stream(req: QueryRequest, request: Request, chain=Depends(_get_chain)):
    """
    Server-Sent Events streaming endpoint.

    Event schema:
    - First event: {"sources": [{chunk_id, filename}, …]}
    - Subsequent events: {"token": "…"}
    - Final event: [DONE]
    """
    sources, token_stream = chain.stream(req.question)

    def _generator():
        src_payload = json.dumps(
            {
                "sources": [
                    {"chunk_id": s.chunk_id, "filename": s.metadata.get("filename", "")}
                    for s in sources
                ]
            }
        )
        yield f"data: {src_payload}\n\n"
        for token in token_stream:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generator(), media_type="text/event-stream")


@router.post("/ingest", response_model=IngestResponse, tags=["ops"],
             dependencies=[Security(require_api_key)])
def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    """
    Ingest documents from *source_path* (file or directory) into both
    the FAISS vector store and the BM25 index, then hot-reload the chain.

    The resolved path must be inside the configured ALLOWED_INGEST_DIR
    (default: data/) to prevent path-traversal attacks.
    """
    from askdocs.config import settings as cfg
    from askdocs.generation.chain import RAGChain
    from askdocs.generation.llm import LLMClient
    from askdocs.ingestion.pipeline import run_ingestion
    from askdocs.retrieval.embedder import Embedder
    from askdocs.retrieval.hybrid import HybridRetriever
    from askdocs.retrieval.reranker import CrossEncoderReranker

    # ── Path traversal guard ─────────────────────────────────────────────────
    try:
        requested = Path(req.source_path).resolve()
        allowed  = cfg.allowed_ingest_dir.resolve()
        requested.relative_to(allowed)  # raises ValueError if outside
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"source_path must be inside the allowed directory "
                f"({cfg.allowed_ingest_dir}). Path traversal is not permitted."
            ),
        )

    try:
        vs, bm25 = run_ingestion(requested, reset=req.reset)
    except Exception:
        logger.exception("Ingestion failed for path: %r", req.source_path)
        raise HTTPException(status_code=500, detail="Ingestion failed. Check server logs.")

    embedder = Embedder()
    retriever = HybridRetriever(vs, bm25, embedder)
    reranker = CrossEncoderReranker()
    llm = LLMClient()
    chain = RAGChain(retriever, reranker, llm)

    request.app.state.chain = chain
    request.app.state.embedder = embedder

    return IngestResponse(status="success", chunks_indexed=len(vs._texts))

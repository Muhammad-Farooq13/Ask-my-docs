"""FastAPI application factory with lifespan index auto-loading."""
from __future__ import annotations

import json
import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from askdocs.api.middleware import RequestIDMiddleware, SecurityHeadersMiddleware
from askdocs.api.routes import router
from askdocs.config import settings


def _configure_logging() -> None:
    """Text logging for dev; JSON logging for production (LOG_FORMAT=json)."""
    if settings.log_format == "json":

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps(
                    {
                        "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                        "level": record.levelname,
                        "logger": record.name,
                        "msg": record.getMessage(),
                    }
                )

        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers = [handler]
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        )
    logging.root.setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-load persisted index on startup if one already exists."""
    app.state.chain = None
    app.state.embedder = None

    vs_path = settings.vector_store_path
    bm25_path = settings.bm25_store_path

    if (vs_path / "index.faiss").exists() and bm25_path.exists():
        logger.info("Existing index found — loading…")
        try:
            from askdocs.generation.chain import RAGChain
            from askdocs.generation.llm import LLMClient
            from askdocs.retrieval.bm25_store import BM25Store
            from askdocs.retrieval.embedder import Embedder
            from askdocs.retrieval.hybrid import HybridRetriever
            from askdocs.retrieval.reranker import CrossEncoderReranker
            from askdocs.retrieval.vector_store import VectorStore

            embedder = Embedder()
            vs = VectorStore.load(vs_path)
            bm25 = BM25Store.load(bm25_path)
            retriever = HybridRetriever(vs, bm25, embedder)
            reranker = CrossEncoderReranker()
            llm = LLMClient()

            app.state.chain = RAGChain(retriever, reranker, llm)
            app.state.embedder = embedder
            logger.info("Index loaded — %d chunks ready.", len(vs._texts))
        except Exception as exc:
            logger.warning("Could not load existing index: %s", exc)
    else:
        logger.info("No existing index — POST /api/v1/ingest to create one.")

    yield  # application is running

    logger.info("AskMyDocs API shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AskMyDocs",
        description=(
            "Production RAG API — hybrid retrieval (BM25 + vector), "
            "cross-encoder reranking, and citation-enforced generation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware is applied in reverse order (last added = first executed).
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "askdocs.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

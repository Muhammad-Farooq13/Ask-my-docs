"""Full RAG chain: retrieve → rerank → generate with citation audit."""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

from askdocs.generation.llm import LLMClient
from askdocs.generation.prompts import build_prompt
from askdocs.retrieval.hybrid import HybridRetriever
from askdocs.retrieval.reranker import CrossEncoderReranker
from askdocs.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[([^\[\]\s]+)\]")


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[SearchResult]
    cited_ids: list[str]          # chunk IDs that appear in the answer
    missing_citations: list[str]  # source IDs NOT cited (citation recall gap)


# ── Citation helpers ──────────────────────────────────────────────────────────

def extract_citations(text: str) -> list[str]:
    """Return all [chunk_id] tokens found in *text*, in order of appearance."""
    return _CITATION_RE.findall(text)


def audit_citations(answer: str, sources: list[SearchResult]) -> list[str]:
    """Return IDs of source chunks that are NOT cited anywhere in *answer*."""
    cited = set(extract_citations(answer))
    return [s.chunk_id for s in sources if s.chunk_id not in cited]


# ── RAG Chain ─────────────────────────────────────────────────────────────────

class RAGChain:
    """
    Orchestrates the full RAG pipeline:

        query
          └─► HybridRetriever  (BM25 + vector → RRF)
                └─► CrossEncoderReranker  (top-k refinement)
                      └─► LLMClient  (citation-enforced generation)
                            └─► citation audit → RAGResponse
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm: LLMClient,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    def run(self, query: str) -> RAGResponse:
        """Execute the pipeline end-to-end. Blocks until the full answer is ready."""
        # 1. Hybrid retrieval
        candidates = self.retriever.retrieve(query)
        logger.debug("Hybrid retrieval: %d candidates", len(candidates))

        # 2. Cross-encoder reranking
        sources = self.reranker.rerank(query, candidates)
        logger.debug("After reranking: %d sources", len(sources))

        # 3. Citation-enforced generation
        system, user = build_prompt(query, sources)
        answer = self.llm.complete(system, user)

        # 4. Citation audit
        cited_ids = extract_citations(answer)
        missing = audit_citations(answer, sources)
        if missing:
            logger.warning("Citation gap — %d source(s) not cited: %s", len(missing), missing)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=sources,
            cited_ids=cited_ids,
            missing_citations=missing,
        )

    def stream(self, query: str) -> tuple[list[SearchResult], Iterator[str]]:
        """
        Streaming variant for the SSE endpoint.

        Returns:
            (sources, token_iterator) — sources are resolved before streaming
            begins so the API can emit them as the first SSE event.
        """
        candidates = self.retriever.retrieve(query)
        sources = self.reranker.rerank(query, candidates)
        system, user = build_prompt(query, sources)
        return sources, self.llm.stream(system, user)

"""Hybrid retriever: BM25 sparse + dense vector fused via Reciprocal Rank Fusion."""
from __future__ import annotations

from askdocs.config import settings
from askdocs.retrieval.bm25_store import BM25Store
from askdocs.retrieval.embedder import Embedder
from askdocs.retrieval.vector_store import SearchResult, VectorStore

# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    *ranked_lists: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """
    Combine N ranked lists using RRF [Cormack et al., 2009].

    RRF score = Σ  1 / (k + rank_i)
    where rank_i is the 0-based position of the document in list i.

    Returns a deduplicated, descending-score list of SearchResults whose
    `.score` field holds the aggregate RRF score for transparency.
    """
    rrf_scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            cid = result.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            # Keep the highest original score as the "canonical" result
            if cid not in best_result or result.score > best_result[cid].score:
                best_result[cid] = result

    fused = sorted(best_result.values(), key=lambda r: rrf_scores[r.chunk_id], reverse=True)
    for r in fused:
        r.score = rrf_scores[r.chunk_id]
    return fused


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Two-stage retriever:
    1. Dense (FAISS cosine) + Sparse (BM25Okapi) independently retrieve top-k candidates.
    2. RRF fuses both ranked lists into a single deduped ranking.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: Embedder,
        vector_top_k: int | None = None,
        bm25_top_k: int | None = None,
        hybrid_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.vs = vector_store
        self.bm25 = bm25_store
        self.embedder = embedder
        self.vector_top_k = vector_top_k or settings.vector_top_k
        self.bm25_top_k = bm25_top_k or settings.bm25_top_k
        self.hybrid_top_k = hybrid_top_k or settings.hybrid_top_k
        self.rrf_k = rrf_k or settings.rrf_k

    def retrieve(self, query: str) -> list[SearchResult]:
        """Return up to *hybrid_top_k* fused results for *query*."""
        q_emb = self.embedder.embed_query(query)
        dense = self.vs.search(q_emb, top_k=self.vector_top_k)
        sparse = self.bm25.search(query, top_k=self.bm25_top_k)
        fused = reciprocal_rank_fusion(dense, sparse, k=self.rrf_k)
        return fused[: self.hybrid_top_k]

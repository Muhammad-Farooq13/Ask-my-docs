"""Cross-encoder reranker that refines the hybrid retrieval candidates."""
from __future__ import annotations

from askdocs.config import settings
from askdocs.retrieval.vector_store import SearchResult


class CrossEncoderReranker:
    """
    Loads a cross-encoder (query, passage) model from sentence-transformers
    and re-scores the candidates retrieved by HybridRetriever.

    Cross-encoders are significantly more accurate than bi-encoders for
    relevance scoring because they jointly model query and passage, but are
    too slow for full-corpus search — hence they are applied only to the
    top-k hybrid candidates (typically 10–20).
    """

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self.top_k = top_k or settings.reranker_top_k
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """
        Score each (query, passage) pair and return the top-k sorted descending.

        The `.score` field of each result is overwritten with the cross-encoder
        logit so downstream code can inspect raw ranking signals.
        """
        if not results:
            return []
        pairs = [(query, r.text) for r in results]
        raw_scores = self.model.predict(pairs)
        ranked = sorted(
            zip(results, raw_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        reranked: list[SearchResult] = []
        for result, score in ranked[: self.top_k]:
            result.score = float(score)
            reranked.append(result)
        return reranked

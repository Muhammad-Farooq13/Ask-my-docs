"""BM25 sparse retrieval store with serialisation support."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from askdocs.retrieval.vector_store import SearchResult


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokeniser — fast and sufficient for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Store:
    """
    Wraps rank-bm25's BM25Okapi with the same SearchResult interface as
    VectorStore so both can be fused uniformly by the hybrid retriever.

    Note: BM25Okapi is not incrementally updatable — adds rebuild the index
    from scratch. This is acceptable for batch ingestion pipelines.
    """

    def __init__(
        self,
        bm25,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        self._bm25 = bm25
        self._texts = list(texts)
        self._metadatas = list(metadatas)
        self._ids = list(ids)

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> BM25Store:
        from rank_bm25 import BM25Okapi
        corpus = [_tokenise(t) for t in texts]
        return cls(BM25Okapi(corpus), texts, metadatas, ids)

    def add(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Append documents and rebuild the BM25 index."""
        from rank_bm25 import BM25Okapi
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)
        corpus = [_tokenise(t) for t in self._texts]
        self._bm25 = BM25Okapi(corpus)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "texts": self._texts,
                    "metadatas": self._metadatas,
                    "ids": self._ids,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> BM25Store:
        with open(Path(path), "rb") as fh:
            d = pickle.load(fh)
        return cls(d["bm25"], d["texts"], d["metadatas"], d["ids"])

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Return up to *top_k* chunks with positive BM25 score."""
        tokens = _tokenise(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self._texts[i],
                chunk_id=self._ids[i],
                score=float(scores[i]),
                metadata=self._metadatas[i],
            )
            for i in ranked
            if scores[i] > 0
        ]

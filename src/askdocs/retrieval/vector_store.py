"""FAISS-backed dense vector store with persistence."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    text: str
    chunk_id: str
    score: float
    metadata: dict


class VectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP index.

    Because all embeddings are L2-normalised, inner-product search is
    equivalent to cosine similarity — no extra normalisation step needed.
    """

    def __init__(
        self,
        index,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        self._index = index
        self._texts = list(texts)
        self._metadatas = list(metadatas)
        self._ids = list(ids)

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> VectorStore:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return cls(index, texts, metadatas, ids)

    def add(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        self._index.add(embeddings)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        import faiss
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as fh:
            pickle.dump(
                {"texts": self._texts, "metadatas": self._metadatas, "ids": self._ids}, fh
            )

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        import faiss
        path = Path(path)
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as fh:
            data = pickle.load(fh)
        return cls(index, data["texts"], data["metadatas"], data["ids"])

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[SearchResult]:
        """Return up to *top_k* nearest chunks by cosine similarity."""
        q = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self._texts))
        if k == 0:
            return []
        scores, indices = self._index.search(q, k)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                SearchResult(
                    text=self._texts[idx],
                    chunk_id=self._ids[idx],
                    score=float(score),
                    metadata=self._metadatas[idx],
                )
            )
        return results

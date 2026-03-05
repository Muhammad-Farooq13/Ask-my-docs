"""Dense text embedder backed by sentence-transformers (lazy-loaded)."""
from __future__ import annotations

import numpy as np


class Embedder:
    """
    Wraps a SentenceTransformer bi-encoder.

    The underlying model is loaded on first use so import time stays fast.
    All returned embeddings are L2-normalised (unit vectors), making inner
    product equivalent to cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.  Returns float32 array of shape (N, D)."""
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Convenience wrapper for a single query string. Returns shape (D,)."""
        return self.embed([text])[0]

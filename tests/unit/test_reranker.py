"""Unit tests for the CrossEncoderReranker (model mocked out)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from askdocs.retrieval.reranker import CrossEncoderReranker
from askdocs.retrieval.vector_store import SearchResult

# ── Helpers ────────────────────────────────────────────────────────────────────

def _res(chunk_id: str, text: str = "some text") -> SearchResult:
    return SearchResult(text=text, chunk_id=chunk_id, score=0.5, metadata={})


def _reranker_with_scores(scores: list[float], top_k: int = 10) -> CrossEncoderReranker:
    rr = CrossEncoderReranker(top_k=top_k)
    rr._model = MagicMock()
    rr._model.predict.return_value = scores
    return rr


# ── Sorting ────────────────────────────────────────────────────────────────────

def test_reranker_sorts_descending():
    rr = _reranker_with_scores([0.2, 0.9, 0.5])
    results = [_res("a"), _res("b"), _res("c")]
    reranked = rr.rerank("query", results)
    assert [r.chunk_id for r in reranked] == ["b", "c", "a"]


def test_reranker_overwrites_score_field():
    rr = _reranker_with_scores([0.3, 0.8])
    results = [_res("a"), _res("b")]
    reranked = rr.rerank("query", results)
    # Scores should reflect cross-encoder output, not original 0.5
    assert reranked[0].score == pytest.approx(0.8)
    assert reranked[1].score == pytest.approx(0.3)


# ── top_k enforcement ─────────────────────────────────────────────────────────

def test_reranker_respects_top_k():
    rr = _reranker_with_scores([0.9, 0.7, 0.5, 0.3], top_k=2)
    results = [_res(f"doc_{i}") for i in range(4)]
    reranked = rr.rerank("query", results)
    assert len(reranked) == 2


def test_reranker_top_k_larger_than_input():
    rr = _reranker_with_scores([0.5, 0.8], top_k=100)
    results = [_res("x"), _res("y")]
    reranked = rr.rerank("query", results)
    assert len(reranked) == 2


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_reranker_empty_input():
    rr = CrossEncoderReranker(top_k=5)
    rr._model = MagicMock()
    assert rr.rerank("query", []) == []
    rr._model.predict.assert_not_called()


def test_reranker_single_result():
    rr = _reranker_with_scores([0.75])
    reranked = rr.rerank("query", [_res("solo")])
    assert len(reranked) == 1
    assert reranked[0].chunk_id == "solo"
    assert reranked[0].score == pytest.approx(0.75)


def test_reranker_predict_called_with_correct_pairs():
    rr = _reranker_with_scores([0.5, 0.6])
    results = [_res("a", "text a"), _res("b", "text b")]
    rr.rerank("my question", results)
    rr._model.predict.assert_called_once_with(
        [("my question", "text a"), ("my question", "text b")]
    )

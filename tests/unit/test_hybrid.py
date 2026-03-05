"""Unit tests for hybrid retrieval and Reciprocal Rank Fusion."""
from __future__ import annotations

from askdocs.retrieval.hybrid import reciprocal_rank_fusion
from askdocs.retrieval.vector_store import SearchResult

# ── Helpers ────────────────────────────────────────────────────────────────────

def _res(chunk_id: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        text=f"text for {chunk_id}",
        chunk_id=chunk_id,
        score=score,
        metadata={},
    )


# ── RRF correctness ───────────────────────────────────────────────────────────

def test_rrf_merges_two_lists():
    list_a = [_res("a", 0.9), _res("b", 0.8), _res("c", 0.5)]
    list_b = [_res("b", 0.7), _res("d", 0.6), _res("a", 0.4)]
    fused = reciprocal_rank_fusion(list_a, list_b)
    ids = [r.chunk_id for r in fused]
    # "a" and "b" appear in both lists → should score highest
    assert ids.index("a") < ids.index("d")
    assert ids.index("b") < ids.index("d")


def test_rrf_deduplicates_chunk_ids():
    duped = [_res("x", 0.9), _res("x", 0.5)]
    fused = reciprocal_rank_fusion(duped)
    assert [r.chunk_id for r in fused].count("x") == 1


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_single_empty_list():
    list_a = [_res(f"doc_{i}") for i in range(5)]
    fused = reciprocal_rank_fusion(list_a, [])
    assert len(fused) == 5


def test_rrf_single_list_preserves_order():
    list_a = [_res(f"doc_{i}", float(10 - i)) for i in range(5)]
    fused = reciprocal_rank_fusion(list_a)
    # Higher-ranked doc in original list should still rank higher after fusion
    assert fused[0].chunk_id == "doc_0"


def test_rrf_score_field_updated():
    list_a = [_res("a", 0.99)]
    fused = reciprocal_rank_fusion(list_a, k=60)
    # Score should be the RRF value 1/(60+1) ≈ 0.0164, not the original 0.99
    assert fused[0].score < 0.1


def test_rrf_three_lists_combined():
    list_a = [_res("a"), _res("b")]
    list_b = [_res("b"), _res("c")]
    list_c = [_res("a"), _res("c"), _res("b")]
    fused = reciprocal_rank_fusion(list_a, list_b, list_c)
    ids = [r.chunk_id for r in fused]
    # "b" appears in all three lists → should rank first
    assert ids[0] == "b"


def test_rrf_result_count():
    list_a = [_res(f"a{i}") for i in range(10)]
    list_b = [_res(f"b{i}") for i in range(10)]
    fused = reciprocal_rank_fusion(list_a, list_b)
    # 10 unique a-docs + 10 unique b-docs = 20 total
    assert len(fused) == 20

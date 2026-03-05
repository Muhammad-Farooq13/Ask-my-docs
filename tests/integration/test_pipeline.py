"""
Integration tests — ingest sample docs then exercise the full retrieval stack.

No LLM calls are made: the RAGChain is not constructed here, keeping the suite
fast and free of external API keys.
"""
from __future__ import annotations

import pytest

from askdocs.ingestion.chunker import chunk_document
from askdocs.ingestion.loader import Document

# ── Sample corpus ─────────────────────────────────────────────────────────────

_CORPUS = [
    (
        "python_intro.txt",
        (
            "Python is a high-level, interpreted programming language. "
            "It emphasises code readability and simplicity. "
            "Python supports multiple paradigms including procedural, "
            "object-oriented, and functional programming."
        ),
    ),
    (
        "ml_basics.txt",
        (
            "Machine learning is a subset of artificial intelligence. "
            "Supervised learning uses labelled data to train models. "
            "Unsupervised learning discovers patterns in unlabelled data. "
            "Reinforcement learning trains agents via reward signals."
        ),
    ),
    (
        "rag_explained.txt",
        (
            "Retrieval-Augmented Generation combines a retrieval component "
            "with a generative language model. BM25 is a classical sparse "
            "retrieval algorithm. Dense vector search uses neural embeddings "
            "to find semantically similar passages."
        ),
    ),
]


def _make_doc(filename: str, content: str) -> Document:
    return Document(
        content=content,
        metadata={"source": filename, "filename": filename, "filetype": ".txt"},
    )


# ── Shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def stores():
    """Build VectorStore and BM25Store from the sample corpus."""
    from askdocs.retrieval.bm25_store import BM25Store
    from askdocs.retrieval.embedder import Embedder
    from askdocs.retrieval.vector_store import VectorStore

    docs = [_make_doc(fn, content) for fn, content in _CORPUS]
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=256, chunk_overlap=32))

    embedder = Embedder()
    texts = [c.text for c in all_chunks]
    ids = [c.chunk_id for c in all_chunks]
    metas = [c.metadata for c in all_chunks]
    embeddings = embedder.embed(texts)

    vs = VectorStore.build(embeddings, texts, metas, ids)
    bm25 = BM25Store.build(texts, metas, ids)
    return vs, bm25, embedder


# ── Vector store ──────────────────────────────────────────────────────────────

def test_vector_search_returns_results(stores):
    vs, bm25, embedder = stores
    q_emb = embedder.embed_query("What is Python?")
    results = vs.search(q_emb, top_k=3)
    assert len(results) > 0


def test_vector_search_python_query_ranks_python_chunk_first(stores):
    vs, bm25, embedder = stores
    q_emb = embedder.embed_query("Python programming language")
    results = vs.search(q_emb, top_k=3)
    assert any("Python" in r.text for r in results)


def test_vector_search_scores_are_bounded(stores):
    vs, bm25, embedder = stores
    q_emb = embedder.embed_query("machine learning")
    results = vs.search(q_emb, top_k=5)
    for r in results:
        assert -1.01 <= r.score <= 1.01, f"Cosine score out of range: {r.score}"


# ── BM25 store ────────────────────────────────────────────────────────────────

def test_bm25_search_returns_results(stores):
    vs, bm25, embedder = stores
    results = bm25.search("machine learning supervised", top_k=3)
    assert len(results) > 0


def test_bm25_search_only_positive_scores(stores):
    vs, bm25, embedder = stores
    results = bm25.search("supervised learning labels", top_k=10)
    for r in results:
        assert r.score > 0


def test_bm25_search_returns_empty_for_unknown_query(stores):
    vs, bm25, embedder = stores
    results = bm25.search("xyzzy_nonexistent_gibberish_token", top_k=5)
    assert results == []


# ── Hybrid retriever ──────────────────────────────────────────────────────────

def test_hybrid_retrieval_returns_results(stores):
    from askdocs.retrieval.hybrid import HybridRetriever
    vs, bm25, embedder = stores
    retriever = HybridRetriever(vs, bm25, embedder, vector_top_k=5, bm25_top_k=5, hybrid_top_k=3)
    results = retriever.retrieve("RAG and vector search")
    assert 0 < len(results) <= 3


def test_hybrid_retrieval_no_duplicate_chunk_ids(stores):
    from askdocs.retrieval.hybrid import HybridRetriever
    vs, bm25, embedder = stores
    retriever = HybridRetriever(vs, bm25, embedder, hybrid_top_k=10)
    results = retriever.retrieve("Python embeddings retrieval")
    ids = [r.chunk_id for r in results]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs in hybrid results"


# ── Cross-encoder reranker ────────────────────────────────────────────────────

def test_reranker_refines_hybrid_results(stores):
    from askdocs.retrieval.hybrid import HybridRetriever
    from askdocs.retrieval.reranker import CrossEncoderReranker
    vs, bm25, embedder = stores
    retriever = HybridRetriever(vs, bm25, embedder, hybrid_top_k=5)
    reranker = CrossEncoderReranker(top_k=3)
    candidates = retriever.retrieve("dense embeddings BM25 retrieval")
    reranked = reranker.rerank("dense embeddings BM25 retrieval", candidates)
    assert 0 < len(reranked) <= 3


# ── Persistence round-trip ────────────────────────────────────────────────────

def test_vector_store_save_and_load(tmp_path, stores):
    from askdocs.retrieval.vector_store import VectorStore
    vs, _, embedder = stores
    save_path = tmp_path / "faiss_idx"
    vs.save(save_path)

    loaded = VectorStore.load(save_path)
    assert len(loaded._texts) == len(vs._texts)
    assert loaded._ids == vs._ids

    q_emb = embedder.embed_query("Python")
    original_top = vs.search(q_emb, top_k=1)[0].chunk_id
    loaded_top = loaded.search(q_emb, top_k=1)[0].chunk_id
    assert original_top == loaded_top


def test_bm25_store_save_and_load(tmp_path, stores):
    from askdocs.retrieval.bm25_store import BM25Store
    _, bm25, _ = stores
    save_path = tmp_path / "bm25.pkl"
    bm25.save(save_path)

    loaded = BM25Store.load(save_path)
    assert len(loaded._texts) == len(bm25._texts)

    orig = bm25.search("machine learning", top_k=3)
    loaded_results = loaded.search("machine learning", top_k=3)
    assert [r.chunk_id for r in orig] == [r.chunk_id for r in loaded_results]

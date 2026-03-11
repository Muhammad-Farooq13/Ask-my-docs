"""
Build and persist the demo BM25 index used by streamlit_app.py.

Run once before launching the Streamlit app:
    python build_demo.py

Output: models/demo_bm25.pkl
"""
from __future__ import annotations

import os
import pickle
import sys

# Make the local package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rank_bm25 import BM25Okapi  # noqa: E402

SAMPLE_DOCS = [
    {
        "title": "Retrieval-Augmented Generation (RAG) Overview",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
            "language model to ground answers in real documents. The retriever fetches "
            "relevant passages from a corpus; the generator synthesises a final answer. "
            "RAG reduces hallucinations, keeps knowledge up to date without retraining, "
            "and makes model decisions explainable through source citations."
        ),
    },
    {
        "title": "BM25 Sparse Retrieval",
        "text": (
            "BM25 (Best Match 25) is a probabilistic bag-of-words ranking function. "
            "It scores each document by the TF-IDF weight of query terms, dampened by "
            "document length normalisation. BM25Okapi is the most widely used variant. "
            "Despite its simplicity BM25 remains surprisingly competitive with dense "
            "vector models, especially on short keyword queries."
        ),
    },
    {
        "title": "Dense Vector Retrieval with FAISS",
        "text": (
            "Dense retrieval encodes queries and passages into high-dimensional vectors "
            "using a bi-encoder (e.g. sentence-transformers). Nearest-neighbour search "
            "over those vectors — powered by FAISS — finds semantically similar passages "
            "even when they share no keywords with the query. L2-normalised inner-product "
            "search is equivalent to cosine similarity."
        ),
    },
    {
        "title": "Hybrid Retrieval and Reciprocal Rank Fusion",
        "text": (
            "Hybrid retrieval combines sparse (BM25) and dense signals. Each method "
            "independently retrieves a candidate list; Reciprocal Rank Fusion (RRF) then "
            "merges them: score = Σ 1/(k + rank_i) for each list i. RRF is robust to "
            "score-scale mismatches between BM25 and cosine similarities and consistently "
            "outperforms either method alone."
        ),
    },
    {
        "title": "Cross-Encoder Reranking",
        "text": (
            "A cross-encoder reads the query and passage together as a single sequence, "
            "producing a relevance score that captures fine-grained interactions. Unlike "
            "bi-encoders, cross-encoders are too slow for full-corpus retrieval but excel "
            "at reranking a small candidate set (10–20 documents). The ms-marco-MiniLM "
            "family offers a good accuracy/latency trade-off."
        ),
    },
    {
        "title": "Citation Enforcement in LLM Answers",
        "text": (
            "Citation enforcement prompts the LLM to reference retrieved chunk IDs "
            "inline (e.g. [chunk-123]). A post-generation audit regex extracts all cited "
            "IDs and compares them against the provided context. Any claim without a "
            "matching citation is flagged as a potential hallucination, reported in the "
            "missing_citations field of the API response."
        ),
    },
    {
        "title": "FastAPI REST API Design",
        "text": (
            "The AskMyDocs API exposes four endpoints: POST /ingest (upload documents), "
            "POST /api/v1/query (blocking Q&A), POST /api/v1/query/stream (SSE token "
            "streaming), GET /health (index stats). Bearer-token auth guards the write "
            "endpoints. Security headers middleware and path-traversal validation prevent "
            "common web vulnerabilities."
        ),
    },
    {
        "title": "Pydantic Settings and Configuration",
        "text": (
            "All runtime parameters are managed by a Pydantic BaseSettings class. "
            "Values cascade: defaults < .env file < environment variables. This makes "
            "the system fully reproducible: clone the repo, copy .env.example to .env, "
            "adjust API keys and model names, then docker compose up."
        ),
    },
]


def build() -> None:
    os.makedirs("models", exist_ok=True)
    dest = os.path.join("models", "demo_bm25.pkl")

    tokenised = [doc["text"].lower().split() for doc in SAMPLE_DOCS]
    model = BM25Okapi(tokenised)

    payload = {"model": model, "docs": SAMPLE_DOCS}
    with open(dest, "wb") as fh:
        pickle.dump(payload, fh)

    print(f"Saved BM25 demo index to {dest}  ({len(SAMPLE_DOCS)} documents)")


if __name__ == "__main__":
    build()

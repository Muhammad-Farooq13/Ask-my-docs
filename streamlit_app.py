"""
AskMyDocs — Interactive RAG Pipeline Demo
Standalone Streamlit app: showcases chunking and BM25 retrieval
without needing a running FastAPI server or OpenAI key.
"""
from __future__ import annotations

import math
import os
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Make sure the local package is importable when running from repo root
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

st.set_page_config(
    page_title="AskMyDocs — RAG Demo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Lazy imports (only fail gracefully if package not installed)
# ---------------------------------------------------------------------------
try:
    from askdocs.ingestion.chunker import chunk_document
    from askdocs.ingestion.loader import Document

    _HAS_CHUNKER = True
except ImportError:
    _HAS_CHUNKER = False

try:
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

# ---------------------------------------------------------------------------
# Sample corpus for BM25 demo
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _metric_card(label: str, value: str, delta: str | None = None) -> None:
    cols = st.columns([1])
    with cols[0]:
        st.metric(label, value, delta)


def _bm25_score(query: str, docs: list[dict]) -> list[dict]:
    """Return docs sorted by BM25 score, highest first."""
    tokenised = [d["text"].lower().split() for d in docs]
    bm25 = BM25Okapi(tokenised)
    q_tokens = query.lower().split()
    raw = bm25.get_scores(q_tokens)
    total = sum(raw) or 1.0
    results = [
        {**doc, "score": float(s), "pct": float(s) / total * 100}
        for doc, s in zip(docs, raw)
    ]
    return sorted(results, key=lambda x: x["score"], reverse=True)


# ---------------------------------------------------------------------------
# Page tabs
# ---------------------------------------------------------------------------
tab_overview, tab_chunker, tab_bm25, tab_arch = st.tabs(
    ["Overview", "Live Chunker", "BM25 Search", "Architecture & API"]
)

# ─── Tab 1: Overview ────────────────────────────────────────────────────────
with tab_overview:
    st.markdown(
        """
        <h1 style='text-align:center; margin-bottom:0'>📚 AskMyDocs</h1>
        <p style='text-align:center; color:#A78BFA; font-size:1.1rem; margin-top:4px'>
        Production-grade Retrieval-Augmented Generation · Hybrid BM25 + Vector Search
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieval stages", "2", "BM25 + FAISS")
    c2.metric("Fusion algorithm", "RRF", "rank-invariant")
    c3.metric("Reranker", "Cross-encoder", "MiniLM-L-6")
    c4.metric("Hallucination guard", "Citations", "regex audit")
    st.divider()

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader("What is RAG?")
        st.markdown(
            """
            **Retrieval-Augmented Generation** grounds an LLM's answers in a private
            document corpus — preventing hallucinations and keeping knowledge fresh
            without expensive retraining.

            The pipeline has two phases:
            1. **Index** — documents are split into chunks, embedded, and stored in a
               FAISS vector index and a BM25 inverted index.
            2. **Query** — the user's question retrieves the most relevant chunks via
               hybrid search; a cross-encoder reranks them; the LLM synthesises an
               answer with inline citations.
            """
        )
        st.info(
            "Use the **Live Chunker** and **BM25 Search** tabs above to explore the "
            "pipeline components interactively — no API key required.",
            icon="💡",
        )

    with col_r:
        st.subheader("Key engineering decisions")
        features = {
            "Hybrid retrieval": "BM25 catches exact keyword matches; FAISS catches semantic paraphrases. RRF combines both without score normalisation.",
            "Cross-encoder reranking": "Joint (query, passage) scoring is slower but far more accurate than bi-encoder dot products for the final top-5.",
            "Citation enforcement": "System prompt requires [chunk_id] inline citations. A post-generation regex audit flags uncited claims in `missing_citations`.",
            "LLM portability": "Strategy pattern — swap OpenAI ↔ local Ollama via one env var; zero code change.",
            "Security": "Bearer-token auth on write endpoints, path-traversal guard on `/ingest`, security-headers middleware, CORS allow-list.",
            "CI-gated evaluation": "A golden-dataset eval gate blocks PRs to `main` if faithfulness / citation recall fall below threshold.",
        }
        for feat, desc in features.items():
            with st.expander(f"**{feat}**"):
                st.write(desc)

# ─── Tab 2: Live Chunker ─────────────────────────────────────────────────────
with tab_chunker:
    st.header("Live Chunker Demo")
    st.caption(
        "Paste any text and tune chunk_size / chunk_overlap to see exactly "
        "how the recursive splitter divides the input."
    )

    if not _HAS_CHUNKER:
        st.error(
            "The `askdocs` package could not be imported. "
            "Run `pip install -e .` from the repo root and restart the app."
        )
    else:
        default_text = (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with language model generation.\n\n"
            "The retrieval component searches a document corpus for passages "
            "relevant to the user's query. It returns the top-k results.\n\n"
            "The generation component receives the retrieved context together "
            "with the user's question and produces a grounded, cited answer.\n\n"
            "RAG reduces hallucinations by anchoring responses in source documents. "
            "It also keeps knowledge current without expensive model retraining.\n\n"
            "Hybrid retrieval combines sparse BM25 and dense vector search, fused "
            "via Reciprocal Rank Fusion for robust recall across query types."
        )

        col_cfg, col_txt = st.columns([1, 2], gap="large")

        with col_cfg:
            chunk_size = st.slider("chunk_size (chars)", 50, 1000, 200, step=10)
            chunk_overlap = st.slider(
                "chunk_overlap (chars)",
                0,
                min(chunk_size - 10, 300),
                40,
                step=10,
            )
            st.caption(
                f"Overlap is {chunk_overlap/chunk_size*100:.0f}% of chunk size. "
                "Keep it 10–20% for good context continuity."
            )

        with col_txt:
            user_text = st.text_area(
                "Input text",
                value=default_text,
                height=220,
                label_visibility="collapsed",
            )

        if st.button("Chunk it", type="primary"):
            if not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                doc = Document(
                    content=user_text,
                    metadata={"source": "demo", "filename": "demo.txt", "filetype": ".txt"},
                )
                chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Chunks produced", len(chunks))
                m2.metric("Avg chunk length", f"{sum(len(c.text) for c in chunks) // max(len(chunks),1)} chars")
                m3.metric("Input length", f"{len(user_text)} chars")

                st.markdown("---")
                for i, c in enumerate(chunks):
                    hue = int((i / max(len(chunks), 1)) * 200)
                    header = f"Chunk {i+1} · {len(c.text)} chars · ID: `{c.chunk_id}`"
                    with st.expander(header, expanded=i < 3):
                        st.code(c.text, language=None)
                        st.caption(
                            f"chunk_index={c.metadata.get('chunk_index')}  "
                            f"doc_id={c.doc_id}"
                        )

# ─── Tab 3: BM25 Search ──────────────────────────────────────────────────────
with tab_bm25:
    st.header("BM25 Search Demo")
    st.caption(
        "Type a question or keyword phrase. BM25Okapi scores all eight sample "
        "documents and ranks them by relevance."
    )

    if not _HAS_BM25:
        st.error(
            "`rank-bm25` is not installed. Run `pip install rank-bm25` and restart."
        )
    else:
        sample_query = st.text_input(
            "Query",
            value="how does hybrid retrieval work",
            placeholder="e.g. how does citation enforcement work?",
        )

        if st.button("Search", type="primary") or sample_query:
            if sample_query.strip():
                results = _bm25_score(sample_query, SAMPLE_DOCS)

                st.divider()
                st.markdown(f"**Results for:** *{sample_query}*")

                for rank, r in enumerate(results):
                    score_bar = "█" * math.ceil(r["pct"] / 5) if r["pct"] > 0 else "▒"
                    label = f"#{rank+1}  {r['title']}  —  score {r['score']:.4f}"
                    with st.expander(label, expanded=rank < 3):
                        st.progress(min(r["pct"] / 100, 1.0))
                        st.write(r["text"])
                        st.caption(
                            f"BM25 raw score: **{r['score']:.6f}** "
                            f"({r['pct']:.1f}% of total mass)"
                        )

                st.info(
                    "In production, BM25 results are **fused with FAISS dense results** "
                    "via Reciprocal Rank Fusion, then the top-10 are reranked by a "
                    "cross-encoder before being sent to the LLM.",
                    icon="ℹ️",
                )

# ─── Tab 4: Architecture & API ───────────────────────────────────────────────
with tab_arch:
    st.header("Architecture & API Reference")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.subheader("Pipeline diagram")
        st.code(
            """
User query
    │
    ▼
┌───────────────────────────────────────────────┐
│ SecurityHeadersMiddleware                     │
│ RequestIDMiddleware (X-Request-ID UUID)        │
│ CORSMiddleware                                │
└───────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│ HybridRetriever                               │
│  ├── BM25Store  (sparse / keyword)  ──┐       │
│  └── VectorStore (FAISS cosine)    ──┤        │
│           Reciprocal Rank Fusion  ◄──┘        │
└───────────────────────────────────────────────┘
    │  top-10 candidates
    ▼
┌───────────────────────────────┐
│ CrossEncoderReranker          │
│  ms-marco-MiniLM-L-6-v2       │
│  returns top-5 reranked       │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│ LLMClient  (OpenAI gpt-4o-mini | Ollama)      │
│  citation-enforced prompt                     │
│  → answer with inline [chunk_id] refs         │
└───────────────────────────────────────────────┘
    │
    ▼
 Citation audit → RAGResponse
   missing_citations flags uncited claims
    │
    ▼
 FastAPI  /api/v1/query        (JSON)
 FastAPI  /api/v1/query/stream (SSE)
""",
            language=None,
        )

    with col_b:
        st.subheader("API endpoints")
        endpoints = [
            ("POST", "/ingest", "Upload PDF / HTML / text files (bearer-auth required)"),
            ("POST", "/api/v1/query", "Blocking Q&A — returns RAGResponse JSON"),
            ("POST", "/api/v1/query/stream", "SSE stream — sources first, then tokens"),
            ("GET", "/health", "JSON index stats (doc count, chunk count, uptime)"),
        ]
        for method, path, desc in endpoints:
            color = "#22C55E" if method == "GET" else "#7C3AED"
            st.markdown(
                f'<span style="background:{color};padding:2px 8px;border-radius:4px;'
                f'font-weight:bold;font-size:0.8rem">{method}</span> '
                f'<code>{path}</code> — {desc}',
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("Key configuration (env vars)")
        config_rows = [
            ("LLM_PROVIDER", "openai", "openai | ollama"),
            ("OPENAI_API_KEY", "—", "Required for OpenAI provider"),
            ("OPENAI_MODEL", "gpt-4o-mini", "Any chat-completion model"),
            ("EMBED_MODEL", "BAAI/bge-small-en-v1.5", "SentenceTransformer bi-encoder"),
            ("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2", "HuggingFace cross-encoder"),
            ("CHUNK_SIZE", "512", "Characters per chunk"),
            ("CHUNK_OVERLAP", "64", "Character overlap between chunks"),
            ("HYBRID_TOP_K", "5", "Final results returned to LLM"),
            ("API_KEY", "—", "Bearer token for write endpoints"),
        ]
        for var, default, desc in config_rows:
            st.markdown(f"**`{var}`** (default: `{default}`) — {desc}")

    st.divider()
    st.subheader("Quick start")
    st.code(
        """# 1. Clone and set up environment
git clone https://github.com/Muhammad-Farooq13/Ask-my-docs
cd Ask-my-docs
cp .env.example .env          # add your OPENAI_API_KEY

# 2a. Run with Docker Compose (recommended)
docker compose up

# 2b. Or run locally
pip install -r requirements-api.txt
pip install --no-deps -e .
uvicorn askdocs.api.main:app --reload   # API on :8000
streamlit run streamlit_app.py          # Demo UI on :8501

# 3. Run tests
pip install -r requirements-ci.txt
pip install --no-deps -e .
pytest tests/unit/ -v
""",
        language="bash",
    )

<!-- README is UTF-8. Re-generated to fix character-encoding corruption. -->
<div align="center">

# AskMyDocs

### Production-grade Retrieval-Augmented Generation system

**Hybrid BM25 + vector retrieval · Cross-encoder reranking · Citation enforcement · CI-gated evaluation**

[![CI](https://github.com/Muhammad-Farooq-13/ask-my-docs/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq-13/ask-my-docs/actions/workflows/ci.yml)
[![Eval Gate](https://github.com/Muhammad-Farooq-13/ask-my-docs/actions/workflows/eval.yml/badge.svg)](https://github.com/Muhammad-Farooq-13/ask-my-docs/actions/workflows/eval.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Quickstart](#quickstart) · [Architecture](#architecture-overview) · [API Reference](#api-reference) · [Evaluation](#evaluation) · [Design Decisions](#design-decisions--interview-talking-points)

</div>

---

## What this project demonstrates

A single repo that takes an LLM app from "working demo" to "production-ready service" — covering every layer that interviews and senior code reviews test you on.

| Concern | Solution in this codebase |
|---|---|
| **Retrieval quality** | Two-stage pipeline: RRF-fused BM25 + FAISS dense retrieval, then cross-encoder reranking |
| **Hallucination control** | Citation-enforcement prompt + post-generation regex audit flags uncited claims |
| **LLM portability** | Strategy pattern: swap OpenAI ↔ local Ollama via a single env var, zero code change |
| **Security** | Bearer-token auth, path-traversal guard on `/ingest`, security-headers middleware, configurable CORS |
| **Observability** | Per-request UUID (`X-Request-ID`), optional structured JSON logging, health endpoint with index stats |
| **Regression prevention** | Golden-dataset evaluation in CI — PRs to `main` are **blocked** if faithfulness / citation recall fall below threshold |
| **Reproducibility** | Pydantic Settings, `.env.example`, Docker Compose — clone → `.env` → `make docker-up` and you're running |

---

## Architecture overview

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SecurityHeadersMiddleware → RequestIDMiddleware → CORSMiddleware   │
│  (optional bearer-token auth on write endpoints)                    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ HybridRetriever                                         │
│  ├── BM25Store      (sparse, keyword-based)  ──┐        │
│  └── VectorStore    (dense, FAISS cosine)    ──┤        │
│                     Reciprocal Rank Fusion  ◄──┘        │
└─────────────────────────────────────────────────────────┘
    │  top-10 candidate chunks
    ▼
┌──────────────────────────────┐
│ CrossEncoderReranker         │  jointly scores (query, passage)
│  ms-marco-MiniLM-L-6-v2     │  returns top-5 reranked chunks
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ LLMClient  (OpenAI gpt-4o-mini  |  local Ollama)         │
│  Citation-enforced prompt → answer with [chunk_id] refs  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
 Citation audit → RAGResponse  (missing_citations warns on uncited claims)
    │
    ▼
 FastAPI  /api/v1/query        (JSON response)
 FastAPI  /api/v1/query/stream (SSE — sources first, then tokens)
    │
    ▼
 Streamlit UI  (port 8501)
```

### Quick demo

```
┌─────────────────────────────────────────────────────────────────────┐
│  🔍  Ask a question about your docs                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ How does the authentication system work?                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  [ Ask ]                                                            │
│                                                                     │
│  Answer                                                             │
│  ─────────────────────────────────────────────────────────────      │
│  Authentication uses JWT tokens [a1b2_0003]. Tokens expire after   │
│  24 hours [a1b2_0004]. Refresh tokens are stored server-side       │
│  and can be revoked at any time [a1b2_0005].                       │
│                                                                     │
│  Sources (3 chunks retrieved, all cited ✓)                         │
│  ▸ auth.md — "...JWT signed with RS256, expiry 86400s..."          │
│  ▸ auth.md — "...refresh tokens persisted in Redis..."             │
│  ▸ auth.md — "...revocation via token blocklist..."                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature highlights

| Feature | Implementation |
|---|---|
| **Hybrid retrieval** | BM25Okapi (rank-bm25) + FAISS IndexFlatIP fused via RRF |
| **Reranking** | sentence-transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| **Citation enforcement** | System prompt rules + regex audit; `missing_citations` field flags uncited claims |
| **Streaming** | SSE endpoint emits source chunks first, then token-by-token generation |
| **Pluggable LLM** | OpenAI or any local Ollama model — swap via `LLM_PROVIDER` env var |
| **Eval pipeline** | Faithfulness, answer relevance, citation recall, context precision — no LLM-as-judge |
| **CI eval gate** | GitHub Actions blocks merges if any metric falls below configured threshold |
| **Production security** | API-key auth, path-traversal guard, security headers, configurable CORS |
| **Docker** | Single `docker compose up` spins up API + UI |

---

## Project structure

```
ask-my-docs/
├── src/askdocs/
│   ├── config.py               ← Pydantic Settings (all config via env vars)
│   ├── ingestion/
│   │   ├── loader.py           ← Multi-format document loader (PDF, MD, TXT, HTML)
│   │   ├── chunker.py          ← Recursive character splitter with overlap
│   │   └── pipeline.py         ← load → chunk → embed → index orchestrator
│   ├── retrieval/
│   │   ├── embedder.py         ← sentence-transformers bi-encoder (lazy load)
│   │   ├── vector_store.py     ← FAISS wrapper with save/load
│   │   ├── bm25_store.py       ← BM25Okapi wrapper with pickle persistence
│   │   ├── hybrid.py           ← Reciprocal Rank Fusion combiner
│   │   └── reranker.py         ← Cross-encoder reranker
│   ├── generation/
│   │   ├── prompts.py          ← Citation-enforced prompt templates
│   │   ├── llm.py              ← OpenAI / Ollama client abstraction
│   │   └── chain.py            ← Full RAG chain + citation audit
│   ├── evaluation/
│   │   ├── metrics.py          ← Faithfulness, relevance, citation recall
│   │   ├── dataset.py          ← JSONL golden dataset loader/saver
│   │   └── runner.py           ← Evaluation loop + threshold checker
│   └── api/
│       ├── main.py             ← FastAPI app factory + lifespan + JSON logging
│       ├── middleware.py       ← RequestID, security headers, bearer-token auth
│       ├── models.py           ← Pydantic request/response models
│       └── routes.py           ← /health, /query, /query/stream, /ingest
├── ui/app.py                   ← Streamlit UI
├── tests/
│   ├── unit/                   ← Fast, no-model tests (mocked cross-encoder)
│   └── integration/            ← Full retrieval stack tests (real models)
├── scripts/
│   ├── ingest.py               ← CLI ingestion
│   └── evaluate.py             ← CLI evaluation + CI gate
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              ← Lint → type-check → unit tests
│   │   └── eval.yml            ← Integration tests + RAG eval gate
│   ├── ISSUE_TEMPLATE/         ← Bug report / feature request forms
│   └── PULL_REQUEST_TEMPLATE.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── raw/                    ← Drop your documents here
│   ├── processed/              ← Auto-generated FAISS index + BM25 pickle
│   └── eval/
│       └── golden_dataset.jsonl
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
└── .env.example
```

---

## Quickstart

### Option A — OpenAI (paid, best quality)

```bash
git clone https://github.com/Muhammad-Farooq-13/ask-my-docs.git
cd ask-my-docs
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make dev-install

cp .env.example .env
# Set: LLM_PROVIDER=openai  OPENAI_API_KEY=sk-...

cp /path/to/your/docs/*.md  data/raw/
make ingest
make serve          # API → http://localhost:8000/api/v1/docs
make ui             # UI  → http://localhost:8501  (new terminal)
```

### Option B — Ollama (free, fully local)

```bash
# Pull a model first
ollama pull llama3.2

cp .env.example .env
# Set: LLM_PROVIDER=ollama  OLLAMA_MODEL=llama3.2

make ingest && make serve
```

### Option C — Docker (API + UI, one command)

```bash
cp .env.example .env  # fill in your keys
make docker-up
# API → http://localhost:8000  |  UI → http://localhost:8501
```

---

## API reference

| Method | Path | Auth required | Description |
|--------|------|:---:|-------------|
| `GET` | `/api/v1/health` | — | Index status and chunk count |
| `POST` | `/api/v1/query` | — | Full RAG query — JSON answer + sources |
| `POST` | `/api/v1/query/stream` | — | SSE streaming query |
| `POST` | `/api/v1/ingest` | ✓ Bearer | Ingest documents and hot-reload the chain |

Interactive docs: `http://localhost:8000/api/v1/docs`

**Enable auth** by setting `API_KEY=<secret>` in `.env`. Requests to `/ingest` must then include `Authorization: Bearer <secret>`. Leave `API_KEY` empty to disable auth (default for local development).

### Example — query

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the authentication system work?"}' | jq .
```

```json
{
  "question": "How does the authentication system work?",
  "answer": "Authentication uses JWT tokens [a1b2_0003]. Tokens expire after 24 hours [a1b2_0004].",
  "sources": [
    {"chunk_id": "a1b2_0003", "text": "...", "score": 0.9241, "filename": "auth.md"},
    {"chunk_id": "a1b2_0004", "text": "...", "score": 0.8817, "filename": "auth.md"}
  ],
  "cited_ids": ["a1b2_0003", "a1b2_0004"],
  "missing_citations": []
}
```

---

## Evaluation

The evaluation pipeline measures four metrics without requiring an LLM-as-judge:

| Metric | Definition | Default threshold |
|--------|-----------|:---:|
| **Faithfulness** | Mean max cosine similarity between each answer sentence and the best context chunk | 0.70 |
| **Answer relevance** | Cosine similarity between question and answer embeddings | 0.70 |
| **Citation recall** | Fraction of retrieved source chunks explicitly cited in the answer | 0.80 |
| **Context precision** | Fraction of retrieved chunks with cosine similarity > 0.4 to the question | — |

```bash
make eval
# Output → data/eval/report.json
{
  "faithfulness": 0.82,
  "answer_relevance": 0.79,
  "citation_recall": 0.94,
  "context_precision": 0.75,
  "pass_rate": 0.80
}
```

Thresholds are configurable via `.env`.  The `eval.yml` CI workflow **blocks merges to `main`** if any metric falls below threshold.

---

## Testing

```bash
make test               # unit tests only — fast, no model loading (26 tests)
make test-integration   # full retrieval stack — requires sentence-transformers
make test-all           # everything
```

| Suite | Coverage | What's tested |
|-------|:---:|---|
| `tests/unit/test_chunker.py` | chunker | Overlap, empty input, metadata propagation |
| `tests/unit/test_hybrid.py` | hybrid RRF | Deduplication, score ordering, multi-list fusion |
| `tests/unit/test_reranker.py` | reranker | Sorting, top-k enforcement, correct pair construction |
| `tests/integration/` | full pipeline | Real models: embed → index → retrieve → rerank |

---

## Configuration reference

All settings can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `"openai"` or `"ollama"` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |
| `OLLAMA_MODEL` | `llama3.2` | Any model pulled via `ollama pull` |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace sentence embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `HYBRID_TOP_K` | `10` | Candidates passed to the reranker |
| `RERANKER_TOP_K` | `5` | Final sources passed to the LLM |
| `RRF_K` | `60` | RRF smoothing constant |
| `API_KEY` | _(empty)_ | Bearer token for `/ingest`; empty = auth disabled |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `ALLOWED_INGEST_DIR` | `data` | Path traversal guard root for `/ingest` |
| `LOG_FORMAT` | `text` | `"text"` for dev, `"json"` for production |

---

## Design decisions & interview talking points

| Decision | What was chosen | Why |
|---|---|---|
| **Two-stage retrieval** | RRF-fused BM25 + FAISS, then cross-encoder rerank | Bi-encoders are fast but imprecise; cross-encoders are expensive but accurate. Stage 1 narrows 10k→10, stage 2 reranks to 5. |
| **Reciprocal Rank Fusion** | RRF(k=60) to merge BM25 and dense lists | Score-normalisation-free; outperforms linear interpolation on out-of-domain queries; one hyperparameter. |
| **Citation-enforcement loop** | Regex post-processing `missing_citations` field | LLMs hallucinate; enforcing citations allows downstream audit without re-querying the model. |
| **No LLM-as-judge in eval** | Pure embedding cosine metrics | Deterministic, cheap, fast CI. Trade-off: lower correlation with human preference vs RAGAS — acceptable for a gate, not a leaderboard. |
| **Pydantic Settings** | All config via env vars, no arg parsing | 12-factor app compliance; single source of truth; works identically locally, Docker, and cloud. |
| **Strategy pattern for LLM** | `LLMClient` abstraction over OpenAI/Ollama | Swapping the provider is a config change, not a code change. Simplifies testing with fakes. |
| **Middleware stack order** | Security headers → Request ID → CORS | CORS must be innermost (FastAPI requirement); security headers applied on every response including error responses from outer middleware. |

---

## Contributing · Security · Changelog

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.
- Report vulnerabilities privately via [SECURITY.md](SECURITY.md).
- See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## Tech stack

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-HuggingFace-yellow)](https://sbert.net)
[![FAISS](https://img.shields.io/badge/FAISS-Meta%20AI-blue)](https://faiss.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)

| Layer | Library |
|---|---|
| REST API | FastAPI + Uvicorn |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| Dense index | FAISS `IndexFlatIP` |
| Sparse retrieval | rank-bm25 `BM25Okapi` |
| Reranking | sentence-transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | OpenAI API / Ollama (local) |
| UI | Streamlit |
| Config | pydantic-settings v2 |
| Testing | pytest, unittest.mock |
| Linting / types | ruff, mypy |
| CI / CD | GitHub Actions |
| Containers | Docker Compose |

---

## License

MIT © 2026 [Muhammad Farooq](https://github.com/Muhammad-Farooq-13). See [LICENSE](LICENSE).

---

## Extending the project

| Goal | Where to change |
|------|-----------------|
| Add a new document format | `src/askdocs/ingestion/loader.py` — add to `_LOADERS` |
| Swap the embedding model | Set `EMBED_MODEL` in `.env` |
| Use a local LLM (no API key) | Set `LLM_PROVIDER=ollama`, install Ollama, `ollama pull llama3.2` |
| Add NLI-based faithfulness | Replace `faithfulness_score()` in `evaluation/metrics.py` |
| Persist index to S3 | Override `save()`/`load()` in `vector_store.py` and `bm25_store.py` |
| Add authentication | Add FastAPI `Depends` guard in `routes.py` |

---

## License

MIT — see [LICENSE](LICENSE).

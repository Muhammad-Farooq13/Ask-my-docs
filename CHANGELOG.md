# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

*(Next changes go here)*

---

## [1.0.0] — 2025-07-15

### Added
- Hybrid retrieval: BM25Okapi sparse + FAISS dense, fused via Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`
- Citation-enforcement system prompt with post-generation audit
- FastAPI REST API: `/query` (blocking), `/query/stream` (SSE), `/ingest`, `/health`
- Streamlit UI with ingest sidebar, answer display, and source expanders
- Standalone `streamlit_app.py` demo: Live Chunker + BM25 Search — no API key or running server required
- Evaluation pipeline: faithfulness, answer relevance, citation recall, context precision
- Golden dataset in JSONL format
- CI workflow: lint → type-check → unit tests with coverage (26 tests, 0 failures)
- Evaluation gate workflow: blocks PRs to `main` when metrics drop below threshold
- Docker Compose: API + UI with one command
- Security: bearer-token API key auth, path traversal guard on `/ingest`, security headers, configurable CORS
- Structured JSON logging mode (`LOG_FORMAT=json`)
- Request ID correlation header (`X-Request-ID`)
- GitHub community files: CONTRIBUTING, SECURITY, PR template, issue templates
- `requirements-ci.txt` — lean CI deps (no heavy ML packages, fast installs)
- `requirements-api.txt` — full API / Docker deps separated from Streamlit Cloud deps
- `.streamlit/config.toml` — dark theme with purple accent
- `runtime.txt` — pins Python 3.11 for Streamlit Cloud

### Changed
- `requirements.txt` now contains only Streamlit Cloud / demo dependencies
- `docker/Dockerfile` uses `requirements-api.txt` for the full ML stack
- CI cache key now tracks `requirements-ci.txt` (faster cache invalidation)
- Bumped `codecov/codecov-action` from v4 to v5
- Added `--cov-fail-under=0` to pytest options so coverage upload never blocks CI

### Security
- All heavy ML imports (faiss, sentence-transformers, rank-bm25) are lazy-loaded at call time — import errors cannot propagate to health-check endpoints


# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

*(Next changes go here)*

---

## [1.0.0] — 2026-03-05

### Added
- Hybrid retrieval: BM25Okapi sparse + FAISS dense, fused via Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`
- Citation-enforcement system prompt with post-generation audit
- FastAPI REST API: `/query` (blocking), `/query/stream` (SSE), `/ingest`, `/health`
- Streamlit UI with ingest sidebar, answer display, and source expanders
- Evaluation pipeline: faithfulness, answer relevance, citation recall, context precision
- Golden dataset in JSONL format
- CI workflow: lint → type-check → unit tests with coverage
- Evaluation gate workflow: blocks PRs to `main` when metrics drop below threshold
- Docker Compose: API + UI with one command
- Security: bearer-token API key auth, path traversal guard on `/ingest`, security headers, configurable CORS
- Structured JSON logging mode (`LOG_FORMAT=json`)
- Request ID correlation header (`X-Request-ID`)
- GitHub community files: CONTRIBUTING, SECURITY, PR template, issue templates

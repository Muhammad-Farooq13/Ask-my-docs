.PHONY: install dev-install test test-integration test-all lint lint-fix type-check ingest eval serve ui docker-up docker-down clean

install:
	pip install -r requirements.txt
	pip install -e .

dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/unit/ -v --tb=short --cov=src/askdocs --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v --tb=short

test-all:
	pytest tests/ -v --tb=short --cov=src/askdocs

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

type-check:
	mypy src/askdocs --ignore-missing-imports

# ── Data pipeline ─────────────────────────────────────────────────────────────
ingest:
	python scripts/ingest.py --source data/raw --reset

eval:
	python scripts/evaluate.py --output data/eval/report.json --fail-on-threshold

# ── Run locally ───────────────────────────────────────────────────────────────
serve:
	uvicorn askdocs.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/app.py

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name "*.pyc" -delete
	rm -rf .coverage coverage.xml .mypy_cache .ruff_cache dist build

FROM python:3.11-slim

WORKDIR /app

# System dependencies for faiss-cpu and pypdf
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and install package
COPY . .
RUN pip install --no-cache-dir -e .

# Pre-download embedding + reranker models so the container starts instantly
# Comment out these lines to keep the image smaller (models download on first request)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

CMD ["uvicorn", "askdocs.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

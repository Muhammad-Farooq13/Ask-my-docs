#!/usr/bin/env python3
"""CLI — run the RAG evaluation pipeline and optionally gate on metric thresholds."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AskMyDocs RAG pipeline against a golden dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to JSONL golden dataset (defaults to settings.eval_dataset_path)",
    )
    parser.add_argument(
        "--output",
        default="data/eval/report.json",
        help="Path to write the JSON evaluation report",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with code 1 if any metric is below its configured threshold",
    )
    args = parser.parse_args()

    from askdocs.config import settings
    from askdocs.evaluation.runner import check_thresholds, run_evaluation
    from askdocs.generation.chain import RAGChain
    from askdocs.generation.llm import LLMClient
    from askdocs.retrieval.bm25_store import BM25Store
    from askdocs.retrieval.embedder import Embedder
    from askdocs.retrieval.hybrid import HybridRetriever
    from askdocs.retrieval.reranker import CrossEncoderReranker
    from askdocs.retrieval.vector_store import VectorStore

    # Load existing index
    try:
        embedder = Embedder()
        vs = VectorStore.load(settings.vector_store_path)
        bm25 = BM25Store.load(settings.bm25_store_path)
    except FileNotFoundError:
        logging.getLogger(__name__).error(
            "No index found. Run `python scripts/ingest.py` first."
        )
        sys.exit(1)

    retriever = HybridRetriever(vs, bm25, embedder)
    reranker = CrossEncoderReranker()
    llm = LLMClient()
    chain = RAGChain(retriever, reranker, llm)

    dataset_path = Path(args.dataset) if args.dataset else None
    report = run_evaluation(
        chain=chain,
        embedder=embedder,
        dataset_path=dataset_path,
        output_path=Path(args.output),
    )

    print("\n=== Evaluation Report ===")
    print(json.dumps(report["aggregate"], indent=2))

    if args.fail_on_threshold and not check_thresholds(report):
        sys.exit(1)


if __name__ == "__main__":
    main()

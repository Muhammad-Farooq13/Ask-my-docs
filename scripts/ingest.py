#!/usr/bin/env python3
"""CLI — ingest documents into the FAISS vector store and BM25 index."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into AskMyDocs indices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a file or directory of documents to ingest",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing index before ingesting",
    )
    args = parser.parse_args()

    from askdocs.ingestion.pipeline import run_ingestion

    try:
        vs, bm25 = run_ingestion(
            source=Path(args.source),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            reset=args.reset,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("Ingestion failed: %s", exc)
        sys.exit(1)

    print(f"✓ Indexed {len(vs._texts)} chunks successfully.")


if __name__ == "__main__":
    main()

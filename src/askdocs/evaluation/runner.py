"""
Evaluation runner — iterates over a golden dataset, collects metrics,
writes a JSON report, and returns a pass/fail boolean for the CI gate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from askdocs.config import settings
from askdocs.evaluation.dataset import GoldenSample, load_golden_dataset
from askdocs.evaluation.metrics import EvalResult, evaluate
from askdocs.generation.chain import RAGChain

logger = logging.getLogger(__name__)


def run_evaluation(
    chain: RAGChain,
    embedder,
    samples: list[GoldenSample] | None = None,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Run the full evaluation loop.

    Returns a report dict with keys:
      - "aggregate": averaged metrics across all samples
      - "samples":   per-sample results including the generated answer
    """
    if samples is None:
        path = dataset_path or settings.eval_dataset_path
        samples = load_golden_dataset(path)

    if not samples:
        raise ValueError("Evaluation dataset is empty.")

    per_sample: list[dict] = []
    for sample in samples:
        response = chain.run(sample.question)
        contexts = [s.text for s in response.sources]
        source_ids = [s.chunk_id for s in response.sources]

        metric: EvalResult = evaluate(
            question=sample.question,
            answer=response.answer,
            source_ids=source_ids,
            contexts=contexts,
            embedder=embedder,
        )
        passed = metric.passes_thresholds(
            faithfulness_thr=settings.eval_faithfulness_threshold,
            relevance_thr=settings.eval_relevance_threshold,
            citation_thr=settings.eval_citation_threshold,
        )
        per_sample.append(
            {
                "question": sample.question,
                "answer": response.answer,
                "metrics": metric.to_dict(),
                "passes": passed,
            }
        )
        logger.info(
            "[eval] %s | %s | pass=%s",
            sample.question[:60],
            metric.to_dict(),
            passed,
        )

    n = len(per_sample)
    aggregate = {
        "faithfulness": sum(r["metrics"]["faithfulness"] for r in per_sample) / n,
        "answer_relevance": sum(r["metrics"]["answer_relevance"] for r in per_sample) / n,
        "citation_recall": sum(r["metrics"]["citation_recall"] for r in per_sample) / n,
        "context_precision": sum(r["metrics"]["context_precision"] for r in per_sample) / n,
        "pass_rate": sum(1 for r in per_sample if r["passes"]) / n,
        "n_samples": n,
    }

    report = {"aggregate": aggregate, "samples": per_sample}

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Eval report written to %s", out)

    return report


def check_thresholds(report: dict) -> bool:
    """Return True if ALL aggregate metrics meet the configured thresholds."""
    agg = report["aggregate"]
    ok = (
        agg["faithfulness"] >= settings.eval_faithfulness_threshold
        and agg["answer_relevance"] >= settings.eval_relevance_threshold
        and agg["citation_recall"] >= settings.eval_citation_threshold
    )
    if ok:
        logger.info("✓ All evaluation thresholds met: %s", agg)
    else:
        logger.error("✗ Evaluation thresholds NOT met: %s", agg)
    return ok

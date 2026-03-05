"""
RAG evaluation metrics (embedding-based, no external judge API required).

Metrics implemented:
- faithfulness       — are all answer sentences grounded in the context?
- answer_relevance   — is the answer semantically close to the question?
- citation_recall    — what fraction of source chunks are explicitly cited?
- context_precision  — what fraction of retrieved chunks are actually relevant?
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_CITATION_RE = re.compile(r"\[([^\[\]\s]+)\]")


@dataclass
class EvalResult:
    faithfulness: float        # 0–1
    answer_relevance: float    # 0–1
    citation_recall: float     # 0–1
    context_precision: float   # 0–1

    def passes_thresholds(
        self,
        faithfulness_thr: float = 0.7,
        relevance_thr: float = 0.7,
        citation_thr: float = 0.8,
    ) -> bool:
        return (
            self.faithfulness >= faithfulness_thr
            and self.answer_relevance >= relevance_thr
            and self.citation_recall >= citation_thr
        )

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "citation_recall": round(self.citation_recall, 4),
            "context_precision": round(self.context_precision, 4),
        }


# ── Individual metric functions ───────────────────────────────────────────────

def citation_recall(answer: str, source_ids: list[str]) -> float:
    """Fraction of source chunk IDs that appear as citations in *answer*."""
    if not source_ids:
        return 1.0
    cited = set(_CITATION_RE.findall(answer))
    return len(cited & set(source_ids)) / len(source_ids)


def faithfulness_score(answer: str, contexts: list[str], embedder) -> float:
    """
    Heuristic faithfulness via sentence-level grounding.

    For each sentence in the answer, find the most similar context chunk
    (cosine sim on normalised embeddings).  Return the mean similarity.
    A proper NLI-based approach (e.g. TRUE / MiniCheck) is a drop-in upgrade.
    """
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", answer)
        if len(s.strip()) > 15
    ]
    if not sentences or not contexts:
        return 0.0

    ctx_emb = embedder.embed(contexts)   # (C, D) — already normalised
    scores: list[float] = []
    for sent in sentences:
        s_emb = embedder.embed_query(sent)            # (D,)
        sims = ctx_emb @ s_emb                        # (C,) cosine sims
        scores.append(float(np.max(sims)))
    return float(np.mean(scores))


def answer_relevance_score(question: str, answer: str, embedder) -> float:
    """Cosine similarity between the question and the answer embeddings."""
    q_emb = embedder.embed_query(question)
    a_emb = embedder.embed_query(answer)
    return float(np.dot(q_emb, a_emb))


def context_precision_score(
    question: str, contexts: list[str], embedder, threshold: float = 0.40
) -> float:
    """
    Fraction of retrieved chunks with cosine similarity > *threshold* to the
    question.  Measures whether the retriever surfaced useful context.
    """
    if not contexts:
        return 0.0
    q_emb = embedder.embed_query(question)
    ctx_emb = embedder.embed(contexts)
    sims = ctx_emb @ q_emb
    return float(np.mean(sims > threshold))


# ── Aggregate evaluator ───────────────────────────────────────────────────────

def evaluate(
    question: str,
    answer: str,
    source_ids: list[str],
    contexts: list[str],
    embedder,
) -> EvalResult:
    return EvalResult(
        faithfulness=faithfulness_score(answer, contexts, embedder),
        answer_relevance=answer_relevance_score(question, answer, embedder),
        citation_recall=citation_recall(answer, source_ids),
        context_precision=context_precision_score(question, contexts, embedder),
    )

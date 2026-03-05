"""Golden dataset — JSONL loader/saver for evaluation samples."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenSample:
    question: str
    expected_answer: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def load_golden_dataset(path: Path) -> list[GoldenSample]:
    """Load a JSONL file where each line is one evaluation sample."""
    samples: list[GoldenSample] = []
    with open(Path(path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(
                GoldenSample(
                    question=data["question"],
                    expected_answer=data.get("expected_answer", ""),
                    relevant_chunk_ids=data.get("relevant_chunk_ids", []),
                    metadata=data.get("metadata", {}),
                )
            )
    return samples


def save_golden_dataset(samples: list[GoldenSample], path: Path) -> None:
    """Serialise samples to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(
                json.dumps(
                    {
                        "question": s.question,
                        "expected_answer": s.expected_answer,
                        "relevant_chunk_ids": s.relevant_chunk_ids,
                        "metadata": s.metadata,
                    }
                )
                + "\n"
            )

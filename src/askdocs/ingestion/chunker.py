"""Recursive character text splitter with configurable size and overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from askdocs.ingestion.loader import Document

# Ordered from most-semantic to least-semantic break points
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


@dataclass
class Chunk:
    text: str
    chunk_id: str
    doc_id: str
    metadata: dict = field(default_factory=dict)


# ── Core splitter ─────────────────────────────────────────────────────────────

def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively split *text* at the highest-priority separator that fits.
    Falls back to finer-grained separators when a segment is still too large.
    """
    seps = separators if separators is not None else _SEPARATORS
    if not seps:
        # Character-level fallback
        return [text[i : i + chunk_size] for i in range(0, len(text), max(1, chunk_size - chunk_overlap))]

    sep, remaining_seps = seps[0], seps[1:]
    parts = re.split(re.escape(sep), text) if sep else list(text)

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                # Recurse with finer separators
                chunks.extend(_split_text(part, chunk_size, chunk_overlap, remaining_seps))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Stitch overlap between adjacent chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-chunk_overlap:]
            # Only prepend tail if it is not already present at the start
            new_chunk = (tail + chunks[i]) if not chunks[i].startswith(tail) else chunks[i]
            overlapped.append(new_chunk)
        return overlapped

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split *doc* into a list of Chunks with stable IDs and inherited metadata."""
    raw_chunks = _split_text(doc.content.strip(), chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for idx, text in enumerate(raw_chunks):
        text = text.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                text=text,
                chunk_id=f"{doc.doc_id}_{idx:04d}",
                doc_id=doc.doc_id,
                metadata={**doc.metadata, "chunk_index": idx, "doc_id": doc.doc_id},
            )
        )
    return chunks

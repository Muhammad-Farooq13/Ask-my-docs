"""Unit tests for the text chunker."""
from __future__ import annotations

from askdocs.ingestion.chunker import Chunk, _split_text, chunk_document
from askdocs.ingestion.loader import Document

# ── Helpers ────────────────────────────────────────────────────────────────────

def _doc(text: str, extra_meta: dict | None = None) -> Document:
    d = Document(
        content=text,
        metadata={"source": "test.txt", "filename": "test.txt", "filetype": ".txt"},
    )
    if extra_meta:
        d.metadata.update(extra_meta)
    return d


# ── _split_text ────────────────────────────────────────────────────────────────

def test_split_text_short_input_returns_single_chunk():
    result = _split_text("Hello world.", chunk_size=512, chunk_overlap=0)
    assert result == ["Hello world."]


def test_split_text_respects_chunk_size():
    text = "word " * 300  # 1500 chars
    parts = _split_text(text, chunk_size=100, chunk_overlap=0)
    for part in parts:
        assert len(part) <= 120, f"Chunk too large: {len(part)}"


def test_split_text_overlap_applied():
    # Build two paragraphs clearly separated by \n\n
    para1 = "A" * 60
    para2 = "B" * 60
    text = para1 + "\n\n" + para2
    parts = _split_text(text, chunk_size=80, chunk_overlap=20)
    # The second chunk should start with tail chars from para1
    assert len(parts) >= 2
    assert parts[1].startswith("A"), "Overlap not applied to second chunk"


def test_split_text_empty_string():
    assert _split_text("", chunk_size=100, chunk_overlap=0) == []


# ── chunk_document ─────────────────────────────────────────────────────────────

def test_chunk_document_short_text():
    chunks = chunk_document(_doc("Hello, world."), chunk_size=512, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello, world."


def test_chunk_ids_are_unique():
    text = " ".join(["token"] * 2000)
    chunks = chunk_document(_doc(text), chunk_size=100, chunk_overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


def test_chunk_ids_contain_doc_id():
    doc = _doc("Some content " * 10)
    chunks = chunk_document(doc, chunk_size=50)
    for c in chunks:
        assert c.chunk_id.startswith(doc.doc_id)


def test_metadata_propagated_to_all_chunks():
    doc = _doc("Content " * 50, extra_meta={"project": "test_project"})
    chunks = chunk_document(doc, chunk_size=100)
    for c in chunks:
        assert c.metadata["project"] == "test_project"
        assert c.metadata["doc_id"] == doc.doc_id
        assert "chunk_index" in c.metadata


def test_whitespace_only_doc_returns_no_chunks():
    chunks = chunk_document(_doc("   \n\n  \t "))
    assert chunks == []


def test_each_chunk_has_nonempty_text():
    text = "paragraph\n\n" * 30
    chunks = chunk_document(_doc(text), chunk_size=50)
    for c in chunks:
        assert c.text.strip(), "Empty chunk found"


def test_chunk_returns_correct_type():
    chunks = chunk_document(_doc("test text"))
    assert all(isinstance(c, Chunk) for c in chunks)

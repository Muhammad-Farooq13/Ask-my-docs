"""Multi-format document loader supporting .txt, .md, .rst, .pdf, .html."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ── Format-specific loaders ───────────────────────────────────────────────────

def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError("Install pypdf: pip install pypdf") from exc
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _load_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4") from exc
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    return soup.get_text(separator="\n")


_LOADERS = {
    ".txt": _load_text,
    ".md": _load_text,
    ".rst": _load_text,
    ".pdf": _load_pdf,
    ".html": _load_html,
    ".htm": _load_html,
}


# ── Public API ────────────────────────────────────────────────────────────────

def load_document(path: Path) -> Document:
    """Load a single file into a Document."""
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix!r}. Supported: {list(_LOADERS)}")
    content = loader(path)
    return Document(
        content=content,
        metadata={
            "source": str(path),
            "filename": path.name,
            "filetype": path.suffix.lower(),
        },
    )


def load_directory(directory: Path) -> Iterator[Document]:
    """Recursively yield Documents from all supported files in *directory*."""
    directory = Path(directory)
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            try:
                yield load_document(path)
            except Exception as exc:
                logger.warning("Skipping %s — %s", path, exc)

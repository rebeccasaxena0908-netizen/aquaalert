"""Load guideline documents (PDF / txt / md) from data/docs/ and split them into passages."""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

CHUNK_CHARS = 900
OVERLAP_CHARS = 150
MIN_CHARS = 60
SUFFIXES = {".pdf", ".txt", ".md"}


class Passage(BaseModel):
    text: str
    source: str  # e.g. "Flood Guidelines.pdf" or "Daily brief: Alappuzha 2018-08-15"
    kind: str = "guideline"  # "guideline" | "brief"
    page: int | None = None
    ref: str | None = None  # extra pointer, e.g. the post ids behind a brief

    def citation(self) -> str:
        return f"{self.source}, p. {self.page}" if self.page else self.source


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split on paragraph/sentence boundaries into ~``size``-char chunks with a little overlap."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    pieces = [s.strip() for s in re.split(r"(?<=[.!?।])\s+|\n{2,}", text) if s.strip()]
    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 1 > size:
            chunks.append(cur)
            cur = cur[-overlap:].split(" ", 1)[-1] if overlap else ""
        cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) >= MIN_CHARS]


def _pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return [(i + 1, page.extract_text() or "") for i, page in enumerate(PdfReader(str(path)).pages)]
    return [(None, path.read_text(errors="ignore"))]


def load_documents(docs_dir: str | Path) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(Path(docs_dir).glob("*")):
        if path.suffix.lower() not in SUFFIXES:
            continue
        try:
            pages = _pages(path)
        except Exception as exc:  # corrupt / encrypted PDF: skip the file, don't break the chatbot
            print(f"skipping {path.name}: {exc}")
            continue
        for page, text in pages:
            passages += [Passage(text=c, source=path.name, page=page) for c in chunk_text(text)]
    return passages


def docs_fingerprint(docs_dir: str | Path) -> str:
    """Changes when a document is added, removed or modified."""
    return "|".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}"
                    for p in sorted(Path(docs_dir).glob("*")) if p.suffix.lower() in SUFFIXES)

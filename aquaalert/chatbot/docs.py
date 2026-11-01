"""Load guideline documents (PDF / HTML / txt / md) from data/docs/ and split them into passages.

Optional ``data/docs/sources.json`` gives per-file metadata:
    {"ndma_floods_dos_donts.html": {"title": "NDMA Floods: Do's & Don'ts", "url": "https://...", "audience": "citizen"}}
``audience`` is "citizen" (household / personal-safety guidance) or "policy" (guidelines for governments).
Files without an entry get a guess from their name, size and wording.
"""
from __future__ import annotations

import html as html_lib
import json
import re
from pathlib import Path

from pydantic import BaseModel

# The multilingual MiniLM encoder reads at most 128 tokens; ~450 English chars is ~100 tokens, so a chunk is
# embedded in full (900-char chunks lost their last third). Part of the index fingerprint.
CHUNK_CHARS = 450
OVERLAP_CHARS = 100
MIN_CHARS = 60
SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm"}
SOURCES_FILE = "sources.json"
_CITIZEN_HINT = re.compile(
    r"do['’]?s\b|don['’]?ts|safety\s+tips|what\s+to\s+do|before\b.*\bduring\b.*\bafter\b|emergency\s+kit", re.I | re.S)


class Passage(BaseModel):
    text: str
    source: str  # file name, or "Daily brief: Alappuzha 2018-08-15"
    kind: str = "guideline"  # "guideline" | "brief"
    audience: str = "policy"  # "citizen" | "policy" (briefs: "situation")
    title: str | None = None
    url: str | None = None
    page: int | None = None
    ref: str | None = None  # extra pointer, e.g. the post ids behind a brief

    def citation(self) -> str:
        name = self.title or self.source
        return f"{name}, p. {self.page}" if self.page else name


# Small-caps running headers come out of PDFs letter-spaced ("na t i o n a l DiSaSt e r ma n a g e m e n t").
# a piece: 1-3 letters, a small-caps fragment with a lower->upper switch ("DiSaSt", "MaNa"), or bare punctuation
_SPACED_PIECE = re.compile(r"^[^\w\s]*(?:[A-Za-z]{1,3}|[A-Za-z]*[a-z][A-Z][A-Za-z]*)?[^\w\s]*$")
_REAL_SINGLE = {"a", "i", "A", "I", "&"}
# short real words a dropped run may have swallowed at its end ("... gu iDe l i n eS: The central ...")
_COMMON_SHORT = {"the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "and", "or", "is", "it", "as", "be",
                 "we", "if", "no", "do", "up", "so", "all", "any", "can", "may", "has", "are", "was", "not"}


def strip_spaced_headings(text: str) -> str:
    """Drop runs of letter-spaced fragments (>=5 short pieces, >=3 stray single letters); keep normal prose."""
    tokens = text.split(" ")
    out, run = [], []

    def flush() -> None:
        letters = [re.sub(r"[^A-Za-z]", "", t) for t in run]
        singles = sum(1 for t in letters if len(t) == 1 and t not in _REAL_SINGLE)
        if not (len(run) >= 5 and singles >= 3):
            out.extend(run)
        else:
            tail = []
            while run and run[-1].lower().strip(".,:;") in _COMMON_SHORT and not re.search(r"[a-z][A-Z]", run[-1]):
                tail.insert(0, run.pop())
            out.extend(tail)
        run.clear()

    for tok in tokens:
        if tok and _SPACED_PIECE.match(tok) and "\n" not in tok:
            run.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return re.sub(r" {2,}", " ", " ".join(out))


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


def html_to_text(raw: str) -> str:
    """Page title + main content of a saved web page (<article>, else <main>, else <body>), tags removed."""
    title = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I)
    title_line = " ".join(html_lib.unescape(title.group(1)).split()) if title else ""
    for tag in ("article", "main", "body"):
        m = re.search(rf"<{tag}\b.*?</{tag}>", raw, re.S | re.I)
        if m:
            raw = m.group(0)
            break
    raw = re.sub(r"<(script|style|nav|footer|header)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<(?:br|/p|/li|/h\d|/div|/tr)\b[^>]*>", "\n", raw, flags=re.I)
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw))
    body = "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())
    return f"{title_line}\n{body}" if title_line else body


def _pages(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return [(i + 1, strip_spaced_headings(page.extract_text() or ""))
                for i, page in enumerate(PdfReader(str(path)).pages)]
    raw = path.read_text(errors="ignore")
    return [(None, html_to_text(raw) if suffix in (".html", ".htm") else raw)]


def _metadata(docs_dir: Path) -> dict[str, dict]:
    path = docs_dir / SOURCES_FILE
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        print(f"ignoring malformed {path}")
        return {}


def guess_audience(name: str, pages: list[tuple[int | None, str]]) -> str:
    sample = name + " " + " ".join(t[:3000] for _, t in pages[:3])
    return "citizen" if len(pages) <= 6 and _CITIZEN_HINT.search(sample) or _CITIZEN_HINT.search(name) else "policy"


def load_documents(docs_dir: str | Path) -> list[Passage]:
    docs_dir = Path(docs_dir)
    meta = _metadata(docs_dir)
    passages: list[Passage] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() not in SUFFIXES or path.name == SOURCES_FILE:
            continue
        try:
            pages = _pages(path)
        except Exception as exc:  # corrupt / encrypted file: skip it, don't break the chatbot
            print(f"skipping {path.name}: {exc}")
            continue
        info = meta.get(path.name, {})
        audience = info.get("audience") or guess_audience(path.name, pages)
        for page, text in pages:
            passages += [Passage(text=c, source=path.name, audience=audience, title=info.get("title"),
                                 url=info.get("url"), page=page) for c in chunk_text(text)]
    return passages


def docs_fingerprint(docs_dir: str | Path) -> str:
    """Changes when a document (or sources.json) is added, removed or modified."""
    files = [p for p in sorted(Path(docs_dir).glob("*")) if p.suffix.lower() in SUFFIXES | {".json"}]
    chunking = f"chunk={CHUNK_CHARS}/{OVERLAP_CHARS}"
    return chunking + "|" + "|".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in files)

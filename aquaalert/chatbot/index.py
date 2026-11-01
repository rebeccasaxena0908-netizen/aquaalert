"""FAISS index over guideline passages (persisted) plus daily-brief passages (in memory)."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np

from aquaalert.chatbot.docs import Passage, docs_fingerprint, load_documents

Encoder = Callable[[list[str]], np.ndarray]  # texts -> L2-normalised (n, d) float array


def _index(passages: list[Passage], encoder: Encoder) -> faiss.Index | None:
    if not passages:
        return None
    emb = np.ascontiguousarray(encoder([p.text for p in passages]), dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])  # inner product == cosine on normalised vectors
    index.add(emb)
    return index


class KnowledgeBase:
    """Guideline documents (cached on disk) + latest daily briefs (rebuilt per query batch)."""

    def __init__(self, docs_dir: str | Path, cache_dir: str | Path, encoder: Encoder, encoder_id: str) -> None:
        self.docs_dir, self.cache_dir = Path(docs_dir), Path(cache_dir)
        self.encoder, self.encoder_id = encoder, encoder_id
        self.doc_passages: list[Passage] = []
        self.doc_index: faiss.Index | None = None
        self.brief_passages: list[Passage] = []
        self.brief_index: faiss.Index | None = None
        self._load_or_build_docs()

    def _load_or_build_docs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fp = hashlib.sha256(f"{self.encoder_id}#{docs_fingerprint(self.docs_dir)}".encode()).hexdigest()
        meta_path, idx_path, pas_path = (self.cache_dir / n for n in ("meta.json", "docs.faiss", "docs.json"))
        if meta_path.exists() and json.loads(meta_path.read_text()).get("fingerprint") == fp:
            self.doc_passages = [Passage(**p) for p in json.loads(pas_path.read_text())]
            self.doc_index = faiss.read_index(str(idx_path)) if idx_path.exists() else None
            return
        self.doc_passages = load_documents(self.docs_dir)
        self.doc_index = _index(self.doc_passages, self.encoder)
        pas_path.write_text(json.dumps([p.model_dump() for p in self.doc_passages]))
        if self.doc_index is not None:
            faiss.write_index(self.doc_index, str(idx_path))
        elif idx_path.exists():
            idx_path.unlink()
        meta_path.write_text(json.dumps({"fingerprint": fp, "n_passages": len(self.doc_passages)}))

    def set_briefs(self, passages: list[Passage]) -> None:
        self.brief_passages = passages
        self.brief_index = _index(passages, self.encoder)

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[tuple[Passage, float]]:
        """Top-``k`` passages across guidelines and briefs, best first."""
        q = np.ascontiguousarray(self.encoder([query]), dtype="float32")
        hits: list[tuple[Passage, float]] = []
        for index, passages in ((self.doc_index, self.doc_passages), (self.brief_index, self.brief_passages)):
            if index is None:
                continue
            scores, ids = index.search(q, min(k, index.ntotal))
            hits += [(passages[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0 and s >= min_score]
        return sorted(hits, key=lambda h: -h[1])[:k]

    @property
    def has_guidelines(self) -> bool:
        return bool(self.doc_passages)

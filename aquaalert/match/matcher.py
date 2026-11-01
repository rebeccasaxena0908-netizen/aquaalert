"""Needs-vs-offers matching: embedding cosine similarity combined with haversine proximity.

    python -m aquaalert.match.matcher data/sample_posts.csv
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel

from aquaalert.match.geo import haversine_km
from aquaalert.schemas import Label, Post

NEED_LABELS = {Label.RESCUE_NEEDED, Label.NEED_SUPPLIES}
OFFER_LABELS = {Label.OFFER_HELP}
Encoder = Callable[[list[str]], np.ndarray]  # texts -> L2-normalised (n, d) array


class Match(BaseModel):
    need_id: str
    offer_id: str
    need_text: str
    offer_text: str
    need_label: str
    need_place: str | None
    offer_place: str | None
    similarity: float  # cosine, clipped to [0, 1]
    distance_km: float
    proximity: float  # 1 at 0 km, 0 at >= max_km
    score: float  # alpha * similarity + (1 - alpha) * proximity


def hashing_encoder(texts: list[str], dim: int = 512) -> np.ndarray:
    """Dependency-free bag-of-words fallback (stable hashing); used if sentence-transformers is unavailable."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


def default_encoder() -> Encoder:
    """MiniLM multilingual sentence embeddings (MPS/CUDA if available), else the hashing fallback."""
    try:
        from aquaalert.classify.embed_lr import ENCODER, encode, load_encoder

        model = load_encoder(ENCODER)
    except Exception:
        return hashing_encoder
    return lambda texts: encode(texts, model)


def _located(posts: list[Post], labels: set[Label]) -> list[Post]:
    return [p for p in posts if p.label in labels and p.extraction and p.extraction.geo]


def match_posts(
    posts: list[Post],
    encoder: Encoder | None = None,
    alpha: float = 0.6,
    max_km: float = 50.0,
    min_score: float = 0.35,
    top_k: int = 3,
    one_to_one: bool = False,
) -> list[Match]:
    """Rank offers for each need.

    ``alpha`` weights text similarity against proximity; offers farther than ``max_km`` are never matched.
    ``one_to_one`` greedily assigns each offer to at most one need (best score first).
    """
    needs, offers = _located(posts, NEED_LABELS), _located(posts, OFFER_LABELS)
    if not needs or not offers:
        return []
    encoder = encoder or default_encoder()
    emb = encoder([p.text for p in needs] + [p.text for p in offers])
    sims = np.clip(emb[: len(needs)] @ emb[len(needs):].T, 0, 1)

    cands: list[Match] = []
    for i, n in enumerate(needs):
        for j, o in enumerate(offers):
            km = haversine_km(n.extraction.geo.lat, n.extraction.geo.lon, o.extraction.geo.lat, o.extraction.geo.lon)
            if km > max_km:
                continue
            prox = 1 - km / max_km
            score = alpha * float(sims[i, j]) + (1 - alpha) * prox
            if score >= min_score:
                cands.append(Match(
                    need_id=n.id, offer_id=o.id, need_text=n.text, offer_text=o.text, need_label=n.label.value,
                    need_place=n.extraction.geo_source, offer_place=o.extraction.geo_source,
                    similarity=float(sims[i, j]), distance_km=km, proximity=prox, score=score))
    cands.sort(key=lambda m: -m.score)
    if one_to_one:
        used_n: set[str] = set()
        used_o: set[str] = set()
        kept = []
        for m in cands:
            if m.need_id not in used_n and m.offer_id not in used_o:
                kept.append(m)
                used_n.add(m.need_id)
                used_o.add(m.offer_id)
        return kept
    per_need: dict[str, int] = {}
    kept = []
    for m in cands:
        if per_need.get(m.need_id, 0) < top_k:
            per_need[m.need_id] = per_need.get(m.need_id, 0) + 1
            kept.append(m)
    return kept


def main() -> None:
    import sys

    from aquaalert.pipeline import run

    for m in match_posts(run(sys.argv[1] if len(sys.argv) > 1 else "data/sample_posts.csv"), one_to_one=True):
        print(f"{m.score:.2f}  sim={m.similarity:.2f} {m.distance_km:5.1f}km  [{m.need_id}] {m.need_place} <- [{m.offer_id}] {m.offer_place}")
        print(f"      need:  {m.need_text[:90]}\n      offer: {m.offer_text[:90]}")


if __name__ == "__main__":
    main()

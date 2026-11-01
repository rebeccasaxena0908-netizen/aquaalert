"""Phase 1 pipeline: ingest -> classify -> extract(geo)."""
from __future__ import annotations

from pathlib import Path

from aquaalert.classify.base import Classifier, classify_posts
from aquaalert.classify.model import get_classifier
from aquaalert.config import get_settings
from aquaalert.extract.entities import extract_posts
from aquaalert.extract.geocode import Geocoder
from aquaalert.ingest.loader import load_posts
from aquaalert.schemas import Post


def process_posts(
    posts: list[Post], classifier: Classifier | None = None, geocoder: Geocoder | None = None
) -> list[Post]:
    classified = classify_posts(posts, classifier or get_classifier(), rules=get_settings().rules)
    return extract_posts(classified, geocoder)


def run(path: str | Path, limit: int | None = None) -> list[Post]:
    return process_posts(load_posts(path, limit=limit))


if __name__ == "__main__":
    import sys

    for p in run(sys.argv[1] if len(sys.argv) > 1 else "data/sample_posts.csv"):
        loc = p.extraction.geo_source if p.extraction else None
        print(f"{p.id}\t{p.label.value}\t{p.label_score:.2f}\t{loc}\t{p.text[:60]}")

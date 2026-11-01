"""Pipeline: ingest -> language -> classify (+ distress rules) -> extract(geo) -> credibility."""
from __future__ import annotations

from pathlib import Path

from aquaalert.classify.base import Classifier, classify_posts
from aquaalert.classify.model import get_classifier
from aquaalert.config import get_settings
from aquaalert.extract.entities import extract_posts
from aquaalert.extract.geocode import Geocoder
from aquaalert.ingest.loader import load_many, load_posts  # noqa: F401
from aquaalert.language.detect import detect_posts
from aquaalert.credibility.heuristics import score_posts
from aquaalert.schemas import Post


def process_posts(
    posts: list[Post], classifier: Classifier | None = None, geocoder: Geocoder | None = None
) -> list[Post]:
    from aquaalert.summarize.brief import daily_briefs  # local: summarize imports this module's callers
    from aquaalert.summarize.providers import TemplateProvider

    classified = classify_posts(detect_posts(posts), classifier or get_classifier(), rules=get_settings().rules)
    located = extract_posts(classified, geocoder)
    # credibility needs locations (nearby contradictions) and the offline district briefs (no LLM calls here)
    return score_posts(located, daily_briefs(located, TemplateProvider()))


def run(path: str | Path, limit: int | None = None) -> list[Post]:
    return process_posts(load_many(path, limit=limit))


if __name__ == "__main__":
    import sys

    for p in run(sys.argv[1] if len(sys.argv) > 1 else "data/sample_posts.csv"):
        loc = p.extraction.geo_source if p.extraction else None
        print(f"{p.id}\t{p.label.value}\t{p.label_score:.2f}\t{loc}\t{p.text[:60]}")

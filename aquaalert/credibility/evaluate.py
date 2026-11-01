"""Score the sample posts and compare with the hand-made misinfo labels in data/sample_posts_rumours.csv.

    python -m aquaalert.credibility.evaluate

The rumour set is synthetic and was written alongside the heuristics, so these numbers only show that each rule
does what it says; they are not an estimate of real-world accuracy.
"""
from __future__ import annotations

import pandas as pd

from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.config import ROOT
from aquaalert.extract.geocode import Geocoder
from aquaalert.ingest.loader import load_posts
from aquaalert.pipeline import process_posts

FILES = ("data/sample_posts.csv", "data/sample_posts_multilingual.csv", "data/sample_posts_rumours.csv")


def scored_sample(classifier=None):
    posts = [p for f in FILES for p in load_posts(ROOT / f)]
    return process_posts(posts, classifier or KeywordZeroShot(), Geocoder(online=False))


def precision_recall(posts, gold: dict[str, int], threshold: float) -> dict:
    flagged = {p.id for p in posts if (p.misinfo_score or 0) >= threshold}
    positives = {i for i, g in gold.items() if g == 1}
    tp = len(flagged & positives)
    return {"threshold": threshold, "flagged": len(flagged), "precision": round(tp / len(flagged), 3) if flagged else None,
            "recall": round(tp / len(positives), 3) if positives else None}


def main() -> None:
    gold_df = pd.read_csv(ROOT / "data/sample_posts_rumours.csv")
    gold = dict(zip(gold_df["id"], gold_df["misinfo"]))
    # posts outside the rumour file are treated as genuine (misinfo = 0)
    posts = scored_sample()
    gold_all = {p.id: gold.get(p.id, 0) for p in posts}
    for p in sorted(posts, key=lambda p: -(p.misinfo_score or 0)):
        if p.misinfo_score or p.id in gold:
            print(f"{p.id:5s} gold={gold_all[p.id]} score={p.misinfo_score:.2f} {p.text[:60]!r}")
    for t in (0.3, 0.5):
        print(precision_recall(posts, gold_all, t))


if __name__ == "__main__":
    main()

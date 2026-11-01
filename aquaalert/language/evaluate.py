"""Per-language evaluation on the labelled sample posts: language ID, classifier accuracy, geolocation.

    python -m aquaalert.language.evaluate [--out models/language_eval.json]

Uses data/sample_posts.csv (English + a few romanised posts) and data/sample_posts_multilingual.csv
(synthetic Hindi / Malayalam / Bengali posts in native script, romanised and code-mixed).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from aquaalert.classify.base import Classifier, classify_posts
from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.config import ROOT
from aquaalert.extract.entities import extract_locations
from aquaalert.ingest.loader import load_posts
from aquaalert.language.detect import detect_posts
from aquaalert.schemas import Post

DATASETS = ("data/sample_posts.csv", "data/sample_posts_multilingual.csv")


def load_eval_posts() -> list[Post]:
    return [p for d in DATASETS for p in load_posts(ROOT / d)]


def per_language(posts: list[Post], classifiers: dict[str, Classifier]) -> dict:
    detected = detect_posts(posts)
    results: dict = defaultdict(dict)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(posts):
        groups[p.gold_language or "?"].append(i)
    preds = {}
    for name, clf in classifiers.items():
        preds[name] = classify_posts(posts, clf, rules=False)
        preds[name + "+rules"] = classify_posts(posts, clf, rules=True)
    for lang, idx in sorted(groups.items()):
        r = results[lang]
        r["n"] = len(idx)
        r["lang_id_accuracy"] = round(sum(detected[i].language == lang for i in idx) / len(idx), 3)
        r["lang_id_errors"] = sorted({detected[i].language for i in idx if detected[i].language != lang})
        r["located"] = round(sum(bool(extract_locations(posts[i].text)) for i in idx) / len(idx), 3)
        for name, ps in preds.items():
            r[f"acc_{name}"] = round(sum(ps[i].label == posts[i].gold_label for i in idx) / len(idx), 3)
    return dict(results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from aquaalert.classify.model import get_classifier

    classifiers: dict[str, Classifier] = {"keyword": KeywordZeroShot()}
    for backend in ("embed_lr", "muril"):
        try:
            classifiers[backend] = get_classifier(backend)
        except (FileNotFoundError, ImportError) as exc:
            print(f"skipping {backend}: {exc}")
    res = per_language(load_eval_posts(), classifiers)
    names = [n for c in classifiers for n in (c, c + "+rules")]
    print(f"{'lang':8s} {'n':>3s} {'langID':>7s} {'located':>8s} " + " ".join(f"{n:>14s}" for n in names))
    for lang, r in res.items():
        print(f"{lang:8s} {r['n']:3d} {r['lang_id_accuracy']:7.2f} {r['located']:8.2f} "
              + " ".join(f"{r['acc_' + n]:14.2f}" for n in names)
              + (f"   langID errors: {r['lang_id_errors']}" if r["lang_id_errors"] else ""))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

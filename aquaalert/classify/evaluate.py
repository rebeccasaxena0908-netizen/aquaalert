"""Compare classifiers on the same labeled test split.

    python -m aquaalert.classify.evaluate --test data/raw/humaid/test.csv --label-dataset humaid \
        --model-dir models/muril-crisis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquaalert.classify.base import Classifier
from aquaalert.classify.train import LABEL2ID, compute_metrics
from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.ingest.loader import load_labeled
from aquaalert.schemas import LABELS


def evaluate(clf: Classifier, texts: list[str], y_true: list[int]) -> dict:
    scores = clf.predict_scores(texts)
    y_pred = [LABEL2ID[max(s, key=s.__getitem__)] for s in scores]
    return compute_metrics(y_true, y_pred)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True)
    p.add_argument("--label-dataset", default=None)
    p.add_argument("--model-dir", default=None, help="fine-tuned model to compare against the fallback")
    p.add_argument("--out", default=None, help="write comparison JSON here")
    a = p.parse_args()

    pairs = load_labeled(a.test, a.label_dataset)
    texts, y_true = [t for t, _ in pairs], [LABEL2ID[l.value] for _, l in pairs]
    support = {l: y_true.count(i) for i, l in enumerate(LABELS)}
    results: dict = {"n_test": len(texts), "support": support}

    classifiers: dict[str, Classifier] = {"keyword_fallback": KeywordZeroShot()}
    if a.model_dir:
        from aquaalert.classify.model import TransformerClassifier

        classifiers["trained"] = TransformerClassifier(a.model_dir)
    for name, clf in classifiers.items():
        results[name] = evaluate(clf, texts, y_true)
        print(f"{name:18s} macro_f1={results[name]['macro_f1']:.4f} accuracy={results[name]['accuracy']:.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

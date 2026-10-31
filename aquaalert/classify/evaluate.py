"""Compare classifiers on the same labeled test split and write per-class F1 + bootstrap CIs.

    python -m aquaalert.classify.evaluate --test data/raw/humaid/test.csv --label-dataset humaid \
        --model-dir models/muril --out models/muril/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquaalert.classify.base import Classifier
from aquaalert.classify.metrics import format_table, full_report
from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.ingest.loader import load_labeled
from aquaalert.schemas import LABELS

LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def predict_ids(clf: Classifier, texts: list[str]) -> list[int]:
    return [LABEL2ID[max(s, key=s.__getitem__)] for s in clf.predict_scores(texts)]


def compare(classifiers: dict[str, Classifier], test_path: str, label_dataset: str | None,
            n_boot: int = 1000) -> dict:
    pairs = load_labeled(test_path, label_dataset)
    texts, y_true = [t for t, _ in pairs], [LABEL2ID[l.value] for _, l in pairs]
    reports = {name: full_report(y_true, predict_ids(clf, texts), n_boot) for name, clf in classifiers.items()}
    print(format_table(reports))
    return {"n_test": len(texts), "n_bootstrap": n_boot, "models": reports}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True)
    p.add_argument("--label-dataset", default=None)
    p.add_argument("--model-dir", default=None, help="fine-tuned model to compare against the fallback")
    p.add_argument("--out", default=None, help="write comparison JSON here")
    a = p.parse_args()

    clfs: dict[str, Classifier] = {"keyword_fallback": KeywordZeroShot()}
    if a.model_dir:
        from aquaalert.classify.model import TransformerClassifier

        clfs = {"trained": TransformerClassifier(a.model_dir), **clfs}
    result = compare(clfs, a.test, a.label_dataset)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

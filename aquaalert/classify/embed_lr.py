"""Lightweight baseline: multilingual sentence embeddings + class-weighted logistic regression.

    python -m aquaalert.classify.embed_lr --train data/raw/humaid/train.csv --val data/raw/humaid/dev.csv \
        --test data/raw/humaid/test.csv --label-dataset humaid --out models/embed_lr

Runs on a laptop (MPS/CUDA used when available). Writes ``classifier.joblib``, ``meta.json`` and
``metrics.json`` (same metrics as the MuRIL run: per-class F1, macro-F1 over 7 / 5 classes, bootstrap CI).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aquaalert.config import ROOT
from aquaalert.ingest.loader import load_labeled
from aquaalert.schemas import LABELS

ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
C_GRID = (0.1, 1.0, 10.0)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def encode(texts: list[str], encoder, batch_size: int = 128) -> np.ndarray:
    return encoder.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True,
                          normalize_embeddings=True)


def load_encoder(name: str = ENCODER):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=pick_device())


class EmbedLRClassifier:
    """Loads a directory written by ``train_embed``."""

    def __init__(self, model_dir: str | Path) -> None:
        import joblib

        model_dir = Path(model_dir)
        self.name = f"embed-lr:{model_dir.name}"
        payload = joblib.load(model_dir / "classifier.joblib")
        self._lr, self._classes = payload["lr"], payload["classes"]  # classes: label indices LR knows
        self._encoder = load_encoder(json.loads((model_dir / "meta.json").read_text())["encoder"])

    def predict_scores(self, texts: list[str]) -> list[dict[str, float]]:
        probs = self._lr.predict_proba(encode(texts, self._encoder))
        out = []
        for row in probs:
            scores = {l: 0.0 for l in LABELS}
            for cls, p in zip(self._classes, row):
                scores[LABELS[cls]] = float(p)
            out.append(scores)
        return out


def train_embed(train_path: str, val_path: str, test_path: str, out: str,
                label_dataset: str | None = None, encoder_name: str = ENCODER, n_boot: int = 1000) -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression

    from aquaalert.classify.evaluate import compare
    from aquaalert.classify.metrics import full_report
    from aquaalert.classify.zeroshot import KeywordZeroShot

    out_dir = Path(out) if Path(out).is_absolute() else ROOT / out
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va = load_labeled(train_path, label_dataset), load_labeled(val_path, label_dataset)
    y_tr = np.array([LABEL2ID[l.value] for _, l in tr])
    y_va = np.array([LABEL2ID[l.value] for _, l in va])

    encoder = load_encoder(encoder_name)
    x_tr, x_va = encode([t for t, _ in tr], encoder), encode([t for t, _ in va], encoder)

    best = (-1.0, None, None)  # dev macro-F1, C, model
    grid = {}
    for c in C_GRID:
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(x_tr, y_tr)
        f1 = full_report(y_va.tolist(), lr.predict(x_va).tolist(), n_boot=1)["macro_f1_all7"]
        grid[str(c)] = f1
        if f1 > best[0]:
            best = (f1, c, lr)
        print(f"C={c}: dev macro-F1 (all 7) = {f1:.4f}")
    dev_f1, c, lr = best

    joblib.dump({"lr": lr, "classes": [int(k) for k in lr.classes_]}, out_dir / "classifier.joblib")
    (out_dir / "meta.json").write_text(json.dumps({"encoder": encoder_name, "labels": LABELS}, indent=2))

    result = compare({"embed_lr": EmbedLRClassifier(out_dir), "keyword_fallback": KeywordZeroShot()},
                     test_path, label_dataset, n_boot)
    result["training"] = {"encoder": encoder_name, "class_weight": "balanced (inverse frequency)",
                          "C_grid_dev_macro_f1": grid, "best_C": c, "best_dev_macro_f1": dev_f1,
                          "n_train": len(tr), "n_dev": len(va)}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {out_dir / 'metrics.json'}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", dest="train_path", required=True)
    p.add_argument("--val", dest="val_path", required=True)
    p.add_argument("--test", dest="test_path", required=True)
    p.add_argument("--out", default="models/embed_lr")
    p.add_argument("--label-dataset", choices=["humaid", "trec_is"], default=None)
    a = p.parse_args()
    train_embed(a.train_path, a.val_path, a.test_path, a.out, a.label_dataset)


if __name__ == "__main__":
    main()

"""Fine-tune MuRIL (or xlm-roberta-base) on a labeled dataset.

    python -m aquaalert.classify.train --data data/raw/humaid.csv --label-dataset humaid \
        --out models/muril-crisis

Writes the model + tokenizer and ``metrics.json`` (accuracy, macro-F1, per-class report).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from aquaalert.config import ROOT, get_settings
from aquaalert.ingest.loader import load_posts
from aquaalert.schemas import LABELS


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Accuracy, macro-F1 and a per-class report keyed by label name."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=range(len(LABELS)), zero_division=0),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def build_splits(texts: list[str], labels: list[int], test_size: float, seed: int):
    """Stratify when every class has >=2 examples, otherwise fall back to a plain split."""
    counts = {l: labels.count(l) for l in set(labels)}
    stratify = labels if min(counts.values()) >= 2 and int(len(labels) * test_size) >= len(counts) else None
    return train_test_split(texts, labels, test_size=test_size, random_state=seed, stratify=stratify)


def train(
    data: str,
    out: str,
    label_dataset: str | None = None,
    base_model: str | None = None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    base_model = base_model or get_settings().base_model
    posts = [p for p in load_posts(data, label_dataset=label_dataset) if p.gold_label]
    if len(posts) < 10:
        raise SystemExit(f"Only {len(posts)} labeled posts in {data}; need more to fine-tune.")
    label2id = {l: i for i, l in enumerate(LABELS)}
    texts = [p.text for p in posts]
    y = [label2id[p.gold_label.value] for p in posts]
    x_tr, x_te, y_tr, y_te = build_splits(texts, y, test_size, seed)

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=label2id
    )

    class _DS(torch.utils.data.Dataset):
        def __init__(self, xs: list[str], ys: list[int]) -> None:
            self.enc, self.ys = tok(xs, truncation=True, max_length=128, padding=True), ys

        def __len__(self) -> int:
            return len(self.ys)

        def __getitem__(self, i: int) -> dict:
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.ys[i])
            return item

    out_dir = Path(out) if Path(out).is_absolute() else ROOT / out
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=_DS(x_tr, y_tr), eval_dataset=_DS(x_te, y_te))
    trainer.train()
    preds = np.argmax(trainer.predict(_DS(x_te, y_te)).predictions, axis=-1).tolist()
    metrics = compute_metrics(y_te, preds)
    metrics.update(base_model=base_model, n_train=len(x_tr), n_test=len(x_te))

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f} -> {out_dir}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="models/muril-crisis")
    p.add_argument("--label-dataset", choices=["humaid", "trec_is"], default=None)
    p.add_argument("--base-model", default=None, help="default: AQUAALERT_BASE_MODEL (google/muril-base-cased)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    a = p.parse_args()
    train(a.data, a.out, a.label_dataset, a.base_model, a.epochs, a.batch_size, a.lr)


if __name__ == "__main__":
    main()

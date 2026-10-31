"""Fine-tune MuRIL (or xlm-roberta-base) on a labeled dataset.

Pre-split data (e.g. HumAID train/dev/test):
    python -m aquaalert.classify.train --train data/raw/humaid/train.csv \
        --val data/raw/humaid/dev.csv --test data/raw/humaid/test.csv \
        --label-dataset humaid --out models/muril-crisis

Single file (random stratified split):
    python -m aquaalert.classify.train --data labeled.csv --out models/muril-crisis

Writes the model + tokenizer and ``metrics.json`` (accuracy, macro-F1, per-class report).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from aquaalert.config import ROOT, get_settings
from aquaalert.ingest.loader import load_labeled
from aquaalert.schemas import LABELS

LABEL2ID = {l: i for i, l in enumerate(LABELS)}


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


def _encode(pairs) -> tuple[list[str], list[int]]:
    return [t for t, _ in pairs], [LABEL2ID[l.value] for _, l in pairs]


def train(
    out: str,
    train_path: str | None = None,
    val_path: str | None = None,
    test_path: str | None = None,
    data: str | None = None,
    label_dataset: str | None = None,
    base_model: str | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-5,
    max_length: int = 96,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    base_model = base_model or get_settings().base_model
    if train_path and test_path:
        x_tr, y_tr = _encode(load_labeled(train_path, label_dataset))
        x_te, y_te = _encode(load_labeled(test_path, label_dataset))
        x_va, y_va = _encode(load_labeled(val_path, label_dataset)) if val_path else (x_te, y_te)
    elif data:
        x, y = _encode(load_labeled(data, label_dataset))
        if len(x) < 10:
            raise SystemExit(f"Only {len(x)} labeled posts in {data}; need more to fine-tune.")
        x_tr, x_te, y_tr, y_te = build_splits(x, y, test_size, seed)
        x_va, y_va = x_te, y_te
    else:
        raise SystemExit("Pass either --data, or --train and --test (optionally --val).")

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID
    )

    class _DS(torch.utils.data.Dataset):
        def __init__(self, xs: list[str], ys: list[int]) -> None:
            self.enc, self.ys = tok(xs, truncation=True, max_length=max_length), ys

        def __len__(self) -> int:
            return len(self.ys)

        def __getitem__(self, i: int) -> dict:
            return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.ys[i]}

    out_dir = Path(out) if Path(out).is_absolute() else ROOT / out
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_DS(x_tr, y_tr),
        eval_dataset=_DS(x_va, y_va),
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()
    preds = np.argmax(trainer.predict(_DS(x_te, y_te)).predictions, axis=-1).tolist()
    metrics = compute_metrics(y_te, preds)
    metrics.update(base_model=base_model, n_train=len(x_tr), n_test=len(x_te), epochs=epochs, max_length=max_length)

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f} -> {out_dir}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="single labeled file; split randomly")
    p.add_argument("--train", dest="train_path")
    p.add_argument("--val", dest="val_path")
    p.add_argument("--test", dest="test_path")
    p.add_argument("--out", default="models/muril-crisis")
    p.add_argument("--label-dataset", choices=["humaid", "trec_is"], default=None)
    p.add_argument("--base-model", default=None, help="default: AQUAALERT_BASE_MODEL (google/muril-base-cased)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=96)
    a = p.parse_args()
    train(a.out, a.train_path, a.val_path, a.test_path, a.data, a.label_dataset, a.base_model,
          a.epochs, a.batch_size, a.lr, a.max_length)


if __name__ == "__main__":
    main()

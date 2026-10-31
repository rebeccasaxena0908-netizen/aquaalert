"""Fine-tune MuRIL (or xlm-roberta-base) with class-weighted cross-entropy.

    python -m aquaalert.classify.train --train data/raw/humaid/train.csv \
        --val data/raw/humaid/dev.csv --test data/raw/humaid/test.csv \
        --label-dataset humaid --out models/muril

Loss weights are inverse class frequency on the training set. The checkpoint with the best dev
macro-F1 (over all 7 classes) is kept. Afterwards the trained model and the keyword fallback are
evaluated on the test split and everything is written to ``<out>/metrics.json``.
A single ``--data`` file (random stratified split) is also supported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from aquaalert.classify.metrics import inverse_frequency_weights
from aquaalert.config import ROOT, get_settings
from aquaalert.ingest.loader import load_labeled
from aquaalert.schemas import LABELS

LABEL2ID = {l: i for i, l in enumerate(LABELS)}


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
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 3e-5,
    max_length: int = 96,
    seed: int = 42,
    n_boot: int = 1000,
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

    from aquaalert.classify.evaluate import compare
    from aquaalert.classify.metrics import full_report
    from aquaalert.classify.model import TransformerClassifier
    from aquaalert.classify.zeroshot import KeywordZeroShot

    base_model = base_model or get_settings().base_model
    out_dir = Path(out) if Path(out).is_absolute() else ROOT / out
    if train_path and val_path and test_path:
        x_tr, y_tr = _encode(load_labeled(train_path, label_dataset))
        x_va, y_va = _encode(load_labeled(val_path, label_dataset))
        test_file = test_path
    elif data:
        x, y = _encode(load_labeled(data, label_dataset))
        x_tr, x_rest, y_tr, y_rest = build_splits(x, y, 0.3, seed)
        x_va, x_te, y_va, y_te = build_splits(x_rest, y_rest, 0.5, seed)
        test_file = str(out_dir / "test_split.csv")
        out_dir.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame({"text": x_te, "label": [LABELS[i] for i in y_te]}).to_csv(test_file, index=False)
        label_dataset = None
    else:
        raise SystemExit("Pass --train, --val and --test (or a single --data file).")

    weights = inverse_frequency_weights(y_tr)
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

    class _WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            w = torch.tensor(weights, dtype=torch.float, device=outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
            return (loss, outputs) if return_outputs else loss

    def dev_metrics(pred) -> dict:
        report = full_report(pred.label_ids.tolist(), np.argmax(pred.predictions, -1).tolist(), n_boot=1)
        return {"macro_f1": report["macro_f1_all7"], "macro_f1_large5": report["macro_f1_large5"]}

    # explicit step count: `warmup_ratio` was removed in transformers 5
    warmup_steps = int(0.06 * -(-len(x_tr) // batch_size) * epochs)
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        seed=seed,
        report_to=[],
    )
    trainer = _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=_DS(x_tr, y_tr),
        eval_dataset=_DS(x_va, y_va),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=dev_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    history = [
        {k: v for k, v in e.items() if k in ("epoch", "eval_loss", "eval_macro_f1", "eval_macro_f1_large5")}
        for e in trainer.state.log_history if "eval_macro_f1" in e
    ]
    result = compare(
        {"trained": TransformerClassifier(out_dir), "keyword_fallback": KeywordZeroShot()},
        test_file, label_dataset, n_boot,
    )
    result["training"] = {
        "base_model": base_model, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "max_length": max_length, "seed": seed, "n_train": len(x_tr), "n_dev": len(x_va),
        "class_weights": dict(zip(LABELS, weights)),
        "dev_history": history,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_macro_f1": trainer.state.best_metric,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {out_dir / 'metrics.json'}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="single labeled file; split randomly 70/15/15")
    p.add_argument("--train", dest="train_path")
    p.add_argument("--val", dest="val_path")
    p.add_argument("--test", dest="test_path")
    p.add_argument("--out", default="models/muril")
    p.add_argument("--label-dataset", choices=["humaid", "trec_is"], default=None)
    p.add_argument("--base-model", default=None, help="default: AQUAALERT_BASE_MODEL (google/muril-base-cased)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=96)
    a = p.parse_args()
    train(a.out, a.train_path, a.val_path, a.test_path, a.data, a.label_dataset, a.base_model,
          a.epochs, a.batch_size, a.lr, a.max_length)


if __name__ == "__main__":
    main()

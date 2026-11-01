"""Fine-tuned transformer classifier and the backend selector."""
from __future__ import annotations

import logging
from pathlib import Path

from aquaalert.classify.base import Classifier
from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.config import get_settings

log = logging.getLogger(__name__)


class TransformerClassifier:
    """Loads a model directory written by ``aquaalert.classify.train``."""

    def __init__(self, model_dir: str | Path, max_length: int = 128, batch_size: int = 32) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = f"transformer:{Path(model_dir).name}"
        self._torch = torch
        self._max_length, self._batch_size = max_length, batch_size
        self._device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._tok = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self._device).eval()
        id2label = self._model.config.id2label
        self._labels = [id2label[i] for i in range(len(id2label))]

    def predict_scores(self, texts: list[str]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = self._tok(
                texts[i : i + self._batch_size],
                truncation=True,
                max_length=self._max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.no_grad():
                probs = self._model(**batch).logits.softmax(-1).tolist()
            out.extend(dict(zip(self._labels, p)) for p in probs)
        return out


TRANSFORMER_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _ready(path: Path, kind: str) -> bool:
    """A model directory is usable only when its weights are there (a download writes config.json first)."""
    if kind == "transformer":
        return (path / "config.json").exists() and any((path / w).exists() for w in TRANSFORMER_WEIGHTS)
    return (path / "classifier.joblib").exists()


def get_classifier(backend: str | None = None) -> Classifier:
    """Pick a classifier.

    ``auto`` prefers MuRIL fine-tuned with the romanised data, then the English-only MuRIL, then the
    embedding+LR baseline, then the keyword fallback. Explicit backends: ``muril_romanised``, ``muril``
    (alias ``trained``), ``embed_lr``, ``zeroshot``.
    """
    settings = get_settings()
    backend = backend or settings.classifier
    if backend == "trained":
        backend = "muril"

    candidates: list[tuple[str, Path, str]] = [
        ("muril_romanised", settings.romanised_dir, "transformer"),
        ("muril", settings.model_dir, "transformer"),
        ("embed_lr", settings.embed_dir, "embed_lr"),
    ]
    for name, path, kind in candidates:
        if backend not in ("auto", name):
            continue
        if not _ready(path, kind):
            if backend == name:
                raise FileNotFoundError(f"No {name} model at {path}")
            continue
        try:
            if kind == "transformer":
                return TransformerClassifier(path)
            from aquaalert.classify.embed_lr import EmbedLRClassifier

            return EmbedLRClassifier(path)
        except Exception as exc:  # missing deps, corrupt files, ...
            if backend == name:
                raise
            log.warning("Could not load %s (%s); trying the next backend", name, exc)
    if backend not in ("auto", "zeroshot"):
        raise ValueError(f"Unknown classifier backend: {backend}")
    return KeywordZeroShot()

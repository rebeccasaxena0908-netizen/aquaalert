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


def get_classifier(backend: str | None = None) -> Classifier:
    """Pick a classifier. ``auto`` uses the trained model if present, else zero-shot."""
    settings = get_settings()
    backend = backend or settings.classifier
    has_model = (settings.model_dir / "config.json").exists()
    if backend in ("auto", "trained") and has_model:
        try:
            return TransformerClassifier(settings.model_dir)
        except Exception as exc:  # missing torch, corrupt checkpoint, ...
            if backend == "trained":
                raise
            log.warning("Could not load trained model (%s); using zero-shot fallback", exc)
    elif backend == "trained":
        raise FileNotFoundError(f"No trained model at {settings.model_dir}")
    return KeywordZeroShot()



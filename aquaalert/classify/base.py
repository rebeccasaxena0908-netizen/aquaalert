"""Classifier interface shared by the zero-shot fallback and the fine-tuned model."""
from __future__ import annotations

from typing import Protocol

from aquaalert.schemas import Label, Post


class Classifier(Protocol):
    name: str

    def predict_scores(self, texts: list[str]) -> list[dict[str, float]]:
        """Return a {label: probability} dict per text."""
        ...


def classify_posts(posts: list[Post], clf: Classifier) -> list[Post]:
    """Return copies of ``posts`` with label, label_score and label_scores filled in."""
    if not posts:
        return []
    results = clf.predict_scores([p.text for p in posts])
    out: list[Post] = []
    for post, scores in zip(posts, results, strict=True):
        best = max(scores, key=scores.__getitem__)
        out.append(
            post.model_copy(
                update={"label": Label(best), "label_score": scores[best], "label_scores": scores}
            )
        )
    return out

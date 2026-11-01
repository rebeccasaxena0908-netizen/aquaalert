"""Classifier interface shared by the zero-shot fallback and the fine-tuned model."""
from __future__ import annotations

from typing import Protocol

from aquaalert.classify.rules import apply_rules
from aquaalert.schemas import Label, Post


class Classifier(Protocol):
    name: str

    def predict_scores(self, texts: list[str]) -> list[dict[str, float]]:
        """Return a {label: probability} dict per text."""
        ...


def classify_posts(posts: list[Post], clf: Classifier, rules: bool = True) -> list[Post]:
    """Return copies of ``posts`` with label, label_score and label_scores filled in.

    With ``rules`` (default) the distress rule layer may promote offer_help/info_update to rescue_needed;
    the model's label and the rule name are then recorded on the Post.
    """
    if not posts:
        return []
    results = clf.predict_scores([p.text for p in posts])
    out: list[Post] = []
    for post, scores in zip(posts, results, strict=True):
        best = Label(max(scores, key=scores.__getitem__))
        update = {"label": best, "label_score": scores[best.value], "label_scores": scores}
        if rules:
            final, rule, conf = apply_rules(post.text, best)
            if rule:
                update.update(label=final, label_score=conf, model_label=best, rule=rule)
        out.append(post.model_copy(update=update))
    return out

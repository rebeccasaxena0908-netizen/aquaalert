import pytest

from aquaalert.classify.base import classify_posts
from aquaalert.classify.label_map import map_label
from aquaalert.classify.model import get_classifier
from aquaalert.classify.train import build_splits, compute_metrics
from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.ingest.loader import load_posts
from aquaalert.schemas import LABELS, Label


@pytest.mark.parametrize("text,expected", [
    ("12 people stranded on the roof, send boats", "rescue_needed"),
    ("The bridge has collapsed and the road is cut off", "infrastructure_damage"),
    ("We need drinking water and food at the camp", "need_supplies"),
    ("Volunteers with boats available, DM us", "offer_help"),
    ("My father is missing, last seen at the station", "missing_person"),
    ("IMD issues red alert, dam level rising", "info_update"),
    ("Anyone know a good biryani place?", "irrelevant"),
    ("Hum chhat par phase hue hain, bachao", "rescue_needed"),
])
def test_zeroshot_examples(text, expected):
    scores = KeywordZeroShot().predict_scores([text])[0]
    assert max(scores, key=scores.get) == expected
    assert set(scores) == set(LABELS)
    assert sum(scores.values()) == pytest.approx(1.0)


def test_zeroshot_on_sample_is_accurate(sample_path):
    posts = classify_posts(load_posts(sample_path), KeywordZeroShot())
    acc = sum(p.label == p.gold_label for p in posts) / len(posts)
    assert acc >= 0.85


def test_classify_posts_fills_fields(make_post):
    out = classify_posts([make_post("people trapped, rescue needed")], KeywordZeroShot())
    assert out[0].label == Label.RESCUE_NEEDED and 0 < out[0].label_score <= 1
    assert classify_posts([], KeywordZeroShot()) == []


def test_get_classifier_falls_back_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("AQUAALERT_MODEL_DIR", str(tmp_path / "missing"))
    assert isinstance(get_classifier("auto"), KeywordZeroShot)
    with pytest.raises(FileNotFoundError):
        get_classifier("trained")


def test_metrics_and_splits():
    m = compute_metrics([0, 1, 2], [0, 1, 1])
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert set(LABELS) <= set(m["per_class"])
    x_tr, x_te, y_tr, y_te = build_splits(list("abcdefghij"), [0, 1] * 5, 0.4, 0)
    assert len(x_te) == 4 and len(x_tr) == 6
    # a class with a single example must not break the split
    build_splits(list("abcdefghij"), [0] * 9 + [1], 0.2, 0)
    assert map_label("offer_help") == Label.OFFER_HELP

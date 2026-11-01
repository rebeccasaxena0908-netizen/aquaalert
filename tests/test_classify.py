import pytest

from aquaalert.classify.base import classify_posts
from aquaalert.classify.label_map import map_label
from aquaalert.classify.model import get_classifier
from aquaalert.classify.train import build_splits
from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.extract.geocode import Geocoder
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


def _isolate(monkeypatch, tmp_path):
    dirs = {k: tmp_path / k for k in ("rom", "muril", "embed")}
    monkeypatch.setenv("AQUAALERT_ROMANISED_DIR", str(dirs["rom"]))
    monkeypatch.setenv("AQUAALERT_MODEL_DIR", str(dirs["muril"]))
    monkeypatch.setenv("AQUAALERT_EMBED_DIR", str(dirs["embed"]))
    return dirs


def test_get_classifier_falls_back_without_model(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert isinstance(get_classifier("auto"), KeywordZeroShot)
    assert isinstance(get_classifier("zeroshot"), KeywordZeroShot)
    for backend in ("trained", "muril", "muril_romanised", "embed_lr"):
        with pytest.raises(FileNotFoundError):
            get_classifier(backend)
    with pytest.raises(ValueError):
        get_classifier("bogus")


def test_auto_order_romanised_then_muril_then_embed_lr(monkeypatch, tmp_path):
    import aquaalert.classify.embed_lr as embed_mod
    import aquaalert.classify.model as model_mod

    d = _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(model_mod, "TransformerClassifier", lambda p: ("transformer", p))
    monkeypatch.setattr(embed_mod, "EmbedLRClassifier", lambda p: ("embed", p))
    for k in d.values():
        k.mkdir()
    (d["embed"] / "classifier.joblib").write_text("x")
    assert model_mod.get_classifier("auto") == ("embed", d["embed"])
    (d["muril"] / "config.json").write_text("{}")
    (d["muril"] / "model.safetensors").write_text("w")
    assert model_mod.get_classifier("auto") == ("transformer", d["muril"])
    (d["rom"] / "config.json").write_text("{}")  # download in progress: config but no weights yet
    assert model_mod.get_classifier("auto") == ("transformer", d["muril"])
    (d["rom"] / "model.safetensors").write_text("w")
    assert model_mod.get_classifier("auto") == ("transformer", d["rom"])
    assert model_mod.get_classifier("muril") == ("transformer", d["muril"])
    assert model_mod.get_classifier("embed_lr") == ("embed", d["embed"])


def test_splits():
    x_tr, x_te, y_tr, y_te = build_splits(list("abcdefghij"), [0, 1] * 5, 0.4, 0)
    assert len(x_te) == 4 and len(x_tr) == 6
    # a class with a single example must not break the split
    build_splits(list("abcdefghij"), [0] * 9 + [1], 0.2, 0)
    assert map_label("offer_help") == Label.OFFER_HELP


def test_pipeline_respects_rules_flag(monkeypatch, tmp_path, make_post):
    from aquaalert.pipeline import process_posts

    class AlwaysOffer:
        name = "offer"

        def predict_scores(self, texts):
            return [{**{l: 0.0 for l in LABELS}, "offer_help": 1.0} for _ in texts]

    g = Geocoder(cache_path=tmp_path / "g.json", online=False)
    post = make_post("Need rescue, we are stuck on the roof in Aluva")
    monkeypatch.setenv("AQUAALERT_RULES", "on")
    assert process_posts([post], AlwaysOffer(), g)[0].label == Label.RESCUE_NEEDED
    monkeypatch.setenv("AQUAALERT_RULES", "off")
    assert process_posts([post], AlwaysOffer(), g)[0].label == Label.OFFER_HELP


def test_train_cli_has_extra_train_flag():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "aquaalert.classify.train", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--extra-train" in out.stdout

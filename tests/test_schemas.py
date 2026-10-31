import pytest
from pydantic import ValidationError

from aquaalert.schemas import LABELS, GeoPoint, Label, Post


def test_labels_match_spec():
    assert LABELS == ["rescue_needed", "infrastructure_damage", "need_supplies", "offer_help",
                      "missing_person", "info_update", "irrelevant"]


def test_post_defaults_and_roundtrip(make_post):
    post = make_post("hello")
    assert post.label is None and post.extraction is None
    assert Post.model_validate_json(post.model_dump_json()) == post


def test_validation(make_post):
    with pytest.raises(ValidationError):
        GeoPoint(lat=100, lon=0)
    with pytest.raises(ValidationError):
        Post.model_validate({"id": "1", "text": "x",
                         "timestamp": "2018-08-15", "label": "bogus"})
    assert Label("offer_help") is Label.OFFER_HELP

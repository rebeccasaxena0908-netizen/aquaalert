import json

import pytest

from aquaalert.classify.label_map import map_label
from aquaalert.ingest.loader import load_posts
from aquaalert.ingest.replay import replay
from aquaalert.schemas import Label


def test_load_sample(sample_path):
    posts = load_posts(sample_path)
    assert len(posts) == 50
    assert [p.timestamp for p in posts] == sorted(p.timestamp for p in posts)
    assert all(p.gold_label for p in posts)
    assert load_posts(sample_path, limit=5)[0].id == "s001"


def test_load_json_with_alt_columns_and_label_map(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([
        {"tweet_id": "1", "tweet_text": "b", "created_at": "2018-08-16 01:00", "class_label": "requests_or_urgent_needs"},
        {"tweet_id": "2", "tweet_text": "a", "created_at": "2018-08-16 00:00", "class_label": "weird"},
        {"tweet_id": "3", "tweet_text": "", "created_at": "2018-08-16 00:00", "class_label": "x"},
    ]))
    posts = load_posts(path, label_dataset="humaid")
    assert [p.id for p in posts] == ["2", "1"]  # sorted, empty text dropped
    assert posts[0].gold_label is None
    assert posts[1].gold_label == Label.NEED_SUPPLIES


def test_load_errors(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_posts(bad)
    with pytest.raises(ValueError):
        load_posts(tmp_path / "x.txt")


def test_map_label():
    assert map_label("rescue_needed") == Label.RESCUE_NEEDED
    assert map_label("Not Humanitarian", "humaid") == Label.IRRELEVANT
    assert map_label("nope") is None and map_label(None) is None


def test_replay_order_and_delays(sample_path):
    posts = list(reversed(load_posts(sample_path, limit=4)))
    waits: list[float] = []
    out = list(replay(posts, speed=60, sleep=waits.append))
    assert [p.id for p in out] == ["s001", "s002", "s003", "s004"]
    assert len(waits) == 3 and all(0 < w <= 5 for w in waits)
    # s001->s002 is 35 minutes of event time = 2100s / 60
    assert waits[0] == pytest.approx(5.0)  # capped by max_delay


def test_replay_no_wait(sample_path):
    waits: list[float] = []
    assert len(list(replay(load_posts(sample_path), speed=0, sleep=waits.append))) == 50
    assert waits == []


def test_humaid_rescue_rule():
    req = "requests_or_urgent_needs"
    assert map_label(req, "humaid", "Families stranded on rooftops, need rescue now") == Label.RESCUE_NEEDED
    assert map_label(req, "humaid", "We need food and water at the shelter") == Label.NEED_SUPPLIES
    assert map_label(req, "humaid", "Rescue teams are distributing blankets") == Label.NEED_SUPPLIES
    assert map_label(req, "humaid") == Label.NEED_SUPPLIES  # no text -> default
    # the rule applies only to the urgent-needs class
    assert map_label("caution_and_advice", "humaid", "people stranded") == Label.INFO_UPDATE


def test_humaid_all_eleven_categories_covered():
    from aquaalert.classify.label_map import HUMAID_MAP

    assert len(HUMAID_MAP) == 11
    assert map_label("Dont know cant judge", "humaid") is None
    assert map_label("Rescue volunteering or donation effort", "humaid") == Label.OFFER_HELP


def test_load_labeled_tsv(tmp_path):
    from aquaalert.ingest.loader import load_labeled

    path = tmp_path / "e.tsv"
    path.write_text("tweet_id\ttweet_text\tclass_label\n1\tstuck on the roof\trequests_or_urgent_needs\n"
                    "2\tthoughts with everyone\tsympathy_and_support\n3\t?\tdont_know_cant_judge\n")
    assert load_labeled(path, "humaid") == [("stuck on the roof", Label.RESCUE_NEEDED),
                                            ("thoughts with everyone", Label.IRRELEVANT)]

import json

import pandas as pd
import pytest

from scripts import make_romanised_data as mrd


@pytest.fixture
def humaid(tmp_path):
    d = tmp_path / "humaid"
    d.mkdir()
    rows = ([("people stuck on the roof need to be rescued now", "requests_or_urgent_needs")] * 3
            + [("my brother is missing since the flood", "missing_or_found_people")] * 4
            + [(f"donate to the relief fund number {i}", "rescue_volunteering_or_donation_effort") for i in range(40)]
            + [(f"bridge damaged on highway {i}", "infrastructure_and_utility_damage") for i in range(20)]
            + [("this exact tweet is also in test", "not_humanitarian")])
    pd.DataFrame(rows, columns=["tweet_text", "class_label"]).to_csv(d / "train.csv", index=False)
    pd.DataFrame([("this exact tweet is also in test", "not_humanitarian")],
                 columns=["tweet_text", "class_label"]).to_csv(d / "test.csv", index=False)
    pd.DataFrame([("a dev tweet", "caution_and_advice")], columns=["tweet_text", "class_label"]).to_csv(d / "dev.csv", index=False)
    return d


def test_only_train_split_and_no_test_overlap(humaid):
    for bad in ("test.csv", "dev.csv"):
        with pytest.raises(SystemExit, match="only the train split"):
            mrd.load_train(humaid / bad)
    df = mrd.load_train(humaid / "train.csv")
    assert "this exact tweet is also in test" not in set(df.tweet_text)
    assert df.source_id.str.startswith("humaid-train-").all()


def test_plan_keeps_all_rare_and_stratifies_the_rest(humaid):
    df = mrd.load_train(humaid / "train.csv")
    items = mrd.plan(df, per_language=27, langs=["hi", "ml"], seed=1)
    for lang in ("hi", "ml"):
        mine = [i for i in items if i["lang"] == lang]
        labels = pd.Series([i["label"] for i in mine]).value_counts().to_dict()
        assert labels["rescue_needed"] == 3 and labels["missing_person"] == 4  # every rare tweet
        assert labels["offer_help"] == 13 and labels["infrastructure_damage"] == 7  # 20 split 2:1
    floored = mrd.plan(df, per_language=27, langs=["hi"], seed=1, min_per_class=10)
    counts = pd.Series([i["label"] for i in floored]).value_counts().to_dict()
    assert counts["infrastructure_damage"] == 10 and counts["offer_help"] == 13  # raised to the floor / unchanged
    assert counts["rescue_needed"] == 3  # rare classes: all available, even below the floor
    capped = mrd.plan(df, per_language=27, langs=["hi"], seed=1, min_per_class=500)
    assert pd.Series([i["label"] for i in capped]).value_counts()["infrastructure_damage"] == 20  # pool size cap
    hi_common = {i["source_id"] for i in items if i["lang"] == "hi" and i["label"] == "offer_help"}
    ml_common = {i["source_id"] for i in items if i["lang"] == "ml" and i["label"] == "offer_help"}
    assert hi_common != ml_common  # different common-class tweets per language


@pytest.mark.parametrize("text,source,problem", [
    ("", "x y z", "empty"),
    ("thudangikkഴിഞ്ഞu ennu parayunnu", "they say it started", "not Latin script"),
    ("Same text here", "same text, here!", "copy of the source"),
    ("ok", "a much longer original tweet about the flood relief camp", "length far from the source"),
    ("Kochi camp me khana chahiye", "Kochi camp needs food", None),
])
def test_valid(text, source, problem):
    assert mrd.valid(text, source) == problem


def test_wrong_language_is_rejected():
    hinglish = "Greek wildfires me ek Irish man missing hai, sabko dar lag raha hai"
    assert mrd.valid(hinglish, "An Irish man is missing in the Greek wildfires", "bn-Latn") == "wrong language (hi-Latn)"
    assert mrd.valid(hinglish, "An Irish man is missing in the Greek wildfires", "hi-Latn") is None
    banglish = "Maryland floods e ekjon hero soldier bhese geche, rescuer ra take khujche"
    assert mrd.valid(banglish, "Rescuers search for hero soldier swept away in Maryland floods", "bn-Latn") is None


def test_clean_strips_rt_mentions_urls():
    assert mrd.clean("RT @AbbTakk: AJK quake me 22 log dead http://t.co/x via @news") == "AJK quake me 22 log dead via"
    assert mrd.clean("RT ShashiTharoor: Trivandrum NGO relief collect kar rahi") == "Trivandrum NGO relief collect kar rahi"


class FakeGemini:
    model = "fake-flash-lite"

    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def complete(self, system, user, max_tokens=600):
        self.calls.append((system, user))
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("LLM request failed: HTTP 503")
        items = json.loads(user)
        return "```json\n" + json.dumps([{"id": i["id"], "text": f"romanised {i['text']} hai bhai"} for i in items]) + "\n```"


def test_generate_is_cached_and_resumable(humaid, tmp_path, monkeypatch):
    monkeypatch.setattr(mrd, "CACHE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(mrd.time, "sleep", lambda s: None)
    items = mrd.plan(mrd.load_train(humaid / "train.csv"), 12, ["hi"], seed=1)
    fake = FakeGemini(fail_first=1)  # first attempt fails, retried
    cache, stats = mrd.generate(items, fake, batch_size=5, sleep=0, cache_path=mrd.CACHE, max_requests=1)
    assert stats["retries"] == 1 and stats["generated"] == 5 and stats["stopped_at_request_cap"] == 1
    assert "romanised Hindi" in fake.calls[0][0] and "Devanagari" in fake.calls[0][0]
    # second run resumes: only the remaining items are requested
    fake2 = FakeGemini()
    cache, stats = mrd.generate(items, fake2, batch_size=5, sleep=0, cache_path=mrd.CACHE, max_requests=None)
    assert stats["cached"] == 5 and stats["generated"] == len(items) - 5
    df = mrd.to_frame(items, cache)
    assert len(df) == len(items) and set(df.lang) == {"hi-Latn"} and set(df.model) == {"fake-flash-lite"}
    assert {"id", "source_id", "label", "text", "source_text", "langid_ok"} <= set(df.columns)
    # usable by train.py's --extra-train loader
    from aquaalert.ingest.loader import load_labeled

    path = tmp_path / "out.csv"
    df.to_csv(path, index=False)
    assert len(load_labeled(path)) == len(items)


def test_unparseable_reply_does_not_crash(humaid, tmp_path, monkeypatch):
    class Garbage(FakeGemini):
        def complete(self, system, user, max_tokens=600):
            return "sorry, I can't"

    items = mrd.plan(mrd.load_train(humaid / "train.csv"), 8, ["bn"], seed=1)
    cache, stats = mrd.generate(items, Garbage(), batch_size=10, sleep=0, cache_path=None, max_requests=None)
    assert stats["unparseable_batches"] == 1 and cache == {}

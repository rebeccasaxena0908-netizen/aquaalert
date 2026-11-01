from datetime import datetime

import pytest

from aquaalert.credibility.evaluate import precision_recall, scored_sample
from aquaalert.credibility.heuristics import combine, contradiction_flags, dam_names, duplicate_flags, score_posts
from aquaalert.schemas import Extraction, GeoPoint, Label, Post

ALAPPUZHA = (9.4981, 76.3388)


def post(id, text, ts="2018-08-15 10:00", label=Label.INFO_UPDATE, author=None, geo=ALAPPUZHA, district="Alappuzha"):
    return Post(id=id, text=text, timestamp=datetime.fromisoformat(ts), label=label, author=author or id,
                extraction=Extraction(geo=GeoPoint(lat=geo[0], lon=geo[1]), geo_source="Alappuzha", district=district))


def score(text, **kw):
    p = score_posts([post("x", text, **kw)])[0]
    return p.misinfo_score, p.misinfo_reasons


def test_combine_is_noisy_or():
    assert combine([]) == 0 and combine([0.5]) == 0.5 and combine([0.5, 0.5]) == pytest.approx(0.75)


@pytest.mark.parametrize("text,code_words", [
    ("Mullaperiyar dam has broken, water will reach Kochi in 2 hours", "dam breach"),
    ("500 people dead in Chengannur", "death toll"),
    ("Forward to all groups, very urgent", "forward"),
    ("Share before it is deleted", "forward"),
    ("Sabko forward karo, bahut zaroori", "forward"),
    ("മുല്ലപ്പെരിയാർ ഡാം പൊട്ടി", "dam breach"),
    ("Video of the Kedarnath flood, Kerala now", "older event"),
    ("Remember the 2013 floods? Same thing happening now", "mentions 2013"),
    ("The government is hiding the real numbers", "hidden"),
])
def test_each_signal_fires_with_a_reason(text, code_words):
    s, reasons = score(text)
    assert s > 0 and any(code_words in r for r in reasons), (text, reasons)


@pytest.mark.parametrize("text", [
    "Need rescue for 6 people near Pandalam bridge, elderly woman with us",
    "Relief camp at Thiruvalla needs rice and blankets",
    "As per IMD bulletin, red alert in Idukki for the next 24 hours",
    "Water level at Mullaperiyar dam is 142 ft, authorities issue red alert",
    "Volunteers with 3 boats available in Thrissur",
])
def test_normal_posts_score_zero(text):
    assert score(text) == (0.0, [])


def test_source_and_debunk_lower_the_score():
    unsourced, _ = score("Idukki dam breached!! forward to all")
    sourced, reasons = score("Idukki dam breached according to the District Collector, forward to all")
    assert 0 < sourced < unsourced and any("official source" in r for r in reasons)
    # a debunk describes the rumour (old year, forwarding) without being one
    assert score("This photo is from the 2015 Chennai floods, not Aluva. Please don't forward it.") == (0.0, [])


def test_url_counts_as_source():
    p = post("x", "Dam breached at Idukki, 30 dead")
    linked = score_posts([p.model_copy(update={"url": "https://news.example/report"})])[0]
    assert linked.misinfo_score == 0  # attributed claims aren't 'unverifiable'


def test_duplicates_across_accounts():
    text = "URGENT 50 people trapped in Ranni school send boats now"
    ps = [post(f"d{i}", text + ("!" * i), author=f"a{i}") for i in range(3)] + [post("o", "something else entirely here")]
    flags = duplicate_flags(ps)
    assert set(flags) == {"d0", "d1", "d2"} and "3 times by 3 account" in flags["d0"].reason
    assert duplicate_flags([post("a", "help"), post("b", "help")]) == {}  # too short to call a copy


def test_calm_claim_contradicted_by_nearby_rescues_and_brief():
    calm = post("c", "Water has receded, everyone is safe now", ts="2018-08-15 10:00")
    near = [post(f"r{i}", "stuck on roof", ts=f"2018-08-15 {9 + i:02d}:00", label=Label.RESCUE_NEEDED) for i in range(2)]
    far = post("f", "stuck on roof", ts="2018-08-15 23:00", label=Label.RESCUE_NEEDED)  # 13 h later
    flags = contradiction_flags([calm, *near, far])
    assert flags["c"].code == "contradicts_nearby" and "2 rescue requests" in flags["c"].reason
    assert "c" not in contradiction_flags([calm, far])

    class Brief:
        district, date, counts = "Alappuzha", datetime(2018, 8, 15).date(), {"rescue_needed": 4}

    assert contradiction_flags([calm], [Brief()])["c"].code == "contradicts_brief"


def test_dam_claim_vs_attributed_denial_matches_dam_names():
    denial = post("d", "Mullaperiyar dam is safe, no breach - District Collector", ts="2018-08-16 02:00")
    same = post("a", "Mullaperiyar dam has broken!!", ts="2018-08-16 01:00")
    malayalam = post("m", "മുല്ലപ്പെരിയാർ ഡാം പൊട്ടി!", ts="2018-08-16 01:30")
    other = post("o", "Idukki dam toot gaya hai", ts="2018-08-16 01:30")
    flags = contradiction_flags([denial, same, malayalam, other])
    assert flags["a"].code == flags["m"].code == "contradicted_by_official"
    assert "o" not in flags  # a different dam
    assert dam_names("മുല്ലപ്പെരിയാർ ഡാം പൊട്ടി") == {"mullaperiyar"}


def test_scores_written_to_post_and_credibility_is_complement():
    p = score_posts([post("x", "Forward to all: dam broke!!!")])[0]
    assert 0 < p.misinfo_score <= 1 and p.credibility == pytest.approx(1 - p.misinfo_score)
    assert len(p.misinfo_reasons) >= 2


def test_sample_rumours_separate_from_normal_posts():
    import pandas as pd

    from aquaalert.config import ROOT

    gold = dict(zip(*[pd.read_csv(ROOT / "data/sample_posts_rumours.csv")[c] for c in ("id", "misinfo")]))
    posts = scored_sample()
    gold_all = {p.id: gold.get(p.id, 0) for p in posts}
    pr = precision_recall(posts, gold_all, 0.3)
    assert pr["precision"] == 1.0 and pr["recall"] == 1.0
    assert all(p.misinfo_score == 0 for p in posts if p.id.startswith(("s", "m")))  # English + multilingual samples

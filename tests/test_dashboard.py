import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.extract.geocode import Geocoder
from aquaalert.ingest.loader import load_posts
from aquaalert.pipeline import process_posts
from aquaalert.match.matcher import hashing_encoder, match_posts
from aquaalert.extract.gazetteer import ALL_PLACES, _load_csv
from dashboard.helpers import (IRRELEVANT_HIDE_SCORE, LABEL_COLORS, SEVERITY_COLORS, STATE_POINTS, alert_popup, build_map, build_match_map,
                               hidden_as_irrelevant, jitter, place_alerts, posts_to_frame)
from aquaalert.schemas import LABELS, Label

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def kpis(at) -> dict[str, int]:
    """Values of the glass KPI cards (rendered as HTML with data-kpi / data-value attributes)."""
    return {k: int(v) for md in at.markdown
            for k, v in re.findall(r'data-kpi="(\w+)" data-value="(\d+)"', md.value)}


def page_html(at) -> str:
    return "\n".join(md.value for md in at.markdown)


def _processed(sample_path, tmp_path):
    return process_posts(load_posts(sample_path), KeywordZeroShot(), Geocoder(cache_path=tmp_path / "g.json", online=False))


def test_colors_cover_all_labels():
    assert set(LABEL_COLORS) == set(LABELS)


def test_frame_and_map(sample_path, tmp_path):
    posts = _processed(sample_path, tmp_path)
    df = posts_to_frame(posts)
    assert len(df) == 50 and df["lat"].notna().sum() > 40
    html = build_map(posts).get_root().render()
    assert html.count("L.circleMarker(") == df["lat"].notna().sum()


def test_jitter_deterministic_and_small():
    assert jitter("a") == jitter("a") != jitter("b")
    assert all(abs(v) < 0.01 for v in jitter("zzz"))


def test_app_runs_end_to_end(monkeypatch):
    for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(k, "")  # tests never depend on (or call) a real LLM
    at = AppTest.from_file(str(APP), default_timeout=600).run()  # a fresh machine downloads the 470 MB embedding model here
    assert not at.exception and not at.error
    k = kpis(at)
    assert k["posts"] == 110 and k["rescue"] > 0 and {"warnings", "flagged"} <= set(k)
    html = page_html(at)
    assert "aa-header" in html and "aa-sky" in html and "--navy" in html  # theme + background injected
    assert [t.label for t in at.tabs] == ["Map", "Warnings", "Daily brief", "Needs ↔ Offers", "Chatbot"]
    assert re.search(r"\d+ matches", html) and any("rescue request" in e.label for e in at.expander)  # briefs rendered


def test_match_map(sample_path, tmp_path):
    posts = _processed(sample_path, tmp_path)
    matches = match_posts(posts, hashing_encoder, min_score=0.0)
    html = build_match_map(matches, posts).get_root().render()
    assert matches and html.count("L.polyline(") == len(matches)


def test_rule_fired_is_shown(make_post, tmp_path):
    from aquaalert.classify.base import classify_posts
    from aquaalert.extract.entities import extract_posts

    class AlwaysOffer:
        name = "offer"

        def predict_scores(self, texts):
            return [{"offer_help": 1.0} for _ in texts]

    posts = extract_posts(classify_posts([make_post("Need rescue! stuck on the rooftop in Aluva")], AlwaysOffer()),
                          Geocoder(cache_path=tmp_path / "g.json", online=False))
    row = posts_to_frame(posts).iloc[0]
    assert row["rule"] == "distress_strong" and row["model_label"] == "offer_help" and row["label"] == "rescue_needed"
    rendered = build_map(posts).get_root().render()
    assert "rule: distress strong" in rendered and "the classifier said offer_help" in rendered


def test_chatbot_tab_answers_from_briefs_offline(monkeypatch):
    for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(k, "")  # force the offline provider
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.chat_input[0].set_value("Where are rescue requests in Alappuzha?").run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "Alappuzha" in at.chat_message[1].markdown[0].value


def test_brief_tab_waits_for_click_when_an_llm_is_configured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-not-used")
    monkeypatch.setenv("AQUAALERT_LLM_PROVIDER", "gemini")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    brief_panels = [e for e in at.expander if "rescue request" in e.label]
    assert not at.exception and brief_panels == []  # no briefs generated, so no network call was made
    assert any("Generate briefs" in b.label for b in at.button)


def test_low_credibility_filter_hides_flagged_posts(monkeypatch):
    for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(k, "")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    before_html = page_html(at)
    before = int(re.search(r"(\d+) shown with current filters", before_html).group(1))
    flagged = kpis(at)["flagged"]
    assert flagged > 0
    at.sidebar.checkbox[0].check().run()
    after = int(re.search(r"(\d+) shown with current filters", page_html(at)).group(1))
    assert not at.exception and after < before


def test_frame_has_language_and_misinfo_columns(sample_path, tmp_path):
    from aquaalert.credibility.heuristics import score_posts
    from aquaalert.language.detect import detect_posts

    posts = score_posts(detect_posts(_processed(sample_path, tmp_path)))
    df = posts_to_frame(posts)
    assert {"language", "misinfo", "misinfo_reasons"} <= set(df.columns) and set(df["language"]) >= {"en", "hi-Latn"}


def test_risk_layer_and_warnings_tab(monkeypatch, sample_path, tmp_path):
    from datetime import date

    from aquaalert.fuse.warnings import compute_warnings

    posts = _processed(sample_path, tmp_path)
    ws = compute_warnings(posts, day=date(2018, 8, 15))
    html = build_map(posts, ws).get_root().render()
    assert html.count("L.circle(") == len([w for w in ws if w.district.lower() in __import__("aquaalert.extract.gazetteer", fromlist=["x"]).KERALA_PLACES])
    for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(k, "")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    k = kpis(at)
    assert not at.exception and {"warnings_tab", "watch_tab"} <= set(k)


def test_pause_background_toggle(monkeypatch):
    for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    paused_rule = "animation-play-state: paused !important"
    assert paused_rule not in page_html(at)
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception and paused_rule in page_html(at)


def test_post_text_is_escaped_in_popups(make_post, tmp_path):
    from aquaalert.classify.base import classify_posts
    from aquaalert.classify.zeroshot import KeywordZeroShot
    from aquaalert.extract.entities import extract_posts

    evil = make_post('<img src=x onerror="alert(1)"> people trapped in Aluva, need rescue')
    posts = extract_posts(classify_posts([evil], KeywordZeroShot()), Geocoder(cache_path=tmp_path / "g.json", online=False))
    rendered = build_map(posts).get_root().render()
    assert "<img src=x" not in rendered and "onerror" not in rendered.split("&lt;img")[0]


def test_brief_shows_phone_redaction_marker(monkeypatch):
    for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    day = next(s for s in at.selectbox if s.label == "Day")
    day.set_value(next(o for o in day.options if "2018-08-15" in str(o))).run()  # the day with Chengannur's SOS
    briefs = [md.value for e in at.expander if "rescue request" in e.label for md in e.markdown]
    assert any("&lt;phone&gt;" in b for b in briefs)  # rendered as the visible text "<phone>"


# ---------------------------------------------------------------- Live mode (no network: poller off, temp DB)
def live_app(monkeypatch, tmp_path):
    for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("AQUAALERT_MODE", "live")
    monkeypatch.setenv("AQUAALERT_LIVE_POLL", "off")
    monkeypatch.setenv("AQUAALERT_LIVE_DB", str(tmp_path / "live.sqlite"))
    monkeypatch.setenv("AQUAALERT_CLASSIFIER", "zeroshot")
    return tmp_path / "live.sqlite"


def source_counts(at) -> dict[str, int]:
    return {k: int(v) for k, v in re.findall(r'data-source="(\w+)" data-count="(\d+)"', page_html(at))}


def test_live_mode_empty_state_never_pads_with_samples(monkeypatch, tmp_path):
    live_app(monkeypatch, tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    html = page_html(at)
    assert not at.exception and "No flood-related items in the last 24h" in html
    assert kpis(at)["posts"] == 0 and source_counts(at) == {"sachet": 0, "news": 0, "bluesky": 0}
    assert 'class="aa-ds aa-ds-live"' in html and 'class="aa-ds aa-ds-synthetic"' not in html


def test_live_mode_sources_official_alerts_and_simulation(monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone

    from aquaalert.ingest.live import LiveItem, LiveStore, parse_cap

    db = live_app(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    cap = parse_cap((Path(__file__).parent / "fixtures" / "live" / "sachet_cap_flood.xml").read_bytes())
    LiveStore(db).set_news_enabled(True)
    LiveStore(db).add([
        LiveItem(id="sachet:T1", source="sachet", kind="official_alert", text=cap["headline"],
                 url="https://sachet.ndma.gov.in/x", published_at=now - timedelta(hours=1), extra=cap),
        LiveItem(id="news:N1", source="news", kind="news", text="Heavy rain triggers flood alert in Aluva, schools shut",
                 url="https://example-news.in/a", published_at=now - timedelta(hours=2), lang="en"),
        LiveItem(id="news:OLD", source="news", kind="news", text="Flood in Aluva last week", url="https://x/old",
                 published_at=now - timedelta(days=3)),  # outside the 24 h window
    ])
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    html = page_html(at)
    assert not at.exception and source_counts(at) == {"sachet": 1, "news": 1, "bluesky": 0}
    assert "Official alert" in html and "Periyar river above danger level" in html and "CAP source" in html
    assert "No flood-related items" not in html and kpis(at)["posts"] == 1

    at.sidebar.button(key="sim_inject").click().run()
    k = kpis(at)
    assert not at.exception and k["posts"] == 1 + 8 and k["flagged"] >= 1  # the dam-breach rumour
    assert source_counts(at) == {"sachet": 1, "news": 1, "bluesky": 0}  # simulated posts never counted as sources
    assert re.search(r"[1-9]\d* matches", page_html(at))  # the boats offer matches rescue requests
    at.sidebar.button(key="sim_clear").click().run()
    assert kpis(at)["posts"] == 1


def test_replay_sample_is_marked_synthetic(monkeypatch):
    for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    html = page_html(at)
    assert 'class="aa-ds aa-ds-replay"' in html and 'class="aa-ds aa-ds-synthetic"' in html
    assert 'class="aa-ds aa-ds-live"' not in html


def test_simulated_badge_in_popup_and_table(tmp_path):
    from aquaalert.classify.zeroshot import KeywordZeroShot
    from aquaalert.ingest.live import to_posts
    from aquaalert.ingest.simulate import simulated_items
    from aquaalert.pipeline import process_posts

    posts = process_posts(to_posts(simulated_items()), KeywordZeroShot(), Geocoder(cache_path=tmp_path / "g.json", online=False))
    assert all(p.source == "simulated" for p in posts)
    rendered = build_map(posts).get_root().render()
    assert rendered.count("aa-badge aa-sim") == sum(1 for p in posts if p.extraction and p.extraction.geo)
    assert set(posts_to_frame(posts)["source"]) == {"SIMULATED"}


def test_live_source_cards_show_state_last_success_and_note(monkeypatch, tmp_path):
    from aquaalert.ingest.live import LiveStore

    db = live_app(monkeypatch, tmp_path)
    st = LiveStore(db)
    st.set_news_enabled(True)
    st.set_status("sachet", state="slow", last_success="2026-11-01T16:05:00+00:00", duration=118.0,
                  note="time budget reached after 9 of 25 new CAP documents")
    st.set_status("news", state="failed", last_error="rate limited by api.gdeltproject.org (HTTP 429)", duration=6.0)
    st.set_status("bluesky", state="skipped", note="not configured (no BLUESKY_HANDLE / BLUESKY_APP_PASSWORD)")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    html = page_html(at)
    states = dict(re.findall(r'data-source="(\w+)" data-count="\d+" data-state="(\w+)"', html))
    assert not at.exception and states == {"sachet": "slow", "news": "failed", "bluesky": "skipped"}
    assert "last success 16:05 UTC · took 118s" in html and "time budget reached" in html and "HTTP 429" in html
    assert "aa-badge aa-err\">failed" in html and "aa-badge aa-warn\">slow" in html


def test_hidden_as_irrelevant_needs_high_confidence(make_post):
    def labelled(label, score):
        return make_post("x").model_copy(update={"label": label, "label_score": score})

    assert hidden_as_irrelevant(labelled(Label.IRRELEVANT, IRRELEVANT_HIDE_SCORE))
    assert not hidden_as_irrelevant(labelled(Label.IRRELEVANT, IRRELEVANT_HIDE_SCORE - 0.01))
    assert not hidden_as_irrelevant(labelled(Label.INFO_UPDATE, 0.99))


class ScriptedClassifier:
    """Confidently irrelevant for "biryani", unsure for "maybe", otherwise an info update."""
    name = "scripted"

    def predict_scores(self, texts):
        def score(t):
            label, conf = ("irrelevant", 0.95) if "biryani" in t else ("irrelevant", 0.5) if "maybe" in t else ("info_update", 0.9)
            rest = (1 - conf) / (len(LABELS) - 1)
            return {l: conf if l == label else rest for l in LABELS}
        return [score(t) for t in texts]


def test_live_mode_hides_confident_irrelevant_but_category_filter_reaches_them(monkeypatch, tmp_path):
    from aquaalert.ingest.live import LiveItem, LiveStore

    db = live_app(monkeypatch, tmp_path)
    monkeypatch.setattr("aquaalert.classify.model.get_classifier", lambda *a, **k: ScriptedClassifier())
    st.cache_resource.clear()  # the cached classifier from earlier tests must not be reused, nor this one leak out
    st.cache_data.clear()
    now = datetime.now(timezone.utc)
    texts = ("Flood alert issued for Patna", "Flood themed biryani night", "maybe a flood somewhere")
    LiveStore(db).set_news_enabled(True)
    LiveStore(db).add([LiveItem(id=f"news:{i}", source="news", kind="news", text=t, url="https://x/a",
                                published_at=now - timedelta(hours=1)) for i, t in enumerate(texts)])
    try:
        at = AppTest.from_file(str(APP), default_timeout=120).run()
        assert not at.exception, at.exception
        shown = set(at.dataframe[0].value["text"])
        assert shown == {texts[0], texts[2]}  # the unsure irrelevant post stays on screen
        assert any("1 live posts hidden" in c.value for c in at.caption)
        assert kpis(at)["posts"] == 3
        box = next(m for m in at.sidebar.multiselect if m.label == "Categories")
        box.set_value([*box.value, "irrelevant"]).run()
        assert set(at.dataframe[0].value["text"]) == set(texts)
    finally:
        st.cache_resource.clear()
        st.cache_data.clear()


def test_nothing_identifying_a_bluesky_account_reaches_the_screen(monkeypatch, tmp_path):
    from aquaalert.ingest.live import LiveStore, parse_bsky_post

    fixture = json.loads((Path(__file__).parent / "fixtures" / "live" / "bsky_search.json").read_text())
    db = live_app(monkeypatch, tmp_path)
    post = fixture["posts"][0]
    post = {**post, "record": {**post["record"], "createdAt": datetime.now(timezone.utc).isoformat()},
            "author": {**post["author"], "displayName": "Example Person"}}
    LiveStore(db).add([parse_bsky_post(post)])
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    frame = at.dataframe[0].value
    shown = page_html(at) + frame.to_csv()
    assert "resident.example" not in shown and "Example Person" not in shown and "author" not in frame.columns
    assert "Water entering houses near Aluva" in shown and "/post/3kabc111" in shown


# ---------------------------------------------------------------- official alerts on the map
def alert(id="a", severity="Severe", area="Ernakulam district: Aluva, Paravur", polygons=(), msg_type="Alert",
          expires="2099-01-01T00:00:00+05:30", url="https://sachet.ndma.gov.in/x?identifier=1", sender="", headline=""):
    from aquaalert.ingest.live import LiveItem

    return LiveItem(id=f"sachet:{id}", source="sachet", kind="official_alert", text=f"headline {id}", url=url,
                    published_at=datetime.now(timezone.utc),
                    extra={"event": "Flood", "severity": severity, "area": area, "expires": expires, "msg_type": msg_type,
                           "polygons": list(polygons), "sender": sender, "headline": headline})


RING = [[10.0, 76.2], [10.2, 76.2], [10.2, 76.4], [10.0, 76.2]]


def test_place_alerts_polygon_first_then_district_then_skip():
    placed, skipped = place_alerts([
        alert("poly", polygons=[RING]), alert("district"), alert("two", area="Patna, Gaya and Kochi"),
        alert("file", area="heavy_rainfall_2026-11-02"), alert("taluka", area="Dharbandora taluka,Sanguem taluka"),
        alert("cancel", msg_type="Cancel"), alert("old", expires="2020-01-01T00:00:00+05:30"),
    ])
    by_id = {p.item.id: p for p in placed}
    assert by_id["sachet:poly"].polygons == [RING] and by_id["sachet:poly"].markers == []
    assert [d for d, _ in by_id["sachet:district"].markers] == ["Ernakulam"]  # three places, one district, one marker
    assert by_id["sachet:district"].markers[0][1] == ALL_PLACES["ernakulam"]
    assert [d for d, _ in by_id["sachet:two"].markers] == ["Patna", "Gaya", "Ernakulam"]
    assert skipped == {"no polygon, and the area names no known district": 2, "cancelled": 1, "expired": 1}


def test_alerts_are_drawn_differently_from_posts_with_a_full_popup():
    evil = alert("x", area='Ernakulam <img src=x onerror="a()">', url="https://sachet.ndma.gov.in/x?identifier=1")
    placed, _ = place_alerts([alert("poly", severity="Extreme", polygons=[RING]), alert("mod", severity="Moderate"), evil])
    page = build_map([], alerts=placed).get_root().render()
    assert page.count("L.polygon(") == 1 and page.count("L.circleMarker(") == 0 and "M12 2.5 23 21.5H1Z" in page
    assert SEVERITY_COLORS["extreme"] in page and SEVERITY_COLORS["moderate"] in page and SEVERITY_COLORS["severe"] in page
    assert "Official alert (NDMA SACHET)" in page and "Severity:" in page and "Area:" in page and "Expires:" in page
    assert "2099-01-01 00:00:00+05:30" in page and "sachet.ndma.gov.in/x?identifier=1" in page and "original alert" in page
    assert "<img src=x" not in page


def test_alert_popup_drops_a_link_that_is_not_http():
    assert "href" not in alert_popup(alert(url="javascript:alert(1)"))


def test_live_map_shows_alert_count_and_legend(monkeypatch, tmp_path):
    from aquaalert.ingest.live import LiveStore

    db = live_app(monkeypatch, tmp_path)
    LiveStore(db).add([alert("a"), alert("b", area="heavy_rainfall_2026-11-02"), alert("c", msg_type="Cancel")])
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert "Official alert (NDMA SACHET)" in page_html(at) and "Cancelled by the issuer" in page_html(at)
    captions = " ".join(c.value for c in at.caption)
    assert "1 of 3 official alerts on the map" in captions and "1 cancelled" in captions
    assert "1 no polygon, and the area names no known district" in captions


def test_replay_map_has_no_official_alert_legend(monkeypatch):
    monkeypatch.delenv("AQUAALERT_MODE", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception and "Official alert (NDMA SACHET)" not in page_html(at)


def test_gdelt_card_is_experimental_off_by_default_and_toggle_enables_it(monkeypatch, tmp_path):
    from aquaalert.ingest.live import LiveItem, LiveStore

    db = live_app(monkeypatch, tmp_path)
    store = LiveStore(db)
    store.add([LiveItem(id="news:1", source="news", kind="news", text="Flood alert issued for Patna", url="https://x/a",
                        published_at=datetime.now(timezone.utc) - timedelta(hours=1))])
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    states = dict(re.findall(r'data-source="(\w+)" data-count="\d+" data-state="(\w+)"', page_html(at)))
    assert not at.exception and states["news"] == "off" and "experimental" in page_html(at)
    assert source_counts(at)["news"] == 0 and kpis(at)["posts"] == 0 and not store.news_enabled()
    at.sidebar.toggle(key="news_enabled").set_value(True).run()
    assert not at.exception and store.news_enabled() and source_counts(at)["news"] == 1 and kpis(at)["posts"] == 1


# ---------------------------------------------------------------- state-level official alerts
AP_TEXT = ("Due to the impact of Cyclone Murasu in the Bay of Bengal, heavy rains accompanied by strong winds are likely in "
           "your area today. The public is advised to remain alert and take necessary precautions. -Government of Andhra Pradesh")


def ap(id="ap", area="heavy_rainfall_2026-11-02", sender="Andhra-Pradesh-SDMA", headline=AP_TEXT, **kw):
    return alert(id, area=area, sender=sender, headline=headline, **kw)


def test_state_level_only_when_no_finer_area_is_given():
    assert place_alerts([ap()])[0][0].state == "Andhra Pradesh"
    assert place_alerts([ap(area="", sender="Kerala-SDMA")])[0][0].state == "Kerala"
    assert place_alerts([alert(area="Odisha", sender="IMD-Bhubaneswar", headline="Heavy rain likely")])[0][0].state == "Odisha"


def test_state_level_is_refused_when_it_would_be_untrue():
    refused = [
        ap("taluka", area="Dharbandora taluka,Sanguem taluka", sender="Goa-SDMA"),  # area is specified, we just can't place it
        ap("imd", sender="IMD-Goa"),  # a weather office, not a state authority
        ap("unknown", sender="Atlantis-SDMA"),
        ap("named", headline="Heavy rain likely in Krishna and Guntur districts"),
        ap("known", headline="Heavy rain likely around Patna"),
        ap("cancel", msg_type="Cancel"), ap("old", expires="2020-01-01T00:00:00+05:30"),
    ]
    placed, skipped = place_alerts(refused)
    assert placed == [] and skipped["cancelled"] == 1 and skipped["expired"] == 1
    assert skipped["no polygon, and the area names no known district"] == 5


def test_finer_placement_wins_over_state():
    placed, _ = place_alerts([ap("p", polygons=[RING]), ap("d", area="Patna")])
    assert [p.state for p in placed] == [None, None] and placed[0].polygons and placed[1].markers


def test_state_alerts_share_one_distinct_banner_with_an_honest_popup():
    placed, _ = place_alerts([ap("a"), ap("b", severity="Extreme"), ap("c", area="very_heavy_rainfall_2026-11-02"),
                              alert("d", area="Patna")])
    page = build_map([], alerts=placed).get_root().render()
    assert page.count("M5 3h16l-4 5 4 5H5Z") == 1 and page.count("M12 2.5 23 21.5H1Z") == 1  # one banner, one triangle
    assert "3 state-level official alert(s)" in page and SEVERITY_COLORS["extreme"] in page  # worst severity colours it
    assert "the affected districts are not specified" in page and "not the affected area" in page
    assert "Andhra-Pradesh-SDMA" in page and "16.85" in page and "80.8667" in page
    assert "heavy_rainfall_2026" not in page  # a file name is not shown as if it were an area


def test_state_points_are_sourced_and_inside_india():
    rows = _load_csv("gazetteer_states.csv")
    assert len(rows) == 36 == len({r["wikidata"] for r in rows})
    for r in rows:
        assert re.fullmatch(r"Q\d+", r["wikidata"]) and 6 < float(r["lat"]) < 37 and 68 < float(r["lon"]) < 98, r["state"]
    assert set(STATE_POINTS) == {r["state"] for r in rows}


def test_live_page_counts_state_alerts_separately(monkeypatch, tmp_path):
    from aquaalert.ingest.live import LiveStore

    db = live_app(monkeypatch, tmp_path)
    LiveStore(db).add([ap("a"), ap("b", area="very_heavy_rainfall_2026-11-02"), alert("c", area="Patna"),
                       alert("t", area="Dharbandora taluka")])
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert "Official alert, state-level (area not specified)" in page_html(at)
    assert "3 of 4 official alerts on the map (2 at state level, affected districts not specified)" in " ".join(
        c.value for c in at.caption)

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.extract.geocode import Geocoder
from aquaalert.ingest.loader import load_posts
from aquaalert.pipeline import process_posts
from aquaalert.match.matcher import hashing_encoder, match_posts
from dashboard.helpers import LABEL_COLORS, build_map, build_match_map, jitter, posts_to_frame
from aquaalert.schemas import LABELS

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
    at = AppTest.from_file(str(APP), default_timeout=60).run()
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

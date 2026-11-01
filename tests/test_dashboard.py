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
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Posts shown"]) > 0 and int(metrics["On map"]) > 0
    assert [t.label for t in at.tabs] == ["Map", "Daily brief", "Needs ↔ Offers", "Chatbot"]
    assert "Matches" in metrics and len(at.expander) > 0  # district briefs rendered


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
    assert "rule: distress_strong (model said offer_help)" in build_map(posts).get_root().render()


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
    assert not at.exception and len(at.expander) == 0  # nothing generated, no network call made
    assert any("Generate briefs" in b.label for b in at.button)

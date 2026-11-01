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


def test_app_runs_end_to_end():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception and not at.error
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Posts shown"]) > 0 and int(metrics["On map"]) > 0
    assert [t.label for t in at.tabs] == ["Map", "Daily brief", "Needs ↔ Offers"]
    assert "Matches" in metrics and len(at.expander) > 0  # district briefs rendered


def test_match_map(sample_path, tmp_path):
    posts = _processed(sample_path, tmp_path)
    matches = match_posts(posts, hashing_encoder, min_score=0.0)
    html = build_match_map(matches, posts).get_root().render()
    assert matches and html.count("L.polyline(") == len(matches)

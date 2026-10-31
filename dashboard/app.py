"""AquaAlert dashboard. Run from the repo root: streamlit run dashboard/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from aquaalert.classify.model import get_classifier  # noqa: E402
from aquaalert.config import get_settings  # noqa: E402
from aquaalert.extract.geocode import Geocoder  # noqa: E402
from aquaalert.ingest.loader import load_posts  # noqa: E402
from aquaalert.pipeline import process_posts  # noqa: E402
from dashboard.helpers import LABEL_COLORS, build_map, posts_to_frame  # noqa: E402

st.set_page_config(page_title="AquaAlert", page_icon="🌊", layout="wide")


@st.cache_resource
def _classifier():
    return get_classifier()


@st.cache_data(show_spinner="Classifying and geocoding posts...")
def _processed(path: str):
    return process_posts(load_posts(path), _classifier(), Geocoder())


st.title("🌊 AquaAlert — flood crisis map")
settings = get_settings()
path = st.sidebar.text_input("Dataset (CSV/JSON/JSONL)", str(settings.dataset))
try:
    posts = _processed(path)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()
if not posts:
    st.warning("Dataset has no usable posts.")
    st.stop()

st.sidebar.caption(f"Classifier: `{_classifier().name}`")
labels = st.sidebar.multiselect("Categories", list(LABEL_COLORS), default=[l for l in LABEL_COLORS if l != "irrelevant"])
min_conf = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

# Replay: reveal posts up to a simulated clock, like a live feed.
times = [p.timestamp for p in posts]
replay_at = st.sidebar.slider(
    "Replay time", min_value=min(times), max_value=max(times), value=max(times), format="MMM D, HH:mm",
)
visible = [
    p for p in posts
    if p.timestamp <= replay_at and p.label and p.label.value in labels and (p.label_score or 0) >= min_conf
]

df = posts_to_frame(visible)
mapped = int(df["lat"].notna().sum()) if len(df) else 0
c1, c2, c3 = st.columns(3)
c1.metric("Posts shown", len(visible))
c2.metric("On map", mapped)
c3.metric("Rescue requests", int((df["label"] == "rescue_needed").sum()) if len(df) else 0)

map_col, legend_col = st.columns([4, 1])
with map_col:
    st_folium(build_map(visible), height=560, use_container_width=True, returned_objects=[])
with legend_col:
    st.subheader("Legend")
    for label, color in LABEL_COLORS.items():
        st.markdown(f"<span style='color:{color}'>●</span> {label}", unsafe_allow_html=True)
    if len(df):
        st.subheader("Counts")
        st.bar_chart(df["label"].value_counts())

st.subheader("Posts")
st.dataframe(df.sort_values("time", ascending=False) if len(df) else df, width="stretch", hide_index=True)

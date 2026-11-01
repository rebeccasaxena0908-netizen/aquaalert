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
from aquaalert.match.matcher import default_encoder, match_posts  # noqa: E402
from aquaalert.chatbot.rag import brief_passages, default_chatbot  # noqa: E402
from aquaalert.summarize.brief import daily_briefs  # noqa: E402
from aquaalert.summarize.providers import TemplateProvider, get_provider  # noqa: E402
from dashboard.helpers import LABEL_COLORS, build_map, build_match_map, posts_to_frame  # noqa: E402

st.set_page_config(page_title="AquaAlert", page_icon="🌊", layout="wide")


@st.cache_resource
def _classifier():
    return get_classifier()


@st.cache_resource
def _encoder():
    return default_encoder()


@st.cache_resource(show_spinner="Loading guideline index...")
def _chatbot():
    return default_chatbot()


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

# Briefs and matching look at everything up to the replay clock, regardless of the category filter.
so_far = [p for p in posts if p.timestamp <= replay_at]
tab_map, tab_brief, tab_match, tab_chat = st.tabs(["Map", "Daily brief", "Needs ↔ Offers", "Chatbot"])

with tab_map:
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

with tab_brief:
    try:
        provider = get_provider()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    days = sorted({p.timestamp.date() for p in so_far})
    st.caption(f"Provider: `{provider.name}` / `{provider.model}`"
               + ("" if provider.uses_network else " — offline summaries (add GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY to .env for LLM summaries)"))
    if not days:
        st.info("No posts yet at this replay time.")
    else:
        day = st.selectbox("Day", days, index=len(days) - 1)
        # LLM providers cost money: only call them on request (results are cached on disk).
        if provider.uses_network and not st.button("Generate briefs", key="gen_briefs"):
            st.info("Click **Generate briefs** to call the LLM (responses are cached in data/cache/summaries.json).")
        else:
            briefs = daily_briefs(so_far, provider, day=day)
            if not briefs:
                st.info("No district-level posts for this day.")
            for b in briefs:
                with st.expander(f"{b.district} — {b.counts.get('rescue_needed', 0)} rescue request(s)", expanded=b.counts.get("rescue_needed", 0) > 0):
                    st.write(b.summary)
                    st.caption(", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in b.counts.items())
                               + (" · cached" if b.cached else ""))
                    if b.fallback_reason:
                        st.warning(f"LLM unavailable ({b.fallback_reason}); showing the offline summary.")
                    if b.dropped_citations:
                        st.warning(f"Removed citations to unknown posts: {', '.join(b.dropped_citations)}")
                    st.markdown("**Sources**")
                    for src in b.sources:
                        link = f"[{src.id}]({src.url})" if src.url else f"`{src.id}` (no link in dataset)"
                        st.markdown(f"- {link} · {src.label} — {src.text}")

with tab_match:
    m1, m2, m3, m4 = st.columns(4)
    alpha = m1.slider("Text weight", 0.0, 1.0, 0.6, 0.05, help="Rest of the score is proximity")
    max_km = m2.slider("Max distance (km)", 5, 200, 50, 5)
    min_score = m3.slider("Min score", 0.0, 1.0, 0.35, 0.05)
    one_to_one = m4.checkbox("One offer per need", value=True)
    matches = match_posts(so_far, _encoder(), alpha, float(max_km), min_score, one_to_one=one_to_one)
    st.metric("Matches", len(matches))
    if matches:
        st_folium(build_match_map(matches, so_far), height=480, use_container_width=True, returned_objects=[], key="match_map")
        st.dataframe(
            [{"score": round(m.score, 2), "similarity": round(m.similarity, 2), "km": round(m.distance_km, 1),
              "need": m.need_id, "need place": m.need_place, "need text": m.need_text,
              "offer": m.offer_id, "offer place": m.offer_place, "offer text": m.offer_text} for m in matches],
            width="stretch", hide_index=True)
    else:
        st.info("No need/offer pairs within the current limits.")


with tab_chat:
    st.caption("Ask about flood preparedness and safety. Answers cite NDMA guideline pages and the latest district briefs. "
               "In immediate danger call **112** first.")
    try:
        bot = _chatbot()
    except ValueError as exc:  # e.g. provider forced to anthropic without a key
        st.error(str(exc))
        st.stop()
    if not bot.kb.has_guidelines:
        st.warning("No guideline documents found. Put NDMA flood PDFs in `data/docs/` and reload (see README). "
                   "Until then answers can only draw on the daily briefs.")
    # Briefs for the latest day so far, from the free offline summarizer (the chatbot never triggers paid LLM calls for them).
    latest = max((p.timestamp.date() for p in so_far), default=None)
    bot.kb.set_briefs(brief_passages(daily_briefs(so_far, TemplateProvider(), day=latest)) if latest else [])
    st.caption(f"Provider: `{bot.provider.name}`" + ("" if bot.provider.uses_network else " — offline: shows the most relevant passages (add an API key to .env for written answers)"))

    history = st.session_state.setdefault("chat", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("citations"):
                with st.expander("Sources"):
                    for c in msg["citations"]:
                        where = f"{c['source']}, p. {c['page']}" if c["page"] else c["source"]
                        st.markdown(f"**[{c['n']}] {where}**" + (f" ({c['ref']})" if c.get("ref") else "") + f"  \n{c['excerpt']}")
    if question := st.chat_input("Ask a question (English, Hindi, ...)"):
        history.append({"role": "user", "text": question})
        ans = bot.ask(question)
        history.append({"role": "assistant", "text": ans.text, "citations": [c.model_dump() for c in ans.citations]})
        st.rerun()

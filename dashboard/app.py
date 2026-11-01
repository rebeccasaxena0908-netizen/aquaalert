"""AquaAlert dashboard. Run from the repo root: streamlit run dashboard/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from aquaalert.chatbot.rag import AUDIENCE_LABEL, brief_passages, default_chatbot  # noqa: E402
from aquaalert.classify.model import get_classifier  # noqa: E402
from aquaalert.config import get_settings  # noqa: E402
from aquaalert.extract.geocode import Geocoder  # noqa: E402
from aquaalert.fuse.sensors import load_sensors  # noqa: E402
from aquaalert.fuse.warnings import compute_warnings  # noqa: E402
from aquaalert.ingest.loader import load_many  # noqa: E402
from aquaalert.match.matcher import default_encoder, match_posts  # noqa: E402
from aquaalert.pipeline import process_posts  # noqa: E402
from aquaalert.summarize.brief import daily_briefs, latest_per_district  # noqa: E402
from aquaalert.summarize.providers import TemplateProvider, get_provider  # noqa: E402
from dashboard import theme as T  # noqa: E402
from dashboard.helpers import FLAG_THRESHOLD, LABEL_COLORS, build_map, build_match_map, posts_to_frame  # noqa: E402

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


@st.cache_data(show_spinner="Loading sensor data...")
def _sensors():
    return load_sensors()


@st.cache_data(show_spinner="Classifying and geocoding posts...")
def _processed(path: str):
    return process_posts(load_many(path), _classifier(), Geocoder())


def md_safe(text: str) -> str:
    """Plain text for st.markdown: no LaTeX from "$" amounts, and "<phone>" (the redaction marker) shown as text
    instead of being swallowed as an HTML tag."""
    return str(text).replace("$", "\\$").replace("<", "&lt;").replace(">", "&gt;")


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ---------- theme (injected once per run, before any content) ----------
st.sidebar.markdown('<span class="aa-label">Display</span>', unsafe_allow_html=True)
paused = st.sidebar.toggle("Pause background", value=False, help="Stop the drifting blobs and rising bubbles")
html(T.theme_css(paused))
html(T.sky_html())

settings = get_settings()
st.sidebar.markdown('<span class="aa-label">Data</span>', unsafe_allow_html=True)
path = st.sidebar.text_input("Dataset(s) (CSV/JSON/JSONL, comma-separated)", settings.dataset)
try:
    posts = _processed(path)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()
if not posts:
    st.warning("Dataset has no usable posts.")
    st.stop()

st.sidebar.markdown('<span class="aa-label">Filters</span>', unsafe_allow_html=True)
labels = st.sidebar.multiselect("Categories", list(LABEL_COLORS), default=[l for l in LABEL_COLORS if l != "irrelevant"])
min_conf = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
languages = sorted({p.language or "und" for p in posts})
lang_pick = st.sidebar.multiselect("Languages", languages, default=languages)
hide_doubtful = st.sidebar.checkbox("Hide low-credibility posts", value=False,
                                    help="Heuristic misinformation score, not a trained model (see README)")
max_misinfo = st.sidebar.slider("…with misinformation score at or above", 0.1, 1.0, FLAG_THRESHOLD, 0.05,
                                disabled=not hide_doubtful)


def credible(p) -> bool:
    return not hide_doubtful or (p.misinfo_score or 0) < max_misinfo


# Replay: reveal posts up to a simulated clock, like a live feed.
times = [p.timestamp for p in posts]
replay_at = st.sidebar.slider(
    "Replay time", min_value=min(times), max_value=max(times), value=max(times), format="MMM D, HH:mm",
)
st.sidebar.caption(f"Classifier: `{_classifier().name}`")

visible = [
    p for p in posts
    if p.timestamp <= replay_at and p.label and p.label.value in labels and (p.label_score or 0) >= min_conf
    and (p.language or "und") in lang_pick and credible(p)
]
analysed = [p for p in posts if p.timestamp <= replay_at]
# Briefs, matching, warnings and the chatbot use everything up to the replay clock (minus hidden low-credibility
# posts), regardless of the category and language filters.
so_far = [p for p in analysed if credible(p)]
readings = _sensors()
replay_day = replay_at.date() if hasattr(replay_at, "date") else replay_at
warnings = compute_warnings(so_far, readings, day=replay_day)
try:
    provider = get_provider()
    provider_error = None
except ValueError as exc:
    provider, provider_error = TemplateProvider(), str(exc)

# ---------- header + KPIs ----------
html(T.header_html([
    (f"Replay · {replay_at:%d %b %Y %H:%M}", True),
    (f"{len(languages)} languages", False),
    (_classifier().name.split(":")[-1], False),
    ("LLM: " + (provider.name if provider.uses_network else "offline"), False),
]))
df = posts_to_frame(visible)
mapped = int(df["lat"].notna().sum()) if len(df) else 0
rescue = [p for p in analysed if p.label and p.label.value == "rescue_needed"]
flagged = [p for p in analysed if (p.misinfo_score or 0) >= FLAG_THRESHOLD]
n_warn = sum(w.level == "warning" for w in warnings)
n_watch = sum(w.level == "watch" for w in warnings)
html(T.kpi_html([
    {"key": "posts", "label": "Posts analysed", "value": len(analysed), "tone": "accent",
     "note": f"{len(visible)} shown with current filters · {mapped} on the map"},
    {"key": "rescue", "label": "Rescue requests", "value": len(rescue), "tone": "err",
     "note": f"{sum(1 for p in rescue if p.rule)} promoted by the distress rules"},
    {"key": "warnings", "label": "Active warnings", "value": n_warn, "tone": "err" if n_warn else "",
     "note": f"+{n_watch} on watch · {replay_day:%d %b}"},
    {"key": "flagged", "label": "Low-credibility posts", "value": len(flagged), "tone": "warn" if flagged else "",
     "note": ("hidden by the sidebar filter" if hide_doubtful else "shown dashed on the map")
             + " · heuristic score"},
]))

tab_map, tab_warn, tab_brief, tab_match, tab_chat = st.tabs(
    ["Map", "Warnings", "Daily brief", "Needs ↔ Offers", "Chatbot"])

# ---------- map ----------
with tab_map:
    map_col, legend_col = st.columns([4, 1])
    with map_col:
        show_risk = st.toggle("District risk layer", value=True, help="Early-warning level per district for the replay day")
        st_folium(build_map(visible, warnings if show_risk else None), height=560, use_container_width=True,
                  returned_objects=[])
    with legend_col:
        counts = df["label"].value_counts().to_dict() if len(df) else {}
        html(T.legend_html(LABEL_COLORS, counts))
        if show_risk:
            html('<div class="aa-glass aa-legend" style="margin-top:12px"><span class="aa-label">District risk</span>'
                 + "".join(f'<div class="aa-legend-row">{T.level_badge(l)}</div>' for l in ("warning", "watch", "normal"))
                 + "</div>")
    html(T.section("Posts", f"{len(df)} posts · newest first · click a column to sort"))
    if len(df):
        cols = ["time", "label", "confidence", "language", "location", "misinfo", "rule", "people", "phones", "text",
                "misinfo_reasons", "model_label", "id"]
        st.dataframe(
            # blank cells instead of "None" for the optional columns
            # blank cells instead of "None"; people as text so a blank and a number can share the column
            df.sort_values("time", ascending=False)[cols]
            .assign(people=lambda d: d["people"].map(lambda v: "" if pd.isna(v) else str(int(v))))
            .fillna({"rule": "", "model_label": "", "location": "", "misinfo_reasons": ""}),
            width="stretch", hide_index=True,
            column_config={
                "time": st.column_config.DatetimeColumn("time", format="D MMM, HH:mm"),
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                "misinfo": st.column_config.ProgressColumn("misinfo", min_value=0.0, max_value=1.0, format="%.2f",
                                                           help="Heuristic misinformation score (0 = no red flags)"),
                "text": st.column_config.TextColumn("text", width="large"),
                "misinfo_reasons": st.column_config.TextColumn("misinfo reasons", width="medium"),
            },
        )

# ---------- warnings ----------
with tab_warn:
    html(T.section(f"Early warnings · {replay_day:%d %b %Y}",
                   "WARNING = rescue/damage posts spike and rain ≥ 115.6 mm or a river above danger level; "
                   "WATCH = only one holds. Sample sensor files are synthetic (see README)."))
    if not readings:
        st.warning("No sensor CSVs found in data/sensors/ — warnings use post volume only.")
    html(T.kpi_html([
        {"key": "warnings_tab", "label": "Warnings", "value": n_warn, "tone": "err" if n_warn else "",
         "note": "spike + sensor threshold"},
        {"key": "watch_tab", "label": "Watch", "value": n_watch, "tone": "warn" if n_watch else "",
         "note": "one signal only, or river ≥ 90 % of danger"},
    ]))
    raised = [w for w in warnings if w.level != "normal"]
    for w in raised:
        reasons = "".join(f"<li>{T.esc(r)}</li>" for r in w.reasons)
        ids = " ".join(T.tag(i) for i in w.post_ids[:12])
        html(T.card(f'{T.level_badge(w.level)}<span class="aa-panel-title">{T.esc(w.district)}</span>',
                    f'<ul class="aa-reasons">{reasons}</ul>' + (f'<div class="aa-sources">{ids}</div>' if ids else "")))
    if not raised:
        st.success("No district warnings for this day.")
    html(T.table_html(
        ["district", "level", "posts", "baseline/day", "rain mm", "river", "level m", "danger m"],
        [[w.district, T.Html(T.level_badge(w.level)), w.post_count, w.baseline, w.rainfall_mm, w.river,
          w.river_level_m, w.danger_level_m] for w in warnings],
        numeric={"posts", "baseline/day", "rain mm", "level m", "danger m"}))
    districts_with_data = sorted({d for d, _ in readings})
    if districts_with_data:
        pick = st.selectbox("Sensor history", districts_with_data,
                            index=districts_with_data.index(raised[0].district)
                            if raised and raised[0].district in districts_with_data else 0)
        hist = sorted((r for (d, _), r in readings.items() if d == pick), key=lambda r: r.date)
        chart = {"rainfall mm": [r.rainfall_mm for r in hist]}
        if any(r.river_level_m is not None for r in hist):
            chart["river % of danger"] = [round(100 * r.river_ratio, 1) if r.river_ratio else None for r in hist]
        st.line_chart(pd.DataFrame(chart, index=[r.date for r in hist]), color=["#35659E", "#8FB9E0"][: len(chart)])

# ---------- daily brief ----------
with tab_brief:
    if provider_error:
        st.error(provider_error)
    days = sorted({p.timestamp.date() for p in so_far})
    html(T.section("Daily brief", f"provider {provider.name} / {provider.model}"
                   + ("" if provider.uses_network else " · offline summaries — add GEMINI_API_KEY, ANTHROPIC_API_KEY "
                      "or OPENAI_API_KEY to .env for LLM summaries")))
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
                n_rescue = b.counts.get("rescue_needed", 0)
                with st.expander(f"{b.district} — {n_rescue} rescue request(s)", expanded=n_rescue > 0):
                    st.markdown(md_safe(b.summary))
                    chips = " ".join(T.tag(f"{k.replace('_', ' ')}: {v}") for k, v in b.counts.items())
                    if b.cached:
                        chips += " " + T.badge("cached", "neu")
                    if b.excluded_doubtful:
                        chips += " " + T.badge(f"{b.excluded_doubtful} doubtful post(s) left out", "warn")
                    html(f'<div class="aa-card-head">{chips}</div>')
                    if b.fallback_reason:
                        st.warning(f"LLM unavailable ({b.fallback_reason}); showing the offline summary.")
                    if b.dropped_citations:
                        st.warning(f"Removed citations to unknown posts: {', '.join(b.dropped_citations)}")
                    sources = "".join(
                        f'<div class="aa-source">{T.tag(s.id)}{T.label_badge(s.label)} '
                        + (f'<a href="{T.esc(s.url)}" target="_blank">link</a> ' if s.url else "")
                        + f"{T.esc(s.text)}</div>" for s in b.sources)
                    html(f'<span class="aa-label">Sources</span><div class="aa-sources">{sources}</div>')

# ---------- needs <-> offers ----------
with tab_match:
    m1, m2, m3, m4 = st.columns(4)
    alpha = m1.slider("Text weight", 0.0, 1.0, 0.6, 0.05, help="Rest of the score is proximity")
    max_km = m2.slider("Max distance (km)", 5, 200, 50, 5)
    min_score = m3.slider("Min score", 0.0, 1.0, 0.35, 0.05)
    one_to_one = m4.toggle("One offer per need", value=True)
    matches = match_posts(so_far, _encoder(), alpha, float(max_km), min_score, one_to_one=one_to_one)
    html(T.section("Needs ↔ Offers", f"{len(matches)} matches · score = {alpha:.2f} × text similarity "
                   f"+ {1 - alpha:.2f} × proximity (0 at {max_km} km)"))
    if matches:
        st_folium(build_match_map(matches, so_far), height=480, use_container_width=True, returned_objects=[],
                  key="match_map")
        html(T.table_html(
            ["score", "km", "need", "need text", "offer", "offer text"],
            [[T.Html(T.meter(m.score)), round(m.distance_km, 1),
              T.Html(f"{T.tag(m.need_id)} {T.label_badge(m.need_label)}<br>{T.esc(m.need_place or '')}"),
              m.need_text, T.Html(f"{T.tag(m.offer_id)}<br>{T.esc(m.offer_place or '')}"), m.offer_text]
             for m in matches],
            numeric={"km"}))
    else:
        st.info("No need/offer pairs within the current limits.")

# ---------- chatbot ----------
with tab_chat:
    html(T.section("Preparedness chatbot", "answers cite guideline pages and the latest district briefs · "
                   "in immediate danger call 112 first"))
    try:
        bot = _chatbot()
    except ValueError as exc:  # e.g. provider forced to anthropic without a key
        st.error(str(exc))
        st.stop()
    if not bot.kb.has_guidelines:
        st.warning("No guideline documents found. Run `python scripts/fetch_docs.py` (see README). "
                   "Until then answers can only draw on the daily briefs.")
    # each district's most recent brief, from the free offline summarizer (no paid LLM calls for chatbot context)
    bot.kb.set_briefs(brief_passages(latest_per_district(daily_briefs(so_far, TemplateProvider()))))
    st.caption(f"Provider: `{bot.provider.name}`" + ("" if bot.provider.uses_network
                                                     else " — offline: shows the most relevant passages"))

    history = st.session_state.setdefault("chat", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(md_safe(msg["text"]))
            if msg.get("citations"):
                with st.expander("Sources"):
                    for c in msg["citations"]:
                        name = c.get("title") or c["source"]
                        where = f"{name}, p. {c['page']}" if c["page"] else name
                        where_html = (f'<a href="{T.esc(c["url"])}" target="_blank">{T.esc(where)}</a>'
                                      if c.get("url") else T.esc(where))
                        kind = "neu"  # audience is not a status: keep the status colours for status
                        num = T.tag(f"[{c['n']}]")
                        html(f'<div class="aa-cite"><div class="aa-cite-head">{num}{where_html}'
                             f'{T.badge(AUDIENCE_LABEL.get(c.get("audience"), "source"), kind)}'
                             + (f'<span class="aa-muted">{T.esc(c["ref"])}</span>' if c.get("ref") else "")
                             + f'</div><div>{T.esc(c["excerpt"])}</div></div>')
    if question := st.chat_input("Ask a question (English, Hindi, Malayalam, ...)"):
        history.append({"role": "user", "text": question})
        ans = bot.ask(question)
        history.append({"role": "assistant", "text": ans.text, "citations": [c.model_dump() for c in ans.citations]})
        st.rerun()

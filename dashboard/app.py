"""AquaAlert dashboard. Run from the repo root: streamlit run dashboard/app.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from aquaalert.chatbot.rag import AUDIENCE_LABEL, brief_passages, default_chatbot  # noqa: E402
from aquaalert.classify.model import get_classifier  # noqa: E402
from aquaalert.config import INDIA_BBOX, get_settings  # noqa: E402
from aquaalert.extract.geocode import Geocoder  # noqa: E402
from aquaalert.fuse.sensors import load_sensors  # noqa: E402
from aquaalert.fuse.warnings import compute_warnings  # noqa: E402
from aquaalert.ingest.loader import load_many  # noqa: E402
from aquaalert.match.matcher import default_encoder, match_posts  # noqa: E402
from aquaalert.pipeline import process_posts  # noqa: E402
from aquaalert.schemas import Label  # noqa: E402
from aquaalert.summarize.brief import daily_briefs, latest_per_district  # noqa: E402
from aquaalert.summarize.providers import TemplateProvider, get_provider  # noqa: E402
from dashboard import theme as T  # noqa: E402
from dashboard.helpers import (FLAG_THRESHOLD, IRRELEVANT_HIDE_SCORE, LABEL_COLORS, SEVERITY_COLORS,  # noqa: E402
                               alert_icon_svg, build_map, build_match_map, hidden_as_irrelevant, place_alerts,
                               posts_to_frame, state_banner_svg)

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


# ---------- live store + background poller (Live mode only) ----------
def _live_db_path() -> str:
    from aquaalert.ingest.live import DB_PATH

    return str(os.getenv("AQUAALERT_LIVE_DB") or DB_PATH)


@st.cache_resource
def _live_store(path: str):
    """Cached per database path, so a different AQUAALERT_LIVE_DB never reuses another store."""
    from aquaalert.ingest.live import LiveStore

    return LiveStore(path)


@st.cache_resource
def _poller_thread(path: str):
    """One background poller per server process and database. AQUAALERT_LIVE_POLL=off disables it (tests, offline demos)."""
    if os.getenv("AQUAALERT_LIVE_POLL", "on").lower() in ("off", "0", "false"):
        return None
    import threading

    from aquaalert.ingest.live import Poller

    stop = threading.Event()
    t = threading.Thread(target=Poller(_live_store(path)).run_forever, args=(stop,), daemon=True, name="aquaalert-live")
    t.start()
    return stop


@st.cache_data(show_spinner="Classifying and geocoding live posts...", max_entries=8)
def _processed_live(key: tuple, _posts: list):
    return process_posts(_posts, _classifier(), Geocoder())


MODES = {
    "sample": ("REPLAY · Kerala Aug 2018 sample", ("replay", "synthetic")),
    "live": ("LIVE · NDMA SACHET, Bluesky", ("live",)),
    "custom": ("REPLAY · your dataset file(s)", ("replay",)),
}
st.sidebar.markdown('<span class="aa-label">Dataset</span>', unsafe_allow_html=True)
default_mode = os.getenv("AQUAALERT_MODE", "sample")
mode = st.sidebar.radio("Dataset", list(MODES), index=list(MODES).index(default_mode) if default_mode in MODES else 0,
                        format_func=lambda m: MODES[m][0], label_visibility="collapsed")
html("".join(T.dataset_chip(k) + " " for k in MODES[mode][1]))
official_alerts: list = []
live_status: dict = {}
live_counts: dict = {}
sim_active = 0
live_updated = None
window_h = 24
if mode == "live":
    from datetime import datetime, timedelta, timezone

    from aquaalert.ingest import simulate
    from aquaalert.ingest.live import SOURCES, to_posts

    store = _live_store(_live_db_path())
    _poller_thread(_live_db_path())
    window_h = st.sidebar.select_slider("Live window (hours)", [6, 12, 24, 48], value=24)
    news_on = st.sidebar.toggle("News (GDELT): experimental", value=store.news_enabled(), key="news_enabled",
                                help="Off by default: GDELT has not been verified live (it rate-limited every test). "
                                     "The background poller picks this up within about 30 s.")
    if news_on != store.news_enabled():
        store.set_news_enabled(news_on)
    now_utc = datetime.now(timezone.utc)
    items = [i for i in store.items(since=now_utc - timedelta(hours=window_h)) if news_on or i.source != "news"]
    official_alerts = [i for i in items if i.kind == "official_alert"]
    live_counts = {s: sum(1 for i in items if i.source == s) for s in SOURCES}  # real sources only
    sim_active = sum(1 for i in items if i.source == simulate.SIM_SOURCE)
    live_status, live_updated = store.status(), store.last_updated()
    raw_posts = to_posts(items)
    posts = _processed_live(tuple((p.id, p.text) for p in raw_posts), raw_posts) if raw_posts else []
    replay_at = now_utc.replace(tzinfo=None)  # live: everything up to now (UTC)
else:
    path = settings.dataset if mode == "sample" else st.sidebar.text_input(
        "Dataset file(s) (CSV/JSON/JSONL, comma-separated)", settings.dataset)
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
languages = sorted({p.language or "und" for p in posts}) or ["en"]
lang_pick = st.sidebar.multiselect("Languages", languages, default=languages)
hide_doubtful = st.sidebar.checkbox("Hide low-credibility posts", value=False,
                                    help="Heuristic misinformation score, not a trained model (see README)")
max_misinfo = st.sidebar.slider("…with misinformation score at or above", 0.1, 1.0, FLAG_THRESHOLD, 0.05,
                                disabled=not hide_doubtful)


def credible(p) -> bool:
    return not hide_doubtful or (p.misinfo_score or 0) < max_misinfo


if mode != "live":
    # Replay: reveal posts up to a simulated clock, like a live feed.
    times = [p.timestamp for p in posts]
    replay_at = st.sidebar.slider(
        "Replay time", min_value=min(times), max_value=max(times), value=max(times), format="MMM D, HH:mm",
    )
st.sidebar.caption(f"Classifier: `{_classifier().name}`")



def category_shown(p) -> bool:
    """The category filter, except that Live mode keeps irrelevant posts the classifier is unsure about on screen."""
    if not p.label:
        return False
    unsure = mode == "live" and p.label == Label.IRRELEVANT and not hidden_as_irrelevant(p)
    return p.label.value in labels or unsure


visible = [
    p for p in posts
    if p.timestamp <= replay_at and category_shown(p) and (p.label_score or 0) >= min_conf
    and (p.language or "und") in lang_pick and credible(p)
]
analysed = [p for p in posts if p.timestamp <= replay_at]
hidden_irrelevant = [p for p in analysed if hidden_as_irrelevant(p)] if mode == "live" else []
# Briefs, matching, warnings and the chatbot use everything up to the replay clock (minus hidden low-credibility
# posts), regardless of the category and language filters. Live mode also leaves out confidently irrelevant posts.
so_far = [p for p in analysed if credible(p) and p not in hidden_irrelevant]
readings = _sensors()
replay_day = replay_at.date() if hasattr(replay_at, "date") else replay_at
warnings = compute_warnings(so_far, readings, day=replay_day)
try:
    provider = get_provider()
    provider_error = None
except ValueError as exc:
    provider, provider_error = TemplateProvider(), str(exc)

# ---------- header + KPIs ----------
if mode != "live":
    clock_chip = f"Replay · {replay_at:%d %b %Y %H:%M}"
elif live_updated:
    clock_chip = f"Live · updated {live_updated:%d %b %H:%M} UTC"
elif any(v.get("state") == "failed" for v in live_status.values()):
    clock_chip = "Live · last poll failed"
else:
    clock_chip = "Live · waiting for first poll"
html(T.header_html([
    (clock_chip, True),
    (f"{len(languages)} languages", False),
    (_classifier().name.split(":")[-1], False),
    ("LLM: " + (provider.name if provider.uses_network else "offline"), False),
]))

if mode == "live":
    # per-source status + counts (simulated posts are never counted here)
    cards = []
    for src, label in (("sachet", "NDMA SACHET alerts"), ("bluesky", "Bluesky posts"), ("news", "News (GDELT) · experimental")):
        stt = live_status.get(src, {})
        state = stt.get("state") or "waiting"
        if src == "news" and not news_on:  # right away, not only after the next poll
            state, stt = "off", {"note": "experimental; enable it in the sidebar"}
        kind = {"ok": "ok", "slow": "warn", "failed": "err", "skipped": "neu"}.get(state, "neu")
        when = f"last success {stt['last_success'][11:16]} UTC" if stt.get("last_success") else "no successful poll yet"
        took = f" · took {stt['duration']:.0f}s" if stt.get("duration") is not None else ""
        detail = stt.get("note") or stt.get("last_error") or ""
        cards.append(f'<div class="aa-glass aa-source-card" data-source="{src}" data-count="{live_counts.get(src, 0)}" '
                     f'data-state="{T.esc(state)}"><span class="aa-label">{T.esc(label)}</span>'
                     f'<span class="aa-big">{live_counts.get(src, 0)}</span><div>{T.badge(state, kind)}</div>'
                     f'<span class="aa-kpi-note">{T.esc(when + took)}</span>'
                     + (f'<span class="aa-kpi-note aa-muted">{T.esc(detail[:160])}</span>' if detail else "")
                     + "</div>")
    html(f'<div class="aa-sources-row">{"".join(cards)}</div>')
    real_flood_posts = [p for p in posts if p.source != "simulated"]
    if not real_flood_posts and not official_alerts:
        html('<div class="aa-glass aa-empty"><div class="aa-panel-title">No flood-related items in the last '
             f'{window_h}h</div>Nothing from SACHET or Bluesky passed the flood filter in this window. '
             'This view is never padded with sample data; switch the dataset to Replay to explore the 2018 sample.</div>')

    @st.fragment(run_every="30s")
    def _auto_refresh():
        """Rerun the page when the poller stored something new (checked every 30 s)."""
        snapshot = (len(store.items(since=datetime.now(timezone.utc) - timedelta(hours=window_h))),
                    str(store.last_updated()))
        if st.session_state.get("live_snapshot") not in (None, snapshot):
            st.session_state["live_snapshot"] = snapshot
            st.rerun(scope="app")
        st.session_state["live_snapshot"] = snapshot

    _auto_refresh()
    with st.sidebar.expander("Simulate incident", expanded=sim_active > 0):
        st.caption(f"Injects {len(simulate.SIM_POSTS)} clearly synthetic posts (Hinglish rescue request, Malayalam SOS, "
                   "offer of boats, missing person, dam-breach rumour, …) into the live pipeline. They carry a "
                   "SIMULATED badge and are never counted as source items.")
        if st.button("Simulate incident", key="sim_inject"):
            simulate.inject(store)
            st.rerun()
        if st.button(f"Clear simulated posts ({sim_active})", key="sim_clear", disabled=sim_active == 0):
            simulate.clear(store)
            st.rerun()

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
     "note": f"{sum(1 for p in rescue if p.rule and p.model_label != p.label)} promoted by the distress rules"
             + (f" · {n_conf} confirmed" if (n_conf := sum(1 for p in rescue if p.rule and p.model_label == p.label)) else "")},
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
        map_view = {"bbox": INDIA_BBOX, "zoom": 5} if mode == "live" else {}
        placed_alerts, unplaced_alerts = place_alerts(official_alerts)
        st_folium(build_map(visible, warnings if show_risk else None, alerts=placed_alerts, **map_view), height=560,
                  use_container_width=True, returned_objects=[])
        if official_alerts:
            why = "; ".join(f"{n} {reason}" for reason, n in unplaced_alerts.items())
            at_state = sum(1 for a in placed_alerts if a.state)
            st.caption(f"{len(placed_alerts)} of {len(official_alerts)} official alerts on the map"
                       + (f" ({at_state} at state level, affected districts not specified)" if at_state else "")
                       + (f". Not drawn: {why}." if why else "."))
    with legend_col:
        counts = df["label"].value_counts().to_dict() if len(df) else {}
        n_state = sum(1 for a in placed_alerts if a.state)
        official_legend = (len(placed_alerts) - n_state, alert_icon_svg(SEVERITY_COLORS["severe"], 14)) if mode == "live" else None
        state_legend = (n_state, state_banner_svg(SEVERITY_COLORS["severe"], 16)) if mode == "live" else None
        html(T.legend_html(LABEL_COLORS, counts, official_legend, state_legend))
        if show_risk:
            html('<div class="aa-glass aa-legend" style="margin-top:12px"><span class="aa-label">District risk</span>'
                 + "".join(f'<div class="aa-legend-row">{T.level_badge(l)}</div>' for l in ("warning", "watch", "normal"))
                 + "</div>")
    if hidden_irrelevant and "irrelevant" not in labels:
        st.caption(f"{len(hidden_irrelevant)} live posts hidden: the classifier is at least "
                   f"{IRRELEVANT_HIDE_SCORE:.0%} sure they are irrelevant. Add 'irrelevant' under Categories to see them.")
    html(T.section("Posts", f"{len(df)} posts · newest first · click a column to sort"))
    if len(df):
        cols = ["time", "source", "label", "confidence", "language", "location", "misinfo", "rule", "people", "phones",
                "text", "link", "misinfo_reasons", "model_label", "id"]
        st.dataframe(
            # blank cells instead of "None" for the optional columns
            # blank cells instead of "None"; people as text so a blank and a number can share the column
            df.sort_values("time", ascending=False)[cols]
            .assign(people=lambda d: d["people"].map(lambda v: "" if pd.isna(v) else str(int(v))))
            .fillna({"rule": "", "model_label": "", "location": "", "misinfo_reasons": "", "source": "", "link": ""}),
            width="stretch", hide_index=True,
            column_config={
                "time": st.column_config.DatetimeColumn("time", format="D MMM, HH:mm"),
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                "misinfo": st.column_config.ProgressColumn("misinfo", min_value=0.0, max_value=1.0, format="%.2f",
                                                           help="Heuristic misinformation score (0 = no red flags)"),
                "text": st.column_config.TextColumn("text", width="large"),
                "misinfo_reasons": st.column_config.TextColumn("misinfo reasons", width="medium"),
                "source": st.column_config.TextColumn("source", help="SIMULATED = injected by 'Simulate incident'"),
                "link": st.column_config.LinkColumn("link", display_text="open ↗"),
            },
        )

# ---------- warnings ----------
with tab_warn:
    html(T.section(f"Early warnings · {replay_day:%d %b %Y}",
                   "WARNING = rescue/damage posts spike and rain ≥ 115.6 mm or a river above danger level; "
                   "WATCH = only one holds. Sample sensor files are synthetic (see README)."))
    html(T.section("Official alerts", "NDMA SACHET CAP alerts (IMD, CWC, state SDMAs) — shown as issued, separate "
                   "from AquaAlert's own post-volume warnings below"))
    if mode != "live":
        st.caption("Official alerts are only fetched in Live mode (SACHET has no alerts for the 2018 replay).")
    elif not official_alerts:
        st.caption(f"No flood, heavy-rain or river alerts from SACHET in the last {window_h}h.")
    for a in official_alerts:
        e = a.extra
        meta = " · ".join(x for x in (e.get("sender"), e.get("area") and f"Area: {e['area']}",
                                      e.get("effective") and f"{e['effective'][:16].replace('T', ' ')} → "
                                      f"{(e.get('expires') or '')[:16].replace('T', ' ')}") if x)
        body = (f'<div class="aa-alert-meta">{T.esc(meta)}</div><p>{T.esc(e.get("headline") or a.text)}</p>'
                + (f'<p><b>Instruction:</b> {T.esc(e["instruction"])}</p>' if e.get("instruction") else "")
                + f'<a class="aa-label" href="{T.esc(a.url)}" target="_blank" rel="noopener noreferrer">CAP source ↗</a>')
        cancelled = (e.get("msg_type") or "").lower() == "cancel"
        html(T.card(f'{T.official_badge()}{T.severity_badge(e.get("severity", ""))}'
                    f'{T.badge("Cancelled by the issuer", "neu") if cancelled else ""}'
                    f'<span class="aa-panel-title">{T.esc(e.get("event") or "Alert")}</span>', body))
    html(T.section("AquaAlert warnings", "post-volume spike × sensor thresholds"))
    if not readings:
        st.warning("No sensor CSVs found in data/sensors/ — warnings use post volume only.")
    elif mode == "live" and not any(d == replay_day for _, d in readings):
        st.caption("The sensor files in data/sensors/ do not cover today, so live warnings use post volume only "
                   "(a spike alone is a WATCH).")
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
                        f'<div class="aa-source">{T.tag(s.id)}{T.source_badge(s.source) if s.source in T.SOURCE_LABELS else ""}'
                        f'{T.label_badge(s.label)} '
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
        st_folium(build_match_map(matches, so_far, **map_view), height=480, use_container_width=True, returned_objects=[],
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

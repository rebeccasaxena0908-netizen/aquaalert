"""Pure helpers for the dashboard (kept separate from Streamlit so they can be tested)."""
from __future__ import annotations

import hashlib
import html

import folium
import pandas as pd

from aquaalert.config import KERALA_BBOX
from aquaalert.extract.gazetteer import ALL_PLACES
from aquaalert.schemas import Label, Post

FLAG_THRESHOLD = 0.5  # misinfo score at which a marker is drawn as "doubtful"
IRRELEVANT_HIDE_SCORE = 0.8  # Live mode: hide posts labelled irrelevant at or above this confidence

# Map category colours: distinct on the light map and the sky background. Only rescue requests use the status red;
# amber is reserved for WATCH / low credibility, green (ok) is reused for offers of help.
LABEL_COLORS: dict[str, str] = {
    "rescue_needed": "#C4484F",
    "infrastructure_damage": "#8C5A2B",
    "need_supplies": "#13838F",
    "offer_help": "#2F8F6E",
    "missing_person": "#7A4FA0",
    "info_update": "#35659E",
    "irrelevant": "#8A99AA",
}

# Themed Leaflet popups/tooltips inside the map iframe (the page CSS can't reach into it)
MAP_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Karla:wght@400;700&display=swap" rel="stylesheet">
<style>
.leaflet-popup-content-wrapper { background: rgba(252,253,255,.97); color: #212936; border-radius: 14px;
  border: 1px solid #B7CCE6; box-shadow: 0 10px 28px rgba(45,85,135,.22); }
.leaflet-popup-tip { background: rgba(252,253,255,.97); border: 1px solid #B7CCE6; }
.leaflet-popup-content { font-family: 'Karla', system-ui, sans-serif; font-size: 13.5px; line-height: 1.5; margin: 12px 14px; }
.leaflet-popup-content b, .aa-pop-title { font-family: 'Montserrat', system-ui, sans-serif; }
.aa-pop-title { font-weight: 700; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #35659E; }
.aa-pop-meta { color: #667789; font-size: 12px; }
.aa-pop-flag { color: #8D611E; font-weight: 700; }
.leaflet-popup-content ul { margin: 4px 0 0; padding-left: 18px; }
.aa-badge { display: inline-block; border-radius: 999px; font-size: 11.5px; font-weight: 700; padding: 2px 8px;
  border: 1px solid; line-height: 1.5; margin: 4px 4px 0 0; }
.aa-err { color: #B04147; background: #F8ECEE; border-color: rgba(196,72,79,.3); }
.aa-warn { color: #8D611E; background: #FAF3E8; border-color: rgba(201,138,43,.3); }
.aa-ok { color: #27755A; background: #EAF4F0; border-color: rgba(47,143,110,.3); }
.aa-neu { color: #3D4A59; background: #E7F0FA; border-color: #B7CCE6; }
.aa-sim { color: #4A3B8C; background: #fff; border: 1.5px dashed #6A58B8; letter-spacing: .08em; }
.aa-pop-link { font-family: 'Montserrat', system-ui, sans-serif; font-size: 11px; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; color: #35659E; }
.leaflet-tooltip { font-family: 'Karla', system-ui, sans-serif; font-size: 12.5px; color: #212936;
  background: rgba(252,253,255,.97); border: 1px solid #B7CCE6; border-radius: 8px; box-shadow: 0 4px 12px rgba(45,85,135,.15); }
</style>
"""


def _themed(fmap: folium.Map) -> folium.Map:
    fmap.get_root().header.add_child(folium.Element(MAP_CSS))
    return fmap


def posts_to_frame(posts: list[Post]) -> pd.DataFrame:
    rows = []
    for p in posts:
        ex = p.extraction
        rows.append(
            {
                "id": p.id,
                "time": p.timestamp,
                "label": p.label.value if p.label else None,
                "confidence": round(p.label_score, 2) if p.label_score is not None else None,
                "language": p.language,
                "misinfo": p.misinfo_score,
                "misinfo_reasons": "; ".join(p.misinfo_reasons),
                "rule": p.rule,
                "model_label": p.model_label.value if p.model_label else None,
                "source": "SIMULATED" if p.source == "simulated" else p.source,
                "link": p.url,
                "location": ex.geo_source if ex else None,
                "lat": ex.geo.lat if ex and ex.geo else None,
                "lon": ex.geo.lon if ex and ex.geo else None,
                "people": ex.people_count if ex else None,
                "phones": ", ".join(ex.phones) if ex else "",
                "text": p.text,
            }
        )
    return pd.DataFrame(rows)


def jitter(post_id: str, spread: float = 0.012) -> tuple[float, float]:
    """Deterministic small offset so posts from the same town don't stack on one pixel."""
    digest = hashlib.md5(post_id.encode()).digest()
    return ((digest[0] / 255 - 0.5) * spread, (digest[1] / 255 - 0.5) * spread)


RISK_COLORS = {"warning": "#C4484F", "watch": "#C98A2B", "normal": "#2F8F6E"}


def add_risk_layer(fmap: folium.Map, warnings: list) -> folium.Map:
    """District early-warning circles at district centres (no district polygons are bundled)."""
    layer = folium.FeatureGroup(name="District risk")
    for w in warnings:
        center = ALL_PLACES.get(w.district.lower())
        if center is None:
            continue
        tip = f"{w.district}: {w.level.upper()}" + ("<br>" + "<br>".join(html.escape(r) for r in w.reasons) if w.reasons else "")
        folium.Circle(center, radius=9000 + 2500 * min(w.post_count, 10), color=RISK_COLORS[w.level],
                      fill=True, fill_opacity=0.18 if w.level != "normal" else 0.06, weight=2,
                      tooltip=folium.Tooltip(tip)).add_to(layer)
    layer.add_to(fmap)
    return fmap


def hidden_as_irrelevant(post: Post) -> bool:
    """Live mode hides posts the classifier is sure are irrelevant; unsure ones stay on screen for a person to judge."""
    return post.label == Label.IRRELEVANT and (post.label_score or 0) >= IRRELEVANT_HIDE_SCORE


def build_map(posts: list[Post], warnings: list | None = None, bbox: tuple[float, float, float, float] = KERALA_BBOX,
              zoom: int = 7) -> folium.Map:
    south, west, north, east = bbox
    fmap = _themed(folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=zoom, tiles="OpenStreetMap"))
    if warnings:
        add_risk_layer(fmap, warnings)
    for p in posts:
        geo = p.extraction.geo if p.extraction else None
        if geo is None or p.label is None:
            continue
        dlat, dlon = jitter(p.id)
        # post text is untrusted: always escaped before it goes into the popup HTML
        from dashboard.theme import cred_badge, label_badge, rule_badge, source_badge  # local: no import cycle

        popup = (f'<div class="aa-pop-title">{html.escape(p.label.value.replace("_", " "))}</div>'
                 f'<div class="aa-pop-meta">{p.label_score:.0%} confidence · {html.escape(p.language or "und")} · '
                 f'{html.escape(p.id)}</div>{html.escape(p.text)}'
                 f'<div>{source_badge(p.source) if p.source in ("sachet", "news", "bluesky", "simulated") else ""}'
                 f'{label_badge(p.label.value)}{rule_badge(p.rule, confirms=p.model_label == p.label)}'
                 f'{cred_badge(p.misinfo_score, FLAG_THRESHOLD)}</div>')
        if p.url:
            popup += (f'<div><a class="aa-pop-link" href="{html.escape(p.url, quote=True)}" target="_blank" '
                      f'rel="noopener noreferrer">open original ↗</a></div>')
        if p.rule:
            if p.model_label and p.model_label != p.label:
                popup += (f'<div class="aa-pop-meta">the classifier said {html.escape(p.model_label.value)}; '
                          f'a distress rule overrode it</div>')
            else:
                popup += '<div class="aa-pop-meta">a distress rule agrees with the classifier</div>'
        flagged = (p.misinfo_score or 0) >= FLAG_THRESHOLD
        if p.misinfo_reasons:
            popup += (f'<span class="aa-pop-flag">why the misinformation score is {p.misinfo_score:.2f}:</span><ul>'
                      + "".join(f"<li>{html.escape(r)}</li>" for r in p.misinfo_reasons) + "</ul>")
        if p.extraction.phones:
            popup += f"<br>Phone: {html.escape(', '.join(p.extraction.phones))}"
        folium.CircleMarker(
            [geo.lat + dlat, geo.lon + dlon],
            radius=8,
            color=LABEL_COLORS[p.label.value],
            fill=True,
            fill_opacity=0.25 if flagged else 0.8,
            dash_array="4" if flagged else None,  # flagged posts: hollow, dashed outline
            popup=folium.Popup(popup, max_width=320),
            tooltip=html.escape(f"{p.label.value.replace('_', ' ')} · {p.extraction.geo_source}"),
        ).add_to(fmap)
    return fmap


def build_match_map(matches: list, posts: list[Post], bbox: tuple[float, float, float, float] = KERALA_BBOX,
                    zoom: int = 7) -> folium.Map:
    """Map with a line from each need (red) to its matched offer (green)."""
    south, west, north, east = bbox
    fmap = _themed(folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=zoom))
    geo = {p.id: p.extraction.geo for p in posts if p.extraction and p.extraction.geo}
    for m in matches:
        n, o = geo.get(m.need_id), geo.get(m.offer_id)
        if n is None or o is None:
            continue
        nd, od = jitter(m.need_id), jitter(m.offer_id)
        npt, opt = [n.lat + nd[0], n.lon + nd[1]], [o.lat + od[0], o.lon + od[1]]
        tip = f"{m.score:.2f} | {m.distance_km:.1f} km"
        folium.PolyLine([npt, opt], color="#35659E", weight=2, dash_array="5", tooltip=tip).add_to(fmap)
        folium.CircleMarker(npt, radius=7, color=LABEL_COLORS["rescue_needed"], fill=True, fill_opacity=0.85,
                            tooltip=html.escape(f"NEED {m.need_id}: {m.need_text[:80]}")).add_to(fmap)
        folium.CircleMarker(opt, radius=7, color=LABEL_COLORS["offer_help"], fill=True, fill_opacity=0.85,
                            tooltip=html.escape(f"OFFER {m.offer_id}: {m.offer_text[:80]}")).add_to(fmap)
    return fmap

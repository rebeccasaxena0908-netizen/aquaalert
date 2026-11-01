"""Pure helpers for the dashboard (kept separate from Streamlit so they can be tested)."""
from __future__ import annotations

import hashlib
import html

import folium
import pandas as pd

from aquaalert.config import KERALA_BBOX
from aquaalert.extract.gazetteer import KERALA_PLACES
from aquaalert.schemas import Post

FLAG_THRESHOLD = 0.5  # misinfo score at which a marker is drawn as "doubtful"

LABEL_COLORS: dict[str, str] = {
    "rescue_needed": "red",
    "infrastructure_damage": "darkred",
    "need_supplies": "orange",
    "offer_help": "green",
    "missing_person": "purple",
    "info_update": "blue",
    "irrelevant": "gray",
}


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


RISK_COLORS = {"warning": "#d7191c", "watch": "#fdae61", "normal": "#1a9641"}


def add_risk_layer(fmap: folium.Map, warnings: list) -> folium.Map:
    """District early-warning circles at district centres (no district polygons are bundled)."""
    layer = folium.FeatureGroup(name="District risk")
    for w in warnings:
        center = KERALA_PLACES.get(w.district.lower())
        if center is None:
            continue
        tip = f"{w.district}: {w.level.upper()}" + ("<br>" + "<br>".join(html.escape(r) for r in w.reasons) if w.reasons else "")
        folium.Circle(center, radius=9000 + 2500 * min(w.post_count, 10), color=RISK_COLORS[w.level],
                      fill=True, fill_opacity=0.18 if w.level != "normal" else 0.06, weight=2,
                      tooltip=folium.Tooltip(tip)).add_to(layer)
    layer.add_to(fmap)
    return fmap


def build_map(posts: list[Post], warnings: list | None = None) -> folium.Map:
    south, west, north, east = KERALA_BBOX
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=7, tiles="OpenStreetMap")
    if warnings:
        add_risk_layer(fmap, warnings)
    for p in posts:
        geo = p.extraction.geo if p.extraction else None
        if geo is None or p.label is None:
            continue
        dlat, dlon = jitter(p.id)
        popup = f"<b>{p.label.value}</b> ({p.label_score:.0%})<br>{p.text}"
        if p.rule:
            popup += f"<br><i>rule: {p.rule} (model said {p.model_label.value})</i>"
        flagged = (p.misinfo_score or 0) >= FLAG_THRESHOLD
        if p.misinfo_reasons:
            popup += (f"<br><b>misinformation score {p.misinfo_score:.2f}</b><ul>"
                      + "".join(f"<li>{html.escape(r)}</li>" for r in p.misinfo_reasons) + "</ul>")
        if p.extraction.phones:
            popup += f"<br>Phone: {', '.join(p.extraction.phones)}"
        folium.CircleMarker(
            [geo.lat + dlat, geo.lon + dlon],
            radius=8,
            color=LABEL_COLORS[p.label.value],
            fill=True,
            fill_opacity=0.25 if flagged else 0.8,
            dash_array="4" if flagged else None,  # flagged posts: hollow, dashed outline
            popup=folium.Popup(popup, max_width=320),
            tooltip=f"{p.label.value} - {p.extraction.geo_source}",
        ).add_to(fmap)
    return fmap


def build_match_map(matches: list, posts: list[Post]) -> folium.Map:
    """Map with a line from each need (red) to its matched offer (green)."""
    south, west, north, east = KERALA_BBOX
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=7)
    geo = {p.id: p.extraction.geo for p in posts if p.extraction and p.extraction.geo}
    for m in matches:
        n, o = geo.get(m.need_id), geo.get(m.offer_id)
        if n is None or o is None:
            continue
        nd, od = jitter(m.need_id), jitter(m.offer_id)
        npt, opt = [n.lat + nd[0], n.lon + nd[1]], [o.lat + od[0], o.lon + od[1]]
        tip = f"{m.score:.2f} | {m.distance_km:.1f} km"
        folium.PolyLine([npt, opt], color="#555", weight=2, dash_array="5", tooltip=tip).add_to(fmap)
        folium.CircleMarker(npt, radius=7, color="red", fill=True, tooltip=f"NEED {m.need_id}: {m.need_text[:80]}").add_to(fmap)
        folium.CircleMarker(opt, radius=7, color="green", fill=True, tooltip=f"OFFER {m.offer_id}: {m.offer_text[:80]}").add_to(fmap)
    return fmap

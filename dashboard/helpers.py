"""Pure helpers for the dashboard (kept separate from Streamlit so they can be tested)."""
from __future__ import annotations

import hashlib

import folium
import pandas as pd

from aquaalert.config import KERALA_BBOX
from aquaalert.schemas import Post

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


def build_map(posts: list[Post]) -> folium.Map:
    south, west, north, east = KERALA_BBOX
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=7, tiles="OpenStreetMap")
    for p in posts:
        geo = p.extraction.geo if p.extraction else None
        if geo is None or p.label is None:
            continue
        dlat, dlon = jitter(p.id)
        popup = f"<b>{p.label.value}</b> ({p.label_score:.0%})<br>{p.text}"
        if p.extraction.phones:
            popup += f"<br>Phone: {', '.join(p.extraction.phones)}"
        folium.CircleMarker(
            [geo.lat + dlat, geo.lon + dlon],
            radius=8,
            color=LABEL_COLORS[p.label.value],
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=320),
            tooltip=f"{p.label.value} - {p.extraction.geo_source}",
        ).add_to(fmap)
    return fmap

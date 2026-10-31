"""Cached geocoding: offline gazetteer first, optional Nominatim biased to Kerala."""
from __future__ import annotations

import json
from pathlib import Path

from aquaalert.config import KERALA_BBOX, get_settings
from aquaalert.extract.gazetteer import KERALA_PLACES
from aquaalert.schemas import GeoPoint


def in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float] = KERALA_BBOX) -> bool:
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


class Geocoder:
    """Resolve a place name to a GeoPoint. Results (including misses) are cached as JSON."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        online: bool | None = None,
        user_agent: str | None = None,
        bbox: tuple[float, float, float, float] = KERALA_BBOX,
    ) -> None:
        settings = get_settings()
        self.cache_path = Path(cache_path) if cache_path else settings.cache_dir / "geocode.json"
        self.online = settings.geocode_online if online is None else online
        self.user_agent = user_agent or settings.geocoder_user_agent
        self.bbox = bbox
        self._nominatim = None
        self._cache: dict[str, list[float] | None] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    def geocode(self, place: str) -> GeoPoint | None:
        key = place.strip().lower()
        if not key:
            return None
        if key in self._cache:
            return self._to_point(self._cache[key])
        coords = KERALA_PLACES.get(key) or (self._lookup_online(place) if self.online else None)
        if coords is None and self.online:
            self._cache[key] = None  # only cache misses once the network was actually consulted
            self._save()
        elif coords is not None:
            self._cache[key] = list(coords)
            self._save()
        return self._to_point(coords)

    def _lookup_online(self, place: str) -> tuple[float, float] | None:
        from geopy.geocoders import Nominatim  # lazy

        if self._nominatim is None:
            self._nominatim = Nominatim(user_agent=self.user_agent, timeout=10)
        south, west, north, east = self.bbox
        try:
            hit = self._nominatim.geocode(
                f"{place}, Kerala, India",
                viewbox=[(north, west), (south, east)],
                bounded=True,
                country_codes="in",
            )
        except Exception:  # network/rate-limit errors: treat as a miss, don't cache
            return None
        return (hit.latitude, hit.longitude) if hit else None

    def _to_point(self, coords: list[float] | tuple[float, float] | None) -> GeoPoint | None:
        if coords is None or not in_bbox(coords[0], coords[1], self.bbox):
            return None
        return GeoPoint(lat=coords[0], lon=coords[1])

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=1, sort_keys=True))

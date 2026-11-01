"""Load district-level rainfall / river-level CSVs from data/sensors/.

Any CSV there with columns ``district, date`` plus ``rainfall_mm`` and/or ``river_level_m`` (+ ``danger_level_m``)
is read; extra columns (e.g. ``river``) are kept for display. Several gauges for one district-day are combined by
taking the highest rainfall and the river gauge closest to (or furthest above) its danger level.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from aquaalert.config import ROOT

SENSOR_DIR = ROOT / "data" / "sensors"


class SensorReading(BaseModel):
    district: str
    date: date
    rainfall_mm: float | None = None
    river_level_m: float | None = None
    danger_level_m: float | None = None
    river: str | None = None

    @property
    def river_ratio(self) -> float | None:
        if self.river_level_m is None or not self.danger_level_m:
            return None
        return self.river_level_m / self.danger_level_m


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"district", "date"} <= set(df.columns) or not ({"rainfall_mm", "river_level_m"} & set(df.columns)):
        raise ValueError(f"{path.name}: needs district, date and rainfall_mm and/or river_level_m")
    if "river_level_m" in df.columns and "danger_level_m" not in df.columns:
        raise ValueError(f"{path.name}: river_level_m needs a danger_level_m column")
    df["district"] = df["district"].astype(str).str.strip().str.title()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df.dropna(subset=["date"])


def load_sensors(sensor_dir: str | Path = SENSOR_DIR) -> dict[tuple[str, date], SensorReading]:
    """{(district, date): SensorReading} merged across all CSV files in ``sensor_dir``."""
    frames = [_read(p) for p in sorted(Path(sensor_dir).glob("*.csv"))]
    readings: dict[tuple[str, date], SensorReading] = {}
    for df in frames:
        for row in df.to_dict("records"):
            key = (row["district"], row["date"])
            r = readings.setdefault(key, SensorReading(district=key[0], date=key[1]))
            rain = row.get("rainfall_mm")
            if rain is not None and not pd.isna(rain):
                r.rainfall_mm = max(float(rain), r.rainfall_mm or 0.0)
            level, danger = row.get("river_level_m"), row.get("danger_level_m")
            if level is not None and not pd.isna(level) and danger and not pd.isna(danger):
                if r.river_ratio is None or float(level) / float(danger) > r.river_ratio:
                    r.river_level_m, r.danger_level_m = float(level), float(danger)
                    river = row.get("river")
                    r.river = None if river is None or pd.isna(river) else str(river)
    return readings

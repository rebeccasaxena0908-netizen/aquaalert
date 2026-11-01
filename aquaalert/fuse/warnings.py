"""District early-warning flags: fuse post-volume spikes with rainfall / river levels.

    python -m aquaalert.fuse.warnings [--day 2018-08-15]

A district-day is a **WARNING** when both hold:
  * posts spike: rescue_needed + infrastructure_damage posts (excluding likely misinformation, score >= 0.5) reach
    MIN_POSTS and at least SPIKE_RATIO x (mean of the previous BASELINE_DAYS days + 1), and
  * a sensor threshold is crossed: rainfall >= RAIN_MM (IMD "very heavy", 115.6 mm/day) or river at/above danger level.
It is a **WATCH** when only one of the two holds (or the river is within RIVER_WATCH of its danger level).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta

from pydantic import BaseModel, Field

from aquaalert.fuse.sensors import SensorReading, load_sensors
from aquaalert.schemas import Label, Post

SIGNAL_LABELS = {Label.RESCUE_NEEDED, Label.INFRASTRUCTURE_DAMAGE}
MAX_MISINFO = 0.5
MIN_POSTS = 3
SPIKE_RATIO = 2.0
BASELINE_DAYS = 3
RAIN_MM = 115.6  # IMD: very heavy rainfall is 115.6-204.4 mm/day; extremely heavy >= 204.5
RIVER_WATCH = 0.9  # share of the danger level that already counts as "approaching"
LEVELS = {"warning": 2, "watch": 1, "normal": 0}


class DistrictWarning(BaseModel):
    district: str
    date: date
    level: str  # "warning" | "watch" | "normal"
    post_count: int
    baseline: float
    spike: bool
    rainfall_mm: float | None = None
    river: str | None = None
    river_level_m: float | None = None
    danger_level_m: float | None = None
    sensor_trigger: bool
    reasons: list[str] = Field(default_factory=list)
    post_ids: list[str] = Field(default_factory=list)


def signal_counts(posts: list[Post]) -> tuple[Counter, dict[tuple[str, date], list[str]]]:
    counts: Counter = Counter()
    ids: dict[tuple[str, date], list[str]] = defaultdict(list)
    for p in posts:
        district = p.extraction.district if p.extraction else None
        if district and p.label in SIGNAL_LABELS and (p.misinfo_score or 0) < MAX_MISINFO:
            key = (district, p.timestamp.date())
            counts[key] += 1
            ids[key].append(p.id)
    return counts, ids


def assess(district: str, day: date, counts: Counter, ids: dict, reading: SensorReading | None) -> DistrictWarning:
    n = counts.get((district, day), 0)
    baseline = sum(counts.get((district, day - timedelta(days=i)), 0) for i in range(1, BASELINE_DAYS + 1)) / BASELINE_DAYS
    spike = n >= MIN_POSTS and n >= SPIKE_RATIO * (baseline + 1)
    reasons: list[str] = []
    if spike:
        reasons.append(f"{n} rescue/damage posts vs {baseline:.1f}/day over the previous {BASELINE_DAYS} days")
    elif n:
        reasons.append(f"{n} rescue/damage post(s): below the spike threshold")
    rain = reading.rainfall_mm if reading else None
    ratio = reading.river_ratio if reading else None
    rain_hit = rain is not None and rain >= RAIN_MM
    river_hit = ratio is not None and ratio >= 1.0
    if rain_hit:
        kind = "extremely heavy" if rain >= 204.5 else "very heavy"
        reasons.append(f"rainfall {rain:.0f} mm ({kind}; threshold {RAIN_MM} mm)")
    if river_hit:
        reasons.append(f"{reading.river or 'river'} at {reading.river_level_m:.2f} m, above danger level {reading.danger_level_m:.2f} m")
    elif ratio is not None and ratio >= RIVER_WATCH:
        reasons.append(f"{reading.river or 'river'} at {ratio:.0%} of its danger level")
    sensor = rain_hit or river_hit
    near = ratio is not None and ratio >= RIVER_WATCH
    level = "warning" if spike and sensor else "watch" if spike or sensor or near else "normal"
    if reading is None:
        reasons.append("no sensor data for this district-day")
    return DistrictWarning(
        district=district, date=day, level=level, post_count=n, baseline=round(baseline, 2), spike=spike,
        rainfall_mm=rain, river=reading.river if reading else None,
        river_level_m=reading.river_level_m if reading else None,
        danger_level_m=reading.danger_level_m if reading else None,
        sensor_trigger=sensor, reasons=reasons, post_ids=ids.get((district, day), []),
    )


def compute_warnings(posts: list[Post], readings: dict | None = None, day: date | None = None) -> list[DistrictWarning]:
    """Assess every district with posts or sensor data on ``day`` (default: latest post day), most severe first."""
    readings = load_sensors() if readings is None else readings
    counts, ids = signal_counts(posts)
    if day is None:
        day = max((p.timestamp.date() for p in posts), default=None) or max((d for _, d in readings), default=None)
    if day is None:
        return []
    districts = {d for d, dd in counts if dd == day} | {d for d, dd in readings if dd == day}
    out = [assess(d, day, counts, ids, readings.get((d, day))) for d in sorted(districts)]
    return sorted(out, key=lambda w: (-LEVELS[w.level], -w.post_count, w.district))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", type=date.fromisoformat, default=None)
    ap.add_argument("--data", default=None, help="dataset(s); default: AQUAALERT_DATASET")
    a = ap.parse_args()
    from aquaalert.config import get_settings
    from aquaalert.pipeline import run

    for w in compute_warnings(run(a.data or get_settings().dataset), day=a.day):
        if w.level != "normal":
            print(f"{w.level.upper():8s} {w.district:18s} {w.date}  " + " | ".join(w.reasons))


if __name__ == "__main__":
    main()

"""Generate the synthetic Kerala August 2018 sensor files in data/sensors/ (seeded, reproducible).

    python scripts/make_sensor_samples.py

The shape follows the 2018 event (a first heavy spell around 8-10 Aug, the peak on 14-17 Aug, highest in Idukki,
Wayanad and Pathanamthitta), but every number is made up. Replace these files with real IMD / CWC data for analysis.
"""
from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "sensors"
DAYS = [date(2018, 8, 1) + timedelta(days=i) for i in range(31)]

# relative rainfall intensity per district (higher = wetter during the event)
INTENSITY = {
    "Idukki": 1.5, "Wayanad": 1.35, "Pathanamthitta": 1.3, "Ernakulam": 1.15, "Thrissur": 1.1, "Palakkad": 1.05,
    "Kottayam": 1.0, "Alappuzha": 0.9, "Malappuram": 1.0, "Kozhikode": 0.95, "Kannur": 0.85, "Kollam": 0.75,
    "Kasaragod": 0.7, "Thiruvananthapuram": 0.55,
}
# one river gauge per district: (river, danger level m, normal level m)
RIVERS = {
    "Pathanamthitta": ("Pamba", 9.0, 4.5), "Alappuzha": ("Pamba", 2.5, 0.9), "Ernakulam": ("Periyar", 10.2, 5.2),
    "Thrissur": ("Chalakudy", 7.1, 3.0), "Idukki": ("Periyar", 12.0, 6.5), "Kottayam": ("Meenachil", 6.0, 2.6),
    "Palakkad": ("Bharathapuzha", 8.5, 3.8), "Wayanad": ("Kabini", 7.4, 3.1), "Malappuram": ("Chaliyar", 9.4, 4.0),
    "Kozhikode": ("Chaliyar", 8.0, 3.5),
}


def storm(d: date) -> float:
    """0-1 storm profile: a spell around 9 Aug and the main peak around 15-16 Aug."""
    t = d.day
    return min(1.0, 0.55 * math.exp(-((t - 9) ** 2) / 3) + math.exp(-((t - 15.5) ** 2) / 4.5))


def main() -> None:
    rng = random.Random(2018)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rainfall_kerala_aug2018.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["district", "date", "rainfall_mm"])
        for district, k in INTENSITY.items():
            for d in DAYS:
                mm = max(0.0, k * (8 + 190 * storm(d)) * rng.uniform(0.7, 1.3) - rng.uniform(0, 6))
                w.writerow([district, d.isoformat(), round(mm, 1)])
    with open(OUT / "river_levels_kerala_aug2018.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["district", "date", "river", "river_level_m", "danger_level_m"])
        for district, (river, danger, normal) in RIVERS.items():
            k = INTENSITY[district]
            level = normal
            for d in DAYS:
                # the river responds to rain with a lag and recedes slowly
                target = normal + (danger - normal) * 1.25 * k * storm(d - timedelta(days=1))
                level += 0.55 * (target - level) + rng.uniform(-0.05, 0.05)
                w.writerow([district, d.isoformat(), river, round(level, 2), danger])
    print("wrote", OUT / "rainfall_kerala_aug2018.csv", "and", OUT / "river_levels_kerala_aug2018.csv")


if __name__ == "__main__":
    main()

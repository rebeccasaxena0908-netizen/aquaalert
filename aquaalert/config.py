"""Environment-driven configuration (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Kerala bounding box: (south, west, north, east)
KERALA_BBOX = (8.17, 74.85, 12.80, 77.42)


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    classifier: str
    model_dir: Path
    base_model: str
    geocoder_user_agent: str
    geocode_online: bool
    dataset: Path
    cache_dir: Path


def get_settings() -> Settings:
    return Settings(
        classifier=os.getenv("AQUAALERT_CLASSIFIER", "auto"),
        model_dir=_path(os.getenv("AQUAALERT_MODEL_DIR", "models/muril")),
        base_model=os.getenv("AQUAALERT_BASE_MODEL", "google/muril-base-cased"),
        geocoder_user_agent=os.getenv("AQUAALERT_GEOCODER_USER_AGENT", "aquaalert-dev"),
        geocode_online=os.getenv("AQUAALERT_GEOCODE_ONLINE", "false").lower() == "true",
        dataset=_path(os.getenv("AQUAALERT_DATASET", "data/sample_posts.csv")),
        cache_dir=_path("data/cache"),
    )

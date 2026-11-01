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


DEFAULT_DATASET = "data/sample_posts.csv,data/sample_posts_multilingual.csv,data/sample_posts_rumours.csv"


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def dataset_paths(value: str) -> list[Path]:
    return [_path(v.strip()) for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    classifier: str
    model_dir: Path
    embed_dir: Path
    base_model: str
    geocoder_user_agent: str
    geocode_online: bool
    dataset: str  # one path or several, comma-separated (relative to the repo root)
    cache_dir: Path
    rules: bool
    llm_provider: str
    llm_model: str | None


def get_settings() -> Settings:
    return Settings(
        classifier=os.getenv("AQUAALERT_CLASSIFIER", "auto"),
        model_dir=_path(os.getenv("AQUAALERT_MODEL_DIR", "models/muril")),
        embed_dir=_path(os.getenv("AQUAALERT_EMBED_DIR", "models/embed_lr")),
        base_model=os.getenv("AQUAALERT_BASE_MODEL", "google/muril-base-cased"),
        geocoder_user_agent=os.getenv("AQUAALERT_GEOCODER_USER_AGENT", "aquaalert-dev"),
        geocode_online=os.getenv("AQUAALERT_GEOCODE_ONLINE", "false").lower() == "true",
        dataset=os.getenv("AQUAALERT_DATASET") or DEFAULT_DATASET,
        cache_dir=_path("data/cache"),
        rules=(os.getenv("AQUAALERT_RULES") or "on").lower() not in ("off", "false", "0"),
        llm_provider=os.getenv("AQUAALERT_LLM_PROVIDER") or "auto",
        llm_model=os.getenv("AQUAALERT_LLM_MODEL") or None,
    )

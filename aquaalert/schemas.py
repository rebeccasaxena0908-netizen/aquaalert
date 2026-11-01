"""Shared Post schema passed between every pipeline stage."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Label(str, Enum):
    RESCUE_NEEDED = "rescue_needed"
    INFRASTRUCTURE_DAMAGE = "infrastructure_damage"
    NEED_SUPPLIES = "need_supplies"
    OFFER_HELP = "offer_help"
    MISSING_PERSON = "missing_person"
    INFO_UPDATE = "info_update"
    IRRELEVANT = "irrelevant"


LABELS: list[str] = [label.value for label in Label]


class GeoPoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Extraction(BaseModel):
    """Output of the extract/ stage."""

    locations: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    people_count: int | None = None
    geo: GeoPoint | None = None
    geo_source: str | None = None  # the location string that was geocoded
    district: str | None = None  # Kerala district of geo_source (gazetteer lookup)


class Post(BaseModel):
    id: str
    text: str
    timestamp: datetime
    source: str = "unknown"  # twitter, news, sample, ...
    author: str | None = None
    url: str | None = None  # link to the original post/article, when the dataset has one
    language: str | None = None  # language/: en, hi, ml, bn, as, hi-Latn, ml-Latn, bn-Latn, ...
    code_mixed: bool | None = None  # language/: Indian language mixed with English
    gold_language: str | None = None  # language tag from the dataset, when present (evaluation only)

    # Ground-truth label when the dataset provides one (used for eval, not by stages)
    gold_label: Label | None = None

    # classify/
    label: Label | None = None
    label_score: float | None = Field(default=None, ge=0, le=1)
    label_scores: dict[str, float] | None = None
    model_label: Label | None = None  # the classifier's own label when a rule overrode it
    rule: str | None = None  # name of the post-classification rule that fired, if any

    # credibility/: heuristic misinformation score (0 = no red flags) and the reasons behind it
    misinfo_score: float | None = Field(default=None, ge=0, le=1)
    misinfo_reasons: list[str] = Field(default_factory=list)
    credibility: float | None = Field(default=None, ge=0, le=1)  # 1 - misinfo_score

    # extract/
    extraction: Extraction | None = None

"""AquaAlert REST API: the pipeline's outputs over HTTP.

    uvicorn api.main:app --reload        # docs at http://127.0.0.1:8000/docs (ReDoc at /redoc)

Posts are loaded and processed once (dataset from AQUAALERT_DATASET) and kept in memory. No authentication:
run it locally or behind your own gateway. Endpoints that can call a paid LLM (briefs with llm=true, chat) only do
so when an API key is configured.
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from aquaalert.chatbot.rag import Answer, Chatbot, brief_passages, default_chatbot
from aquaalert.classify.base import Classifier, classify_posts
from aquaalert.classify.model import get_classifier
from aquaalert.config import get_settings
from aquaalert.credibility.heuristics import combine, text_flags
from aquaalert.extract.entities import extract_post
from aquaalert.extract.geocode import Geocoder
from aquaalert.fuse.sensors import load_sensors
from aquaalert.fuse.warnings import DistrictWarning, compute_warnings
from aquaalert.ingest.loader import load_many
from aquaalert.language.detect import detect_language
from aquaalert.match.matcher import Match, default_encoder, match_posts
from aquaalert.pipeline import process_posts
from aquaalert.schemas import LABELS, Extraction, Label, Post
from aquaalert.summarize.brief import DistrictBrief, daily_briefs, latest_per_district
from aquaalert.summarize.providers import TemplateProvider, get_provider

app = FastAPI(
    title="AquaAlert API",
    version="0.4.0",
    description="Multilingual flood-crisis intelligence: classified posts, needs-offers matches, district briefs, "
                "early warnings and a preparedness chatbot. Misinformation scores are heuristic, not a trained model.",
)


class State:
    """Everything the endpoints share; built once, replaceable in tests via dependency overrides."""

    def __init__(self, posts: list[Post], classifier: Classifier, geocoder: Geocoder, readings: dict,
                 chatbot_factory=default_chatbot, encoder_factory=default_encoder) -> None:
        self.posts, self.classifier, self.geocoder, self.readings = posts, classifier, geocoder, readings
        self._chatbot_factory, self._encoder_factory = chatbot_factory, encoder_factory
        self._chatbot: Chatbot | None = None
        self._encoder = None

    @property
    def chatbot(self) -> Chatbot:
        if self._chatbot is None:
            self._chatbot = self._chatbot_factory()
        return self._chatbot

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = self._encoder_factory()
        return self._encoder


@lru_cache(maxsize=1)
def _default_state() -> State:
    clf, geo = get_classifier(), Geocoder()
    posts = process_posts(load_many(get_settings().dataset), clf, geo)
    return State(posts, clf, geo, load_sensors())


def get_state() -> State:
    return _default_state()


# ---------- models ----------
class Health(BaseModel):
    status: str
    classifier: str
    posts: int
    llm_provider: str
    sensor_readings: int


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000, examples=["Hum chhat par phase hue hain, please help! Aluva"])


class ClassifyResponse(BaseModel):
    label: Label
    label_score: float
    label_scores: dict[str, float]
    rule: str | None
    model_label: Label | None
    language: str
    code_mixed: bool
    extraction: Extraction
    misinfo_score: float
    misinfo_reasons: list[str]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What should I do if water enters my house?"])


# ---------- endpoints ----------
@app.get("/health", response_model=Health, tags=["meta"])
def health(state: State = Depends(get_state)) -> Health:
    try:
        provider = get_provider().name
    except ValueError as exc:
        provider = f"misconfigured: {exc}"
    return Health(status="ok", classifier=state.classifier.name, posts=len(state.posts), llm_provider=provider,
                  sensor_readings=len(state.readings))


@app.get("/posts", response_model=list[Post], tags=["posts"])
def list_posts(
    state: State = Depends(get_state),
    label: list[Label] | None = Query(None, description="repeat for several labels"),
    language: list[str] | None = Query(None, description="e.g. en, ml, hi-Latn"),
    district: str | None = None,
    min_confidence: float = Query(0.0, ge=0, le=1),
    max_misinfo: float = Query(1.0, ge=0, le=1, description="hide posts at or above this misinformation score"),
    since: datetime | None = None,
    until: datetime | None = None,
    located: bool | None = Query(None, description="only posts with (true) / without (false) coordinates"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[Post]:
    out = []
    for p in state.posts:
        geo = p.extraction.geo if p.extraction else None
        if label and p.label not in label:
            continue
        if language and (p.language or "und") not in language:
            continue
        if district and (not p.extraction or (p.extraction.district or "").lower() != district.lower()):
            continue
        if (p.label_score or 0) < min_confidence or (max_misinfo < 1 and (p.misinfo_score or 0) >= max_misinfo):
            continue
        if (since and p.timestamp < since) or (until and p.timestamp > until):
            continue
        if located is not None and (geo is not None) != located:
            continue
        out.append(p)
    return out[offset: offset + limit]


@app.post("/classify", response_model=ClassifyResponse, tags=["posts"])
def classify(req: ClassifyRequest, state: State = Depends(get_state)) -> ClassifyResponse:
    lang = detect_language(req.text)
    raw = Post(id="request", text=req.text, timestamp=datetime.now(), language=lang.lang, code_mixed=lang.code_mixed)
    post = extract_post(classify_posts([raw], state.classifier, rules=get_settings().rules)[0], state.geocoder)
    flags = text_flags(post)  # text-only signals; nearby/duplicate checks need the whole feed
    return ClassifyResponse(
        label=post.label, label_score=post.label_score, label_scores=post.label_scores, rule=post.rule,
        model_label=post.model_label, language=lang.lang, code_mixed=lang.code_mixed, extraction=post.extraction,
        misinfo_score=round(combine([f.weight for f in flags]), 2), misinfo_reasons=[f.reason for f in flags],
    )


@app.get("/matches", response_model=list[Match], tags=["response"])
def matches(
    state: State = Depends(get_state),
    alpha: float = Query(0.6, ge=0, le=1, description="weight of text similarity vs proximity"),
    max_km: float = Query(50.0, gt=0, le=500),
    min_score: float = Query(0.35, ge=0, le=1),
    one_to_one: bool = True,
    max_misinfo: float = Query(0.5, ge=0, le=1),
) -> list[Match]:
    posts = [p for p in state.posts if (p.misinfo_score or 0) < max_misinfo]
    return match_posts(posts, state.encoder, alpha, max_km, min_score, one_to_one=one_to_one)


@app.get("/briefs", response_model=list[DistrictBrief], tags=["response"])
def briefs(
    state: State = Depends(get_state),
    day: date | None = Query(None, description="default: every district's latest brief"),
    district: str | None = None,
    llm: bool = Query(False, description="use the configured LLM provider (costs money; cached) instead of the offline template"),
) -> list[DistrictBrief]:
    try:
        provider = get_provider() if llm else TemplateProvider()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    out = daily_briefs(state.posts, provider, day=day)
    if day is None:
        out = latest_per_district(out)
    if district:
        out = [b for b in out if b.district.lower() == district.lower()]
    return out


@app.get("/warnings", response_model=list[DistrictWarning], tags=["response"])
def warnings(
    state: State = Depends(get_state),
    day: date | None = Query(None, description="default: the latest day with posts"),
    level: str | None = Query(None, pattern="^(warning|watch|normal)$"),
) -> list[DistrictWarning]:
    out = compute_warnings(state.posts, state.readings, day=day)
    return [w for w in out if level is None or w.level == level]


@app.post("/chat/ask", response_model=Answer, tags=["chatbot"])
def ask(req: AskRequest, state: State = Depends(get_state)) -> Answer:
    bot = state.chatbot
    bot.kb.set_briefs(brief_passages(latest_per_district(daily_briefs(state.posts, TemplateProvider()))))
    return bot.ask(req.question)


@app.get("/labels", response_model=list[str], tags=["meta"])
def labels() -> list[str]:
    return LABELS

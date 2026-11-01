"""District-wise daily brief: group classified posts, summarize with a provider, cite sources.

    python -m aquaalert.summarize.brief data/sample_posts.csv
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

from aquaalert.config import get_settings
from aquaalert.extract.entities import PHONE_RE
from aquaalert.schemas import Label, Post
from aquaalert.summarize.providers import BriefInput, BriefPost, Provider, build_user_prompt, get_provider

MAX_POSTS = 20  # posts shown to the model per district-day
MAX_TEXT = 280
# most urgent first
PRIORITY = [Label.RESCUE_NEEDED, Label.MISSING_PERSON, Label.INFRASTRUCTURE_DAMAGE, Label.NEED_SUPPLIES,
            Label.OFFER_HELP, Label.INFO_UPDATE]
_RANK = {l: i for i, l in enumerate(PRIORITY)}
_CITE = re.compile(r"\[([^\[\]\s]+)\]")


class SourceRef(BaseModel):
    id: str
    label: str
    text: str
    url: str | None = None  # None when the dataset has no link (e.g. the synthetic sample)


class DistrictBrief(BaseModel):
    district: str
    date: date
    summary: str
    counts: dict[str, int]
    sources: list[SourceRef] = Field(default_factory=list)
    provider: str
    model: str
    cached: bool = False
    dropped_citations: list[str] = Field(default_factory=list)  # ids the model cited that were not in the input


def _select(posts: list[Post]) -> list[Post]:
    usable = [p for p in posts if p.label and p.label != Label.IRRELEVANT]
    usable.sort(key=lambda p: (_RANK[p.label], -(p.label_score or 0), p.timestamp))
    return usable


def build_inputs(posts: list[Post]) -> dict[tuple[str, date], tuple[BriefInput, list[Post]]]:
    """Group posts by (district, day). Posts with no district are skipped."""
    groups: dict[tuple[str, date], list[Post]] = defaultdict(list)
    for p in posts:
        if p.extraction and p.extraction.district:
            groups[(p.extraction.district, p.timestamp.date())].append(p)
    out = {}
    for (district, day), members in sorted(groups.items()):
        ranked = _select(members)
        if not ranked:
            continue
        counts = Counter(p.label.value for p in ranked)
        shown = ranked[:MAX_POSTS]
        # phone numbers are personal data: keep them out of text sent to a third-party LLM
        brief_posts = [BriefPost(p.id, p.label.value, PHONE_RE.sub("<phone>", p.text)[:MAX_TEXT], p.extraction.geo_source,
                                 p.extraction.people_count, p.url) for p in shown]
        out[(district, day)] = (BriefInput(district, day.isoformat(), dict(counts), brief_posts), shown)
    return out


def _cache_key(provider: Provider, brief: BriefInput) -> str:
    return hashlib.sha256(f"{provider.name}|{provider.model}|{build_user_prompt(brief)}".encode()).hexdigest()


class _Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, value: str) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))


def _verify_citations(text: str, known: set[str]) -> tuple[str, list[str]]:
    """Strip citations of ids that were not in the input (guards against invented sources)."""
    dropped: list[str] = []

    def repl(m: re.Match) -> str:
        if m.group(1) in known:
            return m.group(0)
        dropped.append(m.group(1))
        return ""

    return re.sub(r"\s{2,}", " ", _CITE.sub(repl, text)).strip(), dropped


def summarize_district(brief: BriefInput, shown: list[Post], provider: Provider, cache: _Cache | None) -> DistrictBrief:
    cached = False
    text = None
    key = _cache_key(provider, brief) if provider.uses_network and cache else None
    if key:
        text = cache.get(key)
        cached = text is not None
    if text is None:
        text = provider.summarize(brief)
        if key:
            cache.put(key, text)
    by_id = {p.id: p for p in shown}
    text, dropped = _verify_citations(text, set(by_id))
    cited = list(dict.fromkeys(i for i in _CITE.findall(text) if i in by_id))
    source_posts = [by_id[i] for i in cited] or shown[:3]  # uncited summary: fall back to the top posts
    return DistrictBrief(
        district=brief.district, date=date.fromisoformat(brief.date), summary=text, counts=brief.counts,
        sources=[SourceRef(id=p.id, label=p.label.value, text=p.text, url=p.url) for p in source_posts],
        provider=provider.name, model=provider.model, cached=cached, dropped_citations=dropped,
    )


def daily_briefs(posts: list[Post], provider: Provider | None = None, cache_path: Path | None = None,
                 day: date | None = None) -> list[DistrictBrief]:
    """Briefs for every district-day (or one day), most rescue requests first."""
    provider = provider or get_provider()
    cache = _Cache(cache_path or get_settings().cache_dir / "summaries.json") if provider.uses_network else None
    briefs = [
        summarize_district(inp, shown, provider, cache)
        for (district, d), (inp, shown) in build_inputs(posts).items()
        if day is None or d == day
    ]
    return sorted(briefs, key=lambda b: (-b.counts.get("rescue_needed", 0), b.district))


def main() -> None:
    import sys

    from aquaalert.pipeline import run

    for b in daily_briefs(run(sys.argv[1] if len(sys.argv) > 1 else "data/sample_posts.csv")):
        print(f"\n## {b.district} — {b.date}  ({b.provider}/{b.model})\n{b.summary}")
        for s in b.sources:
            print(f"   [{s.id}] {s.url or '(no link in dataset)'}")


if __name__ == "__main__":
    main()

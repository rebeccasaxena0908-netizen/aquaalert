"""Rule-based extraction of locations, phone numbers and people counts."""
from __future__ import annotations

import re
import unicodedata

from aquaalert.extract.gazetteer import DISTRICT_OF, KERALA_PLACES, NATIVE_ALIASES
from aquaalert.extract.geocode import Geocoder
from aquaalert.schemas import Extraction, Post

PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?([6-9]\d{9})(?!\d)")
_PLACE_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, KERALA_PLACES), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
# "in Foo Bar", "at Foo", "near Foo" for places the gazetteer doesn't know (geocoded online)
_PREP_RE = re.compile(r"\b(?:in|at|near|from|around)\s+((?:[A-Z][a-z]+)(?:\s[A-Z][a-z]+)?)")
_STOPWORDS = {"The", "This", "That", "Kerala", "India", "Please", "Aug", "Tuesday", "Monday"}

_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20}
_COUNT_RE = re.compile(
    r"\b(\d{1,4}|" + "|".join(_NUMBER_WORDS) + r")\s+(?:\w+\s)?(people|persons?|kids|children|adults|"
    r"families|family members|women|men|log|jon manush|manush|elderly)\b",
    re.IGNORECASE,
)
_FAMILY_OF_RE = re.compile(r"\bfamily of (\d{1,3})\b", re.IGNORECASE)


def extract_phones(text: str) -> list[str]:
    seen: list[str] = []
    for m in PHONE_RE.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def _native_hits(text: str) -> list[tuple[int, str]]:
    """(position, canonical place) for native-script names; prefix match so case endings don't block it."""
    hits = []
    for alias, place in NATIVE_ALIASES.items():
        start = text.find(alias)
        while start != -1:
            before = text[start - 1] if start else " "
            if not (before.isalpha() or unicodedata.category(before).startswith("M")):  # start of a word
                hits.append((start, place))
                break
            start = text.find(alias, start + 1)
    return hits


def extract_locations(text: str) -> list[str]:
    """Gazetteer hits (Latin and native script, in text order), then capitalised 'in/at/near X'."""
    found: list[str] = []
    ordered = [(m.start(), m.group(1).lower()) for m in _PLACE_RE.finditer(text)] + _native_hits(text)
    for _, place in sorted(ordered):
        name = place.title()
        if name not in found:
            found.append(name)
    for m in _PREP_RE.finditer(text):
        name = m.group(1)
        if name in _STOPWORDS or name.split()[0] in _STOPWORDS:
            continue
        if name.lower() not in KERALA_PLACES and name not in found:
            found.append(name)
    return found


def extract_people_count(text: str) -> int | None:
    m = _FAMILY_OF_RE.search(text)
    if m:
        return int(m.group(1))
    counts: list[int] = []
    for m in _COUNT_RE.finditer(text):
        raw = m.group(1).lower()
        counts.append(int(raw) if raw.isdigit() else _NUMBER_WORDS[raw])
    return max(counts) if counts else None


def extract_post(post: Post, geocoder: Geocoder | None = None) -> Post:
    """Return a copy of ``post`` with ``extraction`` filled in (geocoding the first resolvable place)."""
    geocoder = geocoder or Geocoder()
    locations = extract_locations(post.text)
    geo, geo_source = None, None
    for loc in locations:
        geo = geocoder.geocode(loc)
        if geo:
            geo_source = loc
            break
    extraction = Extraction(
        locations=locations,
        phones=extract_phones(post.text),
        people_count=extract_people_count(post.text),
        geo=geo,
        geo_source=geo_source,
        district=DISTRICT_OF.get(geo_source.lower()) if geo_source else None,
    )
    return post.model_copy(update={"extraction": extraction})


def extract_posts(posts: list[Post], geocoder: Geocoder | None = None) -> list[Post]:
    geocoder = geocoder or Geocoder()
    return [extract_post(p, geocoder) for p in posts]

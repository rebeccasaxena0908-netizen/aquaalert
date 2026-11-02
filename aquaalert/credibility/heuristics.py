"""Heuristic misinformation score (0 = no red flags, 1 = many). Transparent rules, not a trained model.

Each signal adds a weighted flag with a human-readable reason; flags combine as independent evidence,
score = 1 - prod(1 - weight). Citing an official source, or being a correction/debunk, lowers the score.

Signals: old/recycled event references, unverifiable alarming claims (dam breach, death tolls) without a source,
urgency-forwarding language, contradiction with nearby posts or the latest district brief, copy-pasted text.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import timedelta

from pydantic import BaseModel

from aquaalert.extract.entities import PHONE_RE
from aquaalert.extract.gazetteer import NATIVE_ALIASES
from aquaalert.match.geo import haversine_km
from aquaalert.schemas import Label, Post

NEARBY_KM = 25.0
NEARBY_HOURS = 6
DENIAL_HOURS = 24

_I = re.IGNORECASE
YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
RECYCLED_EVENT = re.compile(
    r"\b(?:uttarakhand|kedarnath|chennai\s+floods?|mumbai\s+(?:floods?\s+)?2005|nepal\s+earthquake|tsunami\s+2004|"
    r"srinagar\s+floods?|assam\s+floods?\s+20[01]\d)\b", _I)
DAM_BREACH = re.compile(
    r"\b(?:dams?|shutters?|bund)\b[^.!?\n]{0,40}\b(?:broke|broken|breach(?:ed)?|burst|collapsed?|cracked|gave\s+way|"
    r"will\s+break|is\s+breaking|toot\s+gaya|toot\s+gayi|phat\s+gaya)\b"
    r"|\bbreach\s+in\b[^.!?\n]{0,30}\bdam\b"
    r"|ഡാം\s*(?:പൊട്ടി|തകർന്നു)|(?:बांध|डैम)\s*(?:टूट|फट)|বাঁধ\s*ভেঙে", _I)
DEATH_TOLL = re.compile(
    r"\b(?:\d{2,}|hundreds|thousands)\s+(?:people\s+|persons\s+)?(?:are\s+|have\s+)?(?:dead|died|killed|bodies|drowned)\b"
    r"|\bdeath\s+toll\b|\d+\s*(?:लोगों\s+की\s+मौत|लोग\s+मरे)|മരിച്ചു|মারা\s+গেছে", _I)
CONSPIRACY = re.compile(r"\b(?:hiding\s+the\s+(?:real\s+)?(?:numbers|truth|deaths)|media\s+(?:is\s+)?not\s+showing|"
                        r"they\s+won'?t\s+tell\s+you|government\s+is\s+hiding)\b", _I)
FORWARD = re.compile(
    r"\bforward\s+(?:this|it|to\s+(?:all|everyone|every\s*one)|karo|kar\s+do)\b|\bfwd:|\bshare\s+(?:before|maximum|max|"
    r"widely|to\s+all|as\s+much)\b|\bbefore\s+(?:it'?s|it\s+is|they)\s+(?:deleted|removed|taken\s+down)\b"
    r"|\bsend\s+(?:this\s+)?to\s+all\b|\bviral\s+karo\b|\bsabko\s+(?:bhejo|forward)\b|\bellavarkkum\s+(?:share|forward)\b"
    r"|ഫോർവേഡ്|ഷെയർ\s+ചെയ്യ|फॉरवर्ड|सबको\s+भेज|ফরওয়ার্ড|সবাইকে\s+পাঠান|\bas\s+received\b", _I)
SOURCE = re.compile(
    r"\baccording\s+to\b|\bas\s+per\b|\bper\s+(?:the\s+)?(?:imd|ksdma|collector|police|bulletin|village\s+officer)\b"
    r"|\bofficials?\b|\bofficially\b|\bdistrict\s+collector\b|\bcollector\b|\bdistrict\s+administration\b|\bksdma\b"
    r"|\bndma\b|\bimd\b|\bcwc\b|\bpolice\s+said\b|\bchief\s+minister\b|\bcm\s+said\b|\bbulletin\b|\bpress\s+release\b"
    r"|https?://|മൗസം\s+വിഭാഗ|मौसम\s+विभाग|जिला\s+प्रशासन|জেলা\s+প্রশাসন", _I)
DEBUNK = re.compile(
    r"\b(?:fake|false|rumou?rs?|hoax|not\s+true|misleading|fact\s*-?check|old\s+(?:photo|video|picture|image|footage))\b"
    r"|\bplease\s+(?:don'?t|do\s+not)\s+(?:forward|share)\b|\bcheck\s+before\s+sharing\b|വ്യാജ|झूठ|अफवाह|গুজব", _I)
CALM = re.compile(
    r"\b(?:everyone|all)\s+(?:is\s+|are\s+)?safe\b|\bno\s+(?:more\s+)?(?:rescue|help)\s+(?:is\s+)?(?:needed|required)\b"
    r"|\bwater\s+(?:has\s+)?(?:receded|receding|gone\s+down)\b|\bsituation\s+(?:is\s+)?(?:normal|under\s+control)\b"
    r"|\ball\s+clear\b|\bno\s+danger\b", _I)
DAM_SAFE = re.compile(r"\bdams?\b[^.!?\n]{0,30}\b(?:is\s+)?(?:safe|stable|intact)\b|\bno\s+breach\b|\bnot\s+breached\b", _I)
SHOUTING = re.compile(r"!{3,}")
_DAM_NAME = re.compile(r"(\S+)\s+(?:dam|dams|ഡാം|डैम|बांध|বাঁধ)", _I)


def dam_names(text: str) -> set[str]:
    """Canonical names of dams mentioned ("Mullaperiyar dam", "മുല്ലപ്പെരിയാർ ഡാം" -> {"mullaperiyar"})."""
    names = set()
    for m in _DAM_NAME.finditer(text):
        tok = m.group(1).strip(".,:;!?'\"()")
        names.add(NATIVE_ALIASES.get(tok, tok.lower()))
    return names - {"the", "a", "this", "that", "of", "our", "ke", "er"}


class Flag(BaseModel):
    code: str
    weight: float
    reason: str


def combine(weights: list[float]) -> float:
    return 1 - math.prod(1 - w for w in weights)


def text_flags(post: Post) -> list[Flag]:
    """Signals that only need the post itself."""
    t = post.text
    flags: list[Flag] = []
    debunk = bool(DEBUNK.search(t))
    if not debunk:  # a debunk describes the old event / forwarding it is correcting
        m = RECYCLED_EVENT.search(t)
        old_years = sorted({int(y) for y in YEAR.findall(t) if int(y) < post.timestamp.year})
        if m:
            flags.append(Flag(code="recycled_event", weight=0.35,
                              reason=f"refers to a different, older event ('{m.group(0)}')"))
        elif old_years:
            flags.append(Flag(code="old_reference", weight=0.3,
                              reason=f"mentions {old_years[0]}, but the post is from {post.timestamp.year}"))
        if FORWARD.search(t):
            flags.append(Flag(code="forwarding", weight=0.35,
                              reason="urges people to forward/share ('forward to all', 'share before deleted')"))
    sourced = bool(SOURCE.search(t)) or bool(post.url)
    if not sourced and not debunk:
        if DAM_BREACH.search(t):
            flags.append(Flag(code="dam_breach_claim", weight=0.4, reason="claims a dam breach without naming a source"))
        if DEATH_TOLL.search(t):
            flags.append(Flag(code="death_toll_claim", weight=0.3, reason="gives a death toll without naming a source"))
    if CONSPIRACY.search(t):
        flags.append(Flag(code="conspiracy", weight=0.2, reason="says the truth is being hidden"))
    if SHOUTING.search(t):
        flags.append(Flag(code="shouting", weight=0.1, reason="sensational punctuation (!!!)"))
    return flags


def _normalise(text: str) -> str:
    text = PHONE_RE.sub(" ", re.sub(r"https?://\S+", " ", text.lower()))
    return " ".join(re.findall(r"\w+", text))


def duplicate_flags(posts: list[Post]) -> dict[str, Flag]:
    """Same text (ignoring case, punctuation, links, phone numbers) posted more than once."""
    groups: dict[str, list[Post]] = defaultdict(list)
    for p in posts:
        key = _normalise(p.text)
        if len(key.split()) >= 5:  # very short texts collide by chance
            groups[key].append(p)
    out = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        # live Bluesky posts carry no author, so accounts can't be counted for them
        by = f" by {len({p.author for p in members})} account(s)" if all(p.author for p in members) else ""
        weight = 0.25 if len(members) >= 3 else 0.15
        for p in members:
            out[p.id] = Flag(code="copy_paste", weight=weight,
                             reason=f"identical text posted {len(members)} times{by}; "
                                    "copy-pasted appeals are often outdated")
    return out


def _near(a: Post, b: Post, hours: float) -> bool:
    if abs((a.timestamp - b.timestamp).total_seconds()) > hours * 3600:
        return False
    ea, eb = a.extraction, b.extraction
    if ea and eb and ea.geo and eb.geo:
        return haversine_km(ea.geo.lat, ea.geo.lon, eb.geo.lat, eb.geo.lon) <= NEARBY_KM
    return bool(ea and eb and ea.district and ea.district == eb.district)


def contradiction_flags(posts: list[Post], briefs: list | None = None) -> dict[str, Flag]:
    """'All safe' claims next to rescue requests (posts or the latest brief); dam-breach claims vs attributed denials."""
    out: dict[str, Flag] = {}
    rescues = [p for p in posts if p.label == Label.RESCUE_NEEDED]
    denials = [p for p in posts if DAM_SAFE.search(p.text) and SOURCE.search(p.text)]
    brief_rescues = {(b.district, b.date): b.counts.get("rescue_needed", 0) for b in briefs or []}
    for p in posts:
        district = p.extraction.district if p.extraction else None
        if CALM.search(p.text) and not DAM_SAFE.search(p.text):
            nearby = [r for r in rescues if r.id != p.id and _near(p, r, NEARBY_HOURS)]
            in_brief = brief_rescues.get((district, p.timestamp.date()), 0)
            if len(nearby) >= 2:
                out[p.id] = Flag(code="contradicts_nearby", weight=0.3,
                                 reason=f"says the situation is calm, but {len(nearby)} rescue requests were posted "
                                        f"within {NEARBY_KM:.0f} km and {NEARBY_HOURS} h ("
                                        + ", ".join(r.id for r in nearby[:3]) + ")")
            elif in_brief >= 2:
                out[p.id] = Flag(code="contradicts_brief", weight=0.3,
                                 reason=f"says the situation is calm, but today's {district} brief lists "
                                        f"{in_brief} rescue requests")
        if DAM_BREACH.search(p.text):
            claimed = dam_names(p.text)
            against = [d for d in denials if d.id != p.id
                       and abs(d.timestamp - p.timestamp) <= timedelta(hours=DENIAL_HOURS)
                       and not (claimed and dam_names(d.text) and claimed.isdisjoint(dam_names(d.text)))]
            if against:
                out[p.id] = Flag(code="contradicted_by_official", weight=0.3,
                                 reason=f"contradicted by an attributed update ({against[0].id}: "
                                        f"'{against[0].text[:60]}')")
    return out


def score_posts(posts: list[Post], briefs: list | None = None) -> list[Post]:
    """Return copies of ``posts`` with ``misinfo_score``, ``misinfo_reasons`` and ``credibility`` filled in."""
    dups = duplicate_flags(posts)
    contra = contradiction_flags(posts, briefs)
    out = []
    for p in posts:
        flags = text_flags(p) + [f for f in (dups.get(p.id), contra.get(p.id)) if f]
        score = combine([f.weight for f in flags])
        reasons = [f.reason for f in flags]
        if flags and (SOURCE.search(p.text) or p.url):
            score *= 0.5
            reasons.append("cites an official source or link (halves the score)")
        if flags and DEBUNK.search(p.text):
            score *= 0.6
            reasons.append("reads as a correction/debunk (lowers the score)")
        score = round(score, 2)
        out.append(p.model_copy(update={"misinfo_score": score, "misinfo_reasons": reasons,
                                        "credibility": round(1 - score, 2)}))
    return out

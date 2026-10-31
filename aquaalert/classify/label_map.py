"""Map dataset-specific labels onto AquaAlert's label set.

HumAID has 11 humanitarian categories and no rescue class; rescue requests are hidden
inside ``requests_or_urgent_needs``, so that class is split by ``RESCUE_RE``.
"""
from __future__ import annotations

import re

from aquaalert.schemas import LABELS, Label

# HumAID categories -> AquaAlert labels. None = drop the example (no usable signal).
HUMAID_MAP: dict[str, Label | None] = {
    "caution_and_advice": Label.INFO_UPDATE,
    "displaced_people_and_evacuations": Label.INFO_UPDATE,  # mostly reports, not requests
    "dont_know_cant_judge": None,
    "infrastructure_and_utility_damage": Label.INFRASTRUCTURE_DAMAGE,
    "injured_or_dead_people": Label.INFO_UPDATE,  # mostly casualty reports, not requests
    "missing_or_found_people": Label.MISSING_PERSON,
    "not_humanitarian": Label.IRRELEVANT,
    "other_relevant_information": Label.INFO_UPDATE,
    "requests_or_urgent_needs": Label.NEED_SUPPLIES,  # -> rescue_needed if RESCUE_RE matches
    "rescue_volunteering_or_donation_effort": Label.OFFER_HELP,
    "sympathy_and_support": Label.IRRELEVANT,
}

# Dataset classes whose posts are re-routed to rescue_needed when they use rescue language.
RESCUE_SOURCE_CLASSES: dict[str, set[str]] = {"humaid": {"requests_or_urgent_needs"}}

# Rescue-request language (not mere mentions of "rescue efforts").
RESCUE_RE = re.compile(
    r"\b(stranded|trapped|stuck|marooned|sos)\b"
    r"|\bneed(s|ed)?\s+(to\s+be\s+)?(rescue[d]?|evacuat\w+|saved|boats?|helicopters?)\b"
    r"|\b(please\s+|pls\s+)?(rescue|evacuate|save)\s+(us|me|them|my|our)\b"
    r"|\b(send|need|require)\w*\s+(a\s+)?(rescue|boats?|helicopters?|ndrf)\b"
    r"|\b(on|atop)\s+(the\s+)?(roof|rooftop|terrace)\b"
    r"|\bhelp\s+(us|me)\s+(get out|evacuate)\b",
    re.IGNORECASE,
)

# TREC-IS / CrisisLex style information types
TREC_IS_MAP: dict[str, Label | None] = {
    "search_and_rescue": Label.RESCUE_NEEDED,
    "movemententities": Label.RESCUE_NEEDED,
    "goodsservices": Label.NEED_SUPPLIES,
    "volunteer": Label.OFFER_HELP,
    "donations": Label.OFFER_HELP,
    "missingpeople": Label.MISSING_PERSON,
    "infrastructure": Label.INFRASTRUCTURE_DAMAGE,
    "serviceavailable": Label.INFO_UPDATE,
    "report-firsthandobservation": Label.INFO_UPDATE,
    "report-emergingthreats": Label.INFO_UPDATE,
    "news": Label.INFO_UPDATE,
    "irrelevant": Label.IRRELEVANT,
}

DATASET_MAPS: dict[str, dict[str, Label | None]] = {"humaid": HUMAID_MAP, "trec_is": TREC_IS_MAP}


def normalize(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_").replace("-", "_") if raw else ""


def map_label(raw: str | None, dataset: str | None = None, text: str | None = None) -> Label | None:
    """Return the AquaAlert label for a raw dataset label, or None if unknown/dropped.

    ``text`` enables text-dependent routing (the HumAID rescue rule).
    """
    if not raw:
        return None
    norm = normalize(raw)
    table = DATASET_MAPS.get(dataset or "")
    if table is not None:
        key = norm if norm in table else raw.strip().lower()
        if key in table:
            label = table[key]
            if label and text and norm in RESCUE_SOURCE_CLASSES.get(dataset or "", ()) and RESCUE_RE.search(text):
                return Label.RESCUE_NEEDED
            return label
    return Label(norm) if norm in LABELS else None

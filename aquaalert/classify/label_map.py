"""Map dataset-specific labels onto AquaAlert's label set."""
from __future__ import annotations

from aquaalert.schemas import LABELS, Label

# HumAID information types -> AquaAlert labels
HUMAID_MAP: dict[str, Label] = {
    "rescue_volunteering_or_donation_effort": Label.OFFER_HELP,
    "requests_or_urgent_needs": Label.NEED_SUPPLIES,
    "injured_or_dead_people": Label.RESCUE_NEEDED,
    "missing_or_found_people": Label.MISSING_PERSON,
    "infrastructure_and_utility_damage": Label.INFRASTRUCTURE_DAMAGE,
    "displaced_people_and_evacuations": Label.RESCUE_NEEDED,
    "caution_and_advice": Label.INFO_UPDATE,
    "sympathy_and_support": Label.IRRELEVANT,
    "other_relevant_information": Label.INFO_UPDATE,
    "not_humanitarian": Label.IRRELEVANT,
    "dont_know_cant_judge": Label.IRRELEVANT,
}

# TREC-IS / CrisisLex style information types
TREC_IS_MAP: dict[str, Label] = {
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

DATASET_MAPS: dict[str, dict[str, Label]] = {"humaid": HUMAID_MAP, "trec_is": TREC_IS_MAP}


def normalize(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_").replace("-", "_") if raw else ""


def map_label(raw: str | None, dataset: str | None = None) -> Label | None:
    """Return the AquaAlert label for a raw dataset label, or None if unknown."""
    if not raw:
        return None
    key = raw.strip().lower()
    if dataset and dataset in DATASET_MAPS:
        mapped = DATASET_MAPS[dataset].get(key) or DATASET_MAPS[dataset].get(normalize(raw))
        if mapped:
            return mapped
    norm = normalize(raw)
    return Label(norm) if norm in LABELS else None

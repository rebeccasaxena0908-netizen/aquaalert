"""Zero-shot fallback classifier: weighted multilingual keyword rules.

Works with no training and no model download, so the demo runs out of the box. Covers
English plus romanised Hindi/Hinglish, Bengali and Malayalam cues and a few native-script
terms. Optionally an NLI pipeline (``NLIZeroShot``) can be used when transformers is installed.
"""
from __future__ import annotations

import math
import re

from aquaalert.schemas import LABELS

# label -> [(regex, weight)]
RULES: dict[str, list[tuple[str, float]]] = {
    "rescue_needed": [
        (r"\b(stranded|trapped|stuck|marooned)\b", 3),
        (r"\b(rescue|evacuat\w*|sos)\b", 2),
        (r"\b(roof|rooftop|terrace)\b", 2),
        (r"\b(rising fast|water (is )?(rising|above|entering)|submerged)\b", 1.5),
        (r"\bboats? (needed|required)\b|\bneed\w* (a )?boats?\b|\bsend boats\b", 2.5),
        (r"\b(phase|phans)\w*\b|\bchhat par\b|\bbachao\b|\bbondi\b|\bnouka\b", 3),
        (r"\b(labour pain|pregnant|elderly (couple|people))\b", 1.5),
        (r"\bhelp!|\burgent(ly)?\b", 1),
        (r"രക്ഷ|बचाओ", 3),
    ],
    "infrastructure_damage": [
        (r"\b(bridge|road|roads|railway|tracks?|highway)\b[^.]{0,40}\b(collapsed?|damaged?|washed|cut off|destroyed|broken|blocked)\b", 3.5),
        (r"\b(collapsed|destroyed|damaged|washed away|burst|blocked)\b", 2),
        (r"\b(power lines?|transformer|electricity|pipelines?|landslide)\b", 2),
        (r"\b(toot gayi|band)\b", 2),
        (r"\b(houses?|property)\b[^.]{0,30}\b(collapsed|damage\w*)\b", 1.5),
    ],
    "need_supplies": [
        (r"\b(need|needs|needed|required|chahiye|dorkar)\b[^.]{0,60}\b(food|water|medicines?|insulin|blankets?|clothes|rice|diesel|sanitary|packets?|khaana|kapde)\b", 4),
        (r"\b(food|water|medicines?|insulin|blankets?|clothes|rice|diesel|khaana|kapde)\b[^.]{0,30}\b(needed|required|urgent\w*)\b", 3.5),
        (r"\brelief camps?\b|\bcamp\b", 1),
        (r"\b(food|drinking water|medicines?|blankets?|clothes|khaana)\b", 1.5),
        (r"\bplease send\b", 1),
    ],
    "offer_help": [
        (r"\b(offering|can donate|we can|volunteers?|ready to assist|ready to help)\b", 3),
        (r"\b(available|free (stay|medical)|donate|trucks ready|dm (us|me)|message me)\b", 2.5),
        (r"\b(bringing (their )?boats|team of)\b", 3),
        (r"\b(have|with) (gloves|masks|boats?)\b", 2),
    ],
    "missing_person": [
        (r"\b(missing|laapata|lapata|nikhoj|looking for my|have you seen|separated from|last seen|last contact)\b", 4),
        (r"\b(whereabouts|untraced)\b|കാണാനില്ല|लापता", 3),
    ],
    "info_update": [
        (r"\b(alert|forecast|level|levels|ft|feet|announces?|update|suspended|cancelled|resumes?|receding|closed|opened|shutters?)\b", 2),
        (r"\b(imd|dam|cm|ndma|authorities|government|relief fund)\b", 1.5),
        (r"\b(heavy|extreme) rain(fall)?\b|\bflood level\b|\bred alert\b", 2),
        (r"\bstay calm\b|\bon the way\b", 1),
    ],
}

_COMPILED = {
    label: [(re.compile(pat, re.IGNORECASE), w) for pat, w in rules] for label, rules in RULES.items()
}
_FLOODISH = re.compile(r"\b(flood\w*|rain\w*|water|dam|river|kerala|baadh)\b|बाढ़|വെള്ളപ്പൊക്കം", re.IGNORECASE)
_IRRELEVANT_BASE = 1.0  # score of the irrelevant class when nothing else fires


def _score_text(text: str) -> dict[str, float]:
    raw = {label: 0.0 for label in LABELS}
    for label, rules in _COMPILED.items():
        raw[label] = sum(w for rx, w in rules if rx.search(text))
    if max(raw.values()) == 0:
        # Nothing matched: weakly info_update if flood-related, otherwise irrelevant.
        if _FLOODISH.search(text):
            raw["info_update"] = 1.0
        raw["irrelevant"] = _IRRELEVANT_BASE
    return raw


def _softmax(raw: dict[str, float], temperature: float = 1.5) -> dict[str, float]:
    exps = {k: math.exp(v / temperature) for k, v in raw.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


class KeywordZeroShot:
    name = "keyword-zeroshot"

    def predict_scores(self, texts: list[str]) -> list[dict[str, float]]:
        return [_softmax(_score_text(t)) for t in texts]


class NLIZeroShot:
    """Optional NLI-based zero-shot (needs transformers + a model download)."""

    name = "nli-zeroshot"
    HYPOTHESES = {
        "rescue_needed": "people are trapped and need to be rescued",
        "infrastructure_damage": "roads, bridges or buildings are damaged",
        "need_supplies": "people need food, water or medicine",
        "offer_help": "someone is offering help or donations",
        "missing_person": "a person is missing",
        "info_update": "this is a news or status update about the flood",
        "irrelevant": "this is unrelated to the flood",
    }

    def __init__(self, model: str = "joeddav/xlm-roberta-large-xnli") -> None:
        from transformers import pipeline  # lazy: heavy import

        self._pipe = pipeline("zero-shot-classification", model=model)

    def predict_scores(self, texts: list[str]) -> list[dict[str, float]]:
        inverse = {v: k for k, v in self.HYPOTHESES.items()}
        out = []
        for text in texts:
            res = self._pipe(text, list(inverse), multi_label=False)
            out.append({inverse[l]: float(s) for l, s in zip(res["labels"], res["scores"])})
        return out

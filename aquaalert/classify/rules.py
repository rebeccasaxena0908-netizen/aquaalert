"""Post-classification rule layer: promote clear first-person distress to ``rescue_needed``.

Models trained on HumAID read "rescue"/"boats"/"help" as volunteering, so urgent calls for help are sometimes
labelled ``offer_help`` or ``info_update``. Only those two labels are ever overridden, and only when:

* a **strong** cue is present (an explicit request for rescue/help), or
* a **contextual** cue (trapped, stuck, rooftop, water entering, ...) appears together with a first-person marker
  (we / our / I'm / my / hum / mere ...).

A genuine offer ("we are rescuing", "boats available", "volunteers ready") or an appeal (fundraising, thanks,
volunteer recruiting: "help us raise money", "SOS blood bank") blocks every cue except unambiguous requests such as
"need rescue" / "boats needed" / "bachao".
The firing rule is recorded on the Post (``rule``), with the model's own label kept in ``model_label``.
"""
from __future__ import annotations

import re

from aquaalert.schemas import Label

OVERRIDABLE = {Label.OFFER_HELP, Label.INFO_UPDATE}

# Unambiguous requests for rescue: fire regardless of surrounding wording
STRONG_EXPLICIT = re.compile(
    r"\bneeds?\s+(?:to\s+be\s+)?(?:rescue[d]?|evacuat\w+|saved)\b"
    r"|\b(?:rescue|evacuation)\s+(?:is\s+)?(?:needed|required|urgently)\b"
    r"|\bboats?\s+(?:needed|required|chahiye)\b|\bneed\s+(?:a\s+)?boats?\b"
    r"|\b(?:bachao|bachaiye|bachalo)\b|\b(?:help|madad)\s+karo\b|\bmadad\s+chahiye\b",
    re.IGNORECASE,
)
# "no more rescue needed", "rescue is not needed": the explicit cue negated
NEGATED = re.compile(
    r"\b(?:no\s+(?:more\s+|longer\s+|further\s+)?|not\s+|don'?t\s+|doesn'?t\s+)"
    r"(?:\w+\s+){0,2}?(?:rescue|evacuation|boats?|help)\b[^.!?]{0,15}\b(?:needed|required|need)\b"
    r"|\b(?:rescue|evacuation|boats?|help)\s+(?:is\s+|are\s+)?(?:not|no\s+longer)\s+(?:needed|required)\b",
    re.IGNORECASE,
)
# Phrases that also appear in fundraising / volunteer appeals ("help us raise money"): blocked by APPEAL
STRONG_AMBIGUOUS = re.compile(
    # "help us" only when it ends the clause or continues like a distress call ("help us, we are trapped")
    r"\b(?:help|save|rescue)\s+us\b(?=\s*(?:[!.,?;:]|$|please|pls|now|asap|urgent\w*|before|get\s+out|evacuate|from|we\b|our\b))"
    r"|\bplease\s+(?:rescue|save)\s+(?:us|me|my|our)\b|\bplease\s+help\s+me\b"
    r"|\bsos\b|\bplease\s+send\s+(?:a\s+)?(?:boats?|helicopter|rescue|help)\b",
    re.IGNORECASE,
)
# Fundraising, thanks, volunteer recruiting, past-tense accounts: not a live call for rescue
APPEAL = re.compile(
    r"\b(?:donat\w*|fundrais\w*|raise[ds]?|raising|telethon|charity|crowdfund\w*|campaign|register|sign\s*up|"
    r"volunteer(?:s)?\s+(?:here|now)|thank(?:s|\s+you)?|grateful|appreciat\w+|proud\s+of|blood\s*bank|"
    r"helped\s+(?:to\s+)?save|saved\s+(?:my|our|the|\d)|rescued\s+(?:my|our|the|\d))\b",
    re.IGNORECASE,
)

# Situation cues; only count with a first-person marker
CONTEXTUAL = re.compile(
    r"\b(?:trapped|stranded|stuck|marooned)\b(?!\s+(?:in|due\s+to)\s+(?:traffic|jam))"
    r"|\b(?:on|atop)\s+(?:the\s+|our\s+|my\s+)?(?:roof|rooftop|terrace)\b|\brooftop\b"
    r"|\bwater\s+(?:is\s+)?(?:entering|entered|coming\s+into|rising\s+in)\b"
    r"|\bpaani\s+(?:ghar|andar)\b|\bchhat\s+par\b|\b(?:phans|phas|fas|phase)\s*(?:gaye|gaya|gayi|gae|hue|hain)\b",
    re.IGNORECASE,
)
FIRST_PERSON = re.compile(
    r"\b(?:we|we're|we've|our|i|i'm|im|i've|my|me)\b|\b(?:hum|hamare|hamari|main|mere|meri|mujhe)\b",
    re.IGNORECASE,
)
# "DM us/me" is an offer idiom, not distress; strip it before looking for first-person words
_DM = re.compile(r"\b(?:dm|message|contact|call)\s+(?:us|me)\b", re.IGNORECASE)
# Narration of something that already happened ("I got stuck here after the hurricane") is not a live call
PAST_SITUATION = re.compile(r"\b(?:was|were|got|had\s+been)\s+(?:\w+\s+)?(?:trapped|stuck|stranded|marooned)\b", re.IGNORECASE)
# Genuine offers of help
OFFER = re.compile(
    r"\b(?:i|we)\s+(?:was|were)\s+(?:rescuing|helping|evacuating)\b|\b(?:i|we)\s+(?:rescued|evacuated|helped)\b"
    r"|\bassignment\s+is\s+to\b|\b(?:emergency\s+)?response\s+teams?\b|\b(?:we|our\s+team)\s+(?:continues?|continue)\s+to\b"
    r"|\bwe\s+(?:are|re|'re)\s+(?:rescuing|helping|evacuating|distributing|providing|sending)\b"
    r"|\bwe\s+(?:can|will|could)\s+(?:rescue|help|send|provide|donate|evacuate|assist)\b"
    r"|\b(?:boats?|rescue\s+teams?|trucks?|shelter|ambulances?)\b[^.!?]{0,25}\b(?:available|ready|on\s+the\s+way|arranged)\b"
    r"|\bvolunteers?\b[^.!?]{0,40}\b(?:ready|available|needed\s+to\s+help|reached)\b"
    r"|\b(?:offering|ready\s+to\s+(?:help|assist|rescue))\b|\bour\s+(?:team|volunteers)\b"
    r"|\bcan\s+(?:rescue|help|host|shelter)\b",
    re.IGNORECASE,
)

CONF_STRONG, CONF_CONTEXTUAL = 0.80, 0.65  # heuristic confidences for rule-made labels


def rescue_rule(text: str) -> tuple[str, float] | None:
    """Return (rule name, confidence) if ``text`` shows clear distress, else None."""
    if STRONG_EXPLICIT.search(NEGATED.sub(" ", text)):
        return "distress_strong", CONF_STRONG
    if APPEAL.search(text) or OFFER.search(text):
        return None
    if STRONG_AMBIGUOUS.search(text):
        return "distress_strong", CONF_STRONG
    if CONTEXTUAL.search(text) and FIRST_PERSON.search(_DM.sub("", text)) and not PAST_SITUATION.search(text):
        return "distress_contextual", CONF_CONTEXTUAL
    return None


def apply_rules(text: str, label: Label) -> tuple[Label, str | None, float | None]:
    """Return (final label, rule name or None, rule confidence or None)."""
    if label in OVERRIDABLE:
        hit = rescue_rule(text)
        if hit:
            return Label.RESCUE_NEEDED, hit[0], hit[1]
    return label, None, None

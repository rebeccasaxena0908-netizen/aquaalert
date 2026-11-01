"""Post-classification rule layer: catch clear first-person distress the classifier missed.

Models trained on HumAID read "rescue"/"boats"/"help" as volunteering and romanised Indian-language posts as supply
requests, so urgent calls for help are sometimes labelled ``offer_help``, ``info_update`` or ``need_supplies``.
Cues come in three strengths, in English, romanised Hindi/Malayalam/Bengali (Hinglish, Manglish, Banglish) and the
native scripts:

* **explicit**: an unambiguous request ("need rescue", "boats needed", "bachao", "boat bhejo", "rakshikkane",
  "nouka pathan", "बचाओ") - fires regardless of surrounding wording;
* **ambiguous**: phrases that also occur in appeals ("help us", "SOS", "please send boats") - blocked by appeal,
  thanks, rescuer or past-tense wording;
* **contextual**: a situation ("trapped", "chhat pe", "fase hue hain", "kudungi", "atke achhi", "ഞങ്ങൾ … കുടുങ്ങി")
  - counts only with a first-person marker (we / hum / njangal / amra / हम / ഞങ്ങൾ / আমরা …) and the same guards.

Which model labels a cue may act on:

* ``offer_help`` / ``info_update``: any cue promotes to ``rescue_needed``;
* ``need_supplies``: an explicit cue, or a first-person situation cue when the post asks for no supplies; never an
  ambiguous appeal ("please send help to Puerto Rico" is a request in HumAID, not a rescue);
* ``irrelevant``: explicit cues only;
* ``rescue_needed``: the rule is recorded as *confirming* the model (label unchanged, visible as a rule badge);
* ``missing_person`` / ``infrastructure_damage``: never touched.

The firing rule is recorded on the Post (``rule``), with the model's own label in ``model_label``.
"""
from __future__ import annotations

import re

from aquaalert.schemas import Label

_I = re.IGNORECASE

# ---------------------------------------------------------------- cues
# Unambiguous requests for rescue
STRONG_EXPLICIT = re.compile(
    r"\bneeds?\s+(?:to\s+be\s+)?(?:rescue[d]?|evacuat\w+|saved)\b"
    r"|\b(?:rescue|evacuation)\s+(?:is\s+)?(?:needed|required|urgently)\b"
    r"|\bboats?\s+(?:needed|required|chahiye|venam|dorkar|lagbe)\b|\bneed\s+(?:a\s+)?boats?\b"
    r"|\bplease\s+send\s+(?:a\s+)?(?:boats?|helicopters?|rescue\s+teams?|rescue)\b"
    # Hinglish
    r"|\b(?:bachao|bachaiye|bachalo|bacha\s+lo)\b|\b(?:help|madad)\s+karo\b|\bmadad\s+chahiye\b"
    r"|\b(?:boat|naav|nav|nau?ka)\s+(?:bhejo|bhej\s+do|bhejiye|chahiye)\b|\brescue\s+karo\b"
    r"|\b(?:hume|humein|hamein)\s+nikalo\b|\bnikalo\s+(?:hume|humein|hamein)\b"
    # Manglish
    r"|\brakshikk(?:ane|anam|oo|uka)\b|\b(?:thoni|vallam|boat)\s+(?:venam|ayakku|ayakkoo|ayakkane)\b"
    r"|\brescue\s+cheyy(?:ane|anam|u)\b"
    # Banglish
    r"|\buddhar\s+(?:korun|koro|kore\s+din)\b|\bbachan\b|\bnouka\s+(?:pathan|pathao|dorkar|lagbe)\b"
    # native scripts (no \b: it breaks inside Indic words)
    r"|बचाओ|बचाइए|बचा\s*लो|नाव\s*भेज|रक्षा\s*करो|രക്ഷിക്ക|ബോട്ട്\s*അയക്ക|തോണി\s*വേണം"
    r"|উদ্ধার\s*কর|বাঁচাও|নৌকা\s*পাঠা",
    _I,
)
# "no more rescue needed", "rescue is not needed": the explicit cue negated
NEGATED = re.compile(
    r"\b(?:no\s+(?:more\s+|longer\s+|further\s+)?|not\s+|don'?t\s+|doesn'?t\s+)"
    r"(?:\w+\s+){0,2}?(?:rescue|evacuation|boats?|help)\b[^.!?]{0,15}\b(?:needed|required|need)\b"
    r"|\b(?:rescue|evacuation|boats?|help)\s+(?:is\s+|are\s+)?(?:not|no\s+longer)\s+(?:needed|required)\b"
    r"|\b(?:boat|madad|rescue)\s+(?:ki\s+)?(?:zarurat|zaroorat|jarurat)\s+nahi\b",
    _I,
)
# Phrases that also appear in fundraising / volunteer appeals ("help us raise money"): blocked by the guards
STRONG_AMBIGUOUS = re.compile(
    r"\b(?:help|save|rescue)\s+us\b(?=\s*(?:[!.,?;:]|$|please|pls|now|asap|urgent\w*|before|get\s+out|evacuate|from|we\b|our\b))"
    r"|\bplease\s+(?:rescue|save)\s+(?:us|me|my|our)\b|\bplease\s+help\s+me\b"
    r"|\bsos\b|\bplease\s+send\s+(?:some\s+)?help\b"
    r"|\bjaldi\s+(?:madad|help)\b|\b(?:pettennu|vegam)\s+sahayikk\w*\b|\btaratari\s+(?:sahajyo|help)\b",
    _I,
)
# Situation cues; only count with a first-person marker
CONTEXTUAL = re.compile(
    r"\b(?:trapped|stranded|stuck|marooned)\b(?!\s+(?:in|due\s+to)\s+(?:traffic|jam))"
    r"|\b(?:on|atop)\s+(?:the\s+|our\s+|my\s+)?(?:roof|rooftop|terrace)\b|\brooftop\b"
    r"|\bwater\s+(?:is\s+)?(?:entering|entered|coming\s+into|rising\s+in)\b"
    # Hinglish: "fase hue hain", "phanse hain", "chhat pe", "paani ghar mein", "doob rahe"
    r"|\b(?:ph|f)a?n?s(?:e|i|a)?\s+(?:hue|huye|gaye|gaya|gayi|gae|hain|hai|he)\b"
    r"|\b(?:chhat|chat|chhath)\s+(?:par|pe|per)\b|\bpaani\s+(?:ghar|andar)\b|\bghar\s+(?:me|mein|main)\s+paani\b"
    r"|\b(?:paani|pani)\s+(?:badh|bad|chadh)\s+raha\b|\bdoob\s+(?:rahe|raha|gaye|jayenge)\b"
    # Manglish: "kudungi", "vellam kayari", "terrace il", "mukalil"
    r"|\bkudung\w*\b|\bvellam\s+(?:kayar\w*|keri\w*)\b|\bveett?il\s+vellam\b|\bterrace\s*-?\s*il\b|\bmukalil\b"
    # Banglish: "atke achhi", "chhade", "jol barche"
    r"|\batke\s+(?:achhi|achi|ache|achhe|gechi|gelam|achhen)\b|\b(?:chhade|chade|chhad\s+e)\b"
    r"|\bjol\s+(?:barche|barchhe|dhukche|dhukchhe|uthche)\b|\bbari(?:te)?\s+jol\b"
    # native scripts
    r"|फंसे|फँसे|फंस\s*गए|छत\s*पर|घर\s*में\s*पानी|കുടുങ്ങ|വെള്ളം\s*കയറ|മുകളിൽ|আটকে|ছাদে|জল\s*ঢুক",
    _I,
)
FIRST_PERSON = re.compile(
    r"\b(?:we|we're|we've|our|i|i'm|im|i've|my|me)\b"
    r"|\b(?:hum|humein|hamein|hamare|hamari|main|mere|meri|mujhe)\b"  # Hinglish
    r"|\b(?:njan|njangal|nammal|ente|enikku|njangalude)\b"  # Manglish
    r"|\b(?:ami|amra|amar|amader|amake)\b"  # Banglish
    r"|हम|मैं|हमारे|हमें|ഞങ്ങൾ|ഞാൻ|ഞങ്ങളുടെ|എന്റെ|আমরা|আমি|আমাদের|আমার",
    _I,
)
# "DM us/me" is an offer idiom, not distress; strip it before looking for first-person words
_DM = re.compile(r"\b(?:dm|message|contact|call)\s+(?:us|me)\b", _I)

# ---------------------------------------------------------------- guards
# Fundraising, thanks, volunteer recruiting, being rescued already: not a live call for rescue
APPEAL = re.compile(
    r"\b(?:donat\w*|fundrais\w*|raise[ds]?|raising|telethon|charity|crowdfund\w*|campaign|register|sign\s*up|"
    r"volunteer(?:s)?\s+(?:here|now)|thank(?:s|\s+you)?|grateful|appreciat\w+|proud\s+of|blood\s*bank|"
    r"helped\s+(?:to\s+)?save|saved\s+(?:my|our|the|\d)|rescued\s+(?:my|our|the|\d))\b"
    r"|\b(?:dhanyavad|dhanyawad|shukriya|chanda|daan\s+karein|nanni|dhonnobad|dhonyobad)\b"
    r"|धन्यवाद|शुक्रिया|दान\s*करें|നന്ദി|ধন্যবাদ",
    _I,
)
# Narration of something that already happened ("I got stuck here", "hum phanse the", "rakshichu")
PAST_SITUATION = re.compile(
    r"\b(?:was|were|got|had\s+been)\s+(?:\w+\s+)?(?:trapped|stuck|stranded|marooned)\b"
    r"|\b(?:ph|f)a?n?s(?:e|i|a)?\s+(?:hue\s+)?the\b|\bbacha\s+(?:liya|liye|li)\b|\bbachaya\s+gaya\b"
    r"|\bnikal\s+(?:liya|liye)\b|\brescue\s+(?:ho\s+gaye|kar\s+liya|kar\s+liye)\b"
    r"|\brakshi(?:chu|chhu)\b|\brakshappett\w*\b|\bkudungiyirunnu\b"
    r"|\batke\s+(?:chhilam|chilam|chhilo|chilo)\b|\buddhar\s+kora\s+hoyeche\b|\bbache\s+gechi\b",
    _I,
)
# Genuine offers of help / rescuer wording
OFFER = re.compile(
    r"\b(?:i|we)\s+(?:was|were)\s+(?:rescuing|helping|evacuating)\b|\b(?:i|we)\s+(?:rescued|evacuated|helped)\b"
    r"|\bassignment\s+is\s+to\b|\b(?:emergency\s+)?response\s+teams?\b|\b(?:we|our\s+team)\s+(?:continues?|continue)\s+to\b"
    r"|\bwe\s+(?:are|re|'re)\s+(?:rescuing|helping|evacuating|distributing|providing|sending)\b"
    r"|\bwe\s+(?:can|will|could)\s+(?:rescue|help|send|provide|donate|evacuate|assist)\b"
    r"|\b(?:boats?|rescue\s+teams?|trucks?|shelter|ambulances?)\b[^.!?]{0,25}\b(?:available|ready|on\s+the\s+way|arranged)\b"
    r"|\bvolunteers?\b[^.!?]{0,40}\b(?:ready|available|needed\s+to\s+help|reached)\b"
    r"|\b(?:offering|ready\s+to\s+(?:help|assist|rescue))\b|\bour\s+(?:team|volunteers)\b"
    r"|\bcan\s+(?:rescue|help|host|shelter)\b"
    # Hinglish / Manglish / Banglish rescuer wording
    r"|\bhum\s+(?:log\s+)?(?:bacha|nikal|rescue\s+kar)\s+rahe\b|\b(?:hamare|mere)\s+paas\s+(?:naav|nav|boat|boats|truck)\b"
    r"|\b(?:naav|nav|boat)\s+hai\s+(?:hamare|mere)\s+paas\b|\bmadad\s+kar\s+sakt\w*\b|\bbhej\s+sakt\w*\b"
    r"|\b(?:thoni|vallam|boat|vandi)\s+und[ue]?\b|\bnjangal\s+rakshikkunnu\b|\bsahayikkam\b|\bethikkam\b"
    r"|\bamader\s+(?:nouka|boat)\s+ache\b|\bamra\s+uddhar\s+korchi\b|\bsahajyo\s+korte\s+(?:pari|parbo)\b",
    _I,
)
# Supply words: a "need_supplies" post that only describes a situation stays a supply request
SUPPLIES = re.compile(
    r"\b(?:food|meals?|drinking\s+water|water\s+bottles?|medicines?|insulin|blankets?|clothes|rice|milk|diesel|"
    r"sanitary|ration|khana|khaana|dawai|dawa|kapde|doodh|bhakshanam|marunnu|khabar|oshudh)\b"
    r"|खाना|दवा|राशन|ഭക്ഷണം|മരുന്ന്|খাবার|ওষুধ",
    _I,
)

# Hashtags (#BJP_भगाओ_देश_बचाओ), @mentions and URLs carry slogans and campaigns, not personal requests; so do
# "<x> bachao" slogans (desh bachao, beti bachao). They are removed before any cue is looked for.
_NOISE = re.compile(r"#\S+|@\w+|https?://\S+", _I)
SLOGAN = re.compile(r"\b(?:desh|beti|loktantra|samvidhan|lokatantra|paryavaran|jungle|kisan)\s*[_ ]?\s*bachao\b"
                    r"|देश\s*बचाओ|बेटी\s*बचाओ|लोकतंत्र\s*बचाओ|संविधान\s*बचाओ", _I)

CONF_STRONG, CONF_CONTEXTUAL = 0.80, 0.65  # heuristic confidences for rule-made labels


def _clean(text: str) -> str:
    return SLOGAN.sub(" ", _NOISE.sub(" ", text))


def rescue_rule(text: str) -> tuple[str, float] | None:
    """Return (rule name, confidence) if ``text`` shows clear distress, else None.

    Rule names: ``distress_strong`` (explicit or ambiguous cue) or ``distress_contextual``.
    """
    text = _clean(text)
    if STRONG_EXPLICIT.search(NEGATED.sub(" ", text)):
        return "distress_strong", CONF_STRONG
    if APPEAL.search(text) or OFFER.search(text) or PAST_SITUATION.search(text):
        return None
    if STRONG_AMBIGUOUS.search(text):
        return "distress_strong", CONF_STRONG
    if CONTEXTUAL.search(text) and FIRST_PERSON.search(_DM.sub("", text)):
        return "distress_contextual", CONF_CONTEXTUAL
    return None


def _explicit(text: str) -> bool:
    return bool(STRONG_EXPLICIT.search(NEGATED.sub(" ", _clean(text))))


def _first_person_situation(text: str) -> bool:
    """A contextual cue with a first-person marker and none of the guards (checked even when an ambiguous cue also matched)."""
    t = _clean(text)
    return bool(CONTEXTUAL.search(t) and FIRST_PERSON.search(_DM.sub("", t))
                and not (APPEAL.search(t) or OFFER.search(t) or PAST_SITUATION.search(t)))


def apply_rules(text: str, label: Label) -> tuple[Label, str | None, float | None]:
    """Return (final label, rule name or None, rule confidence or None).

    A rule that only confirms the model (label already ``rescue_needed``) returns confidence None, so callers keep
    the model's own score.
    """
    if label in (Label.MISSING_PERSON, Label.INFRASTRUCTURE_DAMAGE):
        return label, None, None
    hit = rescue_rule(text)
    if not hit:
        return label, None, None
    name, conf = hit
    if label == Label.RESCUE_NEEDED:
        return label, name, None  # confirms the model
    if label in (Label.OFFER_HELP, Label.INFO_UPDATE):
        return Label.RESCUE_NEEDED, name, conf
    if label == Label.NEED_SUPPLIES and (_explicit(text) or (_first_person_situation(text)
                                                             and not SUPPLIES.search(text))):
        return Label.RESCUE_NEEDED, name, conf
    if label == Label.IRRELEVANT and _explicit(text):
        return Label.RESCUE_NEEDED, name, conf
    return label, None, None

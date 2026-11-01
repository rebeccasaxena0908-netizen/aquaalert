"""Language identification for Indian flood posts, including romanised (Latin-script) Hindi, Malayalam and Bengali.

Native scripts are identified from Unicode blocks (Bengali vs Assamese by the Assamese-only letters ৰ/ৱ). Latin-script
text is scored against small lexicons of frequent, distinctive romanised words ("hai", "njan", "ami", ...) versus English
function words. Codes: ``en``, ``hi``, ``ml``, ``bn``, ``as``, ``ta``, ``te``, ``kn``, ``gu``, ``pa``, ``or``, romanised
``hi-Latn`` / ``ml-Latn`` / ``bn-Latn``, or ``und``. ``code_mixed`` is set when an Indian language is mixed with English.

    python -m aquaalert.language.detect "Hum chhat par phase hue hain, please help"
"""
from __future__ import annotations

import re
import sys
from collections import Counter

from pydantic import BaseModel

SCRIPTS: list[tuple[str, int, int]] = [
    ("hi", 0x0900, 0x097F),  # Devanagari (Hindi; Marathi/Nepali also use it)
    ("bn", 0x0980, 0x09FF),  # Bengali-Assamese
    ("pa", 0x0A00, 0x0A7F), ("gu", 0x0A80, 0x0AFF), ("or", 0x0B00, 0x0B7F), ("ta", 0x0B80, 0x0BFF),
    ("te", 0x0C00, 0x0C7F), ("kn", 0x0C80, 0x0CFF),
    ("ml", 0x0D00, 0x0D7F),  # Malayalam
]
ASSAMESE_ONLY = set("ৰৱ")

# Distinctive romanised words (shared or English-looking words such as "me", "to", "pani", "amma" left out)
ROMANISED: dict[str, set[str]] = {
    "hi-Latn": set("""hai hain nahi nahin kya kyun kyon hum humein hamara hamare hamari mera mere meri aap aapka aapke
        tum tumhara unka unke yahan wahan paani madad chahiye raha rahe rahi gaya gaye gayi karo karke kuch sab sabko
        logon ghar chhat baadh barish khana khaana kapde abhi jaldi bahut aur lekin phir toh bhi mein ko ka ki ke
        se phase phans phas phanse koi kisi karna karein kijiye batao bataiye laapata lapata pitaji beta beti dikkat
        musibat bachaiye toot band""".split()),
    "ml-Latn": set("""njan njangal ningal nammal enthu entha enthanu illa alle aanu undu venam vendi cheyyu cheyyanam
        vellam veedu veettil veetil il sahayam rakshikkane rakshikkuka kudungi kudungiyirikkunnu kudungikidakkunnu pettu
        poyi varu avide ivide ente enikku ningalude kuttikal makkal chettan chechi kayari kayariyittundu mukalil
        thoni boat-il vegam ethrayum pettennu sahayikkuka athe ithu athu karanam onnum kittunnilla bhakshanam""".split()),
    "bn-Latn": set("""ami amra tumi apni amar amader achhi ache achhe nei jol bari barite sahajyo sahajya dorkar lagbe
        khabar nouka manush jon ekhane okhane kintu kothay kore korun dao bondi bhashche bonna bonya taratari
        aschhe jachhe paachhi pachchi theke hoyeche hoye keu kichu ekta""".split()),
}
ENGLISH = set("""the a an is are was were be been am and or but of in on at to for from with by this that these those
    we our us you your they their he she it its i my me please help need needed urgent people water food rescue
    stuck trapped house road near here there now today flood floods boat boats send all any no not have has can will
    family children camp relief district water level dam rain""".split())
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)?")
# words common to several romanised languages carry no signal
_SHARED = set.intersection(*ROMANISED.values()) | (ROMANISED["hi-Latn"] & ROMANISED["bn-Latn"]) \
    | (ROMANISED["hi-Latn"] & ROMANISED["ml-Latn"]) | (ROMANISED["ml-Latn"] & ROMANISED["bn-Latn"])


class LangResult(BaseModel):
    lang: str
    code_mixed: bool = False
    script: str  # "Latn", "Deva", "Beng", "Mlym", ... or "Zyyy" when unknown
    confidence: float  # share of evidence supporting ``lang``
    evidence: dict[str, int]


_SCRIPT_TAG = {"hi": "Deva", "bn": "Beng", "as": "Beng", "ml": "Mlym", "ta": "Taml", "te": "Telu", "kn": "Knda",
               "gu": "Gujr", "pa": "Guru", "or": "Orya"}


def _script_counts(text: str) -> Counter:
    counts: Counter = Counter()
    for ch in text:
        cp = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                counts["latin"] += 1
            continue
        for code, lo, hi in SCRIPTS:
            if lo <= cp <= hi:
                counts["as" if code == "bn" and ch in ASSAMESE_ONLY else code] += 1
                break
    if counts["as"]:  # any Assamese-only letter makes the whole Bengali-script text Assamese
        counts["as"] += counts.pop("bn", 0)
    return counts


def detect_language(text: str) -> LangResult:
    chars = _script_counts(text)
    words = _WORD.findall(text.lower())
    english = sum(w in ENGLISH for w in words)
    native = {k: v for k, v in chars.items() if k != "latin" and v}
    if native:
        lang, n = max(native.items(), key=lambda kv: kv[1])
        total = sum(native.values()) + chars["latin"]
        # English words inside native-script text -> code-mixed (brand names / place names alone don't count)
        return LangResult(lang=lang, code_mixed=english >= 2, script=_SCRIPT_TAG.get(lang, "Zyyy"),
                          confidence=round(n / max(total, 1), 3), evidence={**native, "english_words": english})
    scores = {lang: sum(w in lex and w not in _SHARED for w in words) for lang, lex in ROMANISED.items()}
    best, hits = max(scores.items(), key=lambda kv: kv[1])
    # two distinctive words, or one when the sentence has no English function words at all
    if hits >= 2 or (hits == 1 and english == 0):
        # code-mixed when English carries a real part of the sentence
        mixed = english >= 2 and english >= 0.2 * len(words)
        return LangResult(lang=best, code_mixed=mixed, script="Latn",
                          confidence=round(hits / max(hits + english, 1), 3), evidence={**scores, "english_words": english})
    if english or words:
        return LangResult(lang="en", script="Latn", confidence=round(english / max(len(words), 1), 3),
                          evidence={**scores, "english_words": english})
    return LangResult(lang="und", script="Zyyy", confidence=0.0, evidence={})


def detect_posts(posts: list) -> list:
    """Return copies of ``posts`` with ``language`` and ``code_mixed`` filled in."""
    out = []
    for p in posts:
        r = detect_language(p.text)
        out.append(p.model_copy(update={"language": r.lang, "code_mixed": r.code_mixed}))
    return out


if __name__ == "__main__":
    print(detect_language(" ".join(sys.argv[1:])).model_dump_json(indent=1))

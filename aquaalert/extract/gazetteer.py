"""Offline gazetteer (name -> lat, lon) used before any network lookup.

gazetteer_kerala.csv (Kerala) and gazetteer_india.csv (Assam, Odisha, West Bengal, Bihar, Uttar Pradesh) are generated
from Wikidata by scripts/build_gazetteer.py; each row keeps its Wikidata id. A row with blank coordinates is a river that
is recognised in text but has no usable point, so it never gets a pin.
"""
from __future__ import annotations

import csv
from pathlib import Path


def _load_csv(name: str) -> list[dict]:
    with Path(__file__).with_name(name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


_KERALA_ROWS = _load_csv("gazetteer_kerala.csv")
KERALA_PLACES: dict[str, tuple[float, float]] = {r["name"]: (float(r["lat"]), float(r["lon"])) for r in _KERALA_ROWS}

DISTRICTS: tuple[str, ...] = (
    "Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha", "Kottayam", "Idukki", "Ernakulam",
    "Thrissur", "Palakkad", "Malappuram", "Kozhikode", "Wayanad", "Kannur", "Kasaragod",
)

# place (lowercase) -> district
DISTRICT_OF: dict[str, str] = {r["name"]: r["district"] for r in _KERALA_ROWS}

# Native-script names -> canonical (lowercase) place. Malayalam and Bengali attach case endings to names
# ("ചെങ്ങന്നൂരിൽ" = in Chengannur), so these are matched as prefixes, see extract/entities.py.
NATIVE_ALIASES: dict[str, str] = {
    # Malayalam
    "ആലപ്പുഴ": "alappuzha", "ചെങ്ങന്നൂർ": "chengannur", "ആലുവ": "aluva", "കുട്ടനാട്": "kuttanad",
    "പന്തളം": "pandalam", "റാന്നി": "ranni", "ആറന്മുള": "aranmula", "ചാലക്കുടി": "chalakudy",
    "തൃശൂർ": "thrissur", "തൃശ്ശൂർ": "thrissur", "എറണാകുളം": "ernakulam", "കൊച്ചി": "kochi", "ഇടുക്കി": "idukki",
    "വയനാട്": "wayanad", "പത്തനംതിട്ട": "pathanamthitta", "കോട്ടയം": "kottayam", "തിരുവല്ല": "thiruvalla",
    "പറവൂർ": "north paravur", "മൂന്നാർ": "munnar", "കോഴിക്കോട്": "kozhikode", "പാലക്കാട്": "palakkad",
    "കണ്ണൂർ": "kannur", "മലപ്പുറം": "malappuram", "കൊല്ലം": "kollam", "തിരുവനന്തപുരം": "thiruvananthapuram",
    "എടത്വ": "edathua", "പെരുമ്പാവൂർ": "perumbavoor", "ചെറുതോണി": "cheruthoni", "മുല്ലപ്പെരിയാർ": "mullaperiyar",
    "കാക്കനാട്": "kakkanad", "വൈപ്പിൻ": "vypin",
    # Hindi (Devanagari)
    "अलप्पुझा": "alappuzha", "चेंगन्नूर": "chengannur", "आलुवा": "aluva", "कोच्चि": "kochi", "एर्नाकुलम": "ernakulam",
    "त्रिशूर": "thrissur", "इडुक्की": "idukki", "वायनाड": "wayanad", "कुट्टनाड": "kuttanad",
    "पथानामथिट्टा": "pathanamthitta", "कोट्टायम": "kottayam", "मुन्नार": "munnar", "चालाकुडी": "chalakudy",
    "कोझिकोड": "kozhikode", "पलक्कड़": "palakkad",
    # Bengali
    "আলাপ্পুঝা": "alappuzha", "কোচি": "kochi", "এর্নাকুলাম": "ernakulam", "ত্রিশুর": "thrissur", "আলুভা": "aluva",
    "চেঙ্গান্নুর": "chengannur", "ইডুক্কি": "idukki", "ওয়ানাড": "wayanad", "কুট্টানাড": "kuttanad",
    "পাথানামথিট্টা": "pathanamthitta", "কোট্টায়াম": "kottayam", "মুন্নার": "munnar",
}

# Malayalam chillu letters at the end of a name turn back into consonants before a case ending
# (ചെങ്ങന്നൂർ -> ചെങ്ങന്നൂരിൽ), so also match the stem with the base consonant.
# ർ can return as ര or റ (മൂന്നാർ -> മൂന്നാറിൽ).
_CHILLU = {"ൻ": ("ന",), "ർ": ("ര", "റ"), "ൽ": ("ല",), "ൾ": ("ള",), "ൺ": ("ണ",)}
for _alias, _place in list(NATIVE_ALIASES.items()):
    for _base in _CHILLU.get(_alias[-1], ()):
        NATIVE_ALIASES.setdefault(_alias[:-1] + _base, _place)
    # names ending in anusvara take -ത്ത് before an ending (എറണാകുളം -> എറണാകുളത്ത്). Matching the bare stem is
    # unsafe ("കൊല്ല" also starts "കൊല്ലപ്പെട്ടു", "was killed"), so the stem is kept with its "ത്ത".
    if _alias.endswith("ം"):
        NATIVE_ALIASES.setdefault(_alias[:-1] + "ത്ത", _place)


_INDIA_ROWS = _load_csv("gazetteer_india.csv")
INDIA_PLACES: dict[str, tuple[float, float]] = {
    r["name"]: (float(r["lat"]), float(r["lon"])) for r in _INDIA_ROWS if r["lat"] and r["name"] not in KERALA_PLACES}
ALL_PLACES: dict[str, tuple[float, float]] = {**INDIA_PLACES, **KERALA_PLACES}
RIVERS: frozenset[str] = frozenset(r["name"] for r in _INDIA_ROWS if r["kind"] == "river")
# Names recognised in text (including rivers without a pin), split by whether case can be ignored
PLACE_NAMES: frozenset[str] = frozenset(KERALA_PLACES) | {r["name"] for r in _INDIA_ROWS if r["case_sensitive"] == "0"}
CASED_NAMES: frozenset[str] = frozenset(r["name"] for r in _INDIA_ROWS if r["case_sensitive"] == "1")
# Devanagari / Bengali names of those places. Matched as whole words, see extract/entities.py.
INDIA_ALIASES: dict[str, str] = {a: r["name"] for r in _INDIA_ROWS for a in r["aliases"].split("|") if a}
for _r in _INDIA_ROWS:
    if _r["district"]:
        DISTRICT_OF.setdefault(_r["name"], _r["district"])

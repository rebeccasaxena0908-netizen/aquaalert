"""Small offline gazetteer of Kerala places (name -> lat, lon) used before any network lookup."""
from __future__ import annotations

KERALA_PLACES: dict[str, tuple[float, float]] = {
    "thiruvananthapuram": (8.5241, 76.9366),
    "kollam": (8.8932, 76.6141),
    "pathanamthitta": (9.2648, 76.7870),
    "alappuzha": (9.4981, 76.3388),
    "kottayam": (9.5916, 76.5222),
    "idukki": (9.8494, 76.9710),
    "ernakulam": (9.9816, 76.2999),
    "kochi": (9.9312, 76.2673),
    "thrissur": (10.5276, 76.2144),
    "palakkad": (10.7867, 76.6548),
    "malappuram": (11.0510, 76.0711),
    "kozhikode": (11.2588, 75.7804),
    "wayanad": (11.6854, 76.1320),
    "kannur": (11.8745, 75.3704),
    "kasaragod": (12.4996, 74.9869),
    "chengannur": (9.3158, 76.6150),
    "aluva": (10.1004, 76.3570),
    "kuttanad": (9.3800, 76.4200),
    "pandalam": (9.2243, 76.6770),
    "ranni": (9.3866, 76.7845),
    "aranmula": (9.3190, 76.6970),
    "chalakudy": (10.3070, 76.3340),
    "cheruthoni": (9.8460, 76.9760),
    "thiruvalla": (9.3840, 76.5740),
    "paravur": (10.1500, 76.2230),
    "edathua": (9.3560, 76.4650),
    "kakkanad": (10.0158, 76.3419),
    "munnar": (10.0889, 77.0595),
    "vythiri": (11.5500, 76.0400),
    "shoranur": (10.7630, 76.2740),
    "edappally": (10.0261, 76.3080),
    "vypin": (10.0000, 76.2300),
    "perumbavoor": (10.1130, 76.4750),
    "mullaperiyar": (9.5280, 77.1440),
}

DISTRICTS: tuple[str, ...] = (
    "Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha", "Kottayam", "Idukki", "Ernakulam",
    "Thrissur", "Palakkad", "Malappuram", "Kozhikode", "Wayanad", "Kannur", "Kasaragod",
)

# place (lowercase) -> district; district names and Kochi map to themselves / Ernakulam.
DISTRICT_OF: dict[str, str] = {
    **{d.lower(): d for d in DISTRICTS},
    "kochi": "Ernakulam", "chengannur": "Alappuzha", "aluva": "Ernakulam", "kuttanad": "Alappuzha",
    "pandalam": "Pathanamthitta", "ranni": "Pathanamthitta", "aranmula": "Pathanamthitta",
    "chalakudy": "Thrissur", "cheruthoni": "Idukki", "thiruvalla": "Pathanamthitta",
    "paravur": "Ernakulam", "edathua": "Alappuzha", "kakkanad": "Ernakulam", "munnar": "Idukki",
    "vythiri": "Wayanad", "shoranur": "Palakkad", "edappally": "Ernakulam", "vypin": "Ernakulam",
    "perumbavoor": "Ernakulam", "mullaperiyar": "Idukki",
}

# Native-script names -> canonical (lowercase) place. Malayalam and Bengali attach case endings to names
# ("ചെങ്ങന്നൂരിൽ" = in Chengannur), so these are matched as prefixes, see extract/entities.py.
NATIVE_ALIASES: dict[str, str] = {
    # Malayalam
    "ആലപ്പുഴ": "alappuzha", "ചെങ്ങന്നൂർ": "chengannur", "ആലുവ": "aluva", "കുട്ടനാട്": "kuttanad",
    "പന്തളം": "pandalam", "റാന്നി": "ranni", "ആറന്മുള": "aranmula", "ചാലക്കുടി": "chalakudy",
    "തൃശൂർ": "thrissur", "തൃശ്ശൂർ": "thrissur", "എറണാകുളം": "ernakulam", "കൊച്ചി": "kochi", "ഇടുക്കി": "idukki",
    "വയനാട്": "wayanad", "പത്തനംതിട്ട": "pathanamthitta", "കോട്ടയം": "kottayam", "തിരുവല്ല": "thiruvalla",
    "പറവൂർ": "paravur", "മൂന്നാർ": "munnar", "കോഴിക്കോട്": "kozhikode", "പാലക്കാട്": "palakkad",
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

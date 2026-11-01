import pytest

from aquaalert.extract.entities import extract_locations
from aquaalert.ingest.loader import load_posts
from aquaalert.language.detect import detect_language, detect_posts
from aquaalert.language.evaluate import per_language
from aquaalert.classify.zeroshot import KeywordZeroShot


@pytest.mark.parametrize("text,lang,mixed", [
    ("चेंगन्नूर में 15 लोग फंसे हैं", "hi", False),
    ("Kochi में flood बहुत serious है, please help us, हम stuck हैं", "hi", True),
    ("ചെങ്ങന്നൂരിൽ 20 പേർ കുടുങ്ങി കിടക്കുന്നു", "ml", False),
    ("ആലുവയിൽ water level rising fast, ഞങ്ങൾ terrace il ആണ്, please send boat", "ml", True),
    ("আলাপ্পুঝায় আমরা আটকে আছি", "bn", False),
    ("বানপানীত আমাৰ ঘৰ ডুবিল", "as", False),  # Assamese-only letter ৰ
    ("Hum chhat par phase hue hain, please help us now", "hi-Latn", True),
    ("Mera bhai Rahul Chengannur se laapata hai, kisi ko dikhe toh batao", "hi-Latn", False),
    ("Njangal veettil kudungi kidakkunnu, vellam kayari, pettennu sahayikkuka", "ml-Latn", False),
    ("Ami Alappuzha te ache, 6 jon manush bondi, nouka dorkar", "bn-Latn", False),
    ("Water level rising at Idukki dam, please send boats to the camp", "en", False),
])
def test_detect_language(text, lang, mixed):
    r = detect_language(text)
    assert (r.lang, r.code_mixed) == (lang, mixed), (text, r)
    assert 0 <= r.confidence <= 1


def test_undetermined_and_numbers_only():
    assert detect_language("").lang == "und"
    assert detect_language("9847012345 !!!").lang == "und"


def test_shared_words_do_not_decide():
    # "bhai" is Hindi and Bengali: alone it must not tip the language
    assert detect_language("bhai").lang == "en"


def test_detect_posts_and_pipeline(make_post, tmp_path):
    from aquaalert.extract.geocode import Geocoder
    from aquaalert.pipeline import process_posts

    p = detect_posts([make_post("ചെങ്ങന്നൂരിൽ ഞങ്ങൾ കുടുങ്ങി")])[0]
    assert p.language == "ml" and p.code_mixed is False
    out = process_posts([make_post("ചെങ്ങന്നൂരിൽ ഞങ്ങൾ കുടുങ്ങി")], KeywordZeroShot(),
                        Geocoder(cache_path=tmp_path / "g.json", online=False))[0]
    assert out.language == "ml" and out.extraction.district == "Alappuzha"


@pytest.mark.parametrize("text,places", [
    ("ചെങ്ങന്നൂരിൽ 20 പേർ കുടുങ്ങി", ["Chengannur"]),            # chillu ർ -> ര before the ending
    ("മൂന്നാറിൽ മണ്ണിടിച്ചിൽ", ["Munnar"]),                       # chillu ർ -> റ
    ("എറണാകുളത്ത് 20 കുടുംബങ്ങൾക്ക്", ["Ernakulam"]),           # anusvara ം -> ത്ത്
    ("ആലപ്പുഴയിലെ ക്യാമ്പിൽ", ["Alappuzha"]),
    ("कुट्टनाड में नाव चाहिए, आलुवा से भी", ["Kuttanad", "Aluva"]),
    ("আলাপ্পুঝায় আমরা আটকে আছি", ["Alappuzha"]),
    ("കൊല്ലപ്പെട്ടു എന്ന വാർത്ത", []),                            # "was killed" must not match Kollam
])
def test_native_script_place_names(text, places):
    assert extract_locations(text) == places


def test_multilingual_sample_and_eval(sample_path):
    ml = load_posts(sample_path.parent / "sample_posts_multilingual.csv")
    assert len(ml) == 44 and all(p.gold_language and p.gold_label for p in ml)
    assert {p.gold_language for p in ml} == {"hi", "hi-Latn", "ml", "ml-Latn", "bn", "bn-Latn"}
    res = per_language(load_posts(sample_path) + ml, {"keyword": KeywordZeroShot()})
    assert set(res) >= {"en", "hi", "ml", "bn"} and res["ml"]["n"] == 11
    assert all(res[l]["lang_id_accuracy"] == 1.0 for l in ("hi", "ml", "bn", "en"))
    assert {"acc_keyword", "acc_keyword+rules", "located"} <= set(res["hi"])

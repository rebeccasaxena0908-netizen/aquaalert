import json
import re
from datetime import datetime

import pytest

from aquaalert.extract import gazetteer
from aquaalert.extract.entities import (extract_locations, extract_people_count, extract_phones,
                                        extract_post)
from aquaalert.extract.gazetteer import _load_csv
from aquaalert.extract.geocode import Geocoder, in_bbox
from aquaalert.schemas import Post


def test_phones():
    assert extract_phones("call 9847012345 or +91 9895123456, not 12345") == ["9847012345", "9895123456"]
    assert extract_phones("no numbers") == []


def test_people_count():
    assert extract_people_count("12 people stranded") == 12
    assert extract_people_count("Family of 5 trapped") == 5
    assert extract_people_count("three kids on the roof") == 3
    assert extract_people_count("nothing here") is None


def test_locations():
    assert extract_locations("Stranded in chengannur near Aluva") == ["Chengannur", "Aluva"]
    assert extract_locations("stuck at Mannar Junction") == ["Mannar Junction"]  # unknown place kept
    assert extract_locations("Please help in Kerala") == []


def test_geocoder_offline_and_cache(tmp_path):
    cache = tmp_path / "geo.json"
    g = Geocoder(cache_path=cache, online=False)
    pt = g.geocode("Chengannur")
    assert pt and in_bbox(pt.lat, pt.lon)
    assert g.geocode("Nowhereville") is None
    assert json.loads(cache.read_text())["chengannur"] is not None
    assert "nowhereville" not in json.loads(cache.read_text())  # offline misses aren't cached
    assert Geocoder(cache_path=cache, online=False).geocode("chengannur") == pt  # served from cache


def test_geocoder_online_uses_cache(tmp_path, monkeypatch):
    calls = []
    g = Geocoder(cache_path=tmp_path / "g.json", online=True)
    monkeypatch.setattr(g, "_lookup_online", lambda p: calls.append(p) or (9.5, 76.5))
    assert g.geocode("Mannar Junction") and g.geocode("Mannar Junction")
    assert calls == ["Mannar Junction"]
    monkeypatch.setattr(g, "_lookup_online", lambda p: calls.append(p) or (40.0, -74.0))
    assert g.geocode("Elsewhere") is None  # outside the Kerala bbox is rejected


def test_extract_post(make_post, tmp_path):
    g = Geocoder(cache_path=tmp_path / "g.json", online=False)
    out = extract_post(make_post("8 people trapped at Aluva, call 9895123456"), g)
    ex = out.extraction
    assert ex.locations == ["Aluva"] and ex.geo_source == "Aluva" and ex.geo
    assert ex.people_count == 8 and ex.phones == ["9895123456"]
    assert extract_post(make_post("hello"), g).extraction.geo is None


def test_district_and_url(make_post, tmp_path, sample_path):
    g = Geocoder(cache_path=tmp_path / "g.json", online=False)
    assert extract_post(make_post("stuck in Chengannur"), g).extraction.district == "Alappuzha"
    assert extract_post(make_post("Kochi roads flooded"), g).extraction.district == "Ernakulam"
    assert extract_post(make_post("nothing"), g).extraction.district is None
    from aquaalert.ingest.loader import load_posts

    f = tmp_path / "u.csv"
    f.write_text("id,text,timestamp,url\n1,hello,2018-08-15 01:00,https://example.org/1\n2,hi,2018-08-15 02:00,\n")
    assert [p.url for p in load_posts(f)] == ["https://example.org/1", None]


@pytest.mark.parametrize("text, place, district", [
    ("Flood waters rising in Guwahati", "Guwahati", "Kamrup"), ("Dhemaji villages under water", "Dhemaji", "Dhemaji"),
    ("Waterlogging across Bhubaneswar", "Bhubaneswar", "Khordha"), ("Kanpur ghats closed", "Kanpur", "Kanpur Nagar"),
    ("Heavy rain in Lucknow", "Lucknow", "Lucknow"), ("Patna on alert", "Patna", "Patna"),
    ("Gandak overflows near Bagaha", "Bagaha", "West Champaran"),
    ("गुवाहाटी में बाढ़", "Guwahati", "Kamrup"), ("पटना में जलभराव", "Patna", "Patna"), ("भुवनेश्वर में बारिश", "Bhubaneswar", "Khordha"),
    ("কানপুরে বন্যা", "Kanpur", "Kanpur Nagar"), ("লখনউ শহরে জল", "Lucknow", "Lucknow"),
])
def test_other_states_get_pins(tmp_path, text, place, district):
    post = extract_post(Post(id="x", text=text, timestamp=datetime(2026, 11, 2)), Geocoder(cache_path=tmp_path / "g.json", online=False))
    assert post.extraction.geo_source == place and post.extraction.geo and post.extraction.district == district
    assert not in_bbox(post.extraction.geo.lat, post.extraction.geo.lon)  # outside Kerala


def test_gazetteer_rows_are_sourced_and_in_range():
    for row in _load_csv("gazetteer_india.csv"):
        assert re.fullmatch(r"Q\d+", row["wikidata"]), row["name"]
        if row["lat"]:
            assert 17 < float(row["lat"]) < 31 and 77 < float(row["lon"]) < 97, row["name"]
        else:
            assert row["kind"] == "river", row["name"]


def test_river_is_a_fallback_pin_and_ganga_has_none(tmp_path):
    geo = Geocoder(cache_path=tmp_path / "g.json", online=False)
    make = lambda t: extract_post(Post(id="x", text=t, timestamp=datetime(2026, 11, 2)), geo).extraction  # noqa: E731
    assert make("Alert along the Sankosh River").geo_source == "Sankosh"
    assert make("Brahmaputra rising near Dhubri").geo_source == "Dhubri"
    ganga = make("Ganga above danger mark")
    assert "Ganga" in ganga.locations and ganga.geo is None
    assert make("गंगा का जलस्तर बढ़ा").locations == ["Ganga"] and make("Ganges in spate").locations == ["Ganges"]


@pytest.mark.parametrize("text", ["paani aa gaya hai", "hum basti mein hain", "sambhal ke rehna", "ye banda kaun hai", "पूरा गया बस्ती में", "संभल कर चलो"])
def test_everyday_words_are_not_places(text):
    assert extract_locations(text) == []


def test_everyday_word_places_match_when_capitalised():
    assert extract_locations("Flood in Gaya and Mau") == ["Gaya", "Mau"]


def test_same_name_different_state_is_left_out():
    assert "lakhimpur" not in gazetteer.ALL_PLACES and "lakhimpur kheri" in gazetteer.ALL_PLACES


def test_every_kerala_row_is_sourced_and_inside_kerala():
    rows = _load_csv("gazetteer_kerala.csv")
    assert len({r["wikidata"] for r in rows}) == 34
    for row in rows:
        assert re.fullmatch(r"Q\d+", row["wikidata"]) and in_bbox(float(row["lat"]), float(row["lon"])), row["name"]
        assert row["district"] in gazetteer.DISTRICTS, row["name"]


def test_paravur_means_north_paravur_in_ernakulam(tmp_path):
    geo = Geocoder(cache_path=tmp_path / "g.json", online=False)
    want = (10.15, 76.23)
    for text in ("SOS at North Paravur", "SOS at Paravur", "North Parur flooded", "പറവൂരിൽ വെള്ളം"):
        ex = extract_post(Post(id="x", text=text, timestamp=datetime(2018, 8, 15)), geo).extraction
        assert (round(ex.geo.lat, 2), round(ex.geo.lon, 2)) == want and ex.district == "Ernakulam", text
    assert extract_locations("North Paravur") == ["North Paravur"]


@pytest.mark.parametrize("place, district, lat, lon", [
    ("cheruthoni", "Idukki", 9.8453, 76.9669), ("mullaperiyar", "Idukki", 9.5286, 77.1442),
    ("ranni", "Pathanamthitta", 9.3667, 76.7667), ("thiruvalla", "Pathanamthitta", 9.385, 76.575),
    ("wayanad", "Wayanad", 11.6264, 76.0889), ("vypin", "Ernakulam", 10.1024, 76.2044),
    ("pandalam", "Pathanamthitta", 9.2166, 76.6645),
])
def test_corrected_kerala_pins(tmp_path, place, district, lat, lon):
    pt = Geocoder(cache_path=tmp_path / "g.json", online=False).geocode(place)
    assert (round(pt.lat, 4), round(pt.lon, 4)) == (lat, lon) and gazetteer.DISTRICT_OF[place] == district

import json

from aquaalert.extract.entities import (extract_locations, extract_people_count, extract_phones,
                                        extract_post)
from aquaalert.extract.geocode import Geocoder, in_bbox


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

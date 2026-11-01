from datetime import datetime

import numpy as np
import pytest

from aquaalert.match.geo import haversine_km
from aquaalert.match.matcher import hashing_encoder, match_posts
from aquaalert.schemas import Extraction, GeoPoint, Label, Post

PLACES = {"alappuzha": (9.4981, 76.3388), "chengannur": (9.3158, 76.6150), "kasaragod": (12.4996, 74.9869)}


def post(id: str, text: str, label: Label, place: str) -> Post:
    lat, lon = PLACES[place]
    return Post(id=id, text=text, timestamp=datetime(2018, 8, 15), label=label, label_score=0.9,
                extraction=Extraction(geo=GeoPoint(lat=lat, lon=lon), geo_source=place.title()))


def test_haversine():
    assert haversine_km(10, 76, 10, 76) == 0
    assert haversine_km(9.9312, 76.2673, 10.5276, 76.2144) == pytest.approx(66.5, abs=2)  # Kochi-Thrissur
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.2, abs=0.5)


def test_hashing_encoder_is_normalised():
    e = hashing_encoder(["food and water", "", "food and water"])
    assert np.linalg.norm(e[0]) == pytest.approx(1) and np.linalg.norm(e[1]) == 0
    assert e[0] @ e[2] == pytest.approx(1)


@pytest.fixture
def posts():
    return [
        post("n1", "need food and drinking water packets", Label.NEED_SUPPLIES, "alappuzha"),
        post("o_food", "we can send food and drinking water packets", Label.OFFER_HELP, "chengannur"),
        post("o_far", "we can send food and drinking water packets", Label.OFFER_HELP, "kasaragod"),
        post("o_boat", "volunteers with boats available", Label.OFFER_HELP, "alappuzha"),
        post("x", "road collapsed", Label.INFRASTRUCTURE_DAMAGE, "alappuzha"),
    ]


def test_match_prefers_similar_nearby_offer_and_drops_far_ones(posts):
    ms = match_posts(posts, hashing_encoder, max_km=50)
    assert [m.offer_id for m in ms][0] == "o_food"
    assert "o_far" not in {m.offer_id for m in ms}  # ~370 km away
    top = ms[0]
    assert top.need_id == "n1" and 0 < top.distance_km < 50
    assert top.score == pytest.approx(0.6 * top.similarity + 0.4 * top.proximity)
    assert top.proximity == pytest.approx(1 - top.distance_km / 50)


def test_alpha_shifts_weight_to_distance(posts):
    text_first = match_posts(posts, hashing_encoder, alpha=1.0, min_score=0, max_km=500)
    near_first = match_posts(posts, hashing_encoder, alpha=0.0, min_score=0, max_km=500)
    assert text_first[0].offer_id in {"o_food"}
    assert near_first[0].distance_km == min(m.distance_km for m in near_first)


def test_top_k_and_one_to_one():
    ps = [post("n1", "need food", Label.NEED_SUPPLIES, "alappuzha"),
          post("n2", "need food", Label.RESCUE_NEEDED, "alappuzha"),
          post("o1", "food available", Label.OFFER_HELP, "alappuzha"),
          post("o2", "food available", Label.OFFER_HELP, "chengannur")]
    assert len(match_posts(ps, hashing_encoder, top_k=1, min_score=0)) == 2  # one per need
    ms = match_posts(ps, hashing_encoder, one_to_one=True, min_score=0)
    assert len({m.offer_id for m in ms}) == len(ms) == 2 and len({m.need_id for m in ms}) == 2


def test_no_needs_or_offers_or_geo(posts):
    assert match_posts([p for p in posts if p.label != Label.OFFER_HELP], hashing_encoder) == []
    assert match_posts([p for p in posts if p.label == Label.OFFER_HELP], hashing_encoder) == []
    no_geo = [p.model_copy(update={"extraction": None}) for p in posts]
    assert match_posts(no_geo, hashing_encoder) == []

from datetime import date, datetime

import pytest

from aquaalert.fuse.sensors import SensorReading, load_sensors
from aquaalert.fuse.warnings import RAIN_MM, compute_warnings, signal_counts
from aquaalert.schemas import Extraction, GeoPoint, Label, Post

DAY = date(2018, 8, 15)


def post(id, label=Label.RESCUE_NEEDED, district="Alappuzha", day=DAY, misinfo=0.0):
    return Post(id=id, text="x", timestamp=datetime(day.year, day.month, day.day, 10), label=label,
                misinfo_score=misinfo, extraction=Extraction(geo=GeoPoint(lat=9.5, lon=76.3), district=district))


def reading(district="Alappuzha", rain=None, level=None, danger=None, day=DAY):
    return {(district, day): SensorReading(district=district, date=day, rainfall_mm=rain, river_level_m=level,
                                           danger_level_m=danger, river="Pamba" if level else None)}


def test_sample_sensor_files_load():
    r = load_sensors()
    assert len({d for d, _ in r}) == 14 and all(isinstance(k[1], date) for k in r)
    idk = r[("Idukki", DAY)]
    assert idk.rainfall_mm > RAIN_MM and idk.river == "Periyar" and 0 < idk.river_ratio


def test_loader_merges_files_and_validates(tmp_path):
    (tmp_path / "rain.csv").write_text("district,date,rainfall_mm\nalappuzha,2018-08-15,120\nAlappuzha,2018-08-15,140\n")
    (tmp_path / "river.csv").write_text("district,date,river,river_level_m,danger_level_m\n"
                                        "Alappuzha,2018-08-15,Pamba,2.0,2.5\nAlappuzha,2018-08-15,Achankovil,3.0,2.8\n")
    r = load_sensors(tmp_path)[("Alappuzha", DAY)]
    assert r.rainfall_mm == 140  # highest gauge
    assert r.river == "Achankovil" and r.river_ratio == pytest.approx(3.0 / 2.8)  # gauge closest to danger
    (tmp_path / "bad.csv").write_text("district,date,river_level_m\nAlappuzha,2018-08-15,2\n")
    with pytest.raises(ValueError, match="danger_level_m"):
        load_sensors(tmp_path)


def test_signal_counts_exclude_doubtful_and_other_labels():
    posts = [post("a"), post("b", Label.INFRASTRUCTURE_DAMAGE), post("c", misinfo=0.7), post("d", Label.OFFER_HELP),
             post("e", district=None)]
    counts, ids = signal_counts(posts)
    assert counts[("Alappuzha", DAY)] == 2 and ids[("Alappuzha", DAY)] == ["a", "b"]


def spike(n=4, **kw):
    return [post(f"p{i}", **kw) for i in range(n)]


def test_warning_needs_spike_and_sensor():
    w = compute_warnings(spike(), reading(rain=150), day=DAY)[0]
    assert w.level == "warning" and w.spike and w.sensor_trigger
    assert any("rainfall 150 mm" in r for r in w.reasons) and any("4 rescue/damage posts" in r for r in w.reasons)
    assert compute_warnings(spike(), reading(rain=40), day=DAY)[0].level == "watch"       # posts only
    assert compute_warnings(spike(1), reading(rain=210), day=DAY)[0].level == "watch"    # sensor only
    assert compute_warnings([], reading(rain=10), day=DAY)[0].level == "normal"
    river = compute_warnings(spike(), reading(level=2.6, danger=2.5), day=DAY)[0]
    assert river.level == "warning" and any("above danger level" in r for r in river.reasons)
    near = compute_warnings([], reading(level=2.3, danger=2.5), day=DAY)[0]
    assert near.level == "watch" and any("92% of its danger level" in r for r in near.reasons)


def test_spike_is_relative_to_previous_days():
    history = [post(f"h{d}{i}", day=date(2018, 8, d)) for d in (12, 13, 14) for i in range(3)]  # 3/day before
    w = compute_warnings(history + spike(4), reading(rain=150), day=DAY)[0]
    assert not w.spike and w.level == "watch" and w.baseline == 3.0  # 4 < 2 x (3 + 1)
    assert compute_warnings(history + spike(8), reading(rain=150), day=DAY)[0].level == "warning"


def test_doubtful_posts_cannot_trigger_a_warning():
    rumours = spike(6, misinfo=0.9)
    assert compute_warnings(rumours, reading(rain=200), day=DAY)[0].level == "watch"


def test_sample_data_end_to_end():
    from aquaalert.classify.zeroshot import KeywordZeroShot
    from aquaalert.credibility.evaluate import scored_sample

    posts = scored_sample(KeywordZeroShot())
    ws = compute_warnings(posts, day=DAY)
    assert {w.district for w in ws if w.level == "warning"} >= {"Alappuzha", "Ernakulam"}
    assert ws[0].level == "warning" and all(w.reasons for w in ws if w.level != "normal")
    assert compute_warnings(posts, {}, day=DAY)[0].reasons[-1] == "no sensor data for this district-day"

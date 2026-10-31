import numpy as np
import pytest

from aquaalert.classify.metrics import (LARGE_IDX, bootstrap_ci, confusion, f1_from_confusion,
                                        format_table, full_report, inverse_frequency_weights)
from aquaalert.schemas import LABELS


def test_perfect_and_empty_classes():
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    f1 = f1_from_confusion(confusion(y, y))
    assert f1.tolist() == [1.0] * 7
    assert f1_from_confusion(confusion(np.array([0, 0]), np.array([0, 0])))[1] == 0.0


def test_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(1)
    yt, yp = rng.integers(0, 7, 500), rng.integers(0, 7, 500)
    ours = f1_from_confusion(confusion(yt, yp))
    assert ours == pytest.approx(f1_score(yt, yp, average=None, labels=range(7)))


def test_large5_excludes_small_classes():
    assert [LABELS[i] for i in LARGE_IDX] == ["infrastructure_damage", "need_supplies", "offer_help",
                                             "info_update", "irrelevant"]
    yt = np.array([0, 4] * 50 + [1, 2, 3, 5, 6] * 20)
    rep = full_report(yt.tolist(), yt.tolist(), n_boot=20)
    assert rep["macro_f1_all7"] == rep["macro_f1_large5"] == 1.0
    yp = np.where(np.isin(yt, [0, 4]), 6, yt)  # break only the two small classes
    rep = full_report(yt.tolist(), yp.tolist(), n_boot=20)
    assert rep["macro_f1_large5"] > rep["macro_f1_all7"]


def test_bootstrap_ci_brackets_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 7, 800)
    yp = np.where(rng.random(800) < 0.7, yt, rng.integers(0, 7, 800))
    rep = full_report(yt.tolist(), yp.tolist(), n_boot=200)
    lo, hi = rep["ci95"]["macro_f1_all7"]
    assert lo < rep["macro_f1_all7"] < hi
    assert bootstrap_ci(yt, yp, 50, seed=3) == bootstrap_ci(yt, yp, 50, seed=3)
    assert "macro-F1 (all 7)" in format_table({"m": rep})


def test_inverse_frequency_weights():
    w = inverse_frequency_weights([0] * 90 + [1] * 10)
    assert w[1] == pytest.approx(9 * w[0]) and w[2] == 0.0

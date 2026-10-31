"""Classification metrics: per-class F1, macro-F1 variants and a bootstrap CI (numpy only)."""
from __future__ import annotations

import numpy as np

from aquaalert.schemas import LABELS

K = len(LABELS)
# Classes with too few HumAID examples for a stable F1 (86 / 250 train, 22 / 72 test).
SMALL_CLASSES = ("rescue_needed", "missing_person")
LARGE_IDX = [i for i, l in enumerate(LABELS) if l not in SMALL_CLASSES]


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.bincount(y_true * K + y_pred, minlength=K * K).reshape(K, K)


def f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class F1; a class with no true and no predicted examples scores 0."""
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)


def summarize(f1: np.ndarray) -> dict[str, float]:
    return {"macro_f1_all7": float(f1.mean()), "macro_f1_large5": float(f1[LARGE_IDX].mean())}


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, list[float]]:
    """Percentile bootstrap CI for both macro-F1 variants, resampling test examples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    all7, large5 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = summarize(f1_from_confusion(confusion(y_true[idx], y_pred[idx])))
        all7.append(s["macro_f1_all7"])
        large5.append(s["macro_f1_large5"])
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {
        "macro_f1_all7": [float(v) for v in np.percentile(all7, q)],
        "macro_f1_large5": [float(v) for v in np.percentile(large5, q)],
    }


def full_report(y_true: list[int], y_pred: list[int], n_boot: int = 1000, seed: int = 0) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion(yt, yp)
    f1 = f1_from_confusion(cm)
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        **summarize(f1),
        "ci95": bootstrap_ci(yt, yp, n_boot, seed),
        "per_class_f1": {l: float(f1[i]) for i, l in enumerate(LABELS)},
        "per_class_support": {l: int(cm[i].sum()) for i, l in enumerate(LABELS)},
    }


def inverse_frequency_weights(y_train: list[int]) -> list[float]:
    """w_c = N / (K * n_c); a class absent from training gets weight 0."""
    counts = np.bincount(np.asarray(y_train), minlength=K).astype(float)
    return [float(len(y_train) / (K * c)) if c else 0.0 for c in counts]


def format_table(reports: dict[str, dict]) -> str:
    names = list(reports)
    lines = [f"{'class':24s}{'support':>8s}" + "".join(f"{n:>18s}" for n in names)]
    first = next(iter(reports.values()))
    for l in LABELS:
        lines.append(f"{l:24s}{first['per_class_support'][l]:>8d}" + "".join(f"{r['per_class_f1'][l]:>18.3f}" for r in reports.values()))
    for key, title in (("macro_f1_all7", "macro-F1 (all 7)"), ("macro_f1_large5", "macro-F1 (large 5)")):
        lines.append(f"{title:24s}{'':>8s}" + "".join(
            f"{r[key]:>9.3f} [{r['ci95'][key][0]:.3f},{r['ci95'][key][1]:.3f}]".rjust(18) for r in reports.values()))
    return "\n".join(lines)

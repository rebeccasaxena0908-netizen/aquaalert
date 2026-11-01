"""Compare the English-only and romanised MuRIL models on HumAID test and the hand-written multilingual posts.

    python scripts/compare_models.py [--out models/comparison_romanised.json]

Uses the existing metrics code (classify/evaluate.compare, language/evaluate.per_language), then prints the numbers
Kaggle saved alongside the romanised model (metrics.json, language_eval.json) and the figures reported from the
Kaggle run, flagging every difference instead of hiding it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aquaalert.classify.evaluate import compare  # noqa: E402
from aquaalert.classify.model import TransformerClassifier  # noqa: E402
from aquaalert.classify.zeroshot import KeywordZeroShot  # noqa: E402
from aquaalert.config import get_settings  # noqa: E402
from aquaalert.language.evaluate import load_eval_posts, per_language  # noqa: E402

# Reported from the Kaggle run of notebooks/train_kaggle.ipynb (romanised model)
REPORTED = {"macro_f1_all7": 0.736, "macro_f1_large5": 0.763,
            "lang": {"hi": 1.00, "bn": 1.00, "ml": 0.73, "hi-Latn": 0.67, "ml-Latn": 0.86, "bn-Latn": 0.50}}
LANG_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali", "ml": "Malayalam", "hi-Latn": "Hinglish",
              "ml-Latn": "Manglish", "bn-Latn": "Banglish"}
TOL = 0.0051  # numbers were reported to 2-3 decimals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "models" / "comparison_romanised.json"))
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    s = get_settings()
    models = {"muril": TransformerClassifier(s.model_dir), "muril_romanised": TransformerClassifier(s.romanised_dir)}

    print("== HumAID test")
    humaid = compare({**models, "keyword": KeywordZeroShot()}, str(ROOT / "data/raw/humaid/test.csv"), "humaid", a.n_boot)
    print("\n== hand-written multilingual posts + English sample (accuracy)")
    lang = per_language(load_eval_posts(), models)
    for code, r in lang.items():
        print(f"  {LANG_NAMES.get(code, code):10s} n={r['n']:2d}  muril {r['acc_muril']:.2f} (+rules {r['acc_muril+rules']:.2f})"
              f"  romanised {r['acc_muril_romanised']:.2f} (+rules {r['acc_muril_romanised+rules']:.2f})")

    print("\n== check against the Kaggle run")
    kaggle_metrics = json.loads((s.romanised_dir / "metrics.json").read_text())
    kaggle_lang_path = s.romanised_dir / "language_eval.json"
    kaggle_lang = json.loads(kaggle_lang_path.read_text()) if kaggle_lang_path.exists() else {}
    local = humaid["models"]["muril_romanised"]
    saved = kaggle_metrics["models"]["trained"]
    mismatches = []
    for key in ("macro_f1_all7", "macro_f1_large5"):
        row = f"  {key:16s} local {local[key]:.4f} | Kaggle metrics.json {saved[key]:.4f} | reported {REPORTED[key]:.3f}"
        if abs(local[key] - saved[key]) > 1e-4 or abs(local[key] - REPORTED[key]) > TOL:
            mismatches.append(row)
            row += "  <-- MISMATCH"
        print(row)
    for code, rep in REPORTED["lang"].items():
        loc, loc_r = lang[code]["acc_muril_romanised"], lang[code]["acc_muril_romanised+rules"]
        k = kaggle_lang.get(code, {})
        k_plain, k_rules = k.get("acc_muril"), k.get("acc_muril+rules")  # on Kaggle the model was called "muril"
        row = (f"  {LANG_NAMES[code]:10s} local {loc:.2f} (+rules {loc_r:.2f}) | Kaggle file "
               f"{k_plain if k_plain is not None else '-'} (+rules {k_rules if k_rules is not None else '-'}) | reported {rep:.2f}")
        if min(abs(loc - rep), abs(loc_r - rep)) > TOL or (k_plain is not None and abs(k_plain - loc) > 1e-6):
            mismatches.append(row)
            row += "  <-- MISMATCH"
        print(row)
    print(f"\n{len(mismatches)} mismatch(es)")
    Path(a.out).write_text(json.dumps({"humaid": humaid, "languages": lang, "kaggle_metrics": saved,
                                       "kaggle_language_eval": kaggle_lang, "reported": REPORTED,
                                       "mismatches": mismatches}, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

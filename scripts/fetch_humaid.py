"""Download the HumAID HuggingFace mirror (QCRI/HumAID-all) to data/raw/humaid/{train,dev,test}.csv.

Reads the repo's parquet files directly (the repo's loading script fails split verification
in recent `datasets` versions).

    python scripts/fetch_humaid.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP resumes; the Xet client failed after 72 min on 2026-11-03

import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "humaid"
FILES = {"train": "train", "validation": "dev", "test": "test"}


def download(filename: str, attempts: int = 4) -> str:
    for attempt in range(1, attempts + 1):
        try:
            return hf_hub_download("QCRI/HumAID-all", filename, repo_type="dataset")
        except (OSError, RuntimeError) as exc:
            if attempt == attempts:
                raise
            print(f"{filename}: {type(exc).__name__}, retrying ({attempt}/{attempts})")
            time.sleep(15 * attempt)
    raise AssertionError("unreachable")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split, name in FILES.items():
        path = download(f"data/{split}-00000-of-00001.parquet")
        df = pd.read_parquet(path)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(df), list(df.columns), sorted(df["class_label"].unique()))


if __name__ == "__main__":
    main()

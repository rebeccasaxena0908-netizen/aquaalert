"""Download the HumAID HuggingFace mirror (QCRI/HumAID-all) to data/raw/humaid/{train,dev,test}.csv.

Reads the repo's parquet files directly (the repo's loading script fails split verification
in recent `datasets` versions).

    python scripts/fetch_humaid.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "humaid"
FILES = {"train": "train", "validation": "dev", "test": "test"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split, name in FILES.items():
        path = hf_hub_download("QCRI/HumAID-all", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(path)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(df), list(df.columns), sorted(df["class_label"].unique()))


if __name__ == "__main__":
    main()

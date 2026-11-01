"""Load posts from CSV / JSON / JSONL files into Post objects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aquaalert.classify.label_map import map_label
from aquaalert.schemas import Label, Post

TEXT_COLUMNS = ("text", "tweet_text", "content", "full_text")
TIME_COLUMNS = ("timestamp", "created_at", "tweet_time", "date")
ID_COLUMNS = ("id", "tweet_id")
URL_COLUMNS = ("url", "link", "tweet_url")
LABEL_COLUMNS = ("label", "class_label", "informativeness")


def _first(columns: list[str], options: tuple[str, ...]) -> str | None:
    return next((c for c in options if c in columns), None)


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if suffix == ".json":
        return pd.DataFrame(json.loads(path.read_text()))
    raise ValueError(f"Unsupported dataset format: {suffix}")


def load_posts(
    path: str | Path,
    source: str | None = None,
    label_dataset: str | None = None,
    limit: int | None = None,
) -> list[Post]:
    """Load a dataset file. Posts are returned sorted by timestamp.

    Column names are auto-detected (see the *_COLUMNS tuples). Raw labels, if present,
    are mapped through classify/label_map.py into ``gold_label``.
    """
    path = Path(path)
    df = _read_frame(path)
    cols = list(df.columns)
    text_col, time_col = _first(cols, TEXT_COLUMNS), _first(cols, TIME_COLUMNS)
    if text_col is None or time_col is None:
        raise ValueError(f"{path} needs a text column {TEXT_COLUMNS} and time column {TIME_COLUMNS}")
    id_col, label_col = _first(cols, ID_COLUMNS), _first(cols, LABEL_COLUMNS)
    url_col = _first(cols, URL_COLUMNS)

    posts: list[Post] = []
    for i, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text:
            continue
        ts = pd.to_datetime(row[time_col], errors="coerce")
        if pd.isna(ts):
            continue
        posts.append(
            Post(
                id=str(row[id_col]) if id_col else f"{path.stem}-{i}",
                text=text,
                timestamp=ts.to_pydatetime(),
                source=source or str(row.get("source") or path.stem),
                author=(str(row["author"]) or None) if "author" in cols else None,
                url=(str(row[url_col]).strip() or None) if url_col else None,
                gold_label=map_label(str(row[label_col]), label_dataset, text) if label_col else None,
            )
        )
    posts.sort(key=lambda p: p.timestamp)
    return posts[:limit] if limit else posts


def load_labeled(path: str | Path, label_dataset: str | None = None) -> list[tuple[str, Label]]:
    """Load (text, label) pairs from a file with no timestamps (e.g. a HumAID split).

    Rows with empty text or an unknown/dropped label are skipped.
    """
    df = _read_frame(Path(path))
    cols = list(df.columns)
    text_col, label_col = _first(cols, TEXT_COLUMNS), _first(cols, LABEL_COLUMNS)
    if text_col is None or label_col is None:
        raise ValueError(f"{path} needs a text column {TEXT_COLUMNS} and label column {LABEL_COLUMNS}")
    pairs = []
    for text, raw in zip(df[text_col], df[label_col], strict=True):
        text = str(text).strip()
        label = map_label(str(raw), label_dataset, text)
        if text and label:
            pairs.append((text, label))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a dataset file")
    parser.add_argument("path")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    for post in load_posts(args.path, limit=args.limit):
        print(post.model_dump_json())


if __name__ == "__main__":
    main()

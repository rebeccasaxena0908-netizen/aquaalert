"""Download the chatbot's guideline documents listed in data/docs/sources.json into data/docs/.

    python scripts/fetch_docs.py

Each download is checked (PDF magic bytes / HTML title) before it is saved, and retried on a dropped connection.
Already-saved files are skipped, so it is safe to run again. Links were verified on 2026-11-01.
"""
from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.request
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def fetch(url: str, attempts: int = 4, wait: float = 10.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "aquaalert-docs-fetch"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except (OSError, http.client.HTTPException):  # dropped connections: the government sites are slow at times
            if attempt == attempts:
                raise
            time.sleep(wait * attempt)
    raise AssertionError("unreachable")


def main() -> None:
    sources = json.loads((DOCS / "sources.json").read_text())
    failed = 0
    for name, info in sources.items():
        dest = DOCS / name
        if dest.exists():
            print(f"exists   {name}")
            continue
        try:
            data = fetch(info["url"])
        except Exception as exc:
            print(f"FAILED   {name}: {exc}")
            failed += 1
            continue
        ok = data.startswith(b"%PDF") if name.endswith(".pdf") else b"<title" in data[:20000].lower()
        if not ok:
            print(f"FAILED   {name}: unexpected content from {info['url']}")
            failed += 1
            continue
        dest.write_bytes(data)
        print(f"saved    {name} ({len(data) // 1024} KB)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

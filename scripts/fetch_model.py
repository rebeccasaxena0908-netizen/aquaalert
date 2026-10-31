"""Download the trained model from the private HF Hub repo `aquaalert-muril` into models/muril/.

Reads HF_TOKEN from .env (see .env.example). The repo is `<your-hf-username>/aquaalert-muril`,
resolved from the token; set AQUAALERT_HF_REPO to override with a full `user/name` id.

    python scripts/fetch_model.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
REPO_NAME = "aquaalert-muril"


def resolve_repo_id(token: str, whoami=None) -> str:
    override = os.getenv("AQUAALERT_HF_REPO")
    if override:
        return override
    if whoami is None:
        from huggingface_hub import HfApi

        whoami = HfApi().whoami
    return f"{whoami(token=token)['name']}/{REPO_NAME}"


def main() -> None:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Add it to .env (see .env.example).")

    from huggingface_hub import snapshot_download

    repo_id = resolve_repo_id(token)
    dest = ROOT / "models" / "muril"
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=dest, token=token)
    print(f"Downloaded {repo_id} -> {dest}")


if __name__ == "__main__":
    main()

"""Download a trained model from a private HF Hub repo into models/.

Reads HF_TOKEN from .env (see .env.example). The token's account must have access to the repo. An interrupted download
resumes when run again.

    python scripts/fetch_model.py                                   # <you>/aquaalert-muril -> models/muril
    python scripts/fetch_model.py --repo <user>/aquaalert-muril-romanised --out models/muril_romanised

Without --repo the repo is `<your-hf-username>/aquaalert-muril` (resolved from the token), or AQUAALERT_HF_REPO if set.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
REPO_NAME = "aquaalert-muril"
ATTEMPTS = 4


def resolve_repo_id(token: str, whoami=None, repo: str | None = None) -> str:
    override = repo or os.getenv("AQUAALERT_HF_REPO")
    if override:
        return override
    if whoami is None:
        from huggingface_hub import HfApi

        whoami = HfApi().whoami
    return f"{whoami(token=token)['name']}/{REPO_NAME}"


def resolve_out(out: str | None) -> Path:
    if not out:
        return ROOT / "models" / "muril"
    path = Path(out)
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None, help="full HF repo id (default: AQUAALERT_HF_REPO or <you>/aquaalert-muril)")
    ap.add_argument("--out", default=None, help="target directory (default: models/muril)")
    a = ap.parse_args(argv)
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Add it to .env (see .env.example).")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP resumes; set before huggingface_hub is imported
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    repo_id = resolve_repo_id(token, repo=a.repo)
    dest = resolve_out(a.out)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            snapshot_download(repo_id=repo_id, repo_type="model", local_dir=dest, token=token)
            break
        except RepositoryNotFoundError:
            sys.exit(f"{repo_id}: not found, or this token has no access to it. The AquaAlert models are private: ask the "
                     "owner for access, or train your own (see README).")
        except (OSError, RuntimeError, HfHubHTTPError) as exc:
            if attempt == ATTEMPTS:
                raise
            print(f"{type(exc).__name__}, retrying ({attempt}/{ATTEMPTS})")
            time.sleep(15 * attempt)
    print(f"Downloaded {repo_id} -> {dest}")


if __name__ == "__main__":
    main()

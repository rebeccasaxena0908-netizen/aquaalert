import pytest

from scripts.fetch_model import resolve_repo_id


def test_repo_id_from_whoami(monkeypatch):
    monkeypatch.delenv("AQUAALERT_HF_REPO", raising=False)
    assert resolve_repo_id("tok", whoami=lambda token: {"name": "alice"}) == "alice/aquaalert-muril"


def test_repo_id_override(monkeypatch):
    monkeypatch.setenv("AQUAALERT_HF_REPO", "org/custom")
    assert resolve_repo_id("tok", whoami=lambda token: pytest.fail("should not call")) == "org/custom"

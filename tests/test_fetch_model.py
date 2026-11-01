import pytest

from scripts.fetch_model import resolve_repo_id


def test_repo_id_from_whoami(monkeypatch):
    monkeypatch.delenv("AQUAALERT_HF_REPO", raising=False)
    assert resolve_repo_id("tok", whoami=lambda token: {"name": "alice"}) == "alice/aquaalert-muril"


def test_repo_id_override(monkeypatch):
    monkeypatch.setenv("AQUAALERT_HF_REPO", "org/custom")
    assert resolve_repo_id("tok", whoami=lambda token: pytest.fail("should not call")) == "org/custom"


def test_repo_argument_beats_env_and_whoami(monkeypatch):
    monkeypatch.setenv("AQUAALERT_HF_REPO", "org/from-env")
    assert resolve_repo_id("tok", whoami=lambda token: {"name": "x"}, repo="user/aquaalert-muril-romanised") \
        == "user/aquaalert-muril-romanised"


def test_out_directory_defaults_and_relative_paths():
    from scripts.fetch_model import ROOT, resolve_out

    assert resolve_out(None) == ROOT / "models" / "muril"  # the English-only model is never overwritten by --out
    assert resolve_out("models/muril_romanised") == ROOT / "models" / "muril_romanised"
    assert str(resolve_out("/tmp/elsewhere")) == "/tmp/elsewhere"

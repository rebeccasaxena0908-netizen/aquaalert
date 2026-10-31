from datetime import datetime
from pathlib import Path

import pytest

from aquaalert.schemas import Post

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_posts.csv"


@pytest.fixture
def sample_path() -> Path:
    return SAMPLE


@pytest.fixture
def make_post():
    def _make(text: str, id: str = "p1", ts: str = "2018-08-15 06:00:00") -> Post:
        return Post(id=id, text=text, timestamp=datetime.fromisoformat(ts))

    return _make

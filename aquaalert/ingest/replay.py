"""Replay simulator: stream a historical dataset in time order to mimic a live feed."""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable, Iterator

from aquaalert.ingest.loader import load_posts
from aquaalert.schemas import Post


def replay(
    posts: Iterable[Post],
    speed: float = 60.0,
    max_delay: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[Post]:
    """Yield posts in timestamp order, waiting (event gap / speed) seconds between them.

    ``speed=60`` means one minute of event time passes per real second. ``speed<=0``
    disables waiting entirely. Individual gaps are capped at ``max_delay`` real seconds.
    """
    previous = None
    for post in sorted(posts, key=lambda p: p.timestamp):
        if previous is not None and speed > 0:
            gap = (post.timestamp - previous).total_seconds() / speed
            if gap > 0:
                sleep(min(gap, max_delay))
        previous = post.timestamp
        yield post


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a dataset as a live feed")
    parser.add_argument("path")
    parser.add_argument("--speed", type=float, default=600.0)
    args = parser.parse_args()
    for post in replay(load_posts(args.path), speed=args.speed):
        print(f"[{post.timestamp:%H:%M:%S}] {post.text}")


if __name__ == "__main__":
    main()

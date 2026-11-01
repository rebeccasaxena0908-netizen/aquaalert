"""CLI: python -m aquaalert.chatbot.ask "question" [--data data/sample_posts.csv]"""
from __future__ import annotations

import argparse

from aquaalert.chatbot.rag import brief_passages, default_chatbot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--data", default="data/sample_posts.csv", help="dataset used for the daily briefs")
    a = p.parse_args()

    from aquaalert.pipeline import run
    from aquaalert.summarize.brief import daily_briefs

    bot = default_chatbot()
    bot.kb.set_briefs(brief_passages(daily_briefs(run(a.data), bot.provider)))
    ans = bot.ask(a.question)
    print(ans.text)
    for c in ans.citations:
        print(f"  [{c.n}] {c.source}{f', p. {c.page}' if c.page else ''} (score {c.score})")


if __name__ == "__main__":
    main()

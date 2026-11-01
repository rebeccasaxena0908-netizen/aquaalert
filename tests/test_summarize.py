from datetime import datetime

import pytest

from aquaalert.classify.base import classify_posts
from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.extract.entities import extract_posts
from aquaalert.extract.geocode import Geocoder
from aquaalert.ingest.loader import load_posts
from aquaalert.summarize import brief as brief_mod
from aquaalert.summarize.brief import build_inputs, daily_briefs
from aquaalert.summarize.providers import (AnthropicProvider, OpenAIProvider, TemplateProvider,
                                           build_user_prompt, get_provider)


@pytest.fixture
def posts(sample_path, tmp_path):
    ps = classify_posts(load_posts(sample_path), KeywordZeroShot())
    return extract_posts(ps, Geocoder(cache_path=tmp_path / "g.json", online=False))


def test_districts_assigned(posts):
    by_id = {p.id: p.extraction.district for p in posts}
    assert by_id["s001"] == "Alappuzha"  # Chengannur
    assert by_id["s002"] == "Ernakulam"  # Aluva
    assert by_id["s008"] is None  # no location


def test_build_inputs_groups_ranks_and_redacts(posts):
    groups = build_inputs(posts)
    inp, shown = groups[("Alappuzha", datetime(2018, 8, 15).date())]
    assert all(p.label.value != "irrelevant" for p in shown)
    assert inp.posts[0].label == "rescue_needed"  # most urgent first
    assert inp.counts["rescue_needed"] >= 2
    prompt = build_user_prompt(inp)
    assert "9847012345" not in prompt and "<phone>" in prompt  # phone numbers never reach the LLM
    assert "<posts>" in prompt


def test_template_brief_cites_sources_offline(posts):
    briefs = daily_briefs(posts, TemplateProvider())
    assert briefs and briefs[0].counts.get("rescue_needed", 0) >= briefs[-1].counts.get("rescue_needed", 0)
    b = next(b for b in briefs if b.district == "Alappuzha")
    assert "rescue request" in b.summary and "9847012345" not in b.summary
    assert b.sources and all(f"[{s.id}]" in b.summary for s in b.sources)
    assert not b.cached


class FakeHTTP:
    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, []

    def anthropic(self, url, headers, payload):
        self.calls.append((url, headers, payload))
        return {"content": [{"type": "text", "text": self.reply}]}

    def openai(self, url, headers, payload):
        self.calls.append((url, headers, payload))
        return {"choices": [{"message": {"content": self.reply}}]}


def test_anthropic_provider_request_cache_and_citation_check(posts, tmp_path):
    first_id = next(iter(build_inputs(posts).values()))[0].posts[0].id
    http = FakeHTTP(f"People are trapped [{first_id}]. Invented claim [zzz99].")
    provider = AnthropicProvider("sk-test", http=http.anthropic)
    cache = tmp_path / "summaries.json"
    b1 = daily_briefs(posts, provider, cache)[0]
    url, headers, payload = http.calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "sk-test" and headers["anthropic-version"] == "2023-06-01"
    assert payload["model"] == "claude-haiku-5-5" and "untrusted" in payload["system"]
    assert b1.dropped_citations == ["zzz99"] and "zzz99" not in b1.summary
    assert [s.id for s in b1.sources] == [first_id]
    n_calls = len(http.calls)
    b2 = daily_briefs(posts, provider, cache)[0]
    assert len(http.calls) == n_calls and b2.cached and b2.summary == b1.summary  # served from disk cache


def test_uncited_llm_summary_falls_back_to_top_sources(posts, tmp_path):
    provider = AnthropicProvider("k", http=FakeHTTP("No citations here.").anthropic)
    b = daily_briefs(posts, provider, tmp_path / "c.json")[0]
    assert 1 <= len(b.sources) <= 3


def test_source_urls_propagate(posts, tmp_path):
    linked = [p.model_copy(update={"url": f"https://example.org/{p.id}"}) for p in posts]
    b = next(b for b in daily_briefs(linked, TemplateProvider()) if b.district == "Alappuzha")
    assert all(s.url == f"https://example.org/{s.id}" for s in b.sources)


def test_openai_provider_payload():
    http = FakeHTTP("ok [s1]")
    inp = brief_mod.BriefInput("Idukki", "2018-08-15", {"info_update": 1},
                               [brief_mod.BriefPost("s1", "info_update", "Dam open")])
    out = OpenAIProvider("k", base_url="http://llm.local/v1/", http=http.openai).summarize(inp)
    url, headers, payload = http.calls[0]
    assert out == "ok [s1]" and url == "http://llm.local/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k" and payload["messages"][0]["role"] == "system"


def test_get_provider_selection(monkeypatch):
    monkeypatch.setenv("AQUAALERT_LLM_PROVIDER", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(get_provider(), TemplateProvider)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert isinstance(get_provider(), OpenAIProvider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    assert isinstance(get_provider(), AnthropicProvider)  # Anthropic wins in auto
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_provider("anthropic")
    with pytest.raises(ValueError):
        get_provider("bogus")

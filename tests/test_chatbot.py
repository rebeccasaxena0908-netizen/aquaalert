import numpy as np
import pytest

from aquaalert.chatbot.docs import Passage, chunk_text, docs_fingerprint, load_documents
from aquaalert.chatbot.index import KnowledgeBase
from aquaalert.chatbot.rag import Chatbot, brief_passages, build_prompt
from aquaalert.match.matcher import hashing_encoder
from aquaalert.summarize.brief import DistrictBrief, SourceRef
from aquaalert.summarize.providers import AnthropicProvider, GeminiProvider, TemplateProvider
from tests.conftest import write_pdf

SAFETY = ("Move to higher ground immediately when a flood warning is issued. Do not walk or drive through "
          "flowing floodwater. Switch off electricity at the main switch before water enters the house.")
FOOD = ("Boil drinking water for at least ten minutes after the flood. Discard any food that has touched "
        "floodwater and clean utensils with disinfectant before use.")


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    write_pdf(d / "guide.pdf", [SAFETY, FOOD])
    (d / "notes.txt").write_text("Keep an emergency kit with torch, first aid, dry food and medicines ready at home.")
    (d / "ignore.png").write_bytes(b"x")
    return d


def counting_encoder():
    calls = []

    def enc(texts):
        calls.append(len(texts))
        return hashing_encoder(texts)

    enc.calls = calls
    return enc


def make_kb(docs, tmp_path, enc=hashing_encoder):
    return KnowledgeBase(docs, tmp_path / "cache", enc, "hash")


def test_chunk_text():
    text = " ".join(f"Sentence number {i} about flood safety measures." for i in range(60))
    chunks = chunk_text(text, size=300, overlap=60)
    assert len(chunks) > 3 and all(len(c) <= 420 for c in chunks)
    assert chunks[0].split()[-3:] == chunks[1].split()[:20][-3:] or set(chunks[0].split()) & set(chunks[1].split())
    assert chunk_text("tiny") == [] and chunk_text("") == []


def test_load_documents_reads_pdf_pages_and_txt(docs):
    ps = load_documents(docs)
    by_src = {(p.source, p.page) for p in ps}
    assert ("guide.pdf", 1) in by_src and ("guide.pdf", 2) in by_src and ("notes.txt", None) in by_src
    assert not any(p.source == "ignore.png" for p in ps)
    assert next(p for p in ps if p.page == 2).citation() == "guide.pdf, p. 2"
    assert "floodwater" in next(p for p in ps if p.page == 1).text


def test_corrupt_pdf_is_skipped(docs):
    (docs / "bad.pdf").write_bytes(b"not a pdf")
    assert {p.source for p in load_documents(docs)} == {"guide.pdf", "notes.txt"}


def test_search_ranks_relevant_passage_first(docs, tmp_path):
    kb = make_kb(docs, tmp_path)
    top, score = kb.search("how to make drinking water safe, boil it", k=3)[0]
    assert top.page == 2 and score > 0.2
    top, _ = kb.search("switch off electricity before water enters the house", k=1)[0]
    assert top.page == 1


def test_index_is_cached_and_rebuilt_when_docs_change(docs, tmp_path):
    enc = counting_encoder()
    make_kb(docs, tmp_path, enc)
    n_build = len(enc.calls)
    kb2 = make_kb(docs, tmp_path, enc)
    assert len(enc.calls) == n_build and kb2.has_guidelines  # loaded from disk, no re-embedding
    (docs / "more.txt").write_text("Store drinking water in clean covered containers before the monsoon arrives.")
    fp = docs_fingerprint(docs)
    kb3 = make_kb(docs, tmp_path, enc)
    assert len(enc.calls) > n_build and "more.txt" in fp
    assert any(p.source == "more.txt" for p in kb3.doc_passages)


def _brief():
    return DistrictBrief(district="Alappuzha", date="2018-08-15", counts={"rescue_needed": 2},
                         summary="Urgent: 2 rescue requests near Kuttanad, boats needed [s038].",
                         sources=[SourceRef(id="s038", label="rescue_needed", text="boats needed")],
                         provider="template", model="x")


def test_briefs_are_searchable_with_post_refs(docs, tmp_path):
    kb = make_kb(docs, tmp_path)
    ps = brief_passages([_brief()])
    assert ps[0].kind == "brief" and ps[0].ref == "posts s038" and "Alappuzha" in ps[0].source
    kb.set_briefs(ps)
    top, _ = kb.search("rescue requests boats Kuttanad Alappuzha", k=1)[0]
    assert top.kind == "brief"


def test_offline_fallback_returns_cited_passages(docs, tmp_path):
    bot = Chatbot(make_kb(docs, tmp_path), TemplateProvider(), tmp_path / "chat.json")
    ans = bot.ask("boil drinking water after the flood")
    assert ans.mode == "offline" and ans.citations and "[1]" in ans.text
    assert ans.citations[0].source == "guide.pdf" and ans.citations[0].page == 2


def test_no_match_and_missing_guidelines_message(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    bot = Chatbot(make_kb(empty, tmp_path), TemplateProvider(), tmp_path / "chat.json")
    ans = bot.ask("anything about floods")
    assert ans.mode == "no_match" and ans.citations == [] and "data/docs" in ans.text


class FakeHTTP:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, url, headers, payload):
        self.calls.append(payload)
        return {"content": [{"type": "text", "text": self.reply}]}


def test_llm_answer_prompt_citation_check_and_cache(docs, tmp_path):
    http = FakeHTTP("Boil water for ten minutes [1]. Also visit the moon [9].")
    bot = Chatbot(make_kb(docs, tmp_path), AnthropicProvider("k", http=http), tmp_path / "chat.json")
    q = "boil drinking water after the flood"
    ans = bot.ask(q)
    payload = http.calls[0]
    assert "untrusted" in payload["system"] and "112" in payload["system"]
    user = payload["messages"][0]["content"]
    assert "<passages>" in user and "[1] (guide.pdf, p. 2)" in user and q in user
    assert "[9]" not in ans.text and "[1]" in ans.text and ans.mode == "llm" and not ans.cached
    assert [c.n for c in ans.citations] == [1]
    again = bot.ask(q)
    assert len(http.calls) == 1 and again.cached and again.text == ans.text


def test_build_prompt_numbers_passages():
    hits = [(Passage(text="A", source="a.pdf", page=3), 0.9), (Passage(text="B", source="b.txt"), 0.8)]
    p = build_prompt("q?", hits)
    assert "[1] (a.pdf, p. 3) A" in p and "[2] (b.txt) B" in p and p.endswith("Question: q?")


def test_chatbot_reuses_gemini_provider(docs, tmp_path):
    calls = []

    def fake(url, headers, payload):
        calls.append((url, headers, payload))
        return {"choices": [{"message": {"content": "Boil it for ten minutes [1]."}}]}

    bot = Chatbot(make_kb(docs, tmp_path), GeminiProvider("k", http=fake), tmp_path / "chat.json")
    ans = bot.ask("boil drinking water after the flood")
    url, headers, payload = calls[0]
    assert url.endswith("/v1beta/openai/chat/completions") and payload["model"] == "gemini-3.5-flash-lite"
    assert "untrusted" in payload["messages"][0]["content"] and "[1] (guide.pdf, p. 2)" in payload["messages"][1]["content"]
    assert ans.mode == "llm" and ans.provider == "gemini" and [c.n for c in ans.citations] == [1]


def test_chatbot_falls_back_to_passages_when_llm_fails(docs, tmp_path):
    def down(url, headers, payload):
        raise RuntimeError("LLM request timed out after 60s")

    bot = Chatbot(make_kb(docs, tmp_path), GeminiProvider("k", http=down), tmp_path / "chat.json")
    ans = bot.ask("boil drinking water after the flood")
    assert ans.mode == "offline" and ans.fallback_reason and "timed out" in ans.text and ans.citations
    assert not (tmp_path / "chat.json").exists()

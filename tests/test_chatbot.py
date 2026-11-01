import numpy as np
import pytest

from aquaalert.chatbot.docs import (CHUNK_CHARS, Passage, chunk_text, docs_fingerprint, html_to_text, load_documents,
                                    strip_spaced_headings)
from aquaalert.chatbot.index import KnowledgeBase
from aquaalert.chatbot.rag import SYSTEM_PROMPT, Chatbot, brief_passages, build_prompt, question_kind, select_passages
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
    assert "<passages>" in user and "[1] (policy document · guide.pdf, p. 2)" in user and q in user
    assert "[9]" not in ans.text and "[1]" in ans.text and ans.mode == "llm" and not ans.cached
    assert [c.n for c in ans.citations] == [1]
    again = bot.ask(q)
    assert len(http.calls) == 1 and again.cached and again.text == ans.text


def test_build_prompt_numbers_passages():
    hits = [(Passage(text="A", source="a.pdf", page=3), 0.9), (Passage(text="B", source="b.txt"), 0.8)]
    p = build_prompt("q?", hits)
    assert "[1] (policy document · a.pdf, p. 3) A" in p and "[2] (policy document · b.txt) B" in p
    assert p.endswith("Question: q?")


def test_chatbot_reuses_gemini_provider(docs, tmp_path):
    calls = []

    def fake(url, headers, payload):
        calls.append((url, headers, payload))
        return {"choices": [{"message": {"content": "Boil it for ten minutes [1]."}}]}

    bot = Chatbot(make_kb(docs, tmp_path), GeminiProvider("k", http=fake), tmp_path / "chat.json")
    ans = bot.ask("boil drinking water after the flood")
    url, headers, payload = calls[0]
    assert url.endswith("/v1beta/openai/chat/completions") and payload["model"] == "gemini-3.5-flash-lite"
    assert "untrusted" in payload["messages"][0]["content"] and "[1] (policy document · guide.pdf, p. 2)" in payload["messages"][1]["content"]
    assert ans.mode == "llm" and ans.provider == "gemini" and [c.n for c in ans.citations] == [1]


def test_chatbot_falls_back_to_passages_when_llm_fails(docs, tmp_path):
    def down(url, headers, payload):
        raise RuntimeError("LLM request timed out after 60s")

    bot = Chatbot(make_kb(docs, tmp_path), GeminiProvider("k", http=down), tmp_path / "chat.json")
    ans = bot.ask("boil drinking water after the flood")
    assert ans.mode == "offline" and ans.fallback_reason and "timed out" in ans.text and ans.citations
    assert not (tmp_path / "chat.json").exists()



def test_strip_spaced_headings_keeps_prose():
    header = "na t i o n a l DiSaSt e r ma n a g e m e n t gu iDe l i n eS: The central government shall act."
    assert strip_spaced_headings(header).strip().startswith("The central government shall act")
    for prose in ("it is up to me to go", "Rs 5 lakh to 3 of 4 HQs", "CWC & IMD in J & K", "plan A or B or C by 2 pm"):
        assert strip_spaced_headings(prose) == prose


def test_html_loader_reads_article_only(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "page.html").write_text(
        "<html><head><title>Floods: Do's & Don'ts</title><script>var x=1;</script></head><body>"
        "<nav>Home | About | Contact menu</nav><article><h2>What to do during a flood</h2>"
        "<p>Do not walk through moving water. Six inches of moving water can make you fall.</p>"
        "<p>Move immediately to higher ground if there is any possibility of a flash flood.</p></article>"
        "<footer>Copyright</footer></body></html>")
    text = html_to_text((d / "page.html").read_text())
    assert "moving water" in text and "menu" not in text and "var x" not in text and "Copyright" not in text
    ps = load_documents(d)
    assert ps and ps[0].audience == "citizen"  # guessed from "Do's & Don'ts"/"during"


def test_sources_json_sets_title_url_audience(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("Governments shall prepare flood management plans for every river basin in the state.")
    (d / "sources.json").write_text('{"a.txt": {"title": "Guide A", "url": "https://example.org/a", "audience": "citizen"}}')
    p = load_documents(d)[0]
    assert (p.title, p.url, p.audience, p.citation()) == ("Guide A", "https://example.org/a", "citizen", "Guide A")
    (d / "sources.json").write_text("{not json")
    assert load_documents(d)[0].audience == "policy"  # malformed metadata ignored, guess used


def test_chunks_fit_encoder_window_and_change_fingerprint(docs):
    long = " ".join(f"Sentence {i} explains a flood safety measure in some detail." for i in range(80))
    assert all(len(c) <= CHUNK_CHARS + 80 for c in chunk_text(long))
    assert docs_fingerprint(docs).startswith(f"chunk={CHUNK_CHARS}/")


def test_question_kind():
    assert question_kind("What is the role of the state government in flood forecasting?") == "policy"
    assert question_kind("What should NDRF do during evacuation?") == "policy"
    assert question_kind("Should I walk through flood water?") == "personal"
    assert question_kind("बाढ़ से पहले क्या तैयारी करनी चाहिए?") == "personal"
    assert question_kind("सरकार क्या कर रही है?") == "policy"


def _p(aud, score, text="x"):
    return (Passage(text=text, source=aud, audience=aud), score)


def test_select_passages_guarantees_preferred_audience():
    hits = [_p("policy", 0.9), _p("policy", 0.8), _p("policy", 0.7), _p("policy", 0.6), _p("citizen", 0.5), _p("citizen", 0.4)]
    personal = select_passages(hits, "personal", k=4)
    assert [p.audience for p, _ in personal].count("citizen") == 2 and len(personal) == 4
    assert [s for _, s in personal] == sorted((s for _, s in personal), reverse=True)
    assert [p.audience for p, _ in select_passages(hits, "policy", k=4)] == ["policy"] * 4
    only_policy = [_p("policy", 0.9), _p("policy", 0.8)]
    assert select_passages(only_policy, "personal", k=4) == only_policy  # nothing to promote
    # a citizen passage found by the separate audience search is promoted even if it wasn't a candidate
    crowded = [_p("policy", 0.9 - i / 100, f"p{i}") for i in range(10)]
    out = select_passages(crowded, "personal", k=4, preferred=[_p("citizen", 0.2, "cross-lingual hit")])
    assert len(out) == 4 and any(p.audience == "citizen" for p, _ in out)


def test_audience_filtered_search(docs, tmp_path):
    (docs / "sources.json").write_text('{"notes.txt": {"audience": "citizen"}}')
    kb = make_kb(docs, tmp_path)
    hits = kb.search("boil drinking water after the flood", k=3, audience="citizen")
    assert hits and all(p.audience == "citizen" and p.source == "notes.txt" for p, _ in hits)


def test_system_prompt_prioritises_citizen_guidance():
    assert "citizen guidance" in SYSTEM_PROMPT and "policy document" in SYSTEM_PROMPT
    assert "personal or family safety" in SYSTEM_PROMPT and "governments or agencies" in SYSTEM_PROMPT


def test_comma_citations_are_validated(docs, tmp_path):
    http = FakeHTTP("Boil water [1, 9]. Nothing valid here [7, 8].")
    bot = Chatbot(make_kb(docs, tmp_path), AnthropicProvider("k", http=http), tmp_path / "chat.json")
    ans = bot.ask("boil drinking water after the flood")
    assert "[1]" in ans.text and "9" not in ans.text and "[7" not in ans.text
    assert [c.n for c in ans.citations] == [1]

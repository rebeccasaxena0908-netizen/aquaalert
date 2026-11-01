import pytest
from fastapi.testclient import TestClient

from api.main import State, app, get_state
from aquaalert.chatbot.index import KnowledgeBase
from aquaalert.chatbot.rag import Chatbot
from aquaalert.classify.zeroshot import KeywordZeroShot
from aquaalert.config import ROOT
from aquaalert.extract.geocode import Geocoder
from aquaalert.fuse.sensors import load_sensors
from aquaalert.ingest.loader import load_many
from aquaalert.match.matcher import hashing_encoder
from aquaalert.pipeline import process_posts
from aquaalert.summarize.providers import TemplateProvider

FILES = ",".join(str(ROOT / f) for f in ("data/sample_posts.csv", "data/sample_posts_multilingual.csv",
                                         "data/sample_posts_rumours.csv"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    docs = tmp / "docs"
    docs.mkdir()
    (docs / "tips.txt").write_text("When water enters the house switch off electricity and move to higher ground. "
                                   "Do not walk through moving flood water.")
    geo = Geocoder(cache_path=tmp / "g.json", online=False)
    clf = KeywordZeroShot()
    posts = process_posts(load_many(FILES), clf, geo)
    bot = lambda: Chatbot(KnowledgeBase(docs, tmp / "kb", hashing_encoder, "hash"), TemplateProvider(), tmp / "chat.json")  # noqa: E731
    state = State(posts, clf, geo, load_sensors(), chatbot_factory=bot, encoder_factory=lambda: hashing_encoder)
    app.dependency_overrides[get_state] = lambda: state
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_and_docs(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["posts"] == 110 and h["classifier"] == "keyword-zeroshot" and h["sensor_readings"] > 0
    assert client.get("/docs").status_code == 200 and client.get("/redoc").status_code == 200
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/health", "/posts", "/classify", "/matches", "/briefs", "/warnings", "/chat/ask"} <= paths
    assert len(client.get("/labels").json()) == 7


def test_posts_filters(client):
    everything = client.get("/posts", params={"limit": 1000}).json()
    assert len(everything) == 110
    ml = client.get("/posts", params={"language": "ml", "limit": 1000}).json()
    assert ml and all(p["language"] == "ml" for p in ml)
    two = client.get("/posts", params=[("label", "rescue_needed"), ("label", "missing_person"), ("limit", 1000)]).json()
    assert two and {p["label"] for p in two} <= {"rescue_needed", "missing_person"}
    clean = client.get("/posts", params={"max_misinfo": 0.5, "limit": 1000}).json()
    assert len(clean) < len(everything) and all(p["misinfo_score"] < 0.5 for p in clean)
    alp = client.get("/posts", params={"district": "alappuzha", "limit": 1000}).json()
    assert alp and all(p["extraction"]["district"] == "Alappuzha" for p in alp)
    window = client.get("/posts", params={"since": "2018-08-16T00:00:00", "limit": 1000}).json()
    assert window and all(p["timestamp"] >= "2018-08-16" for p in window)
    page = client.get("/posts", params={"limit": 5, "offset": 5}).json()
    assert [p["id"] for p in page] == [p["id"] for p in everything[5:10]]
    assert client.get("/posts", params={"limit": 0}).status_code == 422
    assert client.get("/posts", params={"label": "bogus"}).status_code == 422


def test_classify_single_text(client):
    r = client.post("/classify", json={"text": "URGENT: 12 people stranded on the roof in Chengannur, call 9847012345"})
    j = r.json()
    assert r.status_code == 200 and j["label"] == "rescue_needed" and j["language"] == "en"
    assert j["extraction"]["district"] == "Alappuzha" and j["extraction"]["phones"] == ["9847012345"]
    rumour = client.post("/classify", json={"text": "Mullaperiyar dam has broken!! Forward to all"}).json()
    assert rumour["misinfo_score"] > 0.5 and rumour["misinfo_reasons"]
    ml = client.post("/classify", json={"text": "ചെങ്ങന്നൂരിൽ ഞങ്ങൾ കുടുങ്ങി"}).json()
    assert ml["language"] == "ml" and ml["extraction"]["district"] == "Alappuzha"
    assert client.post("/classify", json={"text": ""}).status_code == 422


def test_matches_briefs_warnings(client):
    ms = client.get("/matches", params={"min_score": 0}).json()
    assert ms and {"need_id", "offer_id", "score", "distance_km"} <= set(ms[0])
    assert len({m["offer_id"] for m in ms}) == len(ms)  # one_to_one default
    briefs = client.get("/briefs").json()
    assert briefs and len({b["district"] for b in briefs}) == len(briefs)  # latest per district
    alp = client.get("/briefs", params={"day": "2018-08-15", "district": "Alappuzha"}).json()
    assert len(alp) == 1 and alp[0]["provider"] == "template"
    ws = client.get("/warnings", params={"day": "2018-08-15"}).json()
    levels = {w["district"]: w["level"] for w in ws}
    assert levels["Alappuzha"] == "warning" and any(v == "watch" for v in levels.values())
    only = client.get("/warnings", params={"day": "2018-08-15", "level": "warning"}).json()
    assert only and all(w["level"] == "warning" and w["reasons"] for w in only)
    assert client.get("/warnings", params={"level": "panic"}).status_code == 422


def test_briefs_llm_without_key_is_a_clear_400(client, monkeypatch):
    monkeypatch.setenv("AQUAALERT_LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = client.get("/briefs", params={"llm": True})
    assert r.status_code == 400 and "GEMINI_API_KEY" in r.json()["detail"]


def test_chat_ask_uses_docs_and_briefs(client):
    a = client.post("/chat/ask", json={"question": "Should I walk through moving flood water?"}).json()
    assert a["mode"] == "offline" and any(c["source"] == "tips.txt" for c in a["citations"])
    b = client.post("/chat/ask", json={"question": "rescue requests Alappuzha district boats"}).json()
    assert any(c["kind"] == "brief" for c in b["citations"])
    assert client.post("/chat/ask", json={"question": ""}).status_code == 422

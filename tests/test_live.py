"""Live-mode ingestion, tested on saved / constructed responses only (no network)."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import sqlite3

from aquaalert.ingest import live as L
from aquaalert.ingest import simulate as S

FIX = Path(__file__).resolve().parent / "fixtures" / "live"
NOW = datetime(2026, 11, 1, 16, 30, tzinfo=timezone.utc)


class FakeNet:
    """URL -> (status, body). Records every request; anything unmapped is a 404."""

    def __init__(self, routes: dict[str, tuple[int, bytes]]):
        self.routes, self.calls = routes, []

    def __call__(self, url, headers, body=None):
        self.calls.append((url, headers, body))
        for prefix, reply in self.routes.items():
            if url.startswith(prefix):
                return reply(url, headers, body) if callable(reply) else reply
        return 404, b""


def fx(name: str) -> bytes:
    return (FIX / name).read_bytes()


def sachet_routes(**extra):
    return {
        "https://sachet.ndma.gov.in/robots.txt": (404, b"not found"),
        "https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml": (200, fx("sachet_rss_india.xml")),
        "https://sachet.ndma.gov.in/cap_public_website/FetchXMLFile?identifier=TEST-FLOOD-0001": (200, fx("sachet_cap_flood.xml")),
        "https://sachet.ndma.gov.in/cap_public_website/FetchXMLFile?identifier=": (200, fx("sachet_cap_thunderstorm.xml")),
        **extra,
    }


def fetcher(net):
    return L.PoliteFetcher(transport=net, sleep=lambda s: None)


# ---------------------------------------------------------------- polite fetching
def test_robots_txt_is_respected():
    net = FakeNet({"https://news.google.com/robots.txt": (200, fx("google_news_robots.txt"))})
    f = fetcher(net)
    with pytest.raises(L.FetchError, match="blocked by robots.txt"):
        f.get("https://news.google.com/rss/search?q=flood")
    assert all("rss/search" not in u for u, _, _ in net.calls)  # never requested
    # Python's robotparser applies the first matching rule ("Disallow: /") rather than RFC 9309's most-specific
    # match, so even Google's re-allowed paths count as disallowed: stricter than required, which is safe here.
    assert not f.allowed("https://news.google.com/topics/abc")


def test_robots_unreachable_means_skip_and_404_means_allowed():
    f = fetcher(FakeNet({"https://down.example/robots.txt": (503, b"")}))
    with pytest.raises(L.FetchError, match="robots.txt .* unreachable"):
        f.get("https://down.example/feed.xml")
    f2 = fetcher(FakeNet({"https://ok.example/robots.txt": (404, b""), "https://ok.example/feed": (200, b"x")}))
    assert f2.get("https://ok.example/feed") == b"x"


def test_user_agent_and_per_host_gap():
    waits = []
    net = FakeNet({"https://api.gdeltproject.org/robots.txt": (404, b""), "https://api.gdeltproject.org/api": (200, b"{}")})
    f = L.PoliteFetcher(transport=net, sleep=waits.append)
    f.get("https://api.gdeltproject.org/api/v2/doc/doc?q=1")
    f.get("https://api.gdeltproject.org/api/v2/doc/doc?q=2")
    assert waits and max(waits) > 5  # GDELT asks for at most one request every 5 s
    assert all("AquaAlert/" in h["User-Agent"] for _, h, _ in net.calls)


def test_http_errors_and_rate_limits_raise():
    f = fetcher(FakeNet({"https://x.example/robots.txt": (404, b""), "https://x.example/a": (429, b"slow down"),
                         "https://x.example/b": (500, b"")}))
    with pytest.raises(L.FetchError, match="429"):
        f.get("https://x.example/a")
    with pytest.raises(L.FetchError, match="HTTP 500"):
        f.get("https://x.example/b")


# ---------------------------------------------------------------- SACHET
def test_parse_real_sachet_rss_and_cap():
    items = L.parse_sachet_rss(fx("sachet_rss_india.xml"))
    assert len(items) == 3 and all(i["published"].tzinfo for i in items)
    assert items[0]["link"].startswith("https://sachet.ndma.gov.in/cap_public_website/FetchXMLFile?identifier=")
    cap = L.parse_cap(fx("sachet_cap_thunderstorm.xml"))  # real alert saved 2026-11-01
    assert cap["sender"] == "Uttarakhand-SDMA" and cap["event"] == "Thunderstorm with Lightning"
    assert cap["severity"] == "Moderate" and cap["effective"] and cap["expires"]
    assert not L.is_flood_alert(cap)
    flood = L.parse_cap(fx("sachet_cap_flood.xml"))
    assert L.is_flood_alert(flood) and "Aluva" in flood["area"] and flood["instruction"].startswith("Residents")
    assert flood["headline"].startswith("Periyar river")  # English info block preferred over Malayalam


def test_poll_sachet_keeps_flood_alerts_and_only_fetches_new_items():
    net = FakeNet(sachet_routes())
    alerts, examined, errors = L.poll_sachet(fetcher(net), set(), feeds=("india",))
    assert errors == {} and len(examined) == 3 and len(alerts) == 1
    a = alerts[0]
    assert a.kind == "official_alert" and a.source == "sachet" and a.extra["event"] == "Flood"
    assert a.extra["severity"] == "Severe" and a.url.endswith("TEST-FLOOD-0001") and a.author == "Kerala-SDMA"
    net2 = FakeNet(sachet_routes())
    again, examined2, _ = L.poll_sachet(fetcher(net2), set(examined), feeds=("india",))
    assert again == [] and examined2 == [] and not any("FetchXMLFile" in u for u, _, _ in net2.calls)


def test_one_failing_feed_does_not_sink_sachet_but_all_failing_does():
    net = FakeNet(sachet_routes(**{"https://sachet.ndma.gov.in/cap_public_website/rss/rss_kerala.xml": (500, b"")}))
    alerts, _, errors = L.poll_sachet(fetcher(net), set(), feeds=("india", "kerala"))
    assert len(alerts) == 1 and "kerala" in errors
    dead = FakeNet({"https://sachet.ndma.gov.in/robots.txt": (404, b"")})
    with pytest.raises(L.FetchError, match="all SACHET feeds failed"):
        L.poll_sachet(fetcher(dead), set(), feeds=("india", "kerala"))


# ---------------------------------------------------------------- news (GDELT)
def test_gdelt_parse_and_prefilter():
    net = FakeNet({"https://api.gdeltproject.org/robots.txt": (404, b""),
                   "https://api.gdeltproject.org/api/v2/doc/doc": (200, fx("gdelt_artlist.json"))})
    items = L.poll_news(fetcher(net), queries=("flood",))
    assert [i.text for i in items] == ["Heavy rain triggers flood alert in Ernakulam, schools shut",
                                       "बिहार में बाढ़ से कई गांव जलमग्न"]  # cricket story filtered out
    assert items[0].published_at == datetime(2026, 11, 1, 14, 30, tzinfo=timezone.utc)
    assert items[1].lang == "hi" and items[0].url.startswith("https://") and items[0].kind == "news"
    with pytest.raises(L.FetchError, match="non-JSON"):
        L.parse_gdelt(b"Please limit requests to one every 5 seconds")


# ---------------------------------------------------------------- Bluesky
def bsky_routes(expire_first=False):
    state = {"searches": 0}

    def search(url, headers, body):
        state["searches"] += 1
        if expire_first and state["searches"] == 1:
            return 400, b'{"error":"ExpiredToken"}'
        assert headers["Authorization"].startswith("Bearer access-")
        return 200, fx("bsky_search.json")

    sessions = iter(range(1, 10))
    return {
        "https://bsky.social/robots.txt": (404, b""),
        "https://bsky.social/xrpc/com.atproto.server.createSession":
            lambda u, h, b: (200, json.dumps({"accessJwt": f"access-{next(sessions)}", "refreshJwt": "r",
                                              "did": "did:plc:me", "handle": "me.bsky.social"}).encode()),
        "https://bsky.social/xrpc/app.bsky.feed.searchPosts": search,
    }, state


def test_bluesky_login_search_parse_and_relogin():
    routes, state = bsky_routes(expire_first=True)
    net = FakeNet(routes)
    client = L.BlueskyClient(fetcher(net), "me.bsky.social", "app-pass-xxxx")
    items = L.poll_bluesky(client, NOW, queries=("flood kerala",))
    assert [i.author for i in items] == [None, None]  # privacy: text, time and link only
    assert items[0].url == "https://bsky.app/profile/did:plc:test1/post/3kabc111"
    assert not any("example.bsky.social" in f"{i.id}{i.url}{i.text}{i.extra}" for i in items)
    assert items[1].lang == "ml" and items[0].published_at.tzinfo
    logins = [c for c in net.calls if "createSession" in c[0]]
    assert len(logins) == 2  # expired token -> one fresh login
    assert json.loads(logins[0][2]) == {"identifier": "me.bsky.social", "password": "app-pass-xxxx"}
    search_url = next(u for u, _, _ in net.calls if "searchPosts" in u)
    assert "sort=latest" in search_url and "since=" in search_url


def test_missing_bluesky_credentials_skip_quietly(tmp_path, monkeypatch):
    monkeypatch.delenv("BLUESKY_HANDLE", raising=False)
    monkeypatch.delenv("BLUESKY_APP_PASSWORD", raising=False)
    net = FakeNet(sachet_routes())
    poller = L.Poller(L.LiveStore(tmp_path / "l.sqlite"), fetcher(net), interval=600)
    assert poller.bluesky is None
    res = poller.poll_once(NOW, force=True)
    assert res["bluesky"] == "skipped: not configured" and not any("bsky" in u for u, _, _ in net.calls)
    assert poller.store.status()["bluesky"]["state"] == "skipped"


# ---------------------------------------------------------------- store + poller
def item(i, text="Flood water rising in Aluva", source="news", minutes=0):
    return L.LiveItem(id=f"{source}:{i}", source=source, kind="news", text=text, url=f"https://x/{i}",
                      published_at=NOW - timedelta(minutes=minutes))


def test_store_dedupes_by_id_and_text_and_filters_by_time(tmp_path):
    st = L.LiveStore(tmp_path / "l.sqlite")
    assert st.add([item(1), item(1), item(2, "FLOOD water rising in Aluva!!")], NOW) == 1  # same id / same text
    assert st.add([item(3, "Old flood story", minutes=60 * 30)], NOW) == 1
    assert [i.id for i in st.items(since=NOW - timedelta(hours=24))] == ["news:1"]
    got = st.items()[0]
    assert got.published_at.tzinfo and got.url == "https://x/1" and got.source == "news"


def test_poller_sources_fail_independently_with_backoff(tmp_path):
    routes = sachet_routes(**{"https://api.gdeltproject.org/robots.txt": (404, b""),
                              "https://api.gdeltproject.org/api": (429, b"Please limit requests")})
    st = L.LiveStore(tmp_path / "l.sqlite")
    st.set_news_enabled(True)
    poller = L.Poller(st, fetcher(FakeNet(routes)), interval=600, bluesky=False)
    res = poller.poll_once(NOW, force=True)
    assert res["sachet"].startswith("ok") and res["news"].startswith("failed")
    status = st.status()
    assert status["sachet"]["state"] == "ok" and status["news"]["state"] == "failed" and status["news"]["failures"] == 1
    assert status["sachet"]["duration"] is not None
    assert datetime.fromisoformat(status["news"]["next_allowed"]) == NOW + timedelta(seconds=600)
    assert len(st.items(kinds=("official_alert",))) == 1
    later = NOW + timedelta(seconds=700)
    res2 = poller.poll_once(later)
    assert res2["news"].startswith("failed") and st.status()["news"]["failures"] == 2
    assert datetime.fromisoformat(st.status()["news"]["next_allowed"]) == later + timedelta(seconds=1200)  # doubled
    assert poller.poll_once(later + timedelta(seconds=60))["sachet"] == "not due"
    assert st.last_updated() == later  # SACHET was due again at +700 s and succeeded


def test_to_posts_keeps_source_and_url_but_not_official_alerts():
    alert = L.LiveItem(id="sachet:1", source="sachet", kind="official_alert", text="Flood", url="u", published_at=NOW)
    posts = L.to_posts([alert, item(5)])
    assert [p.id for p in posts] == ["news:5"] and posts[0].source == "news" and posts[0].url == "https://x/5"


def test_flood_prefilter_multilingual():
    for t in ("Flooding in Kochi", "बाढ़ का पानी", "വെള്ളപ്പൊക്കം", "বন্যা পরিস্থিতি", "heavy rainfall alert", "Periyar river level rising"):
        assert L.is_flood_related(t), t
    for t in ("cricket final in Kochi", "new phone launch", "stock market flooded with orders?"):
        assert not L.is_flood_related(t), t


FILTER_EXAMPLES = json.loads(fx("flood_filter_examples.json"))


@pytest.mark.parametrize("ex", FILTER_EXAMPLES["posts"], ids=lambda e: e["why"])
def test_flood_filter_saved_examples(ex):
    assert L.is_flood_related(ex["text"]) == ex["flood"]


def test_flood_filter_precision_on_the_ten_posts_read():
    read = [e for e in FILTER_EXAMPLES["posts"] if e["read"]]
    passed = [e for e in read if L.is_flood_related(e["text"])]
    assert len(read) == 10 and len(passed) == 7 and all(e["flood"] for e in passed)


@pytest.mark.parametrize("text", [
    "Glacial lake outburst flood warning", "GLOF alert in Sikkim", "Inundation of low-lying wards", "severe waterlogging in Patna",
    "water logging on NH-27", "Cloudburst in Uttarkashi", "गुड़गांव में जलभराव", "उत्तरकाशी में बादल फटना", "ग्लेशियर झील फटी",
    "കൊച്ചിയിൽ വെള്ളക്കെട്ട്", "ഇടുക്കിയിൽ മേഘവിസ്ഫോടനം", "কলকাতায় জলমগ্ন রাস্তা", "floodwater enters homes", "Flood alert: thousands flood-hit",
    "Farmers flood relief demand", "Thousands flood Delhi over voter lists as flood alert issued in Assam",
])
def test_flood_filter_new_terms_and_real_floods(text):
    assert L.is_flood_related(text)


@pytest.mark.parametrize("text", [
    "Protesters flood the streets with banners", "A flood of tweets about the match", "Fans flood Mumbai over the final",
    "New floodlights at the stadium", "Inbox flooded with messages", "पूरा देश देख रहा है", "জয়পুর আর বানান প্রতিযোগিতা",
    "The glacial lake is lovely in spring",
])
def test_flood_filter_rejects_non_flood(text):
    assert not L.is_flood_related(text)


def test_old_bluesky_rows_are_scrubbed_of_handles(tmp_path):
    path = tmp_path / "l.sqlite"
    L.LiveStore(path)
    uri = "at://did:plc:test1/app.bsky.feed.post/3kabc111"
    with sqlite3.connect(path) as c:
        c.execute("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (f"bluesky:{uri}", "bluesky", "social", "flood in Aluva", "https://bsky.app/profile/someone.example/post/3kabc111",
                   NOW.isoformat(), NOW.isoformat(), "en", "someone.example", "{}", "h"))
    item_after = L.LiveStore(path).items()[0]
    assert item_after.author is None and item_after.id == L._bsky_id(uri)
    assert item_after.url == "https://bsky.app/profile/did:plc:test1/post/3kabc111"
    assert b"someone.example" not in path.read_bytes()


# ---------------------------------------------------------------- simulated incident
def test_simulated_incident_inject_and_clear(tmp_path):
    st = L.LiveStore(tmp_path / "l.sqlite")
    st.add([item(1)], NOW)
    assert S.inject(st, NOW) == len(S.SIM_POSTS) and 5 <= len(S.SIM_POSTS) <= 8
    sims = [i for i in st.items() if i.source == "simulated"]
    assert all(i.id.startswith("sim-") and i.url == "" and i.extra.get("simulated") for i in sims)
    assert S.clear(st) == len(S.SIM_POSTS) and [i.source for i in st.items()] == ["news"]  # real data untouched



# ---------------------------------------------------------------- never blocking: budgets, slow, skipped
def slow_net(delay: float, routes):
    import time as _t

    def transport(url, headers, body=None):
        if "rss_india" in url:
            _t.sleep(delay)
        return FakeNet(routes)(url, headers, body)
    return transport


def test_slow_source_does_not_block_and_is_marked_slow(tmp_path):
    import time as _t

    st = L.LiveStore(tmp_path / "l.sqlite")
    st.set_news_enabled(True)
    routes = sachet_routes(**{"https://api.gdeltproject.org/robots.txt": (404, b""),
                              "https://api.gdeltproject.org/api": (200, fx("gdelt_artlist.json"))})
    fetch = L.PoliteFetcher(transport=slow_net(1.5, routes), sleep=lambda s: None)
    poller = L.Poller(st, fetch, interval=600, bluesky=False, budgets={"sachet": 0.3, "news": 5})
    t0 = _t.monotonic()
    res = poller.poll_once(NOW, force=True)
    assert _t.monotonic() - t0 < 1.2  # did not wait for SACHET
    assert res["sachet"] == "slow: still running" and res["news"].startswith("ok")
    assert st.status()["sachet"]["state"] == "slow" and "still running" in st.status()["sachet"]["note"]
    assert len(st.items(kinds=("news",))) == 2  # news stored while SACHET was still busy
    again = poller.poll_once(NOW + timedelta(seconds=1), force=True)
    assert again["sachet"] == "skipped: previous poll still running"
    poller._threads["sachet"].join(5)  # the slow poll finishes in the background and records its outcome
    s = st.status()["sachet"]
    assert s["state"] == "slow" and "time budget reached" in s["note"]  # it stopped before the CAP documents
    assert st.items(kinds=("official_alert",)) == []
    poller.budgets["sachet"] = 10  # next poll, with room in the budget, fetches the CAP documents
    poller.poll_once(NOW + timedelta(seconds=2), force=True)
    assert len(st.items(kinds=("official_alert",))) == 1


def test_sachet_deadline_returns_partial_results():
    import time as _t

    net = FakeNet(sachet_routes())
    alerts, examined, errors = L.poll_sachet(fetcher(net), set(), feeds=("india",), deadline=_t.monotonic() - 1)
    assert alerts == [] and examined == [] and "_budget" in errors  # stopped immediately, no exception
    alerts, examined, errors = L.poll_sachet(fetcher(net), set(), feeds=("india",), deadline=_t.monotonic() + 30)
    assert len(alerts) == 1 and "_budget" not in errors


def test_partial_sachet_poll_is_slow_not_failed(tmp_path, monkeypatch):
    st = L.LiveStore(tmp_path / "l.sqlite")
    monkeypatch.setattr(L, "poll_sachet", lambda *a, **k: ([], [], {"_budget": "time budget reached after 3 of 25"}))
    poller = L.Poller(st, fetcher(FakeNet({})), interval=600, bluesky=False, budgets={"news": 0.01})
    poller.poll_once(NOW, force=True)
    s = st.status()["sachet"]
    assert s["state"] == "slow" and "time budget reached" in s["note"] and s["last_success"] == NOW.isoformat()


def test_robots_block_is_skipped_not_failed(tmp_path, monkeypatch):
    st = L.LiveStore(tmp_path / "l.sqlite")
    st.set_news_enabled(True)

    def blocked(*a, **k):
        raise L.FetchError("blocked by robots.txt: https://example/rss")
    monkeypatch.setattr(L, "poll_news", blocked)
    monkeypatch.setattr(L, "poll_sachet", lambda *a, **k: ([], [], {}))
    L.Poller(st, fetcher(FakeNet({})), interval=600, bluesky=False).poll_once(NOW, force=True)
    assert st.status()["news"]["state"] == "skipped" and "robots" in st.status()["news"]["last_error"]


def test_old_database_gets_new_status_columns(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE source_status (source TEXT PRIMARY KEY, state TEXT, last_attempt TEXT, last_success "
                  "TEXT, last_error TEXT, failures INTEGER DEFAULT 0, next_allowed TEXT, last_new INTEGER DEFAULT 0)")
    st = L.LiveStore(db)
    st.set_status("news", state="ok", duration=1.5, note="x")
    assert st.status()["news"]["duration"] == 1.5


def test_gdelt_retries_rate_limits_politely():
    calls = {"n": 0}

    def flaky(url, headers, body=None):
        if url.endswith("robots.txt"):
            return 404, b""
        calls["n"] += 1
        return (429, b"Please limit requests to one every 5 seconds") if calls["n"] < 3 else (200, fx("gdelt_artlist.json"))

    waits = []
    f = L.PoliteFetcher(transport=flaky, sleep=waits.append)
    items = L.poll_news(f, queries=("flood",))
    assert len(items) == 2 and calls["n"] == 3
    assert 15.0 in waits and 30.0 in waits  # backed off 15 s, then 30 s
    always = L.PoliteFetcher(transport=lambda u, h, b=None: (404, b"") if u.endswith("robots.txt") else (429, b""),
                             sleep=lambda s: None)
    with pytest.raises(L.FetchError, match="429"):
        L.poll_news(always, queries=("flood",))


# ---------------------------------------------------------------- GDELT is experimental and off by default
def test_gdelt_is_off_by_default_and_never_contacted(tmp_path):
    st = L.LiveStore(tmp_path / "l.sqlite")
    net = FakeNet(sachet_routes())
    res = L.Poller(st, fetcher(net), interval=600, bluesky=False).poll_once(NOW, force=True)
    assert res["news"] == "off: experimental" and st.status()["news"]["state"] == "off"
    assert not any("gdelt" in u for u, _, _ in net.calls) and res["sachet"].startswith("ok")


def test_gdelt_toggle_is_read_before_each_poll_and_remembered(tmp_path):
    path = tmp_path / "l.sqlite"
    routes = sachet_routes(**{"https://api.gdeltproject.org/robots.txt": (404, b""),
                              "https://api.gdeltproject.org/api": (200, fx("gdelt_artlist.json"))})
    net = FakeNet(routes)
    st = L.LiveStore(path)
    poller = L.Poller(st, fetcher(net), interval=600, bluesky=False)
    poller.poll_once(NOW, force=True)
    st.set_news_enabled(True)
    assert L.LiveStore(path).news_enabled()
    assert poller.poll_once(NOW, force=True)["news"].startswith("ok") and any("gdelt" in u for u, _, _ in net.calls)
    st.set_news_enabled(False)
    assert poller.poll_once(NOW, force=True)["news"] == "off: experimental"


# ---------------------------------------------------------------- CAP geometry
def test_cap_polygon_parsing():
    ring = "10.0,76.2 10.2,76.2 10.2,76.4 10.0,76.2"
    assert L.parse_cap_polygon(ring) == [[10.0, 76.2], [10.2, 76.2], [10.2, 76.4], [10.0, 76.2]]
    for bad in ("", "10.0,76.2 10.2,76.2", "a,b c,d e,f g,h", "95,76 95,77 96,77 95,76", "10,76,1 10,77,1 11,77,1 10,76,1"):
        assert L.parse_cap_polygon(bad) is None, bad


def test_parse_cap_reads_inline_polygons_and_records_the_polygon_url():
    flood = fx("sachet_cap_flood.xml").decode()
    assert L.parse_cap(flood.encode())["polygons"] == []
    inline = flood.replace("<area><areaDesc>Ernakulam district: Aluva, Paravur</areaDesc></area>",
                           "<area><areaDesc>Ernakulam</areaDesc><polygon>10.0,76.2 10.2,76.2 10.2,76.4 10.0,76.2</polygon>"
                           "<polygon>not a polygon</polygon></area>", 1)
    assert L.parse_cap(inline.encode())["polygons"] == [[[10.0, 76.2], [10.2, 76.2], [10.2, 76.4], [10.0, 76.2]]]
    assert "FetchPolygonXMLFile" in L.parse_cap(fx("sachet_cap_thunderstorm.xml"))["polygon_url"]

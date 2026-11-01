"""Live-mode ingestion, tested on saved / constructed responses only (no network)."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aquaalert.ingest import live as L

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
    assert [i.author for i in items] == ["resident.example.bsky.social", "malayali.example.bsky.social"]
    assert items[0].url == "https://bsky.app/profile/resident.example.bsky.social/post/3kabc111"
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
    assert res["bluesky"] == "not configured" and not any("bsky" in u for u, _, _ in net.calls)


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
    poller = L.Poller(st, fetcher(FakeNet(routes)), interval=600, bluesky=False)
    res = poller.poll_once(NOW, force=True)
    assert res["sachet"].startswith("ok") and res["news"].startswith("error")
    status = st.status()
    assert status["sachet"]["state"] == "ok" and status["news"]["failures"] == 1
    assert datetime.fromisoformat(status["news"]["next_allowed"]) == NOW + timedelta(seconds=600)
    assert len(st.items(kinds=("official_alert",))) == 1
    later = NOW + timedelta(seconds=700)
    res2 = poller.poll_once(later)
    assert res2["news"].startswith("error") and st.status()["news"]["failures"] == 2
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
        assert L.is_flood_related(t) == ("flood" in t.lower()), t

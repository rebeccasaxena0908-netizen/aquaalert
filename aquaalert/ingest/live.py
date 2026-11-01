"""Live mode: poll free, legitimate sources into SQLite (data/cache/live.sqlite).

Sources (each fails independently, with exponential backoff):
  * sachet  - NDMA SACHET CAP RSS feeds (all-India + state feeds) -> CAP 1.2 XML per item; keeps flood,
              heavy-rain and river alerts as official alerts.
  * news    - GDELT DOC 2.0 API (English + Hindi flood news from Indian sources). Google News RSS is NOT used:
              news.google.com/robots.txt disallows /rss for generic agents.
  * bluesky - app.bsky.feed.searchPosts with an app-password session (BLUESKY_HANDLE / BLUESKY_APP_PASSWORD);
              skipped quietly when the credentials are missing. Unauthenticated search returns 403.
Every request checks robots.txt (RFC 9309: 4xx = allowed, 5xx/unreachable = skip this poll; Python's robotparser
applies the first matching rule, which is stricter than RFC 9309's most-specific match), keeps a minimum gap
per host and sends a descriptive User-Agent. Phone numbers are redacted before any text reaches an LLM elsewhere in
the pipeline (summarize/brief.py); nothing here calls an LLM.

    python -m aquaalert.ingest.live --once        # one poll of every source, then print the store summary
    python -m aquaalert.ingest.live --loop        # poll forever (AQUAALERT_LIVE_INTERVAL seconds)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from aquaalert.config import ROOT
from aquaalert.schemas import Post

UTC = timezone.utc
DB_PATH = ROOT / "data" / "cache" / "live.sqlite"
USER_AGENT = ("AquaAlert/0.5 (flood-crisis research prototype; polite polling of public feeds; "
              "https://github.com/rebeccasaxena0908-netizen/aquaalert)")
SACHET_BASE = "https://sachet.ndma.gov.in/cap_public_website/rss/"
# Feed files verified on 2026-11-01 (rss_<state>.xml). Multi-word states (Tamil Nadu, West Bengal, ...) returned 404
# for every spelling tried, so they are not listed; the all-India feed covers them.
SACHET_FEEDS = ("india", "kerala", "assam", "bihar", "odisha", "uttarakhand", "karnataka", "maharashtra", "gujarat")
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
BSKY_PDS = "https://bsky.social"
CAP_NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}
SOURCES = ("sachet", "news", "bluesky")
MAX_CAP_PER_POLL = 25  # CAP documents fetched per poll (the all-India feed lists 100 items)

# Flood keywords: English, Hindi, Malayalam, Bengali/Assamese, Marathi, plus romanised forms
FLOOD_TERMS = re.compile(
    r"\b(?:flood\w*|inundat\w*|waterlogg\w*|water-logg\w*|cloud\s*burst|deluge|flash\s+flood|heavy\s+rain\w*|"
    r"very\s+heavy|extremely\s+heavy|river\s+(?:level|water|swell\w*|overflow\w*|in\s+spate)|danger\s+level|"
    r"dam\s+(?:shutters?|gates?|release|breach\w*|overflow\w*)|landslide\w*|baadh|baarish|vellapokkam|bonna|bonya)\b"
    r"|बाढ़|बाढ|जलभराव|भारी\s*(?:वर्षा|बारिश)|अति\s*भारी|बादल\s*फट|മഴ|വെള്ളപ്പൊക്കം|ഉരുൾപൊട്ടൽ|প্লাবন|বন্যা|বান|"
    r"অতি\s*ভারী|पूर|मुसळधार",
    re.IGNORECASE,
)
SACHET_KEEP = re.compile(r"flood|inundat|heavy\s+rain|rainfall|river|dam\b|cloud\s*burst|water\s*level|बाढ़|भारी", re.I)


# ---------------------------------------------------------------- fetching
Transport = Callable[[str, dict, bytes | None], tuple[int, bytes]]


def urllib_transport(url: str, headers: dict, body: bytes | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() if exc.fp else b""
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise FetchError(f"network error for {urllib.parse.urlsplit(url).netloc}: {getattr(exc, 'reason', exc)}") from None


class FetchError(RuntimeError):
    pass


class PoliteFetcher:
    """robots.txt-aware fetcher with a per-host minimum gap and a descriptive User-Agent."""

    def __init__(self, transport: Transport = urllib_transport, user_agent: str = USER_AGENT,
                 min_gap: dict[str, float] | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.transport, self.ua, self.sleep = transport, user_agent, sleep
        self.min_gap = {"api.gdeltproject.org": 6.0, **(min_gap or {})}  # GDELT asks for <= 1 request / 5 s
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()

    def _wait(self, host: str) -> None:
        gap = self.min_gap.get(host, 1.0)
        with self._lock:
            delay = self._last.get(host, 0) + gap - time.monotonic()
            if delay > 0:
                self.sleep(delay)
            self._last[host] = time.monotonic()

    def allowed(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        host = parts.netloc
        if host not in self._robots:
            self._wait(host)
            status, body = self.transport(f"{parts.scheme}://{host}/robots.txt", {"User-Agent": self.ua}, None)
            if 400 <= status < 500:
                self._robots[host] = None  # RFC 9309: unavailable robots.txt -> everything allowed
            elif status >= 500 or status == 0:
                raise FetchError(f"robots.txt for {host} unreachable (HTTP {status}); skipping this poll")
            else:
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(body.decode("utf-8", "replace").splitlines())
                self._robots[host] = rp
        rp = self._robots[host]
        return rp is None or rp.can_fetch(self.ua, url)

    def get(self, url: str, headers: dict | None = None, body: bytes | None = None) -> bytes:
        if not self.allowed(url):
            raise FetchError(f"blocked by robots.txt: {url}")
        host = urllib.parse.urlsplit(url).netloc
        self._wait(host)
        status, data = self.transport(url, {"User-Agent": self.ua, **(headers or {})}, body)
        if status == 429:
            raise FetchError(f"rate limited by {host} (HTTP 429)")
        if status >= 400 or status == 0:
            raise FetchError(f"HTTP {status} from {host}")
        return data


# ---------------------------------------------------------------- items
@dataclass
class LiveItem:
    id: str
    source: str  # sachet | news | bluesky | simulated
    kind: str  # official_alert | news | social
    text: str
    url: str
    published_at: datetime
    lang: str | None = None
    author: str | None = None
    extra: dict = field(default_factory=dict)  # CAP fields for alerts, domain for news, ...

    def content_hash(self) -> str:
        return hashlib.sha1(" ".join(re.findall(r"\w+", self.text.lower()))[:300].encode()).hexdigest()


def is_flood_related(text: str) -> bool:
    return bool(FLOOD_TERMS.search(text or ""))


def _utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


# ---------------------------------------------------------------- SACHET (CAP)
def parse_sachet_rss(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for it in root.findall("./channel/item"):
        pub = it.findtext("pubDate")
        out.append({"guid": it.findtext("guid") or it.findtext("link"), "title": it.findtext("title") or "",
                    "link": it.findtext("link") or "", "category": it.findtext("category"),
                    "author": it.findtext("author"),
                    "published": _utc(parsedate_to_datetime(pub)) if pub else datetime.now(UTC)})
    return out


def parse_cap(xml_bytes: bytes) -> dict:
    """CAP 1.2 alert -> flat dict (English info block preferred, all area descriptions joined)."""
    root = ET.fromstring(xml_bytes)
    infos = root.findall("cap:info", CAP_NS)
    if not infos:
        raise ValueError("no <info> block in CAP alert")
    info = next((i for i in infos if (i.findtext("cap:language", namespaces=CAP_NS) or "").lower().startswith("en")),
                infos[0])
    t = lambda node, tag: (node.findtext(f"cap:{tag}", namespaces=CAP_NS) or "").strip()  # noqa: E731
    areas = [t(a, "areaDesc") for i in infos[:1] for a in i.findall("cap:area", CAP_NS)]
    return {
        "identifier": t(root, "identifier"), "sender": t(root, "sender"), "sent": t(root, "sent"),
        "status": t(root, "status"), "msg_type": t(root, "msgType"),
        "event": t(info, "event"), "category": t(info, "category"), "severity": t(info, "severity"),
        "urgency": t(info, "urgency"), "certainty": t(info, "certainty"), "effective": t(info, "effective"),
        "expires": t(info, "expires"), "headline": t(info, "headline"), "description": t(info, "description"),
        "instruction": t(info, "instruction"), "area": "; ".join(a for a in areas if a),
        "languages": [t(i, "language") for i in infos],
    }


def is_flood_alert(cap: dict) -> bool:
    return bool(SACHET_KEEP.search(" ".join((cap.get("event", ""), cap.get("headline", ""), cap.get("description", "")))))


def poll_sachet(fetch: PoliteFetcher, known: set[str], feeds=SACHET_FEEDS, max_cap: int = MAX_CAP_PER_POLL):
    """Return (flood-related alerts, ids of every CAP item examined, per-feed errors). Only new items are fetched.

    A failing feed or CAP document is skipped; the source as a whole fails only when every feed fails.
    """
    seen: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for feed in feeds:
        try:
            for item in parse_sachet_rss(fetch.get(f"{SACHET_BASE}rss_{feed}.xml")):
                seen.setdefault(item["guid"], {**item, "feed": feed})
        except (FetchError, ET.ParseError) as exc:
            errors[feed] = str(exc)[:120]
    if len(errors) == len(feeds):
        raise FetchError("all SACHET feeds failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items())[:280])
    new = sorted((i for g, i in seen.items() if f"sachet:{g}" not in known), key=lambda i: i["published"], reverse=True)
    alerts, examined = [], []
    for item in new[:max_cap]:
        examined.append(f"sachet:{item['guid']}")
        try:
            cap = parse_cap(fetch.get(item["link"]))
        except (FetchError, ValueError, ET.ParseError):
            continue
        if cap["status"].lower() not in ("actual", "") or not is_flood_alert(cap):
            continue
        alerts.append(LiveItem(
            id=f"sachet:{item['guid']}", source="sachet", kind="official_alert",
            text=cap["headline"] or item["title"], url=item["link"], published_at=item["published"],
            lang="en" if cap["languages"] and str(cap["languages"][0]).lower().startswith("en") else None,
            author=cap["sender"] or item.get("author"), extra={**cap, "feed": item["feed"]}))
    return alerts, examined, errors


# ---------------------------------------------------------------- news (GDELT)
GDELT_QUERIES = (
    '(flood OR floods OR flooding OR inundation OR "heavy rain" OR cloudburst) sourcecountry:india',
    "(बाढ़ OR जलभराव OR भारी बारिश) sourcelang:hindi",
)


def parse_gdelt(json_bytes: bytes) -> list[LiveItem]:
    text = json_bytes.decode("utf-8", "replace").strip()
    if not text.startswith("{"):
        raise FetchError(f"GDELT returned a non-JSON reply: {text[:80]}")
    out = []
    for a in json.loads(text or "{}").get("articles", []):
        when = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        out.append(LiveItem(id=f"news:{hashlib.sha1(a['url'].encode()).hexdigest()[:16]}", source="news", kind="news",
                            text=a.get("title", "").strip(), url=a["url"], published_at=when,
                            lang={"English": "en", "Hindi": "hi"}.get(a.get("language", ""), a.get("language")),
                            author=a.get("domain"),
                            extra={"domain": a.get("domain"), "sourcecountry": a.get("sourcecountry")}))
    return out


def poll_news(fetch: PoliteFetcher, queries=GDELT_QUERIES) -> list[LiveItem]:
    items = []
    for q in queries:
        params = {"query": q, "mode": "artlist", "format": "json", "maxrecords": "75", "sort": "datedesc",
                  "timespan": "24h"}
        items += parse_gdelt(fetch.get(f"{GDELT_URL}?{urllib.parse.urlencode(params)}"))
    return [i for i in items if is_flood_related(i.text)]


# ---------------------------------------------------------------- Bluesky
BSKY_QUERIES = ("flood kerala", "flood india", "flood assam", "बाढ़", "വെള്ളപ്പൊക്കം")


class BlueskyClient:
    """App-password session kept in memory only (never written to disk); reused across polls."""

    def __init__(self, fetch: PoliteFetcher, handle: str, app_password: str, pds: str = BSKY_PDS) -> None:
        self.fetch, self.handle, self._password, self.pds = fetch, handle, app_password, pds.rstrip("/")
        self._access: str | None = None

    def _login(self) -> None:
        body = json.dumps({"identifier": self.handle, "password": self._password}).encode()
        data = json.loads(self.fetch.get(f"{self.pds}/xrpc/com.atproto.server.createSession",
                                         {"Content-Type": "application/json"}, body))
        self._access = data["accessJwt"]

    def search(self, q: str, since: datetime, limit: int = 25) -> list[dict]:
        if self._access is None:
            self._login()
        params = urllib.parse.urlencode({"q": q, "sort": "latest", "limit": limit,
                                         "since": since.strftime("%Y-%m-%dT%H:%M:%SZ")})
        url = f"{self.pds}/xrpc/app.bsky.feed.searchPosts?{params}"
        try:
            raw = self.fetch.get(url, {"Authorization": f"Bearer {self._access}"})
        except FetchError as exc:
            if "HTTP 400" not in str(exc) and "HTTP 401" not in str(exc):
                raise
            self._login()  # expired access token: log in again once
            raw = self.fetch.get(url, {"Authorization": f"Bearer {self._access}"})
        return json.loads(raw).get("posts", [])


def parse_bsky_post(p: dict) -> LiveItem:
    rec = p.get("record", {})
    handle = p.get("author", {}).get("handle", "")
    rkey = p["uri"].rsplit("/", 1)[-1]
    created = datetime.fromisoformat(rec.get("createdAt", p.get("indexedAt")).replace("Z", "+00:00"))
    return LiveItem(id=f"bluesky:{p['uri']}", source="bluesky", kind="social", text=rec.get("text", ""),
                    url=f"https://bsky.app/profile/{handle}/post/{rkey}", published_at=_utc(created),
                    lang=(rec.get("langs") or [None])[0], author=handle)


def poll_bluesky(client: BlueskyClient, now: datetime, queries=BSKY_QUERIES) -> list[LiveItem]:
    items = []
    for q in queries:
        items += [parse_bsky_post(p) for p in client.search(q, now - timedelta(hours=24))]
    return [i for i in items if is_flood_related(i.text)]


# ---------------------------------------------------------------- store
SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY, source TEXT, kind TEXT, text TEXT, url TEXT, published_at TEXT, fetched_at TEXT,
  lang TEXT, author TEXT, extra TEXT, content_hash TEXT);
CREATE INDEX IF NOT EXISTS items_hash ON items(content_hash);
CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, seen_at TEXT);
CREATE TABLE IF NOT EXISTS source_status (
  source TEXT PRIMARY KEY, state TEXT, last_attempt TEXT, last_success TEXT, last_error TEXT,
  failures INTEGER DEFAULT 0, next_allowed TEXT, last_new INTEGER DEFAULT 0);
"""


class LiveStore:
    def __init__(self, path: str | Path = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def add(self, items: list[LiveItem], now: datetime | None = None) -> int:
        """Insert new items; duplicates (same id, or same text from another source) are skipped."""
        now = now or datetime.now(UTC)
        added = 0
        with self._conn() as c:
            for it in items:
                h = it.content_hash()
                if c.execute("SELECT 1 FROM items WHERE id=? OR (content_hash=? AND source!='simulated')",
                             (it.id, h)).fetchone():
                    continue
                c.execute("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (it.id, it.source, it.kind, it.text, it.url, _utc(it.published_at).isoformat(),
                           now.isoformat(), it.lang, it.author, json.dumps(it.extra, ensure_ascii=False), h))
                added += 1
        return added

    def mark_seen(self, ids: list[str], now: datetime | None = None) -> None:
        now = (now or datetime.now(UTC)).isoformat()
        with self._conn() as c:
            c.executemany("INSERT OR IGNORE INTO seen VALUES (?, ?)", [(i, now) for i in ids])

    def seen_ids(self) -> set[str]:
        with self._conn() as c:
            return {r[0] for r in c.execute("SELECT id FROM seen")} | {r[0] for r in c.execute("SELECT id FROM items")}

    def items(self, since: datetime | None = None, kinds: tuple[str, ...] | None = None) -> list[LiveItem]:
        q, args = "SELECT * FROM items WHERE 1=1", []
        if since:
            q += " AND published_at >= ?"
            args.append(_utc(since).isoformat())
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += list(kinds)
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY published_at DESC", args).fetchall()
        return [LiveItem(id=r["id"], source=r["source"], kind=r["kind"], text=r["text"], url=r["url"],
                         published_at=datetime.fromisoformat(r["published_at"]), lang=r["lang"], author=r["author"],
                         extra=json.loads(r["extra"] or "{}")) for r in rows]

    def clear_source(self, source: str) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM items WHERE source=?", (source,)).rowcount

    def status(self) -> dict[str, dict]:
        with self._conn() as c:
            return {r["source"]: dict(r) for r in c.execute("SELECT * FROM source_status")}

    def set_status(self, source: str, **fields) -> None:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO source_status (source) VALUES (?)", (source,))
            for k, v in fields.items():
                c.execute(f"UPDATE source_status SET {k}=? WHERE source=?", (v, source))

    def last_updated(self) -> datetime | None:
        with self._conn() as c:
            row = c.execute("SELECT MAX(last_success) FROM source_status").fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None


# ---------------------------------------------------------------- polling
def interval_seconds() -> int:
    return max(60, int(os.getenv("AQUAALERT_LIVE_INTERVAL", "600")))


def backoff(failures: int, base: int) -> timedelta:
    return timedelta(seconds=min(base * 2 ** max(failures - 1, 0), 6 * 3600))


class Poller:
    """Polls each source when due; a failing source backs off without affecting the others."""

    def __init__(self, store: LiveStore, fetch: PoliteFetcher | None = None, interval: int | None = None,
                 bluesky: BlueskyClient | None | bool = True) -> None:
        self.store, self.fetch = store, fetch or PoliteFetcher()
        self.interval = interval or interval_seconds()
        if bluesky is True:
            handle, pw = os.getenv("BLUESKY_HANDLE", "").strip(), os.getenv("BLUESKY_APP_PASSWORD", "").strip()
            bluesky = BlueskyClient(self.fetch, handle, pw, os.getenv("BLUESKY_PDS") or BSKY_PDS) if handle and pw else None
        self.bluesky = bluesky or None

    def _due(self, source: str, now: datetime) -> bool:
        st = self.store.status().get(source, {})
        nxt = st.get("next_allowed")
        return not nxt or datetime.fromisoformat(nxt) <= now

    def poll_once(self, now: datetime | None = None, force: bool = False) -> dict[str, str]:
        now = now or datetime.now(UTC)
        results: dict[str, str] = {}
        for source in SOURCES:
            if source == "bluesky" and self.bluesky is None:
                self.store.set_status("bluesky", state="not configured", last_error=None)
                results[source] = "not configured"
                continue
            if not force and not self._due(source, now):
                results[source] = "not due"
                continue
            prev = self.store.status().get(source, {})
            self.store.set_status(source, last_attempt=now.isoformat())
            try:
                note = None
                if source == "sachet":
                    items, examined, feed_errors = poll_sachet(self.fetch, self.store.seen_ids())
                    self.store.mark_seen(examined, now)
                    if feed_errors:
                        note = f"{len(feed_errors)} feed(s) failed: " + ", ".join(feed_errors)
                elif source == "news":
                    items = poll_news(self.fetch)
                else:
                    items = poll_bluesky(self.bluesky, now)
                new = self.store.add(items, now)
                self.store.set_status(source, state="ok", last_success=now.isoformat(), last_error=note, failures=0,
                                      last_new=new, next_allowed=(now + timedelta(seconds=self.interval)).isoformat())
                results[source] = f"ok: {new} new"
            except Exception as exc:  # each source fails on its own
                failures = (prev.get("failures") or 0) + 1
                state = "blocked" if "robots.txt" in str(exc) and "blocked" in str(exc) else "error"
                self.store.set_status(source, state=state, last_error=str(exc)[:300], failures=failures,
                                      next_allowed=(now + backoff(failures, self.interval)).isoformat())
                results[source] = f"{state}: {exc}"
        return results

    def run_forever(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            self.poll_once()
            stop.wait(30)  # check every 30 s which sources are due


# ---------------------------------------------------------------- to posts
def to_posts(items: list[LiveItem]) -> list[Post]:
    """News and social items -> Posts for the normal pipeline (official alerts are shown separately)."""
    posts = []
    for it in items:
        if it.kind == "official_alert":
            continue
        posts.append(Post(id=it.id, text=it.text, timestamp=it.published_at.astimezone(UTC).replace(tzinfo=None),
                          source=it.source, author=it.author, url=it.url))
    return posts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    store = LiveStore()
    poller = Poller(store)
    if a.loop:
        poller.run_forever()
        return
    for source, result in poller.poll_once(force=True).items():
        print(f"{source:8s} {result}")
    day = datetime.now(UTC) - timedelta(hours=24)
    items = store.items(since=day)
    print(f"items in the last 24 h: {len(items)}",
          {s: sum(i.source == s for i in items) for s in (*SOURCES, "simulated")})


if __name__ == "__main__":
    main()

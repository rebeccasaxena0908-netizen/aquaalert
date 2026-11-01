"""Simulated incident for Live-mode demos: clearly synthetic posts injected into the live store.

Every post has source="simulated", an id starting with "sim-" and no source URL; the dashboard shows a SIMULATED
badge on them everywhere and leaves them out of the real per-source counts. ``clear`` removes them in one call.
Four rescue requests fall in one district (Alappuzha: Chengannur, Kuttanad, Edathua), so the post-volume
spike logic in fuse/ can react.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aquaalert.ingest.live import LiveItem, LiveStore

SIM_SOURCE = "simulated"
SIM_POSTS: tuple[tuple[str, str, int], ...] = (  # (text, language hint, minutes before "now")
    ("Hum Chengannur me ghar ki chhat par phanse hain, 7 log hain, paani badh raha hai, jaldi boat bhejo", "hi-Latn", 18),
    ("ചെങ്ങന്നൂരിൽ ഞങ്ങൾ 12 പേർ വീടിന്റെ മുകളിൽ കുടുങ്ങി കിടക്കുന്നു, ദയവായി ബോട്ട് അയക്കൂ", "ml", 15),
    ("Need rescue! Family of 5 stuck on the first floor in Kuttanad, water entering the house", "en", 12),
    ("SOS: 6 people trapped on a roof at Edathua, elderly woman with us, please send a boat", "en", 11),
    ("We have 3 boats and 10 volunteers ready at Alappuzha boat jetty for rescue, DM us", "en", 10),
    ("Relief camp at Pandalam school needs drinking water and medicines for 120 people", "en", 8),
    ("Missing: my father Joseph (68) last seen near Aluva bus stand this morning, please share", "en", 6),
    ("BREAKING: Idukki dam has burst!! Water will reach Kochi in 1 hour. Forward to all groups before it is deleted!!",
     "en", 3),
)


def simulated_items(now: datetime | None = None) -> list[LiveItem]:
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%H%M%S")
    return [LiveItem(id=f"sim-{stamp}-{i}", source=SIM_SOURCE, kind="social", text=text, url="",
                     published_at=now - timedelta(minutes=mins), lang=lang, author="simulated",
                     extra={"simulated": True})
            for i, (text, lang, mins) in enumerate(SIM_POSTS, 1)]


def inject(store: LiveStore, now: datetime | None = None) -> int:
    return store.add(simulated_items(now), now)


def clear(store: LiveStore) -> int:
    return store.clear_source(SIM_SOURCE)


def is_simulated(post_or_item) -> bool:
    return getattr(post_or_item, "source", None) == SIM_SOURCE

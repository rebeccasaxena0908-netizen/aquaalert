"""Theme helpers for the dashboard: CSS injection and small HTML builders (all classes prefixed aa-).

Everything returns single-line HTML: Streamlit's markdown turns indented lines into code blocks and treats "$" as
math, so text is always passed through ``esc``.
"""
from __future__ import annotations

import html
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSS_FILES = ("styles.css", "app.css")
PAUSE_CSS = ".aa-blob, .aa-bubble, .aa-bubble i { animation-play-state: paused !important; }"
DROP_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#fff" d="M12 2.5c-.3 0-.6.2-.8.4C9.6 5 5 '
             '10.7 5 14.6 5 18.7 8.1 22 12 22s7-3.3 7-7.4c0-3.9-4.6-9.6-6.2-11.7-.2-.2-.5-.4-.8-.4zm-3.1 12.6c.5 0 '
             '.9.4.9.9 0 1.2 1 2.2 2.2 2.2.5 0 .9.4.9.9s-.4.9-.9.9c-2.2 0-4-1.8-4-4 0-.5.4-.9.9-.9z"/></svg>')


def esc(text: object) -> str:
    """HTML-escape and neutralise "$" (Streamlit markdown would start LaTeX math)."""
    return html.escape(str(text)).replace("$", "&#36;")


def theme_css(paused: bool = False) -> str:
    css = "\n".join((HERE / f).read_text() for f in CSS_FILES)
    return f"<style>{css}{PAUSE_CSS if paused else ''}</style>"


def sky_html(n_bubbles: int = 22, seed: int = 7) -> str:
    """Fixed background layer: 4 drifting blobs + rising bubbles with seeded random inline styles."""
    rng = random.Random(seed)  # fixed seed: bubbles don't jump around on every rerun
    blobs = "".join(f'<span class="aa-blob aa-b{i}"></span>' for i in range(1, 5))
    bubbles = []
    for _ in range(n_bubbles):
        size = rng.randint(8, 40)
        left = rng.uniform(0, 97)
        rise, rise_delay = rng.uniform(14, 28), -rng.uniform(0, 28)
        wobble, wobble_delay = rng.uniform(2, 4), -rng.uniform(0, 4)
        bubbles.append(
            f'<span class="aa-bubble" style="width:{size}px;height:{size}px;left:{left:.1f}vw;'
            f'animation-duration:{rise:.1f}s;animation-delay:{rise_delay:.1f}s">'
            f'<i style="animation-duration:{wobble:.1f}s;animation-delay:{wobble_delay:.1f}s"></i></span>')
    return f'<div class="aa-sky" aria-hidden="true">{blobs}{"".join(bubbles)}</div>'


def chip(text: str, active: bool = False) -> str:
    return f'<span class="aa-chip{" aa-active" if active else ""}">{esc(text)}</span>'


def header_html(chips: list[tuple[str, bool]]) -> str:
    return (f'<div class="aa-glass aa-header"><div class="aa-logo">{DROP_ICON}</div>'
            f'<div class="aa-brand"><span class="aa-wordmark">AquaAlert</span>'
            f'<span class="aa-subtitle">Multilingual AI for flood crisis response</span></div>'
            f'<div class="aa-chips">{"".join(chip(t, a) for t, a in chips)}</div></div>')


def kpi_html(cards: list[dict]) -> str:
    """cards: {key, label, value, note, tone in {"accent", "err", "warn", ""}}; data-kpi lets tests read values."""
    out = []
    for c in cards:
        tone = {"accent": " aa-accent", "err": " aa-err-num", "warn": " aa-warn-num"}.get(c.get("tone", ""), "")
        out.append(f'<div class="aa-glass aa-kpi" data-kpi="{esc(c["key"])}" data-value="{esc(c["value"])}">'
                   f'<span class="aa-label">{esc(c["label"])}</span><span class="aa-big{tone}">{esc(c["value"])}</span>'
                   f'<span class="aa-kpi-note">{esc(c["note"])}</span></div>')
    return f'<div class="aa-kpis">{"".join(out)}</div>'


def badge(text: str, kind: str = "neu") -> str:
    return f'<span class="aa-badge aa-{kind}">{esc(text)}</span>'


def label_badge(label: str | None) -> str:
    """Category badge: only rescue requests use a status colour (red); other categories are neutral."""
    if not label:
        return ""
    return badge(label.replace("_", " "), "err" if label == "rescue_needed" else "neu")


def level_badge(level: str) -> str:
    return badge(level.upper(), {"warning": "err", "watch": "warn"}.get(level, "ok"))


def cred_badge(score: float | None, threshold: float) -> str:
    """Never claims a post is verified: 0 means only that no heuristic fired."""
    if score is None:
        return ""
    if score >= threshold:
        return badge(f"low credibility {score:.2f}", "warn")
    if score > 0:
        return badge(f"minor flags {score:.2f}", "neu")
    return badge("no red flags", "ok")


def rule_badge(rule: str | None) -> str:
    """Neutral: the status colours are reserved for rescue/WARNING, WATCH/low credibility and OK."""
    return badge(f"rule: {rule.replace('_', ' ')}", "neu") if rule else ""


def tag(text: str) -> str:
    return f'<span class="aa-tag">{esc(text)}</span>'


def bar(fraction: float) -> str:
    pct = max(0.0, min(1.0, fraction)) * 100
    return f'<div class="aa-bar"><span style="width:{pct:.0f}%"></span></div>'


def meter(value: float) -> str:
    return f'<div class="aa-meter">{bar(value)}<span class="aa-tag">{value:.2f}</span></div>'


class Html(str):
    """A table cell that is already HTML (not escaped)."""


def table_html(columns: list[str], rows: list[list], numeric: set[str] = frozenset()) -> str:
    head = "".join(f'<th class="{"aa-num" if c in numeric else ""}">{esc(c)}</th>' for c in columns)
    body = []
    for row in rows:
        cells = []
        for c, v in zip(columns, row):
            content = v if isinstance(v, Html) else ("–" if v is None else esc(v))
            cls = "aa-num" if c in numeric else ("aa-text-cell" if isinstance(v, str) and len(v) > 60 else "")
            cells.append(f'<td class="{cls}">{content}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (f'<div class="aa-glass aa-table-wrap"><table class="aa-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def section(title: str, note: str = "") -> str:
    note_html = f'<span class="aa-muted">{esc(note)}</span>' if note else ""
    return f'<div class="aa-section"><span class="aa-panel-title">{esc(title)}</span>{note_html}</div>'


def card(title_html: str, body_html: str) -> str:
    return f'<div class="aa-glass aa-card"><div class="aa-card-head">{title_html}</div>{body_html}</div>'


def legend_html(colors: dict[str, str], counts: dict[str, int]) -> str:
    rows = "".join(f'<div class="aa-legend-row"><span class="aa-dot" style="background:{c}"></span>'
                   f'{esc(l.replace("_", " "))}<span class="aa-count">{counts.get(l, 0)}</span></div>' for l, c in colors.items())
    rows += ('<div class="aa-legend-row"><span class="aa-dot aa-hollow"></span>doubtful (low credibility)</div>')
    return f'<div class="aa-glass aa-legend"><span class="aa-label">Legend</span>{rows}</div>'

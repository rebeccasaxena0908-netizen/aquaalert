import re

from dashboard import theme as T


def test_css_files_carry_tokens_fonts_and_motion_rules():
    css = T.theme_css()
    assert css.startswith("<style>") and css.endswith("</style>")
    for token in ("--paper:#FCFDFF", "--navy:#35659E", "--ink:#212936", "--ok:#2F8F6E", "--warn:#C98A2B", "--err:#C4484F"):
        assert token in css.replace(" ", ""), token
    assert "family=Montserrat" in css and "Karla" in css and "JetBrains+Mono" in css
    assert "prefers-reduced-motion: reduce" in css and "pointer-events: none" in css
    assert "animation-play-state: paused" not in T.theme_css(paused=False).split("prefers-reduced-motion")[0]
    assert T.PAUSE_CSS in T.theme_css(paused=True)
    # every custom class is aa- prefixed
    rules = re.sub(r"@import url\([^)]*\);|/\*.*?\*/", "", css, flags=re.S)  # font URL + comments ("app.css")
    streamlit_own = {"stApp", "block-container", "stButton", "stDownloadButton", "stTabs", "react-aria-SelectionIndicator"}
    custom = set(re.findall(r"\.([a-z][\w-]*)", rules)) - streamlit_own
    assert custom and all(c.startswith("aa-") for c in custom), custom - {c for c in custom if c.startswith("aa-")}


def test_sky_is_seeded_and_within_ranges():
    a, b = T.sky_html(), T.sky_html()
    assert a == b  # fixed seed: no jumping on reruns
    assert T.sky_html(seed=1) != a
    assert a.count('class="aa-blob') == 4 and 18 <= a.count('class="aa-bubble"') <= 24
    for size, left, rise, rdelay, wob, wdelay in re.findall(
            r'width:(\d+)px;height:\d+px;left:([\d.]+)vw;animation-duration:([\d.]+)s;animation-delay:(-[\d.]+)s">'
            r'<i style="animation-duration:([\d.]+)s;animation-delay:(-[\d.]+)s"', a):
        assert 8 <= int(size) <= 40 and 0 <= float(left) <= 97
        assert 14 <= float(rise) <= 28 and -28 <= float(rdelay) <= 0
        assert 2 <= float(wob) <= 4 and -4 <= float(wdelay) <= 0
    assert 'aria-hidden="true"' in a and "\n" not in a  # one line: markdown must not see indented code


def test_escaping_blocks_html_and_latex():
    assert T.esc("<b>$45,027</b>") == "&lt;b&gt;&#36;45,027&lt;/b&gt;"
    table = T.table_html(["text", "n"], [["<script>alert(1)</script> $5", 3]], numeric={"n"})
    assert "<script>" not in table and "&#36;5" in table and 'class="aa-num">3<' in table
    assert T.table_html(["x"], [[T.Html('<span class="aa-badge aa-ok">ok</span>')]]).count("aa-badge") == 1


def test_badge_semantics():
    assert "aa-err" in T.label_badge("rescue_needed") and "aa-neu" in T.label_badge("offer_help")
    assert "aa-err" in T.level_badge("warning") and "aa-warn" in T.level_badge("watch") and "aa-ok" in T.level_badge("normal")
    assert "aa-warn" in T.cred_badge(0.75, 0.5) and "aa-neu" in T.cred_badge(0.2, 0.5)
    assert "no red flags" in T.cred_badge(0.0, 0.5) and "verified" not in T.cred_badge(0.0, 0.5)
    assert "aa-neu" in T.rule_badge("distress_strong") and T.rule_badge(None) == ""


def test_header_kpis_and_bar():
    h = T.header_html([("Replay · 15 Aug", True), ("MuRIL", False)])
    assert "AquaAlert" in h and "Multilingual AI for flood crisis response" in h
    assert len(re.findall(r'class="aa-chip[ "]', h)) == 2
    assert 'aa-chip aa-active">Replay' in h
    k = T.kpi_html([{"key": "posts", "label": "Posts analysed", "value": 110, "note": "x", "tone": "accent"}])
    assert 'data-kpi="posts" data-value="110"' in k and "aa-big aa-accent" in k
    assert 'width:100%' in T.bar(1.7) and 'width:0%' in T.bar(-1)

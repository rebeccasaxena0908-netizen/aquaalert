"""Build aquaalert/extract/gazetteer_india.csv from Wikidata (CC0) through the public QLever SPARQL endpoint.

    python scripts/build_gazetteer.py

Covers districts, towns (population >= MIN_POPULATION, or district headquarters) and named rivers of Assam, Odisha,
West Bengal, Bihar and Uttar Pradesh. Every row keeps its Wikidata id, so each coordinate can be checked. Places
without a coordinate on Wikidata are skipped, never estimated. Requests go through the robots.txt-aware PoliteFetcher.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
import urllib.parse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aquaalert.ingest.live import PoliteFetcher  # noqa: E402

ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"
OUT = ROOT / "aquaalert" / "extract" / "gazetteer_india.csv"
STATES = {"Q1164": "Assam", "Q22048": "Odisha", "Q1356": "West Bengal", "Q1165": "Bihar", "Q1498": "Uttar Pradesh"}
MIN_POPULATION = 50_000
EXTRA_TOWNS = ("Bagaha",)  # flood-prone towns Wikidata has no population figure for
# Rivers by Wikidata id (labels are ambiguous: there are two Kosi rivers). Wikidata stores a river's source and mouth, so
# a river gets a pin only if one of those points lies within RIVER_MAX_KM of a covered district headquarters. No midpoints
# are computed: other rivers are still recognised in text (blank coordinates) but get no pin.
RIVERS = {"Q629894": "sankosh", "Q45403": "brahmaputra", "Q5089": "ganga", "Q695610": "kosi", "Q18340305": "gandak",
          "Q390232": "ghaghara", "Q132726": "yamuna", "Q271044": "mahanadi", "Q1709162": "subansiri", "Q907883": "teesta",
          "Q1971511": "damodar", "Q739687": "hooghly river", "Q1709169": "barak", "Q2015702": "manas",
          "Q3773157": "baitarani", "Q2639171": "brahmani", "Q477676": "gomti", "Q2428539": "mahananda",
          "Q2066287": "torsa", "Q6126710": "jaldhaka", "Q1755951": "dhansiri", "Q7380422": "rupnarayan",
          "Q3822918": "subarnarekha", "Q1418174": "son river", "Q4461769": "bagmati", "Q3630468": "burhi gandak"}
RIVER_MAX_KM = 80.0
# Wikidata's nearest point for these is the mouth in Bangladesh, which would misplace an Indian flood report
NO_PIN_RIVERS = {"teesta", "torsa"}
# Other English spellings of a place already in the table (same Wikidata row, same coordinate)
SPELLINGS = {"allahabad": "prayagraj", "calcutta": "kolkata", "burdwan": "bardhaman", "gauhati": "guwahati",
             "banaras": "varanasi", "benares": "varanasi", "ara": "arrah", "ganges": "ganga"}
# Historic districts, statistical areas and labels that are not useful place names; Aurangabad is left out because the
# name almost always means the Maharashtra city, outside this table.
EXCLUDE = {"24 parganas", "manbhum", "west dinajpur", "kolkata metropolitan area", "nadia dakshin", "deo town",
           "kharagpur railway settlement", "garamur phutuki chapori", "aurangabad"}
# Different places sharing a name are dropped unless within this distance (a district and its headquarters)
SAME_PLACE_KM = 60.0
# Everyday words in English / Hindi / Bengali: matched in Latin script only when capitalised ("aa gaya" != Gaya)
CASE_SENSITIVE = {"gaya", "mau", "basti", "banda", "puri", "saran", "nadia", "bally", "suri", "manas", "barak", "orai",
                  "banka", "kulti", "bhiti", "kosi", "torsa", "etah", "hapur", "gonda", "arwal", "jamui", "sambhal", "vaishali",
                  "kalyani", "ara", "sonari", "son river"}
# Native labels that are everyday words; never used as aliases
BAD_ALIASES = {"गया", "मऊ", "बस्ती", "बांदा", "पुरी", "सारण", "आरा", "बाँका", "बांका", "मानस", "नदिया", "গয়া", "বাঁকা", "পুরী",
               "ওরাই", "বস্তী", "বাস্তি", "বান্দা", "বালি", "बाली", "बक्सा", "हालि", "संभल", "মানস"}
KERALA_QID = "Q1186"
# Coarse (south, west, north, east) sanity boxes: Wikidata has errors, e.g. Banka's headquarters resolved to a point in
# Slovakia. These only reject a coordinate; they never supply one.
STATE_BOXES = {"Assam": (24.0, 89.6, 28.2, 96.1), "Odisha": (17.7, 81.3, 22.7, 87.6), "West Bengal": (21.4, 85.7, 27.3, 89.95),
               "Bihar": (24.2, 83.2, 27.6, 88.4), "Uttar Pradesh": (23.8, 77.0, 30.5, 84.7)}
PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
LABELS = """OPTIONAL { ?p rdfs:label ?hi . FILTER(LANG(?hi) = "hi") }
OPTIONAL { ?p rdfs:label ?bn . FILTER(LANG(?bn) = "bn") }"""
_STATE_VALUES = " ".join(f"wd:{q}" for q in STATES)


def query(fetch: PoliteFetcher, sparql: str) -> list[dict]:
    url = f"{ENDPOINT}?{urllib.parse.urlencode({'query': PREFIXES + sparql})}"
    raw = fetch.get(url, {"Accept": "application/sparql-results+json"})
    return [{k: v["value"] for k, v in row.items()} for row in json.loads(raw)["results"]["bindings"]]


def districts_query() -> str:
    return f"""SELECT DISTINCT ?p ?en ?coord ?state ?hi ?bn WHERE {{
  VALUES ?state {{ {_STATE_VALUES} }}
  ?p wdt:P31 wd:Q1149652 ; wdt:P131+ ?state ; wdt:P625 ?coord ; rdfs:label ?en . FILTER(LANG(?en) = "en")
  {LABELS} }}"""


def towns_query() -> str:
    return f"""SELECT DISTINCT ?p ?en ?coord ?state ?hi ?bn ?pop ?district ?districten WHERE {{
  VALUES ?state {{ {_STATE_VALUES} }}
  ?p wdt:P1082 ?pop ; wdt:P131+ ?state ; wdt:P625 ?coord ; rdfs:label ?en . FILTER(LANG(?en) = "en")
  FILTER(?pop >= {MIN_POPULATION})
  ?p wdt:P31/wdt:P279* wd:Q486972 .
  OPTIONAL {{ ?p wdt:P131+ ?district . ?district wdt:P31 wd:Q1149652 ; rdfs:label ?districten . FILTER(LANG(?districten) = "en") }}
  {LABELS} }}"""


def headquarters_query() -> str:
    return f"""SELECT DISTINCT ?p ?en ?coord ?state ?hi ?bn ?district ?districten WHERE {{
  VALUES ?state {{ {_STATE_VALUES} }}
  ?district wdt:P31 wd:Q1149652 ; wdt:P131+ ?state ; wdt:P36 ?p ; rdfs:label ?districten . FILTER(LANG(?districten) = "en")
  ?p wdt:P625 ?coord ; rdfs:label ?en . FILTER(LANG(?en) = "en")
  {LABELS} }}"""


def rivers_query() -> str:
    ids = " ".join(f"wd:{q}" for q in RIVERS)
    return f"""SELECT DISTINCT ?p ?en ?coord ?hi ?bn WHERE {{
  VALUES ?p {{ {ids} }}
  ?p wdt:P625 ?coord ; rdfs:label ?en . FILTER(LANG(?en) = "en")
  {LABELS} }}"""


def extras_query() -> str:
    names = " ".join(f'"{n}"@en' for n in EXTRA_TOWNS)
    return f"""SELECT DISTINCT ?p ?en ?coord ?state ?hi ?bn ?district ?districten WHERE {{
  VALUES ?state {{ {_STATE_VALUES} }}
  VALUES ?en {{ {names} }}
  ?p rdfs:label ?en ; wdt:P131+ ?state ; wdt:P625 ?coord .
  OPTIONAL {{ ?p wdt:P131+ ?district . ?district wdt:P31 wd:Q1149652 ; rdfs:label ?districten . FILTER(LANG(?districten) = "en") }}
  {LABELS} }}"""


def kerala_check_query(names: list[str]) -> str:
    values = " ".join(f'"{n}"@en' for n in names)
    return f"""SELECT DISTINCT ?en ?coord WHERE {{
  VALUES ?en {{ {values} }}
  ?p rdfs:label ?en ; wdt:P131+ wd:{KERALA_QID} ; wdt:P625 ?coord . }}"""


def parse_point(wkt: str) -> tuple[float, float] | None:
    m = re.match(r"Point\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)", wkt, re.I)
    return (round(float(m.group(2)), 4), round(float(m.group(1)), 4)) if m else None


def clean_name(label: str) -> str:
    """Canonical lowercase ASCII-ish name: "Kanpur Nagar district" -> "kanpur nagar", "Deoria, Uttar Pradesh" -> "deoria"."""
    name = re.sub(r"\s+(district|river|city)$", "", label.split(",")[0].strip(), flags=re.I)
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return name.lower()


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    from aquaalert.match.geo import haversine_km

    return haversine_km(a[0], a[1], b[0], b[1])


def native_alias(label: str | None) -> str | None:
    """A usable native-script label: qualifiers removed, no Latin text, no everyday words, at least 4 code points."""
    if not label:
        return None
    alias = unicodedata.normalize("NFC", label).split(",")[0].strip()
    alias = re.sub(r"\s*\(.*\)\s*$", "", alias)
    alias = re.sub(r"\s*(जिला|ज़िला|जिले|ज़िले|नदी|নদী|নদ|জেলা|शहर|শহর)\s*$", "", alias)
    bad = {unicodedata.normalize("NFC", a) for a in BAD_ALIASES}
    if len(alias) < 4 or alias in bad or re.search(r"[A-Za-z0-9]", alias):
        return None
    return alias


def build(raw: dict[str, list[dict]]) -> tuple[list[dict], dict[str, list[str]]]:
    """Merge the query results into one row per name. Returns (rows, skipped reasons)."""
    skipped: dict[str, list[str]] = defaultdict(list)
    q = lambda uri: uri.rsplit("/", 1)[-1]  # noqa: E731
    hq_points = [parse_point(r["coord"]) for r in raw["headquarters"] if parse_point(r["coord"])]
    candidates: dict[str, dict[str, dict]] = defaultdict(dict)  # name -> qid -> row
    # towns before districts: the town's own point is better than a district's centroid
    for kind in ("headquarters", "extra", "town", "district"):
        for r in sorted(raw[kind], key=lambda r: (r["p"], r["coord"])):
            pt = parse_point(r["coord"])
            if pt is None:
                continue
            name, qid = clean_name(r["en"]), q(r["p"])
            if name in EXCLUDE:
                continue
            south, west, north, east = STATE_BOXES[STATES[q(r["state"])]]
            if not (south <= pt[0] <= north and west <= pt[1] <= east):
                skipped["coordinate outside its state"].append(f"{name} ({qid})")
                continue
            row = candidates[name].setdefault(qid, {
                "name": name, "lat": pt[0], "lon": pt[1], "state": STATES[q(r["state"])], "qid": qid,
                "kind": "district" if kind == "district" else "town", "district": "", "aliases": set()})
            district = r.get("districten")
            if kind == "district":
                row["district"] = re.sub(r"\s+district$", "", r["en"], flags=re.I)
            elif district and (not row["district"] or clean_name(district) == name):
                row["district"] = re.sub(r"\s+district$", "", district, flags=re.I)
            row["aliases"] |= {a for a in (native_alias(r.get("hi")), native_alias(r.get("bn"))) if a}
    rows: dict[str, dict] = {}
    for name, by_qid in candidates.items():
        entities = list(by_qid.values())
        first = entities[0]
        # a district entity in the same state is that town's district, wherever its point is; anything else far away is
        # a different place with the same name, and the name is dropped rather than guessed
        if any(km((first["lat"], first["lon"]), (e["lat"], e["lon"])) > SAME_PLACE_KM
               and (e["state"] != first["state"] or e["kind"] != "district") for e in entities[1:]):
            skipped["ambiguous name (different places)"].append(name)
            continue
        for e in entities[1:]:  # a district and its headquarters town: same place
            first["aliases"] |= e["aliases"]
            first["district"] = first["district"] or e["district"]
        first["district"] = first["district"] or (name.title() if first["kind"] == "district" else "")
        rows[name] = first
    by_river: dict[str, list[dict]] = defaultdict(list)
    for r in raw["river"]:
        by_river[q(r["p"])].append(r)
    for qid, name in RIVERS.items():
        points = [parse_point(r["coord"]) for r in by_river.get(qid, []) if parse_point(r["coord"])]
        scored = sorted((min(km(pt, hq) for hq in hq_points), pt) for pt in points)
        pinned = bool(scored) and scored[0][0] <= RIVER_MAX_KM and name not in NO_PIN_RIVERS
        if not pinned:
            skipped["river recognised in text, no pin"].append(name)
        short = name.removesuffix(" river")
        key = short if short not in rows and name != "son river" else name
        aliases = {a for r in by_river[qid] for a in (native_alias(r.get("hi")), native_alias(r.get("bn"))) if a}
        rows[key] = {"name": key, "lat": scored[0][1][0] if pinned else "", "lon": scored[0][1][1] if pinned else "",
                     "state": "", "qid": qid, "kind": "river", "district": "", "aliases": aliases}
    for spelling, name in SPELLINGS.items():
        if name in rows:
            rows[spelling] = {**rows[name], "name": spelling, "aliases": set()}
        else:
            skipped["spelling of a place not in the table"].append(spelling)
    # an alias claimed by places far apart identifies neither; nearby owners (Kanpur / Kanpur Nagar) share it on the shortest
    owners: dict[str, list[str]] = defaultdict(list)
    for name, row in rows.items():
        for a in row["aliases"]:
            owners[a].append(name)
    for alias, names in owners.items():
        if len(names) < 2:
            continue
        names.sort(key=len)
        pts = [(rows[n]["lat"], rows[n]["lon"]) for n in names]
        close = all(pts[0][0] != "" and p[0] != "" and km(pts[0], p) <= SAME_PLACE_KM for p in pts[1:])
        if not close:
            skipped["alias shared by places far apart"].append(alias)
        for n in names[1:] if close else names:
            rows[n]["aliases"].discard(alias)
    return sorted(rows.values(), key=lambda r: r["name"]), skipped


def write_csv(rows: list[dict], path: Path = OUT) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "lat", "lon", "district", "state", "kind", "wikidata", "case_sensitive", "aliases"])
        for r in rows:
            w.writerow([r["name"], r["lat"], r["lon"], r["district"], r["state"], r["kind"], r["qid"],
                        int(r["name"] in CASE_SENSITIVE), "|".join(sorted(r["aliases"]))])


def main() -> None:
    fetch = PoliteFetcher()
    raw = {"district": query(fetch, districts_query()), "headquarters": query(fetch, headquarters_query()),
           "town": query(fetch, towns_query()), "extra": query(fetch, extras_query()),
           "river": query(fetch, rivers_query())}
    print({k: len(v) for k, v in raw.items()})
    rows, skipped = build(raw)
    write_csv(rows)
    kinds = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    print(f"wrote {len(rows)} places to {OUT}: {dict(kinds)}; aliases: {sum(len(r['aliases']) for r in rows)}")
    for reason, names in skipped.items():
        print(f"skipped, {reason}: {sorted(names)}")
    from aquaalert.extract.gazetteer import KERALA_PLACES

    found: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in query(fetch, kerala_check_query([n.title() for n in KERALA_PLACES])):
        pt = parse_point(r["coord"])
        if pt:
            found[r["en"].lower()].append(pt)
    far = sorted(((min(km(c, p) for p in found[n]), n) for n, c in KERALA_PLACES.items() if n in found), reverse=True)
    print(f"Kerala check: {len(far)}/{len(KERALA_PLACES)} found on Wikidata; largest gaps (km):",
          [(n, round(d, 1)) for d, n in far[:6]], "| not found:", sorted(set(KERALA_PLACES) - set(found)))


if __name__ == "__main__":
    main()

# AquaAlert — Multilingual AI for Flood Crisis Response

Turns flood-time posts (English, Hindi/Hinglish, Bengali, Malayalam, Assamese) into classified
help requests on a map. See [CLAUDE.md](CLAUDE.md) for the full architecture and roadmap.

## Status: all four phases
Every stage in CLAUDE.md is in place and runs with no downloads, training or API keys (synthetic sample data,
offline fallbacks). Trained MuRIL weights and LLM keys make it better, not possible.

## Architecture
```
            ┌──────────── data/sensors/*.csv (rainfall, river levels)
            │
ingest ─► language ─► classify ─► extract ─► credibility ─┬─► summarize ─┐
(CSV/JSON,  (script +   (MuRIL /    (places,   (heuristic  ├─► match       ├─► dashboard (Streamlit)
 replay)     romanised)  embed_lr /  phones,    misinfo     ├─► fuse ◄─────┘     api (FastAPI)
                         keyword +   counts,    score)      └─► chatbot (RAG over data/docs + briefs)
                         rules)      geocode)
```
Every stage reads and writes the shared Pydantic `Post` (`aquaalert/schemas.py`); `aquaalert/pipeline.py` runs
language → classify → extract → credibility over a list of posts.

| Module | Role | Key files | Needs |
|---|---|---|---|
| `ingest/` | Load CSV/JSON/JSONL datasets; replay a dataset in time order; **Live mode** polling of NDMA SACHET and Bluesky into SQLite (GDELT news is an experimental option, off by default); simulated incidents | `loader.py`, `replay.py`, `live.py`, `simulate.py` | Bluesky app password optional |
| `language/` | Language ID: native Indic scripts by Unicode block, romanised Hindi/Malayalam/Bengali by lexicon; code-mixing flag; per-language evaluation | `detect.py`, `evaluate.py` | – |
| `classify/` | 7-class crisis classifier. Fine-tuned MuRIL → embed_lr baseline → keyword fallback (auto); distress rule layer; HumAID label mapping; training + evaluation | `model.py`, `rules.py`, `train.py`, `embed_lr.py`, `label_map.py` | MuRIL weights (`scripts/fetch_model.py`) optional |
| `extract/` | Places (Latin + Malayalam/Devanagari/Bengali names), phones, people counts; offline gazetteer + cached Nominatim; district lookup | `entities.py`, `gazetteer.py`, `geocode.py` | – |
| `credibility/` | Heuristic 0-1 misinformation score with reasons (not a trained model) | `heuristics.py` | – |
| `match/` | Needs ↔ offers: multilingual sentence-embedding similarity + haversine proximity | `matcher.py`, `geo.py` | – |
| `summarize/` | District-wise daily briefs with cited posts; Gemini / Anthropic / OpenAI or offline template; cached | `brief.py`, `providers.py` | an LLM key optional |
| `fuse/` | District early warnings: post-volume spikes × rainfall / river-level thresholds, with reasons | `sensors.py`, `warnings.py` | sensor CSVs (samples included) |
| `chatbot/` | RAG over guideline documents + latest briefs: FAISS, multilingual MiniLM, audience-aware retrieval, cited answers | `docs.py`, `index.py`, `rag.py` | docs in `data/docs/` (`scripts/fetch_docs.py`) |
| `api/` | FastAPI service over all of the above | `main.py` | – |
| `dashboard/` | Streamlit + Folium: map + district risk layer, warnings, briefs, matches, chatbot | `app.py`, `helpers.py` | – |

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # torch/transformers are only needed to run or train MuRIL
cp .env.example .env                 # optional: GEMINI_API_KEY / HF_TOKEN etc.
python scripts/fetch_docs.py         # optional: NDMA / SDMA guideline documents for the chatbot
python scripts/fetch_model.py        # optional: the trained MuRIL model (needs HF_TOKEN)
pytest
```

### Dashboard
```bash
streamlit run dashboard/app.py       # http://localhost:8501
```
Look: a single light "blue sky + frosted glass" theme. Base theme in `dashboard/styles.css`, AquaAlert components
in `dashboard/app.css` (all classes `aa-`), HTML helpers in `dashboard/theme.py`, Streamlit colours in
`.streamlit/config.toml`. The animated background (drifting blobs, rising bubbles) pauses with the sidebar toggle and
respects the OS "reduce motion" setting. A 5-7 minute walkthrough is in [docs/demo_script.md](docs/demo_script.md).

Tabs: **Map** (posts coloured by category, doubtful posts dashed, optional district risk layer), **Warnings**
(district early warnings with reasons and sensor history), **Daily brief**, **Needs ↔ Offers**, **Chatbot**. The
sidebar sets the dataset(s), replay time, categories, languages and the low-credibility filter.

### API
```bash
uvicorn api.main:app --reload        # interactive docs: http://127.0.0.1:8000/docs  (ReDoc: /redoc)
```
| Endpoint | What it returns |
|---|---|
| `GET /health` | status, active classifier, post count, LLM provider, sensor readings |
| `GET /posts` | processed posts; filters `label` (repeatable), `language`, `district`, `min_confidence`, `max_misinfo`, `since`, `until`, `located`, `limit`, `offset` |
| `POST /classify` | `{"text": ...}` → label + scores, rule, language, places/phones/geo/district, text-level misinfo score |
| `GET /matches` | needs ↔ offers (`alpha`, `max_km`, `min_score`, `one_to_one`, `max_misinfo`) |
| `GET /briefs` | district briefs (`day`, `district`; `llm=true` uses the configured LLM, otherwise the free offline template) |
| `GET /warnings` | district early warnings for `day` (`level` filter) |
| `POST /chat/ask` | `{"question": ...}` → cited answer from guidelines + briefs |
| `GET /labels` | the 7 categories |

Posts are processed once at startup from `AQUAALERT_DATASET`. There is no authentication: run it locally or behind
your own gateway. Example: `curl -s localhost:8000/warnings?day=2018-08-15&level=warning`.

### CLIs
```bash
python -m aquaalert.ingest.replay data/sample_posts.csv --speed 600
python -m aquaalert.pipeline data/sample_posts.csv
python -m aquaalert.fuse.warnings --day 2018-08-15
python -m aquaalert.summarize.brief data/sample_posts.csv
python -m aquaalert.chatbot.ask "What should be in an emergency kit?"
python -m aquaalert.language.evaluate
python -m aquaalert.credibility.evaluate
python -m aquaalert.classify.train --train ... --val ... --test ... --label-dataset humaid --out models/muril  # on Colab/Kaggle
```
With `AQUAALERT_CLASSIFIER=auto` the classifier is MuRIL if `models/muril/` exists, then `embed_lr`, then the
keyword fallback.

## Live mode (ingest/live.py)
The sidebar **Dataset** selector marks every dataset: **LIVE** (polled sources), **REPLAY · SYNTHETIC** (the
Kerala 2018 sample, all made up) or **REPLAY** (your own files). Live mode polls in a background thread every
`AQUAALERT_LIVE_INTERVAL` seconds (default 600), stores everything in `data/cache/live.sqlite`, refreshes the page
when something new arrives, and shows "last updated", per-source counts and status, a source badge and a link to the
original on every post. When nothing flood-related arrived it says so ("No flood-related items in the last 24h") and
never pads the view with sample data. CLI: `python -m aquaalert.ingest.live --once` or `--loop`.

| Source | What | Access | Status |
|---|---|---|---|
| **NDMA SACHET** | CAP 1.2 alerts from IMD, CWC and state SDMAs. Feeds: `https://sachet.ndma.gov.in/cap_public_website/rss/rss_<feed>.xml` for `india`, `kerala`, `assam`, `bihar`, `odisha`, `uttarakhand`, `karnataka`, `maharashtra`, `gujarat`; each item links to its CAP XML. Flood, heavy-rain and river alerts are kept (event, severity, urgency, area, sender, effective/expires, headline, instruction, link). | Public, no key; no robots.txt (404 = allowed) | **Verified live on 2 Nov 2026** (first on 1 Nov: all 9 feeds fetched; 2 heavy-rain alerts kept (Andhra Pradesh SDMA, cyclone rain)). Multi-word states (Tamil Nadu, West Bengal, …) returned 404 for every spelling tried, so they come only via the all-India feed. Slow at times (a full poll took ~3 min). |
| **Bluesky** | `app.bsky.feed.searchPosts` for flood, cloudburst, waterlogging and glacial-lake terms in English, Hindi, Malayalam (`sort=latest`, last 24 h). Only text, time and link are kept: no handle or display name is stored or shown (the link uses the account's DID, and opening it on Bluesky shows the author there). | Needs a login: unauthenticated search returns 403. Session via `com.atproto.server.createSession` on `bsky.social`, kept in memory only | **Verified live on 2 Nov 2026** with an app password. Skipped quietly when the credentials are missing. |
| ~~Google News RSS~~ | – | **Not used**: `news.google.com/robots.txt` disallows `/` (incl. `/rss`) for generic agents | – |

**Experimental, not a live source: News (GDELT DOC 2.0).** The code polls English and Hindi flood headlines, but this has
never been verified live: on 2026-11-01 every request returned HTTP 429 even 12–42 s apart, and on 2026-11-02 a poll
returned nothing in 117 s. It is off by default and its card shows `off`; **News (GDELT): experimental** in the sidebar
turns it on (remembered in the store; the poller picks it up within about 30 s, and turning it off hides stored news).
CLI: `python -m aquaalert.ingest.live --once --news on`. Tested only on a constructed response.

**Official alerts on the map:** Live mode draws SACHET alerts as dashed outlines (a CAP polygon) or warning triangles
(a district), coloured by severity, with event, severity, area, expiry and a link to the original. A district marker sits
at the district's gazetteer point, usually its headquarters town rather than a true centroid (see Geocoding). Cancelled and
expired alerts, and alerts whose area can't be placed, are left off, and the caption under the map says how many and why;
nothing is placed by guesswork. **Most real alerts can't be placed today:** SACHET's CAP files carry no inline polygon but
link one (`FetchPolygonXMLFile`), and that URL answers HTTP 403 to this client (checked 2026-11-02; it is not worked
around). Their `areaDesc` is often a file name ("heavy_rainfall_2026-11-02") or a taluka outside the gazetteer.

**Never blocking:** each source polls in its own background thread with a time budget (120 s each for SACHET, news
and Bluesky) and 25 s per-request timeouts. SACHET stops fetching CAP documents near its budget and keeps what it
has. The dashboard only reads SQLite, so a slow source never freezes it. Each source shows one of **ok**, **slow**
(used more than half its budget, returned partial results, or is still running; stored results stay visible),
**failed** (error, exponential backoff) or **skipped** (not configured, blocked by robots.txt, or the previous poll is
still running), with its last success time, duration and the reason.

Politeness: every request checks robots.txt first (unreachable robots.txt = skip this poll; Python's robotparser
applies the first matching rule, which is stricter than RFC 9309), keeps a minimum gap per host, and sends a
descriptive User-Agent. Each source fails and backs off on its own (up to 6 h). Items are prefiltered with flood
keywords (English, Hindi, Malayalam, Bengali, Marathi and romanised forms, including GLOF, inundation, waterlogging and
cloudburst), after removing non-flood uses of the word ("flood the streets with", "a flood of tweets", "flood Delhi
over"), deduplicated by id and by text, and keep their real publish time, source and URL. Phone numbers are redacted
before any text reaches an LLM (briefs).

**Relevance:** the active classifier then runs on every post that passed the filter. Live mode hides posts it labels
*irrelevant* with at least 80 % confidence, from the map, table, warnings, briefs and matching; add *irrelevant* under
**Categories** to see them on the map and table. Less confident *irrelevant* posts stay visible.

**Adding Bluesky**: on bsky.app go to *Settings → Privacy and security → App passwords → Add App Password*, then put
`BLUESKY_HANDLE=yourname.bsky.social` and `BLUESKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx` in `.env` (never your main
password). Restart the dashboard.

**Limits** — be honest about these when showing Live mode:
- **Social data is sparse.** Outside a real flood there may be nothing for hours; Bluesky is small in India, and
  X/Twitter (where 2018's posts were) is a paid API and not used.
- **Live posts are mostly news-style**, so rescue and matching are shown with **Simulate incident**.
- **Headlines are not citizen posts.** Even on Bluesky, many posts are news headlines, which the classifier mostly labels
  *info update*; rescue requests, offers and missing-person posts need citizen posts.
- **Classifier shift.** MuRIL was trained on 2016-2019 disaster tweets (HumAID); 2026 headlines, Bluesky posts and other
  regions differ in style and vocabulary, so expect lower accuracy than the HumAID numbers below.
- **Geography.** The gazetteer maps Kerala, Assam, Odisha, West Bengal, Bihar and Uttar Pradesh (see Geocoding); posts
  from other states are listed but not mapped unless Nominatim finds them (`AQUAALERT_GEOCODE_ONLINE=true`, Kerala-bounded).
  Posts naming only a state ("Odisha", "Assam") get no pin. SACHET areas are often free text ("At isolated places").
- **Warnings.** The bundled sensor files cover August 2018 only, so live warnings use post volume alone (a spike is a
  WATCH); official SACHET alerts are shown separately as "Official alert" entries in the Warnings tab.

**Simulate incident** (Live mode, sidebar): injects 8 clearly synthetic posts — a Hinglish rooftop rescue request,
a Malayalam SOS, two English rescue requests (Kuttanad, Edathua), an offer of boats, a relief-camp supply request, a
missing-person post and a dam-breach rumour — into the live store. They go through the normal pipeline (classifier,
distress rules, credibility, matching, warnings), carry a **SIMULATED** badge in the table, map popups and briefs,
are excluded from the real source counts, and **Clear simulated posts** removes them in one click.

## Early warnings (fuse/)
For each district and day, rescue + damage posts are counted (posts with a misinformation score ≥ 0.5 don't count) and
compared with the previous 3 days. A **WARNING** needs a post spike (≥ 3 posts and ≥ 2 × (baseline + 1)) **and** a
sensor trigger: rainfall ≥ 115.6 mm/day (IMD "very heavy"; ≥ 204.5 mm is "extremely heavy") or a river at/above its
danger level. **WATCH** means only one of the two, or a river within 90 % of danger level. Each flag lists its reasons.

Sensor CSVs go in `data/sensors/` with columns `district, date` and `rainfall_mm` and/or `river_level_m, danger_level_m`
(optional `river`). Several gauges per district-day are combined (highest rainfall; the river gauge nearest its
danger level). **The included files are synthetic** (`scripts/make_sensor_samples.py`): shaped like August 2018 but
made-up numbers. Because the sample posts only cover 15-16 Aug, their baseline is 0 and any 3 posts count as a spike;
with a real multi-week feed the baseline does real work.

Real data (portals checked 2026-11-01; none offers a simple bulk CSV, so expect some manual export/reshaping):
- **Rainfall:** [India-WRIS](https://indiawris.gov.in/wris/) has district-level daily rainfall (actual vs normal) built
  from IMD grids and state gauges; IMD Pune's [gridded daily rainfall](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html)
  (0.25°, 1901-present) can be averaged per district.
- **River levels:** CWC's [Flood Forecasting portal](https://ffs.india-water.gov.in/) shows station water levels with
  warning and danger levels; [cwc.gov.in](https://cwc.gov.in/) publishes daily flood bulletins (PDF) with the same
  tables. India-WRIS also has river water-level data.

## Daily briefs
Without a key the dashboard's **Daily brief** tab writes offline extractive summaries. For LLM summaries add one key to `.env`:
- `GEMINI_API_KEY` (default model `gemini-3.5-flash-lite`, via Gemini's OpenAI-compatible endpoint; `AQUAALERT_LLM_MODEL=gemini-3.8-flash` for the newest Flash, which was timing out when tested), or
- `ANTHROPIC_API_KEY` (default model `claude-haiku-5-5`), or
- `OPENAI_API_KEY` (optional `OPENAI_BASE_URL` for compatible endpoints; default model `gpt-4o-mini`).

`AQUAALERT_LLM_PROVIDER=auto` picks Gemini, then Anthropic, then OpenAI, then the template; `AQUAALERT_LLM_MODEL` overrides the model.
Responses are cached in `data/cache/summaries.json`, citations to posts not in the input are stripped, and phone numbers are
redacted from the text sent to the API. LLM calls only happen when you click **Generate briefs**. Posts only get a source
link if the dataset has a `url` column (the synthetic sample has none).

## Needs ↔ Offers
`rescue_needed`/`need_supplies` posts are matched to nearby `offer_help` posts: `score = alpha * cosine + (1 - alpha) * (1 - km / max_km)`.
Matching inherits classifier mistakes (e.g. a "need rescue" post labelled as an offer will be treated as an offer).

## Distress rules
`classify/rules.py` runs after the classifier and catches clear first-person distress the model missed, in English,
romanised Hindi / Malayalam / Bengali (Hinglish, Manglish, Banglish) and the native scripts:

- **explicit** requests fire on their own: "need rescue", "boats needed", "please send boats", "bachao",
  "boat bhejo", "humein nikalo", "rakshikkane", "thoni venam", "uddhar korun", "nouka pathan", "बचाओ", "രക്ഷിക്ക", "উদ্ধার করুন";
- **ambiguous** phrases ("help us", "SOS", "please send help", "jaldi madad", "pettennu sahayikkuka") and
- **situation** cues with a first-person marker ("we are trapped", "hum chhat pe fase hue hain", "paani ghar mein",
  "njangal kudungi", "vellam kayari", "amra chhade atke achhi", "ഞങ്ങൾ … കുടുങ്ങി", "আমরা … আটকে")

are blocked by appeal and thanks wording (donate, chanda, dhanyavad, nanni, dhonnobad…), past tense ("hum phanse the",
"rakshappettu", "atke chhilam", "bacha liya") and rescuer wording ("hamare paas naav hai", "thoni undu", "amader
nouka ache"). Hashtags, @mentions, URLs and slogans ("desh bachao", `#BJP_भगाओ_देश_बचाओ`) are ignored.

What a cue may change: `offer_help` / `info_update` → `rescue_needed`; `need_supplies` → `rescue_needed` only for an
explicit cue or a first-person situation with no supplies asked for (an appeal like "please send help to Puerto Rico"
stays a request); `irrelevant` → `rescue_needed` only for explicit cues; `missing_person` and `infrastructure_damage` are
never touched. When the model already said `rescue_needed`, the rule is recorded as **confirming** it. The Post keeps
`rule` and the model's `model_label`; the dashboard shows a rule badge ("overrode …" or "· confirms") in map popups and
the table. Turn it off with `AQUAALERT_RULES=off`.

Effect (rules tuned on the sample and HumAID **dev**; HumAID **test** evaluated once):

| | No rules | Previous rules | Current rules |
|---|---:|---:|---:|
| HumAID test macro-F1, MuRIL (English-only) | 0.727 | 0.716 | 0.722 |
| HumAID test macro-F1, MuRIL + romanised | 0.736 | 0.726 | 0.731 |
| … large 5 classes (both models) | 0.759 / 0.763 | unchanged | unchanged |

On the 94 English + multilingual sample posts the current rules change 7 labels with the English-only model and 4 with
the romanised one, every one from wrong to right (e.g. Banglish 0.17 → 0.50 and Manglish 0.43 → 0.71 with the English
model; Hinglish 0.67 → 0.78 with the romanised model), and confirm 14 rescue posts the romanised model already had.
On HumAID the rules still cost ~0.005 macro-F1 against no rules, all in `rescue_needed` (22 gold tweets, defined by a
regex), so the **Results** table reports the models without rules.

## Chatbot
The **Chatbot** tab answers flood-preparedness questions from the PDFs in `data/docs/` plus the latest district briefs,
citing the file and page. It uses the same LLM provider as the briefs (Gemini / Anthropic / OpenAI); with no key it shows
the most relevant passages instead. Passages and the question are treated as untrusted text in the prompt, citations to
passages that weren't retrieved are stripped, and answers are cached in `data/cache/chat.json`. The FAISS index is
cached in `data/cache/chatbot/` and rebuilt automatically when files in `data/docs/` change.

Each document is tagged in `data/docs/sources.json` as **citizen guidance** (household do's and don'ts) or a
**policy document** (written for governments). Personal-safety questions always get up to two citizen passages, and
questions about government/agency action (state, district, NDRF, plans, forecasting...) get policy passages; the prompt
tells the model to answer from the matching kind. Untagged files are guessed from their name and wording.

Download the verified documents (checked 2026-11-01) with `python scripts/fetch_docs.py`, which saves them into
`data/docs/` (git-ignored apart from `sources.json`):

| File | Audience | Source |
|---|---|---|
| `ndma_floods_dos_donts.html` | citizen | NDMA, *Floods: Do's & Don'ts* web page: https://ndma.gov.in/node/143 |
| `manipur_sdma_floods.pdf` | citizen | Manipur SDMA one-page before/during/after sheet: https://msdma.mn.gov.in/uploads/site1/docs/Floods.pdf |
| `ndma_guidelines_management_of_floods.pdf` | policy | NDMA *Guidelines: Management of Floods* (2008), Goa SDMA copy: https://sdma.goa.gov.in/sites/default/files/2022-05/flood.pdf |

KSDMA's flood brochure (https://sdma.kerala.gov.in/wp-content/uploads/2018/11/5.Floods.pdf) is a scanned image with no
text layer, so it can't be indexed without OCR, and its Bengali flood sheet is a JPEG. Supported formats: PDF (text
layer), HTML (main `<article>` content), `.txt`, `.md`. Letter-spaced small-caps PDF headers are stripped during loading;
chunks are ~450 characters so the 128-token multilingual encoder embeds each one in full.

CLI: `python -m aquaalert.chatbot.ask "What should be in an emergency kit?"`

## Languages
`language/detect.py` labels each post `en`, `hi`, `ml`, `bn`, `as` (and other Indic scripts) from Unicode blocks, or
`hi-Latn` / `ml-Latn` / `bn-Latn` for romanised text, using small lexicons of frequent romanised words; `code_mixed` marks
Indian-language posts mixed with English. Place names are matched in Malayalam, Devanagari and Bengali script too,
including Malayalam case endings (ചെങ്ങന്നൂരിൽ, എറണാകുളത്ത് → Chengannur, Ernakulam).

`data/sample_posts_multilingual.csv` adds 44 synthetic labelled posts (native script, romanised and code-mixed Hindi,
Malayalam, Bengali). `python -m aquaalert.language.evaluate` reports, per language (accuracy on the gold labels):

| Language | Posts | Language ID | MuRIL | MuRIL + rules | embed_lr | Keyword |
|---|---:|---:|---:|---:|---:|---:|
| English | 45 | 1.00 | 0.73 | 0.80 | 0.69 | 1.00* |
| Hindi (Devanagari) | 9 | 1.00 | 0.89 | 0.89 | 0.89 | 0.33 |
| Malayalam (script) | 11 | 1.00 | 0.73 | 0.73 | 0.46 | 0.27 |
| Bengali (script) | 7 | 1.00 | 0.86 | 0.86 | 0.14 | 0.00 |
| Hinglish | 9 | 1.00 | 0.44 | 0.56 | 0.11 | 0.89* |
| Manglish | 7 | 1.00 | 0.43 | 0.43 | 0.14 | 0.00 |
| Banglish | 6 | 0.83 | 0.17 | 0.17 | 0.00 | 0.83* |

\* the keyword fallback's rules were written against these same sample posts, so its scores here are not meaningful.

Takeaways (from very small synthetic sets written by us, so directional only): MuRIL transfers reasonably to
native-script Hindi/Bengali/Malayalam despite being fine-tuned on English HumAID, but romanised and code-mixed posts are
weak (0.17-0.56): missing-person posts ("laapata", "kanathayi", "nikhoj") are read as irrelevant and rescue requests as
supply requests. **Recommendation:** before retraining, collect a few hundred real labelled romanised/code-mixed posts
(or transliterate part of HumAID into romanised Hindi/Malayalam with `indic-transliteration`) and fine-tune with them
mixed in; native-script performance does not justify retraining on its own. Language-ID accuracy is on the same
synthetic posts the lexicons were developed against, so treat it as optimistic.

### Romanised training data (LLM-generated)
`scripts/make_romanised_data.py` builds `data/processed/romanised_humaid.csv`: HumAID **train** tweets rewritten by
Gemini (`gemini-3.5-flash-lite`) into romanised, code-mixed Hindi, Malayalam and Bengali social-media posts.
**This data is LLM-generated, not real posts.** It is kept in this private repository only.

How it is built: for each language every `rescue_needed` and `missing_person` train tweet is used, and the other
classes are sampled class-proportionally with a floor of 150 per class (plan: 1,637 per language, 4,911 total).
Each row keeps the AquaAlert label, the source row ID (`humaid-train-<row>`, because the HF mirror has no tweet IDs)
and the source text. Dev and test tweets are never used (the script refuses those files and drops train tweets whose
text also appears in dev/test). Rewrites are rejected if they contain any Indic-script character, copy the source,
have an implausible length, or read as another romanised language; RT prefixes, @mentions and URLs are stripped.
Results are cached per row, so a rerun only fetches what is missing.

Current file (2026-11-01): **4,552 rows** (hi-Latn 1,612, ml-Latn 1,539, bn-Latn 1,401), all passing the checks.

| Class | hi-Latn | ml-Latn | bn-Latn | target per language |
|---|---:|---:|---:|---:|
| rescue_needed | 85 | 84 | **65** | 86 (all available) |
| missing_person | 248 | 227 | **201** | 250 (all available) |
| infrastructure_damage | 150 | 147 | **119** | 150 |
| need_supplies | 148 | 142 | **127** | 150 |
| offer_help | 316 | 305 | 317 | 326 |
| info_update | 438 | 410 | 387 | 442 |
| irrelevant | 227 | 224 | 185 | 233 |

**Bengali is short** (bold): 194 Bengali rewrites (12 %) came back as Hinglish and were removed by the
wrong-language check, and the top-up hit the Gemini quota (HTTP 429). To fill the gaps once the quota resets, rerun
`python scripts/make_romanised_data.py`: it only requests the 359 missing rows (~36 requests).

Quality, from a manual spot check of 30 random rows (10 per language):
- **Label drift:** none seen; every rewrite still fits its label.
- **Meaning drift: ~10 % (3/30).** One rewrite turned an argumentative reply into a plea for help, one added
  urgency ("urgent zaroorat") the source didn't have, one added a comment. These can't be detected automatically, so
  expect roughly one row in ten to be looser than its source.
- **Language:** the wrong-language drift above was found here and is now filtered; light Hindi words inside Bengali
  rows ("ke bachao") remain.
- **Context:** the sources are global disasters (US hurricanes, Nepal earthquakes...), so many posts are e.g.
  "romanised Hindi about Hurricane Harvey" rather than about Kerala; spelling is casual and place names can be misspelt.

Use it only as training data: `python -m aquaalert.classify.train ... --extra-train data/processed/romanised_humaid.csv`
(dev and test stay HumAID), or `USE_ROMANISED = True` in `notebooks/train_colab.ipynb` (Google Colab) or
`notebooks/train_kaggle.ipynb` (Kaggle Notebooks, T4; setup steps in its first cell), which also evaluate the 44
hand-written multilingual posts and upload the model to `<user>/aquaalert-muril-romanised`.

## Credibility (heuristic, not a trained model)
`credibility/heuristics.py` gives every post a **misinformation score from 0 (no red flags) to 1** from transparent
rules, and lists the reasons on the dashboard (table column and map popup; flagged markers are hollow and dashed).
It is a set of hand-written heuristics, **not a trained or validated model**: treat a high score as "check this before
acting", never as proof, and a zero score as "no known red flags", not as verified.

| Signal | Weight | Example |
|---|---:|---|
| Older / recycled event | 0.30-0.35 | "Kedarnath flood 2013 video", a year before the post's year |
| Dam-breach claim without a source | 0.40 | "Mullaperiyar dam has broken" (also Malayalam/Hindi/Bengali forms) |
| Death toll without a source | 0.30 | "500 people dead in Chengannur" |
| Urgency-forwarding | 0.35 | "forward to all", "share before it is deleted", "sabko forward karo", "ഫോർവേഡ്" |
| Contradiction | 0.30 | "everyone is safe" while ≥2 rescue requests are within 25 km / 6 h or in today's district brief; a dam-breach claim within 24 h of an attributed denial about the same dam |
| Copy-pasted text | 0.15-0.25 | the same appeal posted by several accounts (often outdated) |
| "Truth is being hidden" / "!!!" | 0.20 / 0.10 | |

Flags combine as independent evidence (`1 - Π(1 - w)`); citing an official source or link halves the score, and
corrections ("this photo is from 2015, don't forward") are recognised as debunks. Daily briefs leave out posts scoring
≥ 0.5 (and say how many), and the sidebar can hide them everywhere. `data/sample_posts_rumours.csv` holds 16 synthetic
rumour / debunk / control posts; `python -m aquaalert.credibility.evaluate` shows all 10 rumours scoring ≥ 0.3 and no
other sample post scoring above 0 — but the set was written together with the rules, so this only shows that each rule
does what it says.

## Geocoding
Offline gazetteer first, then (if `AQUAALERT_GEOCODE_ONLINE=true`) Nominatim bounded to Kerala. Results are
cached in `data/cache/geocode.json`.

The gazetteer maps **398 places**:
- **364 in Assam, Odisha, West Bengal, Bihar and Uttar Pradesh**: districts, district headquarters, towns of at least
  50,000 people, Bagaha and 23 rivers, in [`aquaalert/extract/gazetteer_india.csv`](aquaalert/extract/gazetteer_india.csv).
  **Source: [Wikidata](https://www.wikidata.org) (CC0)**, read through the public
  [QLever SPARQL endpoint](https://qlever.cs.uni-freiburg.de/wikidata) by `scripts/build_gazetteer.py`, because
  robots.txt disallows automated access to GeoNames downloads and to Wikidata's own query service for generic agents.
  Every row keeps its Wikidata id, and the English, Devanagari and Bengali names come from the same entity.
- **34 in Kerala**, in [`aquaalert/extract/gazetteer_kerala.csv`](aquaalert/extract/gazetteer_kerala.csv), from the same
  source with the entity chosen by id in `scripts/build_gazetteer.py`. These replaced coordinates typed by hand in Phase 1;
  the largest move is Vypin at 11.7 km. Bare "Paravur" means **North Paravur** (Ernakulam), the 2018 flood town, not the
  Paravur in Kollam. Four entries are stand-ins for the place: Cheruthoni uses Cheruthoni Dam, Ranni uses "Ranni village",
  Pandalam uses Pandalam Municipality (the town entity is 12 km from the 15 other Pandalam entities), and Kuttanad is a
  region, so its point is only approximate. OpenStreetMap was not used: robots.txt disallows Nominatim and Overpass for
  generic agents.

**No true district centroids:** no district polygons are bundled, and Wikidata's point for a district is usually its
headquarters. So distance to that point is not an error check: 14 correct Kerala places are more than 15 km from it (for
example Shoranur, 40 km, and Mullaperiyar, 42 km). Wikidata's "located in" chain confirms the district for most of them.

Nothing is estimated. Rejected rather than guessed: coordinates outside the place's state (Wikidata puts one Banka entity
in Slovakia), and names shared by different places (Lakhimpur, Akbarpur, Midnapore; use "Lakhimpur Kheri" or
"North Lakhimpur"). Same-named places outside these states cannot be told apart, so a "Hamirpur" in Himachal Pradesh would
pin in Uttar Pradesh; Aurangabad is left out for that reason.

River pins are coarse: one Wikidata point on the river, usually its mouth, used only when a post names no town. Ganga,
Teesta and Torsa are recognised in text but get no pin, because their nearest Wikidata point is too far from a covered
town or lies in Bangladesh. Rapti could not be sourced and is not included. Names that are also everyday words (Gaya, Mau,
Basti, Puri, ...) match only when capitalised, and their ambiguous native spellings are not used.

## Romanised model: before / after
`<user>/aquaalert-muril-romanised` is MuRIL fine-tuned on HumAID train **plus the 4,552 LLM-generated romanised posts**
(Kaggle T4, same recipe otherwise). Fetch it with
`python scripts/fetch_model.py --repo <user>/aquaalert-muril-romanised --out models/muril_romanised`; with
`AQUAALERT_CLASSIFIER=auto` it is preferred over `models/muril`, then `embed_lr`, then the keyword fallback.
Numbers below were recomputed locally with `python scripts/compare_models.py` (distress rules off).

| HumAID test (15,160 tweets) | MuRIL (English-only) | MuRIL + romanised data |
|---|---:|---:|
| Macro-F1, all 7 (95% CI) | 0.727 (0.700–0.752) | **0.736 (0.706–0.764)** |
| Macro-F1, 5 large classes | 0.759 (0.750–0.768) | **0.763 (0.754–0.773)** |
| Accuracy | 0.800 | 0.807 |
| F1 `rescue_needed` | 0.571 | 0.558 |
| F1 `missing_person` | 0.725 | 0.776 |
| F1 `need_supplies` | 0.575 | 0.567 |

| Hand-written posts (accuracy) | Posts | MuRIL (English-only) | MuRIL + romanised data |
|---|---:|---:|---:|
| English | 45 | 0.73 | **0.76** |
| Hindi (Devanagari) | 9 | 0.89 | **1.00** |
| Bengali (script) | 7 | 0.86 | **1.00** |
| Malayalam (script) | 11 | 0.73 | **0.73** |
| Hinglish | 9 | 0.44 | **0.67** |
| Manglish | 7 | 0.43 | **0.86** |
| Banglish | 6 | 0.17 | **0.50** |

All numbers match what the Kaggle run printed and saved (`metrics.json`, `language_eval.json` in the model repo).

Caveats:
- **The per-language sets are tiny** (6–11 posts per language), so one post changes accuracy by 9–17 points; treat the
  gains as directional, not measured.
- **Training and test posts for the romanised languages are both LLM-written or hand-written by us**: the training
  data is Gemini rewrites of HumAID tweets and the 44 test posts are synthetic, so they may share style that real
  WhatsApp/Twitter text does not. The real test of this model is real romanised posts.
- **Bengali training data is short of target** (1,401 rows; `rescue_needed` 65 and `missing_person` 201 of the
  intended 86 and 250), and Banglish remains the weakest language.
- HumAID test is English-only, so its change mainly shows the romanised data did not hurt English performance.
- **Not everything improved:** `rescue_needed` F1 dips slightly (0.571 → 0.558, 22 test tweets) and so does
  `need_supplies` (0.575 → 0.567); Malayalam script is unchanged; on the English sample the old model *with* the
  distress rules (0.80) still beats the new one (0.76), where the rules no longer add anything.

## Results

All three classifiers on the same HumAID test split (15,160 tweets, labels mapped to AquaAlert's 7 classes by
`classify/label_map.py`). Cells are per-class F1; bold marks the best model per row. Bootstrap 95% CIs use 1000
test-set resamples. "5 large classes" excludes `rescue_needed` and `missing_person`.

| Class | Test examples | Keyword fallback | embed_lr | MuRIL |
|---|---:|---:|---:|---:|
| `rescue_needed` | 22 | 0.028 | 0.280 | **0.571** |
| `infrastructure_damage` | 1617 | 0.416 | 0.741 | **0.803** |
| `need_supplies` | 499 | 0.181 | 0.433 | **0.575** |
| `offer_help` | 4219 | 0.374 | 0.810 | **0.862** |
| `missing_person` | 72 | 0.471 | 0.521 | **0.725** |
| `info_update` | 5714 | 0.342 | 0.749 | **0.803** |
| `irrelevant` | 3017 | 0.395 | 0.701 | **0.752** |
| **Macro-F1, all 7** (95% CI) | | 0.315 (0.302–0.327) | 0.605 (0.582–0.625) | 0.727 (0.700–0.752) |
| **Macro-F1, 5 large classes** (95% CI) | | 0.342 (0.332–0.352) | 0.687 (0.677–0.696) | 0.759 (0.750–0.768) |
| Accuracy | | 0.354 | 0.734 | 0.800 |

- **Keyword fallback:** zero-shot rules, no training. Its keywords were tuned on the 50 synthetic sample posts, not on HumAID.
- **embed_lr:** frozen `paraphrase-multilingual-MiniLM-L12-v2` embeddings + class-weighted logistic regression (C=300 chosen on dev). Trains on a Mac in minutes.
- **MuRIL:** `google/muril-base-cased` fine-tuned on Colab (4 epochs, inverse-frequency class-weighted loss, checkpoint chosen by dev macro-F1). Stored in a private Hugging Face repo; pull it with `python scripts/fetch_model.py`.
- `rescue_needed` (22 test examples) and `missing_person` (72) are small, so their F1 and the all-7 macro-F1 have wide intervals. Gold `rescue_needed` labels come from a regex rule over HumAID's `requests_or_urgent_needs` class.
- `AQUAALERT_CLASSIFIER=auto` uses MuRIL if `models/muril/` exists, then `embed_lr`, then the keyword fallback.

## Data
See [scripts/download_data.md](scripts/download_data.md). `data/sample_posts.csv` is synthetic.

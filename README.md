# AquaAlert — Multilingual AI for Flood Crisis Response

Turns flood-time posts (English, Hindi/Hinglish, Bengali, Malayalam, Assamese) into classified
help requests on a map. See [CLAUDE.md](CLAUDE.md) for the full architecture and roadmap.

## Status: Phases 1-3
`ingest -> language -> classify (+ distress rules) -> extract(geo) -> credibility -> summarize / match / chatbot -> Streamlit`,
runnable with no downloads, training or API keys. Phase 4 (sensor fusion, FastAPI) is next.

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # torch/transformers only needed for training
cp .env.example .env
streamlit run dashboard/app.py       # runs on the synthetic samples in data/ (English, multilingual, rumours)
pytest
```

## Modules
| Path | What it does |
|---|---|
| `aquaalert/schemas.py` | Pydantic `Post` shared by all stages |
| `aquaalert/ingest/` | `loader.py` (CSV/JSON/JSONL), `replay.py` (time-ordered feed simulator) |
| `aquaalert/classify/` | `zeroshot.py` keyword fallback, `model.py` backend selector, `train.py` MuRIL fine-tuning, `label_map.py` |
| `aquaalert/extract/` | locations / phones / people counts, offline Kerala gazetteer, cached geocoder |
| `aquaalert/summarize/` | district-wise daily brief (`brief.py`), providers: offline template, Anthropic, OpenAI-compatible (`providers.py`); cited sources, disk cache |
| `aquaalert/match/` | needs-vs-offers matching: sentence-embedding cosine + haversine proximity (`matcher.py`, `geo.py`) |
| `aquaalert/classify/rules.py` | post-classification distress rules (promote clear first-person distress to `rescue_needed`) |
| `aquaalert/chatbot/` | RAG over guideline PDFs in `data/docs/` + latest daily briefs: FAISS, multilingual MiniLM embeddings, cited answers |
| `aquaalert/language/` | language ID: native scripts + romanised Hindi/Malayalam/Bengali, code-mixing flag; per-language evaluation |
| `aquaalert/credibility/` | heuristic misinformation score (0-1) with reasons |
| `aquaalert/pipeline.py` | ingest -> classify -> extract |
| `dashboard/` | Streamlit + Folium app |

## CLI
```bash
python -m aquaalert.ingest.replay data/sample_posts.csv --speed 600
python -m aquaalert.pipeline data/sample_posts.csv
python -m aquaalert.classify.train --data data/raw/humaid.csv --label-dataset humaid --out models/muril
```
Training writes the model and `metrics.json` (accuracy, macro-F1, per-class report) to `--out`.
With `AQUAALERT_CLASSIFIER=auto` the dashboard uses the trained model if `AQUAALERT_MODEL_DIR` exists,
otherwise the zero-shot keyword fallback.

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
`classify/rules.py` runs after the classifier and promotes `offer_help` / `info_update` to `rescue_needed` when a post
has an explicit request ("need rescue", "boats needed", "bachao") or a situation cue (trapped, stuck, rooftop, water
entering, "chhat par") together with first-person wording. Offers, fundraising/volunteer appeals, thanks and past-tense
accounts are left alone. When a rule fires the Post records `rule` and the model's own `model_label`; the dashboard
shows both. Turn it off with `AQUAALERT_RULES=off`.

Effect: on the 50 sample posts 3 labels change (s020, s021, s045; all to the correct `rescue_needed`; accuracy vs the
sample's labels 0.70 → 0.76 with MuRIL). On the HumAID test split it fires on 20 of 15,160 tweets (MuRIL):
macro-F1 (all 7) 0.727 → 0.716, macro-F1 (5 large classes) 0.759 → 0.759. The drop is all in `rescue_needed`
(22 gold tweets, defined by a regex over one HumAID class), so HumAID can't really measure this rule; the numbers in
**Results** above are without it.

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

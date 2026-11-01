# AquaAlert — Multilingual AI for Flood Crisis Response

Turns flood-time posts (English, Hindi/Hinglish, Bengali, Malayalam, Assamese) into classified
help requests on a map. See [CLAUDE.md](CLAUDE.md) for the full architecture and roadmap.

## Status: Phase 1 + Phase 3 quick wins
`ingest -> classify -> extract(geo) -> summarize / match -> Streamlit`, runnable with no downloads, training or API keys
(Phase 2, language and credibility, is not built yet).

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # torch/transformers only needed for training
cp .env.example .env
streamlit run dashboard/app.py       # runs on data/sample_posts.csv (50 fake posts)
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
- `ANTHROPIC_API_KEY` (default model `claude-haiku-5-5`), or
- `OPENAI_API_KEY` (optional `OPENAI_BASE_URL` for compatible endpoints; default model `gpt-4o-mini`).

`AQUAALERT_LLM_PROVIDER=auto` picks Anthropic, then OpenAI, then the template; `AQUAALERT_LLM_MODEL` overrides the model.
Responses are cached in `data/cache/summaries.json`, citations to posts not in the input are stripped, and phone numbers are
redacted from the text sent to the API. LLM calls only happen when you click **Generate briefs**. Posts only get a source
link if the dataset has a `url` column (the synthetic sample has none).

## Needs ↔ Offers
`rescue_needed`/`need_supplies` posts are matched to nearby `offer_help` posts: `score = alpha * cosine + (1 - alpha) * (1 - km / max_km)`.
Matching inherits classifier mistakes (e.g. a "need rescue" post labelled as an offer will be treated as an offer).

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

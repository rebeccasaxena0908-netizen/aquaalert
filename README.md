# AquaAlert — Multilingual AI for Flood Crisis Response

Turns flood-time posts (English, Hindi/Hinglish, Bengali, Malayalam, Assamese) into classified
help requests on a map. See [CLAUDE.md](CLAUDE.md) for the full architecture and roadmap.

## Status: Phase 1
`ingest -> classify -> extract(geo) -> Streamlit map`, runnable with no downloads or training.

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
| `aquaalert/pipeline.py` | ingest -> classify -> extract |
| `dashboard/` | Streamlit + Folium app |

## CLI
```bash
python -m aquaalert.ingest.replay data/sample_posts.csv --speed 600
python -m aquaalert.pipeline data/sample_posts.csv
python -m aquaalert.classify.train --data data/raw/humaid.csv --label-dataset humaid --out models/muril-crisis
```
Training writes the model and `metrics.json` (accuracy, macro-F1, per-class report) to `--out`.
With `AQUAALERT_CLASSIFIER=auto` the dashboard uses the trained model if `AQUAALERT_MODEL_DIR` exists,
otherwise the zero-shot keyword fallback.

## Geocoding
Offline gazetteer first, then (if `AQUAALERT_GEOCODE_ONLINE=true`) Nominatim bounded to Kerala. Results are
cached in `data/cache/geocode.json`.

## Data
See [scripts/download_data.md](scripts/download_data.md). `data/sample_posts.csv` is synthetic.

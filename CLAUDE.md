# AquaAlert — Multilingual AI for Flood Crisis Response

## What this project is
AquaAlert is an NLP platform that turns flood-time text (social media posts, news) into
actionable intelligence: classified help requests, rescue locations on a map, rumor flags,
needs-vs-offers matches, daily summaries, early-warning signals, and a preparedness chatbot.
Focus region: India (English, Hindi, Bengali, Malayalam, Assamese, and code-mixed Hinglish).
Primary test event: Kerala floods 2018 (secondary: Assam floods 2022).

## Architecture (one pipeline, modular stages)
```
ingest -> language -> classify -> credibility -> extract(geo) -> match -> summarize -> fuse -> serve
                                                                                      \-> chatbot (RAG)
```
Every stage reads and writes a shared `Post` schema (see `aquaalert/schemas.py`) so stages
can be developed and tested independently.

### Modules
1. `ingest/` — load posts from datasets (CSV/JSON) and a **replay simulator** that streams a
   historical dataset in time order to mimic a live feed. News via RSS (feedparser).
   Live X/Twitter API is optional and off by default (paid API).
2. `language/` — language ID (fastText lid.176 or `langdetect`), transliteration/normalization
   for code-mixed text (`indic-transliteration`).
3. `classify/` — multi-class crisis classifier. Labels: `rescue_needed`, `infrastructure_damage`,
   `need_supplies`, `offer_help`, `missing_person`, `info_update`, `irrelevant`.
   Backbone: `google/muril-base-cased` (fallback `xlm-roberta-base`). Fine-tune with HF Trainer.
4. `credibility/` — misinformation score 0–1. Start with a fine-tuned classifier plus heuristics
   (account signals if available, "old media reshared", contradiction with verified sources).
5. `extract/` — NER for locations, landmarks, phone numbers, people counts (spaCy + regex,
   optionally fine-tuned token classifier). Geocode with geopy/Nominatim, cached locally,
   biased to the target state's bounding box.
6. `match/` — needs-vs-offers matching: `sentence-transformers`
   (`paraphrase-multilingual-MiniLM-L12-v2`) cosine similarity + haversine distance score.
7. `summarize/` — district-wise daily brief from verified posts + news. Use an LLM API
   (configurable provider via env var); include source links in output.
8. `fuse/` — combine text signal volume per district with rainfall/river-level CSVs
   (IMD/CWC data downloaded manually into `data/sensors/`) to raise early-warning flags.
9. `chatbot/` — RAG over NDMA flood guidelines (PDFs in `data/docs/`) + latest summaries.
   LangChain or LlamaIndex, FAISS vector store, multilingual embeddings.
10. `api/` — FastAPI backend exposing each stage's outputs.
11. `dashboard/` — Streamlit + Folium: live map of geolocated requests, colored by category,
    rumor flags, matches table, daily summary panel, chatbot tab.

## Build phases (always keep a working demo)
- **Phase 1:** schemas, ingest (dataset loader + replay), classifier, NER + geocoding,
  Streamlit map. <- START HERE
- **Phase 2:** multilingual layer, credibility scoring.
- **Phase 3:** summarizer, needs-offers matching.
- **Phase 4:** chatbot (RAG), sensor fusion, FastAPI backend.

## Datasets
- HumAID (CrisisNLP) — labeled disaster tweets incl. Kerala floods 2018.
- CrisisLex / CrisisMMD — additional crisis-labeled tweets.
- TREC-IS / CrisisFACTS — information-type labels and summarization references.
- Kaggle Kerala floods tweet datasets — extra unlabeled text for replay.
- NDMA flood guidelines (PDF) — chatbot knowledge base.
- IMD rainfall / CWC river level data — sensor fusion.
Raw data goes in `data/raw/` (git-ignored). Write a `scripts/download_data.md` noting sources
and any that require manual download or registration.
Map dataset-specific labels onto AquaAlert's label set in `classify/label_map.py`.

## Project layout
```
aquaalert/
  schemas.py  config.py
  ingest/ language/ classify/ credibility/ extract/ match/ summarize/ fuse/ chatbot/
api/  dashboard/  scripts/  notebooks/  tests/
data/{raw,processed,sensors,docs,cache}/   models/
```

## Tech stack
Python 3.11, transformers, datasets, torch, scikit-learn, spaCy, sentence-transformers,
geopy, faiss-cpu, langchain or llama-index, fastapi, uvicorn, streamlit, folium,
streamlit-folium, pandas, pydantic, python-dotenv, pytest.

## Workflow
- **Training and heavy experiments run on Colab (T4)** through the VS Code Colab extension, or on **Kaggle Notebooks
  (T4)** when the Colab GPU quota is used up (`notebooks/train_kaggle.ipynb`, tokens from Kaggle Secrets), using
  notebooks in `notebooks/`. Never train large models on this Mac (small baselines such as `classify/embed_lr.py` are fine).
- **Trained models live in private Hugging Face Hub repos** (e.g. `<user>/aquaalert-muril`). Notebooks must upload
  the model at the end, because the Colab disk is wiped on disconnect.
- **The Mac runs the code, tests, dashboard and demo.** Pull models with `scripts/fetch_model.py`, using
  `HF_TOKEN` from `.env`.
- **New training notebooks follow `notebooks/train_colab.ipynb` / `train_kaggle.ipynb`:** GPU check first, tokens from
  the platform's secrets store or hidden `getpass` prompts (never printed or saved), and the HF upload as the final step.

## Conventions
- Config and secrets via `.env` (provide `.env.example`); never hardcode keys.
- Pydantic models for all data passed between stages.
- Each module has a CLI entry (`python -m aquaalert.classify.train ...`) and unit tests.
- Training scripts log metrics (accuracy, macro-F1, per-class report) to `models/<name>/metrics.json`.
- Keep the dashboard runnable with pretrained/fallback models so the demo works before training.
- Cache geocoding and LLM calls to `data/cache/`.
- Prefer small, readable functions; type hints everywhere; README updated each phase.

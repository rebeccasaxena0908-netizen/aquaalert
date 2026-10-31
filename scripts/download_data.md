# Datasets (put raw files in `data/raw/`, which is git-ignored)

| Dataset | Use | Access |
|---|---|---|
| HumAID (CrisisNLP) — incl. Kerala floods 2018 | labeled tweets for fine-tuning | see below |
| CrisisLex / CrisisMMD | extra labeled tweets | https://crisisnlp.qcri.org/ — direct download |
| TREC-IS / CrisisFACTS | information-type labels, summarization refs | http://dcs.gla.ac.uk/~richardm/TREC_IS/ — registration/request may be required |
| Kaggle Kerala floods tweets | unlabeled replay text | Kaggle account + API token required (manual) |
| NDMA flood guidelines (PDF) | chatbot KB (Phase 4) | https://ndma.gov.in — manual download to `data/docs/` |
| IMD rainfall / CWC river levels | sensor fusion (Phase 4) | Manual download to `data/sensors/` |

## HumAID
- Official page: https://crisisnlp.qcri.org/humaid_dataset
- Download the **event-wise set1** archive: it includes the tweet text (other variants ship tweet IDs only).
  Files are TSV; the loader reads `.tsv` and detects `tweet_text` / `class_label`. Use `--label-dataset humaid`.
- HuggingFace mirror: `QCRI/HumAID-all` (all events; columns `tweet_text`, `class_label`; train/validation/test
  of ~53.5k / 7.8k / 15.2k rows). `python scripts/fetch_humaid.py` saves it to `data/raw/humaid/{train,dev,test}.csv`.
- 11 categories, mapped in `aquaalert/classify/label_map.py`. HumAID has no rescue class: `requests_or_urgent_needs`
  posts using rescue language (stranded, trapped, need rescue, ...) are routed to `rescue_needed`, the rest to
  `need_supplies`. `dont_know_cant_judge` is dropped.
- HumAID has no timestamps in these files, so it is used for training/eval (`load_labeled`), not the replay feed.

Loader needs a text column and a timestamp column (auto-detected); a label column is optional.
Only the HumAID details above were verified; the other URLs were written from memory and are unchecked.

# Datasets (put raw files in `data/raw/`, which is git-ignored)

| Dataset | Use | Access |
|---|---|---|
| HumAID (CrisisNLP) — incl. Kerala floods 2018 | labeled tweets for fine-tuning | https://crisisnlp.qcri.org/humaid_dataset — direct download; use `--label-dataset humaid` |
| CrisisLex / CrisisMMD | extra labeled tweets | https://crisisnlp.qcri.org/ — direct download |
| TREC-IS / CrisisFACTS | information-type labels, summarization refs | http://dcs.gla.ac.uk/~richardm/TREC_IS/ — registration/request may be required |
| Kaggle Kerala floods tweets | unlabeled replay text | Kaggle account + API token required (manual) |
| NDMA flood guidelines (PDF) | chatbot KB (Phase 4) | https://ndma.gov.in — manual download to `data/docs/` |
| IMD rainfall / CWC river levels | sensor fusion (Phase 4) | Manual download to `data/sensors/` |

Loader needs a text column and a timestamp column (auto-detected); a label column is optional.
Verify the URLs above — they were not checked from this environment.

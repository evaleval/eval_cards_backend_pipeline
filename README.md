# eval-cards-backend-pipeline

Python pipeline for materializing static evaluation artifacts from `evaleval/EEE_datastore` and publishing them to the Hugging Face dataset `evaleval/card_backend`.

## What it does

- Loads live parquet data from the 17 active EEE configs through DuckDB.
- Loads curated benchmark cards from `metadata/benchmark_card_*.json`.
- Normalizes model identities into stable family and route IDs.
- Precomputes:
  - `model-cards.json`
  - `eval-list.json`
  - `peer-ranks.json`
  - `benchmark-metadata.json`
  - `developers.json`
  - `models/*.json`
  - `evals/*.json`
  - `developers/*.json`
  - `manifest.json`
- Uploads the full `output/` directory to `evijit/ev_card_be`.

## Install

```bash
python -m pip install --upgrade pip huggingface_hub
```

## Run

Dry run:

```bash
python scripts/pipeline.py --dry-run
```

Upload to Hugging Face:

```bash
HF_TOKEN=hf_xxx python scripts/pipeline.py
```

## Environment variables

- `HF_TOKEN`: required for non-dry-run uploads.
- `CONFIG_BATCH_SIZE`: optional. Controls how many EEE configs are loaded concurrently. Default: `4`.
- `EEE_LOCAL_DATASET_DIR`: optional local snapshot directory (used in CI to avoid HF rate limits).
- `CONFIGS` / `CONFIG_NAMES`: optional comma-separated config override.
- `CONFIG_LIMIT`: optional limit for quick smoke tests.

Lower values reduce peak disk and memory pressure on GitHub Actions runners as the source dataset grows.

## Notes

- The pipeline cleans and recreates `output/` on each run.
- Config load failures are logged and skipped; the skipped config list is recorded in `output/manifest.json`.
- The workflow uses `npm ci --omit=optional` and a bounded `CONFIG_BATCH_SIZE` to reduce runner space usage.

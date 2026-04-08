# eval-cards-backend-pipeline

Node.js pipeline for materializing static evaluation artifacts from `evaleval/EEE_datastore` and publishing them to the Hugging Face dataset `evijit/ev_card_be`.

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
npm ci
```

## Run

Dry run:

```bash
node scripts/pipeline.mjs --dry-run
```

Upload to Hugging Face:

```bash
HF_TOKEN=hf_xxx node scripts/pipeline.mjs
```

## Environment variables

- `HF_TOKEN`: required for non-dry-run uploads.
- `CONFIG_BATCH_SIZE`: optional. Controls how many EEE configs are loaded concurrently. Default: `4`.

Lower values reduce peak disk and memory pressure on GitHub Actions runners as the source dataset grows.

## Notes

- The pipeline cleans and recreates `output/` on each run.
- Config load failures are logged and skipped; the skipped config list is recorded in `output/manifest.json`.
- The workflow uses `npm ci --omit=optional` and a bounded `CONFIG_BATCH_SIZE` to reduce runner space usage.

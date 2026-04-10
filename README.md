# eval-cards-backend-pipeline

Python pipeline for materializing static evaluation artifacts from `evaleval/EEE_datastore` and publishing them to the Hugging Face dataset `evaleval/card_backend`.

## What it does

- Loads live parquet data from the 17 active EEE configs through DuckDB.
- Caches benchmark cards from `evaleval/auto-benchmarkcards` under `.cache/auto_benchmarkcards/cards/`.
- Normalizes model identities into stable family and route IDs.
- Groups composite benchmarks into single-benchmark eval summaries and nests metrics under each eval.
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

Generate the metric-looking string registry from the local EEE snapshot:

```bash
python scripts/build_metric_looking_registry.py
```

Upload to Hugging Face:

```bash
HF_TOKEN=hf_xxx python scripts/pipeline.py
```

## Environment variables

- `HF_TOKEN`: required for non-dry-run uploads.
- `CONFIG_BATCH_SIZE`: optional. Controls how many EEE configs are loaded concurrently. Default: `4`.
- `EEE_LOCAL_DATASET_DIR`: optional local snapshot directory (used in CI to avoid HF rate limits).
- `BENCHMARK_METADATA_LOCAL_DIR`: optional local cache directory for `evaleval/auto-benchmarkcards`.
- `CONFIGS` / `CONFIG_NAMES`: optional comma-separated config override.
- `CONFIG_LIMIT`: optional limit for quick smoke tests.

Lower values reduce peak disk and memory pressure on GitHub Actions runners as the source dataset grows.

## Notes

- The pipeline cleans and recreates `output/` on each run.
- Benchmark metadata is sourced only from the Hugging Face dataset `evaleval/auto-benchmarkcards`.
- `registry/metric_looking_strings.json` is generated from the local EEE snapshot, can be refreshed with `scripts/build_metric_looking_registry.py`, and is used by the pipeline to canonicalize metric aliases.
- Config load failures are logged and skipped; the skipped config list is recorded in `output/manifest.json`.
- The workflow uses `npm ci --omit=optional` and a bounded `CONFIG_BATCH_SIZE` to reduce runner space usage.

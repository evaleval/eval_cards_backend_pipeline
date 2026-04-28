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
- Materializes `output/eval_cards.duckdb` as a query-ready DuckDB artifact with
  canonical metric identity, pipeline hierarchy keys, metric-scoped instance
  joins, and join-quality diagnostics.
- Provides an optional FastAPI query backend over the same DuckDB schema, plus
  manual aggregate/instance JSONL ingest endpoints for local repair workflows.

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run

Dry run:

```bash
python -m scripts.pipeline --dry-run
```

The pipeline also emits `output/eval_cards.duckdb`, and records its table
counts plus join-integrity summary in `output/manifest.json`.

Run the query API:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Generate the metric-looking string registry from the local EEE snapshot:

```bash
python -m scripts.build_metric_looking_registry
```

Upload to Hugging Face:

```bash
HF_TOKEN=hf_xxx python -m scripts.pipeline
```

## Environment variables

- `HF_TOKEN`: required for non-dry-run uploads.
- `CONFIG_BATCH_SIZE`: optional. Controls how many EEE configs are loaded concurrently. Default: `4`.
- `EEE_LOCAL_DATASET_DIR`: optional local snapshot directory (used in CI to avoid HF rate limits).
- `BENCHMARK_METADATA_LOCAL_DIR`: optional local cache directory for `evaleval/auto-benchmarkcards`.
- `CONFIGS` / `CONFIG_NAMES`: optional comma-separated config override.
- `CONFIG_LIMIT`: optional limit for quick smoke tests.
- `EVAL_CARDS_DUCKDB_PATH`: optional path for the pipeline-generated DuckDB artifact. Default: `output/eval_cards.duckdb`.
- `DUCKDB_PATH`: optional path read by the query API. Default: `output/eval_cards.duckdb`.
- `DEFAULT_AGGREGATE_JSONL_PATH`: optional default aggregate ingest JSONL path.
- `DEFAULT_INSTANCE_JSONL_PATH`: optional default instance ingest JSONL path.
- `INGESTION_BASE_DIR`: optional safe base directory for API file ingestion. Default: `data`.

Lower values reduce peak disk and memory pressure on GitHub Actions runners as the source dataset grows.

## Query API

- `GET /health`
- `POST /ingest/aggregate?jsonl_path=...`
- `POST /ingest/instance?jsonl_path=...`
- `GET /stats`
- `GET /sources?limit=...`
- `GET /models?source_name=...&benchmark_name=...&limit=...&offset=...`
- `GET /benchmarks?source_name=...&benchmark_name=...&metric_kind=...&limit=...`
- `GET /benchmarks/{benchmark_name}/models?source_name=...&metric_identity=...&limit=...`
- `GET /metrics/top-models?metric_kind=...&metric_name=...&source_name=...&limit=...`
- `GET /join-integrity`
- `GET /quality/orphan-runs?limit=...`
- `GET /quality/identifier-issues?limit=...`

## Notes

- The pipeline cleans and recreates `output/` on each run.
- Raw names remain display/audit metadata in DuckDB. Query grouping should use
  backend-owned identifiers such as `model_route_id`, `eval_summary_id`,
  `metric_summary_id`, and the canonical metric tuple.
- Benchmark metadata is sourced only from the Hugging Face dataset `evaleval/auto-benchmarkcards`.
- `registry/metric_looking_strings.json` is generated from the local EEE snapshot, can be refreshed with `python -m scripts.build_metric_looking_registry`, and is used by the pipeline to canonicalize metric aliases.
- Config load failures are logged and skipped; the skipped config list is recorded in `output/manifest.json`.
- The workflow uses `npm ci --omit=optional` and a bounded `CONFIG_BATCH_SIZE` to reduce runner space usage.

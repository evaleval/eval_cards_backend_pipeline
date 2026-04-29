# eval-cards-backend-pipeline

Python pipeline for materializing static evaluation artifacts from `evaleval/EEE_datastore` and publishing them to the Hugging Face dataset `evaleval/card_backend` (or a staging target during the parity migration — see "Upload safety guard" below).

## What it does

- Reads upstream JSON evaluation records from `evaleval/EEE_datastore` (one record per `data/<benchmark>/<dev>/<model>/<uuid>.json`).
- Caches benchmark cards from `evaleval/auto-benchmarkcards` under `.cache/auto_benchmarkcards/cards/`.
- Normalizes model identities into stable family and route IDs.
- Groups composite benchmarks into single-benchmark eval summaries and nests metrics under each eval.
- Precomputes JSON artifacts:
  - `model-cards.json`, `model-cards-lite.json`
  - `eval-list.json`, `eval-list-lite.json`
  - `peer-ranks.json`, `comparison-index.json`
  - `benchmark-metadata.json`, `eval-hierarchy.json`, `corpus-aggregates.json`
  - `developers.json`
  - `models/*.json`, `evals/*.json`, `developers/*.json`, `instances/*.jsonl`, `records/*.json`
  - `manifest.json`
- Emits the parity-layer parquet artifacts under `output/duckdb/v1/` (frontend-ready DuckDB read surface):
  - `model_cards.parquet`, `model_cards_lite.parquet`
  - `eval_list.parquet`, `eval_list_lite.parquet`
  - `eval_summaries.parquet`, `aggregate_eval_summaries.parquet`, `matrix_eval_summaries.parquet`
  - `model_summaries.parquet`
  - `developers.parquet`, `developer_summaries.parquet`
- Uploads the full `output/` directory to the dataset selected by `CARD_BACKEND_OUTPUT_REPO`.

## Install

The pipeline runs via `uv` with no project venv. Required packages are pulled in at run time:

- `huggingface_hub` — for snapshot reads of upstream datasets and uploads.
- `datasets` — Arrow-backed parquet writer (sibling of `huggingface_hub`).

`pyarrow` is a transitive dependency of `datasets`.

## Run

Dry run (no upload, full output written under `output/`):

```bash
uv run --with huggingface_hub --with datasets --no-project python -m scripts.pipeline --dry-run
```

Generate the metric-looking string registry from the local EEE snapshot:

```bash
uv run --with huggingface_hub --no-project python -m scripts.build_metric_looking_registry
```

Upload to Hugging Face (during the parity migration — staging target):

```bash
HF_TOKEN=hf_xxx CARD_BACKEND_OUTPUT_REPO=j-chim/temp_evalcard_backend \
  uv run --with huggingface_hub --with datasets --no-project python -m scripts.pipeline
```

## Upload safety guard

`pipeline.resolve_upload_target()` requires an explicit destination:

- `CARD_BACKEND_OUTPUT_REPO`: target dataset (`j-chim/temp_evalcard_backend` during the migration).
- `CARD_BACKEND_ALLOW_PRODUCTION=1`: extra opt-in needed to publish to `evaleval/card_backend`.

Without either, the upload step raises rather than silently shipping. Migration scripts must NEVER set `CARD_BACKEND_ALLOW_PRODUCTION`.

## Cross-repo parity verification

After running the pipeline, validate the parquet payloads against the canonical TS adapters in `general-eval-card`:

```bash
uv run --with datasets --no-project python -m scripts.verify_parity \
  --pipeline-output ./output \
  --general-eval-card ../general-eval-card
```

Exits non-zero on any unexplained divergence between the parity parquet payloads and the TS-adapter outputs (`hfModelCardToEvaluationCardData`, `hfEvalEntryToListItem`, `hfEvalDetailToSummary`, `createModelFamilySummary`, `hfDeveloperDetailToSummary`).

## Environment variables

- `HF_TOKEN`: required for non-dry-run uploads.
- `CARD_BACKEND_OUTPUT_REPO`: required for non-dry-run uploads. Pinning the staging target keeps the pipeline from publishing to production by default.
- `CARD_BACKEND_ALLOW_PRODUCTION=1`: extra opt-in for `evaleval/card_backend`.
- `CONFIG_BATCH_SIZE`: optional. Controls how many EEE configs are loaded concurrently. Default: `4`.
- `EEE_LOCAL_DATASET_DIR`: optional local snapshot directory (used in CI to avoid HF rate limits).
- `BENCHMARK_METADATA_LOCAL_DIR`: optional local cache directory for `evaleval/auto-benchmarkcards`.
- `EEE_REFRESH_SNAPSHOT=1` / `BENCHMARK_METADATA_REFRESH=1`: force re-download of the corresponding snapshot.
- `CONFIGS` / `CONFIG_NAMES`: optional comma-separated config override.
- `CONFIG_LIMIT`: optional limit for quick smoke tests.
- `LOAD_INSTANCE_IN_DRY_RUN=1`: load detailed instance-level data even in dry-run mode (slow; off by default).

Lower `CONFIG_BATCH_SIZE` reduces peak disk and memory pressure on GitHub Actions runners as the source dataset grows.

## Tests

```bash
uv run --with pytest --with huggingface_hub --with datasets --no-project pytest tests/
```

Coverage is split across:
- `tests/test_signals.py` — interpretive-signals helpers.
- `tests/test_parity.py` — cleaning-spec primitives in `scripts/parity.py`.
- `tests/test_parity_adapters.py` — TS-shape adapter ports in `scripts/parity_adapters.py`.
- `tests/test_pipeline_integration.py` — end-to-end fixture pipeline + parity parquet emission + upload safety guard.
- `tests/test_verify_parity.py` — verifier-of-the-verifier (proves cross-repo verifier catches injected divergences).

## Notes

- The pipeline cleans and recreates `output/` on each run.
- Benchmark metadata is sourced only from the Hugging Face dataset `evaleval/auto-benchmarkcards`.
- `registry/metric_looking_strings.json` is generated from the local EEE snapshot and can be refreshed with `python -m scripts.build_metric_looking_registry`; the pipeline uses it to canonicalize metric aliases.
- Config load failures are logged and skipped; the skipped config list is recorded in `output/manifest.json`.

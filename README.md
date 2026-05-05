# eval-card-backend

Materialises evaluation artifacts from `evaleval/EEE_datastore`,
`evaleval/auto-benchmarkcards`, and `evaleval/entity-registry-data` into
a Parquet warehouse for the Eval Cards frontend.

The pipeline runs end-to-end via DuckDB in-process: it loads the three
upstream HF datasets, resolves identity through `eval-entity-resolver`,
computes the four interpretive signals (reproducibility, completeness,
provenance, comparability), and emits a snapshot of canonical tables
plus a thin view layer shaped to what the frontend renders.

## Install

```bash
uv sync
```

`eval-entity-resolver` is wired as a uv workspace path dep against a
sibling clone at `../eval-card-registry/`. CI overrides this with a git
URL — see `scripts/ci_install_resolver.py`.

## Run

Inspect what's already cached locally:

```bash
uv run eval-card-backend
```

Run the full pipeline (downloads HF snapshots if not cached, materialises
the warehouse):

```bash
uv run eval-card-backend canonicalise
```

Common flags:

```bash
# Limit to specific EEE configs
uv run eval-card-backend canonicalise --configs cnn_dailymail,xsum

# Smoke-test the first N configs
uv run eval-card-backend canonicalise --config-limit 3

# Pin the snapshot id (default: now in UTC)
uv run eval-card-backend canonicalise --snapshot-id 2026-05-04T00:00:00Z

# Custom warehouse / cache locations
uv run eval-card-backend canonicalise --warehouse path/to/warehouse \
                                      --cache-root path/to/stage_cache
```

## Stage caching

Each pipeline stage's terminal output is COPY-ed to
`<cache-root>/<snapshot>/<table>.parquet` so re-runs can resume mid-pipeline:

```bash
# Re-bake the view layer from cached canonical tables (skips Stages A–I)
uv run eval-card-backend canonicalise --from-stage J

# Run only Stages A–D for debugging; cache dir is the result
uv run eval-card-backend canonicalise --to-stage D

# Skip cache writes (cache reads still work for --from-stage)
uv run eval-card-backend canonicalise --no-cache
```

Stage letters: A (load) · B (explode) · C (resolve identity) · D (flatten
+ join dims) · E (per-row signals) · F (group signals) · G (dim
materialisation) · I (canonical-warehouse emit) · J (view-layer emit).

## Output layout

```
warehouse/<snapshot_id>/
├── fact_results.parquet           # one row per atomic score, all signal columns
├── benchmarks.parquet             # one row per resolved benchmark
├── models.parquet                 # one row per resolved model
├── canonical_metrics.parquet      # the registry's metric dim (snapshot-stamped)
├── eval_results_view.parquet      # one row per (model, benchmark, metric) triple
├── models_view.parquet            # one row per model, denormalised for the index page
├── evals_view.parquet             # one row per benchmark, multi-metric pre-pivoted
├── manifest.json                  # corpus scalars (model_count, eval_count, …)
├── headline.json                  # corpus signal aggregates (overall + by_category)
├── hierarchy.json                 # top-level composites[] tree + flat families[] lookup
├── comparison-index.json          # per-(eval, metric) leaderboards + inverse model→peer index
└── snapshot_meta.json             # pipeline run metadata (tables, sidecars, row counts)
```

The four canonical parquets are the source of truth (audit/debug);
`*_view.parquet` + the four JSON sidecars are pre-baked for the
frontend to read without GROUP BYs.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | `None` | Optional for public datasets; required for private. |
| `EEE_LOCAL_DATASET_DIR` | `.cache/eee_datastore` | Local cache for the EEE snapshot. |
| `BENCHMARK_METADATA_LOCAL_DIR` | `.cache/auto_benchmarkcards` | Local cache for benchmark cards. |
| `ENTITY_REGISTRY_LOCAL_DIR` | `.cache/entity_registry` | Local cache for the registry. |
| `WAREHOUSE_DIR` | `warehouse` | Output root; overridden by `--warehouse`. |
| `EEE_REFRESH_SNAPSHOT` | unset | Set to `1` to force-refetch the EEE snapshot. |
| `BENCHMARK_METADATA_REFRESH` | unset | Set to `1` to force-refetch the cards. |
| `ENTITY_REGISTRY_REFRESH` | unset | Set to `1` to force-refetch the registry. |

## Tests

```bash
uv run pytest
```

Tests use hand-built fixtures under `tests/fixtures/` and don't require
HF credentials.

## Continuous integration

`.github/workflows/sync.yml` runs the pipeline daily, then publishes the
warehouse snapshot tree to a target HF dataset (today:
`j-chim/temp_evalcard_backend`; flip `HF_TARGET_DATASET` env at the
workflow level to point elsewhere). The `HF_TOKEN` secret must be set on
the repo.

## Design notes

- `CLAUDE.md` is the operational source of truth (architecture pointers,
  ignored-config policy, hotfix retirement plans).
- `notes/01-` through `notes/08-` are the design specs for the canonical
  schema, producer pipeline, EEE schema vendoring, and Stage J view
  layer.

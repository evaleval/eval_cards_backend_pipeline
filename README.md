# eval-card-backend

Materialises evaluation artifacts from `evaleval/EEE_datastore` (plus
optional processed collection extracts from a private dataset; see below),
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
sibling clone at `../eval-entity-registry/`. CI overrides this with a git
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

Each cache directory records the `CACHE_SCHEMA_VERSION` the pipeline that
wrote it declared (`canonicalise/cache.py`). When a stage's cached output
changes shape the constant is bumped, and `--from-stage` against a cache
written before the bump fails immediately with a message naming the stale
directory. The fix is always a full run from Stage A.

The marker is written only once a stage's outputs are all on disk, and
writing stage X first deletes the cached outputs of every later stage. A run
cut short by `--to-stage` or by an interruption therefore leaves a cache that
either refuses to restore (no marker) or restores a single consistent
generation — never a Stage D rebuild sitting on top of yesterday's Stage J.
A restore also requires every table the requested stages declare; a missing
one fails with the same "re-run from Stage A" message.

## Output layout

```
warehouse/<snapshot_id>/
├── fact_results.parquet           # one row per atomic score, all signal columns
│                                  #   (+ is_headline, re-emitted in Stage J)
├── benchmarks.parquet             # one row per (composite, benchmark) appearance
├── composites.parquet             # one row per composite_slug
├── families.parquet               # one row per benchmark family
├── models.parquet                 # one row per resolved model
├── canonical_metrics.parquet      # the registry's metric dim (snapshot-stamped)
├── eval_results_view.parquet      # one row per (model, benchmark, metric) triple
├── models_view.parquet            # one row per model, denormalised for the index page
├── evals_view.parquet             # one row per benchmark, multi-metric pre-pivoted
├── comparison_evals.parquet       # comparison index (per-(eval, metric) leaderboards): one row per eval entry
├── comparison_metrics.parquet     #   one row per (eval, metric) leaderboard
├── comparison_scores.parquet      #   one row per score cell, in leaderboard order
├── manifest.json                  # corpus scalars (model_count, eval_count, …)
├── headline.json                  # corpus signal aggregates (overall + by_category)
├── hierarchy.json                 # top-level composites[] tree + flat families[] lookup
├── benchmark_index.json           # per-benchmark coverage / signal rollups
├── peer-ranks.json                # per-model peer rankings
└── snapshot_meta.json             # pipeline run metadata (tables, sidecars, row counts)
```

The six canonical parquets are the source of truth (audit/debug);
`*_view.parquet`, `comparison_*.parquet` + the JSON sidecars are pre-baked
for the frontend to read without GROUP BYs. The `comparison_*` tables hold
the comparison index: per-(eval, metric) leaderboards, from which the
inverse model→peer index is a re-pivot of `comparison_scores`. Snapshots
built before the tables existed carry the same content as
`comparison-index.json`; `scripts/backfill_comparison_tables.py` converts
them.

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
| `HF_OPENNESS_PROBE` | `1` | Set to `0` to skip the open-weights backfill (offline bakes, or to pin a snapshot to exactly what the registry states). |
| `EEE_INCLUDE_PRIVATE` | unset | Set to `1` to replace public collection-member rows with processed private extract rows (see below). |
| `EEE_PRIVATE_ALLOW_ZERO_MEMBERS` | unset | Set to `1` to allow a private extract whose member records are not public yet. |
| `EEE_PRIVATE_DATASET_REPO` | `evaleval/aisi-inference-scaling-data` | The private dataset. |
| `EEE_PRIVATE_REVISION` | unset | Commit sha of the private dataset; branches and tags are rejected. Unset resolves HEAD to a sha once per run. |
| `EEE_PRIVATE_LOCAL_DATASET_DIR` | `.cache/eee_private` | Local cache for the private extracts, separate from the public snapshot. |

### Processed private collection extract

Some collections are also served as a processed extract (`manifest.json`,
`results.parquet`, `trajectories.parquet`) from the private dataset
`EEE_PRIVATE_DATASET_REPO` under `collections/<name>/`, while their aggregate
member records are public in `evaleval/EEE_datastore`.

With `EEE_INCLUDE_PRIVATE=1` the pipeline downloads `collections/**` at
`EEE_PRIVATE_REVISION` into its own cache and applies each extract: it checks
every public member record's result count against the manifest, drops those
member rows, and injects the processed cells and trajectories. Manifest members
absent from the public corpus are allowed only when the manifest gives them no
results. A public record that carries the study's source name but is not a
manifest member fails the run. A full run also fails when no members are
public at all, unless `EEE_PRIVATE_ALLOW_ZERO_MEMBERS=1`. Injected cells keep
NULL `eee_record_url` and `instance_file_url`. `EEE_PRIVATE_REVISION` is the
extract's pin, independent of `EEE_REVISION`, and is recorded in
`upstream_pins.eee_private`. The token needs read access to the private dataset.

With the toggle off (the default) nothing private is read and the member
records are processed as ordinary records, so protocol variants collapse into
single cells and no trajectories or collection context are emitted. CI keeps
the toggle on. A stage cache records the toggle and private revision that built
it and cannot be resumed across a change to either.

Each applied extract's accounting (members present and absent, rows dropped and
injected, final row counts) is written to `collection_adapter_accounting` in
`snapshot_meta.json` and `manifest.json` and printed in the Sync Pipeline step
summary.

To update an extract: run its extractor against the local raw data
(`--source-dir`), upload the three files to the private dataset, then set
`EEE_PRIVATE_REVISION` in `.github/workflows/sync.yml` to that commit.

When member records first reach the public datastore: keep `EEE_REVISION`
pinned until a flat rebuild includes them, then in one change move the pin to
that commit and remove `EEE_PRIVATE_ALLOW_ZERO_MEMBERS` from the workflow, and
check the accounting in the step summary.

### Open-weights backfill

The entity registry leaves `canonical_models.open_weights` unset for most
models. Stage A fills the gap from the Hugging Face Hub: a model whose
weights are published has a model repo, so a resolvable id is evidence of
open weights. Gated repos count as open — the gate is a terms click, not a
closed model.

The backfill only ever writes `TRUE`, and only over a `NULL`. A lookup that
misses leaves the row `NULL`, because a miss does not distinguish a
proprietary model from one that was renamed, deleted, made private, or
never spelled the same way upstream. Verdicts are cached under
`<cache-root>/hf_openness.json`; network failures are not cached and never
fail the bake.

## Tests

```bash
uv run pytest
```

Tests use hand-built fixtures under `tests/fixtures/` and don't require
HF credentials. Each fixture directory a loader reads carries an empty
`.hand_built` sentinel, which tells the loaders the directory is the corpus
itself (never wiped, never fetched into) even when a stray download has
left its markers behind; `tests/test_fixtures_offline.py` checks every
fixture directory has one. A refresh flag aimed at such a directory raises
instead of deleting it, and the suite strips every refresh and pin variable
from the environment.

## Development checks

Three checks run as pre-commit hooks: `ruff` (lint only, no rewrites), the
pre-commit-hooks file checks (YAML and TOML parse, no merge-conflict
markers, no large files, no trailing whitespace) and `scripts/check_code_hygiene.py`, which rejects
added comments that point at a spec section, a phase, a working session, or
describe what the code did before. A line containing `hygiene: ok` is
skipped by every rule; use it for a false positive. The hygiene hook reads
the staged lines, so `pre-commit run --all-files` runs ruff and the file
checks over the whole tree but the hygiene gate only over what is staged;
`scripts/check_code_hygiene.py --diff <range>` checks a commit range.

`pre-commit` lives in the `lint` dependency group, which a plain `uv sync`
leaves out and removes again; `uv run --group lint` installs the group on
demand, so run pre-commit through it. Install the hooks once per clone. An
existing clone with `core.hooksPath=hooks` set first unsets it, because
`pre-commit install` refuses while it is set:

```bash
git config --unset core.hooksPath      # the old hooks/ dir held the git-lfs hooks and the hygiene hook
git lfs install                        # reinstalls the lfs hooks into .git/hooks
uv run --group lint pre-commit install # adds the pre-commit hook beside them
```

A fresh clone needs only the last two lines. If `pre-commit install` still
refuses, the setting comes from another config file, which a repository-local
unset leaves in place. Find it with:

```bash
git config --show-scope --show-origin core.hooksPath
```

A `global` (or `system`) scope means `~/.gitconfig` (or the system config)
sets it for every repository on the machine. Remove it there with
`git config --global --unset core.hooksPath` (`--system` for the system
file) once no other repository relies on that hooks directory; pre-commit
cannot install while any scope sets it.

The lfs hooks (`pre-push`,
`post-checkout`, `post-commit`, `post-merge`) and pre-commit's `pre-commit`
hook are separate files in `.git/hooks`, so either install can be rerun
without touching the other.

CI (`.github/workflows/ci.yml`) runs the same hooks on every pull request
and push to `main`, then the hygiene gate over the added lines of the pull
request (`scripts/check_code_hygiene.py --diff origin/<base>...HEAD`) or of
the push (`--diff <before>...<sha>`), and the test suite.

## Continuous integration

`.github/workflows/sync.yml` runs the pipeline daily, then publishes the
warehouse snapshot tree to `evaleval/card_backend` on HF. Override via
the `HF_TARGET_DATASET` env at the workflow level. The `HF_TOKEN` secret
must be set on the repo.

### Upstream revisions: follow or pin

Each run reads the head of `evaleval/EEE_datastore` and of
`evaleval/entity-registry-data`, resolved to commit shas by
`scripts/ci_resolve_revisions.sh` and shown in the run summary. To hold
either one at a known commit, set a repo variable (Settings > Secrets and
variables > Actions > Variables, or `gh variable set`):

| Repo variable | Holds | Empty or absent |
| --- | --- | --- |
| `EEE_REVISION` | commit sha of `evaleval/EEE_datastore` | follow head |
| `ENTITY_REGISTRY_REVISION` | commit sha of `evaleval/entity-registry-data` | follow head |

A value that is not a 40-character lowercase hex sha fails the run. To
unpin, delete the variable (`gh variable delete EEE_REVISION`). The
resolver package and the seed YAMLs are not set separately: they are
checked out at the registry repo commit that the selected registry data
names in its `manifest.json` (`seed_git_sha`). The benchmark cards and the
private collection stay pinned in `sync.yml`.

### Publishing and the shrink guard

Before uploading, `scripts/ci_publish_warehouse.py` compares the new
snapshot with the published `warehouse/latest/snapshot_meta.json` on
`eee_records`, `fact_results` and the number of configs. If any is below
90% of the published value, or the published file cannot be read, nothing
is uploaded and the run fails. For a shrink that is intended, start the
workflow by hand with `allow_shrink` on; that snapshot becomes the new
baseline. Starting it by hand with `publish` off runs everything and only
prints the comparison (`--check-only`).

### Alert issue

`.github/workflows/sync-alert.yml` keeps one open issue labelled
`sync-alert` as the thread for pipeline trouble. It comments there when a
run on `main` fails, when a run succeeds but rejected upstream records at
load time (`stage_a_drops` in `snapshot_meta.json`), and when the published
snapshot is more than 36 hours old or its metadata cannot be read. The
next clean run on `main` that actually publishes comments "recovered" and
closes the issue; a run with `publish` off never does. To replay it
against a past run, start
the workflow by hand with `run_id`; that is a dry run that only prints
what it would post unless `write` is on.

### Rollback

1. Revert the offending commit on `main` and start the workflow by hand.
2. If the bad input is upstream data, pin instead: set the variables above
   to the last good commits and start the workflow by hand. The values
   hardcoded before runs followed head were `EEE_REVISION`
   `0fe996f1b578d43d7e4d16be814c961b6f778445` and
   `ENTITY_REGISTRY_REVISION` `9405e27e6b7a96fe9721b8b33a80fd349fabf6b9`
   (resolver `d3f2248108de432b3d3ba8669f653510e4b07d4b`, which that
   registry commit selects by itself).
3. If `warehouse/latest` is bad in the meantime, point the frontend
   Space's `SNAPSHOT_URL` at the last good dated folder
   (`warehouse/<snapshot_id>`); dated folders are never overwritten.

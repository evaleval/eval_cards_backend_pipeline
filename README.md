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
| `EEE_DATAFIX` | unset | Set to `1` to fill verified missing metadata from source evidence. Default off. |
| `REPRO_EXEMPT_NON_GENERATIVE` | unset | Comma-separated declaration sources to exempt from temperature and max_tokens requirements: `lm_eval_output_type`, `rewardbench_model_type`. Empty means off. |
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

## EEE datafix overlay

Some upstream EEE records omit run settings the source itself publishes
elsewhere, which makes the interpretive signals read a documentation gap as
a reproducibility gap. The opt-in overlay corrects those records at the Stage A boundary, so every stage below it
works from corrected data and nothing downstream needs to know.

These are temporary data hotfixes. Their source rules and appliers live in
`canonicalise/resolution_hotfixes.py`, with an upstream adapter PR pointer
and a removal condition on each builder. Generic accounting, evidence files,
and table-building tools remain in `datafix/`. Remove the fills once the
pinned datastore has been re-ingested with the upstream fixes and no covered
field gaps remain. `EEE_DATAFIX` defaults off so normal runs retain the
upstream records until the overlay is explicitly selected.

Each fix is two separable halves:

- a **patch table** under `datafix/data/`, vendored as canonical JSON and
  keyed by the record identity the upstream datastore uses, so it carries no
  backend paths, no cache state and nothing specific to this pipeline;
- a **pure applier** over one record dict, plus a **builder** that
  regenerates the table from its evidence source. Reruns on the same
  evidence are byte-identical, so a rebuild can be diffed rather than
  trusted.

A fix whose evidence is inside the record itself needs no table and is
registered for every source config instead of one; the registry answers per
config, so a config no fix reaches still opens nothing.

Four fixes ship today:

| Fix | What it fills | Evidence |
| --- | --- | --- |
| `vals_ai_max_tokens` | The output-token cap and the temperature on SAGE, MedCode and MedScribe results that state neither. | The benchmarks' own pages state, in one sentence each, temperature 1 and a 30,000-token cap. The two fields are corroborated independently: a gap is filled only where the same record, or the same model on the same benchmark, states that number for that field elsewhere. |
| `hfopenllm_v2_harness_configs` | Temperature and output-token cap on IFEval and MATH Level 5 results. | The leaderboard's published lm-evaluation-harness dumps, matched to the record by exact model id. |
| `inspect_plan_config` | Temperature and output-token cap on results converted from inspect_ai logs, in any config. | The log's own plan, whose config is the task's generate config. Taken only under a closed rule: a checked inspect_ai version with no other package, a plan whose solvers are all ones the rule was read against, a last step that generates with the run's own model, and no step setting a temperature or a cap of its own. |
| `arc_agi_logged_request_args` | The output-token cap, and a temperature where one was sent, on ARC-AGI public-set scores. | ARC Prize's published run outputs, where every attempt records the arguments the request carried. A run is used only when its logs reconstruct the published score and cost, state one cap throughout, name the pair index of every attempt, finished before the record was read, and no path its directory ever held was touched by a second commit. |

Doctrine, the same for every fix: only ever fill a value that is missing,
never overwrite one that is present, never replace a container that is not
a dict, and leave anything ambiguous missing and counted. The overlay
writes only `generation_config.generation_args`; provenance lives in the
patch tables, the export manifest and `snapshot_meta.datafix`, never as a
marker inside the record. A record is validated against the upstream
contract before the overlay runs (so a fix can never mask malformed input)
and again afterwards if it changed (a patched record that fails is a bug
here, and fails the run).

Every fix reads a record as it arrived, so a value one fills is never what
another reads as already stated. Two fixes landing on the same field of the
same result are therefore two proposals about one gap: agreeing on the
value they fill it once, attributed to whichever the registry lists first
and naming the others on the manifest event; disagreeing, the field stays
missing and each of them is counted `cross_fix_conflict`.

`snapshot_meta.json` carries a top-level `datafix` block: whether the
overlay was on, its revision and fingerprint, and per fix the table
checksum, the evidence revision, and what was filled, skipped or left
alone. A fix that never ran is reported from the registry's own entry, so
identifying the overlay and reporting a run that filled nothing open no
patch table. The counts belong to the run, not to the process. The stage
cache records the same fingerprint alongside the upstream revision and
config selection, so `--from-stage` refuses to run later stages over
records loaded under a different overlay, and a resumed run reports the
counts of the data actually in `eee_raw`.

Rebuilding a table, and replaying the corrections outside the pipeline:

```bash
# Regenerate a patch table from its evidence (writes datafix/data/*.json)
uv run eval-card-backend datafix build-vals-table \
    --eee-root .cache/eee_datastore --eee-revision <sha>
uv run eval-card-backend datafix build-hfopenllm-table \
    --results-dir path/to/open-llm-leaderboard-results/<sha>
# --runs-root holds one bulk snapshot of each published ARC dataset,
# --history-root a clone of each, read once for the commits per file
uv run eval-card-backend datafix build-arc-agi-table \
    --eee-root .cache/eee_datastore --eee-revision <sha> \
    --runs-root .cache/arc_public_eval --history-root .cache/arc_history

# Write patched copies of the affected records, in datastore layout,
# plus patch_manifest.jsonl listing every field written. --out has to be
# empty or absent, so a re-export can't leave records from an earlier one
# standing next to a manifest that does not list them
uv run eval-card-backend datafix export \
    --eee-root .cache/eee_datastore --out path/to/export
```

A build that produces the table already shipped is a no-op. One that
produces something else prints what changed (patches added or removed,
defaults, overrides, conflicts, verified-model counts) and exits non-zero
without touching the vendored file; `--accept-changes` writes it once the
differences have been read.

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

The non-generative exemption is independent of `EEE_DATAFIX`. It accepts only
explicit source declarations from HF Open LLM v2 harness likelihood scoring or RewardBench
classifier/DPO model types. Unknown source names fail configuration loading.
Agentic plan and limit requirements remain active. Rows with no remaining
requirements carry `repro_applicable=false`; model bands exclude them and become `not_applicable`
when no applicable cells remain. Both settings are recorded in snapshot
metadata and stage caches reject a change to either setting on resume.
Cells containing exempt observations publish the union of their observations'
requirements and missing fields, so an exempt representative cannot hide an
applicable observation. The headline retains its all-triple gap rate and also
publishes `applicable_triples` and `applicable_reproducibility_gap_rate`; the
latter is null when no triple is applicable.

# Runbook: re-extracting the AISI inference-scaling collection

How to get new UK AISI inference-scaling records onto the eval cards site, and
what to check before the result is published.

Audience: anyone maintaining `vendor/collections/aisi_inference_scaling/`.
You do not need to have written the extractor.

This file sits next to the extractors it documents. Note that `notes/` is
gitignored, so the `notes/collections-spec.md` referenced from `sync.yml`,
`collections_curated.yaml` and `vendor/README.md` is not in the repository.

---

## 1. Why records in the datastore are not enough

`uk-aisi-inference-scaling` is a **curated collection** (`collections_curated.yaml`).
`enumerate_members` in `scripts/collections/aisi_inference_scaling.py` claims every
datastore record whose `slug(source_metadata.source_name)` equals the study slug —
which is what groups them under one family on the evaluator page.

Claimed records are then **dropped** from the normal canonicalise path (that is
`expected_drop_count`) and re-emitted as synthetic results read from
`vendor/collections/aisi_inference_scaling/results.parquet`.

So for anything in this collection:

> merging records into EEE_datastore does nothing on its own.
> The vendored extract is the only route into the warehouse.

If a new AISI benchmark is not appearing, this is almost always why.

## 2. The three things that must line up

| | what | where |
|---|---|---|
| 1 | the records | `evaleval/EEE_datastore` → `data/<collection>/…` |
| 2 | registry entities | `evaleval/eval-card-registry` → `seed/benchmarks.yaml`, `seed/metrics.yaml` |
| 3 | vendor extract **+** `EEE_REVISION` | this repo |

(2) is what turns a raw slug into a display name on the site. A benchmark with no
`canonical_benchmarks` entry still renders, just as its bare id.

(3) is **one unit of work, not two.** `canonicalise` hard-fails when the manifest's
`eee_revision` and `sync.yml`'s `EEE_REVISION` disagree, so they must land in the
same PR. The workflow below does that for you.

## 3. Running it

Actions → **AISI collection extract** → *Run workflow*.

| input | value |
|---|---|
| `revision` | an EEE_datastore **`cron: flat rebuild`** commit SHA (see §4) |
| `open_pr` | leave on — opens a PR with the regenerated vendor files and the pin bump |
| `allow_rate` | leave empty unless a gate fires and you have diagnosed it (§6) |
| `allow_reconciliation` | same |

Expect **1–3 hours**: ~1.2 GB of aggregate records, ~10 GB of sample transcripts,
then a sequential parse of ~36k trajectories. The transcripts are parsed and
discarded, never committed.

You can also run it locally — same script, same arguments — but there is no reason
to prefer that, and it leaves ~11 GB in `~/.cache/huggingface/hub`:

```sh
uv run python scripts/collections/aisi_inference_scaling.py --revision <sha>
```

## 4. Choosing `revision` — the one real trap

**Do not pin the commit that merged the records.** Pin a flat rebuild that came
*after* it.

`ensure_snapshot` builds its file listing from the datastore's `flat/` index, not by
walking `data/`. `flat/` is rebuilt by a cron in `evaleval/every_eval_ever`, so a
commit from minutes ago resolves to a snapshot that does not contain its own records.
The extractor only **warns** and carries on:

```
flat index lags the pinned commit by 10.4 hours; records merged in between
are not in this snapshot; pin a flat-rebuild commit to avoid this
```

That is how you spend an hour re-deriving the corpus you already had and get a
green run with an unchanged extract. It has happened.

Find a usable pin:

```sh
curl -s "https://huggingface.co/api/datasets/evaleval/EEE_datastore/commits/main?limit=20" \
  | python3 -c "import json,sys;[print(c['id'], c['date'][:19], c['title'][:60]) for c in json.load(sys.stdin)]"
```

Take the newest `cron: flat rebuild … (N/N)` commit — the **last** part of the group,
which is the complete state. If the records landed after the most recent rebuild,
trigger one and wait ~10–60 min:

```sh
gh workflow run flat-rebuild.yml --repo evaleval/every_eval_ever
```

Verify before dispatching the extract — this is also the workflow's pre-flight step,
so a bad pin fails in seconds rather than after the download:

```sh
uv run python scripts/collections/check_flat_pin.py --revision <sha> \
  --require-collection aisi-cyber-ctfs --require-collection aisi-the-last-ones
```

A small lag is normal and passes: a flat rebuild's descriptor is written when the
rebuild *starts*, so it always pre-dates its own commit by the rebuild duration.
The decisive test is that the collections are present.

## 5. Reviewing the PR

The run summary prints member counts per benchmark, `aggregate_only`,
`expected_drop_count` and `cells_dropped`. Check:

- **Member counts moved the way you expect.** New benchmark → its rows appear;
  every other benchmark's count unchanged unless the datastore actually changed.
- **`cells_dropped` is empty**, or every entry has a reason you accept. A dropped
  cell is a (benchmark, model, protocol) combination with no computable score.
- **Reconciliation** lines in the log are at their usual exact rates. This check
  recomputes each member record's own published aggregate from its own rows; a drop
  means the score reading no longer matches upstream, which is a real defect, not
  noise.
- **What moved between pins.** Diff the datastore commit range and say in the PR
  which collections changed — the manifest's `eee_revision_note` follows this
  convention. Anything touching an existing member's records deserves a closer look
  than a cron refresh of an unrelated collection.

## 6. When a gate fires

The extractor fails loudly rather than shipping a partial corpus. In rough order of
likelihood:

| failure | meaning | what to do |
|---|---|---|
| `member records under undeclared benchmark(s)` | a new benchmark joined the collection with no `BENCHMARKS` entry | add its score semantics to `BENCHMARKS` **and** `collections_curated.yaml`; see §7 for the aggregate-only case |
| `N sample file(s) failed to download` | transient Hub errors | just re-run; a partial corpus would bias the aggregates, which is why it refuses |
| `extraction rate(s) above the 10% gate` | some class of row is being lost more than usual | diagnose first. `--allow-rate NAME=explanation` records the reason in the manifest — it is an audit trail, not a mute button |
| `per-record aggregate identity below …` | recomputed aggregates no longer match upstream's own summaries | the declared score reading is wrong for that benchmark. Fix the reading; only override with a written justification |
| `row reconciliation failed` | trajectories + errors + duplicates ≠ rows | a parsing bug; do not override |

## 7. Adding an aggregate-only benchmark

Some benchmarks ship headline aggregates and no transcripts — the two cyber
benchmarks are privately held, so no sample files exist for them. They take the
`published_aggregate` path: the published figure is restated, never recomputed, and
`main()` splits them out **before** `stream_trajectories`, so no sample fetch is ever
attempted.

To add one, declare it in both places (they mirror each other):

```python
# scripts/collections/aisi_inference_scaling.py
"<benchmark-id>": {
    "outcome": "binary", "metric": "accuracy",
    "trajectory_outcome": None, "aggregate": "published_aggregate",
    "upstream_rule": None,
    "aggregate_only": True,
    "result_series": "<series in the dotted evaluation_name>",
},
```

Notes:

- `upstream_rule: None` makes reconciliation skip it — correct, since there is
  nothing to recompute against.
- Each published result becomes one cell on the `token_limit` protocol axis, so a
  record with five token thresholds yields five cells. No threshold is privileged
  as "the" headline.
- If the benchmark reports two series per point, declare the meaningful one as
  `result_series` and the other as `companion_series`; the companion is carried in
  `score_details.details` rather than lost.
- A distinct measurement gets its own `metric` id, never blended with `accuracy`.
  Register it in `eval-card-registry` `seed/metrics.yaml` using the hyphenated slug
  convention (`healthbench-score`, `aisi-mean-progress-score`) or the UI falls back
  to the raw id.

## 8. Known rough edges

- **No incremental mode.** Adding one aggregate-only record re-streams all ~10 GB of
  transcripts for members that did not change. The committed `trajectories.parquet`
  could in principle be reused when no trajectory member changed, but TerminalBench's
  `ever_correct` outcome is recomputed from per-row submission history the parquet
  does not carry, so it needs care.
- **The flat rebuild has been failing** on every scheduled run (195 `livebench`
  records re-emitted under existing UUIDs with different bytes). It still commits, so
  pins remain usable, but those records are excluded from `flat/` and the red job
  masks the real problem.
- **In-place edits to published records are hostile to `flat/`.** Changing a record's
  content under the same UUID gets it excluded as a conflict. Re-emit under a fresh
  UUID instead.
- **`notes/collections-spec.md` is referenced** from `sync.yml`, `vendor/README.md`
  and `collections_curated.yaml` but does not exist in the repo.

# vendor/

Curated lookup tables that the pipeline joins in but does not (yet) compute
itself. Hand-maintained and refreshed out-of-band against a pinned
`evaleval/EEE_datastore` snapshot; regenerating them automatically is future
work.

## `is_verified_evaluator.parquet`

Per evaluation record: whether the evaluation was **submitted by the organisation
that ran it** (vs. re-hosted from someone else's leaderboard). Stage D LEFT-joins
this on `evaluation_id` and emits the boolean `is_verified_evaluator` on
`fact_results`; records absent from the lookup default to `false`.

| column | type | note |
| --- | --- | --- |
| `evaluation_id` | string | join key (matches `fact_results.evaluation_id`) |
| `record_uuid` | string | source record id; cross-check |
| `source_organization_name` | string | the org that ran the eval |
| `submitter_author` | string | who submitted it upstream ("" if direct-pushed) |
| `is_verified_evaluator` | bool | true = submitter belongs to the source org |
| `reason` | string | one-line justification for the call |

## `evaluator_affiliation.parquet`

The 31 distinct `(submitter_author, source_organization_name)` decisions behind
the per-record table, each with a record count and a one-line `reason`. Read this
to audit *why* a record is (un)validated.

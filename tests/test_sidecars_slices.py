"""Sidecars — slice surfacing in `hierarchy.json`.

`write_hierarchy` populates `composites[].benchmarks[].slices[]`: a
per-(composite, benchmark) list of `{key, display_name, is_bare_stem,
metrics[]}`. These tests exercise `_hierarchy_composite_slices` directly.

They build minimal synthetic `fact_results` + `canonical_metrics`
+ `benchmarks` tables and call the sidecar helper. The end-to-end
fixture corpus is single-raw across all benchmarks, so it can't
exercise the populated-slice path; that's why this test stays at the
helper level.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise.sidecars import (
    _hierarchy_composite_slices,
)


_FACT_DDL = """
CREATE TABLE fact_results (
    composite_slug         VARCHAR,
    benchmark_id           VARCHAR,
    slice_key              VARCHAR,
    slice_name             VARCHAR,
    metric_id              VARCHAR,
    model_key              VARCHAR,
    org_raw                VARCHAR,
    -- Aggregation keys mirror the canonical ids in these synthetic
    -- tests (no resolution-failure cases). Computed columns let
    -- positional INSERTs against the original 7-column shape keep
    -- working unchanged.
    benchmark_key          VARCHAR AS (benchmark_id),
    metric_key             VARCHAR AS (metric_id),
    -- No scoring-variant qualifiers in these synthetic rows, so the
    -- registry-lookup key is the observation key.
    metric_base_key        VARCHAR AS (metric_id),
    model_aggregation_key  VARCHAR AS (model_key),
    -- Stage D reads the aggregation level off the raw evaluation_name's
    -- last segment; here the slice key stands in for it, so a synthetic
    -- `<group> overall` row is a group rollup and everything else a leaf.
    aggregate_level        VARCHAR AS (
        CASE WHEN slice_key LIKE '% overall' THEN 'subgroup' ELSE 'leaf' END),
    -- The slice helper filters on the Stage J headline map; synthetic rows
    -- are all headline, so a generated key + a view over it is enough.
    fact_id                VARCHAR AS (
        COALESCE(composite_slug, '') || '|' || COALESCE(benchmark_id, '')
        || '|' || COALESCE(slice_key, '') || '|' || COALESCE(metric_id, '')
        || '|' || COALESCE(model_key, '') || '|' || COALESCE(org_raw, ''))
)
"""

_BENCH_DDL = """
CREATE TABLE benchmarks (
    composite_slug      VARCHAR,
    benchmark_id        VARCHAR,
    parent_benchmark_id VARCHAR,
    is_slice            BOOLEAN
)
"""

_METRICS_DDL = """
CREATE TABLE canonical_metrics (
    id           VARCHAR,
    display_name VARCHAR
)
"""


def _seed_minimal_tables(con):
    """Provide the three tables the slice helper reads, plus a one-row
    `benchmarks` stand-in so queries that join it run."""
    con.execute(_FACT_DDL)
    con.execute(
        "CREATE VIEW fact_headline AS "
        "SELECT fact_id, TRUE AS is_headline FROM fact_results"
    )
    con.execute(_BENCH_DDL)
    con.execute(_METRICS_DDL)
    con.execute("INSERT INTO benchmarks VALUES ('helm-classic', 'mmlu', NULL, FALSE)")


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# per-(composite, benchmark) slices[]
# ---------------------------------------------------------------------------


def test_slices_empty_when_no_slice_rows(con):
    """Standalone single-raw benchmark → empty slices[]."""
    _seed_minimal_tables(con)
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("helm-classic", "mmlu", None, None, "accuracy", "m1", "OpenAI"),
        ],
    )
    assert _hierarchy_composite_slices(con, "helm-classic", "mmlu", []) == []


def test_slices_populated_with_per_slice_metrics(con):
    """Each slice carries its own metric list. Multiple metric rows
    per (slice, metric) pair de-dup at the metric level; orgs land in
    `sources[]` deduplicated."""
    _seed_minimal_tables(con)
    con.executemany(
        "INSERT INTO canonical_metrics VALUES (?, ?)",
        [("accuracy", "Accuracy"), ("exact-match", "Exact Match")],
    )
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("helm-classic", "mmlu", "anatomy",   "Anatomy",   "accuracy",    "m1", "OpenAI"),
            ("helm-classic", "mmlu", "anatomy",   "Anatomy",   "accuracy",    "m2", "Scale AI"),
            ("helm-classic", "mmlu", "anatomy",   "Anatomy",   "exact-match", "m1", "OpenAI"),
            ("helm-classic", "mmlu", "astronomy", "Astronomy", "accuracy",    "m1", "OpenAI"),
        ],
    )

    slices = _hierarchy_composite_slices(con, "helm-classic", "mmlu", [])
    assert len(slices) == 2

    by_key = {s["key"]: s for s in slices}
    anatomy = by_key["anatomy"]
    assert anatomy["display_name"] == "Anatomy"
    assert anatomy["is_bare_stem"] is False
    metric_keys = sorted(m["key"] for m in anatomy["metrics"])
    assert metric_keys == ["accuracy", "exact-match"]
    accuracy = next(m for m in anatomy["metrics"] if m["key"] == "accuracy")
    assert accuracy["display_name"] == "Accuracy"
    assert sorted(accuracy["sources"]) == ["OpenAI", "Scale AI"]

    astronomy = by_key["astronomy"]
    assert astronomy["display_name"] == "Astronomy"
    assert [m["key"] for m in astronomy["metrics"]] == ["accuracy"]
    assert astronomy["metrics"][0]["sources"] == ["OpenAI"]


def test_slices_excludes_null_slice_rows(con):
    """A benchmark can have a mix of slice and non-slice rows. NULL
    slice_key rows are dropped from slices[]; they belong to the parent
    benchmark's metrics[] aggregation, not a sub-slice."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO canonical_metrics VALUES ('accuracy', 'Accuracy')")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("helm-classic", "mmlu", "anatomy", "Anatomy", "accuracy", "m1", "OpenAI"),
            ("helm-classic", "mmlu", None,      None,      "accuracy", "m1", "OpenAI"),
        ],
    )

    slices = _hierarchy_composite_slices(con, "helm-classic", "mmlu", [])
    assert [s["key"] for s in slices] == ["anatomy"]


def test_slices_filtered_by_composite_and_benchmark(con):
    """Calling for one (composite, benchmark) only returns its slices,
    not another multi-slice benchmark in the same fact_results."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO benchmarks VALUES ('mmlu-pro-leaderboard', 'mmlu-pro', NULL, FALSE)")
    con.execute("INSERT INTO canonical_metrics VALUES ('accuracy', 'Accuracy')")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("helm-classic",         "mmlu",     "anatomy", "Anatomy", "accuracy", "m1", "OpenAI"),
            ("mmlu-pro-leaderboard", "mmlu-pro", "physics", "Physics", "accuracy", "m1", "OpenAI"),
        ],
    )
    assert [s["key"] for s in _hierarchy_composite_slices(
        con, "helm-classic", "mmlu", []
    )] == ["anatomy"]
    assert [s["key"] for s in _hierarchy_composite_slices(
        con, "mmlu-pro-leaderboard", "mmlu-pro", []
    )] == ["physics"]


def test_slice_display_name_picks_deterministic_representative(con):
    """When the same slice_key has multiple slice_name casings across
    rows (e.g. 'MMLU' + 'mmlu' folded into slice_key='mmlu'), MIN picks
    the lex-earliest so re-runs are byte-stable."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO canonical_metrics VALUES ('accuracy', 'Accuracy')")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("helm-classic", "mmlu", "mmlu", "MMLU", "accuracy", "m1", "OpenAI"),
            ("helm-classic", "mmlu", "mmlu", "mmlu", "accuracy", "m2", "OpenAI"),
            ("helm-classic", "mmlu", "mmlu", "Mmlu", "accuracy", "m3", "OpenAI"),
            ("helm-classic", "mmlu", "anatomy", "Anatomy", "accuracy", "m1", "OpenAI"),
        ],
    )

    slices = {
        s["key"]: s
        for s in _hierarchy_composite_slices(con, "helm-classic", "mmlu", [])
    }
    # 'MMLU' < 'Mmlu' < 'mmlu' under default ASCII comparison.
    assert slices["mmlu"]["display_name"] == "MMLU"
    # `mmlu` slice key matches the benchmark id → flagged as bare-stem.
    assert slices["mmlu"]["is_bare_stem"] is True


def test_group_rollup_slices_nest_their_member_tasks(con):
    """A source that reports a group rollup beside the tasks inside it
    (global_mmlu's `fr_overall` next to `fr_stem`) used to render all of them
    as peers, so the tree said the French rollup and French STEM were
    siblings. The rollup's key ends in the aggregate marker, and every task
    whose key extends the prefix left over is a member of it."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO canonical_metrics VALUES ('accuracy', 'Accuracy')")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("helm-classic", "mmlu", "global mmlu fr overall",
             "fr_overall", "accuracy", "m1", "OpenAI"),
            ("helm-classic", "mmlu", "global mmlu fr stem",
             "fr_stem", "accuracy", "m1", "OpenAI"),
            ("helm-classic", "mmlu", "global mmlu fr business",
             "fr_business", "accuracy", "m1", "OpenAI"),
            ("helm-classic", "mmlu", "global mmlu de overall",
             "de_overall", "accuracy", "m1", "OpenAI"),
            ("helm-classic", "mmlu", "global mmlu de stem",
             "de_stem", "accuracy", "m1", "OpenAI"),
            # a task with no rollup above it stays directly under the benchmark
            ("helm-classic", "mmlu", "global mmlu", "overall",
             "accuracy", "m1", "OpenAI"),
        ],
    )
    parents = {
        s["key"]: s["parent_key"]
        for s in _hierarchy_composite_slices(con, "helm-classic", "mmlu", [])
    }
    assert parents["global mmlu fr overall"] is None
    assert parents["global mmlu de overall"] is None
    assert parents["global mmlu fr stem"] == "global mmlu fr overall"
    assert parents["global mmlu fr business"] == "global mmlu fr overall"
    assert parents["global mmlu de stem"] == "global mmlu de overall"
    assert parents["global mmlu"] is None


def test_slices_without_any_rollup_keep_a_null_parent(con):
    """MMLU-subject style slices, where no group aggregate exists at all."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO canonical_metrics VALUES ('accuracy', 'Accuracy')")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("helm-classic", "mmlu", "anatomy", "Anatomy", "accuracy", "m1", "OpenAI"),
            ("helm-classic", "mmlu", "astronomy", "Astronomy", "accuracy", "m1", "OpenAI"),
        ],
    )
    slices = _hierarchy_composite_slices(con, "helm-classic", "mmlu", [])
    assert all(s["parent_key"] is None for s in slices)
    assert all(s["aggregate_level"] == "leaf" for s in slices)

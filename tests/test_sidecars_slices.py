"""Sidecars — slice surfacing in `hierarchy.json`.

`write_hierarchy` populates two slice-derived fields:

  - `stats.slice_count`: distinct (benchmark_id, slice_key) pairs in
    `fact_results` where slice_key is non-NULL. Drives the homepage strip.
  - `composites[].benchmarks[].slices[]`: per-(composite, benchmark)
    list of `{key, display_name, is_bare_stem, metrics[]}`.

These tests build minimal synthetic `fact_results` + `canonical_metrics`
+ `benchmarks` tables and call the sidecar helpers directly. The
end-to-end fixture corpus is single-raw across all benchmarks, so it
can't exercise the populated-slice path; that's why this test stays at
the helper level.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise.sidecars import (
    _hierarchy_composite_slices,
    _hierarchy_stats,
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
    model_aggregation_key  VARCHAR AS (model_key)
)
"""

_BENCH_DDL = """
CREATE TABLE benchmarks (
    composite_slug      VARCHAR,
    benchmark_id        VARCHAR,
    parent_benchmark_id VARCHAR
)
"""

_METRICS_DDL = """
CREATE TABLE canonical_metrics (
    id           VARCHAR,
    display_name VARCHAR
)
"""


def _seed_minimal_tables(con):
    """Slice-count SQL also reads `benchmarks` (for benchmark_count).
    Provide a one-row stand-in so the sidecar query runs even though
    those non-slice counts aren't what we're asserting on."""
    con.execute(_FACT_DDL)
    con.execute(_BENCH_DDL)
    con.execute(_METRICS_DDL)
    con.execute("INSERT INTO benchmarks VALUES ('helm-classic', 'mmlu', NULL)")


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# slice_count
# ---------------------------------------------------------------------------


def test_slice_count_zero_when_no_slices(con):
    """Single-raw corpus → every fact row has slice_key NULL → no slices."""
    _seed_minimal_tables(con)
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("helm-classic", "mmlu", None, None, "accuracy", "openai/gpt-4o", "OpenAI"),
            ("helm-classic", "mmlu", None, None, "accuracy", "anthropic/claude", "Anthropic"),
        ],
    )

    stats = _hierarchy_stats(con, [], [])
    assert stats["slice_count"] == 0


def test_slice_count_counts_distinct_benchmark_slice_pairs(con):
    """slice_count is COUNT DISTINCT over (benchmark_id, slice_key);
    duplicate (benchmark, slice) rows from different models / orgs
    don't inflate it."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO benchmarks VALUES ('mmlu-pro-leaderboard', 'mmlu-pro', NULL)")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            # 3 distinct (benchmark, slice) pairs total.
            ("helm-classic",         "mmlu",     "anatomy",   "Anatomy",   "accuracy", "m1", "o1"),
            ("helm-classic",         "mmlu",     "anatomy",   "Anatomy",   "accuracy", "m2", "o1"),  # dup
            ("helm-classic",         "mmlu",     "astronomy", "Astronomy", "accuracy", "m1", "o1"),
            ("mmlu-pro-leaderboard", "mmlu-pro", "physics",   "Physics",   "accuracy", "m1", "o1"),
            # NULL slice_key shouldn't count.
            ("helm-classic",         "mmlu",     None,        None,        "accuracy", "m3", "o2"),
        ],
    )

    stats = _hierarchy_stats(con, [], [])
    assert stats["slice_count"] == 3


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
    con.execute("INSERT INTO benchmarks VALUES ('mmlu-pro-leaderboard', 'mmlu-pro', NULL)")
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

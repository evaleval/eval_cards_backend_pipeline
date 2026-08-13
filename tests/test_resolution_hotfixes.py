"""Scorer-wrapper benchmark hotfix tests.

Upstream inspect_ai/harbor conversions write "<metric> on <task> for
scorer <scorer>" into evaluation_name, minting one unresolvable benchmark
per record (task names embed the model or a run hash). The hotfix must
collapse exactly those rows onto the config's canonical benchmark — and
nothing else. Delete alongside the hotfix when upstream is fixed.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise import resolution_hotfixes


def _make_results_resolved(con, rows):
    con.execute(
        """
        CREATE TABLE results_resolved (
            source_config VARCHAR,
            benchmark_raw VARCHAR,
            benchmark_id VARCHAR,
            benchmark_resolution_strategy VARCHAR,
            metric_raw VARCHAR,
            metric_id VARCHAR,
            metric_resolution_strategy VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO results_resolved VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )


@pytest.fixture()
def con():
    c = duckdb.connect()
    yield c
    c.close()


def test_wrapper_rows_reassigned_per_config(con):
    _make_results_resolved(con, [
        ("l2-bench", "mean on full-solver-gpt-5.4 for scorer replay_scorer",
         None, "no_match", "mean", None, "no_match"),
        ("terminalbench", "accuracy on terminalbench/S-adaptive/+1ep/5dd93e71 for scorer harbor_scorer",
         None, "no_match", "accuracy", "accuracy", "exact"),
        ("swebenchpro", "accuracy on eval_task for scorer harbor_scorer",
         None, "no_match", "accuracy", "accuracy", "exact"),
    ])
    resolution_hotfixes.fix_scorer_wrapper_benchmarks(con)
    got = dict(con.execute(
        "SELECT source_config, benchmark_id FROM results_resolved"
    ).fetchall())
    assert got == {
        "l2-bench": "l2-bench",
        "terminalbench": "terminal-bench",
        "swebenchpro": "swe-bench-pro",
    }
    strategies = {r[0] for r in con.execute(
        "SELECT DISTINCT benchmark_resolution_strategy FROM results_resolved"
    ).fetchall()}
    assert strategies == {"hotfix_scorer_wrapper"}


def test_non_wrapper_rows_untouched(con):
    _make_results_resolved(con, [
        # Real benchmark name in a mapped config — upstream fixed; must pass through.
        ("l2-bench", "l2-bench", "l2-bench", "exact", "mean", None, "no_match"),
        # Wrapper shape in an unmapped config — out of scope by design.
        ("someothercfg", "accuracy on task for scorer x",
         None, "no_match", "accuracy", "accuracy", "exact"),
    ])
    resolution_hotfixes.fix_scorer_wrapper_benchmarks(con)
    rows = con.execute(
        "SELECT source_config, benchmark_id, benchmark_resolution_strategy "
        "FROM results_resolved ORDER BY source_config"
    ).fetchall()
    assert rows == [
        ("l2-bench", "l2-bench", "exact"),
        ("someothercfg", None, "no_match"),
    ]


def test_mean_namespaces_after_wrapper_fix(con):
    """Ordering contract: wrapper fix assigns the benchmark_id that the
    vague-metric fix keys on, so l2-bench's bare "mean" ends up namespaced."""
    _make_results_resolved(con, [
        ("l2-bench", "mean on full-solver-claude-haiku-4.5 for scorer replay_scorer",
         None, "no_match", "mean", None, "no_match"),
    ])
    resolution_hotfixes.fix_scorer_wrapper_benchmarks(con)
    resolution_hotfixes.fix_vague_metric_labels(con)
    bid, mid = con.execute(
        "SELECT benchmark_id, metric_id FROM results_resolved"
    ).fetchone()
    assert (bid, mid) == ("l2-bench", "l2-bench.mean")

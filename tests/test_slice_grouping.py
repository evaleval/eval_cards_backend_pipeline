"""Slice-grouping rules: alias-map only, no suffix heuristic.

With suffix stripping disabled, only the alias map and registry-authored
parent_benchmark_id values create slice relationships. Benchmarks that
share a name prefix (big-bench-hard, math-level-5) stay standalone.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise.slice_grouping import (
    apply_slice_grouping,
    compute_slice_stem,
    group_benchmarks,
    normalize_stem,
)


@pytest.mark.parametrize(
    "benchmark_id, expected",
    [
        # Alias map entries resolve to their stem
        ("hal_gaia",            "gaia"),
        ("hal_gaia_level_1",    "gaia"),
        ("helm_classic_foo",    "helm-classic"),
        ("helm_lite_bar",       "helm-lite"),
        ("mt_bench",            "mt-bench"),
        ("mtbench",             "mt-bench"),
        ("hfopenllm_v2_bbh",    "hf-open-llm-v2"),
        ("videomme-w-sub",      "videomme"),
        ("videomme-w-o-sub",    "videomme"),
        # Everything else returns its own normalised id (no suffix stripping)
        ("gaia",                "gaia"),
        ("gaia-level-1",        "gaia-level-1"),
        ("gpqa",                "gpqa"),
        ("gpqa-diamond",        "gpqa-diamond"),
        ("global-mmlu-lite",    "global-mmlu-lite"),
        ("big-bench-hard",      "big-bench-hard"),
        ("math-level-5",        "math-level-5"),
        ("livecodebench-pro",   "livecodebench-pro"),
        ("arena-hard",          "arena-hard"),
        ("rewardbench-2",       "rewardbench-2"),
        ("ace",                 "ace"),
        ("appworld",            "appworld"),
        ("caparena-vs-gpt-4o",  "caparena-vs-gpt-4o"),
        ("mmlu-cot",            "mmlu-cot"),
    ],
)
def test_compute_slice_stem(benchmark_id, expected):
    assert compute_slice_stem(benchmark_id) == expected


def test_normalize_stem_collapses_separators():
    assert normalize_stem("Hal Gaia") == "hal-gaia"
    assert normalize_stem("hal_gaia") == "hal-gaia"
    assert normalize_stem("hal-gaia") == "hal-gaia"
    assert normalize_stem("__foo--bar  baz_") == "foo-bar-baz"


def test_group_benchmarks_no_suffix_stripping():
    grouped = group_benchmarks(
        ["gaia", "gaia-level-1", "gpqa", "gpqa-diamond", "ace",
         "big-bench-hard", "big-bench"]
    )
    # Without suffix stripping, each is its own stem
    assert grouped["gaia"] == ["gaia"]
    assert grouped["gaia-level-1"] == ["gaia-level-1"]
    assert grouped["gpqa"] == ["gpqa"]
    assert grouped["gpqa-diamond"] == ["gpqa-diamond"]
    assert grouped["ace"] == ["ace"]
    assert grouped["big-bench-hard"] == ["big-bench-hard"]
    assert grouped["big-bench"] == ["big-bench"]


def test_group_benchmarks_alias_map_groups():
    grouped = group_benchmarks(
        ["videomme-w-sub", "videomme-w-o-sub", "ace"]
    )
    assert set(grouped["videomme"]) == {"videomme-w-sub", "videomme-w-o-sub"}
    assert grouped["ace"] == ["ace"]


def test_apply_alias_map_grouping():
    """Alias-mapped benchmarks get parent set; others stay NULL."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [
            ("videomme-w-sub",   None),
            ("videomme-w-o-sub", None),
            ("ace",              None),
            ("big-bench-hard",   None),
        ],
    )
    changed = apply_slice_grouping(con)
    assert changed == 2

    parents = dict(
        con.execute(
            "SELECT id, parent_benchmark_id FROM canonical_benchmarks ORDER BY id"
        ).fetchall()
    )
    assert parents["videomme-w-sub"]   == "videomme"
    assert parents["videomme-w-o-sub"] == "videomme"
    assert parents["ace"]              is None
    assert parents["big-bench-hard"]   is None


def test_apply_slice_grouping_idempotent():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [("videomme-w-sub", None), ("videomme-w-o-sub", None)],
    )
    apply_slice_grouping(con)
    second = apply_slice_grouping(con)
    assert second == 0


def test_apply_slice_grouping_preserves_registry_edges():
    """Registry-set parents stay; the alias map only fills NULLs."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [
            ("bfcl",              None),
            ("bfcl-live",         "bfcl"),
            ("bfcl-multi-turn",   "bfcl"),
        ],
    )
    changed = apply_slice_grouping(con)
    parents = dict(
        con.execute(
            "SELECT id, parent_benchmark_id FROM canonical_benchmarks"
        ).fetchall()
    )
    assert changed == 0
    assert parents == {
        "bfcl": None,
        "bfcl-live": "bfcl",
        "bfcl-multi-turn": "bfcl",
    }


def test_no_suffix_stripping_leaves_benchmarks_standalone():
    """Benchmarks that share a name prefix but aren't in the alias map
    stay standalone — no phantom parent created."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [
            ("big-bench",       None),
            ("big-bench-hard",  None),
            ("math",            None),
            ("math-level-5",    None),
            ("math-500",        None),
            ("arena-hard",      None),
            ("arena-hard-v2",   None),
            ("livecodebench",   None),
            ("livecodebench-pro", None),
            ("global-mmlu",     None),
            ("global-mmlu-lite", None),
        ],
    )
    changed = apply_slice_grouping(con)
    assert changed == 0

    parents = dict(
        con.execute(
            "SELECT id, parent_benchmark_id FROM canonical_benchmarks"
        ).fetchall()
    )
    assert all(p is None for p in parents.values())


def test_apply_slice_grouping_promotion_resets_existing_parent():
    """promote_to_benchmark resets a registry-authored parent to NULL."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [
            ("mmlu",     None),
            ("mmlu-pro", "mmlu"),
        ],
    )
    apply_slice_grouping(con, promote_to_benchmark={"mmlu-pro"})
    parents = dict(
        con.execute(
            "SELECT id, parent_benchmark_id FROM canonical_benchmarks"
        ).fetchall()
    )
    assert parents == {"mmlu": None, "mmlu-pro": None}

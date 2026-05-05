"""Slice-grouping rules + slice-promotion overrides.

Covers the spec §4.3 examples plus the benchmarks the registry hasn't
been hand-curated for (caparena, gaia, gpqa, videomme).
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
        # Suffix rules
        ("gaia",                "gaia"),
        ("gaia-level-1",        "gaia"),
        ("gaia-level-3",        "gaia"),
        ("mmlu-l5",             "mmlu"),
        ("apex-v1",             "apex"),
        ("helm-classic-v1.0",   "helm-classic"),
        ("gpqa-diamond",        "gpqa"),
        ("global-mmlu-lite",    "global-mmlu"),
        ("gpqa-diamond-v2",     "gpqa"),  # iterative
        ("caparena-vs-gpt-4o",  "caparena"),
        ("caparena-auto-avg",   "caparena"),
        ("caparena-caption-length", "caparena"),
        ("mmlu-cot",            "mmlu"),
        ("mmlu_5shot",          "mmlu"),
        # Alias map
        ("hal_gaia",            "gaia"),
        ("hal_gaia_level_1",    "gaia"),
        ("helm_classic_foo",    "helm-classic"),
        ("helm_lite_bar",       "helm-lite"),
        ("mt_bench",            "mt-bench"),
        ("mtbench",             "mt-bench"),
        ("hfopenllm_v2_bbh",    "hf-open-llm-v2"),
        # Variants registered via alias map
        ("videomme-w-sub",      "videomme"),
        ("videomme-w-o-sub",    "videomme"),
        # Things that should NOT collapse
        ("rewardbench-2",       "rewardbench-2"),
        ("ace",                 "ace"),
        ("appworld",            "appworld"),
    ],
)
def test_compute_slice_stem(benchmark_id, expected):
    assert compute_slice_stem(benchmark_id) == expected


def test_normalize_stem_collapses_separators():
    assert normalize_stem("Hal Gaia") == "hal-gaia"
    assert normalize_stem("hal_gaia") == "hal-gaia"
    assert normalize_stem("hal-gaia") == "hal-gaia"
    assert normalize_stem("__foo--bar  baz_") == "foo-bar-baz"


def test_group_benchmarks_buckets_by_stem():
    grouped = group_benchmarks(
        ["gaia", "gaia-level-1", "gpqa", "gpqa-diamond", "ace"]
    )
    assert grouped["gaia"] == ["gaia", "gaia-level-1"]
    assert grouped["gpqa"] == ["gpqa", "gpqa-diamond"]
    assert grouped["ace"] == ["ace"]


def test_apply_slice_grouping_sets_parent_for_siblings():
    """Mutates canonical_benchmarks in-place: caparena variants get
    parent=caparena (registry was missing this edge); ace stays NULL."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [
            ("gaia",                    None),
            ("gaia-level-1",            None),
            ("gaia-level-2",            None),
            ("gaia-level-3",            None),
            ("caparena-auto-avg",       None),
            ("caparena-vs-gpt-4o",      None),
            ("ace",                     None),
            # Singleton stem — should be left alone.
            ("global-mmlu-lite",        None),
        ],
    )

    changed = apply_slice_grouping(con)
    assert changed == 6

    parents = dict(
        con.execute(
            "SELECT id, parent_benchmark_id FROM canonical_benchmarks "
            "ORDER BY id"
        ).fetchall()
    )
    assert parents["gaia"]              == "gaia"   # self-parent
    assert parents["gaia-level-1"]      == "gaia"
    assert parents["gaia-level-2"]      == "gaia"
    assert parents["gaia-level-3"]      == "gaia"
    assert parents["caparena-auto-avg"] == "caparena"
    assert parents["caparena-vs-gpt-4o"] == "caparena"
    assert parents["ace"]               is None
    assert parents["global-mmlu-lite"]  is None


def test_apply_slice_grouping_idempotent():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [("gpqa", None), ("gpqa-diamond", None)],
    )
    apply_slice_grouping(con)
    second = apply_slice_grouping(con)
    assert second == 0


def test_apply_slice_grouping_preserves_registry_edges():
    """Registry-set parents stay; the heuristic only fills NULLs."""
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


def test_apply_slice_grouping_respects_conflicting_registry_parent():
    """rewardbench-chat-hard would strip via "-hard" → stem
    "rewardbench-chat", but the registry has it parented to
    "rewardbench". Registry wins.
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [
            ("rewardbench",            None),
            ("rewardbench-chat",       "rewardbench"),
            ("rewardbench-chat-hard",  "rewardbench"),
            ("rewardbench-reasoning",  "rewardbench"),
            ("rewardbench-safety",     "rewardbench"),
        ],
    )
    apply_slice_grouping(con)
    parents = dict(
        con.execute(
            "SELECT id, parent_benchmark_id FROM canonical_benchmarks"
        ).fetchall()
    )
    assert parents["rewardbench-chat-hard"] == "rewardbench"
    assert parents["rewardbench-chat"]      == "rewardbench"
    assert parents["rewardbench"]           is None


def test_apply_slice_grouping_self_parents_bare_stem():
    """When registry has all variants but bare stem dangling, fill in
    the bare stem as self-parent so it joins its variants in the
    composite (GAIA: 4 benchmarks including bare-stem).
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [
            ("gaia",          None),
            ("gaia-level-1",  None),
            ("gaia-level-2",  None),
        ],
    )
    apply_slice_grouping(con)
    parents = dict(
        con.execute(
            "SELECT id, parent_benchmark_id FROM canonical_benchmarks"
        ).fetchall()
    )
    assert parents == {
        "gaia":         "gaia",
        "gaia-level-1": "gaia",
        "gaia-level-2": "gaia",
    }


def test_apply_slice_grouping_promotion_keeps_parent_null():
    """`bfcl-live`, `bfcl-multi-turn`, etc. share a `bfcl` stem but are
    sibling benchmarks (each its own canonical, BFCL is the family slug
    not a benchmark). promote_to_benchmark={bfcl-live, bfcl-multi-turn,
    ...} keeps each row's parent_benchmark_id NULL.
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [
            ("bfcl-live",       None),
            ("bfcl-multi-turn", None),
            ("bfcl-non-live",   None),
            ("bfcl-web-search", None),
        ],
    )
    apply_slice_grouping(
        con,
        promote_to_benchmark={
            "bfcl-live", "bfcl-multi-turn", "bfcl-non-live", "bfcl-web-search",
        },
    )
    parents = dict(
        con.execute(
            "SELECT id, parent_benchmark_id FROM canonical_benchmarks"
        ).fetchall()
    )
    # All four stay parentless — they're benchmarks in the BFCL family,
    # not slices of a phantom `bfcl` stem.
    assert all(p is None for p in parents.values())


def test_apply_slice_grouping_promotion_resets_existing_parent():
    """If the registry (or a prior heuristic run) already wired
    `mmlu-pro` to `mmlu`, the promotion override resets it to NULL —
    MMLU-Pro is a sibling benchmark in the MMLU family, not a slice
    of MMLU.
    """
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


def test_apply_slice_grouping_promotes_gpqa_diamond_to_benchmark():
    """GPQA-Diamond shares the GPQA stem, but it is a benchmark sibling
    in the GPQA family rather than a slice of bare GPQA.
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE canonical_benchmarks "
        "(id VARCHAR, parent_benchmark_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO canonical_benchmarks VALUES (?, ?)",
        [
            ("gpqa", None),
            ("gpqa-diamond", None),
        ],
    )
    apply_slice_grouping(con, promote_to_benchmark={"gpqa-diamond"})
    parents = dict(
        con.execute(
            "SELECT id, parent_benchmark_id FROM canonical_benchmarks "
            "ORDER BY id"
        ).fetchall()
    )
    assert parents == {"gpqa": None, "gpqa-diamond": None}

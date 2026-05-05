"""Stage C — slice_key / slice_name derivation.

`_apply_slice_key` annotates `results_resolved` with a within-benchmark
subdivision: when ≥2 distinct cleaned `benchmark_raw` values map to one
`benchmark_id`, the raw IS the slice. Single-raw benchmarks get NULL —
no slice axis to differentiate. Case-only variants ("MMLU" / "mmlu")
collapse to one slice via the case-insensitive normalised key.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise.stages import _apply_slice_key


def _make_results_resolved(con, rows):
    """Build a minimal `results_resolved` carrying just the columns the
    slice helper reads. `rows` is a list of (benchmark_id, benchmark_raw)
    tuples; row order is preserved so assertions can index by position."""
    con.execute(
        "CREATE TABLE results_resolved ("
        "  rowid       INTEGER, "
        "  benchmark_id  VARCHAR, "
        "  benchmark_raw VARCHAR"
        ")"
    )
    for i, (bid, braw) in enumerate(rows):
        con.execute(
            "INSERT INTO results_resolved VALUES (?, ?, ?)",
            [i, bid, braw],
        )


def _slices(con):
    return con.execute(
        "SELECT rowid, slice_key, slice_name "
        "FROM results_resolved ORDER BY rowid"
    ).fetchall()


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def test_single_raw_benchmark_gets_null_slice(con):
    """One canonical with one cleaned raw = no slice axis."""
    _make_results_resolved(con, [
        ("mmlu",      "mmlu"),
        ("appworld",  "appworld"),
    ])
    _apply_slice_key(con)

    assert _slices(con) == [
        (0, None, None),
        (1, None, None),
    ]


def test_multi_raw_benchmark_populates_slice(con):
    """≥2 distinct cleaned raws on one canonical → slice_key per row.
    slice_key is the lowercase trimmed form; slice_name preserves the
    original raw casing for downstream display."""
    _make_results_resolved(con, [
        ("mmlu",  "Abstract Algebra"),
        ("mmlu",  "Anatomy"),
        ("mmlu",  "mmlu"),
    ])
    _apply_slice_key(con)

    assert _slices(con) == [
        (0, "abstract algebra", "Abstract Algebra"),
        (1, "anatomy",          "Anatomy"),
        (2, "mmlu",             "mmlu"),
    ]


def test_case_only_variants_collapse_to_one_slice(con):
    """'MMLU' and 'mmlu' are the same slice. The benchmark only earns a
    slice axis if the raws differ after case-insensitive normalisation."""
    _make_results_resolved(con, [
        ("mmlu", "MMLU"),
        ("mmlu", "mmlu"),
        ("mmlu", "Mmlu  "),  # also trailing whitespace
    ])
    _apply_slice_key(con)

    # Single distinct slice_key after LOWER(TRIM(...)) → no slice axis.
    assert _slices(con) == [
        (0, None, None),
        (1, None, None),
        (2, None, None),
    ]


def test_case_variants_alongside_real_slice_collapse_to_two(con):
    """Two distinct slices after normalisation: the 'mmlu' overall and
    the 'anatomy' subject. The two raw-casings of mmlu fold into one."""
    _make_results_resolved(con, [
        ("mmlu", "MMLU"),
        ("mmlu", "mmlu"),
        ("mmlu", "Anatomy"),
    ])
    _apply_slice_key(con)

    rows = _slices(con)
    # All three rows are in a multi-slice benchmark → all populated.
    assert rows[0] == (0, "mmlu",    "MMLU")
    assert rows[1] == (1, "mmlu",    "mmlu")
    assert rows[2] == (2, "anatomy", "Anatomy")


def test_unresolved_rows_get_null_slice(con):
    """benchmark_id IS NULL means the resolver no_match'd — slice_key is
    meaningless without a canonical to subdivide. Stays NULL even when
    the raw text would otherwise match a multi-slice canonical."""
    _make_results_resolved(con, [
        ("mmlu", "Anatomy"),
        ("mmlu", "Astronomy"),
        (None,   "Anatomy"),   # unresolved benchmark
    ])
    _apply_slice_key(con)

    rows = _slices(con)
    assert rows[0] == (0, "anatomy",   "Anatomy")
    assert rows[1] == (1, "astronomy", "Astronomy")
    # Unresolved row: slice_key/slice_name remain NULL.
    assert rows[2] == (2, None, None)


def test_null_benchmark_raw_gets_null_slice(con):
    """A NULL raw can't be a slice key; the row sits in a multi-slice
    benchmark but its own slice is unknown. NULL raws also don't
    contribute to the multi-slice count."""
    _make_results_resolved(con, [
        ("mmlu", "Anatomy"),
        ("mmlu", "Astronomy"),
        ("mmlu", None),
    ])
    _apply_slice_key(con)

    rows = _slices(con)
    assert rows[0] == (0, "anatomy",   "Anatomy")
    assert rows[1] == (1, "astronomy", "Astronomy")
    assert rows[2] == (2, None, None)


def test_mixed_single_and_multi_slice_benchmarks(con):
    """Per-benchmark decision: only the multi-raw ones earn a slice
    axis. Single-raw benchmarks in the same table stay NULL."""
    _make_results_resolved(con, [
        ("mmlu",     "Anatomy"),       # multi-slice
        ("mmlu",     "Astronomy"),     # multi-slice
        ("appworld", "appworld"),      # single-raw → NULL
        ("appworld", "appworld"),      # same raw, still single
    ])
    _apply_slice_key(con)

    rows = _slices(con)
    assert rows[0] == (0, "anatomy",   "Anatomy")
    assert rows[1] == (1, "astronomy", "Astronomy")
    assert rows[2] == (2, None, None)
    assert rows[3] == (3, None, None)

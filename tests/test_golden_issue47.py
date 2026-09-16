"""Golden artifacts for the issue #47 rename-at-resolution change.

These tests are NOT the acceptance gate and cannot be: the full corpus needs
the pinned EEE dataset and registry snapshot, which a unit run has no access
to, so nothing here re-derives the goldens from current code. They guard the
artifacts only — readable, internally consistent, and still describing rules
the registry fixture carries — so a silent edit or a drifted rename rule
shows up as a failure rather than as a reinterpreted golden.

The executed full-corpus gate is `notes/issue47/GOLDEN-runG-H.md`: runs G and
H over the pinned corpus, byte-identical across every parquet and sidecar,
with the corpus invariants and the flag counts recorded there.

Both files are pinned to one corpus (header comment: EEE revision, registry
cache, baseline code revision, snapshot id).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

GOLDEN = Path(__file__).parent / "golden"
FOLD_DIFF = GOLDEN / "issue47-metric-fold-diff.csv"
DEFAULT_CHANGES = GOLDEN / "issue47-default-metric-changes.csv"
FIXTURE_FOLDS = (
    Path(__file__).parent / "fixtures" / "entity_registry"
    / "benchmark_metric_folds.parquet"
)

PINNED = (
    "# pinned: EEE abe4b262c054b8348f6d1253bfb2cd34becccacb; "
    "registry cache 2fb0eb90ae94064780ae065622588dc7c3057848; "
    "baseline code 1398b4c; snapshot 2026-09-13T00-00-00Z"
)


def _read(path: Path) -> tuple[str, list[dict]]:
    """Split the pinned-revision comment line off the CSV body."""
    lines = path.read_text().splitlines()
    assert lines[0].startswith("#"), f"{path.name} has no pinned-revision line"
    return lines[0], list(csv.DictReader(lines[1:]))


def test_metric_fold_golden_totals():
    header, rows = _read(FOLD_DIFF)
    assert header == PINNED
    assert len(rows) == 36
    assert sum(int(r["facts"]) for r in rows) == 93_091
    assert all(int(r["models"]) > 0 for r in rows)
    # one row per benchmark, sorted, and every row is an actual rename
    assert [r["benchmark_id"] for r in rows] == sorted(
        r["benchmark_id"] for r in rows
    )
    assert len({r["benchmark_id"] for r in rows}) == 36
    assert all(r["metric_key_before"] != r["metric_key_after"] for r in rows)


def test_every_renamed_metric_is_a_registry_fold_target():
    """The `after` identity of every golden row is a metric the registry's
    fold table actually names as a rename target. A rule dropped or retargeted
    in the registry makes the golden stop describing the corpus."""
    _, rows = _read(FOLD_DIFF)
    folds = pd.read_parquet(FIXTURE_FOLDS)
    targets = set(folds["to_metric_id"])
    unknown = sorted({r["metric_key_after"] for r in rows} - targets)
    assert unknown == []
    # the benchmarks the fixture registry itself carries must match row for row
    by_benchmark = {
        (r.benchmark_id, r.from_metric_id): r.to_metric_id
        for r in folds.itertuples()
    }
    for row in rows:
        key = (row["benchmark_id"], row["metric_key_before"])
        if key in by_benchmark:
            assert by_benchmark[key] == row["metric_key_after"], key


def test_default_metric_change_golden():
    header, rows = _read(DEFAULT_CHANGES)
    assert header == PINNED
    assert len(rows) == 123
    assert {r["change_class"] for r in rows} <= {
        "label_rename", "registry_preferred", "page_rekeyed",
    }
    # a re-keyed page has no `after`: the key stopped existing. Every other
    # class is a real before → after move on a page that still exists.
    for row in rows:
        if row["change_class"] == "page_rekeyed":
            assert row["after"] == ""
        else:
            assert row["after"] and row["before"] != row["after"]
    assert [(r["composite_slug"], r["benchmark_id"]) for r in rows] == sorted(
        (r["composite_slug"], r["benchmark_id"]) for r in rows
    )
    # no page is listed twice
    assert len({(r["composite_slug"], r["benchmark_id"]) for r in rows}) == 123


def test_label_renames_agree_with_the_fold_golden():
    """`label_rename` means the page default moved because its metric was
    renamed — the same (benchmark, before → after) the fold golden records."""
    _, fold_rows = _read(FOLD_DIFF)
    renames = {
        (r["benchmark_id"], r["metric_key_before"], r["metric_key_after"])
        for r in fold_rows
    }
    _, rows = _read(DEFAULT_CHANGES)
    for row in rows:
        if row["change_class"] == "label_rename":
            assert (row["benchmark_id"], row["before"], row["after"]) in renames

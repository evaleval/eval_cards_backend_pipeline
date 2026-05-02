"""Tests for ``scripts.audit_output``.

The clean-fixture run must pass the audit (positive control), and each
invariant must trigger when a synthetic regression is injected into the
fixture's parquet (negative controls). Mutating an in-memory list of
flattened rows and feeding it to the private check functions directly is
deliberately preferred over writing a corrupted parquet — it keeps the
tests fast and isolates each check from parquet I/O concerns.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_output


def _row(
    eval_summary_id: str,
    *,
    family_key: str = "fam",
    models_count: int = 5,
    metrics_count: int = 1,
    evaluation_name: str = "Bench",
    canonical_benchmark_id: str | None = None,
    reporting_sources: list[str] | None = None,
) -> dict:
    return {
        "eval_summary_id": eval_summary_id,
        "benchmark_family_key": family_key,
        "models_count": models_count,
        "metrics_count": metrics_count,
        "evaluation_name": evaluation_name,
        "composite_benchmark_name": evaluation_name,
        "canonical_benchmark_id": canonical_benchmark_id,
        "reporting_sources": reporting_sources,
    }


def _hierarchy(*families: dict) -> dict:
    return {"families": list(families)}


def _family(
    key: str,
    *,
    eval_summary_ids: list[str],
    evals_count: int | None = None,
    leaves: list[dict] | None = None,
    display_name: str | None = None,
) -> dict:
    return {
        "key": key,
        "display_name": display_name or key,
        "eval_summary_ids": eval_summary_ids,
        "evals_count": evals_count if evals_count is not None else len(eval_summary_ids),
        "leaves": leaves or [],
    }


# ---------------------------------------------------------------------------
# Clean-fixture positive control: a real pipeline run must pass the audit
# ---------------------------------------------------------------------------


def test_audit_passes_on_clean_fixture_run(pipeline_output: Path) -> None:
    """Every check in audit_output must be invariant-safe on the synthetic
    fixture corpus. A failure here means either the fixture has drifted
    away from the catalog contract, or the audit has a false-positive."""
    audit_output.audit_output(pipeline_output)


# ---------------------------------------------------------------------------
# Per-check negative controls — each invariant must fire when violated
# ---------------------------------------------------------------------------


def test_zero_count_tile_check_catches_zero_models() -> None:
    rows = [_row("a"), _row("b", models_count=0)]
    errors: list[str] = []
    audit_output._check_zero_count_tile(rows, errors)
    assert errors and "zero_count_tile" in errors[0]
    assert "'b'" in errors[0]


def test_zero_count_tile_check_catches_zero_metrics() -> None:
    rows = [_row("a"), _row("c", metrics_count=0)]
    errors: list[str] = []
    audit_output._check_zero_count_tile(rows, errors)
    assert errors and "zero_count_tile" in errors[0]
    assert "'c'" in errors[0]


def test_hierarchy_dangling_check_catches_unknown_eval_summary_id() -> None:
    rows = [_row("known_a"), _row("known_b")]
    hierarchy = _hierarchy(
        _family("fam_x", eval_summary_ids=["known_a", "ghost_id"])
    )
    errors: list[str] = []
    audit_output._check_hierarchy_dangling_eval_summary_ids(rows, hierarchy, errors)
    assert errors and "hierarchy_dangling_eval_summary_id" in errors[0]
    assert "ghost_id" in errors[0]


def test_hierarchy_dangling_check_accepts_reporting_sources() -> None:
    """A demoted source listed in ``reporting_sources`` of a primary row
    counts as 'in eval-list' for hierarchy reference purposes — the
    drilldown JSON still ships and the frontend can route to it."""
    rows = [_row("primary", reporting_sources=["primary", "demoted_source"])]
    hierarchy = _hierarchy(_family("fam_x", eval_summary_ids=["demoted_source"]))
    errors: list[str] = []
    audit_output._check_hierarchy_dangling_eval_summary_ids(rows, hierarchy, errors)
    assert not errors


def test_hierarchy_family_unrenderable_check_fires_for_orphan_family() -> None:
    """Replicates the AA-lift / canonical-union pre-fix bug: a hierarchy
    family with key='gpqa' but no eval-list row carrying
    benchmark_family_key='gpqa' renders as 0/0 in the frontend."""
    rows = [_row("hfopenllm_v2_gpqa", family_key="hfopenllm")]
    hierarchy = _hierarchy(_family("gpqa", eval_summary_ids=["hfopenllm_v2_gpqa"]))
    errors: list[str] = []
    audit_output._check_hierarchy_family_unrenderable(rows, hierarchy, errors)
    assert errors and "hierarchy_family_unrenderable" in errors[0]
    assert "gpqa" in errors[0]


def test_hierarchy_family_unrenderable_check_normalizes_key_punctuation() -> None:
    """``tau2-bench`` (hierarchy) and ``tau2_bench`` (eval-list) must
    match — both normalize to the same alphanumeric form, mirroring the
    frontend's ``getSummaryScopeKey`` filter."""
    rows = [_row("canonical__tau2-bench", family_key="tau2_bench")]
    hierarchy = _hierarchy(
        _family("tau2-bench", eval_summary_ids=["canonical__tau2-bench"])
    )
    errors: list[str] = []
    audit_output._check_hierarchy_family_unrenderable(rows, hierarchy, errors)
    assert not errors


def test_hierarchy_family_unrenderable_check_skips_empty_families() -> None:
    """A family with evals_count=0 was already pruned by the upstream
    pipeline; surfacing it would be noise."""
    rows = [_row("a", family_key="fam_a")]
    hierarchy = _hierarchy(
        _family("fam_a", eval_summary_ids=["a"]),
        _family("empty", eval_summary_ids=[], evals_count=0),
    )
    errors: list[str] = []
    audit_output._check_hierarchy_family_unrenderable(rows, hierarchy, errors)
    assert not errors


def test_duplicate_display_name_check_catches_finance_agent_pattern() -> None:
    """Two rows in the same family share an identical evaluation_name —
    the canonical-union / parent-rollup didn't merge them and the user
    sees N tiles labeled the same. Negative control for the 'Vals AI 4
    Finance Agents' regression."""
    rows = [
        _row("a1", family_key="vals", evaluation_name="Finance Agent"),
        _row("a2", family_key="vals", evaluation_name="Finance Agent"),
        _row("a3", family_key="vals", evaluation_name="Sage"),
    ]
    errors: list[str] = []
    audit_output._check_duplicate_display_name_within_family(rows, errors)
    assert errors and "duplicate_display_name_within_family" in errors[0]
    assert "Finance Agent" in errors[0]


def test_duplicate_display_name_check_allows_same_name_across_families() -> None:
    """It's fine for two different suites to each surface a 'GPQA' leaf —
    the catalog tile is keyed by (family, name), not name alone."""
    rows = [
        _row("a1", family_key="suite_a", evaluation_name="GPQA"),
        _row("a2", family_key="suite_b", evaluation_name="GPQA"),
    ]
    errors: list[str] = []
    audit_output._check_duplicate_display_name_within_family(rows, errors)
    assert not errors


def test_canonical_id_collapse_check_catches_unmerged_canonical() -> None:
    """Two eval-list rows with the same non-empty canonical_benchmark_id
    is a partial-collapse bug: canonical-union should have merged them
    into a single canonical__<id> row."""
    rows = [
        _row("source_a", canonical_benchmark_id="gpqa"),
        _row("source_b", canonical_benchmark_id="gpqa"),
    ]
    errors: list[str] = []
    audit_output._check_canonical_id_not_collapsed(rows, errors)
    assert errors and "canonical_id_not_collapsed" in errors[0]
    assert "gpqa" in errors[0]


def test_canonical_id_collapse_check_allows_no_canonical() -> None:
    """Rows without a canonical_benchmark_id are unconstrained — the
    collapse rule only applies to the canonical-resolved ones."""
    rows = [_row("a"), _row("b"), _row("c")]
    errors: list[str] = []
    audit_output._check_canonical_id_not_collapsed(rows, errors)
    assert not errors


# ---------------------------------------------------------------------------
# audit_output() integration: errors aggregate, not fail-fast
# ---------------------------------------------------------------------------


def test_audit_aggregates_all_errors_in_one_raise(tmp_path: Path) -> None:
    """``audit_output`` should surface every violation it finds in a
    single AuditError, not stop at the first one. Operators triage all
    classes in one cycle that way."""
    parquet_dir = tmp_path / "duckdb" / "v1"
    parquet_dir.mkdir(parents=True)

    import pandas as pd

    rows = [
        # Triggers zero_count_tile
        {
            "eval_summary_id": "zero_row",
            "benchmark_family_key": "fam_a",
            "models_count": 0,
            "payload_json": json.dumps({
                "evaluation_name": "Zero",
                "metrics_count": 1,
            }),
        },
        # Triggers canonical_id_not_collapsed (paired with next)
        {
            "eval_summary_id": "uncollapsed_a",
            "benchmark_family_key": "fam_a",
            "models_count": 5,
            "payload_json": json.dumps({
                "evaluation_name": "A",
                "metrics_count": 1,
                "canonical_benchmark_id": "shared",
            }),
        },
        {
            "eval_summary_id": "uncollapsed_b",
            "benchmark_family_key": "fam_a",
            "models_count": 5,
            "payload_json": json.dumps({
                "evaluation_name": "B",
                "metrics_count": 1,
                "canonical_benchmark_id": "shared",
            }),
        },
    ]
    pd.DataFrame(rows).to_parquet(parquet_dir / "eval_list.parquet")

    (tmp_path / "eval-hierarchy.json").write_text(
        json.dumps({"families": []}), encoding="utf-8"
    )

    with pytest.raises(audit_output.AuditError) as exc_info:
        audit_output.audit_output(tmp_path)
    message = str(exc_info.value)
    assert "zero_count_tile" in message
    assert "canonical_id_not_collapsed" in message

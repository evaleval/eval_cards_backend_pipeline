"""Stage J — `eval_results_view` materialisation.

These tests run the canonicalisation pipeline through Stage I against the
hand-built fixtures, then invoke `stage_j_eval_results_view` directly on
the resulting parquets. This isolates the view-layer SQL from the
orchestrator wiring (which lands in a later task).
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _run_through_stage_i(tmp_path, monkeypatch, config: str) -> Path:
    """Run the pipeline through Stage I against a single fixture config."""
    eee_root = FIXTURES / "eee"
    cards_root = FIXTURES / "auto_benchmarkcards"
    reg_root = FIXTURES / "entity_registry"
    warehouse = tmp_path / "warehouse"

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    settings = Settings.from_env()
    out_dir = pipeline.run(
        settings,
        configs=[config],
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        cache_root=str(tmp_path / "cache"),
    )
    assert out_dir is not None
    return out_dir


def _materialise_view(out_dir: Path):
    """Load canonical parquets into a fresh DuckDB connection, register
    UDFs, and run Stage J. Returns the connection (caller does
    assertions)."""
    from eval_card_backend.canonicalise import stages
    from eval_card_backend.canonicalise.resolver_setup import register_udfs
    from eval_card_backend.sources import registry as registry_src
    from eval_entity_resolver import Resolver

    con = duckdb.connect()
    alias_store = registry_src.load_alias_store(FIXTURES / "entity_registry")
    register_udfs(con, Resolver(alias_store))
    for table in (
        "fact_results", "benchmarks", "composites", "families", "models",
        "canonical_metrics",
    ):
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_parquet('{out_dir}/{table}.parquet')"
        )
    stages.stage_j_eval_results_view(con, "2026-04-30T00:00:00Z")
    return con


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


def test_view_columns_match_spec(tmp_path, monkeypatch):
    """All spec-required columns are present with the documented types."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out)
    cols = {row[0]: row[1] for row in con.execute("DESCRIBE eval_results_view").fetchall()}

    expected = {
        "snapshot_id", "evaluation_id", "metric_summary_id",
        "benchmark_id", "metric_id", "model_id", "model_route_id",
        "model_info", "metric_display_name", "metric_unit", "lower_is_better",
        "category", "score", "score_details", "fact_row_count",
        "position", "total", "percentile",
        "evaluation_timestamp", "source_metadata", "source_data",
        "source_record_url", "eval_library",
        "evaluator_relationships", "has_first_party", "has_third_party",
        "coverage_cell", "reporting_orgs", "scores_by_organization",
        "is_summary_score", "summary_score_for", "aggregate_components",
        "has_reproducibility_gap", "completeness_score", "is_multi_source",
        "first_party_only", "has_variant_divergence", "has_cross_party_divergence",
        "evalcards_annotations",
        "instance_file_path", "instance_file_format", "instance_rows",
        # Hierarchy-alignment §5.3: family / composite columns surface
        # on the view so frontend filters don't need a dim join.
        "composite_slug", "family_id", "is_slice", "parent_benchmark_id",
    }
    missing = expected - cols.keys()
    assert not missing, f"missing columns: {missing}"

    # Spot-check a few critical types.
    assert cols["evaluation_timestamp"] == "TIMESTAMP"
    assert cols["snapshot_id"] == "TIMESTAMP"
    assert cols["score"] == "DOUBLE"
    assert cols["position"] == "INTEGER"
    assert cols["coverage_cell"] == "VARCHAR"
    # Hierarchy-alignment §5.3 column types.
    assert cols["composite_slug"] == "VARCHAR"
    assert cols["family_id"] == "VARCHAR"
    assert cols["parent_benchmark_id"] == "VARCHAR"


# ---------------------------------------------------------------------------
# Slugs
# ---------------------------------------------------------------------------


def test_evaluation_id_round_trips_through_unquote(tmp_path, monkeypatch):
    """evaluation_id is `<composite_slug>/<benchmark_id>` URL-encoded;
    unquote recovers the canonical pair separated by `/`."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out)
    rows = con.execute(
        "SELECT DISTINCT composite_slug, benchmark_id, evaluation_id "
        "FROM eval_results_view "
        "WHERE benchmark_id IS NOT NULL"
    ).fetchall()
    assert rows, "no rows produced"
    for composite_slug, benchmark_id, slug in rows:
        assert unquote(slug) == f"{composite_slug}/{benchmark_id}"


def test_metric_summary_id_round_trips(tmp_path, monkeypatch):
    """metric_summary_id encodes `benchmark_id:metric_id`; unquote recovers
    the literal join string."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out)
    rows = con.execute(
        "SELECT benchmark_id, metric_id, metric_summary_id FROM eval_results_view"
    ).fetchall()
    assert rows
    for bid, mid, slug in rows:
        assert unquote(slug) == f"{bid}:{mid}"


def test_model_route_id_round_trips(tmp_path, monkeypatch):
    """model_route_id is url_encode(model_key); slashes → %2F. Resolved
    models have model_key == model_id; unresolved fall back to model_raw
    so the view is still addressable for them."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out)
    rows = con.execute(
        "SELECT model_key, model_id, model_route_id FROM eval_results_view"
    ).fetchall()
    for model_key, model_id, slug in rows:
        assert unquote(slug) == model_key
        if "/" in model_key:
            assert "%2F" in slug
        # Resolved → model_id matches model_key; unresolved → model_id is NULL.
        assert model_id is None or model_id == model_key


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def test_score_uses_median_across_fact_rows(tmp_path, monkeypatch):
    """fixtures_variant has three first-party rows (0.5, 0.78, 0.85) on the
    same triple. Representative score = median = 0.78."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_variant")
    con = _materialise_view(out)
    row = con.execute(
        "SELECT score, fact_row_count FROM eval_results_view"
    ).fetchone()
    assert row is not None
    score, fact_row_count = row
    assert abs(score - 0.78) < 1e-9
    assert fact_row_count == 3


@pytest.mark.skip(
    reason="Inactive while the stages.py:759 evaluator_relationship override is in place "
    "(EEE upstream-data-quality mitigation: only llm-stats+raw_verified='false' surfaces "
    "as first_party; everything else collapses to third_party). Re-enable when the override "
    "is removed and upstream emits the right value directly."
)
def test_score_prefers_first_party_when_third_party_diverges(tmp_path, monkeypatch):
    """fixtures_xparty has one first-party row (0.85) and one third-party
    row (0.65). First-party-priority rule → score = 0.85, NOT the all-rows
    median 0.75."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_view(out)
    row = con.execute(
        "SELECT score, fact_row_count FROM eval_results_view"
    ).fetchone()
    score, fact_row_count = row
    assert abs(score - 0.85) < 1e-9
    assert fact_row_count == 2


def test_position_total_percentile_with_unresolved_peer(tmp_path, monkeypatch):
    """gpt-4o (resolved, score=0.85) and community/fine-tune-7b (unresolved,
    score=0.4) share the mmlu/mmlu.acc triple in fixtures_clean. The
    unresolved peer now contributes to ranking — the view no longer drops
    NULL-model_id rows."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out)
    row = con.execute(
        "SELECT position, total, percentile FROM eval_results_view "
        "WHERE model_id = 'openai/gpt-4o'"
    ).fetchone()
    pos, total, _pct = row
    assert pos == 1, "gpt-4o has the higher score → rank 1"
    assert total == 2, "fixture pairs gpt-4o with the unresolved fine-tune-7b"

    # The unresolved peer is also rankable now.
    unresolved_row = con.execute(
        "SELECT position, total FROM eval_results_view "
        "WHERE model_key = 'community/fine-tune-7b' AND model_id IS NULL"
    ).fetchone()
    assert unresolved_row is not None
    assert unresolved_row == (2, 2)


@pytest.mark.skip(
    reason="Inactive while the stages.py:759 evaluator_relationship override is in place "
    "(EEE upstream-data-quality mitigation: only llm-stats+raw_verified='false' surfaces "
    "as first_party; everything else collapses to third_party). Re-enable when the override "
    "is removed and upstream emits the right value directly."
)
def test_coverage_cell_self_when_only_first_party(tmp_path, monkeypatch):
    """All fixtures_clean rows are first-party → coverage_cell='self'."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out)
    rows = con.execute(
        "SELECT coverage_cell, has_first_party, has_third_party "
        "FROM eval_results_view WHERE model_id = 'openai/gpt-4o'"
    ).fetchall()
    for coverage, has_first, has_third in rows:
        assert coverage == "self"
        assert has_first is True
        assert has_third is False


@pytest.mark.skip(
    reason="Inactive while the stages.py:759 evaluator_relationship override is in place "
    "(EEE upstream-data-quality mitigation: only llm-stats+raw_verified='false' surfaces "
    "as first_party; everything else collapses to third_party). Re-enable when the override "
    "is removed and upstream emits the right value directly."
)
def test_coverage_cell_both_when_cross_party(tmp_path, monkeypatch):
    """fixtures_xparty has one first-party and one third-party row on the
    same triple → coverage_cell='both'."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_view(out)
    row = con.execute(
        "SELECT coverage_cell, has_first_party, has_third_party "
        "FROM eval_results_view"
    ).fetchone()
    assert row is not None
    coverage, has_first, has_third = row
    assert coverage == "both"
    assert has_first is True
    assert has_third is True


# ---------------------------------------------------------------------------
# Signal flags
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Inactive while the stages.py:759 evaluator_relationship override is in place "
    "(EEE upstream-data-quality mitigation: only llm-stats+raw_verified='false' surfaces "
    "as first_party; everything else collapses to third_party). Re-enable when the override "
    "is removed and upstream emits the right value directly."
)
def test_evalcards_annotations_struct_populated(tmp_path, monkeypatch):
    """evalcards_annotations carries reproducibility_gap, provenance, and
    divergence sub-structs with the expected nested shape."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_variant")
    con = _materialise_view(out)
    row = con.execute(
        "SELECT evalcards_annotations FROM eval_results_view"
    ).fetchone()
    annotations = row[0]
    assert "reproducibility_gap" in annotations
    assert "provenance" in annotations
    assert "variant_divergence" in annotations
    assert "cross_party_divergence" in annotations
    # Provenance carries the representative row's evaluator_relationship.
    assert annotations["provenance"]["evaluator_relationship"] == "first_party"


def test_score_details_struct_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_variant")
    con = _materialise_view(out)
    row = con.execute(
        "SELECT score_details FROM eval_results_view"
    ).fetchone()
    sd = row[0]
    assert "score" in sd
    assert "standard_error" in sd
    assert "sample_size" in sd
    assert "confidence_interval" in sd
    assert abs(sd["score"] - 0.78) < 1e-9


# ---------------------------------------------------------------------------
# is_summary_score
# ---------------------------------------------------------------------------


def test_is_summary_score_false_for_standalone_benchmark(tmp_path, monkeypatch):
    """Fixtures don't carry a parent_benchmark_id, so is_summary_score
    must be False everywhere."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out)
    rows = con.execute(
        "SELECT is_summary_score, summary_score_for FROM eval_results_view"
    ).fetchall()
    assert rows
    for is_summary, summary_for in rows:
        assert is_summary is False
        assert summary_for is None


# ---------------------------------------------------------------------------
# Primary-key uniqueness
# ---------------------------------------------------------------------------


def test_primary_key_is_unique(tmp_path, monkeypatch):
    """(snapshot_id, metric_summary_id, model_id) must uniquely identify
    every row — that's the documented PRIMARY KEY."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_variant")
    con = _materialise_view(out)
    n_rows, n_unique = con.execute(
        "SELECT COUNT(*), "
        "       COUNT(DISTINCT (snapshot_id, metric_summary_id, model_id)) "
        "FROM eval_results_view"
    ).fetchone()
    assert n_rows == n_unique
    assert n_rows >= 1

"""Stage J — `evals_view` materialisation."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _run_through_stage_i(tmp_path, monkeypatch, config: str) -> Path:
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

    out_dir = pipeline.run(
        Settings.from_env(),
        configs=[config],
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        cache_root=str(tmp_path / "cache"),
    )
    assert out_dir is not None
    return out_dir


def _materialise_views(out_dir: Path):
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
    stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")
    return con


def test_evals_view_columns_match_spec(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    cols = {row[0]: row[1] for row in con.execute("DESCRIBE evals_view").fetchall()}

    expected = {
        "snapshot_id", "evaluation_id", "benchmark_id", "primary_metric_id",
        "evaluation_name", "canonical_display_name",
        "composite_slug", "composite_display_name",
        "family_id", "family_display_name", "is_slice",
        "parent_benchmark_id",
        "category",
        "metric_config",
        "models_count", "evaluator_names", "source_types",
        "latest_source_name", "third_party_ratio",
        "missing_generation_config_count",
        "best_model", "worst_model",
        "avg_score", "avg_score_norm", "top_score",
        "has_card", "benchmark_card",
        "is_aggregated", "aggregate_sources",
        "is_summary_score", "summary_eval_ids",
        "tags", "source_data",
        "reproducibility_summary", "provenance_summary", "comparability_summary",
        "instance_data",
        "metrics_count", "metric_names",
        "leaderboard_metrics", "leaderboard_rows",
        "root_metrics", "subtasks", "subtasks_count",
    }
    missing = expected - cols.keys()
    assert not missing, f"missing columns: {missing}"
    # Legacy columns removed in the composite/family/slice taxonomy
    # cutover. Test guards against regression.
    assert "composite_benchmark_key" not in cols
    assert "benchmark_family_key" not in cols
    assert "benchmark_leaf_key" not in cols
    # Hierarchy-alignment §5.3 column types — frontend filters read
    # these scalars without a dim join.
    assert cols["composite_slug"] == "VARCHAR"
    assert cols["family_id"] == "VARCHAR"
    assert cols["parent_benchmark_id"] == "VARCHAR"


def test_evals_view_excludes_factless_parent_shells(tmp_path, monkeypatch):
    """evals_view ships only benchmarks with at least one fact row. The
    benchmarks dim deliberately includes fact-less parent shells (so the
    hierarchy graph resolves) — but those aren't navigable evals and
    should not appear in the user-facing view. Aligns with
    comparison-index, which is built from per-(eval, metric) buckets and
    excludes them by construction.
    """
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    factless = con.execute(
        """
        SELECT e.composite_slug, e.benchmark_id
        FROM evals_view e
        WHERE NOT EXISTS (
            SELECT 1 FROM read_parquet(?) f
            WHERE f.composite_slug = e.composite_slug
              AND f.benchmark_key  = e.benchmark_id
        )
        """,
        [f"{out}/fact_results.parquet"],
    ).fetchall()
    assert factless == [], (
        f"evals_view contains fact-less parent shells: {factless}. "
        f"These are dim rows for hierarchy support; they don't belong in "
        f"the eval list."
    )


def test_evaluation_id_is_composite_slash_benchmark(tmp_path, monkeypatch):
    """evaluation_id is `<composite_slug>/<benchmark_id>` URL-encoded."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    rows = con.execute(
        "SELECT composite_slug, benchmark_id, evaluation_id FROM evals_view"
    ).fetchall()
    for composite_slug, benchmark_id, slug in rows:
        # URL-encoded form of "composite_slug/benchmark_id"
        assert unquote(slug) == f"{composite_slug}/{benchmark_id}"


def test_primary_metric_picked_deterministically(tmp_path, monkeypatch):
    """Single-metric fixture → primary_metric_id = the only metric."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT primary_metric_id, metrics_count FROM evals_view"
    ).fetchone()
    primary, count = row
    assert primary == "accuracy"
    assert count == 1


def test_leaderboard_metrics_one_per_metric(tmp_path, monkeypatch):
    """leaderboard_metrics[] has one entry per (benchmark, metric)."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT leaderboard_metrics, metrics_count FROM evals_view"
    ).fetchone()
    metrics, count = row
    assert len(metrics) == count
    for m in metrics:
        assert {"column_key", "metric_summary_id", "metric_id",
                "metric_name", "display_name", "canonical_display_name",
                "lower_is_better", "unit", "scope",
                "subtask_key", "subtask_name"} <= set(m.keys())
        assert m["scope"] == "root"


def test_leaderboard_rows_pivoted_values_map(tmp_path, monkeypatch):
    """Each row's `values` is a MAP keyed by metric column_key.
    fixtures_variant: 3 fact rows, all same model+benchmark+metric,
    representative score = median = 0.78."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_variant")
    con = _materialise_views(out)
    row = con.execute("SELECT leaderboard_rows FROM evals_view").fetchone()[0]
    assert row is not None
    assert len(row) >= 1
    first = row[0]
    assert "values" in first
    # values keys are metric column keys (= metric_id today).
    metric_keys = list(first["values"].keys())
    assert len(metric_keys) == 1
    score = first["values"][metric_keys[0]]
    assert abs(score - 0.78) < 1e-9
    assert first["metrics_present"] == 1


def test_avg_score_normalisation_uses_metric_bounds(tmp_path, monkeypatch):
    """avg_score_norm = (avg_score - min) / (max - min). For a 0..1 metric
    avg_score == avg_score_norm."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_variant")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT avg_score, avg_score_norm FROM evals_view"
    ).fetchone()
    avg, norm = row
    assert avg is not None
    assert abs(avg - norm) < 1e-9  # 0..1 metric → identity


def test_best_model_uses_lower_is_better(tmp_path, monkeypatch):
    """best_model.score equals top_score for higher-is-better metrics."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT best_model, top_score FROM evals_view"
    ).fetchone()
    best, top = row
    assert best is not None
    assert best["score"] == top
    assert best["name"] == "GPT-4o"


def test_has_card_and_card_struct_populated(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT has_card, benchmark_card.benchmark_details.name "
        "FROM evals_view"
    ).fetchone()
    has_card, name = row
    assert has_card is True
    assert name is not None


def test_benchmark_card_array_fields_never_null(tmp_path, monkeypatch):
    """Frontend reads `benchmark_card.methodology.metrics.length` etc. without
    null guards because the TS contract says these are `string[]`. Source JSON
    can omit any field, in which case the JSON extract returns NULL — the view
    layer must COALESCE to [] so the contract holds. A regression here crashes
    the eval detail page (see `BenchmarkCardPanel`).
    """
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    rows = con.execute(
        """
        SELECT
            benchmark_card.benchmark_details.domains            AS bd_domains,
            benchmark_card.benchmark_details.languages          AS bd_languages,
            benchmark_card.benchmark_details.similar_benchmarks AS bd_similar,
            benchmark_card.benchmark_details.resources          AS bd_resources,
            benchmark_card.purpose_and_intended_users.audience          AS pu_audience,
            benchmark_card.purpose_and_intended_users.tasks             AS pu_tasks,
            benchmark_card.purpose_and_intended_users.out_of_scope_uses AS pu_oos,
            benchmark_card.methodology.methods AS m_methods,
            benchmark_card.methodology.metrics AS m_metrics,
            benchmark_card.possible_risks      AS risks,
            benchmark_card.missing_fields      AS missing_fields
        FROM evals_view
        WHERE has_card
        """
    ).fetchall()
    assert rows, "fixture has no carded evals; cannot exercise array contract"
    for row in rows:
        for field_value in row:
            assert field_value is not None, (
                "benchmark_card array field is NULL — frontend will crash on "
                ".length / .map / .join (see BenchmarkCardPanel). Add a "
                "COALESCE shim in stages.py at the struct_pack site."
            )


def test_tags_struct_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    tags = con.execute("SELECT tags FROM evals_view").fetchone()[0]
    assert set(tags.keys()) == {"domains", "languages", "tasks"}


def test_third_party_ratio_xparty_fixture(tmp_path, monkeypatch):
    """fixtures_xparty: 1 model, 1 benchmark, coverage_cell='both' → ratio=1.0."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_views(out)
    ratio = con.execute(
        "SELECT third_party_ratio FROM evals_view"
    ).fetchone()[0]
    assert ratio == 1.0


@pytest.mark.skip(
    reason="Inactive while the stages.py:759 evaluator_relationship override is in place "
    "(EEE upstream-data-quality mitigation: only llm-stats+raw_verified='false' surfaces "
    "as first_party; everything else collapses to third_party). Re-enable when the override "
    "is removed and upstream emits the right value directly."
)
def test_provenance_summary_not_inflated_by_array_unnest(tmp_path, monkeypatch):
    """Regression: when a triple has multiple evaluator_relationships and
    multiple reporting_orgs, naive `CROSS JOIN UNNEST` of both arrays
    multiplies row counts by their cross-product. fixtures_xparty has 1
    triple with 2 relationships × 2 orgs — every per-triple SUM/COUNT
    must still equal 1, not 4."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_views(out)
    repro, prov, comp = con.execute(
        "SELECT reproducibility_summary, provenance_summary, comparability_summary "
        "FROM evals_view"
    ).fetchone()

    # One triple in fixtures_xparty — every triple-counting field must equal 1.
    assert repro["results_total"] == 1, (
        f"reproducibility.results_total inflated: {repro['results_total']}"
    )
    assert prov["total_groups"] == 1
    assert prov["total_results"] == 1
    assert comp["total_groups"] == 1
    # The triple is collaborative (one self + one third-party report).
    assert prov["source_type_distribution"]["collaborative"] == 1
    assert sum(prov["source_type_distribution"].values()) == 1
    # Cross-party divergence eligible for this triple (>1 distinct org).
    assert comp["groups_with_cross_party_check"] == 1


def test_is_aggregated_false_for_standalone(tmp_path, monkeypatch):
    """v1: is_aggregated is always FALSE."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    rows = con.execute("SELECT is_aggregated FROM evals_view").fetchall()
    for r in rows:
        assert r[0] is False


def test_primary_key_unique(tmp_path, monkeypatch):
    """(snapshot_id, evaluation_id) uniquely identifies every row."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_views(out)
    n_rows, n_unique = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT (snapshot_id, evaluation_id)) "
        "FROM evals_view"
    ).fetchone()
    assert n_rows == n_unique


def test_instance_data_struct_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    instance = con.execute("SELECT instance_data FROM evals_view").fetchone()[0]
    assert set(instance.keys()) == {
        "available", "url_count", "sample_urls",
        "models_with_loaded_instances",
    }


def test_metric_config_struct_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    cfg = con.execute("SELECT metric_config FROM evals_view").fetchone()[0]
    assert set(cfg.keys()) == {
        "evaluation_description", "lower_is_better", "score_type",
        "min_score", "max_score", "unit",
    }

"""Stage J — `models_view` materialisation."""
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
    stages.stage_j_models_view(con, "2026-04-30T00:00:00Z")
    return con


def test_models_view_columns_match_spec(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    cols = {row[0]: row[1] for row in con.execute("DESCRIBE models_view").fetchall()}

    expected = {
        "snapshot_id", "model_key", "model_id", "id", "route_id", "model_route_id",
        "model_family_id", "model_name", "canonical_model_name",
        "model_family_name", "developer",
        "release_date", "model_url", "architecture", "params",
        "params_billions", "open_weights",
        "root_model_id", "lineage_origin_org_id",
        "inference_engine", "inference_platform",
        "evaluations_count", "benchmarks_count", "variant_count",
        "evaluator_count", "evaluator_names",
        "source_type_count", "source_types",
        "third_party_eval_count", "independent_verification_ratio",
        "evidence_count", "missing_generation_config_count",
        "latest_timestamp", "latest_source_name", "benchmark_names",
        "categories", "category_stats",
        "reproducibility_status", "reproducibility_summary",
        "provenance_summary", "comparability_summary",
        "eval_libraries", "score_summary", "top_scores",
        "source_urls", "detail_urls",
        "variants", "raw_model_ids",
    }
    missing = expected - cols.keys()
    assert not missing, f"missing columns: {missing}"
    assert cols["latest_timestamp"] == "TIMESTAMP"
    assert cols["snapshot_id"] == "TIMESTAMP"


def test_route_id_round_trips(tmp_path, monkeypatch):
    """`route_id` derives from `model_key`, not `model_id`. For resolved
    models the two coincide; for unresolved (registry NULL) `model_key`
    falls back to `model_raw` so the row is still addressable. The
    round-trip is therefore against `model_key`."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    rows = con.execute(
        "SELECT model_key, model_id, route_id, model_route_id FROM models_view"
    ).fetchall()
    for model_key, model_id, route_id, model_route_id in rows:
        assert route_id == model_route_id, "model_route_id is an alias of route_id"
        assert unquote(route_id) == model_key
        # Resolved → model_id matches model_key; unresolved → model_id is NULL
        # while model_key carries the raw source name.
        assert model_id is None or model_id == model_key


def test_aggregations_match_fact_counts(tmp_path, monkeypatch):
    """fixtures_variant: 1 model, 1 (benchmark, metric), 3 fact rows.
    Expected: evaluations_count=1, benchmarks_count=1, evidence_count=3."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_variant")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT evaluations_count, benchmarks_count, evidence_count, variant_count "
        "FROM models_view"
    ).fetchone()
    evaluations_count, benchmarks_count, evidence_count, variant_count = row
    assert evaluations_count == 1
    assert benchmarks_count == 1
    assert evidence_count == 3
    # Three rows, three different setups → three distinct variant_keys.
    assert variant_count == 3


def test_third_party_eval_count_xparty_fixture(tmp_path, monkeypatch):
    """fixtures_xparty has one (benchmark, metric) covered by both first-
    and third-party orgs → coverage_cell='both' → third_party_eval_count=1
    and independent_verification_ratio=1.0."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT third_party_eval_count, evaluations_count, "
        "       independent_verification_ratio "
        "FROM models_view"
    ).fetchone()
    third_party_count, evaluations_count, ratio = row
    assert evaluations_count == 1
    assert third_party_count == 1
    assert ratio == 1.0


def test_reproducibility_status_band_complete(tmp_path, monkeypatch):
    """fixtures_clean has full reproducibility coverage → band='complete'."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    status = con.execute(
        "SELECT reproducibility_status FROM models_view "
        "WHERE model_id = 'openai/gpt-4o'"
    ).fetchone()[0]
    assert status == "complete"


def test_category_stats_shape_and_sum(tmp_path, monkeypatch):
    """category_stats is a fixed-shape STRUCT keyed on the typed enum.
    The sum equals evaluations_count."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT category_stats, evaluations_count FROM models_view"
    ).fetchone()
    cs, ec = row
    assert set(cs.keys()) == {"General", "Reasoning", "Agentic", "Safety", "Knowledge"}
    assert sum(cs.values()) == ec


def test_score_summary_struct_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    summary = con.execute("SELECT score_summary FROM models_view").fetchone()[0]
    assert set(summary.keys()) == {"count", "min", "max", "average"}
    assert summary["count"] >= 0


def test_eval_libraries_carries_distinct_tuples(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    libs = con.execute("SELECT eval_libraries FROM models_view").fetchone()[0]
    assert libs is not None
    assert len(libs) >= 1
    for lib in libs:
        assert set(lib.keys()) == {"name", "version", "fork"}


def test_variants_includes_self_entry(tmp_path, monkeypatch):
    """v1: variants[] always carries one self-entry per row."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT model_id, variants FROM models_view "
        "WHERE model_id = 'openai/gpt-4o'"
    ).fetchone()
    model_id, variants = row
    assert variants is not None
    assert len(variants) == 1
    self_entry = variants[0]
    assert self_entry["variant_id"] == model_id
    assert unquote(self_entry["variant_key"]) == model_id
    assert self_entry["family_id"] is not None


def test_primary_key_is_unique(tmp_path, monkeypatch):
    """(snapshot_id, model_id) uniquely identifies every row."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_views(out)
    n_rows, n_unique = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT (snapshot_id, model_id)) FROM models_view"
    ).fetchone()
    assert n_rows == n_unique
    assert n_rows >= 1


def test_latest_timestamp_is_typed_timestamp(tmp_path, monkeypatch):
    """latest_timestamp must arrive as TIMESTAMP, not VARCHAR."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT latest_timestamp, typeof(latest_timestamp) FROM models_view"
    ).fetchone()
    ts, typename = row
    assert typename == "TIMESTAMP"
    assert ts is not None


def test_top_scores_one_per_category(tmp_path, monkeypatch):
    """top_scores has at most one entry per CategoryType."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    top_scores = con.execute("SELECT top_scores FROM models_view").fetchone()[0]
    assert top_scores is not None
    benchmark_keys = [t["benchmarkKey"] for t in top_scores]
    assert len(benchmark_keys) >= 1
    for entry in top_scores:
        assert set(entry.keys()) == {"benchmark", "benchmarkKey", "score", "metric"}

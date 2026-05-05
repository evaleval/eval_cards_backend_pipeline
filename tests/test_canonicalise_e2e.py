"""End-to-end smoke test: full pipeline over a hand-built fixture.

Builds a tiny EEE/cards/registry fixture in `tmp_path`, runs the canonicalisation
pipeline, asserts the warehouse parquets exist with non-zero rows. Exercises
the full DuckDB stage chain + UDF registration.

Skipped if duckdb isn't installed (it's now a hard dep so this is just paranoia).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_eee_fixture(eee_root: Path) -> None:
    config_dir = eee_root / "data" / "minibench" / "openai" / "gpt-4o"
    config_dir.mkdir(parents=True)
    record = {
        "evaluation_id": "ev_001",
        "schema_version": "0.2.2",
        "retrieved_timestamp": "2026-04-30T00:00:00Z",
        "model_info": {
            "developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o",
            "inference_platform": "openai-api",
        },
        "source_metadata": {
            "source_name": "OpenAI", "source_type": "documentation",
            "source_organization_name": "OpenAI",
            "evaluator_relationship": "first_party",
        },
        "eval_library": {"name": "minibench", "version": "1.0"},
        "evaluation_results": [
            {
                "evaluation_name": "minibench",
                "source_data": {"dataset_name": "minibench", "source_type": "other"},
                "metric_config": {
                    "metric_id": "minibench.acc",
                    "metric_name": "accuracy",
                    "evaluation_description": "Accuracy on minibench",
                    "lower_is_better": False,
                },
                "score_details": {"score": 0.85},
                "generation_config": {
                    "generation_args": {"temperature": 0.0, "max_tokens": 1024}
                },
            },
            {
                # row with different setup → variant divergence eligible
                "evaluation_name": "minibench",
                "source_data": {"dataset_name": "minibench", "source_type": "other"},
                "metric_config": {
                    "metric_id": "minibench.acc",
                    "metric_name": "accuracy",
                    "evaluation_description": "Accuracy on minibench",
                    "lower_is_better": False,
                },
                "score_details": {"score": 0.50},
                "generation_config": {
                    "generation_args": {"temperature": 0.7, "max_tokens": 1024}
                },
            },
        ],
    }
    (config_dir / "rec.json").write_text(json.dumps(record))


def _write_registry_fixture(reg_root: Path) -> None:
    import pandas as pd

    (reg_root / "aliases").mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "1", "raw_value": "openai/gpt-4o", "entity_type": "model",
             "canonical_id": "openai/gpt-4o", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
            {"id": "2", "raw_value": "minibench", "entity_type": "benchmark",
             "canonical_id": "minibench", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
            {"id": "3", "raw_value": "Accuracy", "entity_type": "metric",
             "canonical_id": "accuracy", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
        ]
    ).to_parquet(reg_root / "aliases" / "part-0.parquet")

    (reg_root / "canonical_orgs").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "openai", "display_name": "OpenAI", "parent_org_id": None,
          "website": "https://openai.com", "hf_org": "openai", "kind": "company",
          "tags": "[]", "metadata": "{}", "review_status": "reviewed",
          "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "canonical_orgs" / "part-0.parquet")

    (reg_root / "canonical_models").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "openai/gpt-4o", "display_name": "GPT-4o", "developer": "OpenAI",
          "org_id": "openai", "family": "GPT-4", "architecture": None,
          "params_billions": None, "parents": "[]",
          "root_model_id": None, "lineage_origin_org_id": "openai",
          "open_weights": False, "release_date": "2024-05",
          "tags": "[]", "metadata": "{}", "review_status": "reviewed",
          "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "canonical_models" / "part-0.parquet")

    (reg_root / "canonical_benchmarks").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "minibench", "display_name": "MiniBench",
          "description": "tiny benchmark", "dataset_repo": None,
          "parent_benchmark_id": None, "tags": "[]", "metadata": "{}",
          "review_status": "reviewed", "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "canonical_benchmarks" / "part-0.parquet")

    (reg_root / "canonical_metrics").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "accuracy", "display_name": "Accuracy",
          "score_type": "continuous", "lower_is_better": False,
          "min_score": 0.0, "max_score": 1.0, "metadata": "{}",
          "review_status": "reviewed", "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "canonical_metrics" / "part-0.parquet")

    (reg_root / "eval_harnesses").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "minibench", "display_name": "MiniBench Harness", "version": "1.0",
          "fork_url": None, "metadata": "{}", "review_status": "reviewed",
          "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "eval_harnesses" / "part-0.parquet")


def _write_minimal_seed_fixture(seed_root: Path) -> None:
    """Lay down a stub composites.yaml so taxonomy.load_and_materialise gets
    a non-empty composite_config_map. Without this, stage A writes a 0-row
    map to cache, which the pipeline's stale-cache check then refuses to
    restore on --from-stage runs. The contents don't have to match the
    fixture's source_configs — the cache check only requires >0 rows.
    """
    seed_root.mkdir(parents=True, exist_ok=True)
    (seed_root / "composites.yaml").write_text(
        "minibench:\n  display: MiniBench\n  configs:\n    - minibench\n"
    )


def _write_cards_fixture(cards_root: Path) -> None:
    (cards_root / "cards").mkdir(parents=True)
    card = {
        "benchmark_card": {
            "benchmark_details": {
                "name": "MiniBench", "overview": "A tiny test benchmark.",
                "data_type": "qa", "domains": ["test"], "languages": ["en"],
                "similar_benchmarks": [], "resources": [],
            },
            "purpose_and_intended_users": {
                "goal": "test the pipeline", "audience": ["devs"],
                "tasks": ["qa"], "limitations": "tiny",
                "out_of_scope_uses": [],
            },
            "data": {"source": "synthetic", "size": "10",
                     "format": "json", "annotation": "manual"},
            "methodology": {
                "methods": ["exact match"], "metrics": ["accuracy"],
                "calculation": "n_correct / n_total",
                "interpretation": "higher is better",
                "baseline_results": "0.5",
                "validation": "manual review",
            },
            "ethical_and_legal_considerations": {
                "privacy_and_anonymity": "n/a", "data_licensing": "MIT",
                "consent_procedures": "n/a", "compliance_with_regulations": "n/a",
            },
        }
    }
    (cards_root / "cards" / "minibench.json").write_text(json.dumps(card))


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    _write_eee_fixture(eee_root)
    _write_registry_fixture(reg_root)
    seed_root = tmp_path / "seed"
    _write_minimal_seed_fixture(seed_root)
    _write_cards_fixture(cards_root)

    # Force the source loaders into local-only mode by setting the env vars.
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    settings = Settings.from_env()

    out_dir = pipeline.run(
        settings,
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
    )

    import duckdb

    con = duckdb.connect()
    fact_path = out_dir / "fact_results.parquet"
    assert fact_path.exists()

    rows = con.execute(
        f"SELECT model_id, benchmark_id, metric_id, score, has_reproducibility_gap, "
        f"has_variant_divergence, distinct_reporting_orgs, "
        f"completeness_score, completeness_populated_count, "
        f"completeness_total_fields_evaluated, "
        f"metric_kind, metric_unit, score_scale_anomaly, "
        f"variant_threshold_basis "
        f"FROM read_parquet('{fact_path}') ORDER BY score"
    ).fetchall()
    assert len(rows) == 2
    # Both rows resolved
    assert all(r[0] == "openai/gpt-4o" for r in rows)
    assert all(r[1] == "minibench" for r in rows)
    assert all(r[2] == "accuracy" for r in rows)
    # Both rows are in the same comparability group with two distinct setups
    # → variant divergence is applicable; magnitude (0.85 - 0.50) > threshold (0.05)
    assert all(r[5] is True for r in rows)
    # Single org → cross-party not applicable
    assert all(r[6] == 1 for r in rows)
    # Per-row completeness: full card (22 + 1 partial-data full) + 3 EEE
    # source_metadata (set in fixture) + 0 reserved = 26 / 28. Both rows
    # share the same EEE record's source_metadata so both should match.
    for r in rows:
        assert r[9] == 28              # total_fields_evaluated
        assert r[8] == 26              # populated_count
        assert abs(r[7] - 26 / 28) < 1e-9
    # Metric-meta hotfix: registry has score_type='continuous',
    # min=0, max=1, lower=False but no metric_kind/metric_unit. EEE per-record
    # has metric_name='accuracy' but no metric_kind/metric_unit either.
    # Hotfix should resolve:
    #   metric_kind = 'accuracy' (regex matched 'accuracy' in metric_name)
    #   metric_unit = 'proportion' (heuristic: min=0, max=1, continuous)
    # → variant_threshold_basis short-circuits to proportion.
    # → score_scale_anomaly evaluates against [0,1]: scores 0.5, 0.85 are in range → False.
    for r in rows:
        assert r[10] == "accuracy", f"metric_kind: {r[10]!r}"
        assert r[11] == "proportion", f"metric_unit: {r[11]!r}"
        assert r[12] is False, f"score_scale_anomaly: {r[12]!r}"
        assert r[13] == "proportion", f"basis: {r[13]!r}"

    bench = out_dir / "benchmarks.parquet"
    assert bench.exists()
    bench_rows = con.execute(
        f"SELECT benchmark_id, display_name, card_present, card_missing_count "
        f"FROM read_parquet('{bench}')"
    ).fetchall()
    # All 23 card-derived field scores are 1 → card_missing_count = 0.
    assert bench_rows == [("minibench", "MiniBench", True, 0)]

    # benchmark_completeness.parquet is not emitted — completeness lives
    # on fact_results columns instead.
    assert not (out_dir / "benchmark_completeness.parquet").exists()

    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    assert snap["row_counts"]["fact_results"] == 2
    assert "benchmark_completeness.parquet" not in snap["tables"]
    # No collisions in this fixture; the counter is still expected to be present.
    assert snap["row_counts"]["dropped_rows_dedup"] == 0
    # Stage J ran end-to-end: tables list extends with the 3 view parquets,
    # sidecars list carries the JSON files emitted by Stage J.
    assert "fact_results.parquet" in snap["tables"]
    assert "eval_results_view.parquet" in snap["tables"]
    assert "models_view.parquet" in snap["tables"]
    assert "evals_view.parquet" in snap["tables"]
    assert set(snap["sidecars"]) == {
        "manifest.json", "headline.json", "hierarchy.json",
        "benchmark_index.json",
    }

    # comparability_group_id is the full md5 (32 hex chars), not truncated.
    group_id = con.execute(
        f"SELECT DISTINCT comparability_group_id FROM read_parquet('{fact_path}') "
        f"WHERE comparability_group_id IS NOT NULL"
    ).fetchall()
    assert len(group_id) == 1
    assert len(group_id[0][0]) == 32

    # retrieved_timestamp survives Stage F.4 onto the published fact_results.
    # The view layer's "latest_*" rollups depend on it; snapshot_id is run-level
    # so it can't substitute.
    fact_cols = {
        r[0] for r in con.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{fact_path}'))"
        ).fetchall()
    }
    assert "retrieved_timestamp" in fact_cols
    ts_rows = con.execute(
        f"SELECT retrieved_timestamp FROM read_parquet('{fact_path}')"
    ).fetchall()
    assert all(r[0] is not None for r in ts_rows)

    # canonical_metrics.parquet carries snapshot_id like every other table.
    cm_path = out_dir / "canonical_metrics.parquet"
    cm_cols = {
        r[0] for r in con.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{cm_path}'))"
        ).fetchall()
    }
    assert "snapshot_id" in cm_cols

    # Stage J: every view-layer artifact must be present alongside the
    # canonical parquets after a default end-to-end run.
    for fname in (
        "eval_results_view.parquet",
        "models_view.parquet",
        "evals_view.parquet",
        "manifest.json",
        "headline.json",
        "hierarchy.json",
    ):
        assert (out_dir / fname).exists(), f"Stage J artifact missing: {fname}"

    # Sidecars are valid JSON with the documented shape.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["model_count"] >= 1
    assert manifest["summary_artifacts"]["corpus_aggregates"] == "headline.json"

    headline = json.loads((out_dir / "headline.json").read_text())
    assert "reproducibility" in headline
    assert "developers" in headline


def test_pipeline_from_stage_j_rebakes_view_layer(tmp_path, monkeypatch):
    """`--from-stage J` restores the canonical tables from cache, rebuilds
    the three view parquets + three JSON sidecars, and skips re-running
    Stages A-I. The warehouse dir keeps the canonical parquets intact."""
    pytest.importorskip("duckdb")

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    _write_eee_fixture(eee_root)
    _write_registry_fixture(reg_root)
    seed_root = tmp_path / "seed"
    _write_minimal_seed_fixture(seed_root)
    _write_cards_fixture(cards_root)

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    settings = Settings.from_env()
    cache_root = tmp_path / "cache"
    snapshot = "2026-04-30T00:00:00Z"

    # First, populate the cache via a full run.
    out_dir = pipeline.run(
        settings,
        snapshot_id=snapshot,
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(cache_root),
    )
    assert out_dir is not None
    erv_path = out_dir / "eval_results_view.parquet"
    erv_mtime_before = erv_path.stat().st_mtime

    # Delete the view parquets to confirm --from-stage J rebuilds them.
    erv_path.unlink()
    (out_dir / "models_view.parquet").unlink()
    (out_dir / "evals_view.parquet").unlink()
    (out_dir / "manifest.json").unlink()

    # Re-run with --from-stage J: should restore canonical tables from
    # cache, build views, write parquets + sidecars.
    out_dir2 = pipeline.run(
        settings,
        snapshot_id=snapshot,
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(cache_root),
        from_stage="J",
    )
    assert out_dir2 == out_dir

    # All view artifacts re-emerged.
    for fname in (
        "eval_results_view.parquet",
        "models_view.parquet",
        "evals_view.parquet",
        "manifest.json",
        "headline.json",
        "hierarchy.json",
    ):
        assert (out_dir2 / fname).exists()

    # The new view parquet's mtime is later than the original — confirms
    # it actually got rebuilt rather than restored from a stale on-disk
    # version.
    assert erv_path.stat().st_mtime >= erv_mtime_before


def test_metric_unit_inconsistency_is_deterministic_and_counted(tmp_path, monkeypatch):
    """Two rows in the same comparability group reporting different
    metric_units must (a) yield the same threshold basis on every re-run
    (deterministic per-group pick) and (b) bump the inconsistency counter
    in snapshot_meta so operators know to backfill the registry."""
    pytest.importorskip("duckdb")

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    _write_registry_fixture(reg_root)
    seed_root = tmp_path / "seed"
    _write_minimal_seed_fixture(seed_root)
    _write_cards_fixture(cards_root)

    config_dir = eee_root / "data" / "minibench" / "openai" / "gpt-4o"
    config_dir.mkdir(parents=True)

    def _record(eval_id: str, score: float, metric_unit: str | None,
                temperature: float) -> dict:
        # metric_name='Accuracy' resolves via the registry alias, so both
        # rows share canonical metric_id and group together. Registry has
        # no metric_unit, so the EEE per-record unit wins via the hotfix.
        # Different temperature on each row keeps differing_setup_fields
        # non-empty so variant divergence is computed.
        return {
            "evaluation_id": eval_id,
            "schema_version": "0.2.2",
            "retrieved_timestamp": "2026-04-30T00:00:00Z",
            "model_info": {
                "developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o",
                "inference_platform": "openai-api",
            },
            "source_metadata": {
                "source_name": "OpenAI", "source_type": "documentation",
                "source_organization_name": "OpenAI",
                "evaluator_relationship": "first_party",
            },
            "eval_library": {"name": "minibench", "version": "1.0"},
            "evaluation_results": [
                {
                    "evaluation_name": "minibench",
                    "source_data": {"dataset_name": "minibench", "source_type": "other"},
                    "metric_config": {
                        "metric_id": "minibench.acc",
                        "metric_name": "Accuracy",
                        "evaluation_description": "Accuracy on minibench",
                        "lower_is_better": False,
                        "metric_unit": metric_unit,
                    },
                    "score_details": {"score": score},
                    "generation_config": {
                        "generation_args": {"temperature": temperature, "max_tokens": 1024}
                    },
                },
            ],
        }

    # Two records, same canonical (model, benchmark, metric=accuracy),
    # but the two report different EEE-side metric_units. Different
    # temperatures make divergence applicable so threshold_basis populates.
    (config_dir / "percent.json").write_text(json.dumps(
        _record("ev_pct", 0.85, metric_unit="percent", temperature=0.0)
    ))
    (config_dir / "proportion.json").write_text(json.dumps(
        _record("ev_prop", 0.95, metric_unit="proportion", temperature=0.7)
    ))

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        snapshot_id="2026-05-03T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
    )

    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    assert snap["row_counts"]["comparability_groups_metric_unit_inconsistent"] == 1

    import duckdb
    con = duckdb.connect()
    fact_path = out_dir / "fact_results.parquet"
    rows = con.execute(
        f"SELECT DISTINCT variant_threshold_basis FROM read_parquet('{fact_path}') "
        f"WHERE variant_threshold_basis IS NOT NULL"
    ).fetchall()
    # Both rows in one group → one (deterministic) basis label.
    assert len(rows) == 1


def test_score_scale_anomaly_fires_on_declared_range_violations(tmp_path, monkeypatch):
    """score_scale_anomaly must fire when score is outside the declared
    [min_score, max_score] range, not only on the proportion-unit case."""
    pytest.importorskip("duckdb")

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    _write_registry_fixture(reg_root)
    seed_root = tmp_path / "seed"
    _write_minimal_seed_fixture(seed_root)
    _write_cards_fixture(cards_root)

    config_dir = eee_root / "data" / "minibench" / "openai" / "gpt-4o"
    config_dir.mkdir(parents=True)

    def _record(eval_id: str, score: float, *, metric_name: str,
                min_score: float, max_score: float) -> dict:
        return {
            "evaluation_id": eval_id,
            "schema_version": "0.2.2",
            "retrieved_timestamp": "2026-04-30T00:00:00Z",
            "model_info": {
                "developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o",
                "inference_platform": "openai-api",
            },
            "source_metadata": {
                "source_name": "OpenAI", "source_type": "documentation",
                "source_organization_name": "OpenAI",
                "evaluator_relationship": "first_party",
            },
            "eval_library": {"name": "minibench", "version": "1.0"},
            "evaluation_results": [
                {
                    "evaluation_name": "minibench",
                    "source_data": {"dataset_name": "minibench", "source_type": "other"},
                    "metric_config": {
                        "metric_id": f"minibench.{metric_name}",
                        "metric_name": metric_name,
                        "evaluation_description": f"{metric_name} on minibench",
                        "lower_is_better": False,
                        "min_score": min_score, "max_score": max_score,
                    },
                    "score_details": {"score": score},
                    "generation_config": {
                        "generation_args": {"temperature": 0.0, "max_tokens": 1024}
                    },
                },
            ],
        }

    # Row 1: percent-scale accuracy with score=120 (above max=100). Registry
    # has min/max=[0,1] for `accuracy` so registry wins → in-range; we use a
    # bespoke unresolved metric_name so EEE per-record [0, 100] survives.
    (config_dir / "above.json").write_text(json.dumps(
        _record("ev_above", 120.0, metric_name="custom_pct",
                min_score=0.0, max_score=100.0)
    ))
    # Row 2: same metric, in-range → no anomaly.
    (config_dir / "ok.json").write_text(json.dumps(
        _record("ev_ok", 50.0, metric_name="custom_pct",
                min_score=0.0, max_score=100.0)
    ))
    # Row 3: below the declared min.
    (config_dir / "below.json").write_text(json.dumps(
        _record("ev_below", -5.0, metric_name="custom_pct",
                min_score=0.0, max_score=100.0)
    ))

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        snapshot_id="2026-05-03T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
    )

    import duckdb
    con = duckdb.connect()
    fact_path = out_dir / "fact_results.parquet"
    rows = con.execute(
        f"SELECT evaluation_id, score_scale_anomaly FROM read_parquet('{fact_path}') "
        f"ORDER BY evaluation_id"
    ).fetchall()
    by_eval = dict(rows)
    assert by_eval["ev_above"] is True, "above max_score must flag anomaly"
    assert by_eval["ev_below"] is True, "below min_score must flag anomaly"
    assert by_eval["ev_ok"] is False, "in-range row must NOT flag anomaly"


def test_harness_raw_strips_unknown_version_sentinel(tmp_path, monkeypatch):
    """Upstream EEE writes eval_library.version='unknown' when the
    version isn't recorded. The producer must NOT feed 'helm unknown' to
    the resolver — strip the sentinel and pass just the harness name."""
    pytest.importorskip("duckdb")

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    _write_registry_fixture(reg_root)
    seed_root = tmp_path / "seed"
    _write_minimal_seed_fixture(seed_root)
    _write_cards_fixture(cards_root)

    config_dir = eee_root / "data" / "minibench" / "openai" / "gpt-4o"
    config_dir.mkdir(parents=True)

    def _record(eval_id: str, version) -> dict:
        return {
            "evaluation_id": eval_id,
            "schema_version": "0.2.2",
            "retrieved_timestamp": "2026-04-30T00:00:00Z",
            "model_info": {
                "developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o",
                "inference_platform": "openai-api",
            },
            "source_metadata": {
                "source_name": "OpenAI", "source_type": "documentation",
                "source_organization_name": "OpenAI",
                "evaluator_relationship": "first_party",
            },
            "eval_library": {"name": "helm", "version": version},
            "evaluation_results": [
                {
                    "evaluation_name": "minibench",
                    "source_data": {"dataset_name": "minibench", "source_type": "other"},
                    "metric_config": {
                        "metric_id": "minibench.acc", "metric_name": "accuracy",
                        "evaluation_description": "Accuracy on minibench",
                        "lower_is_better": False,
                    },
                    "score_details": {"score": 0.5},
                    "generation_config": {
                        "generation_args": {"temperature": 0.0, "max_tokens": 1024}
                    },
                },
            ],
        }

    (config_dir / "unknown.json").write_text(json.dumps(_record("ev_unk", "unknown")))
    (config_dir / "real.json").write_text(json.dumps(_record("ev_real", "1.0")))

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        snapshot_id="2026-05-03T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
    )

    import duckdb
    con = duckdb.connect()
    fact_path = out_dir / "fact_results.parquet"
    rows = con.execute(
        f"SELECT evaluation_id, harness_raw FROM read_parquet('{fact_path}') "
        f"ORDER BY evaluation_id"
    ).fetchall()
    by_eval = dict(rows)
    assert by_eval["ev_unk"] == "helm", \
        f"version='unknown' must be stripped; got {by_eval['ev_unk']!r}"
    assert by_eval["ev_real"] == "helm 1.0", \
        f"real version must round-trip; got {by_eval['ev_real']!r}"


def test_pipeline_drops_score_sentinel_when_scale_excludes_it(tmp_path, monkeypatch):
    """HELM emits score=-1 as 'evaluation failed'. Drop only when the
    declared metric scale (proportion/percent or min_score > -1) tells us
    -1 isn't a legitimate value. A delta-style metric whose declared scale
    includes -1 must NOT be dropped.
    """
    pytest.importorskip("duckdb")

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    _write_registry_fixture(reg_root)
    seed_root = tmp_path / "seed"
    _write_minimal_seed_fixture(seed_root)
    _write_cards_fixture(cards_root)

    config_dir = eee_root / "data" / "minibench" / "openai" / "gpt-4o"
    config_dir.mkdir(parents=True)

    def _record(eval_id: str, score: float, *, metric_name: str,
                min_score: float, max_score: float,
                metric_unit: str | None) -> dict:
        return {
            "evaluation_id": eval_id,
            "schema_version": "0.2.2",
            "retrieved_timestamp": "2026-04-30T00:00:00Z",
            "model_info": {
                "developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o",
                "inference_platform": "openai-api",
            },
            "source_metadata": {
                "source_name": "OpenAI", "source_type": "documentation",
                "source_organization_name": "OpenAI",
                "evaluator_relationship": "first_party",
            },
            "eval_library": {"name": "minibench", "version": "1.0"},
            "evaluation_results": [
                {
                    "evaluation_name": "minibench",
                    "source_data": {"dataset_name": "minibench", "source_type": "other"},
                    "metric_config": {
                        "metric_id": f"minibench.{metric_name}",
                        "metric_name": metric_name,
                        "evaluation_description": f"{metric_name} on minibench",
                        "lower_is_better": False,
                        "min_score": min_score,
                        "max_score": max_score,
                        "metric_unit": metric_unit,
                    },
                    "score_details": {"score": score},
                    "generation_config": {
                        "generation_args": {"temperature": 0.0, "max_tokens": 1024}
                    },
                },
            ],
        }

    # Row 1: score=-1 on a [0,1] proportion accuracy metric. Registry has the
    # accuracy alias → min_score=0 from the canonical metric → sentinel
    # filter fires → row dropped.
    (config_dir / "sentinel.json").write_text(json.dumps(
        _record("ev_sent", -1.0, metric_name="Accuracy",
                min_score=0.0, max_score=1.0, metric_unit="proportion")
    ))
    # Row 2: score=-1 on a metric whose declared scale spans [-1, 1].
    # `correlation` has no registry alias so the per-record EEE min_score=-1
    # wins via the metric_meta layered chain → sentinel filter does NOT fire
    # → row kept. This is the "legitimate -1" case the policy must preserve.
    (config_dir / "delta.json").write_text(json.dumps(
        _record("ev_delta", -1.0, metric_name="correlation",
                min_score=-1.0, max_score=1.0, metric_unit=None)
    ))
    # Row 3: a normal valid row so the rest of the pipeline has data.
    (config_dir / "ok.json").write_text(json.dumps(
        _record("ev_ok", 0.85, metric_name="Accuracy",
                min_score=0.0, max_score=1.0, metric_unit="proportion")
    ))

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        snapshot_id="2026-05-03T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
    )

    import duckdb
    con = duckdb.connect()
    fact_path = out_dir / "fact_results.parquet"
    rows = con.execute(
        f"SELECT evaluation_id, score FROM read_parquet('{fact_path}') "
        f"ORDER BY evaluation_id"
    ).fetchall()
    eval_ids = [r[0] for r in rows]
    assert "ev_sent" not in eval_ids, "sentinel row must be dropped"
    assert "ev_delta" in eval_ids, "legitimate -1 on a [-1,1] metric must survive"
    assert "ev_ok" in eval_ids

    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    assert snap["row_counts"]["dropped_rows_sentinel"] == 1
    assert snap["row_counts"]["dropped_rows_no_score"] == 0
    assert snap["row_counts"]["fact_results"] == 2


def test_pipeline_dedupes_fact_id_collisions(tmp_path, monkeypatch):
    """Two EEE records with the same evaluation_id collide on fact_id.
    Pipeline must keep the row with the latest retrieved_timestamp and surface
    the dedup count in snapshot_meta.
    """
    pytest.importorskip("duckdb")

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    _write_registry_fixture(reg_root)
    seed_root = tmp_path / "seed"
    _write_minimal_seed_fixture(seed_root)
    _write_cards_fixture(cards_root)

    # Two records, same evaluation_id, same one-element evaluation_results[]
    # → both produce fact_id = sha256("ev_dup:0")[:16]. Different
    # retrieved_timestamps decide which wins; different scores let the test
    # observe which row was kept.
    config_dir = eee_root / "data" / "minibench" / "openai" / "gpt-4o"
    config_dir.mkdir(parents=True)

    def _record(timestamp: str, score: float) -> dict:
        return {
            "evaluation_id": "ev_dup",
            "schema_version": "0.2.2",
            "retrieved_timestamp": timestamp,
            "model_info": {
                "developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o",
                "inference_platform": "openai-api",
            },
            "source_metadata": {
                "source_name": "OpenAI", "source_type": "documentation",
                "source_organization_name": "OpenAI",
                "evaluator_relationship": "first_party",
            },
            "eval_library": {"name": "minibench", "version": "1.0"},
            "evaluation_results": [
                {
                    "evaluation_name": "minibench",
                    "source_data": {"dataset_name": "minibench", "source_type": "other"},
                    "metric_config": {
                        "metric_id": "minibench.acc",
                        "metric_name": "accuracy",
                        "evaluation_description": "Accuracy on minibench",
                        "lower_is_better": False,
                    },
                    "score_details": {"score": score},
                    "generation_config": {
                        "generation_args": {"temperature": 0.0, "max_tokens": 1024}
                    },
                },
            ],
        }

    (config_dir / "old.json").write_text(json.dumps(_record("2026-04-30T00:00:00Z", 0.30)))
    (config_dir / "new.json").write_text(json.dumps(_record("2026-05-03T00:00:00Z", 0.85)))

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        snapshot_id="2026-05-03T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
    )

    import duckdb
    con = duckdb.connect()
    fact_path = out_dir / "fact_results.parquet"
    rows = con.execute(
        f"SELECT fact_id, score FROM read_parquet('{fact_path}')"
    ).fetchall()
    assert len(rows) == 1, f"expected 1 row after dedup, got {len(rows)}: {rows}"
    assert rows[0][1] == 0.85, "later retrieved_timestamp must win"

    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    assert snap["row_counts"]["fact_results"] == 1
    assert snap["row_counts"]["dropped_rows_dedup"] == 1
    assert snap["row_counts"]["dropped_rows_no_score"] == 0

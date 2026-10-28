"""End-to-end smoke test: full pipeline over a hand-built fixture.

Builds a tiny EEE/cards/registry fixture in `tmp_path`, runs the canonicalisation
pipeline, asserts the warehouse parquets exist with non-zero rows. Exercises
the full DuckDB stage chain + UDF registration.

Skipped if duckdb isn't installed (it's now a hard dep so this is just paranoia).
"""
from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

import pytest

from tests.eee_layout import write_eee_datastore


def _write_eee_fixture(eee_root: Path) -> None:
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
    write_eee_datastore(eee_root, [("minibench", "rec.json", json.dumps(record))])


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


@pytest.mark.parametrize("card_layout", ["per_file", "flat_only", "both", "none"])
def test_pipeline_end_to_end(tmp_path, monkeypatch, card_layout):
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
    card_file = cards_root / "cards" / "minibench.json"
    if card_layout in {"flat_only", "both"}:
        payload = json.loads(card_file.read_text())["benchmark_card"]
        (cards_root / "benchmark-metadata.json").write_text(json.dumps({"minibench": payload}))
    if card_layout in {"flat_only", "none"}:
        card_file.unlink()
    if card_layout == "none":
        (cards_root / "benchmark-metadata.json").write_text("{}")

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
    populated_count = 3 if card_layout == "none" else 26
    for r in rows:
        assert r[9] == 28              # total_fields_evaluated
        assert r[8] == populated_count
        assert abs(r[7] - populated_count / 28) < 1e-9
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
    if card_layout != "none":
        assert bench_rows == [("minibench", "MiniBench", True, 0)]
    else:
        assert bench_rows[0][:3] == ("minibench", "MiniBench", False)

    expected_path = "cards/minibench.json" if card_layout in {"per_file", "both"} else None
    for table in ("benchmarks", "evals_view"):
        assert con.execute(
            f"SELECT benchmark_card_source_path FROM read_parquet('{out_dir}/{table}.parquet')"
        ).fetchall() == [(expected_path,)]

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
    for table in (
        "comparison_evals.parquet", "comparison_metrics.parquet",
        "comparison_scores.parquet",
    ):
        assert table in snap["tables"]
    assert all((out_dir / t).exists() for t in snap["tables"])
    # The comparison index ships as those three tables only.
    assert not (out_dir / "comparison-index.json").exists()
    assert set(snap["sidecars"]) == {
        "manifest.json", "headline.json", "hierarchy.json",
        "benchmark_index.json", "peer-ranks.json", "collections.json",
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

    # Divergence is a comparability-GROUP signal: the headline counts must equal
    # distinct comparability groups, not eval_results_view rows (a triple reported
    # under several evaluations must be counted once). Guards the row->group count
    # fix; see sensitivity/docs/divergence-count-grain.md.
    comp = headline["comparability"]["overall"]
    for flag, count_key, check_key in (
        ("has_variant_divergence", "variant_divergent_count", "groups_with_variant_check"),
        ("has_cross_party_divergence", "cross_party_divergent_count", "groups_with_cross_party_check"),
    ):
        flagged, eligible = con.execute(
            f"SELECT "
            f"  COUNT(DISTINCT comparability_group_id) FILTER (WHERE {flag}), "
            f"  COUNT(DISTINCT comparability_group_id) FILTER (WHERE {flag} IS NOT NULL) "
            f"FROM read_parquet('{fact_path}')"
        ).fetchone()
        assert comp[count_key] == flagged, (count_key, comp[count_key], flagged)
        assert comp[check_key] == eligible, (check_key, comp[check_key], eligible)


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
    write_eee_datastore(eee_root, [
        ("minibench", "percent.json", json.dumps(
            _record("ev_pct", 0.85, metric_unit="percent", temperature=0.0)
        )),
        ("minibench", "proportion.json", json.dumps(
            _record("ev_prop", 0.95, metric_unit="proportion", temperature=0.7)
        )),
    ])

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

    write_eee_datastore(eee_root, [
        # Row 1: percent-scale accuracy with score=120 (above max=100). Registry
        # has min/max=[0,1] for `accuracy` so registry wins → in-range; we use a
        # bespoke unresolved metric_name so EEE per-record [0, 100] survives.
        ("minibench", "above.json", json.dumps(
            _record("ev_above", 120.0, metric_name="custom_pct",
                    min_score=0.0, max_score=100.0)
        )),
        # Row 2: same metric, in-range → no anomaly.
        ("minibench", "ok.json", json.dumps(
            _record("ev_ok", 50.0, metric_name="custom_pct",
                    min_score=0.0, max_score=100.0)
        )),
        # Row 3: below the declared min.
        ("minibench", "below.json", json.dumps(
            _record("ev_below", -5.0, metric_name="custom_pct",
                    min_score=0.0, max_score=100.0)
        )),
    ])

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

    write_eee_datastore(eee_root, [
        ("minibench", "unknown.json", json.dumps(_record("ev_unk", "unknown"))),
        ("minibench", "real.json", json.dumps(_record("ev_real", "1.0"))),
    ])

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

    write_eee_datastore(eee_root, [
        # Row 1: score=-1 on a [0,1] proportion accuracy metric. Registry has the
        # accuracy alias → min_score=0 from the canonical metric → sentinel
        # filter fires → row dropped.
        ("minibench", "sentinel.json", json.dumps(
            _record("ev_sent", -1.0, metric_name="Accuracy",
                    min_score=0.0, max_score=1.0, metric_unit="proportion")
        )),
        # Row 2: score=-1 on a metric whose declared scale spans [-1, 1].
        # `correlation` has no registry alias so the per-record EEE min_score=-1
        # wins via the metric_meta layered chain → sentinel filter does NOT fire
        # → row kept. This is the "legitimate -1" case the policy must preserve.
        ("minibench", "delta.json", json.dumps(
            _record("ev_delta", -1.0, metric_name="correlation",
                    min_score=-1.0, max_score=1.0, metric_unit=None)
        )),
        # Row 3: a normal valid row so the rest of the pipeline has data.
        ("minibench", "ok.json", json.dumps(
            _record("ev_ok", 0.85, metric_name="Accuracy",
                    min_score=0.0, max_score=1.0, metric_unit="proportion")
        )),
    ])

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


def _assert_dedup_by_config_sums_to_row_counts(snap: dict) -> None:
    """Per-config dedup counts must add up to the global row_counts."""
    by_config = snap["dedup_by_config"]
    totals = snap["dedup_by_config_totals"]
    keys = {
        "staging", "after_fact_id_pass", "after_content_pass",
        "residual_timestamp_only", "residual_generation_args_only",
        "residual_both",
    }
    assert set(snap["configs"]) == set(by_config)
    for counts in by_config.values():
        assert set(counts) == keys
        assert all(isinstance(v, int) for v in counts.values())
    assert totals == {
        k: sum(c[k] for c in by_config.values()) for k in keys
    }
    rc = snap["row_counts"]
    assert totals["staging"] == rc["fact_results_pre_drop"]
    assert totals["after_fact_id_pass"] == (
        rc["fact_results_pre_drop"] - rc["dropped_rows_no_score"]
        - rc["dropped_rows_sentinel"] - rc["dropped_rows_dedup"]
    )
    assert totals["after_content_pass"] == rc["fact_results"]
    assert snap["dedup_by_config_consistent"] is True


def test_fixture_corpus_dedup_stats_add_up_on_both_paths(
    fixture_corpus_build, tmp_path, monkeypatch,
):
    """The per-config counts come from the Stage E decision table, so they
    add up to the global row counts on a fresh bake and again when Stage E
    is restored from its cache for a `--from-stage F` rebuild."""
    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings
    from tests.fixture_corpus import FIXTURES

    snap = json.loads((fixture_corpus_build.out_dir / "snapshot_meta.json").read_text())
    _assert_dedup_by_config_sums_to_row_counts(snap)
    assert snap["row_counts"]["fact_results"] > 0

    # Rebuild from a copy of the stage cache so a failure here cannot
    # disturb the session-shared bake.
    cache_root = tmp_path / "cache"
    shutil.copytree(fixture_corpus_build.cache_dir.parent, cache_root)
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(FIXTURES / "eee"))
    monkeypatch.setenv(
        "BENCHMARK_METADATA_LOCAL_DIR", str(FIXTURES / "auto_benchmarkcards")
    )
    out_dir = pipeline.run(
        Settings.from_env(),
        snapshot_id=fixture_corpus_build.snapshot_id,
        warehouse_dir=str(tmp_path / "warehouse"),
        registry_local_dir=str(fixture_corpus_build.registry_dir),
        cache_root=str(cache_root),
        from_stage="F",
    )
    snap2 = json.loads((out_dir / "snapshot_meta.json").read_text())
    _assert_dedup_by_config_sums_to_row_counts(snap2)
    stage_e_keys = (
        "fact_results_pre_drop", "fact_results", "dropped_rows_no_score",
        "dropped_rows_sentinel", "dropped_rows_dedup", "dropped_rows_content_dedup",
    )
    for key in stage_e_keys:
        assert snap2["row_counts"][key] == snap["row_counts"][key], key
    for key in (
        "dedup_by_config", "dedup_by_config_totals", "dedup_by_config_consistent",
    ):
        assert snap2[key] == snap[key], key


def test_snapshot_meta_refuses_fact_rows_the_dedup_stats_do_not_account_for(
    monkeypatch,
):
    from eval_card_backend.canonicalise import pipeline, stages
    from eval_card_backend.config import Settings

    monkeypatch.setattr(pipeline, "_hf_dataset_snapshot", lambda *a, **k: None)
    by_config = {"minibench": {
        "staging": 3, "after_fact_id_pass": 2, "after_content_pass": 1,
        "residual_timestamp_only": 0, "residual_generation_args_only": 0,
        "residual_both": 0,
    }}
    stats = stages.StageEStats(
        pre=3, n_dropped_no_score=0, n_dropped_sentinel=0,
        n_dropped_dedup=1, n_dropped_content_dedup=1, post=1,
        dedup_by_config=by_config,
    )
    kwargs = dict(
        snapshot_id="2026-05-01T00:00:00Z", chosen=["minibench"],
        settings=Settings.from_env(),
        n_eee=0, n_cards=0, n_exploded=0, n_synth_collisions=0,
        n_unit_inconsistent=0, view_layer_emitted=False,
        eee_root=None, registry_root=None, cards_root=None,
    )
    meta = pipeline._build_snapshot_meta(stage_e_stats=stats, n_fact_results=1, **kwargs)
    assert meta["dedup_by_config_consistent"] is True

    # One more row in the materialised fact_results than the content pass kept.
    with pytest.raises(
        stages.StageEInvariantError,
        match="fact_results has 2 rows but stage_e_dedup_stats "
              "after_content_pass sums to 1",
    ):
        pipeline._build_snapshot_meta(stage_e_stats=stats, n_fact_results=2, **kwargs)

    meta = pipeline._build_snapshot_meta(
        stage_e_stats=stats._replace(dedup_by_config=None), n_fact_results=2,
        **kwargs,
    )
    assert meta["dedup_by_config_consistent"] is None
    assert meta["dedup_by_config_totals"] is None


def test_observation_key_fields_are_pinned():
    """Every field in the Stage E observation key decides what counts as one
    observation; adding or dropping one changes which rows survive the
    content pass, so the list changes only deliberately, here and there."""
    from eval_card_backend.canonicalise import stages

    assert stages._OBSERVATION_KEY_FIELDS == (
        ("source_config", "source_config"),
        ("reporting_org", "org_raw"),
        ("evaluator_relationship", "evaluator_relationship"),
        ("source_type", "source_type"),
        ("model_raw", "model_raw"),
        ("inference_platform", "inference_platform"),
        ("evaluation_name", "evaluation_name"),
        ("benchmark_raw", "benchmark_raw"),
        ("benchmark_id", "benchmark_id"),
        ("benchmark_subset", "benchmark_subset"),
        ("slice_key", "slice_key"),
        ("slice_name", "slice_name"),
        ("aggregate_level", "aggregate_level"),
        ("observation_role", "observation_role"),
        ("collection_id", "collection_id"),
        ("metric_raw", "metric_raw"),
        ("metric_id", "metric_id"),
        ("metric_key", "metric_key"),
        ("metric_qualifier", "metric_qualifier"),
        ("split", "split"),
        ("score", "score"),
        ("n_samples", "n_samples"),
        ("evaluation_timestamp", "evaluation_timestamp"),
        ("generation_args", "generation_args_json"),
        ("generation_details", "generation_additional_details"),
        ("agent_scaffold", "agent_scaffold_raw"),
        ("harness_raw", "harness_raw"),
        ("harness_id", "harness_id"),
        ("eval_library_name", "eval_library_name"),
        ("eval_library_version", "eval_library_version"),
        ("protocol_condition", "protocol_condition"),
        ("judge_condition", "judge_condition"),
        ("instance_checksum", "instance_checksum"),
        ("instance_hash_algorithm", "instance_hash_algorithm"),
        ("instance_rows", "instance_rows"),
    )
    assert len(stages._OBSERVATION_KEY_FIELDS) == 35


def test_observation_key_renders_real_and_looser_struct_from_one_list():
    from eval_card_backend.canonicalise import stages

    real = stages._observation_key_sql()
    no_ts = stages._observation_key_sql(exclude=stages._TIMESTAMP_ONLY_EXCLUDES)
    no_args = stages._observation_key_sql(exclude=stages._GENERATION_ARGS_ONLY_EXCLUDES)
    assert all(s.startswith("struct_pack(") for s in (real, no_ts, no_args))
    assert "evaluation_timestamp := evaluation_timestamp" in real
    assert "generation_args := generation_args_json" in real
    assert "evaluation_timestamp" not in no_ts
    assert "generation_args := generation_args_json" in no_ts
    assert "generation_args :=" not in no_args
    assert "evaluation_timestamp := evaluation_timestamp" in no_args
    assert "generation_details := generation_additional_details" in no_args
    assert len(real.split(", ")) == len(no_ts.split(", ")) + 1 == len(no_args.split(", ")) + 1
    with pytest.raises(ValueError):
        stages._observation_key_sql(exclude=frozenset({"retrieved_timestamp"}))


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

    write_eee_datastore(eee_root, [
        ("minibench", "old.json", json.dumps(_record("2026-04-30T00:00:00Z", 0.30))),
        ("minibench", "new.json", json.dumps(_record("2026-05-03T00:00:00Z", 0.85))),
    ])

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
    _assert_dedup_by_config_sums_to_row_counts(snap)
    assert snap["dedup_by_config"] == {"minibench": {
        "staging": 2, "after_fact_id_pass": 1, "after_content_pass": 1,
        "residual_timestamp_only": 0, "residual_generation_args_only": 0,
        "residual_both": 0,
    }}


def test_pipeline_dedupes_reemitted_observations_by_content(tmp_path, monkeypatch):
    """Fresh record ids do not make identical observations independent.

    The content key must still retain changed values, source configs, and
    generation conditions. Equal-timestamp ties have a deterministic winner.
    snapshot_meta reports the per-config counts at each dedup boundary, and
    the residual of survivors that differ only in evaluation_timestamp,
    generation_args, or both, identically on a fresh run and a from-cache rebuild.
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

    def _record(
        evaluation_id: str,
        retrieved: str,
        score: float,
        *,
        temperature: float = 0.0,
        evaluated: str = "2026-04-29T00:00:00Z",
    ) -> dict:
        return {
            "evaluation_id": evaluation_id,
            "schema_version": "0.2.2",
            "retrieved_timestamp": retrieved,
            "evaluation_timestamp": evaluated,
            "model_info": {
                "developer": "openai", "name": "GPT-4o",
                "id": "openai/gpt-4o", "inference_platform": "openai-api",
            },
            "source_metadata": {
                "source_name": "OpenAI", "source_type": "documentation",
                "source_organization_name": "OpenAI",
                "evaluator_relationship": "first_party",
            },
            "eval_library": {"name": "minibench", "version": "1.0"},
            "evaluation_results": [{
                "evaluation_name": "minibench",
                "source_data": {
                    "dataset_name": "minibench", "source_type": "other",
                },
                "metric_config": {
                    "metric_id": "minibench.acc", "metric_name": "accuracy",
                    "evaluation_description": "Accuracy on minibench",
                    "lower_is_better": False,
                },
                "score_details": {"score": score},
                "generation_config": {
                    "generation_args": {
                        "temperature": temperature, "max_tokens": 1024,
                    }
                },
            }],
        }

    records = [
        # Same source/content under fresh ids: only the latest retrieval stays.
        ("minibench", "same-old.json", json.dumps(_record(
            "same_old", "2026-04-30T00:00:00Z", 0.8,
        ))),
        ("minibench", "same-new.json", json.dumps(_record(
            "same_new", "2026-05-03T00:00:00Z", 0.8,
        ))),
        # A different value and a different generation config both stay.
        ("minibench", "different-value.json", json.dumps(_record(
            "different_value", "2026-05-01T00:00:00Z", 0.9,
        ))),
        ("minibench", "different-generation.json", json.dumps(_record(
            "different_generation", "2026-05-01T00:00:00Z", 0.8,
            temperature=0.7,
        ))),
        # Equal retrieval timestamps use the documented total ordering.
        ("minibench", "tie-a.json", json.dumps(_record(
            "tie_a", "2026-05-02T00:00:00Z", 0.7,
        ))),
        ("minibench", "tie-z.json", json.dumps(_record(
            "tie_z", "2026-05-02T00:00:00Z", 0.7,
        ))),
        # Two results of ONE record under different source-native names are
        # two observations even when they resolve alike and share a value.
        ("minibench", "two-channels.json", json.dumps({
            **_record("two_channels", "2026-05-04T00:00:00Z", 0.6),
            "evaluation_results": [
                {**_record("x", "x", 0.6)["evaluation_results"][0],
                 "evaluation_name": "minibench.channel_a"},
                {**_record("x", "x", 0.6)["evaluation_results"][0],
                 "evaluation_name": "minibench.channel_b"},
            ],
        })),
        # Identical content from another source_config is independent.
        ("other-source", "other.json", json.dumps(_record(
            "other_source", "2026-05-03T00:00:00Z", 0.8,
        ))),
        # Same content under two stated evaluation times: both survive the
        # content pass and show up as that config's residual.
        ("ts-source", "ts-a.json", json.dumps(_record(
            "ts_a", "2026-05-03T00:00:00Z", 0.8, evaluated="2026-04-28T00:00:00Z",
        ))),
        ("ts-source", "ts-b.json", json.dumps(_record(
            "ts_b", "2026-05-03T00:00:00Z", 0.8, evaluated="2026-04-29T00:00:00Z",
        ))),
        # Same content apart from both the stated evaluation time and the
        # generation arguments: only the combined relaxation sees the pair.
        ("both-source", "both-a.json", json.dumps(_record(
            "both_a", "2026-05-03T00:00:00Z", 0.8, evaluated="2026-04-28T00:00:00Z",
        ))),
        ("both-source", "both-b.json", json.dumps(_record(
            "both_b", "2026-05-03T00:00:00Z", 0.8, evaluated="2026-04-29T00:00:00Z",
            temperature=0.7,
        ))),
    ]

    def _with(record: dict, **fields) -> dict:
        """The one result of `record` with `fields` set on it."""
        result = record["evaluation_results"][0]
        for key, value in fields.items():
            if key == "hf_split":
                result["source_data"]["hf_split"] = value
            else:
                result[key] = value
        return record

    records += [
        # Same value on two dataset splits: two observations.
        ("split-source", "split-test.json", json.dumps(_with(
            _record("split_test", "2026-05-03T00:00:00Z", 0.8), hf_split="test",
        ))),
        ("split-source", "split-validation.json", json.dumps(_with(
            _record("split_validation", "2026-05-03T00:00:00Z", 0.8), hf_split="validation",
        ))),
        # Same value on two slices of one benchmark: two observations.
        ("slice-source", "slice-a.json", json.dumps(_with(
            _record("slice_a", "2026-05-03T00:00:00Z", 0.8), evaluation_name="minibench.part_a",
        ))),
        ("slice-source", "slice-b.json", json.dumps(_with(
            _record("slice_b", "2026-05-03T00:00:00Z", 0.8), evaluation_name="minibench.part_b",
        ))),
    ]
    write_eee_datastore(eee_root, records)

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    run_kwargs = dict(
        snapshot_id="2026-05-03T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
    )
    out_dir = pipeline.run(Settings.from_env(), **run_kwargs)

    import duckdb

    fact_path = out_dir / "fact_results.parquet"
    rows = duckdb.connect().execute(
        f"SELECT source_config, evaluation_id, score, temperature "
        f"FROM read_parquet('{fact_path}') "
        f"ORDER BY source_config, score, temperature, evaluation_id"
    ).fetchall()
    assert rows == [
        ("both-source", "both_a", 0.8, 0.0),
        ("both-source", "both_b", 0.8, 0.7),
        ("minibench", "two_channels", 0.6, 0.0),
        ("minibench", "two_channels", 0.6, 0.0),
        ("minibench", "tie_z", 0.7, 0.0),
        ("minibench", "same_new", 0.8, 0.0),
        ("minibench", "different_generation", 0.8, 0.7),
        ("minibench", "different_value", 0.9, 0.0),
        ("other-source", "other_source", 0.8, 0.0),
        ("slice-source", "slice_a", 0.8, 0.0),
        ("slice-source", "slice_b", 0.8, 0.0),
        ("split-source", "split_test", 0.8, 0.0),
        ("split-source", "split_validation", 0.8, 0.0),
        ("ts-source", "ts_a", 0.8, 0.0),
        ("ts-source", "ts_b", 0.8, 0.0),
    ]
    axes = duckdb.connect().execute(
        f"SELECT evaluation_id, benchmark_id, split, slice_key "
        f"FROM read_parquet('{fact_path}') "
        f"WHERE source_config IN ('split-source', 'slice-source') "
        f"ORDER BY evaluation_id"
    ).fetchall()
    # slice_key is the cleaned raw name once a benchmark has more than one
    # raw spelling in the snapshot, so the whole-benchmark rows carry
    # "minibench" rather than NULL.
    assert axes == [
        ("slice_a", "minibench", None, "minibench part a"),
        ("slice_b", "minibench", None, "minibench part b"),
        ("split_test", "minibench", "test", "minibench"),
        ("split_validation", "minibench", "validation", "minibench"),
    ]

    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    assert snap["row_counts"]["fact_results"] == 15
    assert snap["row_counts"]["dropped_rows_dedup"] == 0
    assert snap["row_counts"]["dropped_rows_content_dedup"] == 2
    _assert_dedup_by_config_sums_to_row_counts(snap)
    # minibench's residual is same_new vs different_generation: equal on
    # every key field except generation_args. ts-source's is ts_a vs ts_b:
    # equal except for the stated evaluation time. Both also count once
    # both fields are dropped, and both-source counts only then.
    assert snap["dedup_by_config"] == {
        "both-source": {
            "staging": 2, "after_fact_id_pass": 2, "after_content_pass": 2,
            "residual_timestamp_only": 0, "residual_generation_args_only": 0,
            "residual_both": 1,
        },
        "minibench": {
            "staging": 8, "after_fact_id_pass": 8, "after_content_pass": 6,
            "residual_timestamp_only": 0, "residual_generation_args_only": 1,
            "residual_both": 1,
        },
        "other-source": {
            "staging": 1, "after_fact_id_pass": 1, "after_content_pass": 1,
            "residual_timestamp_only": 0, "residual_generation_args_only": 0,
            "residual_both": 0,
        },
        "slice-source": {
            "staging": 2, "after_fact_id_pass": 2, "after_content_pass": 2,
            "residual_timestamp_only": 0, "residual_generation_args_only": 0,
            "residual_both": 0,
        },
        "split-source": {
            "staging": 2, "after_fact_id_pass": 2, "after_content_pass": 2,
            "residual_timestamp_only": 0, "residual_generation_args_only": 0,
            "residual_both": 0,
        },
        "ts-source": {
            "staging": 2, "after_fact_id_pass": 2, "after_content_pass": 2,
            "residual_timestamp_only": 1, "residual_generation_args_only": 0,
            "residual_both": 1,
        },
    }
    assert snap["dedup_by_config_totals"] == {
        "staging": 17, "after_fact_id_pass": 17, "after_content_pass": 15,
        "residual_timestamp_only": 1, "residual_generation_args_only": 1,
        "residual_both": 3,
    }

    # A from-cache rebuild recovers the same counts from the cached tables.
    out_dir2 = pipeline.run(Settings.from_env(), **run_kwargs, from_stage="F")
    snap2 = json.loads((out_dir2 / "snapshot_meta.json").read_text())
    _assert_dedup_by_config_sums_to_row_counts(snap2)
    assert snap2["row_counts"] == snap["row_counts"]
    assert snap2["dedup_by_config"] == snap["dedup_by_config"]
    assert snap2["dedup_by_config_totals"] == snap["dedup_by_config_totals"]


_RETRIEVED_INSTANT_CASES = [
    ("2026-05-01T00:00:00+02:00", datetime.datetime(2026, 4, 30, 22, 0)),
    ("2026-05-01T00:00:00-07:00", datetime.datetime(2026, 5, 1, 7, 0)),
    ("2026-05-01T00:00:00Z", datetime.datetime(2026, 5, 1, 0, 0)),
    ("2026-05-01T00:00:00", datetime.datetime(2026, 5, 1, 0, 0)),
    ("2026-05-01", datetime.datetime(2026, 5, 1, 0, 0)),
    ("1777766400", datetime.datetime(2026, 5, 3, 0, 0)),
    ("1777766400.5", datetime.datetime(2026, 5, 3, 0, 0, 0, 500000)),
    ("20260501", None),
    ("1777766400000", None),
    ("inf", None),
    ("nan", None),
    ("", None),
    (None, None),
]


@pytest.mark.parametrize("time_zone", ["Asia/Tokyo", "America/Los_Angeles"])
def test_retrieved_instant_sql_is_utc_bounded_and_session_zone_independent(time_zone):
    """An explicit offset shifts the instant, a zoneless string is UTC, an
    epoch outside 2001..2096 (a bare date digit-string, milliseconds, inf)
    is NULL rather than a far-off year, and none of it moves with the
    session time zone."""
    import duckdb

    from eval_card_backend.canonicalise import stages

    con = duckdb.connect()
    con.execute(f"SET TimeZone = '{time_zone}'")
    con.execute("CREATE TABLE t (i INTEGER, retrieved_timestamp VARCHAR)")
    con.executemany(
        "INSERT INTO t VALUES (?, ?)",
        [(i, raw) for i, (raw, _) in enumerate(_RETRIEVED_INSTANT_CASES)],
    )
    rows = con.execute(
        f"SELECT i, {stages.retrieved_instant_sql('retrieved_timestamp')} "
        f"FROM t ORDER BY i"
    ).fetchall()
    assert [parsed for _, parsed in rows] == [
        expected for _, expected in _RETRIEVED_INSTANT_CASES
    ]
    # The +02:00 retrieval (30 Apr 22:00Z) orders before the epoch retrieval
    # on 3 May, as instants, not as text.
    ordered = con.execute(
        f"SELECT retrieved_timestamp FROM t "
        f"WHERE retrieved_timestamp IN ('2026-05-01T00:00:00+02:00', '1777766400') "
        f"ORDER BY {stages.retrieved_instant_sql('retrieved_timestamp')}"
    ).fetchall()
    assert [r[0] for r in ordered] == ["2026-05-01T00:00:00+02:00", "1777766400"]


def test_pipeline_dedup_orders_mixed_retrieved_timestamp_formats(
    tmp_path, monkeypatch,
):
    """retrieved_timestamp arrives as epoch seconds or ISO-8601 text. Both
    dedup passes must keep the later retrieval by parsed instant, not by
    text order (which ranks every ISO string above every epoch string).
    1777766400 is 2026-05-03T00:00:00Z; 1777852800 is 2026-05-04T00:00:00Z.
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

    def _record(evaluation_id: str, retrieved: str, score: float) -> dict:
        return {
            "evaluation_id": evaluation_id,
            "schema_version": "0.2.2",
            "retrieved_timestamp": retrieved,
            "evaluation_timestamp": "2026-04-29T00:00:00Z",
            "model_info": {
                "developer": "openai", "name": "GPT-4o",
                "id": "openai/gpt-4o", "inference_platform": "openai-api",
            },
            "source_metadata": {
                "source_name": "OpenAI", "source_type": "documentation",
                "source_organization_name": "OpenAI",
                "evaluator_relationship": "first_party",
            },
            "eval_library": {"name": "minibench", "version": "1.0"},
            "evaluation_results": [{
                "evaluation_name": "minibench",
                "source_data": {
                    "dataset_name": "minibench", "source_type": "other",
                },
                "metric_config": {
                    "metric_id": "minibench.acc", "metric_name": "accuracy",
                    "evaluation_description": "Accuracy on minibench",
                    "lower_is_better": False,
                },
                "score_details": {"score": score},
                "generation_config": {
                    "generation_args": {"temperature": 0.0, "max_tokens": 1024}
                },
            }],
        }

    write_eee_datastore(eee_root, [
        # fact_id collision: the epoch copy is the later retrieval.
        ("minibench", "fid-iso.json", json.dumps(_record(
            "ev_mixed", "2026-05-01T00:00:00Z", 0.30,
        ))),
        ("minibench", "fid-epoch.json", json.dumps(_record(
            "ev_mixed", "1777766400.5", 0.85,
        ))),
        # Content re-emission under fresh ids: the epoch copy is later.
        ("minibench", "content-iso.json", json.dumps(_record(
            "content_iso", "2026-05-01T00:00:00Z", 0.8,
        ))),
        ("minibench", "content-epoch.json", json.dumps(_record(
            "content_epoch", "1777766400", 0.8,
        ))),
        # Two epoch strings: the later instant wins.
        ("minibench", "epoch-old.json", json.dumps(_record(
            "epoch_old", "1777766400.25", 0.6,
        ))),
        ("minibench", "epoch-new.json", json.dumps(_record(
            "epoch_new", "1777852800.25", 0.6,
        ))),
    ])

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        snapshot_id="2026-05-05T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
    )

    import duckdb

    fact_path = out_dir / "fact_results.parquet"
    rows = duckdb.connect().execute(
        f"SELECT evaluation_id, score FROM read_parquet('{fact_path}') "
        f"ORDER BY score"
    ).fetchall()
    assert rows == [
        ("epoch_new", 0.6),
        ("content_epoch", 0.8),
        ("ev_mixed", 0.85),
    ]

    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    assert snap["row_counts"]["dropped_rows_dedup"] == 1
    assert snap["row_counts"]["dropped_rows_content_dedup"] == 2
    _assert_dedup_by_config_sums_to_row_counts(snap)


def test_metric_name_pre_step_end_to_end(tmp_path, monkeypatch):
    """Stage C tries a record's own metric_name as a registry alias before
    keyword extraction, but only when the description has nothing more
    specific to say. Four records cover the four cases; the registry fixture
    flags `score` as catch-all so the pre-step is armed (without that flag
    both metric pre-steps disable themselves)."""
    pytest.importorskip("duckdb")
    import pandas as pd

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    seed_root = tmp_path / "seed"
    _write_registry_fixture(reg_root)
    _write_minimal_seed_fixture(seed_root)
    _write_cards_fixture(cards_root)

    def metric(mid, name, catch_all=False):
        return {"id": mid, "display_name": name, "score_type": "continuous",
                "lower_is_better": False, "min_score": 0.0, "max_score": 1.0,
                "metadata": '{"catch_all": true}' if catch_all else "{}",
                "review_status": "reviewed", "created_at": "", "updated_at": ""}

    def alias(i, raw, cid):
        return {"id": str(i), "raw_value": raw, "entity_type": "metric",
                "canonical_id": cid, "source_config": None, "source_field": None,
                "status": "active", "strategy": "exact", "confidence": 1.0,
                "notes": None, "created_at": "", "updated_at": ""}

    pd.DataFrame([
        metric("accuracy", "Accuracy"), metric("macro-accuracy", "Macro Accuracy"),
        metric("task-success-rate", "Task Success Rate"), metric("success-rate", "Success Rate"),
        metric("score", "Score", catch_all=True),
    ]).to_parquet(reg_root / "canonical_metrics" / "part-0.parquet")
    aliases = pd.read_parquet(reg_root / "aliases" / "part-0.parquet")
    aliases = pd.concat([aliases, pd.DataFrame([
        alias(10, "Macro Accuracy", "macro-accuracy"),
        alias(11, "Task success rate", "task-success-rate"),
        alias(12, "Success Rate", "success-rate"),
        alias(13, "Score", "score"),
    ])], ignore_index=True)
    aliases.to_parquet(reg_root / "aliases" / "part-0.parquet")

    def result(metric_id, metric_name, description, score):
        cfg = {"metric_id": metric_id, "metric_name": metric_name, "lower_is_better": False}
        if description is not None:
            cfg["evaluation_description"] = description
        return {
            "evaluation_name": "minibench",
            "source_data": {"dataset_name": "minibench", "source_type": "other"},
            "metric_config": cfg,
            "score_details": {"score": score},
            "generation_config": {"generation_args": {"temperature": 0.0}},
        }

    record = {
        "evaluation_id": "ev_metric_name",
        "schema_version": "0.2.2",
        "retrieved_timestamp": "2026-04-30T00:00:00Z",
        "model_info": {"developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o",
                       "inference_platform": "openai-api"},
        "source_metadata": {"source_name": "OpenAI", "source_type": "documentation",
                            "source_organization_name": "OpenAI",
                            "evaluator_relationship": "first_party"},
        "eval_library": {"name": "minibench", "version": "1.0"},
        "evaluation_results": [
            # 1. the name is a registry alias the extractor would reduce to accuracy
            result("minibench.macro_acc", "Macro Accuracy", "Accuracy on minibench", 0.81),
            # 2. the description names a MORE specific metric: it keeps winning
            result("minibench.sr", "Success Rate", "Task success rate", 0.62),
            # 3. no description at all: the name is the best signal
            result("minibench.acc2", "Accuracy", None, 0.43),
            # 4. a catch-all name never outranks the description
            result("minibench.score", "Score", "Accuracy on minibench", 0.24),
        ],
    }
    target = eee_root / "data" / "fixtures_metric_name" / "openai" / "gpt-4o"
    target.mkdir(parents=True)
    (target / "01-metric-name.json").write_text(json.dumps(record))

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(), snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse), registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root), cache_root=str(tmp_path / "cache"),
    )
    import duckdb

    rows = duckdb.connect().execute(
        f"SELECT score, metric_id FROM read_parquet('{out_dir / 'fact_results.parquet'}') ORDER BY score"
    ).fetchall()
    assert rows == [
        (0.24, "accuracy"),          # catch-all name deferred to the description
        (0.43, "accuracy"),          # no description: direct name hit
        (0.62, "task-success-rate"), # specific description beat the generic name
        (0.81, "macro-accuracy"),    # direct name hit beat the lossy extraction
    ]

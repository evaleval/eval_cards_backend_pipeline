"""End-to-end tests reading the Stage J warehouse parquets directly.

Other Stage J tests run the per-view materialisation functions in-memory.
This file runs the full canonicalisation pipeline through the orchestrator
and asserts on the parquet files the consumer (frontend) actually reads.
Two scenarios are covered:

- A multi-model fixture (`_write_two_model_fixture`) — exercises ranking
  position/total/percentile across distinct models and the
  reproducibility band-rule edges (complete vs missing).
- A `lower_is_better=True` ranking direction test using a synthetic
  in-memory fact_results table — fixtures can't easily model a
  registry-side `lower_is_better=True` metric.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote

import duckdb
import pytest


# ---------------------------------------------------------------------------
# Custom 2-model fixture
# ---------------------------------------------------------------------------


def _write_eee_fixture_two_models(eee_root: Path) -> None:
    """Two-model fixture on the same benchmark+metric.

    - gpt-4o    : score=0.9, temperature + max_tokens set        → no repro gap
    - claude-3  : score=0.5, neither temperature nor max_tokens   → repro gap
    """
    base_dir = eee_root / "data" / "fixtures_two_models"

    def _record(eval_id: str, model_id: str, score: float, *, with_repro: bool) -> dict:
        gen_args: dict = {}
        if with_repro:
            gen_args = {"temperature": 0.0, "max_tokens": 1024}
        return {
            "evaluation_id": eval_id,
            "schema_version": "0.2.2",
            "retrieved_timestamp": "2026-04-30T00:00:00Z",
            "model_info": {
                "developer": "openai" if model_id == "openai/gpt-4o" else "anthropic",
                "name": "GPT-4o" if model_id == "openai/gpt-4o" else "Claude 3 Opus",
                "id": model_id,
                "inference_platform": "test",
            },
            "source_metadata": {
                "source_name": "Test", "source_type": "documentation",
                "source_organization_name": (
                    "OpenAI" if model_id == "openai/gpt-4o" else "Anthropic"
                ),
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
                    },
                    "score_details": {"score": score},
                    "generation_config": {"generation_args": gen_args},
                }
            ],
        }

    for path, rec in [
        (
            base_dir / "openai" / "gpt-4o" / "ev_a.json",
            _record("ev_a", "openai/gpt-4o", 0.9, with_repro=True),
        ),
        (
            base_dir / "anthropic" / "claude-3-opus" / "ev_b.json",
            _record("ev_b", "anthropic/claude-3-opus", 0.5, with_repro=False),
        ),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))


def _write_two_model_registry(reg_root: Path) -> None:
    import pandas as pd

    (reg_root / "aliases").mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "1", "raw_value": "openai/gpt-4o", "entity_type": "model",
             "canonical_id": "openai/gpt-4o", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
            {"id": "2", "raw_value": "anthropic/claude-3-opus", "entity_type": "model",
             "canonical_id": "anthropic/claude-3-opus", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
            {"id": "3", "raw_value": "minibench", "entity_type": "benchmark",
             "canonical_id": "minibench", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
            {"id": "4", "raw_value": "Accuracy", "entity_type": "metric",
             "canonical_id": "accuracy", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
        ]
    ).to_parquet(reg_root / "aliases" / "part-0.parquet")

    (reg_root / "canonical_orgs").mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "openai", "display_name": "OpenAI", "parent_org_id": None,
             "website": "https://openai.com", "hf_org": "openai", "kind": "company",
             "tags": "[]", "metadata": "{}", "review_status": "reviewed",
             "created_at": "", "updated_at": ""},
            {"id": "anthropic", "display_name": "Anthropic", "parent_org_id": None,
             "website": "https://anthropic.com", "hf_org": "anthropic", "kind": "company",
             "tags": "[]", "metadata": "{}", "review_status": "reviewed",
             "created_at": "", "updated_at": ""},
        ]
    ).to_parquet(reg_root / "canonical_orgs" / "part-0.parquet")

    (reg_root / "canonical_models").mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "openai/gpt-4o", "display_name": "GPT-4o", "developer": "OpenAI",
             "org_id": "openai", "family": "GPT-4", "architecture": None,
             "params_billions": None, "parents": "[]",
             "root_model_id": None, "lineage_origin_org_id": "openai",
             "open_weights": False, "release_date": "2024-05",
             "tags": "[]", "metadata": "{}", "review_status": "reviewed",
             "created_at": "", "updated_at": ""},
            {"id": "anthropic/claude-3-opus", "display_name": "Claude 3 Opus",
             "developer": "Anthropic", "org_id": "anthropic", "family": "Claude 3",
             "architecture": None, "params_billions": None, "parents": "[]",
             "root_model_id": None, "lineage_origin_org_id": "anthropic",
             "open_weights": False, "release_date": "2024-03-04",
             "tags": "[]", "metadata": "{}", "review_status": "reviewed",
             "created_at": "", "updated_at": ""},
        ]
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


def _run_pipeline(tmp_path, monkeypatch) -> Path:
    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    _write_eee_fixture_two_models(eee_root)
    _write_two_model_registry(reg_root)
    cards_root.mkdir()  # empty cards corpus is fine

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        cache_root=str(tmp_path / "cache"),
    )
    assert out_dir is not None
    return out_dir


# ---------------------------------------------------------------------------
# Warehouse layout
# ---------------------------------------------------------------------------


def test_warehouse_emits_all_eleven_files(tmp_path, monkeypatch):
    """End-to-end run produces 4 canonical + 3 view parquets + 3 JSON
    sidecars + snapshot_meta.json."""
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)
    expected = {
        "fact_results.parquet",
        "benchmarks.parquet",
        "models.parquet",
        "canonical_metrics.parquet",
        "eval_results_view.parquet",
        "models_view.parquet",
        "evals_view.parquet",
        "manifest.json",
        "headline.json",
        "hierarchy.json",
        "snapshot_meta.json",
    }
    actual = {p.name for p in out.iterdir() if p.is_file()}
    assert expected <= actual, f"missing: {expected - actual}"


def test_view_parquets_have_unique_primary_keys(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)
    con = duckdb.connect()

    for table, key in [
        ("eval_results_view", "(snapshot_id, metric_summary_id, model_id)"),
        ("models_view",       "(snapshot_id, model_id)"),
        ("evals_view",        "(snapshot_id, evaluation_id)"),
    ]:
        path = out / f"{table}.parquet"
        n_rows, n_unique = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT {key}) "
            f"FROM read_parquet('{path}')"
        ).fetchone()
        assert n_rows == n_unique, f"{table}: PK collision (rows={n_rows}, unique={n_unique})"
        assert n_rows >= 1, f"{table} is empty"


# ---------------------------------------------------------------------------
# Multi-model ranking
# ---------------------------------------------------------------------------


def test_position_total_percentile_higher_is_better(tmp_path, monkeypatch):
    """Two models on the same (benchmark, metric):
    - gpt-4o     score=0.9 → position=1, percentile=1.0
    - claude-3   score=0.5 → position=2, percentile=0.0
    total=2 for both."""
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)
    con = duckdb.connect()

    rows = con.execute(
        f"SELECT model_id, score, position, total, percentile "
        f"FROM read_parquet('{out / 'eval_results_view.parquet'}') "
        f"ORDER BY position"
    ).fetchall()
    assert len(rows) == 2

    top, bottom = rows
    top_id, top_score, top_pos, top_total, top_pct = top
    bot_id, bot_score, bot_pos, bot_total, bot_pct = bottom

    assert top_id == "openai/gpt-4o"
    assert top_pos == 1
    assert top_total == 2
    assert top_pct == 1.0

    assert bot_id == "anthropic/claude-3-opus"
    assert bot_pos == 2
    assert bot_total == 2
    assert bot_pct == 0.0


def test_evals_view_best_and_worst_model(tmp_path, monkeypatch):
    """`best_model.score` / `worst_model.score` track the lower_is_better
    rule on the primary metric — for a higher-is-better metric they
    equal max(score) / min(score) respectively."""
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)
    con = duckdb.connect()

    row = con.execute(
        f"SELECT best_model, worst_model, top_score, avg_score "
        f"FROM read_parquet('{out / 'evals_view.parquet'}')"
    ).fetchone()
    best, worst, top, avg = row
    assert best["score"] == 0.9
    assert best["name"] == "GPT-4o"
    assert worst["score"] == 0.5
    assert worst["name"] == "Claude 3 Opus"
    assert top == 0.9
    assert abs(avg - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# Reproducibility band rule (complete vs missing)
# ---------------------------------------------------------------------------


def test_reproducibility_band_rule_complete_and_missing(tmp_path, monkeypatch):
    """gpt-4o has temperature + max_tokens set → 'complete'.
    claude-3-opus has neither → 'missing'."""
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)
    con = duckdb.connect()

    rows = con.execute(
        f"SELECT model_id, reproducibility_status "
        f"FROM read_parquet('{out / 'models_view.parquet'}') "
        f"ORDER BY model_id"
    ).fetchall()
    by_model = dict(rows)
    assert by_model["openai/gpt-4o"] == "complete"
    assert by_model["anthropic/claude-3-opus"] == "missing"


def test_reproducibility_summary_counts_match(tmp_path, monkeypatch):
    """Per-model gap_count matches the band rule:
    complete → gap_count=0; missing → gap_count=results_total."""
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT model_id, reproducibility_summary, reproducibility_status "
        f"FROM read_parquet('{out / 'models_view.parquet'}')"
    ).fetchall()
    for model_id, summary, status in rows:
        if status == "complete":
            assert summary["has_reproducibility_gap_count"] == 0
        elif status == "missing":
            assert summary["has_reproducibility_gap_count"] == summary["results_total"]


# ---------------------------------------------------------------------------
# Slug round-trips on the warehouse parquets
# ---------------------------------------------------------------------------


def test_slugs_round_trip_on_warehouse_parquet(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)
    con = duckdb.connect()

    rows = con.execute(
        f"SELECT model_id, model_route_id, composite_slug, benchmark_id, "
        f"       evaluation_id, metric_id, metric_summary_id "
        f"FROM read_parquet('{out / 'eval_results_view.parquet'}')"
    ).fetchall()
    assert len(rows) >= 2
    for (model_id, model_route_id, composite_slug, bench_id, eval_id,
         metric_id, metric_summary_id) in rows:
        assert unquote(model_route_id) == model_id
        # evaluation_id encodes `<composite_slug>/<benchmark_id>`
        assert unquote(eval_id) == f"{composite_slug}/{bench_id}"
        assert unquote(metric_summary_id) == f"{bench_id}:{metric_id}"


# ---------------------------------------------------------------------------
# Sidecar invariants on the corpus
# ---------------------------------------------------------------------------


def test_manifest_counts_match_distinct_entities(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model_count"] == 2
    assert manifest["eval_count"] == 1     # 1 (benchmark, metric) pair
    assert manifest["metric_eval_count"] == 2  # 2 distinct (model, benchmark, metric) triples


def test_headline_developer_list_carries_both_orgs(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)
    h = json.loads((out / "headline.json").read_text())
    devs = {d["developer"] for d in h["developers"]}
    assert {"OpenAI", "Anthropic"} <= devs


# ---------------------------------------------------------------------------
# lower_is_better ranking direction
# ---------------------------------------------------------------------------


def test_lower_is_better_flips_ranking_direction(tmp_path, monkeypatch):
    """Synthesise a 2-model `eval_results_view` with `lower_is_better=True`
    and run only the ranking SQL to confirm position direction flips.

    Constructs a minimal `fact_results` + dim shape in-memory, runs the
    Stage J view materialisers, and asserts on `position`.
    """
    pytest.importorskip("duckdb")
    out = _run_pipeline(tmp_path, monkeypatch)

    from eval_card_backend.canonicalise.resolver_setup import register_udfs
    from eval_card_backend.canonicalise import stages
    from eval_card_backend.sources import registry as registry_src
    from eval_entity_resolver import Resolver

    con = duckdb.connect()
    register_udfs(con, Resolver(registry_src.load_alias_store(tmp_path / "reg")))

    # Reuse the warehouse parquets but flip lower_is_better on
    # canonical_metrics — and regenerate fact_results' lower_is_better
    # column so Stage J's ranking SQL reads the flipped direction.
    for table in (
        "fact_results", "benchmarks", "composites", "families", "models",
        "canonical_metrics",
    ):
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_parquet('{out / f'{table}.parquet'}')"
        )
    con.execute("UPDATE fact_results SET lower_is_better = TRUE")
    con.execute("UPDATE canonical_metrics SET lower_is_better = TRUE WHERE id = 'accuracy'")

    stages.stage_j_eval_results_view(con, "2026-04-30T00:00:00Z")

    rows = con.execute(
        "SELECT model_id, score, position, percentile FROM eval_results_view "
        "ORDER BY position"
    ).fetchall()
    # With lower_is_better=TRUE, the LOWER score should rank position=1.
    top_id, top_score, top_pos, top_pct = rows[0]
    assert top_id == "anthropic/claude-3-opus"
    assert top_score == 0.5
    assert top_pos == 1
    assert top_pct == 1.0

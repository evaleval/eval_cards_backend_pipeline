"""IGNORED_CONFIGS exclusion tests.

These exclusions are unconditional — alphaxiv (and any other entries) must
be filtered out even when a user explicitly passes the config name via
--configs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from eval_card_backend.config import IGNORED_CONFIGS


def test_alphaxiv_is_explicitly_ignored():
    """Regression guard: alphaxiv must stay excluded until upstream fixes."""
    assert "alphaxiv" in IGNORED_CONFIGS


def _write_minimal_eee(eee_root: Path, config: str) -> None:
    cfg_dir = eee_root / "data" / config / "openai" / "gpt-4o"
    cfg_dir.mkdir(parents=True)
    record = {
        "evaluation_id": f"ev_{config}",
        "schema_version": "0.2.2",
        "retrieved_timestamp": "2026-04-30T00:00:00Z",
        "model_info": {"developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o"},
        "source_metadata": {
            "source_type": "documentation",
            "source_organization_name": "OpenAI",
            "evaluator_relationship": "first_party",
        },
        "eval_library": {"name": "x", "version": "1"},
        "evaluation_results": [
            {
                "evaluation_name": config,
                "source_data": {"dataset_name": config},
                "metric_config": {"metric_id": f"{config}.acc", "metric_name": "accuracy"},
                "score_details": {"score": 0.5},
                "generation_config": {"generation_args": {"temperature": 0.0, "max_tokens": 1}},
            }
        ],
    }
    (cfg_dir / "rec.json").write_text(json.dumps(record))


def _write_registry(reg_root: Path) -> None:
    (reg_root / "aliases").mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "1", "raw_value": "openai/gpt-4o", "entity_type": "model",
             "canonical_id": "openai/gpt-4o", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
        ]
    ).to_parquet(reg_root / "aliases" / "part-0.parquet")
    for table in ("canonical_orgs", "canonical_models", "canonical_benchmarks",
                  "canonical_metrics", "eval_harnesses"):
        (reg_root / table).mkdir(parents=True)
        if table == "canonical_metrics":
            pd.DataFrame([{"id": "accuracy", "display_name": "Accuracy",
                           "score_type": "continuous", "lower_is_better": False,
                           "min_score": 0.0, "max_score": 1.0, "metadata": "{}",
                           "review_status": "reviewed", "created_at": "",
                           "updated_at": ""}]).to_parquet(
                reg_root / table / "part-0.parquet"
            )
        else:
            pd.DataFrame([{"id": "x", "display_name": "X", "metadata": "{}",
                           "review_status": "reviewed", "created_at": "",
                           "updated_at": ""}]).to_parquet(
                reg_root / table / "part-0.parquet"
            )


def test_alphaxiv_excluded_from_pipeline_even_when_explicitly_requested(
    tmp_path, monkeypatch, caplog
):
    """If a user passes --configs alphaxiv,foo, alphaxiv is still skipped.
    The pipeline runs over `foo` only, and a WARN log surfaces the exclusion.
    """
    pytest.importorskip("duckdb")

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    cards_root.mkdir()  # empty cards
    warehouse = tmp_path / "warehouse"

    _write_minimal_eee(eee_root, "minicfg")
    _write_minimal_eee(eee_root, "alphaxiv")
    _write_registry(reg_root)

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    settings = Settings.from_env()

    with caplog.at_level(logging.WARNING):
        out_dir = pipeline.run(
            settings,
            configs=["minicfg", "alphaxiv"],   # explicit ask includes alphaxiv
            snapshot_id="2026-04-30T00:00:00Z",
            warehouse_dir=str(warehouse),
            registry_local_dir=str(reg_root),
            cache_root=str(tmp_path / "cache"),
        )

    # alphaxiv is excluded — only minicfg's row(s) make it through
    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    assert snap["configs"] == ["minicfg"]
    assert "alphaxiv" not in snap["configs"]

    # WARN log surfaced the exclusion
    excluded_logs = [
        r for r in caplog.records
        if "ignoring" in r.message.lower() and "alphaxiv" in r.message
    ]
    assert excluded_logs, "expected a WARN log about ignoring alphaxiv"
    assert "upstream_data_quality" in excluded_logs[0].message

"""Contract tests for refactor-path identity normalization (no HF fetch)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.helpers.entity_resolution import resolve_model_identity_for_pipeline
from scripts.metadata import benchmark_normalization as bn
from scripts.metadata import metrics as metrics_mod


def _noop_org_model():
    return {"canonical_id": None, "strategy": None, "confidence": None}


@pytest.fixture
def minimal_metric_registry(tmp_path: Path) -> Path:
    path = tmp_path / "metric_looking_strings.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {"normalized": "accuracy", "display_name": "Accuracy"},
                    {"normalized": "pass_at_1", "display_name": "Pass@1"},
                ],
                "alias_to_normalized": {
                    "acc": "accuracy",
                    "exact_match": "exact_match",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_resolve_model_identity_bundle_shape():
    model_info = {
        "id": "anthropic/claude-3-5-sonnet-20241022",
        "name": "Claude 3.5 Sonnet",
        "developer": "Anthropic",
    }
    with patch("scripts.registry.resolve_org", return_value=_noop_org_model()):
        with patch("scripts.registry.resolve_model", return_value=_noop_org_model()):
            bundle = resolve_model_identity_for_pipeline(model_info)
    assert "canonical" in bundle and "display_route" in bundle
    canon = bundle["canonical"]
    disp = bundle["display_route"]
    assert canon["normalized_id"].startswith("anthropic/")
    assert disp["model_route_id"] == disp["family_id"].replace("/", "__")
    assert disp["merged_setup_alias"] is False


def test_canonicalize_metric_key_pass_at_and_alias(minimal_metric_registry: Path):
    metrics_mod.load_metric_registry(minimal_metric_registry)
    with patch("scripts.registry.resolve_metric", return_value=_noop_org_model()):
        assert metrics_mod.canonicalize_metric_key("Pass @ 1") == "pass_at_1"
        assert metrics_mod.canonicalize_metric_key("acc") == "accuracy"


def test_classify_evaluation_result_slash_slice(minimal_metric_registry: Path):
    metrics_mod.load_metric_registry(minimal_metric_registry)
    with patch("scripts.registry.resolve_metric", return_value=_noop_org_model()):
        evaluation = {"benchmark": "tau-bench-2"}
        result = {
            "evaluation_name": "accuracy",
            "source_data": {"dataset_name": "tau-bench-2/airline"},
            "metric_config": {
                "metric_name": "accuracy",
                "metric_id": "tau-bench-2.airline.accuracy",
                "evaluation_description": "Accuracy",
                "metric_kind": "score",
            },
        }
        out = bn.classify_evaluation_result(evaluation, result, None)
    assert out["benchmark_parent_key"] == "tau_bench_2"
    assert out["slice_key"] == "airline"
    assert out["metric_key"] == "accuracy"


def test_infer_category_from_card_domain():
    card = {
        "benchmark_details": {
            "domains": ["Coding"],
            "name": "Test",
        }
    }
    assert bn.infer_category_from_benchmark("ignored", card) == "coding"


def test_infer_category_from_benchmark_name_mmlu():
    assert bn.infer_category_from_benchmark("some_bench_mmlu_probe") == "reasoning"

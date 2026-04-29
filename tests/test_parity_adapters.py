"""Unit tests for ``scripts.parity_adapters``.

These exercise the TS-shape ports directly (the integration suite only
smoke-tests them via the fixture pipeline). Each test mirrors a TS code
path with explicit input/output shape assertions; quirks called out in
``general-eval-card/notes/transformations/`` are encoded as named tests.
"""
from __future__ import annotations

import json

import pytest

from scripts import parity_adapters


# ---------------------------------------------------------------------------
# slug helpers
# ---------------------------------------------------------------------------


def test_slugify_eval_id_collapses_non_alnum_to_underscore():
    """`lib/model-data.ts:62` — `[^a-z0-9]+` → `_`, strip edge underscores."""
    assert parity_adapters._slugify_eval_id("HELM-Lite v2") == "helm_lite_v2"
    assert parity_adapters._slugify_eval_id("___foo___") == "foo"
    assert parity_adapters._slugify_eval_id("Score / Accuracy") == "score_accuracy"
    assert parity_adapters._slugify_eval_id("") == ""
    assert parity_adapters._slugify_eval_id(None) == ""


def test_get_aggregate_eval_id_prefix():
    assert parity_adapters.get_aggregate_eval_id("helm_lite") == "aggregate__helm_lite"
    assert parity_adapters.get_aggregate_eval_id("HELM Lite v2") == "aggregate__helm_lite_v2"


def test_canonical_instance_results_url_filters_non_canonical():
    """`lib/hf-data.ts:1157` — only allow card_backend instance URLs."""
    canonical = (
        "https://huggingface.co/datasets/evaleval/card_backend/resolve/main/instances/foo.jsonl"
    )
    assert parity_adapters._canonical_instance_results_url(canonical) == canonical
    # Non-canonical URLs (different dataset, no instances/ path) are dropped.
    assert (
        parity_adapters._canonical_instance_results_url(
            "https://huggingface.co/datasets/other/path/instances/foo.jsonl"
        )
        is None
    )
    assert (
        parity_adapters._canonical_instance_results_url(
            "https://huggingface.co/datasets/evaleval/card_backend/main/raw/foo.json"
        )
        is None
    )
    assert parity_adapters._canonical_instance_results_url(None) is None
    assert parity_adapters._canonical_instance_results_url(123) is None


def test_locale_compare_key_lowercase_first_for_case_equivalents():
    """JS `localeCompare` puts `"yi-6b"` before `"Yi-6B"` (lowercase wins
    on case-equivalent inputs); Python codepoint sort would do the
    opposite. Caught by cross-repo verifier on the production corpus."""
    items = ["01-ai/Yi-6B", "01-ai/yi-6b"]
    assert sorted(items, key=parity_adapters._locale_compare_key) == [
        "01-ai/yi-6b",
        "01-ai/Yi-6B",
    ]


def test_locale_compare_key_position_by_position_case_tiebreak():
    """UCA tertiary level: position-by-position lowercase-before-uppercase
    so `"Ita"` (one uppercase) sorts before `"ITA"` (three uppercase).
    Caught by cross-repo verifier on `DeepMount00/Llama-3.1-8b-Ita` vs
    `-ITA` in the production corpus."""
    items = ["DeepMount00/Llama-3.1-8b-ITA", "DeepMount00/Llama-3.1-8b-Ita"]
    assert sorted(items, key=parity_adapters._locale_compare_key) == [
        "DeepMount00/Llama-3.1-8b-Ita",
        "DeepMount00/Llama-3.1-8b-ITA",
    ]


def test_locale_compare_key_mixed_case_namespace_split():
    """`deepseek/DeepSeek-R1` vs `deepseek/deepseek-r1` — position 9 is
    capital `D` (parity) vs lowercase `d` (TS). UCA puts lowercase first,
    so `deepseek/deepseek-r1` comes before `deepseek/DeepSeek-R1`."""
    items = ["deepseek/DeepSeek-R1", "deepseek/deepseek-r1"]
    assert sorted(items, key=parity_adapters._locale_compare_key) == [
        "deepseek/deepseek-r1",
        "deepseek/DeepSeek-R1",
    ]


def test_locale_compare_key_falls_back_to_codepoint():
    """When inputs aren't case-equivalent at all, primary key (casefold)
    decides — alphabetical ordering, NOT raw codepoint."""
    items = ["alpha", "Beta", "gamma"]
    assert sorted(items, key=parity_adapters._locale_compare_key) == [
        "alpha",
        "Beta",
        "gamma",
    ]


def test_coalesce_skips_only_none():
    assert parity_adapters._coalesce(None, None, "x") == "x"
    assert parity_adapters._coalesce(None, 0, "x") == 0  # `0` is not None
    assert parity_adapters._coalesce(None, False, 1) is False  # `False` is not None
    assert parity_adapters._coalesce(None, None) is None


# ---------------------------------------------------------------------------
# hf_model_card_to_evaluation_card_data — TS-shape EvaluationCardData
# ---------------------------------------------------------------------------


def _model_card_fixture() -> dict:
    return {
        "model_family_id": "openai/gpt-5",
        "model_family_name": "GPT-5",
        "developer": "openai",
        "total_evaluations": 13,
        "benchmark_count": 5,
        "benchmark_family_count": 4,
        "categories_covered": ["agentic", "reasoning", "safety"],
        "last_updated": "2024-06-20T00:00:00Z",
        "variants": [
            {"variant_key": "default", "variant_label": "Default", "last_updated": "2024-06-20T00:00:00Z"},
            {"variant_key": "20240701", "variant_label": "2024-07-01", "last_updated": "2024-07-01T00:00:00Z"},
        ],
        "score_summary": {"count": 12, "min": 0.1, "max": 0.9, "average": 0.5},
        "params_billions": "70B",
        "benchmark_names": ["mmlu", "gsm8k"],
    }


def test_hf_model_card_emits_ts_shape_keys():
    """The TS adapter rewrites keys: `id`, `route_id`, `model_name`, etc."""
    out = parity_adapters.hf_model_card_to_evaluation_card_data(_model_card_fixture())
    assert out["id"] == "openai/gpt-5"
    assert out["route_id"] == "openai__gpt-5"
    assert out["model_name"] == "GPT 5"
    assert out["model_id"] == "openai/gpt-5"
    assert out["canonical_model_name"] == "GPT 5"
    assert out["developer"] == "OpenAI"
    assert out["evaluations_count"] == 13


def test_hf_model_card_hardcoded_ts_active_path_defaults():
    """`hfModelCardToEvaluationCardData` hardcodes these per spec 14:
    no per-card aggregation is computed in the TS active path."""
    out = parity_adapters.hf_model_card_to_evaluation_card_data(_model_card_fixture())
    assert out["evaluator_count"] == 0
    assert out["evaluator_names"] == []
    assert out["source_type_count"] == 1
    assert out["source_types"] == ["documentation"]
    assert out["missing_generation_config_count"] == 0
    assert out["third_party_eval_count"] == 0
    assert out["independent_verification_ratio"] == 0
    assert out["reproducibility_status"] == "partial"
    assert out["eval_libraries"] == []
    assert out["source_urls"] == []
    assert out["detail_urls"] == []


def test_hf_model_card_categories_via_map_hf_categories():
    """`mapHFCategories(["agentic","reasoning","safety"])` lookup-table only,
    NOT regex fallback. Pipeline `other` → "General"."""
    fixture = {**_model_card_fixture(), "categories_covered": ["other", "agentic"]}
    out = parity_adapters.hf_model_card_to_evaluation_card_data(fixture)
    # "other" maps to "General"; deduplicated against subsequent mappings.
    assert out["categories"] == ["General", "Agentic"]


def test_hf_model_card_category_stats_path_a_fake_distribution():
    """Path A — ``Math.floor(total / N)`` with last-cat-takes-remainder."""
    out = parity_adapters.hf_model_card_to_evaluation_card_data(_model_card_fixture())
    # total=13, 3 categories → {3, 3, 7}? actually per spec 16 last takes remainder
    # so 3+3+7=13. But Math.floor(13/3) = 4, then remaining = 13-4-4 = 5 for last.
    # Wait: Math.floor(13/3)=4; remaining=13-4=9 then 9-4=5.
    # last_remainder = 5. So {Agentic: 4, Reasoning: 4, Safety: 5}.
    assert sum(out["category_stats"].values()) == 13
    assert len(out["category_stats"]) == 3


def test_hf_model_card_latest_timestamp_picks_max_across_variants():
    """`getModelCardLatestTimestamp` sorts via `normalizeEvalTimestamp`."""
    out = parity_adapters.hf_model_card_to_evaluation_card_data(_model_card_fixture())
    # Variant has 2024-07-01, top-level has 2024-06-20; max wins.
    assert out["latest_timestamp"] == "2024-07-01T00:00:00Z"


def test_hf_model_card_top_scores_average_fallback():
    """When `top_benchmark_scores` is missing, fall back to the cross-benchmark
    average row."""
    fixture = {**_model_card_fixture()}
    out = parity_adapters.hf_model_card_to_evaluation_card_data(fixture)
    assert len(out["top_scores"]) == 1
    assert out["top_scores"][0]["benchmark"] == "Average"
    assert out["top_scores"][0]["score"] == 0.5


def test_hf_model_card_benchmark_names_humanized():
    """Each name is run through `getBenchmarkDisplayName` (BENCHMARK_NAMES map)."""
    out = parity_adapters.hf_model_card_to_evaluation_card_data(_model_card_fixture())
    assert out["benchmark_names"] == ["Mmlu", "Gsm8k"]


def test_hf_model_card_params_billions_parsed():
    out = parity_adapters.hf_model_card_to_evaluation_card_data(_model_card_fixture())
    assert out["params_billions"] == 70.0


def test_hf_model_card_no_benchmarks_count_default():
    """TS `entry.benchmark_family_count || entry.benchmark_count` — no `|| 0`
    final fallback. Missing both fields leaves the value as None (was a bug
    in earlier impl that defaulted to 0)."""
    fixture = {**_model_card_fixture()}
    fixture.pop("benchmark_family_count")
    fixture.pop("benchmark_count")
    out = parity_adapters.hf_model_card_to_evaluation_card_data(fixture)
    assert out["benchmarks_count"] is None


# ---------------------------------------------------------------------------
# hf_eval_entry_to_list_item — BenchmarkEvalListItem shape
# ---------------------------------------------------------------------------


def _list_item_fixture() -> dict:
    return {
        "eval_summary_id": "helm_lite__mmlu",
        "benchmark": "helm_lite",
        "benchmark_parent_name": "HELM Lite",
        "benchmark_family_key": "helm",
        "benchmark_leaf_key": "mmlu",
        "benchmark_leaf_name": "MMLU",
        "evaluation_name": "MMLU",
        "display_name": "HELM Lite / MMLU",
        "category": "knowledge",
        "models_count": 12,
        "metrics_count": 1,
        "subtasks_count": 0,
        "metric_names": ["accuracy"],
        "primary_metric_name": "accuracy",
        "metrics": [
            {"metric_summary_id": "m1", "metric_name": "accuracy", "lower_is_better": False, "models_count": 12, "top_score": 0.95}
        ],
        "tags": {"domains": [], "languages": [], "tasks": []},
        "top_score": 0.95,
        "instance_data": {"available": False, "url_count": 0, "sample_urls": [], "models_with_loaded_instances": 0},
        "source_data": {"dataset_name": "mmlu"},
        "benchmark_card": None,
    }


def test_hf_eval_entry_to_list_item_emits_ts_shape():
    out = parity_adapters.hf_eval_entry_to_list_item(_list_item_fixture())
    assert out["evaluation_id"] == "helm_lite__mmlu"
    assert out["composite_benchmark_key"] == "helm_lite"
    assert out["composite_benchmark_name"] == "HELM Lite"  # display from BENCHMARK_NAMES
    assert out["category"] == "Knowledge"
    assert out["models_count"] == 12
    # Hardcoded TS-active-path defaults
    assert out["evaluator_names"] == []
    assert out["source_types"] == []
    assert out["third_party_ratio"] == 0
    assert out["missing_generation_config_count"] == 0
    assert out["avg_score"] == 0
    assert out["avg_score_norm"] == 0


def test_hf_eval_entry_to_list_item_best_model_from_top_score():
    """TS: `entry.top_score != null ? {name: "", score: top_score} : null`."""
    out = parity_adapters.hf_eval_entry_to_list_item(_list_item_fixture())
    assert out["best_model"] == {"name": "", "score": 0.95}
    assert out["worst_model"] is None


def test_hf_eval_entry_to_list_item_metric_config_synthesis():
    """`metric_config` is synthesized from primary metric: forced 0/1 range."""
    out = parity_adapters.hf_eval_entry_to_list_item(_list_item_fixture())
    cfg = out["metric_config"]
    assert cfg["min_score"] == 0
    assert cfg["max_score"] == 1
    assert cfg["score_type"] == "continuous"
    assert cfg["lower_is_better"] is False


# ---------------------------------------------------------------------------
# hf_eval_detail_to_summary — BenchmarkEvalSummary shape
# ---------------------------------------------------------------------------


def _eval_detail_fixture() -> dict:
    return {
        "eval_summary_id": "gsm8k_main",
        "benchmark": "gsm8k",
        "benchmark_leaf_name": "GSM8K",
        "canonical_display_name": "GSM8K / Accuracy",
        "metrics": [
            {
                "metric_summary_id": "gsm8k.accuracy",
                "metric_name": "accuracy",
                "display_name": "Accuracy",
                "lower_is_better": False,
                "metric_config": {
                    "min_score": 0,
                    "max_score": 1,
                    "evaluation_description": "accuracy",
                    "score_type": "continuous",
                },
                "model_results": [
                    {
                        "model_id": "openai/gpt-5",
                        "model_name": "GPT-5",
                        "developer": "openai",
                        "model_route_id": "openai__gpt-5",
                        "score": 0.88,
                        "retrieved_timestamp": "1700000000",
                        "source_metadata": {
                            "evaluator_relationship": "first_party",
                            "source_type": "leaderboard",
                            "source_name": "OpenAI evals",
                            "source_organization_name": "OpenAI",
                        },
                    },
                    {
                        "model_id": "anthropic/claude",
                        "model_name": "Claude",
                        "developer": "anthropic",
                        "model_route_id": "anthropic__claude",
                        "score": 0.92,
                        "retrieved_timestamp": "1700100000",
                        "source_metadata": {
                            "evaluator_relationship": "third_party",
                            "source_type": "leaderboard",
                            "source_name": "Other lab",
                            "source_organization_name": "Other Lab",
                        },
                    },
                ],
            }
        ],
        "subtasks": [],
        "benchmark_card": None,
    }


def test_hf_eval_detail_to_summary_sorts_by_score_desc():
    """`hfEvalDetailToSummary` sorts model_results by score descending
    (default; lower_is_better=True flips). Reasoning category comes from
    `inferCategoryFromBenchmark(eval_name)`."""
    out = parity_adapters.hf_eval_detail_to_summary(_eval_detail_fixture())
    scores = [r["score"] for r in out["model_results"]]
    assert scores == [0.92, 0.88]  # claude (0.92) before gpt-5 (0.88)
    # GSM8K → Reasoning per regex
    assert out["category"] == "Reasoning"


def test_hf_eval_detail_to_summary_metric_config_min_max_default():
    """When metric_config doesn't specify min/max, default to 0/1."""
    fixture = _eval_detail_fixture()
    fixture["metrics"][0]["metric_config"].pop("min_score", None)
    fixture["metrics"][0]["metric_config"].pop("max_score", None)
    out = parity_adapters.hf_eval_detail_to_summary(fixture)
    assert out["metric_config"]["min_score"] == 0
    assert out["metric_config"]["max_score"] == 1


def test_hf_eval_detail_to_summary_no_metrics_short_circuit():
    """When primary_metric is missing, return base shape with empty
    model_results and avg_score=0."""
    fixture = {**_eval_detail_fixture(), "metrics": [], "subtasks": []}
    out = parity_adapters.hf_eval_detail_to_summary(fixture)
    assert out["model_results"] == []
    assert out["models_count"] == 0
    assert out["avg_score"] == 0
    assert out["avg_score_norm"] == 0
    assert out["best_model"] is None
    # Empty-metric short-circuit puts benchmark display name into latest_source_name
    assert out["latest_source_name"] == "Gsm8k"  # falls through humanize fallback


def test_hf_eval_detail_to_summary_hardcoded_defaults():
    out = parity_adapters.hf_eval_detail_to_summary(_eval_detail_fixture())
    assert out["evaluator_names"] == []
    assert out["source_types"] == []
    assert out["third_party_ratio"] == 0
    assert out["missing_generation_config_count"] == 0


# ---------------------------------------------------------------------------
# flatten_model_evaluations — BenchmarkEvaluation[] post-#2-dedup
# ---------------------------------------------------------------------------


def _model_detail_fixture() -> dict:
    return {
        "model_family_id": "openai/gpt-5",
        "model_family_name": "GPT-5",
        "model_route_id": "openai__gpt-5",
        "model_info": {"id": "openai/gpt-5", "name": "GPT-5", "developer": "openai"},
        "raw_model_ids": ["openai/gpt-5"],
        "variants": [
            {"variant_key": "default", "variant_label": "Default", "raw_model_ids": ["openai/gpt-5"]}
        ],
        "hierarchy_by_category": {
            "reasoning": [
                {
                    "eval_summary_id": "gsm8k_main",
                    "benchmark": "gsm8k",
                    "benchmark_family_key": "gsm8k",
                    "benchmark_family_name": "GSM8K",
                    "benchmark_leaf_key": "main",
                    "benchmark_leaf_name": "GSM8K",
                    "display_name": "GSM8K",
                    "metrics": [
                        {
                            "metric_summary_id": "gsm8k.accuracy",
                            "metric_name": "accuracy",
                            "display_name": "Accuracy",
                            "metric_config": {
                                "min_score": 0,
                                "max_score": 1,
                                "lower_is_better": False,
                            },
                            "model_results": [
                                {
                                    "model_id": "openai/gpt-5",
                                    "raw_model_id": "openai/gpt-5",
                                    "model_route_id": "openai__gpt-5",
                                    "model_name": "GPT-5",
                                    "developer": "openai",
                                    "score": 0.88,
                                    "retrieved_timestamp": "1700000000",
                                    "source_metadata": {
                                        "evaluator_relationship": "first_party",
                                        "source_type": "leaderboard",
                                        "source_name": "OpenAI evals",
                                        "source_organization_name": "OpenAI",
                                    },
                                }
                            ],
                        }
                    ],
                    "subtasks": [],
                }
            ]
        },
    }


def test_flatten_model_evaluations_emits_one_row_per_metric_variant():
    out = parity_adapters.flatten_model_evaluations(_model_detail_fixture())
    assert len(out) == 1
    row = out[0]
    assert row["benchmark"] == "gsm8k"
    assert row["category"] == "Reasoning"
    assert row["evaluation_id"] == "gsm8k.accuracy__default"
    assert row["model_info"]["id"] == "openai/gpt-5"
    assert len(row["evaluation_results"]) == 1


def test_flatten_model_evaluations_skips_rows_missing_source_metadata(capsys):
    """Per the in-pipeline policy: warn + skip rather than raise."""
    fixture = _model_detail_fixture()
    fixture["hierarchy_by_category"]["reasoning"][0]["metrics"][0]["model_results"][0]["source_metadata"] = None
    out = parity_adapters.flatten_model_evaluations(fixture)
    assert out == []
    err = capsys.readouterr().err
    assert "missing source_metadata" in err


def test_flatten_model_evaluations_filters_other_models():
    """`belongsToModelFamily` filters rows to those matching the detail's
    raw_model_ids set."""
    fixture = _model_detail_fixture()
    other_row = {
        "model_id": "google/gemini-3",
        "raw_model_id": "google/gemini-3",
        "model_route_id": "google__gemini-3",
        "model_name": "Gemini 3",
        "developer": "google",
        "score": 0.85,
        "retrieved_timestamp": "1700000000",
        "source_metadata": {
            "evaluator_relationship": "third_party",
            "source_type": "leaderboard",
            "source_name": "x",
            "source_organization_name": "x",
        },
    }
    fixture["hierarchy_by_category"]["reasoning"][0]["metrics"][0]["model_results"].append(other_row)
    out = parity_adapters.flatten_model_evaluations(fixture)
    # Only the openai/gpt-5 row survives; gemini is filtered out.
    assert len(out) == 1
    assert out[0]["model_info"]["id"] == "openai/gpt-5"


def test_flatten_model_evaluations_evaluation_id_uses_variant_key_suffix():
    """Per spec reshape/03 — `evaluation_id = f"{metric_summary_id}__{variant_key}"`."""
    out = parity_adapters.flatten_model_evaluations(_model_detail_fixture())
    assert out[0]["evaluation_id"].endswith("__default")
    # rsplit on "__" recovers the variant_key from the suffix even when
    # the metric ID itself contains "__".
    assert out[0]["evaluation_id"].rsplit("__", 1)[1] == "default"


# ---------------------------------------------------------------------------
# create_model_family_summary — ModelEvaluationSummary shape
# ---------------------------------------------------------------------------


def _benchmark_evaluation(model_id: str, score: float, ts: str = "1700000000") -> dict:
    return {
        "schema_version": "0.2.2",
        "evaluation_id": "gsm8k.accuracy__default",
        "retrieved_timestamp": ts,
        "benchmark": "gsm8k",
        "category": "Reasoning",
        "source_data": {"dataset_name": "gsm8k"},
        "model_info": {
            "id": model_id,
            "name": "GPT 5",
            "developer": "openai",
        },
        "evaluation_results": [
            {
                "evaluation_name": "accuracy",
                "score_details": {"score": score},
                "metric_config": {"min_score": 0, "max_score": 1},
            }
        ],
    }


def test_create_model_family_summary_emits_canonical_identity():
    """Per spec 01 + reshape/03 — family_id/route_id/family_name from
    `getCanonicalModelIdentity` of the first evaluation."""
    evaluations = [_benchmark_evaluation("openai/gpt-5", 0.88)]
    out = parity_adapters.create_model_family_summary(evaluations)
    assert out["model_family_id"] == "openai/gpt-5"
    assert out["model_route_id"] == "openai__gpt-5"
    assert out["model_family_name"] == "GPT 5"


def test_create_model_family_summary_raw_model_ids_sorted_distinct():
    """`Array.from(new Set(...)).sort(localeCompare)`."""
    evaluations = [
        _benchmark_evaluation("openai/gpt-5", 0.88),
        _benchmark_evaluation("openai/gpt-5", 0.92, ts="1700100000"),
    ]
    out = parity_adapters.create_model_family_summary(evaluations)
    assert out["raw_model_ids"] == ["openai/gpt-5"]


def test_create_model_family_summary_variants_have_model_info_overlay():
    """Per `createModelFamilySummary`, variants[].model_info is overlaid
    with descriptor metadata (name = variantDisplayName, model_version)."""
    evaluations = [_benchmark_evaluation("openai/gpt-5", 0.88)]
    out = parity_adapters.create_model_family_summary(evaluations)
    assert isinstance(out["variants"], list)
    variant = out["variants"][0]
    assert "variant_id" in variant
    assert "variant_key" in variant
    assert "raw_model_ids" in variant
    assert variant["family_id"] == "openai/gpt-5"


def test_create_model_family_summary_rejects_empty_evaluations():
    with pytest.raises(ValueError):
        parity_adapters.create_model_family_summary([])


# ---------------------------------------------------------------------------
# aggregate_benchmark_summaries — composite rollup
# ---------------------------------------------------------------------------


def _eval_summary_for_aggregate(eval_id: str, score: float, lower_is_better: bool = False) -> dict:
    return {
        "evaluation_id": eval_id,
        "evaluation_name": eval_id.replace("__", " / "),
        "composite_benchmark_key": "gsm8k",
        "composite_benchmark_name": "GSM8K",
        "category": "Reasoning",
        "metric_config": {
            "min_score": 0,
            "max_score": 1,
            "lower_is_better": lower_is_better,
            "evaluation_description": "accuracy",
        },
        "evaluator_names": [],
        "source_types": [],
        "missing_generation_config_count": 0,
        "model_results": [
            {
                "model_info": {"id": "openai/gpt-5", "name": "GPT-5", "developer": "openai"},
                "score": score,
                "score_details": {"score": score},
                "evaluation_timestamp": "1700000000",
                "source_metadata": {
                    "evaluator_relationship": "first_party",
                    "source_type": "leaderboard",
                    "source_name": "ev",
                    "source_organization_name": "OpenAI",
                },
                "source_data": {"dataset_name": "gsm8k"},
                "result": {},
            }
        ],
    }


def test_aggregate_benchmark_summaries_returns_none_for_empty():
    assert parity_adapters.aggregate_benchmark_summaries([], "x") is None


def test_aggregate_benchmark_summaries_normalizes_before_averaging():
    """Per spec reshape/05 quirk #1 — normalize each sub-eval first, then avg."""
    summaries = [
        _eval_summary_for_aggregate("a", 0.8),
        _eval_summary_for_aggregate("b", 0.4),
    ]
    out = parity_adapters.aggregate_benchmark_summaries(summaries, "myfamily")
    assert out is not None
    assert out["evaluation_id"] == "aggregate__myfamily"
    # Both sub-evals at min=0,max=1 means normalized==raw. avg-of-per-model-avgs.
    assert out["avg_score"] == pytest.approx(0.6)
    assert out["models_count"] == 1


def test_aggregate_benchmark_summaries_multiple_sources_sentinel():
    """`latest_source_name = "Multiple sources"` when len(sources) > 1."""
    summaries = [
        _eval_summary_for_aggregate("a", 0.8),
        _eval_summary_for_aggregate("b", 0.4),
    ]
    out = parity_adapters.aggregate_benchmark_summaries(summaries, "myfamily")
    assert out["latest_source_name"] == "Multiple sources"


def test_aggregate_benchmark_summaries_metric_config_synthesized():
    """Synthesized config always has min_score=0, max_score=1, unit set."""
    summaries = [
        _eval_summary_for_aggregate("a", 0.8),
        _eval_summary_for_aggregate("b", 0.4),
    ]
    out = parity_adapters.aggregate_benchmark_summaries(summaries, "x")
    assert out["metric_config"]["min_score"] == 0
    assert out["metric_config"]["max_score"] == 1
    assert out["metric_config"]["unit"] == "normalized average"


# ---------------------------------------------------------------------------
# build_single_metric_suite_matrix_summary — matrix leaderboard
# ---------------------------------------------------------------------------


def _single_metric_detail(eval_id: str, score: float) -> dict:
    return {
        "eval_summary_id": eval_id,
        "benchmark_leaf_key": eval_id,
        "benchmark_leaf_name": eval_id.upper(),
        "benchmark": eval_id,
        "metrics": [
            {
                "metric_summary_id": f"{eval_id}.accuracy",
                "metric_name": "accuracy",
                "display_name": "Accuracy",
                "metric_config": {"min_score": 0, "max_score": 1, "lower_is_better": False},
                "model_results": [
                    {
                        "model_id": "openai/gpt-5",
                        "model_name": "GPT-5",
                        "developer": "openai",
                        "score": score,
                        "retrieved_timestamp": "1700000000",
                        "source_metadata": {
                            "evaluator_relationship": "first_party",
                            "source_type": "leaderboard",
                            "source_name": "ev",
                            "source_organization_name": "OpenAI",
                        },
                    }
                ],
            }
        ],
        "subtasks": [],
        "benchmark_card": None,
    }


def test_build_matrix_summary_returns_none_below_floors():
    """Need >=2 sub-evals and >=2 surviving columns."""
    assert parity_adapters.build_single_metric_suite_matrix_summary([], "x") is None
    assert (
        parity_adapters.build_single_metric_suite_matrix_summary(
            [_single_metric_detail("a", 0.5)], "x"
        )
        is None
    )


def test_build_matrix_summary_emits_columns_and_rows():
    details = [
        _single_metric_detail("a", 0.5),
        _single_metric_detail("b", 0.7),
    ]
    out = parity_adapters.build_single_metric_suite_matrix_summary(details, "myfamily")
    assert out is not None
    assert out["evaluation_id"] == "matrix__myfamily"
    assert len(out["leaderboard_metrics"]) == 2
    assert len(out["leaderboard_rows"]) == 1
    row = out["leaderboard_rows"][0]
    assert row["model_info"]["id"] == "openai/gpt-5"


def test_build_matrix_summary_skips_when_metrics_count_not_one():
    """Eligibility: `len(metrics) == 1`. Sub-evals with 2 metrics are dropped."""
    details = [
        _single_metric_detail("a", 0.5),
        _single_metric_detail("b", 0.7),
    ]
    # Add a second metric to detail b — should be excluded.
    details[1]["metrics"].append({**details[1]["metrics"][0]})
    out = parity_adapters.build_single_metric_suite_matrix_summary(details, "x")
    # Only `a` survives → suite floor (≥2) fails → None.
    assert out is None


def test_build_matrix_summary_shared_metric_name_only_when_all_match():
    """`shared_metric_name` is None when sub-evals report different metrics."""
    a = _single_metric_detail("a", 0.5)
    b = _single_metric_detail("b", 0.7)
    b["metrics"][0]["metric_name"] = "f1"  # different metric → not shared
    out = parity_adapters.build_single_metric_suite_matrix_summary([a, b], "x")
    # leaderboard_metrics still has 2 rows (one per sub-eval); shared_metric_name
    # is the "first" sub-eval's name when only one is unique. With two distinct
    # metrics, the suite_metric_config falls back to first metric's config.
    assert out is not None
    # Suite eval description follows TS pattern: shared name OR first sub's
    # description.
    descr = out["metric_config"]["evaluation_description"]
    assert descr in {"f1", "accuracy"}


# ---------------------------------------------------------------------------
# Smoke tests for the supporting helpers
# ---------------------------------------------------------------------------


def test_to_summary_metric_config_string_fields_pass_through():
    metric = {
        "metric_name": "accuracy",
        "display_name": "Accuracy",
        "lower_is_better": False,
        "metric_config": {
            "evaluation_description": "Accuracy",
            "min_score": 0,
            "max_score": 1,
            "score_type": "binary",
            "unit": "%",
        },
    }
    out = parity_adapters._to_summary_metric_config(metric)
    assert out["evaluation_description"] == "Accuracy"
    assert out["score_type"] == "binary"
    assert out["unit"] == "%"
    assert out["min_score"] == 0


def test_to_summary_metric_config_continuous_default_for_unknown_score_type():
    metric = {"metric_name": "x", "metric_config": {"score_type": "garbage"}}
    out = parity_adapters._to_summary_metric_config(metric)
    assert out["score_type"] == "continuous"


def test_extract_detail_subtasks_skips_non_dict_entries():
    detail = {"subtasks": [None, "skip me", {"display_name": "Sub", "metrics": []}]}
    out = parity_adapters._extract_detail_subtasks(detail)
    assert len(out) == 1
    assert out[0]["display_name"] == "Sub"


def test_aggregated_variant_descriptor_setup_alias_collapse():
    """`mode: "fc"` triggers setup-alias collapse → variantKey from
    versionDate when present."""
    desc = parity_adapters._aggregated_variant_descriptor(
        {
            "id": "anthropic/claude-3-5-sonnet-20240620",
            "additional_details": {"mode": "fc"},
        }
    )
    assert desc["mergedSetupAlias"] is True
    # Identity has versionDate=2024-06-20 → variantKey collapses to that date.
    assert desc["variantKey"] == "2024-06-20"


def test_aggregated_variant_descriptor_no_setup_alias():
    """When `mode` is absent or not a known alias, descriptor preserves
    the canonical variant key/label."""
    desc = parity_adapters._aggregated_variant_descriptor(
        {"id": "openai/gpt-5"}
    )
    assert desc["mergedSetupAlias"] is False
    assert desc["variantKey"] == "base"
    assert desc["variantDisplayName"] == "GPT 5"


def test_get_developer_benchmark_stats_dedups_per_model():
    """Each model contributes one count per distinct benchmark name. Same
    benchmark across multiple models bumps the count (`{mmlu: 2}`)."""
    models = [
        {"benchmark_names": ["MMLU", "GSM8K", "MMLU"]},  # MMLU dedup'd within
        {"benchmark_names": ["MMLU"]},
    ]
    out = parity_adapters.get_developer_benchmark_stats(models)
    assert out == {"MMLU": 2, "GSM8K": 1}


def test_get_developer_benchmark_stats_falls_back_to_top_scores():
    """When `benchmark_names` is empty, fall back to `top_benchmark_scores`
    per the TS truthy-coalesce."""
    models = [
        {
            "benchmark_names": [],
            "top_benchmark_scores": [
                {"benchmark": "MMLU"},
                {"benchmark": "MMLU"},  # dedup within model
                {"benchmark": "BBH"},
            ],
        }
    ]
    out = parity_adapters.get_developer_benchmark_stats(models)
    assert out == {"MMLU": 1, "BBH": 1}


def test_hf_developer_detail_to_summary_emits_ts_shape():
    detail = {
        "developer": "openai",
        "models": [
            {
                "model_family_id": "openai/gpt-5",
                "model_family_name": "GPT-5",
                "developer": "openai",
                "total_evaluations": 10,
                "benchmark_count": 2,
                "categories_covered": ["reasoning"],
                "score_summary": {"count": 10, "average": 0.85},
                "benchmark_names": ["MMLU", "GSM8K"],
                "variants": [],
            },
            {
                "model_family_id": "openai/gpt-4",
                "model_family_name": "GPT-4",
                "developer": "openai",
                "total_evaluations": 5,
                "benchmark_count": 1,
                "categories_covered": ["reasoning"],
                "score_summary": {"count": 5, "average": 0.8},
                "benchmark_names": ["MMLU"],
                "variants": [],
            },
        ],
    }
    out = parity_adapters.hf_developer_detail_to_summary(detail)
    assert out["developer"] == "OpenAI"
    assert out["route_id"] == "openai"
    assert out["model_count"] == 2
    assert out["benchmark_count"] == 2  # MMLU + GSM8K
    assert out["evaluation_count"] == 15
    # `humanize_token_first_char` only first-char-cases each token; an
    # all-uppercase input like "MMLU" stays "MMLU" because position 0 is
    # already uppercase. Per spec 08 — TS-as-spec.
    assert out["popular_evals"] == [
        {"benchmark": "MMLU", "model_count": 2},  # both models have MMLU
        {"benchmark": "GSM8K", "model_count": 1},
    ]
    assert len(out["models"]) == 2
    # Each model is a post-adapter EvaluationCardData (TS shape).
    assert out["models"][0]["id"] == "openai/gpt-5"
    assert out["models"][0]["route_id"] == "openai__gpt-5"


def test_hf_developer_detail_to_list_entry_strips_models():
    """List-entry shape == summary shape minus `models[]`."""
    detail = {
        "developer": "anthropic",
        "models": [
            {
                "model_family_id": "anthropic/claude",
                "model_family_name": "Claude",
                "developer": "anthropic",
                "total_evaluations": 3,
                "benchmark_count": 1,
                "categories_covered": ["safety"],
                "score_summary": {"count": 3, "average": 0.9},
                "benchmark_names": ["RewardBench"],
                "variants": [],
            }
        ],
    }
    summary = parity_adapters.hf_developer_detail_to_summary(detail)
    list_entry = parity_adapters.hf_developer_detail_to_list_entry(detail)
    assert "models" in summary
    assert "models" not in list_entry
    # All other fields preserved.
    assert list_entry["developer"] == summary["developer"]
    assert list_entry["route_id"] == summary["route_id"]
    assert list_entry["popular_evals"] == summary["popular_evals"]


def test_hf_developer_detail_to_summary_popular_evals_caps_at_three():
    """`popular_evals` is the top-3 sorted by model_count desc."""
    detail = {
        "developer": "many-bench-co",
        "models": [
            {
                "model_family_id": "x/m",
                "model_family_name": "M",
                "developer": "x",
                "total_evaluations": 1,
                "benchmark_count": 5,
                "categories_covered": [],
                "score_summary": {"count": 1, "average": 0.5},
                "benchmark_names": ["A", "B", "C", "D", "E"],
                "variants": [],
            }
        ],
    }
    out = parity_adapters.hf_developer_detail_to_summary(detail)
    assert len(out["popular_evals"]) == 3


def test_sort_variants_orders_by_date_desc_then_count_desc():
    """`sortVariants` — date DESC, total_evaluations DESC, label ASC."""
    variants = [
        {"version_date": "2024-01-01", "total_evaluations": 5, "variant_label": "A"},
        {"version_date": "2024-06-01", "total_evaluations": 5, "variant_label": "B"},
        {"version_date": "2024-06-01", "total_evaluations": 9, "variant_label": "C"},
        {"version_date": None, "total_evaluations": 100, "variant_label": "Z"},
    ]
    out = parity_adapters._sort_variants(variants)
    # Newest date first; within same date, higher count first.
    assert out[0]["variant_label"] == "C"
    assert out[1]["variant_label"] == "B"
    assert out[2]["variant_label"] == "A"
    # Date-less variants sort last regardless of count.
    assert out[3]["variant_label"] == "Z"

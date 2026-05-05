"""Benchmark identity normalization for the refactor pipeline.

Stable routing/grouping uses underscore-normalized keys (``benchmark_*_key``,
``slice_key``, ``metric_key``). Human-facing strings use ``*_name`` fields and
``canonical_benchmark_display_name`` / ``join_display_name_parts`` from
``scripts.helpers.presentation``.
"""

from __future__ import annotations

import re
from typing import Any

from scripts.helpers.benchmark_constants import (
    COMMON_LANGUAGE_SUBSET_KEYS,
    DOMAIN_CATEGORY_MAP,
    SUMMARY_SCORE_LEAF_KEYS,
)
from scripts.helpers.benchmark_identity import (
    canonical_benchmark_family_key,
    is_agentic_benchmark_name,
    normalize_benchmark_key,
    normalize_subset_key,
)
from scripts.helpers.metric_constants import (
    BENCHMARK_DEFAULT_METRICS,
    FRONTEND_REGEX_CATEGORY_TOKENS,
)
from scripts.helpers.presentation import (
    canonical_benchmark_display_name,
    humanize_token_key,
    join_display_name_parts,
)
from scripts.helpers.slug_utils import as_string
from scripts.metadata.benchmark_cards import (
    benchmark_card_language_keys,
    compact_benchmark_key,
)
from scripts.metadata.metrics import (
    infer_metric_from_benchmark_card,
    infer_metric_from_benchmark_defaults,
    infer_metric_from_score_details,
    infer_metric_from_value,
    metric_namespace_component,
    split_metric_from_evaluation_description,
    split_metric_from_evaluation_name,
)
def infer_top_level_benchmark_name(benchmark: Any, benchmark_family_name: str) -> str:
    benchmark_key = normalize_benchmark_key(benchmark)
    if benchmark_key.startswith("helm_"):
        suffix = benchmark_key.split("_", 1)[1]
        return canonical_benchmark_display_name(suffix, fallback=suffix)
    if (
        benchmark_family_name
        and normalize_benchmark_key(benchmark_family_name) == benchmark_key
    ):
        return benchmark_family_name
    return canonical_benchmark_display_name(
        benchmark, fallback=benchmark or benchmark_family_name
    )


def infer_subset_slice_from_name(
    name: Any, benchmark: Any
) -> tuple[str | None, str | None]:
    text = as_string(name).strip()
    benchmark_key = normalize_benchmark_key(benchmark)
    if not text or not benchmark_key or "/" not in text:
        return None, None

    if text.count("/") != 1:
        return None, None

    prefix_raw, suffix_raw = text.split("/", 1)
    if normalize_benchmark_key(prefix_raw) != benchmark_key:
        return None, None

    suffix_text = suffix_raw.strip(" /")
    if not suffix_text:
        return None, None

    return normalize_benchmark_key(suffix_text), humanize_token_key(suffix_text)


def is_language_subset_name(name: Any, benchmark_card: dict | None) -> bool:
    if not benchmark_card:
        return False

    normalized = normalize_benchmark_key(name)
    compact = compact_benchmark_key(name)
    if not normalized and not compact:
        return False

    language_keys = benchmark_card_language_keys(benchmark_card)
    if normalized in language_keys or compact in language_keys:
        return True
    return (
        normalized in COMMON_LANGUAGE_SUBSET_KEYS
        or compact in COMMON_LANGUAGE_SUBSET_KEYS
    )


def top_level_benchmark_owns_slices(
    benchmark: Any, benchmark_card: dict | None
) -> bool:
    benchmark_key = normalize_benchmark_key(benchmark)
    if benchmark_card:
        return True
    if benchmark_key in {
        normalize_benchmark_key(key) for key in BENCHMARK_DEFAULT_METRICS
    }:
        return True
    if benchmark_key.startswith("helm_"):
        return True
    return False


def infer_benchmark_leaf_and_slice(
    evaluation: dict,
    result: dict,
    benchmark_family_key: str,
    benchmark_family_name: str,
    component_key: str | None,
    component_name: str | None,
    benchmark_card: dict | None,
) -> tuple[str, str, str | None, str | None]:
    benchmark = as_string(evaluation.get("benchmark"))
    source_data = (
        result.get("source_data") if isinstance(result.get("source_data"), dict) else {}
    )
    dataset_name = as_string((source_data or {}).get("dataset_name"))
    raw_name = as_string(result.get("evaluation_name")).strip()
    raw_name_key = normalize_benchmark_key(raw_name)
    dataset_key = normalize_benchmark_key(dataset_name)
    top_level_key = normalize_benchmark_key(benchmark or dataset_name)
    top_level_name = infer_top_level_benchmark_name(
        benchmark or dataset_name, benchmark_family_name
    )

    subset_key, subset_name = infer_subset_slice_from_name(
        dataset_name or raw_name, benchmark or dataset_name
    )
    if subset_key and subset_name:
        return top_level_key, top_level_name, subset_key, subset_name

    paren_match = re.match(r"^(.+?)\s*\(([^()]+)\)\s*$", dataset_name or raw_name or "")
    if paren_match:
        paren_prefix_raw = paren_match.group(1).strip()
        paren_suffix_raw = paren_match.group(2).strip()
        paren_prefix_key = normalize_benchmark_key(paren_prefix_raw)
        if (
            paren_prefix_key
            and paren_suffix_raw
            and top_level_key
            and (
                paren_prefix_key == top_level_key
                or paren_prefix_key.startswith(top_level_key + "_")
                or top_level_key.startswith(paren_prefix_key + "_")
            )
        ):
            slice_key = normalize_subset_key(paren_suffix_raw)
            slice_name = humanize_token_key(paren_suffix_raw)
            if slice_key:
                leaf_name = canonical_benchmark_display_name(
                    paren_prefix_raw,
                    fallback=paren_prefix_raw,
                )
                return paren_prefix_key, leaf_name, slice_key, slice_name

    if raw_name and raw_name_key and dataset_key and raw_name_key == dataset_key:
        canonical_raw_name = canonical_benchmark_display_name(
            raw_name,
            benchmark or dataset_name,
            benchmark_family_name,
            fallback=top_level_name,
        )
        if (
            is_language_subset_name(raw_name, benchmark_card)
            and raw_name_key != top_level_key
        ):
            return top_level_key, top_level_name, raw_name_key, raw_name
        return raw_name_key, canonical_raw_name, None, None

    if component_name:
        component_summary_key = normalize_benchmark_key(component_name)
        if component_summary_key in SUMMARY_SCORE_LEAF_KEYS:
            return top_level_key, top_level_name, None, None
        if top_level_benchmark_owns_slices(
            benchmark or dataset_name, benchmark_card
        ) or is_language_subset_name(component_name, benchmark_card):
            if (
                component_key == top_level_key
                or normalize_benchmark_key(component_name) == top_level_key
            ):
                return top_level_key, top_level_name, None, None
            return top_level_key, top_level_name, component_key, component_name
        return (
            component_key or normalize_benchmark_key(component_name),
            component_name,
            None,
            None,
        )

    return top_level_key, top_level_name, None, None


def classify_evaluation_result(
    evaluation: dict, result: dict, benchmark_card: dict | None
) -> dict:
    benchmark = as_string(evaluation.get("benchmark"))
    source_data = (
        result.get("source_data") if isinstance(result.get("source_data"), dict) else {}
    )
    dataset_name = as_string((source_data or {}).get("dataset_name"))
    benchmark_family_key = canonical_benchmark_family_key(benchmark or dataset_name)
    benchmark_card_name = as_string(
        ((benchmark_card or {}).get("benchmark_details") or {}).get("name")
    )
    benchmark_family_name = (
        canonical_benchmark_display_name(
            benchmark_family_key,
            benchmark_card_name,
            benchmark,
            dataset_name,
            fallback=benchmark_family_key or benchmark or dataset_name,
        )
        or "Unknown Benchmark"
    )
    raw_name = as_string(result.get("evaluation_name")).strip()
    benchmark_keys = [
        candidate
        for candidate in {
            normalize_benchmark_key(benchmark),
            normalize_benchmark_key(dataset_name),
            benchmark_family_key,
        }
        if candidate
    ]

    metric_config = (
        result.get("metric_config")
        if isinstance(result.get("metric_config"), dict)
        else {}
    )
    metric = None
    metric_source = "unknown"
    component_name = None
    component_key = None
    raw_name_consumed_as_metric = False

    explicit_metric = infer_metric_from_value(
        metric_name=metric_config.get("metric_name"),
        metric_id=metric_config.get("metric_id"),
    )
    if explicit_metric:
        metric = explicit_metric
        metric_source = "metric_config"
        component_name, component_key = metric_namespace_component(
            metric["metric_id"], benchmark_family_key
        )
        split_metric = split_metric_from_evaluation_name(raw_name, benchmark_keys)
        if (
            split_metric
            and split_metric["metric"]["metric_key"] == metric["metric_key"]
        ):
            if not component_name and not component_key:
                component_name = split_metric["component_name"]
                component_key = split_metric["component_key"]
            raw_name_consumed_as_metric = (
                split_metric["component_name"] is None
                and split_metric["component_key"] is None
            )

    if metric is None:
        split_metric = split_metric_from_evaluation_name(raw_name, benchmark_keys)
        if split_metric:
            metric = split_metric["metric"]
            metric_source = split_metric["metric_source"]
            component_name = split_metric["component_name"]
            component_key = split_metric["component_key"]
            raw_name_consumed_as_metric = component_name is None

    if metric is None:
        metric = split_metric_from_evaluation_description(
            metric_config.get("evaluation_description")
        )
        if metric:
            metric_source = "evaluation_description"

    if metric is None:
        metric = infer_metric_from_benchmark_card(benchmark_card)
        if metric:
            metric_source = "benchmark_card"

    if metric is None:
        metric = infer_metric_from_benchmark_defaults(benchmark_family_key)
        if metric:
            metric_source = "benchmark_default"

    if metric is None:
        metric = infer_metric_from_score_details(result)
        if metric:
            metric_source = "score_details"

    if metric is None:
        metric = {
            "metric_name": "Score",
            "metric_id": "score",
            "metric_key": "score",
        }
        metric_source = "fallback"

    raw_name_key = normalize_benchmark_key(raw_name)
    if (
        raw_name
        and not component_name
        and not raw_name_consumed_as_metric
        and raw_name_key
        and raw_name_key not in benchmark_keys
        and raw_name_key != metric["metric_key"]
    ):
        component_name = raw_name
        component_key = raw_name_key

    if component_name and not component_key:
        component_key = normalize_benchmark_key(component_name)

    benchmark_leaf_key, benchmark_leaf_name, slice_key, slice_name = (
        infer_benchmark_leaf_and_slice(
            evaluation,
            result,
            benchmark_family_key or normalize_benchmark_key(benchmark or dataset_name),
            benchmark_family_name,
            component_key,
            component_name,
            benchmark_card,
        )
    )

    parent_key = normalize_benchmark_key(benchmark or dataset_name)
    is_summary_score = (
        bool(benchmark_leaf_key)
        and benchmark_leaf_key in SUMMARY_SCORE_LEAF_KEYS
        and benchmark_leaf_key != parent_key
    )
    benchmark_parent_name = canonical_benchmark_display_name(
        parent_key,
        benchmark,
        dataset_name,
        benchmark_card_name,
        fallback=benchmark or dataset_name,
    )
    display_name = join_display_name_parts(component_name, metric["metric_name"])
    if not display_name:
        display_name = (
            benchmark_leaf_name or benchmark_parent_name or benchmark_family_name
        )
    canonical_display_name = join_display_name_parts(
        benchmark_leaf_name or benchmark_parent_name or benchmark_family_name,
        slice_name,
        metric["metric_name"],
    )
    if not canonical_display_name:
        canonical_display_name = display_name

    return {
        "benchmark_family_key": benchmark_family_key or parent_key,
        "benchmark_family_name": benchmark_family_name,
        "benchmark_parent_key": parent_key,
        "benchmark_parent_name": benchmark_parent_name,
        "benchmark_component_key": component_key,
        "benchmark_component_name": component_name,
        "benchmark_leaf_key": benchmark_leaf_key,
        "benchmark_leaf_name": benchmark_leaf_name,
        "slice_key": slice_key,
        "slice_name": slice_name,
        "metric_name": metric["metric_name"],
        "metric_id": metric["metric_id"],
        "metric_key": metric["metric_key"],
        "metric_source": metric_source,
        "display_name": display_name,
        "canonical_display_name": canonical_display_name,
        "raw_evaluation_name": raw_name or None,
        "is_summary_score": is_summary_score,
    }


def infer_category_from_benchmark(
    benchmark_name: str, benchmark_card: dict | None = None
) -> str:
    def _frontend_regex_category(benchmark_name: str) -> str | None:
        if not benchmark_name:
            return None
        text = str(benchmark_name).lower()
        for category, tokens in FRONTEND_REGEX_CATEGORY_TOKENS:
            if any(token in text for token in tokens):
                return category
        return None

    if benchmark_card:
        domains = (benchmark_card.get("benchmark_details") or {}).get("domains") or []
        for domain in domains:
            domain_lower = domain.lower()
            if domain_lower in DOMAIN_CATEGORY_MAP:
                return DOMAIN_CATEGORY_MAP[domain_lower]
            for keyword, category in DOMAIN_CATEGORY_MAP.items():
                if keyword in domain_lower:
                    return category

    key = normalize_benchmark_key(benchmark_name)
    if not key:
        return "other"
    if is_agentic_benchmark_name(key):
        return "agentic"
    if re.search(r"(global_mmlu_lite|boolq|medqa|legalbench|quac|cnn_dailymail)", key):
        return "knowledge"
    if re.search(r"(reward_bench)", key):
        return "general"
    if re.search(r"(math|gsm|gpqa|mmlu|hellaswag|musr)", key):
        return "reasoning"
    if re.search(r"(ifeval)", key):
        return "instruction_following"
    if re.search(r"(hfopenllm|helm)", key):
        return "general"

    frontend = _frontend_regex_category(benchmark_name)
    if frontend is not None:
        return frontend
    return "other"

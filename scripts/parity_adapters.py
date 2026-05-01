"""TS-adapter parity ports.

These produce the exact post-TS-processing shapes the frontend currently
emits. Per PLAN_20260428.md line 55: "The payload must represent the
object after current TypeScript processing, not the raw backend object."

Each function in this module mirrors a TS adapter in
``general-eval-card/lib/`` and reproduces its output verbatim, including
hardcoded defaults that exist only because the TS-active paths never
populate the corresponding values.

  - ``hf_model_card_to_evaluation_card_data``: lib/model-data.ts:354
  - ``hf_eval_entry_to_list_item``:            lib/model-data.ts:421
  - ``hf_eval_detail_to_summary``:             lib/model-data.ts:742
  - ``create_model_family_summary``:           lib/eval-processing.ts:336

Helpers are imported from ``scripts.parity`` — slug, identity,
benchmark-display-name, license shorten, params parse, etc. The
adapters do NOT re-derive what those helpers already provide.
"""
from __future__ import annotations

import math
from typing import Any

from scripts import parity


# ---------------------------------------------------------------------------
# Shared helpers — mirror the small TS utilities that aren't in parity.py
# ---------------------------------------------------------------------------


def _slugify_eval_id(value: Any) -> str:
    """Mirror `lib/model-data.ts:62 slugifyEvalId`. Lowercase, replace
    non-alphanumerics with `_`, strip edge underscores."""
    if value is None:
        return ""
    import re

    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def get_aggregate_eval_id(value: Any) -> str:
    """Mirror `lib/model-data.ts:66 getAggregateEvalId`."""
    return f"aggregate__{_slugify_eval_id(value)}"


def _normalize_model_id_for_lookup(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _canonical_instance_results_url(value: Any) -> str | None:
    """Mirror `lib/hf-data.ts:1157-1165 getCanonicalInstanceResultsUrl`.

    Returns the URL only when it points at an instances/ artifact under
    the production card_backend dataset; otherwise ``None``. The TS guard
    blocks legacy / cross-dataset URLs from leaking into the eval-detail
    surface.
    """
    if not isinstance(value, str):
        return None
    if "/datasets/evaleval/card_backend/" in value and "/instances/" in value:
        return value
    return None


def _coalesce(*values: Any) -> Any:
    """JS `??`-chain — first non-None value."""
    for value in values:
        if value is not None:
            return value
    return None


def _locale_compare_key(value: Any) -> tuple:
    """Approximate JS `String.prototype.localeCompare` for ASCII inputs.

    JS default-locale (Unicode Collation Algorithm) sorts:
      - Primary: case-insensitive alphabetical (`casefold`).
      - Tertiary tie-break: position-by-position lowercase-before-uppercase
        (so `"Ita" < "ITA"` because position 1 is lowercase in `Ita`).
      - Quaternary: codepoint order as final fallback.

    Python's default codepoint sort does the opposite at every level
    (`"ITA" < "Ita"`), so we encode the UCA semantics explicitly:
      - first key: ``casefold`` (ASCII-equivalent of UCA primary)
      - second key: per-position ``isupper`` flags (lowercase=False<True)
      - third key: original string (codepoint tiebreak).
    """
    text = str(value)
    return (text.casefold(), tuple(c.isupper() for c in text), text)


def _to_summary_metric_config(metric: dict) -> dict:
    """Port of `toSummaryMetricConfig` (lib/model-data.ts:503-532)."""
    raw = metric.get("metric_config") or {}
    description = (
        (isinstance(raw.get("evaluation_description"), str) and raw["evaluation_description"])
        or (isinstance(raw.get("metric_name"), str) and raw["metric_name"])
        or metric.get("metric_name")
        or metric.get("display_name")
        or metric.get("evaluation_name")
        or ""
    )
    if isinstance(raw.get("unit"), str):
        unit = raw["unit"]
    elif isinstance(raw.get("metric_unit"), str):
        unit = raw["metric_unit"]
    else:
        unit = None

    score_type_raw = raw.get("score_type")
    score_type = (
        score_type_raw if score_type_raw in {"binary", "discrete"} else "continuous"
    )
    min_score = raw.get("min_score") if isinstance(raw.get("min_score"), (int, float)) else 0
    max_score = raw.get("max_score") if isinstance(raw.get("max_score"), (int, float)) else 1
    out = {
        "evaluation_description": description,
        "lower_is_better": bool(metric.get("lower_is_better")),
        "score_type": score_type,
        "min_score": min_score,
        "max_score": max_score,
    }
    if unit is not None:
        out["unit"] = unit
    return out


def _to_benchmark_summary_metric(metric: dict) -> dict:
    """Port of `toBenchmarkSummaryMetric` (lib/model-data.ts:534-553)."""
    metric_config = _to_summary_metric_config(metric)
    scores: list[float] = []
    for result in metric.get("model_results") or []:
        score = result.get("score")
        if score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score_f):
            scores.append(score_f)
    metric_name = (
        metric.get("metric_name")
        or metric.get("evaluation_name")
        or metric.get("display_name")
        or "Metric"
    )
    display_name = (
        metric.get("display_name")
        or metric.get("metric_name")
        or metric.get("evaluation_name")
        or metric_name
    )
    if scores:
        top_score: float | None = (
            min(scores) if metric.get("lower_is_better") else max(scores)
        )
    else:
        top_score = None
    return {
        "metric_summary_id": metric.get("metric_summary_id"),
        "metric_name": metric_name,
        "display_name": display_name,
        "canonical_display_name": metric.get("canonical_display_name"),
        "metric_key": metric.get("metric_key"),
        "lower_is_better": bool(metric.get("lower_is_better")),
        "models_count": len(metric.get("model_results") or []),
        "top_score": top_score,
        "unit": metric_config.get("unit"),
    }


def _extract_detail_subtasks(detail: dict) -> list[dict]:
    """Port of `extractDetailSubtasks` (lib/model-data.ts:555-587)."""
    out: list[dict] = []
    subtasks = detail.get("subtasks") or []
    if not isinstance(subtasks, list):
        return []
    for subtask in subtasks:
        if not isinstance(subtask, dict):
            continue
        metrics = subtask.get("metrics")
        metrics = metrics if isinstance(metrics, list) else []
        subtask_key = (
            (isinstance(subtask.get("subtask_key"), str) and subtask["subtask_key"])
            or (isinstance(subtask.get("display_name"), str) and _slugify_eval_id(subtask["display_name"]))
            or "subtask"
        )
        subtask_name = (
            (isinstance(subtask.get("subtask_name"), str) and subtask["subtask_name"])
            or (isinstance(subtask.get("display_name"), str) and subtask["display_name"])
            or "Subtask"
        )
        display_name = (
            (isinstance(subtask.get("display_name"), str) and subtask["display_name"])
            or (isinstance(subtask.get("subtask_name"), str) and subtask["subtask_name"])
            or "Subtask"
        )
        canonical = subtask.get("canonical_display_name")
        out.append(
            {
                "subtask_key": subtask_key,
                "subtask_name": subtask_name,
                "display_name": display_name,
                "canonical_display_name": canonical if isinstance(canonical, str) else None,
                "metrics": metrics,
            }
        )
    return out


def _extract_benchmark_subtasks(detail: dict) -> list[dict]:
    out: list[dict] = []
    for subtask in _extract_detail_subtasks(detail):
        out.append(
            {
                "subtask_key": subtask["subtask_key"],
                "subtask_name": subtask["subtask_name"],
                "display_name": subtask["display_name"],
                "canonical_display_name": subtask["canonical_display_name"],
                "metrics": [_to_benchmark_summary_metric(m) for m in subtask["metrics"]],
            }
        )
    return out


def _build_benchmark_leaderboard_matrix(detail: dict) -> dict:
    """Port of `buildBenchmarkLeaderboardMatrix` (lib/model-data.ts:599-697)."""
    benchmark_key = detail.get("benchmark") or ""
    source_data = detail.get("source_data") or {"dataset_name": benchmark_key}
    leaderboard_metrics: list[dict] = []
    row_states: dict[str, dict] = {}

    def register(metric: dict, scope: str, subtask: dict | None = None) -> None:
        summary_metric = _to_benchmark_summary_metric(metric)
        metric_token = (
            summary_metric.get("metric_summary_id")
            or summary_metric.get("metric_key")
            or _slugify_eval_id(summary_metric.get("display_name") or "")
        )
        column_parts = [scope]
        if subtask:
            column_parts.append(str(subtask.get("subtask_key") or ""))
        column_parts.append(str(metric_token or ""))
        column_key = ":".join(p for p in column_parts if p)
        leaderboard_metrics.append(
            {
                "column_key": column_key,
                "metric_summary_id": summary_metric.get("metric_summary_id"),
                "metric_name": summary_metric.get("metric_name"),
                "display_name": summary_metric.get("display_name"),
                "canonical_display_name": summary_metric.get("canonical_display_name"),
                "lower_is_better": summary_metric.get("lower_is_better"),
                "unit": summary_metric.get("unit"),
                "scope": scope,
                "subtask_key": subtask.get("subtask_key") if subtask else None,
                "subtask_name": subtask.get("subtask_name") if subtask else None,
            }
        )
        for model_result in metric.get("model_results") or []:
            model_id = model_result.get("model_id") or model_result.get("model_name")
            if not model_id:
                continue
            ts = parity.normalize_eval_timestamp(model_result.get("retrieved_timestamp") or "")
            existing = row_states.get(model_id)
            if existing is None:
                row_states[model_id] = {
                    "model_info": {
                        "name": model_result.get("model_name") or "",
                        "id": model_id,
                        "developer": model_result.get("developer") or "",
                    },
                    "model_route_id": model_result.get("model_route_id"),
                    "evaluation_timestamp": model_result.get("retrieved_timestamp") or "",
                    "source_metadata": model_result.get("source_metadata"),
                    "source_data": source_data,
                    "values": {column_key: model_result.get("score")},
                    "_ts": ts,
                }
                continue
            existing["values"][column_key] = model_result.get("score")
            if not existing.get("model_route_id") and model_result.get("model_route_id"):
                existing["model_route_id"] = model_result.get("model_route_id")
            if ts >= existing["_ts"]:
                existing["evaluation_timestamp"] = (
                    model_result.get("retrieved_timestamp") or existing["evaluation_timestamp"]
                )
                existing["_ts"] = ts

    for metric in detail.get("metrics") or []:
        register(metric, "root")
    for subtask in _extract_detail_subtasks(detail):
        for metric in subtask["metrics"]:
            register(
                metric,
                "subtask",
                {
                    "subtask_key": subtask["subtask_key"],
                    "subtask_name": subtask.get("display_name") or subtask["subtask_name"],
                },
            )

    leaderboard_rows: list[dict] = []
    for state in row_states.values():
        ts = state.pop("_ts", None)  # noqa: F841 — drop internal field
        metrics_present = sum(
            1
            for m in leaderboard_metrics
            if isinstance(state["values"].get(m["column_key"]), (int, float))
        )
        leaderboard_rows.append({**state, "metrics_present": metrics_present})

    return {
        "leaderboard_metrics": leaderboard_metrics,
        "leaderboard_rows": leaderboard_rows,
    }


def _to_model_results_for_metric(detail: dict, metric: dict) -> list[dict]:
    """Port of `toModelResultsForMetric` (lib/model-data.ts:699-740)."""
    benchmark_key = detail.get("benchmark") or ""
    metric_config = _to_summary_metric_config(metric)
    out: list[dict] = []
    for mr in metric.get("model_results") or []:
        evaluation_timestamp = mr.get("retrieved_timestamp") or ""
        model_info = {
            "name": mr.get("model_name") or "",
            "id": mr.get("model_id") or "",
            "developer": mr.get("developer") or "",
        }
        evaluation_result = {
            "evaluation_name": (
                metric.get("metric_name")
                or metric.get("evaluation_name")
                or metric.get("display_name")
                or ""
            ),
            "display_name": (
                metric.get("display_name")
                or metric.get("metric_name")
                or metric.get("evaluation_name")
            ),
            "canonical_display_name": metric.get("canonical_display_name"),
            "metric_summary_id": metric.get("metric_summary_id"),
            "metric_key": metric.get("metric_key"),
            "evaluation_timestamp": evaluation_timestamp,
            "metric_config": metric_config,
            "score_details": {"score": _coalesce(mr.get("score"), 0)},
            "detailed_evaluation_results_url": _canonical_instance_results_url(
                mr.get("detailed_evaluation_results")
            ),
        }
        out.append(
            {
                "model_info": model_info,
                "model_route_id": mr.get("model_route_id"),
                "score": _coalesce(mr.get("score"), 0),
                "score_details": {"score": _coalesce(mr.get("score"), 0)},
                "evaluation_timestamp": evaluation_timestamp,
                "source_metadata": mr.get("source_metadata"),
                "source_data": detail.get("source_data") or {"dataset_name": benchmark_key},
                "result": evaluation_result,
            }
        )
    return out


def _flat_metric_names(leaderboard_metrics: list[dict]) -> list[str]:
    out: list[str] = []
    for m in leaderboard_metrics:
        if m.get("scope") == "subtask" and m.get("subtask_name"):
            out.append(f"{m['subtask_name']} / {m.get('metric_name')}")
        else:
            name = m.get("metric_name")
            if name:
                out.append(name)
    return out


# ---------------------------------------------------------------------------
# flatten_model_evaluations (lib/hf-data.ts:1384)
#
# Hierarchy → flat ``BenchmarkEvaluation[]``. Output feeds
# ``create_model_family_summary`` and the long-form ``model_results``
# parquet table (one row per `(eval_summary_id, variant_key,
# evaluation_metric)` after the variant-bucket merge).
# ---------------------------------------------------------------------------


def _build_variant_lookup(detail: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for variant in detail.get("variants") or []:
        for raw_id in variant.get("raw_model_ids") or []:
            key = _normalize_model_id_for_lookup(raw_id)
            if key:
                out[key] = {
                    "variantKey": variant.get("variant_key") or "default",
                    "variantLabel": variant.get("variant_label") or "Default",
                }
    return out


def _resolve_variant_meta(detail: dict, variant_lookup: dict, result: dict) -> dict:
    """Mirror `lib/hf-data.ts:1081-1108 resolveVariantMeta`.

    Uses NORMALIZED candidates (lowercase, trimmed) for both the lookup
    AND the fallback variant_key — TS pattern via `normalizeModelIdForLookup`.
    """
    candidates_normalized = [
        _normalize_model_id_for_lookup(result.get("raw_model_id")),
        _normalize_model_id_for_lookup(result.get("model_id")),
    ]
    for normalized in candidates_normalized:
        if normalized and normalized in variant_lookup:
            return variant_lookup[normalized]
    variants = detail.get("variants") or []
    if len(variants) == 1:
        return {
            "variantKey": variants[0].get("variant_key") or "default",
            "variantLabel": variants[0].get("variant_label") or "Default",
        }
    fallback_key = next((c for c in candidates_normalized if c), None)
    return {
        "variantKey": fallback_key
        or (detail.get("model_info") or {}).get("variant_key")
        or "default",
        "variantLabel": (detail.get("model_info") or {}).get("variant_label") or "Default",
    }


def _belongs_to_model_family(detail: dict, result: dict, raw_model_ids: set[str]) -> bool:
    detail_route = _normalize_model_id_for_lookup(detail.get("model_route_id"))
    if detail_route and _normalize_model_id_for_lookup(result.get("model_route_id")) == detail_route:
        return True
    for field in ("raw_model_id", "model_id"):
        normalized = _normalize_model_id_for_lookup(result.get(field))
        if normalized and normalized in raw_model_ids:
            return True
    return False


def _build_model_info_for_variant(detail: dict, result: dict, variant_meta: dict) -> dict:
    """Mirror `lib/hf-data.ts:1133-1155 buildModelInfoForVariant`."""
    detail_info = detail.get("model_info") or {}
    model_id = _coalesce(
        result.get("raw_model_id"),
        result.get("model_id"),
        detail_info.get("id"),
    )
    # TS uses `||` (truthy-coalesce) on the model_name fallback chain.
    model_name = (
        result.get("model_name")
        or detail.get("model_family_name")
        or detail_info.get("name")
    )
    label = variant_meta.get("variantLabel")
    if label and label != "Default":
        version = label
    else:
        version = None
    additional_details = dict(detail_info.get("additional_details") or {})
    additional_details["raw_model_id"] = _coalesce(
        result.get("raw_model_id"), result.get("model_id")
    )
    return {
        **detail_info,
        "id": model_id,
        # TS: `result.developer || detail.model_info.developer` — truthy-coalesce.
        "name": model_name,
        "developer": result.get("developer") or detail_info.get("developer"),
        "model_version": version,
        "additional_details": additional_details,
    }


def _build_flatten_context(node: dict, inherited: dict | None) -> dict:
    benchmark = node.get("benchmark") or (inherited or {}).get("benchmark")
    family_name = node.get("benchmark_family_name") or (inherited or {}).get("benchmark_family_name")
    display = (
        node.get("display_name")
        or node.get("subtask_name")
        or node.get("benchmark_leaf_name")
        or (inherited or {}).get("benchmark_leaf_name")
        or family_name
        or benchmark
        or "Unknown Benchmark"
    )
    canonical = (
        node.get("canonical_display_name")
        or (inherited or {}).get("canonical_display_name")
        or display
    )
    source = (
        node.get("source_data")
        or (inherited or {}).get("sourceData")
        or {"dataset_name": benchmark or display}
    )
    return {
        "eval_summary_id": node.get("eval_summary_id") or (inherited or {}).get("eval_summary_id"),
        "benchmark": benchmark,
        "display_name": display,
        "canonical_display_name": canonical,
        "sourceData": source,
        "benchmark_family_key": node.get("benchmark_family_key") or (inherited or {}).get("benchmark_family_key"),
        "benchmark_family_name": family_name,
        "benchmark_parent_key": node.get("benchmark_parent_key") or (inherited or {}).get("benchmark_parent_key"),
        "benchmark_parent_name": node.get("benchmark_parent_name") or (inherited or {}).get("benchmark_parent_name"),
        "benchmark_leaf_key": node.get("benchmark_leaf_key") or (inherited or {}).get("benchmark_leaf_key"),
        "benchmark_leaf_name": node.get("benchmark_leaf_name") or (inherited or {}).get("benchmark_leaf_name"),
    }


def _flatten_hierarchy_node(
    detail: dict,
    node: dict,
    category: str,
    raw_model_ids: set[str],
    variant_lookup: dict,
    inherited: dict | None = None,
) -> list[dict]:
    out: list[dict] = []
    context = _build_flatten_context(node, inherited)
    source_data = context["sourceData"]

    for metric in node.get("metrics") or []:
        relevant = [
            r
            for r in (metric.get("model_results") or [])
            if _belongs_to_model_family(detail, r, raw_model_ids)
        ]
        if not relevant:
            continue
        results_by_variant: dict[str, dict] = {}
        for result in relevant:
            if not result.get("source_metadata"):
                # TS `assertSourceMetadata` throws here, but throwing crashes
                # the pipeline mid-run when even one EEE record is missing
                # source_metadata. Skip the row instead and emit a warning
                # so downstream consumers can still load the parquet.
                import sys as _sys

                _sys.stderr.write(
                    "[parity] WARN: skipping model_result missing source_metadata: "
                    f"model_family={detail.get('model_family_id')} "
                    f"metric={metric.get('metric_summary_id')}\n"
                )
                continue
            variant_meta = _resolve_variant_meta(detail, variant_lookup, result)
            variant_key = variant_meta.get("variantKey") or "default"
            model_info = _build_model_info_for_variant(detail, result, variant_meta)
            inline_samples = parity.parse_instance_level_data(result.get("instance_level_data"))
            evaluation_result = {
                "evaluation_name": (
                    metric.get("metric_name")
                    or metric.get("evaluation_name")
                    or metric.get("display_name")
                ),
                "display_name": (
                    metric.get("display_name")
                    or metric.get("metric_name")
                    or metric.get("evaluation_name")
                ),
                "canonical_display_name": (
                    metric.get("canonical_display_name")
                    or metric.get("display_name")
                    or f"{context.get('benchmark') or context.get('display_name') or 'Benchmark'} / {metric.get('metric_name')}"
                ),
                "metric_summary_id": metric.get("metric_summary_id"),
                "metric_key": metric.get("metric_key"),
                "evaluation_timestamp": result.get("retrieved_timestamp") or detail.get("last_updated") or "",
                "source_data": source_data,
                "metric_config": metric.get("metric_config"),
                "score_details": {"score": result.get("score")},
                "detailed_evaluation_results_url": _canonical_instance_results_url(
                    result.get("detailed_evaluation_results")
                ),
            }
            existing = results_by_variant.get(variant_key)
            if existing is None:
                results_by_variant[variant_key] = {
                    "modelInfo": model_info,
                    "evaluationResults": [evaluation_result],
                    "inlineSamples": inline_samples if inline_samples else None,
                    "latestTimestamp": result.get("retrieved_timestamp")
                    or detail.get("last_updated")
                    or "",
                    "sourceMetadata": result.get("source_metadata"),
                }
                continue
            existing["evaluationResults"].append(evaluation_result)
            if (not existing.get("inlineSamples")) and inline_samples:
                existing["inlineSamples"] = inline_samples
            new_ts = parity.to_comparable_timestamp(result.get("retrieved_timestamp"))
            old_ts = parity.to_comparable_timestamp(existing.get("latestTimestamp"))
            if new_ts >= old_ts:
                existing["latestTimestamp"] = (
                    result.get("retrieved_timestamp") or existing["latestTimestamp"]
                )
                existing["sourceMetadata"] = result.get("source_metadata")

        for variant_key, group in results_by_variant.items():
            slice_key = metric.get("slice_key") or node.get("subtask_key")
            slice_name = metric.get("slice_name") or node.get("subtask_name")
            display_name_value = (
                node.get("display_name")
                or node.get("subtask_name")
                or metric.get("slice_name")
                or context.get("display_name")
                or context.get("benchmark_leaf_name")
                or context.get("benchmark")
            )
            canonical_display_name = node.get("canonical_display_name")
            if not canonical_display_name:
                if metric.get("slice_name") and (
                    context.get("benchmark_parent_name") or context.get("benchmark")
                ):
                    canonical_display_name = (
                        f"{context.get('benchmark_parent_name') or context.get('benchmark')} / {metric.get('slice_name')}"
                    )
                else:
                    canonical_display_name = (
                        context.get("canonical_display_name") or context.get("benchmark")
                    )
            inline_samples = group.get("inlineSamples")
            out.append(
                {
                    "schema_version": "0.2.2",
                    "eval_summary_id": context.get("eval_summary_id"),
                    "evaluation_id": f"{metric.get('metric_summary_id')}__{variant_key}",
                    "retrieved_timestamp": group["latestTimestamp"],
                    "benchmark": context.get("benchmark"),
                    "display_name": display_name_value,
                    "canonical_display_name": canonical_display_name,
                    "category": category,
                    "benchmark_family_key": context.get("benchmark_family_key"),
                    "benchmark_family_name": context.get("benchmark_family_name"),
                    "benchmark_parent_key": context.get("benchmark_parent_key"),
                    "benchmark_parent_name": context.get("benchmark_parent_name"),
                    "benchmark_leaf_key": metric.get("benchmark_leaf_key") or context.get("benchmark_leaf_key"),
                    "benchmark_leaf_name": metric.get("benchmark_leaf_name") or context.get("benchmark_leaf_name"),
                    "slice_key": slice_key,
                    "slice_name": slice_name,
                    "source_data": source_data,
                    "source_metadata": group["sourceMetadata"],
                    "model_info": group["modelInfo"],
                    "evaluation_results": group["evaluationResults"],
                    "detailed_evaluation_results_per_samples": (
                        inline_samples if inline_samples else None
                    ),
                }
            )

    for subtask in node.get("subtasks") or []:
        out.extend(
            _flatten_hierarchy_node(
                detail, subtask, category, raw_model_ids, variant_lookup, context
            )
        )
    return out


def flatten_model_evaluations(detail: dict) -> list[dict]:
    """Port of ``flattenModelEvaluations`` (lib/hf-data.ts:1384).

    Walks ``hierarchy_by_category`` and emits one row per (eval, variant,
    metric-bucket) per spec reshape/03. The output is the canonical
    ``BenchmarkEvaluation[]`` shape every downstream call site (model
    family summary, score-summary stats) expects.
    """
    raw_ids: set[str] = set()
    for value in (
        list(detail.get("raw_model_ids") or [])
        + [rid for v in detail.get("variants") or [] for rid in v.get("raw_model_ids") or []]
        + [(detail.get("model_info") or {}).get("id")]
        + [detail.get("model_family_id")]
    ):
        normalized = _normalize_model_id_for_lookup(value)
        if normalized:
            raw_ids.add(normalized)

    variant_lookup = _build_variant_lookup(detail)
    out: list[dict] = []
    hierarchy = detail.get("hierarchy_by_category") or {}
    for category_key, nodes in hierarchy.items():
        mapped = parity.map_hf_categories([category_key])[0]
        for node in nodes or []:
            out.extend(
                _flatten_hierarchy_node(detail, node, mapped, raw_ids, variant_lookup)
            )
    return out


# ---------------------------------------------------------------------------
# 1. hfModelCardToEvaluationCardData (lib/model-data.ts:354)
# ---------------------------------------------------------------------------


def _model_card_average_score(entry: dict) -> float | None:
    score_summary = entry.get("score_summary") or {}
    if isinstance(score_summary.get("average"), (int, float)):
        return float(score_summary["average"])
    if isinstance(score_summary.get("avg"), (int, float)):
        return float(score_summary["avg"])
    return None


def _model_card_latest_timestamp(entry: dict) -> str | None:
    candidates: list[str] = []
    last_updated = entry.get("last_updated")
    if last_updated:
        candidates.append(str(last_updated))
    for variant in entry.get("variants") or []:
        if isinstance(variant, dict) and variant.get("last_updated"):
            candidates.append(str(variant["last_updated"]))
    if not candidates:
        return last_updated if last_updated else None
    candidates.sort(key=lambda v: parity.normalize_eval_timestamp(v), reverse=True)
    return candidates[0]


def _model_card_top_scores(entry: dict) -> list[dict]:
    raw = entry.get("top_benchmark_scores")
    if isinstance(raw, list) and raw:
        out: list[dict] = []
        for score in raw:
            if not isinstance(score, dict):
                continue
            try:
                score_value = float(score.get("score"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score_value):
                continue
            out.append(
                {
                    "benchmark": parity.get_benchmark_display_name(score.get("benchmark")),
                    "benchmarkKey": score.get("benchmarkKey"),
                    "score": score_value,
                    "metric": score.get("evaluation_name") or score.get("metric"),
                }
            )
        return out
    average = _model_card_average_score(entry)
    score_summary = entry.get("score_summary") or {}
    count = score_summary.get("count") or 0
    if average is None or not isinstance(count, (int, float)) or count <= 0:
        return []
    return [{"benchmark": "Average", "score": average, "metric": "Cross-benchmark average"}]


def hf_model_card_to_evaluation_card_data(entry: dict) -> dict:
    """Port of ``hfModelCardToEvaluationCardData`` (lib/model-data.ts:354).

    Produces the exact ``EvaluationCardData`` object the frontend renders
    on the model-card grid. Hardcoded defaults
    (``evaluator_count: 0``, ``source_types: ["documentation"]``,
    ``reproducibility_status: "partial"``, etc.) match the TS adapter
    verbatim — these exist on the TS side because the active path never
    populates them, and downstream UI code expects them present.
    """
    canonical_identity = parity.get_canonical_model_identity(
        {
            "id": entry.get("model_family_id"),
            "name": entry.get("model_family_name"),
        }
    )
    raw_categories = entry.get("categories_covered") or []
    categories = parity.map_hf_categories(raw_categories)

    average_score = _model_card_average_score(entry)
    top_scores = _model_card_top_scores(entry)

    total_evaluations = int(entry.get("total_evaluations") or 0)
    category_stats: dict[str, int] = {}
    if categories:
        per_cat = max(1, total_evaluations // len(categories))
        remaining = total_evaluations
        for i, cat in enumerate(categories):
            is_last = i == len(categories) - 1
            count = remaining if is_last else min(per_cat, remaining)
            category_stats[cat] = count
            remaining -= count

    benchmark_names_list = entry.get("benchmark_names") or []
    score_summary_in = entry.get("score_summary") or {}

    out: dict[str, Any] = {
        "id": canonical_identity["familyId"],
        "route_id": parity.get_model_family_route_id(canonical_identity["familyId"]),
        "model_name": canonical_identity["familyName"],
        "model_id": canonical_identity["familyId"],
        "canonical_model_name": canonical_identity["familyName"],
        # Registry-resolved canonical id from the upstream pipeline.
        # Distinct from `id` (which is the slug-derived family id from
        # the TS adapter): this is the evalcard-registry's canonical for
        # the model. None when no registry hit on this row.
        "canonical_model_id": entry.get("canonical_model_id"),
        "developer": parity.normalize_developer_name(entry.get("developer")),
        "evaluations_count": total_evaluations,
        # TS: `entry.benchmark_family_count || entry.benchmark_count` — no
        # final `|| 0` fallback. May surface as `undefined` when both fields
        # are missing/0; we preserve that by leaving as None.
        "benchmarks_count": (
            entry.get("benchmark_family_count") or entry.get("benchmark_count")
        ),
        "variant_count": len(entry.get("variants") or []),
        "categories": categories,
        "category_stats": category_stats,
        "latest_timestamp": _model_card_latest_timestamp(entry),
        "evaluator_count": 0,
        "evaluator_names": [],
        "source_type_count": 1,
        "source_types": ["documentation"],
        "evidence_count": total_evaluations,
        "missing_generation_config_count": 0,
        "third_party_eval_count": 0,
        "independent_verification_ratio": 0,
        "reproducibility_status": "partial",
        "eval_libraries": [],
        "latest_source_name": (
            f"{len(benchmark_names_list)} benchmark"
            f"{'' if len(benchmark_names_list) == 1 else 's'}"
            if benchmark_names_list
            else None
        ),
        "params_billions": parity.parse_params_billions(entry.get("params_billions")),
        "benchmark_names": [
            parity.get_benchmark_display_name(name) for name in benchmark_names_list
        ],
        "score_summary": {
            "count": score_summary_in.get("count"),
            "min": score_summary_in.get("min"),
            "max": score_summary_in.get("max"),
            "average": average_score,
        },
        "top_scores": top_scores,
        "source_urls": [],
        "detail_urls": [],
    }
    return out


# ---------------------------------------------------------------------------
# 2. hfEvalEntryToListItem (lib/model-data.ts:421)
# ---------------------------------------------------------------------------


def hf_eval_entry_to_list_item(entry: dict) -> dict:
    """Port of ``hfEvalEntryToListItem`` (lib/model-data.ts:421)."""
    raw_category = entry.get("category")
    mapped_categories = parity.map_hf_categories([raw_category] if raw_category else [])
    category = mapped_categories[0] if mapped_categories else "General"

    metrics = entry.get("metrics") or []
    primary_metric = next(
        (m for m in metrics if m.get("metric_name") == entry.get("primary_metric_name")),
        metrics[0] if metrics else None,
    )

    benchmark_display_name = parity.get_benchmark_display_name(
        entry.get("benchmark_parent_name") or entry.get("benchmark") or ""
    )
    raw_display_name = (
        entry.get("evaluation_name")
        or entry.get("display_name")
        or entry.get("benchmark_leaf_name")
        or entry.get("eval_summary_id")
    )
    top_score = entry.get("top_score")
    return {
        "evaluation_name": raw_display_name,
        "evaluation_id": entry.get("eval_summary_id"),
        "composite_benchmark_key": entry.get("benchmark") or "",
        "composite_benchmark_name": benchmark_display_name,
        "category": category,
        "metric_config": {
            "evaluation_description": entry.get("primary_metric_name"),
            "lower_is_better": (
                bool(primary_metric.get("lower_is_better")) if primary_metric else False
            ),
            "score_type": "continuous",
            "min_score": 0,
            "max_score": 1,
        },
        "models_count": entry.get("models_count"),
        "evaluator_names": [],
        "source_types": [],
        "latest_source_name": parity.get_benchmark_display_name(entry.get("benchmark")),
        "third_party_ratio": 0,
        "missing_generation_config_count": 0,
        "best_model": {"name": "", "score": top_score} if top_score is not None else None,
        "worst_model": None,
        "avg_score": 0,
        "avg_score_norm": 0,
        "benchmark_card": entry.get("benchmark_card"),
        "tags": entry.get("tags"),
        "metrics_count": entry.get("metrics_count"),
        "metric_names": entry.get("metric_names"),
        "instance_data": entry.get("instance_data"),
        "benchmark_family_key": entry.get("benchmark_family_key"),
        "benchmark_leaf_key": entry.get("benchmark_leaf_key"),
        "source_data": entry.get("source_data"),
        "top_score": top_score,
        "subtasks_count": entry.get("subtasks_count") or 0,
        "is_summary_score": bool(entry.get("is_summary_score")),
        "summary_eval_ids": entry.get("summary_eval_ids") or [],
    }


# ---------------------------------------------------------------------------
# 3. hfEvalDetailToSummary (lib/model-data.ts:742)
# ---------------------------------------------------------------------------


def hf_eval_detail_to_summary(detail: dict) -> dict:
    """Port of ``hfEvalDetailToSummary`` (lib/model-data.ts:742).

    Preserves spec-14 quirks: hardcoded ``evaluator_names: []``,
    ``source_types: []``, ``latest_source_name = display_name``,
    ``third_party_ratio: 0``, ``missing_generation_config_count: 0``;
    sorts the primary metric's results by score and uses
    ``avg_score_norm = avg_score`` for the (assumed-0-to-1) primary metric.
    """
    eval_name = detail.get("benchmark_leaf_name") or detail.get("eval_summary_id") or "Unknown"
    benchmark_key = detail.get("benchmark") or ""
    all_metrics = detail.get("metrics") or []
    root_metrics = [_to_benchmark_summary_metric(m) for m in all_metrics]
    subtasks = _extract_benchmark_subtasks(detail)
    leaderboard_matrix = _build_benchmark_leaderboard_matrix(detail)

    primary_metric: dict | None = all_metrics[0] if all_metrics else None
    if primary_metric is None:
        for subtask in detail.get("subtasks") or []:
            if isinstance(subtask, dict) and isinstance(subtask.get("metrics"), list):
                if subtask["metrics"]:
                    primary_metric = subtask["metrics"][0]
                    break

    bench_display = parity.get_benchmark_display_name(benchmark_key)
    base = {
        "evaluation_name": eval_name,
        "evaluation_id": detail.get("eval_summary_id"),
        "canonical_display_name": detail.get("canonical_display_name"),
        "composite_benchmark_key": benchmark_key,
        "composite_benchmark_name": bench_display,
        "category": parity.infer_category_from_benchmark(eval_name),
        "evaluator_names": [],
        "source_types": [],
        "latest_source_name": bench_display,
        "third_party_ratio": 0,
        "missing_generation_config_count": 0,
        "benchmark_card": detail.get("benchmark_card"),
        "metric_names": _flat_metric_names(leaderboard_matrix["leaderboard_metrics"]),
        "metrics_count": len(leaderboard_matrix["leaderboard_metrics"]),
        "root_metrics": root_metrics,
        "subtasks": subtasks,
        "leaderboard_metrics": leaderboard_matrix["leaderboard_metrics"],
        "leaderboard_rows": leaderboard_matrix["leaderboard_rows"],
    }

    if primary_metric is None:
        base.update(
            {
                "metric_config": {
                    "evaluation_description": "",
                    "lower_is_better": False,
                    "score_type": "continuous",
                },
                "model_results": [],
                "models_count": 0,
                "best_model": None,
                "worst_model": None,
                "avg_score": 0,
                "avg_score_norm": 0,
            }
        )
        return base

    metric_config = _to_summary_metric_config(primary_metric)
    model_results = _to_model_results_for_metric(detail, primary_metric)
    lower_is_better = bool(metric_config.get("lower_is_better"))

    def _row_score(r: dict) -> float:
        # `_to_model_results_for_metric` defaults missing scores to 0
        # (TS: `mr.score ?? 0`); we mirror that here. Critical: don't
        # use truthy `or` — a legitimate score of `0` would silently
        # become NaN/0 from the wrong branch and skew avg_score.
        score = r.get("score")
        if score is None:
            return 0.0
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    model_results.sort(key=_row_score, reverse=not lower_is_better)
    # `Number.isFinite` accepts 0; build the list explicitly so a row
    # with `score == 0` is INCLUDED in the average.
    finite_scores: list[float] = []
    for r in model_results:
        score = r.get("score")
        if score is None:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite_scores.append(value)
    avg_score = sum(finite_scores) / len(finite_scores) if finite_scores else 0
    base.update(
        {
            "metric_config": metric_config,
            "model_results": model_results,
            "models_count": len(model_results),
            "best_model": (
                {
                    "name": model_results[0]["model_info"]["name"],
                    "score": model_results[0]["score"],
                }
                if model_results
                else None
            ),
            "worst_model": (
                {
                    "name": model_results[-1]["model_info"]["name"],
                    "score": model_results[-1]["score"],
                }
                if model_results
                else None
            ),
            "avg_score": avg_score,
            "avg_score_norm": avg_score,
        }
    )
    return base


# ---------------------------------------------------------------------------
# 4. createModelFamilySummary (lib/eval-processing.ts:336)
# ---------------------------------------------------------------------------


def _aggregated_variant_descriptor(model_info: dict) -> dict:
    """Port of `getAggregatedVariantDescriptor` (lib/eval-processing.ts:277)."""
    identity = parity.get_canonical_model_identity(model_info or {})
    additional = (model_info or {}).get("additional_details") or {}
    raw_mode = additional.get("mode") if isinstance(additional, dict) else None
    setup_alias = (
        raw_mode.strip()
        if isinstance(raw_mode, str)
        and parity.is_setup_alias_qualifier(raw_mode)
        else None
    )

    if not setup_alias:
        return {
            "variantKey": identity["variantKey"],
            "variantLabel": identity["variantLabel"],
            "variantDisplayName": identity["variantDisplayName"],
            "familyId": identity["familyId"],
            "familyName": identity["familyName"],
            "versionDate": identity["versionDate"],
            "versionQualifier": identity["versionQualifier"],
            "mergedSetupAlias": False,
        }

    if identity.get("versionDate"):
        return {
            "variantKey": identity["versionDate"],
            "variantLabel": identity["versionDate"],
            "variantDisplayName": f"{identity['familyName']} ({identity['versionDate']})",
            "familyId": identity["familyId"],
            "familyName": identity["familyName"],
            "versionDate": identity["versionDate"],
            "versionQualifier": None,
            "mergedSetupAlias": True,
        }

    return {
        "variantKey": "base",
        "variantLabel": "Current",
        "variantDisplayName": identity["familyName"],
        "familyId": identity["familyId"],
        "familyName": identity["familyName"],
        "versionDate": None,
        "versionQualifier": None,
        "mergedSetupAlias": True,
    }


def _create_model_summary(evaluations: list[dict]) -> dict:
    """Port of ``createModelSummary`` (lib/eval-processing.ts:163-227).

    Bucket evaluations by category (regex fallback when ``eval.category``
    is absent), record latest timestamp, count results.
    """
    if not evaluations:
        raise ValueError("No evaluations provided")
    model_info = evaluations[0].get("model_info") or {}

    evaluations_by_category: dict[str, list[dict]] = {}
    categories_set: list[str] = []
    seen_cats: set[str] = set()

    def _add_category(category: str, evaluation: dict) -> None:
        if category not in seen_cats:
            seen_cats.add(category)
            categories_set.append(category)
        evaluations_by_category.setdefault(category, []).append(evaluation)

    for evaluation in evaluations:
        eval_categories: list[str] = []
        category = evaluation.get("category")
        if category:
            eval_categories.append(category)
        else:
            for result in evaluation.get("evaluation_results") or []:
                inferred = parity.infer_category_from_benchmark(
                    result.get("evaluation_name")
                )
                if inferred == "General":
                    source = evaluation.get("source_data")
                    if isinstance(source, dict):
                        inferred = parity.infer_category_from_benchmark(
                            source.get("dataset_name")
                        )
                if inferred not in eval_categories:
                    eval_categories.append(inferred)
        for cat in eval_categories:
            _add_category(cat, evaluation)

    timestamps_ms: list[float] = []
    for evaluation in evaluations:
        ts = evaluation.get("retrieved_timestamp")
        if ts is None:
            continue
        ms = parity.normalize_eval_timestamp(ts)
        if math.isfinite(ms):
            timestamps_ms.append(ms)
    if timestamps_ms:
        from datetime import datetime, timezone

        latest = max(timestamps_ms)
        dt = datetime.fromtimestamp(latest / 1000.0, tz=timezone.utc)
        # `Date.prototype.toISOString()` truncates microseconds → ms, doesn't round.
        millis = dt.microsecond // 1000
        last_updated = f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{millis:03d}Z"
    else:
        last_updated = None

    total_results = sum(
        len(e.get("evaluation_results") or []) for e in evaluations
    )

    return {
        "model_info": model_info,
        "evaluations_by_category": evaluations_by_category,
        "total_evaluations": total_results,
        "last_updated": last_updated,
        "categories_covered": categories_set,
    }


def _sort_variants(variants: list[dict]) -> list[dict]:
    def _date_value(variant: dict) -> float:
        version_date = variant.get("version_date")
        if not version_date:
            return float("-inf")
        ms = parity.normalize_eval_timestamp(version_date)
        return ms if math.isfinite(ms) else float("-inf")

    return sorted(
        variants,
        key=lambda v: (
            -_date_value(v),
            -int(v.get("total_evaluations") or 0),
            _locale_compare_key(v.get("variant_label") or ""),
        ),
    )


# ---------------------------------------------------------------------------
# 5. aggregateBenchmarkSummaries (lib/model-data.ts:868)
# ---------------------------------------------------------------------------


SYNTHETIC_MATRIX_EVAL_PREFIX = "matrix__"


def aggregate_benchmark_summaries(
    summaries: list[dict], aggregation_key: str
) -> dict | None:
    """Port of ``aggregateBenchmarkSummaries`` (lib/model-data.ts:868).

    Input ``summaries`` are the post-``hf_eval_detail_to_summary`` shape
    (i.e. ``BenchmarkEvalSummary[]``) — caller is responsible for running
    the per-detail adapter first.
    """
    if not summaries:
        return None
    first = summaries[0]
    card = first.get("benchmark_card")

    sources_dedup: dict[str, dict] = {}
    for summary in summaries:
        eval_id = summary.get("evaluation_id")
        if eval_id and eval_id not in sources_dedup:
            sources_dedup[eval_id] = {
                "evaluation_id": eval_id,
                "composite_benchmark_key": summary.get("composite_benchmark_key"),
                "composite_benchmark_name": summary.get("evaluation_name"),
                "models_count": summary.get("models_count"),
                "avg_score_norm": summary.get("avg_score_norm"),
            }
    # TS: `.sort((a, b) => a.composite_benchmark_name.localeCompare(b.composite_benchmark_name))`.
    # Default codepoint sort puts uppercase ahead of lowercase (`CNN` < `Civ`),
    # which disagrees with JS `localeCompare`'s case-folded primary comparison
    # (`Civ` < `CNN`). Use `_locale_compare_key` so positions match the TS
    # adapter exactly.
    aggregate_sources = sorted(
        sources_dedup.values(),
        key=lambda s: _locale_compare_key(s.get("composite_benchmark_name") or ""),
    )

    suite_display_name = parity.get_benchmark_display_name(aggregation_key)

    model_buckets: dict[str, dict] = {}
    for summary in summaries:
        for model_result in summary.get("model_results") or []:
            model_id = (model_result.get("model_info") or {}).get("id")
            if not model_id:
                continue
            slot = model_buckets.setdefault(
                model_id,
                {
                    "model_info": model_result.get("model_info"),
                    "components": [],
                },
            )
            slot["components"].append({"summary": summary, "modelResult": model_result})

    aggregate_metric_config = dict(first.get("metric_config") or {})
    aggregate_metric_config.update(
        {
            "evaluation_description": (
                f"Average normalized score across "
                f"{', '.join(s.get('composite_benchmark_name') or '' for s in aggregate_sources)}"
                if len(aggregate_sources) > 1
                else (first.get("metric_config") or {}).get("evaluation_description")
            ),
            "min_score": 0,
            "max_score": 1,
            "unit": "normalized average",
        }
    )

    def _component_score(summary: dict, model_result: dict) -> float:
        config = summary.get("metric_config") or {}
        return parity.normalize_summary_score(config, model_result.get("score"))

    aggregated_results: list[dict] = []
    for slot in model_buckets.values():
        components = slot["components"]
        normalized_scores = [
            _component_score(c["summary"], c["modelResult"]) for c in components
        ]
        avg_norm = sum(normalized_scores) / len(normalized_scores)

        latest_component = max(
            components,
            key=lambda c: parity.normalize_eval_timestamp(
                (c["modelResult"] or {}).get("evaluation_timestamp") or ""
            ),
        )

        aggregate_components = sorted(
            (
                {
                    "evaluation_id": c["summary"].get("evaluation_id"),
                    "composite_benchmark_key": c["summary"].get("composite_benchmark_key"),
                    "composite_benchmark_name": c["summary"].get("composite_benchmark_name"),
                    "score": c["modelResult"].get("score"),
                    "normalized_score": _component_score(c["summary"], c["modelResult"]),
                    "evaluation_timestamp": c["modelResult"].get("evaluation_timestamp"),
                    "source_name": ((c["modelResult"].get("source_metadata") or {}).get("source_name")),
                    "source_type": ((c["modelResult"].get("source_metadata") or {}).get("source_type")),
                    "source_organization_name": (
                        (c["modelResult"].get("source_metadata") or {}).get("source_organization_name")
                    ),
                    "evaluator_relationship": (
                        (c["modelResult"].get("source_metadata") or {}).get("evaluator_relationship")
                    ),
                }
                for c in components
            ),
            # TS: `.sort((a, b) => a.composite_benchmark_name.localeCompare(...))`.
            # See the parallel sort fix on `aggregate_sources` above; same
            # case-folded primary collation needed here so component order
            # matches the TS adapter.
            key=lambda c: _locale_compare_key(c.get("composite_benchmark_name") or ""),
        )

        sample_size_total = sum(
            ((c["modelResult"].get("score_details") or {}).get("sample_size") or 0)
            for c in components
        )
        latest_mr = latest_component["modelResult"]
        result_block = dict(latest_mr.get("result") or {})
        result_block.update(
            {
                "evaluation_name": (
                    ((card or {}).get("benchmark_details") or {}).get("name")
                    or first.get("evaluation_name")
                ),
                "metric_config": aggregate_metric_config,
                "score_details": {"score": avg_norm},
            }
        )

        aggregated_results.append(
            {
                "model_info": slot["model_info"],
                "score": avg_norm,
                "score_details": {
                    "score": avg_norm,
                    "sample_size": sample_size_total or None,
                },
                "evaluation_timestamp": latest_mr.get("evaluation_timestamp"),
                "source_metadata": latest_mr.get("source_metadata"),
                "source_data": latest_mr.get("source_data"),
                "result": result_block,
                "aggregate_components": aggregate_components,
            }
        )

    lower_is_better = bool((first.get("metric_config") or {}).get("lower_is_better"))
    aggregated_results.sort(key=lambda r: r["score"], reverse=not lower_is_better)
    avg_score = (
        sum(r["score"] for r in aggregated_results) / len(aggregated_results)
        if aggregated_results
        else 0
    )

    evaluator_names = sorted(
        {n for s in summaries for n in (s.get("evaluator_names") or [])}
    )
    source_types = sorted(
        {t for s in summaries for t in (s.get("source_types") or [])}
    )
    total_underlying = sum(
        len(s.get("model_results") or []) for s in summaries
    )
    total_third_party = sum(
        sum(
            1
            for r in (s.get("model_results") or [])
            if (r.get("source_metadata") or {}).get("evaluator_relationship") == "third_party"
        )
        for s in summaries
    )

    if len(aggregate_sources) == 1:
        composite_benchmark_key = aggregate_sources[0].get("composite_benchmark_key")
        latest_source_name = aggregate_sources[0].get("composite_benchmark_name")
    else:
        composite_benchmark_key = aggregation_key
        latest_source_name = "Multiple sources"

    return {
        "evaluation_name": suite_display_name,
        "evaluation_id": get_aggregate_eval_id(aggregation_key),
        "composite_benchmark_key": composite_benchmark_key,
        "composite_benchmark_name": suite_display_name,
        "category": first.get("category"),
        "metric_config": aggregate_metric_config,
        "model_results": aggregated_results,
        "models_count": len(aggregated_results),
        "evaluator_names": evaluator_names,
        "source_types": source_types,
        "latest_source_name": latest_source_name,
        "third_party_ratio": (total_third_party / total_underlying) if total_underlying else 0,
        "missing_generation_config_count": sum(
            int(s.get("missing_generation_config_count") or 0) for s in summaries
        ),
        "best_model": (
            {
                "name": (aggregated_results[0]["model_info"] or {}).get("name"),
                "score": aggregated_results[0]["score"],
            }
            if aggregated_results
            else None
        ),
        "worst_model": (
            {
                "name": (aggregated_results[-1]["model_info"] or {}).get("name"),
                "score": aggregated_results[-1]["score"],
            }
            if aggregated_results
            else None
        ),
        "avg_score": avg_score,
        "avg_score_norm": avg_score,
        "benchmark_card": card,
        "is_aggregated": True,
        "aggregate_sources": aggregate_sources,
    }


# ---------------------------------------------------------------------------
# 6. buildSingleMetricSuiteMatrixSummary (lib/model-data.ts:1039)
# ---------------------------------------------------------------------------


def build_single_metric_suite_matrix_summary(
    details: list[dict], suite_key: str
) -> dict | None:
    """Port of ``buildSingleMetricSuiteMatrixSummary`` (lib/model-data.ts:1039).

    Eligibility: each detail has exactly one metric and zero subtasks; need
    >=2 surviving sub-evals; column floor >=2.
    """
    if len(details) < 2:
        return None
    suite_display_name = parity.get_benchmark_display_name(suite_key)
    valid = sorted(
        [
            d
            for d in details
            if len(d.get("metrics") or []) == 1
            and len(_extract_detail_subtasks(d)) == 0
        ],
        # TS: `(left, right) => (left.benchmark_leaf_name || left.eval_summary_id)
        # .localeCompare(right.benchmark_leaf_name || right.eval_summary_id)`.
        # Default codepoint sort would put `CNN/DailyMail` before `CivilComments`;
        # JS localeCompare puts them the other way. Use `_locale_compare_key` so
        # the matrix's `leaderboard_metrics[i]` ordering matches the TS adapter.
        key=lambda d: _locale_compare_key(
            d.get("benchmark_leaf_name") or d.get("eval_summary_id") or ""
        ),
    )
    if len(valid) < 2:
        return None

    leaderboard_metrics: list[dict] = []
    row_states: dict[str, dict] = {}
    metric_config: dict | None = None
    benchmark_card: dict | None = None
    metric_names: set[str] = set()

    for detail in valid:
        metrics = detail.get("metrics") or []
        if not metrics:
            continue
        metric = metrics[0]
        if metric_config is None:
            metric_config = _to_summary_metric_config(metric)
        if benchmark_card is None and detail.get("benchmark_card"):
            benchmark_card = detail.get("benchmark_card")
        summary_metric = _to_benchmark_summary_metric(metric)
        if summary_metric.get("metric_name"):
            metric_names.add(summary_metric["metric_name"])
        subtask_key = (
            detail.get("benchmark_leaf_key")
            or _slugify_eval_id(detail.get("eval_summary_id") or "")
        )
        subtask_name = (
            detail.get("benchmark_leaf_name")
            or detail.get("canonical_display_name")
            or detail.get("eval_summary_id")
            or subtask_key
        )
        metric_token = (
            summary_metric.get("metric_summary_id")
            or summary_metric.get("metric_key")
            or _slugify_eval_id(summary_metric.get("display_name") or "")
        )
        column_key = ":".join(["subtask", str(subtask_key), str(metric_token)])
        leaderboard_metrics.append(
            {
                "column_key": column_key,
                "metric_summary_id": summary_metric.get("metric_summary_id"),
                "metric_name": summary_metric.get("metric_name"),
                "display_name": summary_metric.get("display_name"),
                "canonical_display_name": summary_metric.get("canonical_display_name"),
                "lower_is_better": summary_metric.get("lower_is_better"),
                "unit": summary_metric.get("unit"),
                "scope": "subtask",
                "subtask_key": subtask_key,
                "subtask_name": subtask_name,
            }
        )
        benchmark_key_local = detail.get("benchmark") or suite_key
        source_data = detail.get("source_data") or {"dataset_name": benchmark_key_local}

        for model_result in metric.get("model_results") or []:
            model_id = model_result.get("model_id") or model_result.get("model_name")
            if not model_id:
                continue
            ts = parity.normalize_eval_timestamp(model_result.get("retrieved_timestamp") or "")
            existing = row_states.get(model_id)
            if existing is None:
                row_states[model_id] = {
                    "model_info": {
                        "name": model_result.get("model_name") or "",
                        "id": model_id,
                        "developer": model_result.get("developer") or "",
                    },
                    "model_route_id": model_result.get("model_route_id"),
                    "evaluation_timestamp": model_result.get("retrieved_timestamp") or "",
                    "source_metadata": model_result.get("source_metadata"),
                    "source_data": source_data,
                    "values": {column_key: model_result.get("score")},
                    "_ts": ts,
                }
                continue
            existing["values"][column_key] = model_result.get("score")
            if not existing.get("model_route_id") and model_result.get("model_route_id"):
                existing["model_route_id"] = model_result.get("model_route_id")
            if ts >= existing["_ts"]:
                existing["evaluation_timestamp"] = (
                    model_result.get("retrieved_timestamp") or existing["evaluation_timestamp"]
                )
                existing["source_metadata"] = model_result.get("source_metadata")
                existing["source_data"] = source_data
                existing["_ts"] = ts

    if len(leaderboard_metrics) < 2:
        return None

    shared_metric_name = next(iter(metric_names)) if len(metric_names) == 1 else None
    if metric_config is None:
        suite_metric_config: dict = {
            "evaluation_description": shared_metric_name or "",
            "lower_is_better": False,
            "score_type": "continuous",
            "min_score": 0,
            "max_score": 1,
        }
    else:
        suite_metric_config = dict(metric_config)
        suite_metric_config["evaluation_description"] = (
            shared_metric_name or metric_config.get("evaluation_description")
        )

    leaderboard_rows: list[dict] = []
    for state in row_states.values():
        state.pop("_ts", None)
        metrics_present = sum(
            1
            for m in leaderboard_metrics
            if isinstance(state["values"].get(m["column_key"]), (int, float))
        )
        leaderboard_rows.append({**state, "metrics_present": metrics_present})

    return {
        "evaluation_name": suite_display_name,
        "evaluation_id": f"{SYNTHETIC_MATRIX_EVAL_PREFIX}{suite_key}",
        "canonical_display_name": suite_display_name,
        "composite_benchmark_key": suite_key,
        "composite_benchmark_name": suite_display_name,
        "category": parity.infer_category_from_benchmark(suite_display_name),
        "metric_config": suite_metric_config,
        "model_results": [],
        "models_count": len(leaderboard_rows),
        "evaluator_names": [],
        "source_types": [],
        "latest_source_name": suite_display_name,
        "third_party_ratio": 0,
        "missing_generation_config_count": 0,
        "best_model": None,
        "worst_model": None,
        "avg_score": 0,
        "avg_score_norm": 0,
        "benchmark_card": benchmark_card,
        "metrics_count": len(leaderboard_metrics),
        "metric_names": [
            f"{m.get('subtask_name')} / {m.get('metric_name')}" for m in leaderboard_metrics
        ],
        "source_data": {"dataset_name": suite_display_name},
        "leaderboard_metrics": leaderboard_metrics,
        "leaderboard_rows": leaderboard_rows,
    }


def create_model_family_summary(evaluations: list[dict]) -> dict:
    """Port of ``createModelFamilySummary`` (lib/eval-processing.ts:336)."""
    if not evaluations:
        raise ValueError("No evaluations provided")

    family_identity = parity.get_canonical_model_identity(evaluations[0].get("model_info") or {})

    variant_groups: dict[str, dict] = {}
    for evaluation in evaluations:
        descriptor = _aggregated_variant_descriptor(evaluation.get("model_info") or {})
        slot = variant_groups.setdefault(
            descriptor["variantKey"],
            {"descriptor": descriptor, "evaluations": []},
        )
        slot["evaluations"].append(evaluation)

    variants_unsorted: list[dict] = []
    for slot in variant_groups.values():
        descriptor = slot["descriptor"]
        variant_evals = slot["evaluations"]
        summary = _create_model_summary(variant_evals)
        if descriptor["mergedSetupAlias"]:
            model_info = dict(summary["model_info"])
            if descriptor["variantKey"] == "base":
                model_info["id"] = descriptor["familyId"]
            else:
                model_info["id"] = f"{descriptor['familyId']}::{descriptor['variantKey']}"
            model_info["name"] = descriptor["variantDisplayName"]
            if descriptor["variantKey"] == "base":
                model_info["model_version"] = None
            else:
                model_info["model_version"] = descriptor["variantLabel"]
        else:
            model_info = summary["model_info"]
        raw_ids = sorted(
            {
                ev.get("model_info", {}).get("id")
                for ev in variant_evals
                if ev.get("model_info", {}).get("id")
            },
            key=_locale_compare_key,
        )
        variants_unsorted.append(
            {
                **summary,
                "model_info": model_info,
                "variant_id": f"{descriptor['familyId']}::{descriptor['variantKey']}",
                "variant_key": descriptor["variantKey"],
                "variant_label": descriptor["variantLabel"],
                "variant_display_name": descriptor["variantDisplayName"],
                "raw_model_ids": raw_ids,
                "family_id": descriptor["familyId"],
                "family_name": descriptor["familyName"],
                "version_date": descriptor["versionDate"],
                "version_qualifier": descriptor["versionQualifier"],
            }
        )

    variants = _sort_variants(variants_unsorted)
    family_summary = _create_model_summary(evaluations)
    representative = variants[0] if variants else family_summary

    family_raw_ids = sorted(
        {
            ev.get("model_info", {}).get("id")
            for ev in evaluations
            if ev.get("model_info", {}).get("id")
        },
        key=_locale_compare_key,
    )

    return {
        **family_summary,
        "model_info": {
            **(representative.get("model_info") or {}),
            "id": family_identity["familyId"],
            "name": family_identity["familyName"],
            "model_version": None,
        },
        "model_family_id": family_identity["familyId"],
        "model_route_id": parity.get_model_family_route_id(family_identity["familyId"]),
        "model_family_name": family_identity["familyName"],
        "raw_model_ids": family_raw_ids,
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# Developer surfaces — port of `getDeveloperBenchmarkStats` and
# `hfDeveloperDetailToSummary` from lib/model-data.ts.
# ---------------------------------------------------------------------------


def get_developer_benchmark_stats(models: list[dict]) -> dict[str, int]:
    """Port of ``getDeveloperBenchmarkStats`` (lib/model-data.ts:289).

    Returns benchmark name → number of distinct models that have it. Falls
    back to ``top_benchmark_scores[].benchmark`` when ``benchmark_names`` is
    empty (matches the TS truthy-coalesce ``benchmarkNames.length > 0 ? …``).

    Uses ``dict.fromkeys`` to mirror JS ``new Set(array)`` insertion-order
    semantics. Python's ``set()`` is hash-ordered; iterating it would shuffle
    the order benchmarks first land in ``counts``, and a later stable sort
    by count would surface tied-count entries in a different order than the
    TS adapter (``IFEval, BBH, MMLU PRO`` vs ``BBH, GPQA, IFEval``).
    """
    counts: dict[str, int] = {}
    for model in models:
        benchmark_names = [b for b in (model.get("benchmark_names") or []) if b]
        if benchmark_names:
            unique = list(dict.fromkeys(benchmark_names))
        else:
            unique = list(
                dict.fromkeys(
                    score.get("benchmark")
                    for score in (model.get("top_benchmark_scores") or [])
                    if score.get("benchmark")
                )
            )
        for benchmark in unique:
            counts[benchmark] = counts.get(benchmark, 0) + 1
    return counts


def hf_developer_detail_to_summary(detail: dict) -> dict:
    """Port of ``hfDeveloperDetailToSummary`` (lib/model-data.ts:1373).

    Produces the `developer-summary` API shape: developer (canonical),
    route_id, model_count, benchmark_count, evaluation_count,
    popular_evals, models[] (each post-`hfModelCardToEvaluationCardData`).
    """
    raw_models = detail.get("models") or []
    model_cards = [hf_model_card_to_evaluation_card_data(m) for m in raw_models]
    benchmark_counts = get_developer_benchmark_stats(raw_models)
    evaluation_count = sum(int(m.get("total_evaluations") or 0) for m in raw_models)
    popular_evals = [
        {
            "benchmark": parity.get_benchmark_display_name(benchmark),
            "model_count": count,
        }
        for benchmark, count in sorted(
            benchmark_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:3]
    ]
    return {
        "developer": parity.normalize_developer_name(detail.get("developer")),
        "route_id": parity.get_developer_route_id(detail.get("developer") or ""),
        "model_count": len(raw_models),
        "benchmark_count": len(benchmark_counts),
        "evaluation_count": evaluation_count,
        "popular_evals": popular_evals,
        "models": model_cards,
    }


def hf_developer_detail_to_list_entry(detail: dict) -> dict:
    """Same shape as `hf_developer_detail_to_summary` minus `models[]` —
    matches what `getDeveloperList` returns (lib/model-data.ts:1305)."""
    summary = hf_developer_detail_to_summary(detail)
    summary.pop("models", None)
    return summary


# ---------------------------------------------------------------------------
# Setup-alias variant merging — port of `normalizeSingleModelCardEntry`
# (lib/hf-data.ts:750). The TS API path runs this on every model-card
# fetch (`fetchModelCardsList` / `fetchModelCardsListLite`) before the
# adapter sees the data, so the parity emitter must run it too.
# ---------------------------------------------------------------------------


def _get_latest_timestamp(a: Any, b: Any) -> Any:
    """Port of `getLatestTimestamp` (lib/hf-data.ts:722). Returns the more
    recent of two timestamp strings; falls back to the finite one if the
    other doesn't parse."""
    if not a:
        return b
    if not b:
        return a
    a_ms = parity.normalize_eval_timestamp(a)
    b_ms = parity.normalize_eval_timestamp(b)
    if not math.isfinite(a_ms):
        return b
    if not math.isfinite(b_ms):
        return a
    return b if b_ms > a_ms else a


def _sort_normalized_model_card_variants_key(variant: dict) -> tuple:
    """Port of `sortNormalizedModelCardVariants` (lib/hf-data.ts:734).

    Order: ``default`` first; then ``last_updated`` descending; then
    ``variant_label`` locale-compare ascending.

    Encoding: tuples sort lexicographically. ``(not is_default)`` is
    False (=0) for default so it sorts FIRST. ``-ts_ms`` flips ascending
    sort to descending. ``_locale_compare_key`` matches JS
    ``localeCompare`` for ASCII-ish strings.
    """
    is_default = variant.get("variant_key") == "default"
    last_updated = variant.get("last_updated")
    if last_updated:
        ts_ms = parity.normalize_eval_timestamp(last_updated)
        if not math.isfinite(ts_ms):
            ts_ms = float("-inf")
    else:
        ts_ms = float("-inf")
    label = variant.get("variant_label") or ""
    return (not is_default, -ts_ms, _locale_compare_key(label))


def normalize_single_model_card_entry(entry: dict) -> dict:
    """Port of `normalizeSingleModelCardEntry` (lib/hf-data.ts:750).

    Walks each variant, computes the normalized variant_key (``base`` →
    ``default``; setup-alias qualifiers like ``prompt`` / ``fc`` /
    ``thinking-*`` collapse onto the underlying ``version_date``), then
    merges duplicate keys (sums ``evaluation_count``, picks max
    ``last_updated``, dedups + sorts ``raw_model_ids``). Re-derives the
    canonical family identity at the top level.

    Without this, parity ``model_card`` payloads carry the raw 6-variant
    counts the pipeline emits, while the user-facing TS path serves the
    post-merge 2-3 counts — diverging on every model with multiple
    "prompt" / "fc" / "thinking" submission rows.
    """
    family_identity = parity.get_canonical_model_identity(
        {"id": entry.get("model_family_id"), "name": entry.get("model_family_name")}
    )
    variants_by_key: dict[str, dict] = {}
    for variant in entry.get("variants") or []:
        original_key = variant.get("variant_key")
        original_label = variant.get("variant_label") or ""

        if original_key == "base":
            normalized_key = "default"
            normalized_label = "Default"
        elif original_key != "default":
            # TS uses both `id` and `name` on the synthetic identity. The
            # name here is identical to the id (a string handle), matching
            # the TS code's literal copy.
            synthetic_handle = f"{family_identity['familyId']}-{original_key}"
            synthetic_identity = parity.get_canonical_model_identity(
                {"id": synthetic_handle, "name": synthetic_handle}
            )
            if (
                synthetic_identity.get("versionDate")
                and parity.is_setup_alias_qualifier(synthetic_identity.get("versionQualifier"))
            ):
                normalized_key = synthetic_identity["versionDate"]
                normalized_label = synthetic_identity["versionDate"]
            else:
                normalized_key = synthetic_identity.get("variantKey") or original_key
                normalized_label = synthetic_identity.get("variantLabel") or original_label
        else:
            # `default` passes through — TS keeps the original label.
            normalized_key = original_key
            normalized_label = original_label

        existing = variants_by_key.get(normalized_key)
        if existing is not None:
            existing["evaluation_count"] = int(existing.get("evaluation_count") or 0) + int(
                variant.get("evaluation_count") or 0
            )
            existing["last_updated"] = _get_latest_timestamp(
                existing.get("last_updated"), variant.get("last_updated")
            )
            # TS: `Array.from(new Set([...existing, ...variant])).sort(localeCompare)`
            # `dict.fromkeys` mirrors JS `new Set(array)` insertion-order semantics.
            merged_ids = list(
                dict.fromkeys(
                    list(existing.get("raw_model_ids") or [])
                    + list(variant.get("raw_model_ids") or [])
                )
            )
            existing["raw_model_ids"] = sorted(merged_ids, key=_locale_compare_key)
            continue

        new_variant = dict(variant)
        new_variant["variant_key"] = normalized_key
        new_variant["variant_label"] = normalized_label
        new_variant["raw_model_ids"] = sorted(
            list(variant.get("raw_model_ids") or []), key=_locale_compare_key
        )
        variants_by_key[normalized_key] = new_variant

    normalized_variants = sorted(
        variants_by_key.values(), key=_sort_normalized_model_card_variants_key
    )
    if normalized_variants:
        total_evals = sum(int(v.get("evaluation_count") or 0) for v in normalized_variants)
    else:
        total_evals = entry.get("total_evaluations")

    out = dict(entry)
    out["model_family_id"] = family_identity["familyId"]
    out["model_route_id"] = parity.get_model_family_route_id(family_identity["familyId"])
    out["model_family_name"] = family_identity["familyName"]
    out["total_evaluations"] = total_evals
    out["variants"] = normalized_variants
    return out

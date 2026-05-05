import json
import re
from pathlib import Path
from typing import Any

from scripts import registry
from scripts.helpers.benchmark_identity import normalize_benchmark_key
from scripts.helpers.metric_constants import (
    BENCHMARK_DEFAULT_METRICS,
    BUILTIN_METRIC_DISPLAY_MAP,
    EVAL_DESCRIPTION_METRIC_REGEX,
    PASS_AT_EXACT_REGEX,
)
from scripts.helpers.presentation import humanize_metric_key, humanize_token_key
from scripts.helpers.slug_utils import as_string, slugify


METRIC_REGISTRY_ALIAS_LOOKUP: dict[str, str] = {}
METRIC_REGISTRY_ENTRIES: dict[str, dict] = {}
METRIC_SUFFIX_ALIAS_CANDIDATES: list[str] = []


def load_metric_registry(path: Path) -> None:
    global METRIC_REGISTRY_ALIAS_LOOKUP
    global METRIC_REGISTRY_ENTRIES
    global METRIC_SUFFIX_ALIAS_CANDIDATES

    METRIC_REGISTRY_ALIAS_LOOKUP = {}
    METRIC_REGISTRY_ENTRIES = {}
    METRIC_SUFFIX_ALIAS_CANDIDATES = []

    if not path.exists():
        return

    parsed = json.loads(path.read_text(encoding="utf-8"))
    entries = parsed.get("entries") if isinstance(parsed, dict) else None
    alias_map = parsed.get("alias_to_normalized") if isinstance(parsed, dict) else None

    if isinstance(entries, list):
        for entry in entries:
            normalized = as_string((entry or {}).get("normalized")).strip()
            if normalized:
                METRIC_REGISTRY_ENTRIES[normalized] = entry

    if isinstance(alias_map, dict):
        for key, value in alias_map.items():
            norm_key = as_string(key).strip()
            norm_value = as_string(value).strip()
            if norm_key and norm_value:
                METRIC_REGISTRY_ALIAS_LOOKUP[norm_key] = norm_value

    candidate_set = set()
    for raw_alias in METRIC_REGISTRY_ALIAS_LOOKUP:
        normalized_alias = normalize_benchmark_key(raw_alias)
        if normalized_alias:
            candidate_set.add(normalized_alias)
    for canonical_key in METRIC_REGISTRY_ENTRIES:
        normalized_alias = normalize_benchmark_key(canonical_key)
        if normalized_alias:
            candidate_set.add(normalized_alias)
    METRIC_SUFFIX_ALIAS_CANDIDATES = sorted(
        candidate_set, key=lambda value: (-len(value.split("_")), -len(value), value)
    )


def canonicalize_metric_key(value: Any) -> str:
    raw = as_string(value).strip()
    if not raw:
        return ""
    pass_match = PASS_AT_EXACT_REGEX.match(raw)
    if pass_match:
        return f"pass_at_{pass_match.group(1)}"

    candidates = [
        raw,
        normalize_benchmark_key(raw),
        normalize_benchmark_key(raw.split(".")[-1]),
    ]
    for candidate in candidates:
        if candidate and candidate in METRIC_REGISTRY_ALIAS_LOOKUP:
            return METRIC_REGISTRY_ALIAS_LOOKUP[candidate]

    registry_result = registry.resolve_metric(raw)
    if registry_result and registry_result.get("canonical_id"):
        return registry_result["canonical_id"].replace("-", "_")

    return normalize_benchmark_key(raw.split(".")[-1]) or normalize_benchmark_key(raw)


def strict_metric_alias_lookup(value: Any) -> str:
    raw = as_string(value).strip()
    if not raw:
        return ""
    pass_match = PASS_AT_EXACT_REGEX.match(raw)
    if pass_match:
        return f"pass_at_{pass_match.group(1)}"

    candidates = [
        raw,
        normalize_benchmark_key(raw),
        normalize_benchmark_key(raw.split(".")[-1]),
    ]
    for candidate in candidates:
        if candidate and candidate in METRIC_REGISTRY_ALIAS_LOOKUP:
            return METRIC_REGISTRY_ALIAS_LOOKUP[candidate]
    return ""


def preferred_metric_display(metric_key: str, raw_label: Any = None) -> str:
    if metric_key in METRIC_REGISTRY_ENTRIES:
        display = as_string(METRIC_REGISTRY_ENTRIES[metric_key].get("display_name")).strip()
        if display:
            return display
    if metric_key in BUILTIN_METRIC_DISPLAY_MAP:
        return BUILTIN_METRIC_DISPLAY_MAP[metric_key]
    if (
        raw_label
        and canonicalize_metric_key(raw_label) == metric_key
        and normalize_benchmark_key(raw_label) == metric_key
    ):
        return as_string(raw_label).strip()
    return humanize_metric_key(metric_key)


def infer_metric_from_value(metric_name: Any = None, metric_id: Any = None) -> dict | None:
    explicit_id = as_string(metric_id).strip()
    explicit_name = as_string(metric_name).strip()

    if explicit_id:
        metric_key = canonicalize_metric_key(explicit_id) or slugify(explicit_id)
        display = preferred_metric_display(metric_key, explicit_name or explicit_id.split(".")[-1])
        return {
            "metric_name": display,
            "metric_id": explicit_id,
            "metric_key": metric_key or "score",
        }

    raw = explicit_name
    if not raw:
        return None

    metric_key = canonicalize_metric_key(raw) or slugify(raw) or "score"
    display = preferred_metric_display(metric_key, explicit_name)
    return {
        "metric_name": display,
        "metric_id": metric_key,
        "metric_key": metric_key,
    }


def infer_metric_from_score_details(result: dict) -> dict | None:
    details = (
        ((result.get("score_details") or {}).get("details") or {})
        if isinstance(result, dict)
        else {}
    )
    if not isinstance(details, dict):
        return None
    tab = as_string(details.get("tab")).strip()
    if not tab:
        return None
    return infer_metric_from_value(metric_name=tab)


def infer_metric_from_benchmark_defaults(benchmark_key: str) -> dict | None:
    default = BENCHMARK_DEFAULT_METRICS.get(normalize_benchmark_key(benchmark_key))
    if not default:
        return None
    metric_name, metric_id_value = default
    return {
        "metric_name": metric_name,
        "metric_id": metric_id_value,
        "metric_key": normalize_benchmark_key(metric_id_value),
    }


def infer_metric_from_benchmark_card(card: dict | None) -> dict | None:
    metrics = (
        (((card or {}).get("methodology") or {}).get("metrics") or [])
        if isinstance(card, dict)
        else []
    )
    if isinstance(metrics, list) and metrics:
        return infer_metric_from_value(metric_name=metrics[0])
    return None


def metric_namespace_component(metric_id: str, benchmark_family_key: str) -> tuple[str | None, str | None]:
    parts = [part for part in re.split(r"[./]+", as_string(metric_id)) if part]
    if len(parts) < 3:
        return None, None
    if normalize_benchmark_key(parts[0]) != normalize_benchmark_key(benchmark_family_key):
        return None, None
    component_parts = parts[1:-1]
    if not component_parts:
        return None, None
    component_key = normalize_benchmark_key("_".join(component_parts))
    return humanize_token_key(" ".join(component_parts)), component_key


def split_metric_from_evaluation_description(description: Any) -> dict | None:
    text = as_string(description).strip()
    if not text:
        return None
    match = EVAL_DESCRIPTION_METRIC_REGEX.match(text)
    if not match:
        return None
    return infer_metric_from_value(metric_name=match.group(1))


def split_metric_from_evaluation_name(raw_name: Any, benchmark_keys: list[str]) -> dict | None:
    name = as_string(raw_name).strip()
    if not name:
        return None

    normalized_name = normalize_benchmark_key(name)
    for benchmark_key in benchmark_keys:
        if benchmark_key and normalized_name.startswith(f"{benchmark_key}_"):
            suffix = normalized_name[len(benchmark_key) + 1 :]
            if strict_metric_alias_lookup(suffix):
                maybe_metric = infer_metric_from_value(metric_name=suffix)
                if maybe_metric:
                    return {
                        "component_name": None,
                        "component_key": None,
                        "metric": maybe_metric,
                        "metric_source": "evaluation_name_suffix",
                    }

    raw_tokens = [token for token in re.split(r"[.\s_-]+", name) if token]
    for split_index in range(1, len(raw_tokens)):
        prefix_raw = " ".join(raw_tokens[:split_index]).strip()
        suffix_raw = " ".join(raw_tokens[split_index:]).strip()
        if not suffix_raw:
            continue
        if not strict_metric_alias_lookup(suffix_raw):
            continue
        metric = infer_metric_from_value(metric_name=suffix_raw)
        if not metric:
            continue
        component_key = normalize_benchmark_key(prefix_raw) if prefix_raw else None
        if component_key and component_key in benchmark_keys:
            prefix_raw = ""
            component_key = None
        return {
            "component_name": humanize_token_key(prefix_raw) if prefix_raw else None,
            "component_key": component_key,
            "metric": metric,
            "metric_source": "evaluation_name_suffix",
        }

    direct_metric_key = strict_metric_alias_lookup(name)
    if direct_metric_key:
        metric = infer_metric_from_value(metric_name=name)
        if metric:
            return {
                "component_name": None,
                "component_key": None,
                "metric": metric,
                "metric_source": "evaluation_name",
            }

    for alias_candidate in METRIC_SUFFIX_ALIAS_CANDIDATES:
        if not alias_candidate or not normalized_name.endswith(f"_{alias_candidate}"):
            continue
        prefix = normalized_name[: -(len(alias_candidate) + 1)]
        if not prefix:
            continue
        if not strict_metric_alias_lookup(alias_candidate):
            continue
        metric = infer_metric_from_value(metric_name=alias_candidate)
        if not metric:
            continue
        component_key = normalize_benchmark_key(prefix)
        if component_key in benchmark_keys:
            component_key = None
            component_name = None
        else:
            component_name = humanize_token_key(prefix)
        return {
            "component_name": component_name,
            "component_key": component_key,
            "metric": metric,
            "metric_source": "evaluation_name_suffix",
        }

    return None


_METRIC_KIND_TO_GROUP = {
    "accuracy": "capability",
    "exact_match": "capability",
    "em": "capability",
    "f1": "capability",
    "win_rate": "capability",
    "winrate": "capability",
    "cost": "cost",
    "latency": "latency",
    "throughput": "latency",
    "time": "latency",
    "rank": "rank",
    "difference": "robustness",
}

# Order matters: first matching pattern wins. Listed most-specific first so
# e.g. "Latency Standard Deviation" lands in latency rather than robustness.
_METRIC_NAME_GROUP_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "cost",
        re.compile(r"\b(?:cost|usd|dollar|price)\b", re.IGNORECASE),
    ),
    (
        "latency",
        re.compile(
            r"\b(?:latency|throughput|elapsed|wall[\s_]?time|"
            r"tokens?[\s_/]?(?:per|sec|s)\b|p\d{2,3}|percentile)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rank",
        re.compile(r"\brank\b", re.IGNORECASE),
    ),
    (
        "robustness",
        re.compile(
            r"\b(?:sensitivity|delta|stddev|standard[\s_]?deviation|"
            r"variance|robustness)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "efficiency",
        re.compile(r"\b(?:attempts|retries|tries)\b", re.IGNORECASE),
    ),
    (
        "capability",
        re.compile(
            r"\b(?:accuracy|acc|elo|score|pass@\d+|win[\s_]?rate|f1|"
            r"exact[\s_]?match|em|bleu|rouge(?:-\d+)?|recall|precision|"
            r"mrr|ndcg|coverage|correct|harmlessness)\b",
            re.IGNORECASE,
        ),
    ),
)

# Tab order in the histogram UI. Capability surfaces first (the actual
# task score), followed by capability-adjacent groups, then instrumental
# groups, with "other" as a fallback bucket.
METRIC_GROUP_ORDER = (
    "capability",
    "robustness",
    "efficiency",
    "cost",
    "latency",
    "rank",
    "other",
)
_METRIC_GROUP_INDEX = {group: i for i, group in enumerate(METRIC_GROUP_ORDER)}


def metric_group(metric: dict) -> str:
    """Classify a metric into one of ``METRIC_GROUP_ORDER``.

    ``metric_kind`` from upstream EEE is authoritative when present (e.g.
    ``kind=cost`` always means ``cost`` regardless of name). Otherwise we
    match the metric name against the rules above. Defaults to ``other``.
    """

    config = metric.get("metric_config") or {}
    kind = (as_string(config.get("metric_kind")) or "").lower()
    if kind in _METRIC_KIND_TO_GROUP:
        return _METRIC_KIND_TO_GROUP[kind]
    name = as_string(metric.get("metric_name"))
    if name:
        for group, pattern in _METRIC_NAME_GROUP_RULES:
            if pattern.search(name):
                return group
    return "other"


def metric_group_order_index(group: str) -> int:
    """Sort key for a metric group. Lower = surfaced first in the UI."""

    return _METRIC_GROUP_INDEX.get(group, _METRIC_GROUP_INDEX["other"])


# Tier mapping derived from the group taxonomy: capability wins, then
# capability-adjacent (robustness / efficiency / other), then instrumental.
# Used only for eval-list.json's ``primary_metric_name`` — comparison-index
# emits all metrics so doesn't pick a primary.
_METRIC_TIER_BY_GROUP = {
    "capability": 0,
    "robustness": 1,
    "efficiency": 1,
    "other": 1,
    "cost": 2,
    "latency": 2,
    "rank": 2,
}


def metric_priority_tier(metric: dict) -> int:
    """Rank a metric for ``primary_metric_name`` selection. Derived from
    ``metric_group`` so the picker and the comparison-index tab order stay
    in lockstep.
    """

    return _METRIC_TIER_BY_GROUP.get(metric_group(metric), 1)


def pick_primary_metric(metrics: list[dict]) -> dict | None:
    """Pick one canonical metric from a list, preferring capability scores
    over instrumental ones. Stable secondary sort matches the existing
    alphabetical order so within-tier choices stay deterministic and align
    with the order metrics appear in eval-list.json. Used for eval-list's
    ``primary_metric_name`` only — the comparison index emits every metric
    via a tabbed UI and no longer needs a single pick.
    """

    if not metrics:
        return None
    return min(
        metrics,
        key=lambda m: (
            metric_priority_tier(m),
            as_string(m.get("metric_name")),
            as_string(m.get("metric_summary_id")),
        ),
    )


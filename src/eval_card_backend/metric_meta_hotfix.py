"""HOTFIX — temporary metric-meta inference layer.

PURPOSE
    Resolve `metric_kind`, `metric_unit`, `min_score`, `max_score`, and
    `lower_is_better` for each fact row. Used by Stage D to populate
    fact_results columns that feed comparability-threshold computation
    (`signals/comparability.py:compute_threshold`) and `score_scale_anomaly`.

WHY HOTFIX
    Both upstream sources of this metadata are partially populated:
      - The registry's `canonical_metrics` table only carries `score_type`,
        `min_score`, `max_score`, and `lower_is_better` today; it has no
        `metric_kind` or `metric_unit`.
      - EEE per-record `metric_config` populates the spec-shaped fields
        for ~27.5% of rows (sampled 2026-05). The remaining ~72.5% have
        NULL `metric_kind` and `metric_unit`, but ~99.8% of those have
        `min_score=0, max_score=1, score_type='continuous'` — i.e. they
        ARE proportion-scale, just unlabelled.

    Without this layer, comparability threshold for those rows falls
    through to `range_5pct` (numerically correct for [0,1] but with the
    wrong basis label) and `score_scale_anomaly` is permanently NULL
    (always-false silently).

RETIREMENT
    When evalcard-registry's `canonical_metrics` gains real `metric_kind`
    / `metric_unit` columns AND coverage is verified against EEE's actual
    vocabulary, retire by:
      1. Stage A loads the new columns directly from canonical_metrics.
      2. Stage D reads metric_kind / metric_unit / min / max / lower
         straight from the JOINed canonical_metrics row.
      3. Delete this module + its UDF wiring + its tests.
    The `_provenance_counter` (logged at end of run) is the retirement
    signal — when the share of rows hitting `'heuristic_*'` drops below
    ~5%, the registry is doing its job and the module can go.

DESIGN
    Layered chain per field. Higher layer wins when populated:

      | field           | registry                          | eee per-record                | heuristic                                   |
      |-----------------|-----------------------------------|-------------------------------|---------------------------------------------|
      | metric_kind     | canonical_metrics.metric_kind     | metric_config.metric_kind     | regex on metric_name; default 'score'       |
      | metric_unit     | canonical_metrics.metric_unit     | metric_config.metric_unit     | [min=0, max=1, continuous] → 'proportion'   |
      | min_score       | canonical_metrics.min_score       | metric_config.min_score       | —                                           |
      | max_score       | canonical_metrics.max_score       | metric_config.max_score       | —                                           |
      | lower_is_better | canonical_metrics.lower_is_better | metric_config.lower_is_better | default False                               |

    Synonym normalisation (last step): metric_unit `'percentage'` → `'percent'`.

    Registry beats EEE because the registry's whole point is canonical
    metadata; if it knows the canonical for a metric, that's authoritative
    over a single record's view. EEE per-record beats heuristics for the
    same reason (real data > inference).

    Today registry's metric_kind / metric_unit are always NULL, so the
    registry layer is a no-op for those two fields. The chain is wired
    for the future.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Provenance counter — logged at end-of-run. Tracks which layer filled each
# field, so we can monitor when the registry has caught up enough to retire
# this module. See `log_metric_meta_summary()`.
# ---------------------------------------------------------------------------

_provenance_counter: Counter[tuple[str, str]] = Counter()


def reset_provenance_counter() -> None:
    _provenance_counter.clear()


def log_metric_meta_summary(log) -> None:
    """Pipeline.run() calls this at end of run alongside other summaries."""
    if not _provenance_counter:
        return
    log.info("--- metric-meta hotfix provenance ---")
    by_field: dict[str, Counter[str]] = {}
    for (field, source), count in _provenance_counter.items():
        by_field.setdefault(field, Counter())[source] += count
    for field in sorted(by_field):
        total = sum(by_field[field].values())
        breakdown = ", ".join(
            f"{src}={n} ({n/total:.0%})"
            for src, n in by_field[field].most_common()
        )
        log.info("  %s: %s", field, breakdown)


# ---------------------------------------------------------------------------
# Heuristic regex: metric_name → metric_kind.
#
# Vocabulary mirrors values upstream EEE itself populates when it does fill
# `metric_kind` (sampled 2026-05): accuracy, exact_match, f1, win_rate,
# pass_rate, elo, cost, latency, throughput, rank, count, score,
# benchmark_score, refusal_rate, standard_deviation, ndcg, bleu, rouge.
#
# Order matters: most-specific first. ASCII metric names are common but case
# varies; match case-insensitively. Word-boundary anchors avoid `accuracy`
# matching `inaccuracy_rate`, etc.
# ---------------------------------------------------------------------------

_NAME_TO_METRIC_KIND_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bpass\s*@?\s*\d+", re.I),                                "pass_rate"),
    (re.compile(r"\b(?:exact[\s_]?match|^em$|\bem\b)", re.I),               "exact_match"),
    (re.compile(r"\bf1\b", re.I),                                           "f1"),
    (re.compile(r"\bwin[\s_]?rate\b", re.I),                                "win_rate"),
    (re.compile(r"\belo\b", re.I),                                          "elo"),
    (re.compile(r"\b(?:refusal|abstain)[\s_]?rate\b", re.I),                "refusal_rate"),
    (re.compile(r"\b(?:cost|usd|dollar|price)\b", re.I),                    "cost"),
    (re.compile(r"\b(?:latency|elapsed|wall[\s_]?time)\b", re.I),           "latency"),
    (re.compile(r"\bthroughput\b|tokens?[\s_/]?per[\s_]?sec", re.I),        "throughput"),
    (re.compile(r"\brank\b", re.I),                                         "rank"),
    (re.compile(r"\b(?:standard[\s_]?deviation|stddev|variance)\b", re.I),  "standard_deviation"),
    (re.compile(r"\b(?:sensitivity|delta)\b", re.I),                        "difference"),
    (re.compile(r"\b(?:ndcg|mrr)\b", re.I),                                 "ndcg"),
    (re.compile(r"\bbleu(?:-\d+)?\b", re.I),                                "bleu"),
    (re.compile(r"\brouge(?:-\d+)?\b", re.I),                               "rouge"),
    (re.compile(r"\b(?:attempts|retries|tries)\b", re.I),                   "count"),
    (re.compile(r"\b(?:accuracy|acc|correct|harmlessness)\b", re.I),        "accuracy"),
    (re.compile(r"\b(?:recall|precision)\b", re.I),                         "accuracy"),
)

# Default when no regex matches. 'score' is the spec's catchall; keeping it
# here (rather than NULL) gives the frontend a non-null kind it can render
# for any metric — provenance counter still flags how often this fired.
_DEFAULT_METRIC_KIND = "score"

# metric_unit synonym normalisation. EEE records spell the same concept
# multiple ways; consolidate so the threshold short-circuit fires uniformly.
_METRIC_UNIT_SYNONYMS: dict[str, str] = {
    "percentage": "percent",
}


# ---------------------------------------------------------------------------
# The core function
# ---------------------------------------------------------------------------

def _coalesce(*values):
    """Return the first non-None argument, or None."""
    for v in values:
        if v is not None:
            return v
    return None


def _is_real_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _infer_metric_kind_from_name(name: str | None) -> str | None:
    if not name or not isinstance(name, str):
        return None
    for pattern, kind in _NAME_TO_METRIC_KIND_RULES:
        if pattern.search(name):
            return kind
    return None


def _normalise_metric_unit(unit: Any) -> Any:
    if isinstance(unit, str):
        return _METRIC_UNIT_SYNONYMS.get(unit.lower(), unit)
    return unit


def _record(field: str, source: str) -> None:
    _provenance_counter[(field, source)] += 1


def derive_metric_meta(
    eee_metric_config: Any,
    registry_metric_kind: Any,
    registry_metric_unit: Any,
    registry_min_score: Any,
    registry_max_score: Any,
    registry_lower_is_better: Any,
    metric_name: str | None,
    registry_score_type: Any = None,
) -> dict:
    """Return resolved metric meta for a single fact row.

    All inputs are nullable (the layered chain handles every absence).

    `eee_metric_config` is the raw per-record `metric_config` dict from EEE
    (already parsed; not a JSON string — the UDF wrapper handles coercion).
    Other registry_* args are the columns from the JOINed canonical_metrics
    row, NULL when the metric didn't resolve or the registry doesn't carry
    the field. `registry_score_type` is the registry's `score_type`
    (`binary | continuous | levels`), passed alongside the rest so the
    proportion-shape heuristic can fire when EEE records omit score_type
    but the registry knows it.
    """
    if not isinstance(eee_metric_config, dict):
        eee_metric_config = {}

    eee_kind            = eee_metric_config.get("metric_kind")
    eee_unit            = eee_metric_config.get("metric_unit")
    eee_min             = eee_metric_config.get("min_score")
    eee_max             = eee_metric_config.get("max_score")
    eee_lower           = eee_metric_config.get("lower_is_better")
    # score_type: prefer per-record EEE; fall back to registry. Used only
    # for the proportion-shape heuristic; not exposed downstream.
    eee_score_type      = eee_metric_config.get("score_type") or registry_score_type

    # min_score / max_score / lower_is_better — straightforward chain.
    min_score = _coalesce(registry_min_score, eee_min)
    max_score = _coalesce(registry_max_score, eee_max)
    if min_score is None and max_score is None:
        _record("min_score", "default_null")
        _record("max_score", "default_null")
    else:
        _record("min_score", "registry" if registry_min_score is not None else "eee_record")
        _record("max_score", "registry" if registry_max_score is not None else "eee_record")

    if registry_lower_is_better is not None:
        lower_is_better = bool(registry_lower_is_better)
        _record("lower_is_better", "registry")
    elif eee_lower is not None:
        lower_is_better = bool(eee_lower)
        _record("lower_is_better", "eee_record")
    else:
        lower_is_better = False
        _record("lower_is_better", "heuristic_default")

    # metric_unit — registry, then EEE, then proportion-from-shape heuristic.
    if registry_metric_unit is not None:
        metric_unit = _normalise_metric_unit(registry_metric_unit)
        metric_unit_provenance = "registry"
    elif eee_unit is not None:
        metric_unit = _normalise_metric_unit(eee_unit)
        metric_unit_provenance = "eee_record"
    elif (
        _is_real_number(min_score) and min_score == 0
        and _is_real_number(max_score) and max_score == 1
        and (isinstance(eee_score_type, str) and eee_score_type.lower() == "continuous")
    ):
        # The most common NULL-unit shape (~99.8% of NULL rows in current
        # data): unlabelled but [0,1] continuous-valued. Defensible inference.
        metric_unit = "proportion"
        metric_unit_provenance = "heuristic_proportion_shape"
    else:
        metric_unit = None
        metric_unit_provenance = "default_null"
    _record("metric_unit", metric_unit_provenance)

    # metric_kind — registry, then EEE, then name-regex, then 'score' default.
    if registry_metric_kind is not None:
        metric_kind = registry_metric_kind
        metric_kind_provenance = "registry"
    elif eee_kind is not None:
        metric_kind = eee_kind
        metric_kind_provenance = "eee_record"
    else:
        regex_kind = _infer_metric_kind_from_name(metric_name)
        if regex_kind is not None:
            metric_kind = regex_kind
            metric_kind_provenance = "heuristic_regex"
        else:
            metric_kind = _DEFAULT_METRIC_KIND
            metric_kind_provenance = "heuristic_default"
    _record("metric_kind", metric_kind_provenance)

    return {
        "metric_kind": metric_kind,
        "metric_unit": metric_unit,
        "min_score": min_score,
        "max_score": max_score,
        "lower_is_better": lower_is_better,
        "metric_kind_provenance": metric_kind_provenance,
        "metric_unit_provenance": metric_unit_provenance,
    }

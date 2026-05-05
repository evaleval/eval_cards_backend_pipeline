"""Comparability-threshold constants and resolver.

The four basis labels and their threshold values are pinned here:
    proportion       → 0.05
    percent          → 5.0  (percentage points)
    range_5pct       → 0.05 * (max_score - min_score)
    fallback_default → 0.05  (absolute)

Inputs come from the per-row resolved metric meta (the
metric_meta_hotfix layered chain), passed as a dict-shaped
`metric_config` at call sites in `signals/comparability.py`.
"""
from __future__ import annotations

from typing import Any


THRESHOLD_PROPORTION: float = 0.05
THRESHOLD_PERCENT: float = 5.0
THRESHOLD_RANGE_5PCT_FACTOR: float = 0.05
THRESHOLD_FALLBACK_DEFAULT: float = 0.05


BASIS_PROPORTION = "proportion"
BASIS_PERCENT = "percent"
BASIS_RANGE_5PCT = "range_5pct"
BASIS_FALLBACK_DEFAULT = "fallback_default"


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compute_threshold(metric_config: Any) -> tuple[float, str]:
    """Return (threshold, basis_label) for a metric. Basis label is one of
    the four BASIS_* constants above."""
    if isinstance(metric_config, dict):
        metric_unit = metric_config.get("metric_unit")
        if metric_unit == "proportion":
            return THRESHOLD_PROPORTION, BASIS_PROPORTION
        if metric_unit == "percent":
            return THRESHOLD_PERCENT, BASIS_PERCENT
        min_score = metric_config.get("min_score")
        max_score = metric_config.get("max_score")
        if (
            _is_real_number(min_score)
            and _is_real_number(max_score)
            and max_score > min_score
        ):
            return THRESHOLD_RANGE_5PCT_FACTOR * (max_score - min_score), BASIS_RANGE_5PCT
    return THRESHOLD_FALLBACK_DEFAULT, BASIS_FALLBACK_DEFAULT

"""Comparability-threshold constants and resolver.

The four basis labels and their threshold values are pinned here:
    proportion       → 0.05
    percent          → 5.0  (percentage points)
    range_5pct       → 0.05 * (max_score - min_score)
    fallback_default → 0.05  (absolute)

Inputs come from the per-row resolved metric meta (the
metric_meta_hotfix layered chain), passed as a dict-shaped
`metric_config` at call sites in `signals/comparability.py`.

Sensitivity sweep: all four bases are the same "5%" boundary expressed in
different units, so they scale together by a single multiplier read from the
`DIVERGENCE_THRESHOLD_FACTOR` env var (default 1.0 → the pinned 5% values
unchanged). A run with the var set to 0.5 / 1.0 / 1.5 / 2.0 evaluates the
divergence signals at 2.5% / 5% / 7.5% / 10%. The factor is applied only to
the returned threshold magnitude — the basis label is unaffected.
"""
from __future__ import annotations

import math
import os
from typing import Any


THRESHOLD_PROPORTION: float = 0.05
THRESHOLD_PERCENT: float = 5.0
THRESHOLD_RANGE_5PCT_FACTOR: float = 0.05
THRESHOLD_FALLBACK_DEFAULT: float = 0.05


BASIS_PROPORTION = "proportion"
BASIS_PERCENT = "percent"
BASIS_RANGE_5PCT = "range_5pct"
BASIS_FALLBACK_DEFAULT = "fallback_default"


THRESHOLD_FACTOR_ENV = "DIVERGENCE_THRESHOLD_FACTOR"


def threshold_factor() -> float:
    """Multiplier applied to every divergence threshold, from
    `DIVERGENCE_THRESHOLD_FACTOR`. Returns 1.0 (the pinned 5% behaviour) when
    the var is unset, empty, non-numeric, or non-positive — so a normal run is
    never altered and a malformed sweep value fails safe to baseline."""
    raw = os.environ.get(THRESHOLD_FACTOR_ENV)
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    # Must be finite and positive; inf/nan/≤0 fail safe to the pinned baseline
    # (inf would silently make every group non-divergent).
    return value if (math.isfinite(value) and value > 0) else 1.0


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _base_threshold(metric_config: Any) -> tuple[float, str]:
    """The pinned 5% threshold and basis label, before the sweep multiplier."""
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


def compute_threshold(metric_config: Any) -> tuple[float, str]:
    """Return (threshold, basis_label) for a metric. Basis label is one of
    the four BASIS_* constants above. The threshold is the pinned 5% value for
    the basis scaled by `threshold_factor()` (1.0 unless a sweep sets the env
    var), so the default behaviour is byte-identical to the pinned values."""
    base, basis = _base_threshold(metric_config)
    return base * threshold_factor(), basis

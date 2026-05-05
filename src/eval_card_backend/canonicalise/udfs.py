"""Python UDF wrappers used by the canonicalisation pipeline.

These are thin adapters over `signals/`, plus DuckDB type-coercion concerns
(`_coerce_json` on JSON-typed UDF params is mandatory, otherwise DuckDB hands
the body a `str` and dict access silently misbehaves).

All UDFs live here so `resolver_setup.py` can import-and-register cleanly.
"""
from __future__ import annotations

import functools
import logging
from collections import Counter, defaultdict
from typing import Any

from eval_card_backend.signals.comparability import (
    compute_cross_party_divergence_py as _compute_cross_party,
)
from eval_card_backend.signals.comparability import (
    compute_variant_divergence_py as _compute_variant,
)
from eval_card_backend.metric_meta_hotfix import derive_metric_meta
from eval_card_backend.signals.completeness import compute_completeness_py
from eval_card_backend.signals.reproducibility import is_agentic_py
from eval_card_backend.signals.setup import (
    canonical_json,
    fact_id_py,
    variant_key_py,
)

# Public surface: every name `resolver_setup.py` reaches as `udfs.<name>` plus
# the run-time helpers `pipeline.run` calls. The two divergence Python entry
# points (`compute_cross_party_divergence_py`, `compute_variant_divergence_py`)
# are NOT re-exported — DuckDB consumes the `*_udf_body` wrappers below
# instead, and the bare Python functions are imported under aliases for
# internal use only. Keeping `__all__` accurate silences F401 on these
# deliberate re-exports without re-introducing the misleading entries.
__all__ = [
    "canonical_json",
    "compute_completeness_py",
    "compute_cross_party_divergence_udf_body",
    "compute_variant_divergence_udf_body",
    "derive_metric_meta",
    "derive_metric_meta_udf_body",
    "fact_id_py",
    "is_agentic_py",
    "log_resolver_summary",
    "make_resolver_udfs",
    "reset_resolver_counters",
    "variant_parent_id_py",
    "variant_key_py",
]

log = logging.getLogger(__name__)


def variant_parent_id_py(parents_json: str | None) -> str | None:
    """Extract the first `variant` edge id from the registry's
    `canonical_models.parents` column. Returns the in-family parent id
    or NULL when no variant edge is present. 
    Tolerant of NULL / empty / malformed input."""
    from eval_entity_resolver.canonical_store import (
        decode_parents,
        variant_parent_id,
    )

    return variant_parent_id(decode_parents(parents_json))


# ---------------------------------------------------------------------------
# Resolver counters (shared module state — populated as Stage C runs).
# ---------------------------------------------------------------------------

miss_counter: Counter[str] = Counter()
miss_examples: dict[str, Counter[str]] = defaultdict(Counter)
exception_seen: set[tuple[str, str]] = set()
exception_counter: Counter[tuple[str, str]] = Counter()


def reset_resolver_counters() -> None:
    miss_counter.clear()
    miss_examples.clear()
    exception_seen.clear()
    exception_counter.clear()


# Module-level handle so `log_resolver_summary` can report cache hit-rate.
_resolve_cache_info: Any = None


def make_resolver_udfs(resolver):
    """Bind `resolver` into UDF closures. Resolver is a `eval_entity_resolver.Resolver`
    instance, but typed as object so this module imports nothing at import time.

    Stage C calls resolve_canonical_id and resolve_strategy with the same
    (raw, entity_type, source_config) per row; caching the underlying resolver
    call halves the work and lets repeated raws across rows (most rows share
    the same model_raw / benchmark_raw / etc.) hit cache.
    """
    # Cache the resolver result struct so both UDFs read from the same memo.
    # 16k entries comfortably covers our scale (a few hundred unique entities
    # × 5 entity types × ~50 source_configs); LRU evicts beyond that.
    @functools.lru_cache(maxsize=16384)
    def _resolve_cached(raw: str, entity_type: str | None, source_config: str | None):
        return resolver.resolve(raw, entity_type, source_config)

    global _resolve_cache_info
    _resolve_cache_info = _resolve_cached.cache_info

    def resolve_canonical_id_py(
        raw: str | None, entity_type: str | None, source_config: str | None
    ) -> str | None:
        if not raw or not isinstance(raw, str) or not raw.strip():
            return None
        try:
            result = _resolve_cached(raw, entity_type, source_config)
        except Exception as e:
            key = (entity_type or "?", type(e).__name__)
            exception_counter[key] += 1
            if key not in exception_seen:
                exception_seen.add(key)
                log.warning(
                    "resolver raised %s on %s (first occurrence): "
                    "raw=%r config=%r err=%s",
                    type(e).__name__, entity_type, raw, source_config, e,
                )
            return None
        if result.canonical_id is None:
            miss_counter[entity_type or "?"] += 1
            miss_examples[entity_type or "?"][raw] += 1
        return result.canonical_id

    def resolve_strategy_py(
        raw: str | None, entity_type: str | None, source_config: str | None
    ) -> str:
        if not raw or not isinstance(raw, str) or not raw.strip():
            return "no_match"
        try:
            return _resolve_cached(raw, entity_type, source_config).strategy
        except Exception:
            return "no_match"

    return resolve_canonical_id_py, resolve_strategy_py


def log_resolver_summary(top_n: int = 10) -> None:
    """End-of-run resolver summary. Called by `pipeline.run` after Stage I."""
    log.info("=== resolver summary ===")
    if _resolve_cache_info is not None:
        info = _resolve_cache_info()
        total = info.hits + info.misses
        rate = (info.hits / total) if total else 0.0
        log.info(
            "  resolve cache: %d calls, %d hits (%.1f%%), %d misses, %d entries",
            total, info.hits, rate * 100, info.misses, info.currsize,
        )
    if not miss_counter:
        log.info("  (no resolver misses)")
    for entity_type, count in miss_counter.most_common():
        examples = miss_examples[entity_type].most_common(top_n)
        sample_str = ", ".join(f"{raw!r}×{n}" for raw, n in examples)
        log.info(
            "  %s: %d no_match across %d distinct raws — top: %s",
            entity_type,
            count,
            len(miss_examples[entity_type]),
            sample_str,
        )
    if exception_counter:
        log.warning("--- resolver exceptions ---")
        for (entity_type, exc), count in exception_counter.most_common():
            log.warning("  %s/%s: %d occurrences", entity_type, exc, count)
    else:
        log.info("(no resolver exceptions)")


# ---------------------------------------------------------------------------
# DuckDB-friendly wrappers around the divergence UDFs.
# ---------------------------------------------------------------------------
#
# DuckDB hands STRUCT-list params over to Python as a list of dicts.
# `compute_variant_divergence_py` consumes that shape directly. The wrapper
# below converts the divergence dict (or None) into the STRUCT shape DuckDB
# expects: when None, every field is None.


def _empty_variant_struct() -> dict:
    return {
        "has_variant_divergence": None,
        "divergence_magnitude": None,
        "threshold_used": None,
        "threshold_basis": None,
        "differing_setup_fields": None,
    }


def _empty_cross_party_struct() -> dict:
    return {
        "has_cross_party_divergence": None,
        "divergence_magnitude": None,
        "threshold_used": None,
        "threshold_basis": None,
        "differing_setup_fields": None,
        "organization_count": None,
        "scores_by_organization": None,
    }


def compute_variant_divergence_udf_body(rows, metric_config) -> dict:
    rows = [dict(r) for r in rows] if rows else []
    out = _compute_variant(rows, metric_config)
    return out if out is not None else _empty_variant_struct()


def compute_cross_party_divergence_udf_body(rows, metric_config) -> dict:
    rows = [dict(r) for r in rows] if rows else []
    out = _compute_cross_party(rows, metric_config)
    return out if out is not None else _empty_cross_party_struct()


def derive_metric_meta_udf_body(
    eee_metric_config,
    registry_metric_kind,
    registry_metric_unit,
    registry_min_score,
    registry_max_score,
    registry_lower_is_better,
    metric_name,
    registry_score_type,
) -> dict:
    """DuckDB wrapper for `metric_meta_hotfix.derive_metric_meta`.

    The first arg arrives from DuckDB as a JSON string (the per-record EEE
    `metric_config` blob). Parse on entry per the JSON-UDF convention.
    """
    from eval_card_backend.signals.setup import _coerce_json
    parsed_mc = _coerce_json(eee_metric_config, caller="derive_metric_meta_udf")
    return derive_metric_meta(
        parsed_mc,
        registry_metric_kind,
        registry_metric_unit,
        registry_min_score,
        registry_max_score,
        registry_lower_is_better,
        metric_name,
        registry_score_type=registry_score_type,
    )

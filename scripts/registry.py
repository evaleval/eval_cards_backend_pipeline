"""Registry-based benchmark identity resolution.

Wraps ``eval_entity_resolver`` so the rest of the pipeline can stamp a
canonical ``benchmark`` identity on each ``eval_summary`` regardless of the
EEE suite the row came from. Resolution is memoized on
``(raw_value, source_config)`` since the corpus has at most a few hundred
unique combinations and they re-occur across thousands of rows.

Source of truth: the HuggingFace dataset ``evaleval/entity-registry-data``.
Override with ``REGISTRY_LOCAL_PARQUET_DIR`` to load from a local parquet
directory (the directory must contain ``aliases.parquet`` per the
``AliasStore.from_parquet`` contract). Set ``REGISTRY_DISABLE=1`` to
short-circuit all lookups — useful for diagnostic runs that want to
isolate non-registry behaviour.

``benchmark`` resolution is the production path; ``model``, ``metric``,
and ``org`` resolvers are also exposed (mirroring the same pattern) to
unblock pipeline-side migration work, even though registry coverage of
those entity types is still maturing — partial migration would create
contradictory taxonomies across artifacts (see CLAUDE.md "Pending:
evalcard-registry integration sweep").
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

REGISTRY_HF_DATASET = "evaleval/entity-registry-data"

_resolver: Any = None
_resolver_init_attempted = False
_resolver_init_error: str | None = None

_canonical_benchmark_displays: dict[str, str] | None = None
_canonical_benchmark_parents: dict[str, str] | None = None


def _log(event: str, **fields: Any) -> None:
    print(f"[pipeline] {json.dumps({'event': event, **fields})}")


def _build_resolver() -> Any:
    """Construct the AliasStore-backed Resolver. Returns None on failure.

    Tries the local parquet override first when ``REGISTRY_LOCAL_PARQUET_DIR``
    is set, otherwise falls back to the published HF dataset. The HF path
    uses ``hf_hub_download`` internally, so subsequent runs reuse the
    standard HF cache without us managing a separate snapshot directory.
    """
    from eval_entity_resolver import AliasStore, Resolver

    local_dir = (os.environ.get("REGISTRY_LOCAL_PARQUET_DIR") or "").strip()
    if local_dir:
        store = AliasStore.from_parquet(local_dir, read_only=True)
        _log(
            "registry.loaded",
            source="local_parquet",
            path=local_dir,
            alias_rows=len(store.to_dataframe()),
        )
        return Resolver(store)

    store = AliasStore.from_hf(REGISTRY_HF_DATASET, read_only=True)
    _log(
        "registry.loaded",
        source="hf_dataset",
        repo=REGISTRY_HF_DATASET,
        alias_rows=len(store.to_dataframe()),
    )
    return Resolver(store)


def _get_resolver() -> Any:
    """Lazy resolver initialization. Returns None if disabled or unreachable.

    Failure is non-fatal: the pipeline keeps running with canonical_id=None
    on every summary, which mirrors the pre-registry behaviour. The
    ``registry.disabled`` / ``registry.init_failed`` log lines are how we
    surface a missing registry post-hoc.
    """
    global _resolver, _resolver_init_attempted, _resolver_init_error
    if _resolver is not None:
        return _resolver
    if _resolver_init_attempted:
        return None
    _resolver_init_attempted = True

    if os.environ.get("REGISTRY_DISABLE") == "1":
        _resolver_init_error = "REGISTRY_DISABLE=1"
        _log("registry.disabled", reason="env_flag")
        return None

    try:
        _resolver = _build_resolver()
        return _resolver
    except Exception as error:
        _resolver_init_error = str(error)
        _log("registry.init_failed", error=str(error))
        return None


@lru_cache(maxsize=2048)
def _resolve_cached(
    raw_value: str | None, source_config: str | None
) -> tuple[str | None, str, float]:
    """Inner cache. Returns (canonical_id, strategy, confidence)."""
    if not raw_value:
        return (None, "empty_input", 0.0)
    resolver = _get_resolver()
    if resolver is None:
        return (None, "registry_unavailable", 0.0)
    result = resolver.resolve(raw_value, "benchmark", source_config)
    return (result.canonical_id, result.strategy, float(result.confidence))


def _load_canonical_benchmark_displays() -> dict[str, str]:
    """Load canonical_id → display_name map from the registry's benchmarks table.

    Source: ``canonical_benchmarks/part-0.parquet`` on the registry HF dataset
    (sibling of ``aliases/part-0.parquet``). Resolver only loads aliases, so
    we read the entity table separately for display_name lookup.

    Honors REGISTRY_LOCAL_PARQUET_DIR. Returns empty dict on failure (matches
    the alias-store fallback policy).
    """
    global _canonical_benchmark_displays
    if _canonical_benchmark_displays is not None:
        return _canonical_benchmark_displays

    if os.environ.get("REGISTRY_DISABLE") == "1":
        _canonical_benchmark_displays = {}
        return _canonical_benchmark_displays

    try:
        import pandas as pd

        local_dir = (os.environ.get("REGISTRY_LOCAL_PARQUET_DIR") or "").strip()
        if local_dir:
            from pathlib import Path

            path = Path(local_dir) / "canonical_benchmarks.parquet"
            if not path.exists():
                path = Path(local_dir) / "canonical_benchmarks" / "part-0.parquet"
            df = pd.read_parquet(path)
        else:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(
                repo_id=REGISTRY_HF_DATASET,
                filename="canonical_benchmarks/part-0.parquet",
                repo_type="dataset",
            )
            df = pd.read_parquet(local)
        _canonical_benchmark_displays = {
            str(row["id"]): str(row["display_name"])
            for _, row in df.iterrows()
            if row.get("id") and row.get("display_name")
        }
        _log(
            "registry.canonical_displays_loaded",
            count=len(_canonical_benchmark_displays),
        )
    except Exception as error:
        _log("registry.canonical_displays_failed", error=str(error))
        _canonical_benchmark_displays = {}
    return _canonical_benchmark_displays


def get_canonical_display_name(canonical_id: str | None) -> str | None:
    """Return the registry display_name for a canonical benchmark id, or None."""
    if not canonical_id:
        return None
    return _load_canonical_benchmark_displays().get(canonical_id)


def _load_canonical_benchmark_parents() -> dict[str, str]:
    """Load canonical_id → parent_benchmark_id map (only entries with non-null
    parent). Used by ``get_canonical_benchmark_root`` to roll per-domain sub-
    benchmarks (e.g. ``tau2-airline``, ``tau2-retail``, ``tau2-telecom``) up to
    their parent canonical (``tau2-bench``) for catalog tile aggregation.

    Reads the same ``canonical_benchmarks/part-0.parquet`` table as
    ``_load_canonical_benchmark_displays`` and follows the same fallback
    policy (REGISTRY_LOCAL_PARQUET_DIR → HF download → empty on failure).
    """
    global _canonical_benchmark_parents
    if _canonical_benchmark_parents is not None:
        return _canonical_benchmark_parents

    if os.environ.get("REGISTRY_DISABLE") == "1":
        _canonical_benchmark_parents = {}
        return _canonical_benchmark_parents

    try:
        import pandas as pd

        local_dir = (os.environ.get("REGISTRY_LOCAL_PARQUET_DIR") or "").strip()
        if local_dir:
            from pathlib import Path

            path = Path(local_dir) / "canonical_benchmarks.parquet"
            if not path.exists():
                path = Path(local_dir) / "canonical_benchmarks" / "part-0.parquet"
            df = pd.read_parquet(path)
        else:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(
                repo_id=REGISTRY_HF_DATASET,
                filename="canonical_benchmarks/part-0.parquet",
                repo_type="dataset",
            )
            df = pd.read_parquet(local)
        parents: dict[str, str] = {}
        for _, row in df.iterrows():
            child = row.get("id")
            parent = row.get("parent_benchmark_id")
            if child and parent and not (isinstance(parent, float) and parent != parent):
                parents[str(child)] = str(parent)
        _canonical_benchmark_parents = parents
        _log("registry.canonical_parents_loaded", count=len(parents))
    except Exception as error:
        _log("registry.canonical_parents_failed", error=str(error))
        _canonical_benchmark_parents = {}
    return _canonical_benchmark_parents


def get_canonical_benchmark_root(canonical_id: str | None) -> str | None:
    """Walk the ``parent_benchmark_id`` chain to return the root canonical id.

    A root has no parent (e.g. ``tau2-bench``), so this collapses
    ``tau2-airline`` → ``tau2-bench`` and ``tau2-retail`` → ``tau2-bench``.
    Used so multi-domain sibling sub-benchmarks roll up into a single
    canonical-union tile in the catalog.

    Cycle-safe: aborts after 16 hops if the registry has a cycle (shouldn't
    happen, but parent_benchmark_id is a free-form FK in the registry seed).
    Returns the input unchanged when there's no parent or the registry isn't
    reachable.
    """
    if not canonical_id:
        return None
    parents = _load_canonical_benchmark_parents()
    current = canonical_id
    for _ in range(16):
        parent = parents.get(current)
        if not parent or parent == current:
            return current
        current = parent
    return current


def resolve_benchmark(
    raw_value: str | None, source_config: str | None
) -> dict[str, Any]:
    """Resolve a benchmark identifier to its canonical registry id.

    ``raw_value`` should be the most specific benchmark name available on
    the eval_summary (typically ``benchmark_leaf_name`` falling back to
    ``benchmark_leaf_key``). ``source_config`` MUST be the raw EEE config
    name with original punctuation (e.g. ``tau-bench-2_airline``,
    ``apex-agents``, ``global-mmlu-lite``) — the pipeline-normalized
    ``benchmark_family_key`` does NOT match the registry's
    ``scoped_aliases`` keys, which preserve hyphens.

    Returns a stable dict shape regardless of resolution outcome so the
    caller can treat the audit trail uniformly.
    """
    canonical_id, strategy, confidence = _resolve_cached(raw_value, source_config)
    return {
        "canonical_id": canonical_id,
        "strategy": strategy,
        "confidence": confidence,
        "raw_value": raw_value,
        "source_config": source_config,
    }


def _resolve_entity(
    entity_type: str,
    raw_value: str | None,
    source_config: str | None,
    failure_event: str,
) -> tuple[str | None, str, float]:
    """Shared resolver call. Returns (canonical_id, strategy, confidence)."""
    if not raw_value:
        return (None, "empty_input", 0.0)
    resolver = _get_resolver()
    if resolver is None:
        return (None, "registry_unavailable", 0.0)
    try:
        result = resolver.resolve(raw_value, entity_type, source_config)
    except Exception as error:
        _log(
            failure_event,
            error=str(error),
            raw_value=raw_value,
            source_config=source_config,
        )
        return (None, "resolver_error", 0.0)
    return (result.canonical_id, result.strategy, float(result.confidence))


@lru_cache(maxsize=2048)
def _resolve_model_cached(
    raw_value: str | None, source_config: str | None
) -> tuple[str | None, str, float]:
    return _resolve_entity(
        "model", raw_value, source_config, "registry.resolve_model_failed"
    )


def resolve_model(
    raw_value: str | None, source_config: str | None = None
) -> dict[str, Any]:
    """Resolve a model identifier to its canonical registry id.

    ``raw_value`` is typically a HuggingFace-style ``developer/model`` slug
    or whatever surface form appears on the EEE record. ``source_config``
    is the raw EEE config name (preserve original punctuation) when
    scoped lookups matter; pass ``None`` for global lookups.

    Returns a stable dict shape regardless of resolution outcome.
    """
    canonical_id, strategy, confidence = _resolve_model_cached(
        raw_value, source_config
    )
    return {
        "canonical_id": canonical_id,
        "strategy": strategy,
        "confidence": confidence,
        "raw_value": raw_value,
        "source_config": source_config,
    }


@lru_cache(maxsize=2048)
def _resolve_metric_cached(
    raw_value: str | None, source_config: str | None
) -> tuple[str | None, str, float]:
    return _resolve_entity(
        "metric", raw_value, source_config, "registry.resolve_metric_failed"
    )


def resolve_metric(
    raw_value: str | None, source_config: str | None = None
) -> dict[str, Any]:
    """Resolve a metric identifier to its canonical registry id.

    ``raw_value`` is the metric name as it appears on the evaluation_result
    (e.g. ``accuracy``, ``exact_match``, ``win_rate``). ``source_config``
    is the raw EEE config name when scoped lookups matter.
    """
    canonical_id, strategy, confidence = _resolve_metric_cached(
        raw_value, source_config
    )
    return {
        "canonical_id": canonical_id,
        "strategy": strategy,
        "confidence": confidence,
        "raw_value": raw_value,
        "source_config": source_config,
    }


@lru_cache(maxsize=2048)
def _resolve_org_cached(
    raw_value: str | None, source_config: str | None
) -> tuple[str | None, str, float]:
    return _resolve_entity(
        "org", raw_value, source_config, "registry.resolve_org_failed"
    )


def resolve_org(
    raw_value: str | None, source_config: str | None = None
) -> dict[str, Any]:
    """Resolve an organization identifier to its canonical registry id.

    ``raw_value`` is the org name as it appears on the source record
    (e.g. ``OpenAI``, ``Anthropic``, ``Stanford CRFM``). May return
    ``canonical_id=None`` until the org schema is seeded — that's
    expected; callers should fall back to existing normalization logic.
    """
    canonical_id, strategy, confidence = _resolve_org_cached(
        raw_value, source_config
    )
    return {
        "canonical_id": canonical_id,
        "strategy": strategy,
        "confidence": confidence,
        "raw_value": raw_value,
        "source_config": source_config,
    }


def reset_for_tests() -> None:
    """Clear the module-level resolver and the lru_cache. Test-only."""
    global _resolver, _resolver_init_attempted, _resolver_init_error
    global _canonical_benchmark_displays, _canonical_benchmark_parents
    _resolver = None
    _resolver_init_attempted = False
    _resolver_init_error = None
    _canonical_benchmark_displays = None
    _canonical_benchmark_parents = None
    _resolve_cached.cache_clear()
    _resolve_model_cached.cache_clear()
    _resolve_metric_cached.cache_clear()
    _resolve_org_cached.cache_clear()

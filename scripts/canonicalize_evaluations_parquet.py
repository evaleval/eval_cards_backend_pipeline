"""Post-hoc model canonicalization on output/duckdb/v1/evaluations.parquet.

Spike — does NOT modify the pipeline. Reads the existing analytics parquet,
resolves each unique `model_route_id` to a registry canonical id (using the
local registry parquet built by `scripts/build_local_model_aliases.py`),
and writes a new parquet with TWO extra columns:

  - canonical_model_id        — the size-specific canonical (e.g.
                                 `meta/llama-3.1-8b`). Tracks the registry's
                                 default grain.
  - canonical_model_family_id — the release-family rollup (e.g. `meta/llama-3.1`).
                                 Treats different sizes/SKUs of the same release
                                 as one analysis unit. Used for the provenance
                                 question per user policy 2026-05-01.
No metric canonicalization is done here. Per user policy 2026-05-01, the
registry's metric coverage (22 canonicals, some questionable aliases like
`cost` ↔ `cost-per-task`) is not yet trustworthy enough to use for
provenance grouping. The pragmatic stance: group provenance at
(model, benchmark) WITHOUT metric_key, and explicitly do not claim
"same statistic reported by multiple parties" — only "same (model, benchmark)
reported by multiple parties." See `notes/provenance-investigation.md`.

Output: `output/duckdb/v1/evaluations_canonicalized.parquet`
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Default to the registry repo's fixtures dir (where `eval-card-registry seed
# --local` writes). Override via REGISTRY_LOCAL_PARQUET_DIR if pointing at
# a different dir (e.g. legacy .cache/local_registry_with_models).
DEFAULT_REGISTRY_DIR = Path("/Users/jchim/projects/evaleval/evalcard-registry/fixtures")
LOCAL_DIR = Path(os.environ.get("REGISTRY_LOCAL_PARQUET_DIR") or str(DEFAULT_REGISTRY_DIR)).resolve()
os.environ["REGISTRY_LOCAL_PARQUET_DIR"] = str(LOCAL_DIR)

import duckdb
import pandas as pd

from eval_entity_resolver import AliasStore, Resolver

SRC = Path("output/duckdb/v1/evaluations.parquet")
DST = Path("output/duckdb/v1/evaluations_canonicalized.parquet")
CANONICAL_MODELS_PARQUET = LOCAL_DIR / "canonical_models.parquet"


def to_hf_form(route_id: str) -> str:
    return route_id.replace("__", "/", 1)


# ----------------------------------------------------------------------------
# Family rollup — collapse different sizes/SKUs of the same release into one
# `canonical_model_family_id`. Two-tier strategy:
#   (1) registry parent_model_id (most authoritative, but only ~15 entries today)
#   (2) pattern rules per-org for the high-volume canonicals
# Anything not matched falls back to canonical_id itself.
# ----------------------------------------------------------------------------

# Pattern rules: ordered list of (regex, replacement) tuples, applied in order.
#
# **Reverted to empty 2026-05-01** per user policy "model size DOES matter".
# Earlier versions of FAMILY_PATTERNS rolled up size variants
# (Llama-3.1-8b/70b/405b → Llama-3.1, Claude-3-haiku/sonnet/opus → Claude-3,
# Gemini-2.0-flash/pro/lite → Gemini-2.0, etc.) at the family level, treating
# each release as one analysis unit. We've reverted that policy: different
# sizes / SKUs / capability tiers are different products and stay separate.
#
# As a result, `canonical_model_family_id` ≈ `canonical_model_id` for most
# rows. The column is preserved in the schema for safety; any rollup that
# IS desired now flows through the registry's `parent_model_id` chain
# (e.g. meta/llama-4-scout-17b-16e → parent: meta/llama-4) which we still
# follow in `collapse_to_family()`. If we want to bring back regex-level
# rollups later, this is the place to add them — but only for true
# same-product cases (snapshots, quantization), not size/SKU collapse.
FAMILY_PATTERNS: list[tuple[re.Pattern, str]] = []


def _build_parent_map() -> dict[str, str]:
    """Read canonical_models.parquet for parent_model_id mappings."""
    if not CANONICAL_MODELS_PARQUET.exists():
        return {}
    df = pd.read_parquet(CANONICAL_MODELS_PARQUET, columns=["id", "parent_model_id"])
    df = df[df["parent_model_id"].notna()]
    return dict(zip(df["id"], df["parent_model_id"]))


def collapse_to_family(canonical_id: str, parent_map: dict[str, str]) -> str:
    """Return the release-family rollup for a canonical_id."""
    if not canonical_id:
        return canonical_id
    # Try parent_model_id first (most authoritative)
    parent = parent_map.get(canonical_id)
    if parent:
        # Recurse to handle multi-level (rare, but cheap)
        return collapse_to_family(parent, parent_map)
    # Try pattern rules
    for pattern, replacement in FAMILY_PATTERNS:
        new = pattern.sub(replacement, canonical_id)
        if new != canonical_id:
            return new
    # Fallback — already at family level
    return canonical_id


def main() -> None:
    store = AliasStore.from_parquet(str(LOCAL_DIR), read_only=True)
    resolver = Resolver(store)
    parent_map = _build_parent_map()
    print(f"  parent_model_id map: {len(parent_map)} entries")

    con = duckdb.connect()
    routes = [r[0] for r in con.execute(
        f"SELECT DISTINCT model_route_id FROM '{SRC}' WHERE model_route_id IS NOT NULL"
    ).fetchall()]
    print(f"  {len(routes)} unique route_ids to resolve")

    canonical_map: dict[str, str | None] = {}
    family_map: dict[str, str | None] = {}
    n_resolved = 0
    for route in routes:
        result = resolver.resolve(to_hf_form(route), "model", None)
        canonical = result.canonical_id
        canonical_map[route] = canonical
        if canonical:
            n_resolved += 1
            family_map[route] = collapse_to_family(canonical, parent_map)
        else:
            family_map[route] = None
    print(f"  resolved: {n_resolved} / {len(routes)} ({100*n_resolved/len(routes):.1f}%)")

    # Re-resolve benchmarks with the latest registry state. The pipeline
    # canonicalized benchmarks at run time using the HF dataset; we want to
    # apply the local registry's newer aliases (e.g. `artificial_analysis.*`
    # → mmlu-pro, gpqa, etc.) on top.
    bench_inputs = con.execute(
        f"""SELECT DISTINCT benchmark_leaf_name, benchmark_family_key
            FROM '{SRC}' WHERE benchmark_leaf_name IS NOT NULL"""
    ).fetchall()
    print(f"  {len(bench_inputs)} unique (benchmark_leaf_name, family_key) pairs to resolve")
    bench_map: dict[tuple[str, str | None], str | None] = {}
    n_bench_resolved = 0
    for leaf, fam in bench_inputs:
        # Try with source_config = family_key first (scoped match), then global
        res = resolver.resolve(leaf, "benchmark", fam)
        if not res.canonical_id:
            res = resolver.resolve(leaf, "benchmark", None)
        bench_map[(leaf, fam)] = res.canonical_id
        if res.canonical_id:
            n_bench_resolved += 1
    print(f"  bench resolved: {n_bench_resolved} / {len(bench_inputs)} "
          f"({100*n_bench_resolved/max(len(bench_inputs),1):.1f}%)")

    # No metric canonicalization. Per user policy 2026-05-01, the registry's
    # metric coverage is too thin and contains questionable aliases (e.g.
    # `cost` → `cost-per-task` is not generally correct). Provenance
    # grouping should be at (model, benchmark) only; do not claim
    # metric-level verification.

    # Show top family collapses for spot-check
    from collections import Counter
    family_counts: Counter[str] = Counter()
    for route, fam in family_map.items():
        if fam:
            family_counts[fam] += 1
    print(f"  top family rollups (route_ids per family):")
    for fam, n in family_counts.most_common(15):
        print(f"    {fam:40s}  {n} routes")

    df = pd.read_parquet(SRC)
    print(f"  loaded {len(df):,} rows from {SRC}")

    df["canonical_model_id"] = df["model_route_id"].map(canonical_map).fillna(df["model_route_id"])
    df["canonical_model_family_id"] = df["model_route_id"].map(family_map).fillna(df["model_route_id"])

    # Re-resolve benchmarks: replace canonical_benchmark_id and
    # benchmark_grouping_key based on local registry. We DO NOT touch
    # benchmark_family_key / benchmark_leaf_key / benchmark_leaf_name.
    bench_keys = list(zip(df["benchmark_leaf_name"], df["benchmark_family_key"]))
    new_canonical_bench = [bench_map.get((l, f)) for l, f in bench_keys]
    # Preserve the pipeline-emission canonical_benchmark_id for diff
    df["canonical_benchmark_id_orig"] = df["canonical_benchmark_id"]
    df["canonical_benchmark_id"] = new_canonical_bench
    # benchmark_grouping_key: canonical-or-summary fallback (matches
    # `pipeline._summary_canonical_grouping_key`)
    df["benchmark_grouping_key"] = [
        c if c else f"summary:{s}"
        for c, s in zip(new_canonical_bench, df["eval_summary_id"])
    ]

    n_can = (df["model_route_id"] != df["canonical_model_id"]).sum()
    n_fam = (df["model_route_id"] != df["canonical_model_family_id"]).sum()
    n_bench_changed = (df["canonical_benchmark_id"] != df["canonical_benchmark_id_orig"]).sum()
    print(f"  rows route != canonical:                 {n_can:,}")
    print(f"  rows route != family:                    {n_fam:,}")
    print(f"  rows benchmark canonical changed:        {n_bench_changed:,}")
    print(f"  unique route_ids:                        {df['model_route_id'].nunique():,}")
    print(f"  unique canonical_model_ids:              {df['canonical_model_id'].nunique():,}")
    print(f"  unique canonical_model_family:           {df['canonical_model_family_id'].nunique():,}")
    print(f"  unique canonical_benchmark_ids (pipeline-emit): {df['canonical_benchmark_id_orig'].nunique():,}")
    print(f"  unique canonical_benchmark_ids (after relocal): {df['canonical_benchmark_id'].nunique():,}")
    print(f"  unique benchmark_grouping_key:           {df['benchmark_grouping_key'].nunique():,}")

    df.to_parquet(DST, index=False)
    print(f"  wrote {DST}")


if __name__ == "__main__":
    main()

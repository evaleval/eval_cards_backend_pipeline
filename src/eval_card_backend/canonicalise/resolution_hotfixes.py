"""Temporary fact-level resolution hot fixes.

Sibling of `hierarchy_hotfixes.py`, but operates one stage earlier: these
functions run inside Stage C on the `results_resolved` table, before slice
keys are derived. They compensate for upstream data issues that can't be
expressed as registry aliases because the fix depends on a *cross-field*
condition the resolver can't see (it resolves each field independently).

Every function has a lifecycle annotation: what removes the need for it.
All functions mutate `results_resolved` in place via the DuckDB connection.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


# ── 1. HELM composite-aggregate rows ─────────────────────────────────
# TODO(upstream): the real fix is in HELM ingestion. HELM emits its
# composite-level aggregate ("Mean win rate" / "Mean score") with the
# *metric name in the benchmark field* and a generic "rank"/"score" in
# the metric field. A registry alias can't repair this: benchmark and
# metric resolve independently, so there's no field the metric resolver
# can read that says "this rank is really a mean-win-rate", and aliasing
# `rank` → `mean win rate` would be both fragile (future data) and
# semantically false. When HELM ingestion is corrected to put the
# aggregate metric in the metric field, delete this function — the rows
# will then resolve cleanly on their own.
#
# Until then: for exactly the malformed rows (benchmark_raw is the
# aggregate label within a HELM composite), correct the field placement —
# benchmark ← the composite-overall canonical, metric ← the shared
# `mean-win-rate` / `mean-score` canonical. The benchmark already carries
# the per-tier namespace (helm-classic-leaderboard vs helm-lite-leaderboard),
# so the metric stays the plain canonical concept. This also kills the
# helm_mmlu → `mmlu` collision (those rows otherwise become a bogus
# "mean win rate" slice on the real MMLU benchmark); it runs before
# `_apply_slice_key` so the collision never forms a slice.

# source_config → (composite-overall benchmark id, canonical metric id).
# Narrow + enumerated on purpose: only these six HELM tiers, only the two
# aggregate labels below. Never touches a genuine rank/score elsewhere.
_HELM_AGGREGATE_MAP = {
    "helm_classic": ("helm-classic-leaderboard", "mean-win-rate"),
    "helm_lite": ("helm-lite-leaderboard", "mean-win-rate"),
    "helm_instruct": ("helm-instruct-leaderboard", "mean-win-rate"),
    "helm_mmlu": ("helm-mmlu-leaderboard", "mean-win-rate"),
    "helm_capabilities": ("helm-capabilities-leaderboard", "mean-score"),
    "helm_safety": ("helm-safety-leaderboard", "mean-score"),
}

_HELM_AGGREGATE_LABELS = ("mean win rate", "mean score")


def fix_helm_composite_aggregates(con) -> None:
    """Reassign benchmark + metric for HELM composite-aggregate rows.

    Fires only on rows where `benchmark_raw` is one of the aggregate
    labels within one of the six HELM composite source_configs.
    """
    mapping_values = ", ".join(
        f"('{sc}', '{bid}', '{mid}')"
        for sc, (bid, mid) in _HELM_AGGREGATE_MAP.items()
    )
    labels_sql = ", ".join(f"'{lbl}'" for lbl in _HELM_AGGREGATE_LABELS)

    con.execute(
        f"""
        UPDATE results_resolved AS r
        SET benchmark_id = m.bid,
            metric_id    = m.mid,
            benchmark_resolution_strategy = 'hotfix_helm_aggregate',
            metric_resolution_strategy    = 'hotfix_helm_aggregate'
        FROM (VALUES {mapping_values}) AS m(sc, bid, mid)
        WHERE r.source_config = m.sc
          AND LOWER(TRIM(r.benchmark_raw)) IN ({labels_sql})
        """
    )
    n = con.execute(
        f"""
        SELECT COUNT(*) FROM results_resolved
        WHERE source_config IN ({", ".join(f"'{sc}'" for sc in _HELM_AGGREGATE_MAP)})
          AND benchmark_resolution_strategy = 'hotfix_helm_aggregate'
        """
    ).fetchone()[0]
    log.info("resolution_hotfixes: reassigned %d HELM composite-aggregate row(s)", n)


# ── 2. Vague / malformed metric labels ───────────────────────────────
# TODO(upstream): the source emits metric labels that don't resolve to a
# canonical, so the metric renders blank. Two kinds handled here:
#   - "mean": a real value but vague — its *meaning* differs per benchmark
#     (cvebench success vs cyse2 exploit vs swebench resolution), so it must
#     not be conflated into one global "mean" metric.
#   - codegolf's metric field holds a junk benchmark-name string
#     ("Codegolf v2.2 benchmark") — same malformed-field class as HELM.
# Until the source emits real metric names, namespace them by benchmark so
# they (a) display and (b) stay distinct. Matching placeholder canonicals
# live in metrics.yaml (lower_is_better unset → direction stays per-row).
# When the source is fixed, delete this + those placeholder canonicals.


def fix_vague_metric_labels(con) -> None:
    """Namespace vague/malformed metric labels by their benchmark.

    Override (not IS-NULL-guarded): these raws ("mean", codegolf's junk
    string) never carry a real canonical, but the registry's normalized/
    fuzzy matching can still mis-resolve them to a same-token namespaced
    placeholder. We overwrite unconditionally so the final id is always the
    correct <benchmark_id>.<suffix> regardless of what the resolver guessed.
    """
    # "mean" → "<benchmark_id>.mean" (benchmark must have resolved).
    con.execute(
        """
        UPDATE results_resolved
        SET metric_id = benchmark_id || '.mean',
            metric_resolution_strategy = 'hotfix_vague_metric'
        WHERE LOWER(TRIM(metric_raw)) = 'mean'
          AND benchmark_id IS NOT NULL
        """
    )
    # codegolf junk metric string → "codegolf.score".
    con.execute(
        """
        UPDATE results_resolved
        SET metric_id = 'codegolf.score',
            metric_resolution_strategy = 'hotfix_vague_metric'
        WHERE metric_raw = 'Codegolf v2.2 benchmark'
          AND benchmark_id = 'codegolf'
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE metric_resolution_strategy = 'hotfix_vague_metric'"
    ).fetchone()[0]
    log.info("resolution_hotfixes: namespaced %d vague/malformed metric label(s)", n)

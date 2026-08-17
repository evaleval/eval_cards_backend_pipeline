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


# ── 3. inspect_ai/harbor scorer-wrapper benchmark labels ─────────────
# TODO(upstream): these EEE submissions were converted from inspect_ai /
# harbor logs whose *task* name is per-model ("full-solver-<model>") or
# per-run ("swebenchpro/S-adaptive/+1ep/<hash>"), and the converter wrote
# the whole "<metric> on <task> for scorer <scorer>" label into
# evaluation_name — the field the pipeline treats as benchmark identity.
# Each record therefore mints a unique unresolvable benchmark, and the
# frontend fans one benchmark out into N single-model "benchmarks".
# A registry alias can't repair this: aliases are literal strings and the
# set is open (every new model/run mints a new one). When the upstream
# submissions carry the real benchmark name in evaluation_name, delete
# this function, its map, and the `l2-bench.mean` placeholder canonical
# in the registry's metrics.yaml (seeded for the same reason).
#
# Scope is deliberately enumerated: only these source_configs, only rows
# whose benchmark_raw matches the wrapper shape. The task-variant suffixes
# (S-adaptive/+Nep/<hash>) collapse into the config's single benchmark;
# they resurface as variant submissions of the same (model, benchmark,
# metric) triple, which is the intended modelling for setup variants.

# source_config → canonical benchmark id.
_SCORER_WRAPPER_MAP = {
    "l2-bench": "l2-bench",
    "terminalbench": "terminal-bench",
    "swebenchpro": "swe-bench-pro",
}

# "<word> on <anything> for scorer <word>" — the converter's label shape.
_SCORER_WRAPPER_REGEX = r"^\S+ on .+ for scorer \S+$"


def fix_scorer_wrapper_benchmarks(con) -> None:
    """Reassign benchmark for scorer-wrapper labelled rows.

    Fires only on rows in the enumerated source_configs whose
    `benchmark_raw` matches the wrapper shape. Runs before
    `fix_vague_metric_labels` so l2-bench's bare "mean" metric gets
    namespaced to `l2-bench.mean` once the benchmark is assigned.
    """
    mapping_values = ", ".join(
        f"('{sc}', '{bid}')" for sc, bid in _SCORER_WRAPPER_MAP.items()
    )
    con.execute(
        f"""
        UPDATE results_resolved AS r
        SET benchmark_id = m.bid,
            benchmark_resolution_strategy = 'hotfix_scorer_wrapper'
        FROM (VALUES {mapping_values}) AS m(sc, bid)
        WHERE r.source_config = m.sc
          AND regexp_matches(r.benchmark_raw, '{_SCORER_WRAPPER_REGEX}')
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE benchmark_resolution_strategy = 'hotfix_scorer_wrapper'"
    ).fetchone()[0]
    log.info(
        "resolution_hotfixes: reassigned %d scorer-wrapper benchmark row(s)", n
    )


# ── 4. HLE leaderboard calibration column mislabelled "score" ────────
# TODO(upstream): the HLE leaderboard feed labels its calibration-error
# column literally "score" (values 34–89, anti-correlated with accuracy),
# so the resolver lands it on the generic `score` canonical. A registry
# alias can't fix a raw called "score" without breaking `score` everywhere
# else. Delete this when the EEE hle ingestion relabels the column; the
# (hle, score → accuracy) registry fold stays valid on its own then.
# Until then this MUST run before apply_metric_folds — the fold would
# otherwise sweep these calibration rows into the accuracy view.
# NOTE: with catch-all-aware registry data, the structured metric-id
# pre-step already resolves hle's `hle.calibration_error` ids to
# calibration-error and this UPDATE matches 0 rows. It stays as the
# fallback for the pre-step's disabled mode (registry data without
# catch_all flags).


def fix_hle_calibration_error(con) -> None:
    """Re-key hle-source rows whose raw metric is 'score' to the
    calibration-error canonical (lower is better)."""
    con.execute(
        """
        UPDATE results_resolved
        SET metric_id = 'calibration-error',
            metric_resolution_strategy = 'hotfix_hle_calibration'
        WHERE source_config = 'hle'
          AND benchmark_id = 'hle'
          AND LOWER(TRIM(metric_raw)) = 'score'
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE metric_resolution_strategy = 'hotfix_hle_calibration'"
    ).fetchone()[0]
    log.info("resolution_hotfixes: re-keyed %d hle calibration row(s)", n)


# ── 5. HAL SciCode main-problem rate labelled generic "score" ────────
# TODO(upstream): HAL's SciCode leaderboard reports the MAIN-problem
# solve rate while every other scicode source reports the sub-problem
# rate; both arrive as raw "score". Different counting style = different
# metric (maintainer ruling 2026-08-16) — re-key HAL's rows to the
# scicode.main canonical so the merged default table never mixes the
# two. Delete when the EEE hal-scicode ingestion labels the metric.


def fix_scicode_hal_main_rate(con) -> None:
    """Re-key hal-scicode rows from generic `score` to `scicode.main`."""
    con.execute(
        """
        UPDATE results_resolved
        SET metric_id = 'scicode.main',
            metric_resolution_strategy = 'hotfix_scicode_main'
        WHERE source_config = 'hal-scicode'
          AND benchmark_id = 'scicode'
          AND LOWER(TRIM(metric_raw)) = 'score'
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE metric_resolution_strategy = 'hotfix_scicode_main'"
    ).fetchone()[0]
    log.info("resolution_hotfixes: re-keyed %d hal scicode main-rate row(s)", n)

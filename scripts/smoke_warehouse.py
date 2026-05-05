"""Distributional smoke tests against a warehouse snapshot.

Run after a pipeline emit to gate the snapshot against silent regressions.

    uv run python scripts/smoke_warehouse.py [warehouse/<snapshot_dir>]

Default: latest snapshot under `warehouse/`. Exit code:
  0 — all hard gates green
  1 — at least one hard gate fired (regression)
  2 — bad invocation (no snapshot found, missing parquet, etc.)

Hard gates fire (exit 1) on:
  - fact_id collisions (the (snapshot_id, fact_id) primary-key contract is broken)
  - any non-harness entity resolution rate < 50% (alias coverage collapse)
  - comparability singletons > 80% (grouping is producing too few peer rows
    to compute divergence on)

Everything else is informational: distribution breakdowns, top-N unresolved
raws driving registry alias backfill, parent-collapse candidates,
threshold/unit consistency, completeness partial-fields rate. Soft warnings
log but do not exit non-zero.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

log = logging.getLogger("smoke")


# ---------------------------------------------------------------------------
# Thresholds — adjust as the dataset matures.
# ---------------------------------------------------------------------------

MIN_RESOLUTION_RATE_NON_HARNESS = 0.50
MAX_SINGLETON_GROUP_RATE = 0.80
TOP_N_DEFAULT = 20
PARENT_COLLAPSE_THRESHOLD = 5


def _open(snapshot_dir: Path) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB and register the warehouse parquets as views."""
    con = duckdb.connect()
    fact = snapshot_dir / "fact_results.parquet"
    if not fact.exists():
        raise FileNotFoundError(f"missing {fact}; not a warehouse snapshot")
    con.execute(f"CREATE VIEW fact_results AS SELECT * FROM read_parquet('{fact}')")
    for table in ("benchmarks", "models", "canonical_metrics"):
        path = snapshot_dir / f"{table}.parquet"
        if path.exists():
            con.execute(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')"
            )
    return con


# Surfaces which metric families are firing the anomaly flag; a sudden empty
# result usually means the anomaly rule was narrowed accidentally.
def score_scale_anomaly_breakdown(con) -> None:
    rows = con.execute(
        """
        SELECT
            metric_unit,
            metric_kind,
            COUNT(*)                                AS n,
            COUNT(*) FILTER (WHERE score_scale_anomaly) AS anomalies
        FROM fact_results
        GROUP BY 1, 2
        HAVING COUNT(*) FILTER (WHERE score_scale_anomaly) > 0
        ORDER BY anomalies DESC
        """
    ).fetchall()
    log.info("score_scale_anomaly breakdown:")
    if not rows:
        log.info("  (no anomalies)")
    for unit, kind, n, anom in rows:
        log.info(
            "  unit=%s kind=%s: %d / %d (%.1f%%)",
            unit, kind, anom, n, 100 * anom / max(n, 1),
        )


# Cross-checks that the anomaly flag covers the bounded violations.
# anomaly_total should be ≥ (below_min + above_max); divergence means
# the anomaly rule has lost coverage of one of the bound clauses.
def score_outside_bounds(con) -> None:
    out = con.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE min_score IS NOT NULL AND score < min_score) AS below_min,
            COUNT(*) FILTER (WHERE max_score IS NOT NULL AND score > max_score) AS above_max,
            COUNT(*) FILTER (WHERE score_scale_anomaly) AS anomaly_total
        FROM fact_results
        """
    ).fetchone()
    below, above, anom = out
    log.info(
        "score-outside-bounds: below_min=%d above_max=%d (anomaly_total=%d)",
        below, above, anom,
    )
    if (below + above) > anom:
        log.warning(
            "  WARN: %d rows outside bounds but only %d flagged — anomaly rule lost coverage",
            below + above, anom,
        )


# (snapshot_id, fact_id) is the primary-key contract for downstream JOINs;
# any duplicate row multiplies join output. HARD gate.
def fact_id_collisions(con) -> bool:
    out = con.execute(
        """
        SELECT
            COUNT(*) - COUNT(DISTINCT fact_id) AS dup_rows,
            COUNT(*) FILTER (WHERE fact_id IS NULL) AS null_fact_ids
        FROM fact_results
        """
    ).fetchone()
    dups, nulls = out
    log.info("fact_id: %d duplicate rows, %d NULL fact_ids", dups, nulls)
    if dups > 0:
        log.error("  HARD: fact_id dedup regressed; %d dup rows in warehouse", dups)
        return False
    return True


# Non-harness resolution dropping below 50% means alias coverage in the
# registry collapsed; downstream leaderboards lose the ability to group rows
# by canonical id. Harness is excluded from the gate because alias coverage
# there is independently tracked (harness_resolution / unresolved_harness)
# and historically lags the others. HARD gate.
def resolution_rates(con) -> bool:
    out = con.execute(
        """
        SELECT
            COUNT(*)                                       AS n,
            COUNT(*) FILTER (WHERE model_id IS NOT NULL)     AS model,
            COUNT(*) FILTER (WHERE benchmark_id IS NOT NULL) AS bench,
            COUNT(*) FILTER (WHERE metric_id IS NOT NULL)    AS metric,
            COUNT(*) FILTER (WHERE org_id IS NOT NULL)       AS org,
            COUNT(*) FILTER (WHERE harness_id IS NOT NULL)   AS harness
        FROM fact_results
        """
    ).fetchone()
    n = out[0] or 1
    model, bench, metric, org, harness = (out[1] / n, out[2] / n, out[3] / n,
                                          out[4] / n, out[5] / n)
    log.info(
        "resolution rate (n=%d): model=%.1f%% bench=%.1f%% metric=%.1f%% org=%.1f%% harness=%.1f%%",
        out[0], 100 * model, 100 * bench, 100 * metric, 100 * org, 100 * harness,
    )
    failed = []
    for label, rate in (("model", model), ("benchmark", bench),
                        ("metric", metric), ("org", org)):
        if rate < MIN_RESOLUTION_RATE_NON_HARNESS:
            failed.append((label, rate))
    if failed:
        for label, rate in failed:
            log.error(
                "  HARD: %s resolution rate %.1f%% below %.0f%% threshold",
                label, 100 * rate, 100 * MIN_RESOLUTION_RATE_NON_HARNESS,
            )
        return False
    return True


# Sentinel-version stripping in Stage C lifts harness resolution above zero;
# sustained improvement requires registry alias coverage. Tracked as a soft
# signal so a temporary alias regression doesn't fail CI.
def harness_resolution(con) -> None:
    out = con.execute(
        """
        SELECT
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE harness_id IS NOT NULL) AS resolved
        FROM fact_results
        """
    ).fetchone()
    n, resolved = out
    rate = resolved / max(n, 1)
    log.info("harness resolution: %.1f%% (%d / %d)", 100 * rate, resolved, n)
    if rate == 0:
        log.warning(
            "  WARN: harness resolution = 0%%; check Stage C eval_library handling"
        )


# Operational output: paste straight into a registry alias-backfill ticket.
def unresolved_harness(con, top_n: int = TOP_N_DEFAULT) -> None:
    rows = con.execute(
        f"""
        SELECT harness_raw, COUNT(*) AS n
        FROM fact_results
        WHERE harness_id IS NULL AND harness_raw IS NOT NULL
        GROUP BY 1
        ORDER BY n DESC
        LIMIT {top_n}
        """
    ).fetchall()
    log.info("top-%d unresolved harness_raw:", top_n)
    for raw, n in rows:
        log.info("  %dx %s", n, raw)


# A canonical_id that absorbs many distinct raws often signals subtask-level
# granularity being collapsed into a parent (e.g. all MMLU subtasks resolving
# to one `mmlu` row), which silently mis-attributes per-subtask variance to
# variant divergence. Same data the producer logs; this exposes it from
# warehouse parquet for ad-hoc inspection.
def canonical_id_collapse(con, top_n: int = TOP_N_DEFAULT) -> None:
    log.info("canonical_id absorbing >%d raws (parent-collapse candidates):",
             PARENT_COLLAPSE_THRESHOLD)
    for entity, raw_col, id_col in (
        ("model", "model_raw", "model_id"),
        ("benchmark", "benchmark_raw", "benchmark_id"),
        ("metric", "metric_raw", "metric_id"),
        ("org", "org_raw", "org_id"),
    ):
        rows = con.execute(
            f"""
            SELECT {id_col}, COUNT(DISTINCT {raw_col}) AS n_raws
            FROM fact_results
            WHERE {id_col} IS NOT NULL
            GROUP BY 1
            HAVING COUNT(DISTINCT {raw_col}) > {PARENT_COLLAPSE_THRESHOLD}
            ORDER BY n_raws DESC
            LIMIT {top_n}
            """
        ).fetchall()
        if not rows:
            continue
        log.info("  %s:", entity)
        for cid, n in rows:
            log.info("    %s ← %d raws", cid, n)


# A high singleton rate means most comparability groups have no peer row and
# divergence can't be computed for them; downstream analytics lose coverage
# even though the rows themselves are present. HARD gate.
def group_size_distribution(con) -> bool:
    out = con.execute(
        """
        WITH per_group AS (
            SELECT comparability_group_id, COUNT(*) AS sz
            FROM fact_results
            WHERE comparability_group_id IS NOT NULL
            GROUP BY 1
        )
        SELECT
            COUNT(*) AS n_groups,
            COUNT(*) FILTER (WHERE sz = 1) AS singletons
        FROM per_group
        """
    ).fetchone()
    n, singletons = out
    rate = singletons / max(n, 1)
    log.info("comparability groups: n=%d singletons=%d (%.1f%%)",
             n, singletons, 100 * rate)
    if rate > MAX_SINGLETON_GROUP_RATE:
        log.error(
            "  HARD: singleton rate %.1f%% > %.0f%% threshold; grouping likely broken",
            100 * rate, 100 * MAX_SINGLETON_GROUP_RATE,
        )
        return False
    return True


# When metric_unit and the row's threshold_basis disagree, the divergence
# threshold the row was judged against doesn't reflect the row's actual unit
# (e.g. a percent-scale row judged at the 0.05 absolute threshold instead of
# 5.0). The fix is registry-side metric_unit consistency; this query
# surfaces the per-row impact.
def threshold_basis_vs_unit(con) -> None:
    out = con.execute(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE metric_unit = 'percent'
                  AND variant_threshold_basis = 'proportion'
            ) AS percent_judged_proportion,
            COUNT(*) FILTER (
                WHERE metric_unit = 'proportion'
                  AND variant_threshold_basis = 'percent'
            ) AS proportion_judged_percent
        FROM fact_results
        WHERE variant_threshold_basis IS NOT NULL
        """
    ).fetchone()
    pp, pp2 = out
    log.info(
        "threshold_basis ↔ metric_unit mismatch: percent_judged_proportion=%d proportion_judged_percent=%d",
        pp, pp2,
    )
    if pp + pp2 > 0:
        log.warning(
            "  WARN: %d rows judged at the wrong threshold scale; "
            "fix in registry.canonical_metrics.metric_unit",
            pp + pp2,
        )


# Tracks the rate at which any partial-coverage completeness field
# (autobenchmarkcard.data subitems today) has 0 < score < 1. A 100% empty
# rate means every card filled either all subitems or none — informational
# until cards heterogeneity changes.
def partial_fields_empty_rate(con) -> None:
    out = con.execute(
        """
        SELECT
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE length(completeness_partial_fields) = 0) AS empty
        FROM fact_results
        """
    ).fetchone()
    n, empty = out
    rate = empty / max(n, 1)
    log.info("completeness_partial_fields empty: %.1f%% (%d / %d)",
             100 * rate, empty, n)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _latest_snapshot(warehouse: Path) -> Path | None:
    if not warehouse.is_dir():
        return None
    candidates = sorted(p for p in warehouse.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot",
        nargs="?",
        type=Path,
        help="Snapshot dir (default: latest under warehouse/)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    snapshot = args.snapshot or _latest_snapshot(Path("warehouse"))
    if snapshot is None or not snapshot.is_dir():
        log.error("no warehouse snapshot found (looked under warehouse/)")
        return 2
    log.info("smoke test against snapshot: %s", snapshot)

    try:
        con = _open(snapshot)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2

    hard_ok = True
    score_scale_anomaly_breakdown(con)
    score_outside_bounds(con)
    hard_ok &= fact_id_collisions(con)
    hard_ok &= resolution_rates(con)
    harness_resolution(con)
    unresolved_harness(con)
    canonical_id_collapse(con)
    hard_ok &= group_size_distribution(con)
    threshold_basis_vs_unit(con)
    partial_fields_empty_rate(con)

    if not hard_ok:
        log.error("smoke FAILED: at least one hard gate tripped")
        return 1
    log.info("smoke OK: all hard gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())

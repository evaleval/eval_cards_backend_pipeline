"""Survey resolution gaps in the canonicalized evaluations parquet.

Prints two prioritized lists:
  1. High-volume unresolved model_route_ids (especially cross-org)
  2. New summary: benchmark buckets (likely registry coverage gaps)

Used after each pipeline rebuild + post-hoc canonicalize step to identify
where the next round of registry expansion should focus.

Usage:
  uv run --with duckdb --no-project python scripts/survey_resolution_gaps.py
"""
from __future__ import annotations

import duckdb

NEW = 'output/duckdb/v1/evaluations_canonicalized.parquet'


def main() -> None:
    con = duckdb.connect()

    print('=' * 72)
    print(' RESOLUTION GAP SURVEY')
    print('=' * 72)

    # Headline coverage stats
    print('\n[Coverage]')
    print(con.execute(
        f"SELECT COUNT(*) AS total_rows, "
        f"COUNT(DISTINCT model_route_id) AS routes, "
        f"COUNT(DISTINCT canonical_model_id) AS canonical_models, "
        f"COUNT(DISTINCT canonical_model_family_id) AS families, "
        f"COUNT(DISTINCT canonical_benchmark_id) AS canonical_benches, "
        f"COUNT(DISTINCT benchmark_grouping_key) AS bgk "
        f"FROM '{NEW}'"
    ).fetchone())

    # Model gap — high-volume unresolved cross-org routes
    print('\n[Model gap] Top 30 unresolved cross-org model_route_ids '
          '(canonical_model_id == model_route_id, ≥2 orgs)')
    rows = con.execute(f"""
    WITH per_route AS (
      SELECT model_route_id, COUNT(*) AS rows,
             COUNT(DISTINCT source_organization_normalized) AS orgs,
             COUNT(DISTINCT canonical_benchmark_id) AS canon_benches,
             STRING_AGG(DISTINCT source_organization_normalized, ',' ORDER BY source_organization_normalized) AS olist
      FROM '{NEW}'
      WHERE model_route_id = canonical_model_id
      GROUP BY 1
    )
    SELECT model_route_id, rows, orgs, canon_benches, olist
    FROM per_route
    WHERE orgs > 1
    ORDER BY orgs DESC, rows DESC
    LIMIT 30
    """).fetchall()
    print(f"  {'route':62s} {'rows':>6} {'orgs':>4} {'cb':>3}  org_list")
    for r in rows:
        print(f"  {r[0]:62s} {r[1]:>6,} {r[2]:>4} {r[3]:>3}  {r[4][:40]}")

    print('\n[Model gap] Top 30 highest-volume unresolved (any org count)')
    rows = con.execute(f"""
    WITH per_route AS (
      SELECT model_route_id, COUNT(*) AS rows,
             COUNT(DISTINCT source_organization_normalized) AS orgs,
             STRING_AGG(DISTINCT source_organization_normalized, ',' ORDER BY source_organization_normalized) AS olist
      FROM '{NEW}'
      WHERE model_route_id = canonical_model_id
      GROUP BY 1
    )
    SELECT model_route_id, rows, orgs, olist
    FROM per_route
    ORDER BY rows DESC
    LIMIT 30
    """).fetchall()
    for r in rows:
        cross = " ←CROSS" if r[2] > 1 else ""
        print(f"  {r[0]:62s} {r[1]:>6,}r  {r[2]} org [{r[3][:40]}]{cross}")

    # Bench gap — new summary: buckets
    print('\n[Bench gap] Top 30 summary: fallback buckets (canonical_benchmark_id IS NULL)')
    rows = con.execute(f"""
    SELECT benchmark_grouping_key, benchmark_leaf_name, eval_summary_id, COUNT(*) AS rows
    FROM '{NEW}'
    WHERE canonical_benchmark_id IS NULL
    GROUP BY 1, 2, 3
    ORDER BY rows DESC
    LIMIT 30
    """).fetchall()
    print(f"  {'leaf_name':40s}  {'eval_summary_id':40s}  rows")
    for r in rows:
        print(f"  {r[1]!r:38s}  {r[2]:38s}  {r[3]:>6,}")

    # New entities not seen pre-refresh — diff against backup if available
    print('\n[Diff vs backup if available]')
    import os
    import glob
    backups = sorted(glob.glob('output_backup_*'), reverse=True)
    if backups:
        bak = f"{backups[0]}/duckdb/v1/evaluations_canonicalized.parquet"
        if not os.path.exists(bak):
            bak = f"{backups[0]}/duckdb/v1/evaluations.parquet"
        if os.path.exists(bak):
            print(f"  Using backup: {bak}")
            new_routes = con.execute(f"""
            SELECT model_route_id FROM '{NEW}'
            EXCEPT
            SELECT model_route_id FROM '{bak}'
            """).fetchall()
            old_routes = con.execute(f"""
            SELECT model_route_id FROM '{bak}'
            EXCEPT
            SELECT model_route_id FROM '{NEW}'
            """).fetchall()
            print(f"  routes added since backup: {len(new_routes)}")
            print(f"  routes dropped since backup: {len(old_routes)}")
            new_benches = con.execute(f"""
            SELECT benchmark_grouping_key FROM '{NEW}'
            EXCEPT
            SELECT benchmark_grouping_key FROM '{bak}'
            """).fetchall()
            print(f"  benchmark_grouping_keys added: {len(new_benches)}")
            if new_benches:
                print(f"  sample new benches: {[r[0] for r in new_benches[:10]]}")
        else:
            print(f"  (backup parquet not found at {bak})")
    else:
        print("  (no output_backup_* found)")


if __name__ == "__main__":
    main()

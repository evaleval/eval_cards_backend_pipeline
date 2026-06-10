"""Stage J — JSON sidecars for the view layer.

Five small documents the frontend reads alongside the view parquets:

- `manifest.json` — corpus-level scalars (model_count, eval_count, …).
- `headline.json` — corpus signal aggregates with stratified by-tag
  blocks. Drives the home-page corpus signal strip.
- `hierarchy.json` — six-level rollout tree (families → composites →
  benchmarks → metrics). Drives the home-page rollout strip + family
  detail page.
- `comparison-index.json` — per-(eval, metric) leaderboards plus an inverse
  model→peer index. Backs the model-detail grid view; without it the grid
  renders empty regardless of how many cells the model has.
- `benchmark_index.json` — per-benchmark cross-composite appearance index.
  For each canonical `benchmark_id`, lists every composite reporting it
  with the primary-metric aggregate stats (avg/top score, models_count).
  Lets a benchmark-detail page render "HELM/MMLU avg=X across N models,
  Open LLM v2/MMLU avg=Y across M models" without scanning the full
  hierarchy tree.

These are emitted by Python serialisation rather than DuckDB COPY because
they're scalar/JSON-shaped, not columnar, and the frontend reads them as
JSON. Stage cache integration deliberately skips them — sidecars are
cheap to re-derive from the cached canonical + view parquets.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from eval_card_backend.canonicalise import evalcard_tags, hierarchy_dedup, hierarchy_hotfixes
from eval_card_backend.config import IGNORED_CONFIGS
from eval_card_backend.signals.reproducibility import (
    AGENTIC_REPRODUCIBILITY_FIELDS,
    BASE_REPRODUCIBILITY_FIELDS,
)
from eval_card_backend.slugs import url_encode


CONFIG_VERSION = 1
SIGNAL_VERSION = "1.0"


def write_manifest(con, out_dir: Path, snapshot_meta: dict) -> Path:
    """Tiny scalar manifest. Loaded once per process by the consumer."""
    counts = con.execute(
        """
        SELECT
            -- Root-grain model count (variants of one identity collapse
            -- to one model). `model_aggregation_key` falls back to the
            -- raw source name when canonical resolution failed, so
            -- unresolved models still count toward the headline.
            COUNT(DISTINCT model_aggregation_key)
                FILTER (WHERE model_aggregation_key IS NOT NULL)
                AS model_count,
            COUNT(DISTINCT (composite_slug, benchmark_key))
                FILTER (WHERE composite_slug IS NOT NULL
                        AND   benchmark_key  IS NOT NULL)
                AS eval_count,
            COUNT(DISTINCT (model_aggregation_key, composite_slug,
                            benchmark_key, metric_key))
                FILTER (WHERE model_aggregation_key IS NOT NULL
                        AND   composite_slug         IS NOT NULL
                        AND   benchmark_key          IS NOT NULL
                        AND   metric_key             IS NOT NULL)
                AS metric_eval_count
        FROM fact_results
        """
    ).fetchone()
    model_count, eval_count, metric_eval_count = counts
    # Read composite_count from the composites dim — that's the
    # filtered "live" set the frontend renders. Counting from
    # fact_results would double-count fully-unresolved composites
    # which never make it into the dim or hierarchy.json.
    composite_count = con.execute(
        "SELECT COUNT(*) FROM composites"
    ).fetchone()[0]

    skipped = sorted(IGNORED_CONFIGS)
    payload = {
        "generated_at":          snapshot_meta["snapshot_id"],
        "config_version":        CONFIG_VERSION,
        "skipped_configs":       skipped,
        "model_count":           int(model_count or 0),
        "eval_count":            int(eval_count or 0),
        "metric_eval_count":     int(metric_eval_count or 0),
        "composite_count":       int(composite_count or 0),
        "source_config_count":   len(snapshot_meta.get("configs") or []),
        "skipped_config_count":  len(skipped),
        # Upstream-input pins — answer "is this snapshot reading stale
        # registry/EEE data?" without a follow-up HF query. Each entry
        # is `{repo_id, sha, last_modified}`; values are None on lookup
        # failure (best-effort, never fatal — see _hf_dataset_snapshot).
        "upstream_pins":         snapshot_meta.get("upstream_pins") or {},
        # raw_verified coverage — drives the 1st/3rd-party classification
        # for llm-stats rows. Tracking coverage
        # makes it visible when upstream EEE starts populating the field
        # consistently.
        "raw_verified_coverage": _raw_verified_coverage(con),
        "summary_artifacts": {
            "corpus_aggregates": "headline.json",
            "eval_hierarchy":    "hierarchy.json",
            "comparison_index":  "comparison-index.json",
            "benchmark_index":   "benchmark_index.json",
        },
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _raw_verified_coverage(con) -> dict:
    """Tally `metric_config.additional_details.raw_verified` presence
    across the corpus, broken out by whether the row's source is the
    llm-stats aggregator (where the field actually drives the
    first/third-party label per the reference's party_label rule).

    Returns a dict the manifest emits verbatim:
      {
        total_rows: int,
        llm_stats: {total: int, raw_verified_true: int,
                    raw_verified_false: int, raw_verified_null: int},
        non_llm_stats: {total: int, raw_verified_true: int,
                        raw_verified_false: int, raw_verified_null: int},
      }

    `raw_verified` is `additional_details.raw_verified` — a string
    'true'/'false' in EEE today, sometimes missing. Read defensively;
    treat anything not in {'true', 'false'} as missing.
    """
    row = con.execute(
        """
        WITH rv AS (
            SELECT
                fr.source_config = 'llm-stats' AS is_llm_stats,
                -- metric_additional_details is a JSON-encoded VARCHAR;
                -- json_extract_string returns NULL when the key is
                -- absent or the value isn't a string. Lowercase to
                -- normalise 'True'/'TRUE'/'true' variants.
                LOWER(COALESCE(
                    json_extract_string(fr.metric_additional_details, '$.raw_verified'),
                    ''
                )) AS rv
            FROM fact_results fr
        )
        SELECT
            COUNT(*)                                                AS total,
            COUNT(*) FILTER (WHERE is_llm_stats)                     AS ls_total,
            COUNT(*) FILTER (WHERE is_llm_stats AND rv = 'true')     AS ls_true,
            COUNT(*) FILTER (WHERE is_llm_stats AND rv = 'false')    AS ls_false,
            COUNT(*) FILTER (WHERE is_llm_stats
                             AND rv NOT IN ('true', 'false'))        AS ls_null,
            COUNT(*) FILTER (WHERE NOT is_llm_stats)                 AS nls_total,
            COUNT(*) FILTER (WHERE NOT is_llm_stats AND rv = 'true') AS nls_true,
            COUNT(*) FILTER (WHERE NOT is_llm_stats AND rv = 'false') AS nls_false,
            COUNT(*) FILTER (WHERE NOT is_llm_stats
                             AND rv NOT IN ('true', 'false'))        AS nls_null
        FROM rv
        """
    ).fetchone()
    return {
        "total_rows":    int(row[0] or 0),
        "llm_stats": {
            "total":              int(row[1] or 0),
            "raw_verified_true":  int(row[2] or 0),
            "raw_verified_false": int(row[3] or 0),
            "raw_verified_null":  int(row[4] or 0),
        },
        "non_llm_stats": {
            "total":              int(row[5] or 0),
            "raw_verified_true":  int(row[6] or 0),
            "raw_verified_false": int(row[7] or 0),
            "raw_verified_null":  int(row[8] or 0),
        },
    }


# ---------------------------------------------------------------------------
# headline.json
# ---------------------------------------------------------------------------


def _tag_filter_clause(tag: str | None, alias: str = "erv") -> str:
    """Return a WHERE fragment filtering to rows whose derived_tags
    JSON array contains `tag` (or empty when None)."""
    if tag is None:
        return ""
    safe = tag.replace("'", "''")
    return (
        f"AND list_contains("
        f"from_json({alias}.derived_tags, '[\"VARCHAR\"]'), '{safe}')"
    )


def _reproducibility_block(con, tag: str | None) -> dict:
    """Triple-level rollups for reproducibility.

    Per-field missingness scans `fact_results` for every fact row's
    `repro_missing_fields[]`, then BOOL_OR's per-triple — a triple is
    flagged as missing field f if *any* of its rows had f in the gap.
    Base fields are denominated against all triples; agentic fields
    against agentic triples only (a triple is agentic if any row in it
    is_agentic).
    """
    if tag is None:
        tag_join = ""
        tag_where = ""
    else:
        tag_join = (
            "JOIN eval_results_view erv "
            "  ON erv.model_key    = fr.model_aggregation_key "
            " AND erv.benchmark_id = fr.benchmark_key "
            " AND erv.metric_id    = fr.metric_key"
        )
        tag_where = _tag_filter_clause(tag, alias="erv")

    field_flags_sql_parts = []
    for f in BASE_REPRODUCIBILITY_FIELDS + AGENTIC_REPRODUCIBILITY_FIELDS:
        field_flags_sql_parts.append(
            f"BOOL_OR(array_contains(fr.repro_missing_fields, '{f}')) "
            f"AS missing_{f}"
        )
    field_flags_sql = ",\n            ".join(field_flags_sql_parts)

    triple_rollup_sql = f"""
        WITH triple_rollups AS (
            SELECT
                fr.model_aggregation_key, fr.benchmark_key, fr.metric_key,
                BOOL_OR(fr.has_reproducibility_gap) AS triple_has_gap,
                BOOL_OR(fr.is_agentic)              AS triple_agentic,
                {field_flags_sql}
            FROM fact_results fr
            {tag_join}
            WHERE fr.model_aggregation_key IS NOT NULL
              AND fr.benchmark_key         IS NOT NULL
              AND fr.metric_key            IS NOT NULL
              {tag_where}
            GROUP BY 1, 2, 3
        )
        SELECT
            COUNT(*)                           AS total_triples,
            SUM(CASE WHEN triple_has_gap THEN 1 ELSE 0 END) AS triples_with_gap,
            AVG(CASE WHEN triple_has_gap THEN 1.0 ELSE 0.0 END)
                                                AS gap_rate,
            SUM(CASE WHEN triple_agentic THEN 1 ELSE 0 END) AS agentic_triples,
            {",".join(
                f"SUM(CASE WHEN missing_{f} THEN 1 ELSE 0 END) AS n_missing_{f}"
                for f in BASE_REPRODUCIBILITY_FIELDS + AGENTIC_REPRODUCIBILITY_FIELDS
            )}
        FROM triple_rollups
        """
    row = con.execute(triple_rollup_sql).fetchone()
    columns = [d[0] for d in con.description]
    rec = dict(zip(columns, row))

    total = int(rec["total_triples"] or 0)
    triples_with_gap = int(rec["triples_with_gap"] or 0)
    gap_rate = rec["gap_rate"]
    agentic = int(rec["agentic_triples"] or 0)

    per_field: dict[str, dict] = {}
    for f in BASE_REPRODUCIBILITY_FIELDS:
        n = int(rec[f"n_missing_{f}"] or 0)
        per_field[f] = {
            "missing_count":     n,
            "missing_rate":      (n / total) if total else None,
            "denominator":       "all_triples",
            "denominator_count": total,
        }
    for f in AGENTIC_REPRODUCIBILITY_FIELDS:
        n = int(rec[f"n_missing_{f}"] or 0)
        per_field[f] = {
            "missing_count":     n,
            "missing_rate":      (n / agentic) if agentic else None,
            "denominator":       "agentic_only",
            "denominator_count": agentic,
        }

    return {
        "total_triples":                       total,
        "triples_with_reproducibility_gap":    triples_with_gap,
        "reproducibility_gap_rate":            gap_rate,
        "agentic_triples":                     agentic,
        "per_field_missingness":               per_field,
    }


def _completeness_block(con, tag: str | None) -> dict:
    """Triple-level completeness rollup.

    A triple's completeness is the AVG of its fact rows' completeness
    scores; the corpus-level rate is AVG over triples (not over rows).
    """
    tag_clause = _tag_filter_clause(tag)
    row = con.execute(
        f"""
        WITH triple_rollups AS (
            SELECT
                erv.model_key, erv.benchmark_id, erv.metric_id,
                erv.completeness_score AS triple_avg_completeness
            FROM eval_results_view erv
            WHERE 1 = 1 {tag_clause}
        )
        SELECT
            COUNT(*)                                AS total_triples,
            AVG(triple_avg_completeness)            AS completeness_avg,
            MIN(triple_avg_completeness)            AS completeness_min,
            MAX(triple_avg_completeness)            AS completeness_max
        FROM triple_rollups
        """
    ).fetchone()
    total, avg, mn, mx = row
    return {
        "total_triples":   int(total or 0),
        "completeness_avg": avg,
        "completeness_min": mn,
        "completeness_max": mx,
    }


def _provenance_block(con, tag: str | None) -> dict:
    """Per-triple provenance rollup. coverage_cell + has_third_party drive
    the 4-way source-type distribution."""
    tag_clause = _tag_filter_clause(tag)
    row = con.execute(
        f"""
        SELECT
            COUNT(*)                                            AS total_triples,
            SUM(CASE WHEN erv.is_multi_source THEN 1 ELSE 0 END)     AS multi_source_triples,
            SUM(CASE WHEN erv.first_party_only THEN 1 ELSE 0 END)    AS first_party_only_triples,
            SUM(CASE WHEN erv.coverage_cell = 'self'                          THEN 1 ELSE 0 END) AS pst_first_party,
            SUM(CASE WHEN erv.coverage_cell = 'third' AND erv.has_third_party THEN 1 ELSE 0 END) AS pst_third_party,
            SUM(CASE WHEN erv.coverage_cell = 'both'                          THEN 1 ELSE 0 END) AS pst_collaborative,
            SUM(CASE WHEN erv.coverage_cell = 'third' AND NOT erv.has_third_party THEN 1 ELSE 0 END) AS pst_unspecified
        FROM eval_results_view erv
        WHERE 1 = 1 {tag_clause}
        """
    ).fetchone()
    (total, multi, first_only, pst_fp, pst_tp, pst_co, pst_un) = row
    return {
        "total_triples":            int(total or 0),
        "multi_source_triples":     int(multi or 0),
        "first_party_only_triples": int(first_only or 0),
        "source_type_distribution": {
            "first_party":   int(pst_fp or 0),
            "third_party":   int(pst_tp or 0),
            "collaborative": int(pst_co or 0),
            "unspecified":   int(pst_un or 0),
        },
    }


def _comparability_block(con, tag: str | None) -> dict:
    tag_clause = _tag_filter_clause(tag)
    row = con.execute(
        f"""
        -- Divergence is a comparability-GROUP signal. The comparability group
        -- grain is slice-aware (comparability_group_id includes slice_key), so
        -- it cannot be counted off eval_results_view, which collapses slices to
        -- one triple row. We source the counts from fact_results, counting each
        -- comparability_group_id once (the flag is constant within a group).
        -- The by_tag filter still works via a deduped derived_tags lookup joined
        -- on the (model, benchmark, metric) triple (tags are benchmark-level,
        -- constant across a triple's slices). total_triples is now the count of
        -- distinct comparability groups. See sensitivity/docs/divergence-count-grain.md.
        SELECT
            COUNT(DISTINCT fr.comparability_group_id)                 AS total_triples,
            COUNT(DISTINCT fr.comparability_group_id)
                FILTER (WHERE fr.has_variant_divergence)              AS variant_divergent,
            COUNT(DISTINCT fr.comparability_group_id)
                FILTER (WHERE fr.has_cross_party_divergence)          AS cross_party_divergent,
            COUNT(DISTINCT fr.comparability_group_id)
                FILTER (WHERE fr.has_variant_divergence IS NOT NULL)  AS variant_eligible,
            COUNT(DISTINCT fr.comparability_group_id)
                FILTER (WHERE fr.has_cross_party_divergence IS NOT NULL) AS cross_party_eligible
        FROM fact_results fr
        LEFT JOIN (
            SELECT DISTINCT model_key, benchmark_id, metric_id, derived_tags
            FROM eval_results_view
        ) erv
          ON erv.model_key    = fr.model_aggregation_key
         AND erv.benchmark_id = fr.benchmark_key
         AND erv.metric_id    = fr.metric_key
        WHERE fr.comparability_group_id IS NOT NULL {tag_clause}
        """
    ).fetchone()
    (total, var_div, cross_div, var_elig, cross_elig) = row
    return {
        "total_triples":               int(total or 0),
        "variant_divergent_count":     int(var_div or 0),
        "cross_party_divergent_count": int(cross_div or 0),
        "groups_with_variant_check":     int(var_elig or 0),
        "groups_with_cross_party_check": int(cross_elig or 0),
    }


def _developers_list(con) -> list[dict]:
    rows = con.execute(
        """
        SELECT
            developer,
            COUNT(DISTINCT model_key)                                    AS model_count,
            COUNT(DISTINCT benchmark_id) FILTER (WHERE benchmark_id IS NOT NULL) AS benchmark_count,
            COUNT(*)                                                     AS evaluation_count
        FROM models_view
        LEFT JOIN UNNEST(benchmark_names) b(benchmark_id) ON TRUE
        WHERE developer IS NOT NULL
        GROUP BY developer
        ORDER BY evaluation_count DESC, developer ASC
        """
    ).fetchall()
    cols = [d[0] for d in con.description]
    out: list[dict] = []
    for r in rows:
        rec = dict(zip(cols, r))
        out.append({
            "developer":        rec["developer"],
            "route_id":         url_encode(rec["developer"]),
            "model_count":      int(rec["model_count"] or 0),
            "benchmark_count":  int(rec["benchmark_count"] or 0),
            "evaluation_count": int(rec["evaluation_count"] or 0),
            "popular_evals":    [],   # placeholder — frontend tolerates empty
        })
    return out


def _model_families_list(con) -> list[dict]:
    """Per-model-group rollup (developer + variant lineage groupings
    derived from `models_view.model_group_id`, the always-present GROUP
    key). Distinct from the benchmark-family hierarchy under `families`
    in hierarchy.json — name kept distinct to avoid the historical
    conflation. The emitted `family_key` JSON field is left unchanged
    (frontend contract); it carries the group key, as it always has.
    """
    rows = con.execute(
        """
        SELECT
            model_group_id   AS family_key,
            ANY_VALUE(model_family_name) AS display_name,
            COUNT(DISTINCT model_key)    AS model_count,
            SUM(evaluations_count)       AS eval_count
        FROM models_view
        WHERE model_group_id IS NOT NULL
        GROUP BY model_group_id
        ORDER BY eval_count DESC NULLS LAST, family_key ASC
        """
    ).fetchall()
    return [
        {
            "family_key":   r[0],
            "display_name": r[1] or r[0],
            "model_count":  int(r[2] or 0),
            "eval_count":   int(r[3] or 0),
        }
        for r in rows
    ]


def _composites_list(con) -> list[dict]:
    """Per-composite rollup. Counts of distinct benchmarks + models +
    triples per composite_slug. Drives the homepage composite strip.
    """
    rows = con.execute(
        """
        SELECT
            composite_slug,
            ANY_VALUE(composite_display_name)            AS display_name,
            COUNT(DISTINCT benchmark_id)                 AS benchmark_count,
            COUNT(DISTINCT model_key)                    AS model_count,
            COUNT(*)                                     AS evaluation_count
        FROM eval_results_view
        WHERE composite_slug IS NOT NULL
        GROUP BY composite_slug
        ORDER BY evaluation_count DESC, composite_slug ASC
        """
    ).fetchall()
    return [
        {
            "composite_slug":   r[0],
            "display_name":     r[1] or r[0],
            "benchmark_count":  int(r[2] or 0),
            "model_count":      int(r[3] or 0),
            "evaluation_count": int(r[4] or 0),
        }
        for r in rows
    ]


def _tags_list(con) -> list[dict]:
    rows = con.execute(
        """
        SELECT
            tag.t                                       AS tag,
            COUNT(DISTINCT erv.model_key)               AS model_count,
            COUNT(DISTINCT (erv.benchmark_id, erv.metric_id)) AS eval_count
        FROM eval_results_view erv,
             UNNEST(from_json(erv.derived_tags, '["VARCHAR"]')) AS tag(t)
        GROUP BY 1
        ORDER BY eval_count DESC, tag ASC
        """
    ).fetchall()
    return [
        {"tag": r[0], "model_count": int(r[1] or 0), "eval_count": int(r[2] or 0)}
        for r in rows
    ]


def _reporting_org_count(con) -> int:
    """Distinct eval-provider identities in the corpus.

    Reads the de-aliased `reporting_orgs` arrays from `eval_results_view`
    (canonical_orgs.display_name when registered, raw upstream string
    otherwise — see the `tris` CTE in stage_j_eval_results_view), unnests,
    and counts distinct. Precomputed here so the frontend doesn't have to
    scan parquet at request time for the home-page headline tile.
    """
    row = con.execute(
        """
        SELECT COUNT(DISTINCT u)
        FROM eval_results_view erv,
             UNNEST(COALESCE(erv.reporting_orgs, [])) AS u_t(u)
        WHERE u IS NOT NULL
        """
    ).fetchone()
    return int(row[0] or 0)


def _total_benchmarks(con) -> int:
    # Denominator for per-model "Benchmarks" / "Coverage" on the listing.
    # Reads `benchmarks` (the canonical dim) so this matches the home-page
    # `hierarchy.stats.benchmark_count`. Excludes slice rows (e.g.
    # gaia-level-1) so the user-facing count reflects distinct evaluation
    # frameworks, not framework × subset cells. Numerator
    # (models_view.benchmarks_count = COUNT(DISTINCT benchmark_id) over
    # eval_results_view) is still a subset since any benchmark with model
    # rows in the view also exists in the dim.
    row = con.execute(
        """
        SELECT COUNT(DISTINCT benchmark_id)
        FROM benchmarks
        WHERE benchmark_id IS NOT NULL
          AND NOT is_slice
        """
    ).fetchone()
    return int(row[0] or 0)


def write_headline(con, out_dir: Path, snapshot_meta: dict) -> Path:
    from eval_card_backend.canonicalise.evalcard_tags import VALID_TAGS

    def _stratified(builder):
        return {
            "overall":  builder(con, None),
            "by_tag":   {t: builder(con, t) for t in sorted(VALID_TAGS)},
        }

    payload = {
        "generated_at":              snapshot_meta["snapshot_id"],
        "signal_version":            SIGNAL_VERSION,
        "stratification_dimensions": ["tag"],
        "reproducibility": _stratified(_reproducibility_block),
        "completeness":    _stratified(_completeness_block),
        "provenance":      _stratified(_provenance_block),
        "comparability":   _stratified(_comparability_block),
        "reporting_org_count": _reporting_org_count(con),
        "total_benchmarks":   _total_benchmarks(con),
        "developers":      _developers_list(con),
        "model_families":  _model_families_list(con),
        "composites":      _composites_list(con),
        "tags":            _tags_list(con),
    }
    path = out_dir / "headline.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


# ---------------------------------------------------------------------------
# hierarchy.json
# ---------------------------------------------------------------------------

_FORCE_LAYOUT: dict[str, str] = {
    "hf-open-llm-v2": "composites",
}


def write_hierarchy(con, out_dir: Path, snapshot_meta: dict) -> Path:
    """Family-rooted hierarchy tree.

    The top-level shape is:

      {
        schema_version: "v3.hierarchy.1",
        generated_at, stats,
        families: [{key, display_name, derivedTags, tags, evals_count,
                    constituent_evaluation_ids, provenance_summary,
                    standalone_benchmarks | benchmarks | composites}],
      }

    Each family chooses ONE of three layouts based on its content:
      - `standalone_benchmarks[]`: single-benchmark family.
      - `benchmarks[]` (flat): multiple benchmarks, no composite layer.
      - `composites[].benchmarks[]`: multiple distinct named groupings
        within the family (e.g. HELM's multiple composites; MMLU-Pro's one).

    Bucketing rules:
      - Composite with `family_id` set in canonical_composites lands
        under that family.
      - Composite without `family_id` becomes its own singleton family
        (family.id == composite.id).
    """
    composites = _hierarchy_composites(con)  # rich per-composite records
    family_records = _hierarchy_families(con, composites)

    # Registry-curated families are exempt from dedup/hotfix removal
    curated_ids: frozenset[str] = frozenset()
    if _table_exists(con, "canonical_families"):
        curated_ids = frozenset(
            r[0] for r in con.execute("SELECT id FROM canonical_families").fetchall()
            if r[0]
        )
        hierarchy_dedup.set_curated_families(curated_ids)

    # Hot fixes (temporary, clearly separated)
    hierarchy_hotfixes.consolidate_air_bench(family_records)
    hierarchy_hotfixes.dedup_vals_ai_aliases(family_records)
    hierarchy_hotfixes.group_same_bench_across_sources(family_records, curated_ids)

    # Systematic dedup (permanent pipeline logic). Both passes are
    # score-gated per data-model-invariants §4 — only true echoes
    # (matching scores) are collapsed; independent cross-source
    # evaluations of the same benchmark are preserved.
    hierarchy_dedup.consolidate_dedicated_home_benchmarks(family_records, con)
    hierarchy_dedup.dedup_aggregator_benchmarks(family_records, con)

    # Tag decoration runs after structural changes, before stats
    evalcard_tags.decorate_hierarchy_tags(family_records)

    # Stats and index computed after dedup so they reflect the cleaned tree
    benchmark_index = _hierarchy_benchmark_index(con, family_records)
    stats = _hierarchy_tree_stats(con, family_records)

    payload = {
        "schema_version":  "v3.hierarchy.1",
        "generated_at":    snapshot_meta["snapshot_id"],
        "stats":           stats,
        "families":        family_records,
        "benchmark_index": benchmark_index,
    }
    path = out_dir / "hierarchy.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _hierarchy_benchmark_index(con, families: list[dict]) -> list[dict]:
    """Cross-suite lookup. One entry per canonical benchmark that
    surfaces under 2+ distinct families (e.g. AIME under llm-stats AND
    artificial-analysis).

    Each entry:
      {
        key:          canonical benchmark id,
        display_name: benchmark display,
        appearances: [{family_key, benchmark_key, constituent_evaluation_ids,
                       models_count, is_canonical_home}],
      }

    `is_canonical_home` flags the appearance whose family is the
    benchmark's "natural" home (family_key == benchmark_key, i.e. the
    benchmark IS the family root somewhere). Useful for the frontend
    to render the headline link.

    Per-(model, metric) cross-suite aggregates are deferred — that's
    data analysis, not structure.
    """
    from collections import defaultdict

    # Walk every benchmark across every family layout. Each appearance
    # is one (family, benchmark) pair carrying its constituent_evaluation_ids.
    # We also walk slices: a within-benchmark slice whose `key` is itself
    # a canonical benchmark (e.g. `aime-2024` as a slice of `aime`)
    # surfaces as a cross-suite hit when it appears under multiple
    # families' parent benchmarks.
    appearances_by_bench: dict[str, list[dict]] = defaultdict(list)
    for fam in families:
        bench_streams = [
            *(fam.get("standalone_benchmarks") or []),
            *(fam.get("benchmarks") or []),
            *(b for c in (fam.get("composites") or [])
                for b in (c.get("benchmarks") or [])),
        ]
        for b in bench_streams:
            bench_key = b["key"]
            appearances_by_bench[bench_key].append({
                "family_key":        fam["key"],
                "benchmark_key":     bench_key,
                "constituent_evaluation_ids": list(b.get("constituent_evaluation_ids") or []),
                "is_canonical_home": (fam["key"] == bench_key),
            })
            for s in b.get("slices") or []:
                slice_key = s.get("key")
                if not slice_key or slice_key == bench_key:
                    continue
                appearances_by_bench[slice_key].append({
                    "family_key":        fam["key"],
                    "benchmark_key":     bench_key,
                    "constituent_evaluation_ids": list(b.get("constituent_evaluation_ids") or []),
                    "is_canonical_home": False,
                })

    # Look up display names from the benchmarks dim — one canonical id
    # may have inconsistent display_name across composites; pick the
    # one tied to the canonical_home appearance when possible.
    display_lookup: dict[str, str] = {}
    for fam in families:
        for stream_name in ("standalone_benchmarks", "benchmarks"):
            for b in fam.get(stream_name) or []:
                display_lookup.setdefault(b["key"], b.get("display_name") or b["key"])
        for c in fam.get("composites") or []:
            for b in c.get("benchmarks") or []:
                display_lookup.setdefault(b["key"], b.get("display_name") or b["key"])

    out: list[dict] = []
    for bench_key in sorted(appearances_by_bench):
        appearances = appearances_by_bench[bench_key]
        distinct_families = {a["family_key"] for a in appearances}
        if len(distinct_families) < 2:
            continue
        # Filter family-rollup artifacts: a slice appearing under many
        # different parent benchmarks within one family creates an entry
        # with many distinct benchmark_keys. Real cross-family entries
        # have few (a benchmark in 2-3 families shares <= 3 parent keys).
        distinct_bench_keys = {a["benchmark_key"] for a in appearances}
        if len(distinct_bench_keys) > 3:
            continue
        out.append({
            "key":          bench_key,
            "display_name": display_lookup.get(bench_key, bench_key),
            "appearances":  appearances,
        })
    return out


def _hierarchy_families(con, composites: list[dict]) -> list[dict]:
    """Bucket composites by family_id (from canonical_composites + a
    dim-level join), then for each family choose a layout.

    Reads `canonical_families` and `canonical_composites` from the
    DuckDB connection (Stage A loaded them via taxonomy.py). Composites
    without a curated family_id become singleton families.
    """
    import json as _json
    from collections import defaultdict

    # --- Pull family / composite curation from registry tables ---
    fam_rows = con.execute(
        "SELECT id, display_name, category, tags, "
        "       benchmark_ids, composite_keys, folder_aliases "
        "  FROM canonical_families"
    ).fetchall() if _table_exists(con, "canonical_families") else []
    families_curated: dict[str, dict] = {}
    for r in fam_rows:
        fid, display, cat, tags, bench_ids, comp_keys, folder_aliases = r
        families_curated[fid] = {
            "display_name":    display or fid,
            "category":        (cat or "other"),
            "tags":            _decode_json_list(tags),
            "benchmark_ids":   _decode_json_list(bench_ids),
            "composite_keys":  _decode_json_list(comp_keys),
            "folder_aliases":  _decode_json_list(folder_aliases),
        }

    comp_rows = con.execute(
        "SELECT id, family_id FROM canonical_composites"
    ).fetchall() if _table_exists(con, "canonical_composites") else []
    composite_to_family: dict[str, str] = {}
    for cid, fid in comp_rows:
        if fid:
            composite_to_family[cid] = fid

    # When `canonical_families.folder_aliases` is populated, map each
    # folder alias to a composite-slug equivalent (`cyse2_interpreter_abuse`
    # → `cyse2-interpreter-abuse`) and treat that composite as belonging
    # to the curated family. Lets families that only specify
    # `folder_aliases` (no `composite_keys` or `benchmark_ids`) still
    # bucket their composites correctly.
    import re as _re
    def _folder_to_composite_slug(folder: str) -> str:
        s = folder.lower()
        s = _re.sub(r"[_\s]+", "-", s)
        s = _re.sub(r"[^a-z0-9-/.]+", "-", s)
        s = _re.sub(r"-+", "-", s).strip("-")
        return s
    for fid, curated in families_curated.items():
        for folder in curated.get("folder_aliases") or []:
            comp_slug = _folder_to_composite_slug(folder)
            composite_to_family.setdefault(comp_slug, fid)

    # --- Bucket composites by family ---
    # Family-slug aliases map a composite's own slug (what we'd emit when no
    # canonical_composites.family_id is curated) onto the canonical family slug.
    # Set in one place so the rename also applies to per-benchmark family_id below.
    family_slug_aliases = {
        "artificial-analysis-llms": "artificial-analysis",
        "caparena-auto":            "caparena",
        "hfopenllm-v2":             "hf-open-llm-v2",
    }

    by_family: dict[str, list[dict]] = defaultdict(list)
    for comp in composites:
        family_id = composite_to_family.get(comp["key"], comp["key"])
        family_id = family_slug_aliases.get(family_id, family_id)
        by_family[family_id].append(comp)

    # Add curated families that have no composites (rare — covers the
    # case of a families.yaml entry whose member benchmarks landed
    # entirely inside composites that *do* have family_id but the
    # curated entry is the parent grouping). Empty buckets are skipped
    # below.
    for fid in families_curated:
        by_family.setdefault(fid, [])

    # --- Build per-family records ---
    out: list[dict] = []
    composite_driven_keys: set[str] = set()  # tracks families built from composites
    for fid in sorted(by_family):
        family_composites = by_family[fid]
        if not family_composites:
            continue
        composite_driven_keys.add(fid)

        curated = families_curated.get(fid)
        # Display name preference: curated > composite display when
        # singleton > family id. Curated wins so HELM family says
        # "HELM" not "HELM Classic" (the first composite alphabetically).
        if curated:
            display_name = curated["display_name"]
            family_tags = curated["tags"]
        else:
            display_name = family_composites[0]["display_name"]
            family_tags = []

        # Roll up benchmarks across all of this family's composites.
        # Each composite record has benchmarks[] already (rich, with
        # slices/metrics/etc).
        all_benchmarks: list[dict] = []
        all_constituent_evaluation_ids: set[str] = set()
        for comp in family_composites:
            for bench in comp.get("benchmarks", []):
                all_benchmarks.append(bench)
                for eid in bench.get("constituent_evaluation_ids", []) or []:
                    all_constituent_evaluation_ids.add(eid)

        # --- Layout selection ---
        # 1. Multi-composite family → composites[] layout (HELM).
        # 2. Single composite + single benchmark → standalone_benchmarks.
        # 3. Single composite, multiple benchmarks → flat benchmarks[].
        # 4. No benchmarks anywhere (curated family with empty composites)
        #    → skip (continue above already filtered empty buckets).
        family_record: dict = {
            "key":              fid,
            "display_name":     display_name,
            "tags":             _merge_family_tags(family_tags, all_benchmarks),
            "evals_count":      sum(int(c.get("evals_count") or 0)
                                    for c in family_composites),
            "constituent_evaluation_ids": sorted(all_constituent_evaluation_ids),
        }

        # Aggregate signal summaries across the family's benchmarks.
        # Reuses the per-composite aggregators which take a list of
        # benchmark records.
        family_record["reproducibility_summary"] = _aggregate_reproducibility(all_benchmarks)
        family_record["provenance_summary"]      = _aggregate_provenance(all_benchmarks)
        family_record["comparability_summary"]   = _aggregate_comparability(all_benchmarks)

        # Mark the headline benchmark within this family. Sets
        # is_primary on each benchmark row across whatever layout the
        # family ends up using.
        _mark_family_primary_benchmark(fid, all_benchmarks)

        force = _FORCE_LAYOUT.get(fid)
        if force == "composites" and family_composites:
            primary_comp = sorted(family_composites, key=lambda c: c["key"])[0]
            for comp in family_composites:
                comp["is_primary"] = (comp["key"] == primary_comp["key"])
            family_record["composites"] = family_composites
        elif len(family_composites) >= 2:
            primary_comp = sorted(family_composites, key=lambda c: c["key"])[0]
            for comp in family_composites:
                comp["is_primary"] = (comp["key"] == primary_comp["key"])
            family_record["composites"] = family_composites
        elif len(all_benchmarks) == 1:
            family_record["standalone_benchmarks"] = all_benchmarks
        elif len(all_benchmarks) >= 2:
            family_record["benchmarks"] = all_benchmarks
        else:
            # Curated family with no benchmarks materialised — skip.
            # (Defensive; should be filtered by the empty-bucket check.)
            continue

        out.append(family_record)

    # --- Benchmark-driven families ---------------------------------------
    # Some benchmarks claim a `family_id` that no composite is curated to
    # (e.g. judgebench-coding lives under the `wasp` composite but its
    # canonical_benchmark.family_id="judgebench"). The composite-driven
    # bucketing above places such benchmarks under the wrong family and
    # drops the natural one. Surface them here as additional top-level
    # families. Each benchmark appears in BOTH its composite-driven family
    # and its benchmark-driven family — benchmark_index naturally cross-
    # links the two.
    #
    # We only surface a benchmark-driven family when:
    #   (a) the family_id matches a curated `canonical_families` entry, OR
    #   (b) ≥2 distinct benchmarks across composites share the same
    #       family_id and that family_id isn't already a top-level family.
    # Singleton family_ids that just happen to match a benchmark's own
    # canonical_id (the auto-fallback for unset family_ids) don't qualify.
    benchmark_family_groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for fam_record in list(out):
        for bench in _walk_family_benchmarks(fam_record):
            bfid = bench.get("family_id")
            if not bfid or bfid == fam_record["key"]:
                continue
            if bfid in composite_driven_keys:
                continue
            # Skip the auto-fallback case where family_id was defaulted
            # to the benchmark's own canonical_id by
            # `_hierarchy_composite_benchmark`. Those aren't curated
            # family edges; surfacing them here creates singleton
            # noise-families.
            if bfid == bench.get("key"):
                continue
            # Dedupe by benchmark_id: the same canonical can live
            # under multiple composites (cross-suite benchmarks like
            # math-500, mmlu, gpqa). We only want one entry per
            # canonical in the benchmark-driven family.
            benchmark_family_groups[bfid].setdefault(bench["key"], bench)

    # Registry-curation-driven families: a `canonical_families` entry can
    # list `benchmark_ids` directly (e.g. cyse2 → [cyse2_interpreter_abuse,
    # cyse2_prompt_injection, cyse2_vulnerability_exploit]). When the
    # member benchmarks don't carry an explicit `family_id` (so the
    # walk above missed them), pull them in by ID match against
    # everything we've already rendered.
    for fid, curated in families_curated.items():
        if fid in composite_driven_keys or fid in benchmark_family_groups:
            continue
        member_ids = set(curated.get("benchmark_ids") or [])
        if not member_ids:
            continue
        for fam_record in out:
            for bench in _walk_family_benchmarks(fam_record):
                if bench.get("key") in member_ids:
                    benchmark_family_groups[fid].setdefault(bench["key"], bench)

    # The inner filter (`bfid != bench.get("key")`) already removed the
    # auto-fallback case, so any remaining group is curated. No further
    # gating required.

    import copy as _copy
    for bfid in sorted(benchmark_family_groups):
        # Clone the benchmark dicts so that `_mark_family_primary_benchmark`
        # below — which mutates `is_primary` — doesn't leak the
        # benchmark-driven family's primary choice back into the
        # composite-driven family that originally owns these benchmarks.
        # Without the clone, a benchmark surfaced under both families
        # ends up with `is_primary=True` set from whichever pass ran
        # last, violating the per-family "at most one primary" invariant.
        benches = [_copy.copy(b) for b in benchmark_family_groups[bfid].values()]
        if not benches:
            continue
        # Reset is_primary on the cloned copies — the benchmark-driven
        # family decides its own primary independently of the
        # composite-driven family's choice.
        for b in benches:
            b["is_primary"] = False
        curated = families_curated.get(bfid)
        display = curated["display_name"] if curated else _prettify_family_slug(bfid)
        family_tags = curated["tags"] if curated else []
        all_constituent_evaluation_ids: set[str] = set()
        for b in benches:
            for eid in b.get("constituent_evaluation_ids", []) or []:
                all_constituent_evaluation_ids.add(eid)
        family_record = {
            "key":              bfid,
            "display_name":     display,
            "tags":             _merge_family_tags(family_tags, benches),
            "evals_count":      len(all_constituent_evaluation_ids),
            "constituent_evaluation_ids": sorted(all_constituent_evaluation_ids),
        }
        family_record["reproducibility_summary"] = _aggregate_reproducibility(benches)
        family_record["provenance_summary"]      = _aggregate_provenance(benches)
        family_record["comparability_summary"]   = _aggregate_comparability(benches)
        _mark_family_primary_benchmark(bfid, benches)
        if len(benches) == 1:
            family_record["standalone_benchmarks"] = benches
        else:
            family_record["benchmarks"] = benches
        out.append(family_record)

    return out


def _walk_family_benchmarks(family: dict):
    """Yield every benchmark dict in a family record across all three
    layouts (standalone_benchmarks / benchmarks / composites[].benchmarks).
    """
    for layout in ("standalone_benchmarks", "benchmarks"):
        for b in family.get(layout) or []:
            yield b
    for c in family.get("composites") or []:
        for b in c.get("benchmarks") or []:
            yield b


def _prettify_family_slug(slug: str) -> str:
    """Lightweight display fallback when no curated display_name exists."""
    return slug.replace("-", " ").replace("_", " ").title()


def _hierarchy_tree_stats(con, families: list[dict]) -> dict:
    """Recompute stats from the cleaned hierarchy tree.

    `benchmark_count` and `slice_count` are derived from the post-dedup
    tree so they reflect what the consumer actually sees.
    `metric_count` and `metric_rows_scanned` stay DB-sourced — they
    represent the raw data corpus, not the presentation hierarchy.
    """
    benchmark_count = 0
    slice_count = 0
    for fam in families:
        for b in _walk_family_benchmarks(fam):
            benchmark_count += 1
            slice_count += len(b.get("slices") or [])

    row = con.execute(
        """
        SELECT
            (SELECT COUNT(DISTINCT (composite_slug, benchmark_key, metric_key))
                 FROM fact_results
                 WHERE composite_slug IS NOT NULL
                   AND benchmark_key  IS NOT NULL
                   AND metric_key     IS NOT NULL)                    AS metric_count,
            (SELECT COUNT(*) FROM fact_results)                       AS metric_rows_scanned
        """
    ).fetchone()
    # A composite is a named sub-surface grouping inside a multi-composite
    # family (HELM's 7 surfaces, MMMU's 2 modes, hf-open-llm-v2). Only
    # families using the composites[] layout have them; flat/standalone
    # families have none. Count the actual composite entries — NOT a
    # hybrid that also added 1 per flat family (that inflated the stat).
    composite_count = sum(len(f.get("composites") or []) for f in families)
    return {
        "family_count":         len(families),
        "composite_count":      composite_count,
        "benchmark_count":      benchmark_count,
        "slice_count":          slice_count,
        "metric_count":         int(row[0] or 0),
        "metric_rows_scanned":  int(row[1] or 0),
    }


def _table_exists(con, name: str) -> bool:
    """True when a table or view exists on the connection. The
    canonical_families / canonical_composites tables are loaded from
    the registry parquets via taxonomy.py; if a deployment is on an
    old registry snapshot without them, write_hierarchy degrades to
    a no-curation flat output."""
    rows = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
        [name],
    ).fetchall()
    return bool(rows)


def _decode_json_list(value) -> list:
    import json as _json
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ("[]", "null"):
            return []
        try:
            d = _json.loads(s)
            return list(d) if isinstance(d, list) else []
        except (ValueError, TypeError):
            return []
    return []


def _merge_family_tags(family_tags: list, benchmarks: list[dict]) -> dict:
    """Combine curated family tags (a flat string list) with per-
    benchmark tag dicts ({domains, languages, tasks}). Family-level
    tags get folded into `domains` since the curated family.tags is
    closest to that semantically."""
    domains = set(d for d in family_tags if isinstance(d, str))
    languages: set[str] = set()
    tasks: set[str] = set()
    for b in benchmarks:
        bench_tags = b.get("tags") or {}
        for d in bench_tags.get("domains") or []:
            domains.add(d)
        for l in bench_tags.get("languages") or []:
            languages.add(l)
        for t in bench_tags.get("tasks") or []:
            tasks.add(t)
    return {
        "domains":   sorted(domains),
        "languages": sorted(languages),
        "tasks":     sorted(tasks),
    }


def _hierarchy_composites(con) -> list[dict]:
    """Build the composites[] tree from the benchmarks dim + per-
    benchmark detail (metrics, slices, signal summaries from
    evals_view). evals_count comes from the composites dim — sum of
    per-(composite, benchmark, metric) triples — not a sum of
    per-benchmark models_count, which would double-count cross-
    benchmark models.
    """
    composite_evals_count = dict(
        con.execute(
            "SELECT composite_slug, evals_count FROM composites"
        ).fetchall()
    )

    # LEFT JOIN preserves the bare-parent shells (e.g. arc-agi) so their level
    # slices have a root to nest under in the hierarchy tree. Shell rows have
    # NULL evaluation_id (they're absent from evals_view, which drops them) —
    # that null signals "structural anchor only, not navigable" to the frontend.
    rows = con.execute(
        """
        SELECT
            b.composite_slug,
            b.composite_display_name,
            b.benchmark_id,
            b.display_name,
            b.family_id,
            b.is_slice,
            b.parent_benchmark_id,
            b.card_present,
            b.domains, b.languages, b.tasks,
            ev.evaluation_id, ev.evaluation_name, ev.derived_tags,
            ev.models_count,
            ev.reproducibility_summary, ev.provenance_summary,
            ev.comparability_summary
        FROM benchmarks b
        LEFT JOIN evals_view ev
          ON ev.composite_slug = b.composite_slug
         AND ev.benchmark_id   = b.benchmark_id
        ORDER BY b.composite_slug, b.benchmark_id
        """
    ).fetchall()
    cols = [d[0] for d in con.description]
    benchmarks = [dict(zip(cols, r)) for r in rows]

    # Bucket by composite_slug. Within each composite, group rows into
    # root benchmarks (is_slice=FALSE) and slice rows attached to their
    # root via parent_benchmark_id.
    from collections import defaultdict
    by_composite: dict[str, list[dict]] = defaultdict(list)
    for b in benchmarks:
        by_composite[b["composite_slug"]].append(b)

    out: list[dict] = []
    for slug in sorted(by_composite):
        members = by_composite[slug]
        display_name = next(
            (m["composite_display_name"] for m in members
             if m["composite_display_name"]),
            slug,
        )
        evals_count = int(composite_evals_count.get(slug) or 0)
        # Group slices under their root benchmark.
        roots = [m for m in members if not m["is_slice"]]
        slices_by_root: dict[str, list[dict]] = defaultdict(list)
        for m in members:
            if m["is_slice"]:
                root_id = m["parent_benchmark_id"] or m["benchmark_id"]
                slices_by_root[root_id].append(m)
            elif (m["parent_benchmark_id"] is not None
                  and m["parent_benchmark_id"] == m["benchmark_id"]):
                # Self-parented bare-stem (e.g. gaia → gaia): include
                # the bare stem as a slice of itself, marked is_bare_stem.
                slices_by_root[m["benchmark_id"]].append({
                    **m, "_is_bare_stem": True,
                })

        # Handle fact-less shell roots (evaluation_id=null).
        # Shells with slices are kept as structural grouping nodes —
        # they aren't navigable but their children carry the data.
        # Shells without slices are truly empty and are dropped.
        navigable_roots = []
        for root in roots:
            if root.get("evaluation_id") is not None:
                navigable_roots.append(root)
            elif root["benchmark_id"] in slices_by_root:
                navigable_roots.append(root)
            # else: truly empty shell with no slices — drop silently
        roots = navigable_roots

        bench_records = [
            _hierarchy_composite_benchmark(con, slug, root, slices_by_root.get(root["benchmark_id"], []))
            for root in roots
        ]
        bench_records.sort(key=lambda r: r["key"])

        # Composite-level rollups: union of tags and counts across roots.
        domains = sorted({t for r in bench_records for t in r["tags"]["domains"]})
        languages = sorted({t for r in bench_records for t in r["tags"]["languages"]})
        tasks = sorted({t for r in bench_records for t in r["tags"]["tasks"]})
        repro = _aggregate_reproducibility(roots)
        prov = _aggregate_provenance(roots)
        comp = _aggregate_comparability(roots)

        out.append({
            "key":          slug,
            "display_name": display_name,
            "tags":         {"domains": domains, "languages": languages, "tasks": tasks},
            "evals_count":  evals_count,
            "benchmarks":   bench_records,
            "reproducibility_summary": repro,
            "provenance_summary":      prov,
            "comparability_summary":   comp,
        })
    return out


# ---------------------------------------------------------------------------
# Display polish: prettify_display + ACRONYMS handling
# ---------------------------------------------------------------------------


# Loaded lazily from `<registry_root>/display_overrides.yaml`. Curators
# add slugs that should render uppercase (e.g. "mmlu" → "MMLU") rather
# than the default title-case ("Mmlu"). Falls back to a hard-coded set
# when the file is unavailable so the producer doesn't degrade silently.
_FALLBACK_ACRONYMS: frozenset[str] = frozenset({
    "gaia", "gpqa", "mmlu", "mmmu", "bfcl", "helm", "ifeval", "bbh",
    "mmlu-pro", "musr", "math", "gsm8k", "arc-agi", "swe-bench",
    "ace", "hle", "aime", "hal", "usaco",
})

_acronyms_cache: frozenset[str] | None = None
_bench_display_names_cache: dict[str, str] | None = None


def _load_display_overrides() -> tuple[frozenset[str], dict[str, str]]:
    """Load both acronyms and benchmark_display_names from display_overrides.yaml."""
    global _acronyms_cache, _bench_display_names_cache
    if _acronyms_cache is not None and _bench_display_names_cache is not None:
        return _acronyms_cache, _bench_display_names_cache
    candidates = [
        Path(".cache/entity_registry/display_overrides.yaml"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            import yaml as _yaml
            data = _yaml.safe_load(path.read_text()) or {}
        except (ImportError, OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        items = data.get("acronyms")
        if isinstance(items, list):
            _acronyms_cache = frozenset(str(s).lower() for s in items if s)
        names = data.get("benchmark_display_names")
        if isinstance(names, dict):
            _bench_display_names_cache = {
                str(k).lower(): str(v) for k, v in names.items()
            }
        break
    if _acronyms_cache is None:
        _acronyms_cache = _FALLBACK_ACRONYMS
    if _bench_display_names_cache is None:
        _bench_display_names_cache = {}
    return _acronyms_cache, _bench_display_names_cache


def _load_acronyms() -> frozenset[str]:
    """Read curated acronyms from the registry cache. Cached at module
    level — the seed YAML is small and immutable per snapshot."""
    global _acronyms_cache
    if _acronyms_cache is not None:
        return _acronyms_cache
    candidates = [
        Path(".cache/entity_registry/display_overrides.yaml"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            import yaml as _yaml
            data = _yaml.safe_load(path.read_text()) or {}
        except (ImportError, OSError, ValueError):
            continue
        items = data.get("acronyms") if isinstance(data, dict) else None
        if isinstance(items, list):
            _acronyms_cache = frozenset(str(s).lower() for s in items if s)
            return _acronyms_cache
    _acronyms_cache = _FALLBACK_ACRONYMS
    return _acronyms_cache


def _title_segment(seg: str, acronyms: frozenset[str]) -> str:
    """Title-case one segment of a slug, preserving curated acronyms.
    Numeric / version-like tokens (1m, v1, 2024) stay as-is."""
    if not seg:
        return seg
    if seg.lower() in acronyms:
        return seg.upper()
    if seg.replace(".", "").isdigit() or re.fullmatch(r"v\d+(\.\d+)*", seg):
        return seg
    if seg[0].isalpha() and seg[0].islower():
        return seg[:1].upper() + seg[1:]
    return seg


def prettify_display(name: str | None) -> str:
    """Cleanup an arbitrary slug or eval label into a tidy human title.

    - None / empty → "" (caller can fall back to id).
    - Underscores become spaces.
    - Mixed-case input (already curated, e.g. "MMLU-Pro" or
      "Humanity's Last Exam") is returned unchanged.
    - All-lowercase input is title-cased per hyphen-segment with
      curated acronyms preserved.
    """
    if not name:
        return ""
    _, bench_names = _load_display_overrides()
    override = bench_names.get(name.lower())
    if override:
        return override
    cleaned = name.replace("_", " ").strip()
    if any(c.isupper() for c in cleaned):
        return cleaned
    acronyms = _load_acronyms()
    words = cleaned.split()
    out: list[str] = []
    for w in words:
        if "-" in w:
            out.append("-".join(_title_segment(s, acronyms) for s in w.split("-")))
        else:
            out.append(_title_segment(w, acronyms))
    return " ".join(out)


# Metric-tail detector. When evaluation_name = "{slice} {metric}" without
# an explicit metric_config.metric_name, the producer's slice_key carries
# the slice portion — but rendering may still benefit from peeling the
# trailing metric off display labels.
_METRIC_TAIL_RE = re.compile(
    r"\s+("
    r"(?:mean\s+)?(?:score|accuracy|f1|em|loss)|"
    r"pass@\d+|pass@k|"
    r"standard\s+error|win\s+rate|"
    r"avg(?:\s+(?:attempts|latency(?:_ms)?))?|"
    r"average\s+attempts|"
    r"mean|rank|elo"
    r")$",
    re.IGNORECASE,
)


def peel_metric_tail(label: str | None) -> tuple[str, str | None]:
    """Split a display label into (slice_label, metric_tail) when it
    looks like "X Pass@1" / "Overall Mean Score". When no tail matches,
    returns (label, None). The peeled metric_tail is the trailing
    metric phrase; consumers usually move it from the slice display
    onto the metric label.
    """
    if not label:
        return ("", None)
    m = _METRIC_TAIL_RE.search(label)
    if m and m.start() > 0:
        return (label[: m.start()].rstrip(), label[m.start():].strip())
    return (label, None)


# Primary-metric preference list, in order.
# Compared case-insensitively against metric_display_name. Earlier
# entries win.
_PRIMARY_METRIC_PREFERENCE: tuple[str, ...] = (
    "overall", "mean win rate", "mean score", "score", "accuracy",
    "exact match", "exact_match", "win rate", "elo", "rank",
    "pass@1", "f1", "mean",
)


# Explicit overrides for which benchmark is the "primary readout" of a
# family or composite — i.e. the row whose primary metric the frontend
# surfaces as the family's headline number.
# Add entries here when the heuristic (first-with-is_overall, else
# first by sort) picks the wrong row for a family.
_FAMILY_PRIMARY_OVERRIDE: dict[str, str] = {
    "artificial-analysis": "artificial-analysis-intelligence-index",
}


def _mark_family_primary_benchmark(
    family_key: str, benchmarks: list[dict]
) -> None:
    """Mutate `benchmarks` in place: set `is_overall` (this row IS the
    family root) and `is_primary` (this row is the family's headline
    reading).

    `is_overall`: True iff `benchmark.key == family_key`. A multi-bench
    family without a head benchmark of the same name (HAL, BFCL family
    with no `bfcl` benchmark) has no overall row — all False.

    `is_primary` selection:
      1. _FAMILY_PRIMARY_OVERRIDE explicit map (curator-supplied).
      2. The benchmark with `is_overall=True` (the family-root row).
      3. The first benchmark by ascending key (stable tie-break).
    """
    if not benchmarks:
        return
    for b in benchmarks:
        b["is_overall"] = (b["key"] == family_key)

    override = _FAMILY_PRIMARY_OVERRIDE.get(family_key)
    if override:
        primary_key = override
    else:
        overall = next((b for b in benchmarks if b["is_overall"]), None)
        primary_key = (overall or sorted(benchmarks, key=lambda x: x["key"])[0])["key"]
    for b in benchmarks:
        b["is_primary"] = (b["key"] == primary_key)


def _pick_primary_metric_key(metrics: list[dict]) -> str | None:
    """Return the metric_key of the primary metric for a benchmark, or
    None when the benchmark has no metrics:

      1. First metric whose display name (case-insensitive) matches an
         entry in `_PRIMARY_METRIC_PREFERENCE`, in preference order.
      2. Fallback: most-reported metric (highest `models_count`,
         tie-break alphabetical on metric_key).
    """
    if not metrics:
        return None
    by_name = {(m.get("display_name") or m["key"]).strip().lower(): m
               for m in metrics}
    for pref in _PRIMARY_METRIC_PREFERENCE:
        if pref in by_name:
            return by_name[pref]["key"]
    best = max(
        metrics,
        key=lambda m: (int(m.get("models_count") or 0), -ord(m["key"][:1] or " ")),
    )
    return best["key"]


def _hierarchy_composite_benchmark(
    con, composite_slug: str, root: dict, slice_rows: list[dict]
) -> dict:
    """Build one benchmark sub-record under composites[].benchmarks[].

    Slices are the within-benchmark cuts for this (composite, benchmark)
    pair. Each slice carries its own metric list. The bare-stem slice
    (when one exists, e.g. `gaia` inside the `gaia` benchmark) is
    flagged with is_bare_stem=true so the frontend can render it as
    "Overall" / "Main".

    Also emits:
      - `primary_metric_key`: the canonical readout among this
        benchmark's metrics (see `_pick_primary_metric_key`).
      - `metrics[].is_primary`: True for the metric matching
        `primary_metric_key`.
      - `is_overall`: True when this benchmark IS the family/composite
        root (canonical_id matches the family or composite key).
        is_primary at the family level (across siblings) is added
        later by the family-rollup pass.
    """
    benchmark_id = root["benchmark_id"]
    # Pull metric meta + per-metric model coverage. models_count drives
    # the primary-metric tie-break when no display name matches the
    # preference list.
    metrics_rows = con.execute(
        """
        SELECT metric_id,
               ANY_VALUE(metric_display_name)              AS display_name,
               ARRAY_AGG(DISTINCT source_metadata.source_organization_name)
                   FILTER (WHERE source_metadata.source_organization_name IS NOT NULL)
                   AS sources,
               COUNT(DISTINCT model_key)                   AS models_count
        FROM eval_results_view
        WHERE composite_slug = ? AND benchmark_id = ?
        GROUP BY metric_id
        ORDER BY metric_id
        """,
        [composite_slug, benchmark_id],
    ).fetchall()

    eval_ids_row = con.execute(
        "SELECT ARRAY_AGG(DISTINCT evaluation_id ORDER BY evaluation_id) "
        "FROM evals_view WHERE composite_slug = ? AND benchmark_id = ?",
        [composite_slug, benchmark_id],
    ).fetchone()
    constituent_ids = eval_ids_row[0] if eval_ids_row and eval_ids_row[0] else []
    if not constituent_ids:
        # Shell/rollup benchmark with no eval row of its own (e.g. bfcl,
        # rewardbench-2): its data lives entirely in slice-child evals
        # (parent_benchmark_id = this benchmark). Surface those real child
        # eval ids so consumers link to a real eval (and eval-detail split
        # navigation works) instead of a synthetic dead id.
        child_row = con.execute(
            "SELECT ARRAY_AGG(DISTINCT evaluation_id ORDER BY evaluation_id) "
            "FROM evals_view WHERE composite_slug = ? AND parent_benchmark_id = ?",
            [composite_slug, benchmark_id],
        ).fetchone()
        constituent_ids = child_row[0] if child_row and child_row[0] else []

    metrics: list[dict] = [
        {
            "key":           m[0],
            "display_name":  m[1] or m[0],
            "sources":       m[2] or [],
            "models_count":  int(m[3] or 0),
        }
        for m in metrics_rows
    ]
    primary_metric_key = _pick_primary_metric_key(metrics)
    for m in metrics:
        m["is_primary"] = (m["key"] == primary_metric_key)

    family_id = root.get("family_id") or benchmark_id
    # is_overall is set by the family rollup pass (it's family-relative,
    # not just benchmark-self-relative — a benchmark only IS the family
    # root in the context of the family it gets rendered under). Default
    # to False here; _mark_family_primary_benchmark flips the right one.

    # Display name: curated `display_name` from canonical_benchmarks
    # wins, but only when it actually differs from the slug. The
    # auto-create path sets `display_name = benchmark_id` for unmatched
    # raw strings (e.g. "format_sensitivity" / "memory"); prettify those
    # so the UI doesn't show "format_sensitivity" verbatim.
    raw_display = root.get("display_name")
    if raw_display and raw_display != benchmark_id:
        bench_display = raw_display
    else:
        bench_display = prettify_display(benchmark_id)

    return {
        "key":          benchmark_id,
        "display_name": bench_display,
        "family_id":    family_id,
        "is_slice":     False,
        "is_overall":   False,
        "primary_metric_key": primary_metric_key,
        "has_card":     bool(root.get("card_present"))
                        if root.get("card_present") is not None else False,
        "tags": {
            "domains":   list(root.get("domains") or []),
            "languages": list(root.get("languages") or []),
            "tasks":     list(root.get("tasks") or []),
        },
        "metrics":   metrics,
        "slices":   _hierarchy_composite_slices(
                        con, composite_slug, benchmark_id, slice_rows),
        "constituent_evaluation_ids": constituent_ids,
        "reproducibility_summary": root.get("reproducibility_summary"),
        "provenance_summary":      root.get("provenance_summary"),
        "comparability_summary":   root.get("comparability_summary"),
    }


def _hierarchy_composite_slices(
    con, composite_slug: str, benchmark_id: str, slice_rows: list[dict]
) -> list[dict]:
    """Per-(composite, benchmark) slices[]: within-benchmark cuts.

    Two slice mechanisms are unioned:

      1. **Sibling-benchmark slices** (`slice_rows`) — separate
         canonical benchmarks parented to the same root. GAIA's
         `gaia-level-1/2/3` and the self-parented `gaia` "Overall"
         row, CapArena's `caparena-vs-X`, GPQA's `gpqa-diamond`. These
         are full (composite, benchmark) rows in the dim with their
         own metrics/orgs in `eval_results_view`; the slice key is
         the slice benchmark's id and `is_bare_stem=true` when it
         equals the root.
      2. **slice_key cuts** (within-benchmark) — MMLU subjects style,
         where multiple raw evaluation_names collapse to one
         canonical id and Stage C keeps the raw on each fact row's
         `slice_key`. Read from `fact_results` keyed on `slice_key`.

    The two never collide in practice (a benchmark uses one mechanism
    or the other), but a UNION at the output level is the safe contract.
    """
    out: list[dict] = []

    # (1) Sibling-benchmark slices from the benchmarks dim.
    for sr in slice_rows:
        slice_id = sr["benchmark_id"]
        rows = con.execute(
            """
            SELECT
                erv.metric_id,
                ANY_VALUE(erv.metric_display_name) AS metric_display,
                ARRAY_AGG(DISTINCT erv.source_metadata.source_organization_name)
                    FILTER (WHERE erv.source_metadata.source_organization_name IS NOT NULL)
                    AS sources
            FROM eval_results_view erv
            WHERE erv.composite_slug = ? AND erv.benchmark_id = ?
            GROUP BY erv.metric_id
            ORDER BY erv.metric_id
            """,
            [composite_slug, slice_id],
        ).fetchall()
        out.append({
            "key":          slice_id,
            "display_name": sr.get("display_name") or slice_id,
            "is_bare_stem": bool(sr.get("_is_bare_stem"))
                            or slice_id == benchmark_id,
            "metrics": [
                {
                    "key":          r[0],
                    "display_name": r[1] or r[0],
                    "sources":      list(r[2] or []),
                }
                for r in rows
            ],
        })

    # (2) Within-benchmark slice_key cuts (MMLU subjects, etc.).
    rows = con.execute(
        """
        WITH per_slice_metric AS (
            -- The bound `?` benchmark parameter comes from the
            -- benchmarks dim's `benchmark_id`, which under the
            -- root-grain refactor is the canonical-or-raw key. Match
            -- it against fr.benchmark_key (not the canonical-only
            -- fr.benchmark_id) so raw-only benchmarks find their
            -- slices.
            SELECT
                fr.slice_key,
                fr.metric_key                            AS metric_id,
                MIN(fr.slice_name)                       AS slice_name_rep,
                ANY_VALUE(cmet.display_name)             AS metric_display,
                ARRAY_AGG(DISTINCT fr.org_raw)
                    FILTER (WHERE fr.org_raw IS NOT NULL) AS sources
            FROM fact_results fr
            LEFT JOIN canonical_metrics cmet ON cmet.id = fr.metric_key
            WHERE fr.composite_slug = ?
              AND fr.benchmark_key = ?
              AND fr.slice_key IS NOT NULL
              AND fr.metric_key IS NOT NULL
            GROUP BY fr.slice_key, fr.metric_key
        )
        SELECT
            slice_key,
            MIN(slice_name_rep) AS slice_display_name,
            ARRAY_AGG(struct_pack(
                metric_id      := metric_id,
                metric_display := metric_display,
                sources        := sources
            ) ORDER BY metric_id) AS metrics
        FROM per_slice_metric
        GROUP BY slice_key
        ORDER BY slice_key
        """,
        [composite_slug, benchmark_id],
    ).fetchall()
    sibling_keys = {s["key"] for s in out}
    for row in rows:
        slice_key = row[0]
        # Skip if a sibling-benchmark slice already covers this id (the
        # bare-stem `gaia` slice_key would otherwise duplicate the
        # `gaia` self-parented sibling slice).
        if slice_key in sibling_keys:
            continue
        out.append({
            "key":          slice_key,
            "display_name": row[1] or slice_key,
            "is_bare_stem": slice_key == benchmark_id,
            "metrics": [
                {
                    "key":          m["metric_id"],
                    "display_name": m["metric_display"] or m["metric_id"],
                    "sources":      list(m["sources"] or []),
                }
                for m in row[2]
            ],
        })

    out.sort(key=lambda s: s["key"])
    return out


def _hierarchy_families_index(con) -> list[dict]:
    """Lightweight family lookup index. One entry per family_id with the
    member benchmark keys. No per-composite info — that lives under
    composites[].
    """
    rows = con.execute(
        """
        SELECT family_id, family_display_name, member_benchmark_keys
        FROM families
        ORDER BY family_id
        """
    ).fetchall()
    return [
        {
            "key":                   r[0],
            "display_name":          r[1] or r[0],
            "member_benchmark_keys": list(r[2] or []),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Phantom / display-name helpers (used by Stage G phantom synthesis and
# the composite hierarchy aggregations).
# ---------------------------------------------------------------------------


# Acronyms preserved verbatim when title-casing a stem fallback display name.
# Keep small — only well-known names where the registry-style display would
# otherwise lose mixed-case (CapArena, GAIA, GPQA, MMLU, etc.).
_KNOWN_ACRONYMS: dict[str, str] = {
    "gaia":     "GAIA",
    "gpqa":     "GPQA",
    "mmlu":     "MMLU",
    "bfcl":     "BFCL",
    "helm":     "HELM",
    "bbh":      "BBH",
    "bbq":      "BBQ",
    "ifeval":   "IFEval",
    "musr":     "MuSR",
    "caparena": "CapArena",
    "videomme": "VideoMME",
    "math":     "MATH",
    "aime":     "AIME",
    "lcb":      "LCB",
    "ace":      "ACE",
    "mmmu":     "MMMU",
    "arc":      "ARC",
    "agi":      "AGI",
    "swe":      "SWE",
}


def _title_case_stem(stem: str) -> str:
    if stem in _KNOWN_ACRONYMS:
        return _KNOWN_ACRONYMS[stem]
    parts = [_KNOWN_ACRONYMS.get(p, p.capitalize()) for p in stem.split("-") if p]
    return " ".join(parts) if parts else stem


def _common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        i = 0
        while i < len(prefix) and i < len(s) and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            return ""
    # If the prefix cuts mid-word in any of the originals (next char in
    # that string is alphanumeric), back off to the last word boundary so
    # we don't return ragged stubs like "ARC-AGI v".
    if any(
        len(s) > len(prefix) and s[len(prefix)].isalnum()
        for s in strings
    ):
        last_ws = max(
            (i for i, ch in enumerate(prefix) if ch.isspace()),
            default=-1,
        )
        prefix = prefix[:last_ws] if last_ws >= 0 else ""
    prefix = prefix.rstrip(" \t-_:/")
    # Drop trailing one-letter alphabetic tokens (e.g. "Videomme W" →
    # "Videomme") — they're artefacts of the LCP. Numeric tokens stay
    # ("RewardBench 2" should not collapse to "RewardBench").
    while True:
        tokens = prefix.split()
        if (
            len(tokens) >= 2
            and len(tokens[-1]) == 1
            and tokens[-1].isalpha()
        ):
            prefix = " ".join(tokens[:-1])
        else:
            break
    return prefix.rstrip(" \t-_:/")


def _aggregate_reproducibility(members: list[dict]) -> dict | None:
    """Sum results_total + has_reproducibility_gap_count; recompute
    populated_ratio_avg as a results-weighted mean."""
    total = 0
    gap_count = 0
    weighted_sum = 0.0
    weight = 0
    seen = False
    for m in members:
        s = m.get("reproducibility_summary")
        if not s:
            continue
        seen = True
        n = int(s.get("results_total") or 0)
        total += n
        gap_count += int(s.get("has_reproducibility_gap_count") or 0)
        ratio = s.get("populated_ratio_avg")
        if ratio is not None and n > 0:
            weighted_sum += float(ratio) * n
            weight += n
    if not seen:
        return None
    return {
        "results_total":                 total,
        "has_reproducibility_gap_count": gap_count,
        "populated_ratio_avg":           (weighted_sum / weight) if weight else None,
    }


def _aggregate_provenance(members: list[dict]) -> dict | None:
    out = {
        "total_results":           0,
        "total_groups":            0,
        "multi_source_groups":     0,
        "first_party_only_groups": 0,
        "source_type_distribution": {
            "first_party":   0,
            "third_party":   0,
            "collaborative": 0,
            "unspecified":   0,
        },
    }
    seen = False
    for m in members:
        s = m.get("provenance_summary")
        if not s:
            continue
        seen = True
        out["total_results"] += int(s.get("total_results") or 0)
        out["total_groups"] += int(s.get("total_groups") or 0)
        out["multi_source_groups"] += int(s.get("multi_source_groups") or 0)
        out["first_party_only_groups"] += int(s.get("first_party_only_groups") or 0)
        dist = s.get("source_type_distribution") or {}
        for k in out["source_type_distribution"]:
            out["source_type_distribution"][k] += int(dist.get(k) or 0)
    return out if seen else None


def _aggregate_comparability(members: list[dict]) -> dict | None:
    out = {
        "total_groups":                  0,
        "groups_with_variant_check":     0,
        "groups_with_cross_party_check": 0,
        "variant_divergent_count":       0,
        "cross_party_divergent_count":   0,
    }
    seen = False
    for m in members:
        s = m.get("comparability_summary")
        if not s:
            continue
        seen = True
        for k in out:
            out[k] += int(s.get(k) or 0)
    return out if seen else None


# ---------------------------------------------------------------------------
# comparison-index.json
# ---------------------------------------------------------------------------


# Tab-strip ordering that the frontend's plotbox expects. Capability surfaces
# first (the actual task score), then capability-adjacent groups, then
# instrumental groups, with "other" as the fallback bucket.
_METRIC_GROUP_ORDER = (
    "capability",
    "robustness",
    "efficiency",
    "cost",
    "latency",
    "rank",
    "other",
)
_METRIC_GROUP_INDEX = {group: i for i, group in enumerate(_METRIC_GROUP_ORDER)}


# Authoritative mapping when `metric_kind` is populated on the fact row
# (registry preferred, EEE_meta fallback, regex inference last — see
# `metric_meta_hotfix._infer_metric_kind_from_name`). Mirrors the legacy
# producer's table so cutover doesn't reshuffle the tab strip.
_METRIC_KIND_TO_GROUP: dict[str, str] = {
    "accuracy":   "capability",
    "elo":        "capability",
    "score":      "capability",
    "pass":       "capability",
    "f1":         "capability",
    "win_rate":   "capability",
    "winrate":    "capability",
    "cost":       "cost",
    "latency":    "latency",
    "throughput": "latency",
    "time":       "latency",
    "rank":       "rank",
    "difference": "robustness",
}

# Order matters: first matching pattern wins, listed most-specific first
# so e.g. "Latency Standard Deviation" lands in latency rather than
# robustness. Used when `metric_kind` is absent — covers the long tail of
# metric names the hotfix UDF couldn't classify upstream.
_METRIC_NAME_GROUP_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("cost",       re.compile(r"\b(?:cost|usd|dollar|price)\b", re.IGNORECASE)),
    ("latency",    re.compile(
        r"\b(?:latency|throughput|elapsed|wall[\s_]?time|"
        r"tokens?[\s_/]?(?:per|sec|s)\b|p\d{2,3}|percentile)\b",
        re.IGNORECASE,
    )),
    ("rank",       re.compile(r"\brank\b", re.IGNORECASE)),
    ("robustness", re.compile(
        r"\b(?:sensitivity|delta|stddev|standard[\s_]?deviation|"
        r"variance|robustness)\b",
        re.IGNORECASE,
    )),
    ("efficiency", re.compile(r"\b(?:attempts|retries|tries)\b", re.IGNORECASE)),
    ("capability", re.compile(
        r"\b(?:accuracy|acc|elo|score|pass@\d+|win[\s_]?rate|f1|"
        r"exact[\s_]?match|em|bleu|rouge(?:-\d+)?|recall|precision|"
        r"mrr|ndcg|coverage|correct|harmlessness)\b",
        re.IGNORECASE,
    )),
)


def _classify_metric_group(metric_kind: str | None, metric_name: str | None) -> str:
    """Return the tab-strip bucket for a metric.

    Precedence: `metric_kind` (registry > EEE > inferred) → name regex →
    `"other"`. Caller is responsible for passing the metric_kind sourced
    from `fact_results`, where the hotfix UDF has already applied that
    precedence at canonicalisation time.
    """
    if metric_kind:
        kind = metric_kind.strip().lower()
        if kind in _METRIC_KIND_TO_GROUP:
            return _METRIC_KIND_TO_GROUP[kind]
    if metric_name:
        for group, pattern in _METRIC_NAME_GROUP_RULES:
            if pattern.search(metric_name):
                return group
    return "other"


def write_comparison_index(con, out_dir: Path, snapshot_meta: dict) -> Path:
    """Per-(eval, metric) leaderboards + inverse model→peer index.

    Backs the grid view on the model detail page. The frontend's
    `plotboxUnits` skips any eval not present here (`comparisonIndex.evals[id]
    ?? continue`), so this artifact's keyset must cover every evaluation_id
    in `eval_results_view`.

    `eval_results_view` already collapses fact rows to one row per
    `(model_key, benchmark_id, metric_id)` triple — the legacy producer's
    submission-tail logic doesn't apply, so every leaderboard row carries
    `submission_count=1, submission_axis="default"`. If/when the view layer
    starts preserving multiple submissions per triple, this is the single
    place that needs to learn about it.
    """
    # `metric_kind` is per-metric within a benchmark; pre-aggregate from
    # fact_results once rather than carry it on every cell row. Mirrors the
    # MAX-FILTER pattern stage I uses when packing it into metric_config.
    rows = con.execute(
        """
        WITH metric_kinds AS (
            SELECT
                benchmark_key,
                metric_key,
                MAX(metric_kind) FILTER (WHERE metric_kind IS NOT NULL) AS metric_kind
            FROM fact_results
            WHERE benchmark_key IS NOT NULL AND metric_key IS NOT NULL
            GROUP BY benchmark_key, metric_key
        )
        SELECT
            erv.evaluation_id,
            erv.metric_summary_id,
            erv.metric_id,
            erv.metric_display_name,
            erv.metric_unit,
            erv.lower_is_better,
            erv.score,
            erv.model_key,
            erv.model_id,
            erv.model_route_id,
            -- Decoder knobs from the row's generation_config struct; the
            -- per-source detail rows on the model page read these off the
            -- score cells (no eval-id join is possible from reported-results).
            erv.generation_config.generation_args.temperature AS temperature,
            erv.generation_config.generation_args.max_tokens  AS max_tokens,
            -- Group key (renamed `model_family_id` -> `model_group_id` on
            -- models_view). Aliased back to `model_family_id` so the rest
            -- of this builder and the emitted JSON field keep their name
            -- (frontend contract) while carrying the same group-key value.
            mv.model_group_id AS model_family_id,
            mv.model_family_name,
            mv.developer,
            mk.metric_kind
        FROM eval_results_view erv
        -- Join on model_key (root-grain identity) so unresolved models
        -- still pick up models_view entries via the raw fallback.
        LEFT JOIN models_view mv
          ON mv.model_key = erv.model_key
        LEFT JOIN metric_kinds mk
          ON mk.benchmark_key = erv.benchmark_id
         AND mk.metric_key    = erv.metric_id
        WHERE erv.score             IS NOT NULL
          AND erv.evaluation_id     IS NOT NULL
          AND erv.metric_summary_id IS NOT NULL
          AND erv.model_route_id    IS NOT NULL
        """
    ).fetchall()
    cols = [d[0] for d in con.description]

    # parent_benchmark_id comes from the benchmarks dim. For roots
    # (the eval *is* the benchmark) this is NULL; for slice rows it
    # carries the parent benchmark's id so the frontend's
    # getCompositeKey() fallback chain has a parent to fall back to.
    # Use is_slice as the gate: canonical_benchmarks stores
    # parent_benchmark_id == benchmark_id for roots in some cases, so
    # null-out the root path explicitly rather than trusting the raw
    # column value.
    eval_meta_rows = con.execute(
        """
        SELECT
            ev.evaluation_id,
            ev.evaluation_name,
            ev.canonical_display_name,
            ev.composite_slug,
            ev.composite_display_name,
            ev.benchmark_id,
            ev.family_id,
            ev.family_display_name,
            ev.is_slice,
            CASE WHEN ev.is_slice THEN b.parent_benchmark_id ELSE NULL END
                AS parent_benchmark_id,
            ev.derived_tags,
            ev.is_summary_score
        FROM evals_view ev
        LEFT JOIN benchmarks b
          ON b.composite_slug = ev.composite_slug
         AND b.benchmark_id   = ev.benchmark_id
        """
    ).fetchall()
    eval_meta_cols = [d[0] for d in con.description]
    eval_meta = {
        r[0]: dict(zip(eval_meta_cols, r))
        for r in eval_meta_rows
        if r[0] is not None
    }

    # Group cells by (evaluation_id, metric_summary_id) — this is the
    # leaderboard key.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        rec = dict(zip(cols, r))
        grouped[(rec["evaluation_id"], rec["metric_summary_id"])].append(rec)

    eval_metric_buckets: dict[str, list[dict]] = defaultdict(list)
    by_model: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for (eval_id, metric_summary_id), peer_rows in grouped.items():
        first = peer_rows[0]
        lower_is_better = bool(first["lower_is_better"])

        # Two-pass stable sort: route id ascending tiebreak, then score in
        # the metric's preferred direction. Mirrors the legacy producer's
        # ordering so existing UI ranks don't shift on cutover.
        peer_rows.sort(key=lambda r: r["model_route_id"])
        peer_rows.sort(
            key=lambda r: r["score"], reverse=not lower_is_better
        )

        total = len(peer_rows)
        scores_out: list[dict] = []
        position = 0
        previous_score = None
        for idx, rec in enumerate(peer_rows, start=1):
            sc = rec["score"]
            # Dense-tie ranking: position only advances when score changes,
            # so peers at the same score share a rank. Matches legacy.
            if previous_score is None or sc != previous_score:
                position = idx
                previous_score = sc

            scores_out.append({
                "model_route_id":    rec["model_route_id"],
                "model_family_id":   rec["model_family_id"]
                                       or rec["model_id"]
                                       or rec["model_key"],
                "model_family_name": rec["model_family_name"] or "",
                "developer":         rec["developer"] or "",
                "variant_key":       "default",
                "score":             sc,
                "rank":              position,
                "total":             total,
                "submission_count":  1,
                "submission_axis":   "default",
                "temperature":       rec["temperature"],
                "max_tokens":        rec["max_tokens"],
            })
            by_model[rec["model_route_id"]][eval_id][metric_summary_id] = {
                "score":            sc,
                "rank":             position,
                "total":            total,
                "submission_count": 1,
                "submission_axis":  "default",
                "temperature":      rec["temperature"],
                "max_tokens":       rec["max_tokens"],
            }

        group = _classify_metric_group(
            first.get("metric_kind"), first.get("metric_display_name")
        )
        eval_metric_buckets[eval_id].append({
            "metric_summary_id": metric_summary_id,
            "metric_name":       first["metric_display_name"] or "",
            "metric_id":         first["metric_id"],
            "metric_key":        first["metric_id"],
            "group":             group,
            "group_order":       _METRIC_GROUP_INDEX[group],
            "lower_is_better":   lower_is_better,
            "unit":              first["metric_unit"],
            "scores":            scores_out,
        })

    evals_out: dict[str, dict] = {}
    for eval_id, metrics in eval_metric_buckets.items():
        meta = eval_meta.get(eval_id, {})
        # Capability tabs surface first (so the actual task score is the
        # default tab on the histogram strip), then the rest of the group
        # taxonomy. Within-group ordering stays alphabetical for determinism.
        metrics.sort(
            key=lambda m: (m["group_order"], m["metric_name"] or "", m["metric_summary_id"])
        )
        evals_out[eval_id] = {
            "evaluation_id":           eval_id,
            "composite_slug":          meta.get("composite_slug"),
            "composite_display_name":  meta.get("composite_display_name"),
            "benchmark_id":            meta.get("benchmark_id"),
            "family_id":               meta.get("family_id"),
            "family_display_name":     meta.get("family_display_name"),
            "parent_benchmark_id":     meta.get("parent_benchmark_id"),
            "is_slice":                bool(meta.get("is_slice")),
            "display_name":            meta.get("canonical_display_name")
                                         or meta.get("evaluation_name"),
            "derived_tags":            json.loads(meta.get("derived_tags") or "[]"),
            "is_summary_score":        bool(meta.get("is_summary_score")),
            "summary_score_for":       None,
            # dropped `summary_eval_ids` — it was sourced from the dead
            # evals_view column (always []) and never read by consumers.
            "metrics":                 metrics,
        }

    payload = {
        "generated_at":       snapshot_meta["snapshot_id"],
        "config_version":     CONFIG_VERSION,
        "metric_group_order": list(_METRIC_GROUP_ORDER),
        "evals":              evals_out,
        "by_model":           {
            route: {ev: dict(metrics) for ev, metrics in evs.items()}
            for route, evs in by_model.items()
        },
    }
    path = out_dir / "comparison-index.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


# ---------------------------------------------------------------------------
# benchmark_index.json
# ---------------------------------------------------------------------------


def write_benchmark_index(con, out_dir: Path, snapshot_meta: dict) -> Path:
    """Per-benchmark cross-composite appearance index.

    Keyed by canonical `benchmark_id`. Each entry lists every composite
    reporting that benchmark, with the primary-metric aggregate stats
    pulled straight from `evals_view`.

    Apples-to-apples warning: different composites can pick different
    primary metrics for the same benchmark (e.g. MMLU as `accuracy` in
    one composite and `normalized_accuracy` in another). Each appearance
    carries its own `primary_metric_id` + display name so consumers can
    detect heterogeneity and choose to compare directly, filter to a
    shared metric, or normalise. We deliberately don't pre-bake any
    cross-appearance roll-up: that's policy, and consumers can compute
    weighted averages from `avg_score` × `models_count` themselves.

    Slice rows (e.g. `gpqa-diamond` under `gpqa`) appear as their own
    keys — they're benchmarks in the dim. `is_slice` and
    `parent_benchmark_id` are at the entry level so consumers can fold
    slices into their parent if they want to.
    """
    rows = con.execute(
        """
        SELECT
            ev.benchmark_id,
            ev.family_id,
            ev.family_display_name,
            ev.is_slice,
            ev.parent_benchmark_id,
            ev.composite_slug,
            ev.composite_display_name,
            ev.evaluation_id,
            ev.primary_metric_id,
            ev.metric_config.evaluation_description AS primary_metric_display_name,
            ev.metric_config.lower_is_better        AS lower_is_better,
            ev.metric_config.unit                   AS metric_unit,
            ev.models_count,
            ev.avg_score,
            ev.top_score
        FROM evals_view ev
        WHERE ev.benchmark_id IS NOT NULL
        ORDER BY ev.benchmark_id, ev.composite_slug
        """
    ).fetchall()
    cols = [d[0] for d in con.description]

    benchmarks: dict[str, dict] = {}
    for r in rows:
        rec = dict(zip(cols, r))
        bid = rec["benchmark_id"]
        entry = benchmarks.get(bid)
        if entry is None:
            entry = {
                "family_id":           rec["family_id"] or bid,
                "family_display_name": rec["family_display_name"]
                                          or rec["family_id"]
                                          or bid,
                "is_slice":            bool(rec["is_slice"]),
                "parent_benchmark_id": rec["parent_benchmark_id"],
                "appearances":         [],
            }
            benchmarks[bid] = entry
        entry["appearances"].append({
            "composite_slug":              rec["composite_slug"],
            "composite_display_name":      rec["composite_display_name"]
                                              or rec["composite_slug"],
            "evaluation_id":               rec["evaluation_id"],
            "primary_metric_id":           rec["primary_metric_id"],
            "primary_metric_display_name": rec["primary_metric_display_name"]
                                              or rec["primary_metric_id"],
            "lower_is_better":             None if rec["lower_is_better"] is None
                                                else bool(rec["lower_is_better"]),
            "metric_unit":                 rec["metric_unit"],
            "avg_score":                   rec["avg_score"],
            "top_score":                   rec["top_score"],
            "models_count":                int(rec["models_count"] or 0),
        })

    payload = {
        "generated_at":    snapshot_meta["snapshot_id"],
        "config_version":  CONFIG_VERSION,
        "benchmark_count": len(benchmarks),
        "benchmarks":      benchmarks,
    }
    path = out_dir / "benchmark_index.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


# ---------------------------------------------------------------------------
# peer-ranks.json
# ---------------------------------------------------------------------------


def write_peer_ranks(con, out_dir: Path, snapshot_meta: dict) -> Path:
    """Per-(eval, model) peer rank using each eval's primary metric.

    Frontend shape — `Record<eval_summary_id, Record<model_route_id,
    {position, total}>>`. Used by the model-detail page to attach a
    peer-rank badge to every benchmark cell. Comparison-index already
    carries per-metric ranks under `by_model`, but peer-ranks.json
    is a much smaller, primary-metric-only roll-up the model-detail
    grid loads as a single fetch (rather than scanning every cell's
    metric histograms).

    Pre-v3 the producer published this only at the dataset root and
    unversioned, so consumers that pinned a `warehouse/<snapshot>/`
    URL still hit the live `main`-branch ranks and could drift from
    the pinned views. Emitting it as a sidecar fixes that drift —
    peer-ranks.json now travels with the snapshot.
    """
    rows = con.execute(
        """
        SELECT
            erv.evaluation_id,
            erv.model_route_id,
            erv.position,
            erv.total
        FROM eval_results_view erv
        JOIN evals_view ev
          ON  ev.evaluation_id      = erv.evaluation_id
          AND ev.primary_metric_id  = erv.metric_id
        WHERE erv.score          IS NOT NULL
          AND erv.position       IS NOT NULL
          AND erv.total          IS NOT NULL
          AND erv.model_route_id IS NOT NULL
        """
    ).fetchall()

    by_eval: dict[str, dict[str, dict]] = defaultdict(dict)
    for evaluation_id, model_route_id, position, total in rows:
        # If a model appears more than once in the primary-metric set
        # for the same eval (shouldn't happen — eval_results_view is
        # collapsed to one row per (model, benchmark, metric)), keep
        # the better rank.
        existing = by_eval[evaluation_id].get(model_route_id)
        rank_entry = {"position": int(position), "total": int(total)}
        if existing is None or rank_entry["position"] < existing["position"]:
            by_eval[evaluation_id][model_route_id] = rank_entry

    payload = {
        "generated_at":   snapshot_meta["snapshot_id"],
        "config_version": CONFIG_VERSION,
        "ranks":          by_eval,
    }
    path = out_dir / "peer-ranks.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_default(obj):
    """Coerce DuckDB-native types JSON can't serialise."""
    import datetime
    import decimal

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Type not serialisable: {type(obj).__name__}")

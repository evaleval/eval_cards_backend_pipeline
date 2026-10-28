"""Compare two warehouse snapshots and report where the signals moved.

Reads the parquet tables and `headline.json` out of a "before" and an "after"
snapshot directory, writes a markdown report of the movement that a
generation-config overlay is allowed to cause (reproducibility, variant keys,
comparability, the generation payload on `eval_results_view`), and then asserts
the things that must not move. Every invariant renders as an explicit PASS or
FAIL line; the process exits non-zero when any of them fails.

The first invariants are about the comparison itself: the before snapshot has
to have been built with the overlay off, the after snapshot with it on and
filling something, and, when `--expect-fingerprint` is given, with exactly the
overlay that sha identifies. A pair that fails those is not the experiment the
rest of the report claims to describe. What the after snapshot says it filled
is then held to the tables: the fact rows in the source configs it names have
to show temperature or an output cap going from no value to one. A pair whose
metadata was edited over unchanged parquet fails there.

Adding fixes to an overlay that already ships is the same comparison one step
along, so `--before-overlay-fingerprint` switches the first invariant over:
the before snapshot then has to carry exactly the overlay that sha names, the
after snapshot a different one, and every fix the after snapshot has and the
before one does not has to have filled something. The source configs to hold
the report to default to the ones those new fixes report reaching, and it is
those new fixes whose claims the tables are held to: what the overlay already
filled in the before snapshot cannot move again.

Both snapshots must sit on the same upstream pins, otherwise the run aborts:
a pin difference makes every delta uninterpretable.

    uv run python scripts/datafix_signal_diff.py <before_dir> <after_dir> [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import duckdb

from eval_card_backend.datafix.table_io import read_json


# Source configs the overlay is allowed to touch. Everything else must come
# through byte-identical.
DEFAULT_TARGET_SOURCE_CONFIGS = ("vals-ai", "hfopenllm_v2")

MOVER_LIMIT = 25

# Fixed caveat: this counter is not the reproducibility rule.
MISSING_GEN_CONFIG_NOTE = (
    "`models_view.missing_generation_config_count` counts fact rows missing any of "
    "temperature, top_p or max_tokens, a three-field rule that is wider than the "
    "reproducibility rule (temperature + max_tokens). An overlay that fills only "
    "temperature and max_tokens leaves every hfopenllm_v2 row short of top_p, so "
    "this counter is expected not to improve for that source."
)

# Comparability groups span sources, so filling setup fields on a target fact can
# move these columns on facts from other sources in the same group. Nothing else
# on those rows may move.
SPILL_COLUMNS: dict[str, tuple[str, ...]] = {
    "fact_results": (
        "comparability_status",
        "has_variant_divergence",
        "variant_divergence_magnitude",
        "variant_divergence_threshold",
        "variant_threshold_basis",
        "variant_differing_fields",
        "has_cross_party_divergence",
        "cross_party_divergence_magnitude",
        "cross_party_divergence_threshold",
        "cross_party_threshold_basis",
        "cross_party_differing_fields",
        "cross_party_org_count",
    ),
    "eval_results_view": (
        "comparability_status",
        "has_variant_divergence",
        "has_cross_party_divergence",
        "evalcards_annotations",
    ),
    "evals_view": ("comparability_summary",),
    "models_view": ("comparability_summary",),
}
# Parts of eval_results_view.evalcards_annotations that comparability does not feed.
ERV_ANNOTATION_STABLE_PARTS = ("reproducibility_gap", "provenance")

# The two generation args the overlay writes, and nothing else. Every other
# setup field is out of its reach and may not move anywhere.
DIRECT_INPUT_FIELDS = ("temperature", "max_tokens")

# The fact columns carrying them: the values themselves and their flags.
DIRECT_INPUT_COLUMNS = (
    "temperature", "max_tokens", "has_temperature", "has_max_tokens",
)

# Read off the setup payload of one fact, so they move with it and only with it.
DERIVED_REPRO_COLUMNS = (
    "repro_missing_fields", "repro_required_count",
    "has_reproducibility_gap", "repro_populated_count",
)

# Compare one fact's setup against the rest of its comparability group, so
# they can move on a fact the overlay never touched, inside a group it did.
VARIANT_COLUMNS = ("variant_key",) + SPILL_COLUMNS["fact_results"]

# Columns compared member by member rather than whole, per table.
MEMBERWISE_COLUMNS: dict[str, tuple[str, ...]] = {
    "eval_results_view": ("generation_config",),
}

# Columns the overlay may move anywhere in the warehouse, per table. It writes
# nothing but two generation args, so movement is confined to those, the
# per-field flags and the reproducibility signal built from them, the variant
# key hashed from the payload, the comparability verdicts that compare
# variants, and the view-layer summaries and status bands rolled up from those.
# A table absent from this mapping may not move at all, and neither may a
# column absent from its entry or from `MEMBERWISE_COLUMNS`.
MOVABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "fact_results": DIRECT_INPUT_COLUMNS + DERIVED_REPRO_COLUMNS + VARIANT_COLUMNS,
    "eval_results_view": (
        "has_reproducibility_gap",
    ) + SPILL_COLUMNS["eval_results_view"],
    "evals_view": (
        "missing_generation_config_count", "reproducibility_summary",
    ) + SPILL_COLUMNS["evals_view"],
    "models_view": (
        "variant_count", "missing_generation_config_count",
        "reproducibility_status", "reproducibility_summary",
    ) + SPILL_COLUMNS["models_view"],
}

# Summaries averaged over unordered floats differ in the last digits run to run.
FLOAT_TOLERANCE = 1e-9

ERV_KEY = (
    "composite_slug",
    "model_key",
    "benchmark_id",
    "metric_id",
    "protocol_condition",
    "judge_condition",
    "split",
)


@dataclass(frozen=True)
class TableSpec:
    name: str
    # Empty key = no unique key on this table; compared as a row multiset.
    key: tuple[str, ...]
    # (attribution kind, column) used to decide whether a row belongs to one of
    # the target source configs. None = the table carries no usable attribution.
    scope: tuple[str, str] | None
    # Key columns that are null by design ("no condition", "no split"). Joins
    # here are null-safe, so these lose no rows; every other key column must be
    # populated on both sides.
    nullable: tuple[str, ...] = ()


TABLES: tuple[TableSpec, ...] = (
    TableSpec("fact_results", ("fact_id",), ("source_config", "source_config")),
    TableSpec("eval_results_view", ERV_KEY, ("composite", "composite_slug"),
              ("protocol_condition", "judge_condition", "split")),
    TableSpec("evals_view", ("evaluation_id",), ("composite", "composite_slug")),
    TableSpec("models_view", ("model_key",), ("model", "model_key")),
    TableSpec("merged_evals_view", ("evaluation_id",), ("benchmark", "benchmark_id")),
    TableSpec("models", ("model_key",), ("model", "model_key")),
    TableSpec("benchmarks", ("composite_slug", "benchmark_id"), ("composite", "composite_slug")),
    TableSpec("composites", ("composite_slug",), ("composite", "composite_slug")),
    TableSpec("canonical_metrics", ("id",), ("metric", "id")),
    TableSpec("families", ("family_id",), None),
    TableSpec("collection_trajectories", (), None),
)

# Columns whose value is the snapshot's own identity, never a signal.
IGNORED_COLUMNS = frozenset({"snapshot_id"})

FACT_IDENTITY_COLUMNS = (
    "evaluation_id", "result_idx", "evaluation_result_id",
    "model_raw", "model_id", "model_leaf_id", "model_aggregation_key", "model_key",
    "inference_platform", "resolution_source", "resolution_granularity",
    "benchmark_raw", "benchmark_id", "benchmark_key", "slice_key", "slice_name",
    "aggregate_level", "benchmark_subset", "is_part", "observation_role", "split",
    "metric_raw", "metric_id", "metric_key", "metric_key_effective",
    "metric_base_key", "metric_qualifier", "metric_source_label",
    "org_raw", "org_id", "org_display", "harness_raw", "harness_id",
    "parent_benchmark_id", "parent_model_id", "root_model_id", "model_family_id",
    "lineage_origin_model_id", "composite_slug", "composite_display_name",
    "source_config", "benchmark_card_id", "collection_id", "protocol_condition",
    "judge_condition", "split_source", "source_record_path",
    "model_resolution_strategy", "benchmark_resolution_strategy",
    "metric_resolution_strategy", "org_resolution_strategy",
    "harness_resolution_strategy",
)

FACT_SCORE_COLUMNS = (
    "score", "score_se", "score_sd", "score_ci_lower", "score_ci_upper",
    "score_ci_level", "n_samples", "scale_conversion", "score_canonical",
    "score_se_canonical", "score_sd_canonical", "score_ci_lower_canonical",
    "score_ci_upper_canonical", "score_scale_anomaly",
)

ERV_IDENTITY_COLUMNS = (
    "metric_summary_id", "composite_display_name", "family_id",
    "family_display_name", "is_slice", "parent_benchmark_id",
    "parent_benchmark_display_name", "metric_base_id", "metric_qualifier",
    "model_id", "model_route_id", "model_info", "metric_display_name",
    "metric_unit", "lower_is_better", "derived_tags", "metric_id_effective",
    "coverage_cell", "reporting_orgs", "evaluator_display_name",
    "is_verified_evaluator", "evaluator_relationships", "has_first_party",
    "has_third_party", "collection_id",
)

ERV_SCORE_COLUMNS = (
    "min_score", "max_score", "score_normalized", "score", "score_published",
    "score_details", "score_canonical", "score_se_canonical", "score_sd_canonical",
    "score_ci_lower_canonical", "score_ci_upper_canonical", "fact_row_count",
    "position", "total", "percentile", "scores_by_organization",
    "aggregate_components", "value_level", "value_aggregation",
    "children_present", "children_expected",
)

# `eval_results_view` does not emit the representative fact id, so the
# representative pick is checked through the columns it sources.
ERV_REPRESENTATIVE_COLUMNS = (
    "eee_record_url", "source_record_url", "evaluation_timestamp",
    "retrieved_timestamp", "benchmark_updated", "source_metadata", "source_data",
    "eval_library", "metric_source_label", "instance_file_path",
    "instance_file_format", "instance_rows", "instance_file_url",
    "is_summary_score", "summary_score_for",
)

# The rep pick orders on these fact columns inside a triple.
FACT_REPRESENTATIVE_ORDER_COLUMNS = (
    "score", "evaluator_relationship", "evaluation_id", "fact_id",
    "aggregate_level", "observation_role", "is_part", "metric_key_effective",
)

TRIPLE_KEY = (
    "composite_slug", "model_aggregation_key", "benchmark_key", "metric_key",
    "protocol_condition", "judge_condition", "split",
)

GENERATION_FIELDS = (
    ("temperature", "generation_config.generation_args.temperature"),
    ("top_p", "generation_config.generation_args.top_p"),
    ("top_k", "generation_config.generation_args.top_k"),
    ("max_tokens", "generation_config.generation_args.max_tokens"),
    ("reasoning", "generation_config.generation_args.reasoning"),
    ("num_few_shot", "generation_config.num_few_shot"),
    ("prompt_template", "generation_config.prompt_template"),
    ("additional_details", "generation_config.additional_details"),
)

# headline.json blocks that the overlay may legitimately move.
HEADLINE_MOVING_BLOCKS = ("reproducibility", "comparability")

# Fewer fact rows can move than a fix reports results patched, and that is
# not a disagreement.
FACT_FILL_NOTE = (
    "A fix counts the raw results it wrote into; the tables count fact rows. "
    "Several raw results can collapse onto one fact id, and Stage E drops "
    "results it cannot score, so observed fact movement is at most what a fix "
    "reports and is normally less."
)

# Which fact a view row was built from is not emitted, so a view row counts as
# reachable when any fact sharing these keys was filled.
ATTRIBUTION_NOTE = (
    "Only `fact_results` carries the fact id, so reachability is attributed "
    "per table through the keys the tables do share: composite, model and "
    "benchmark for `eval_results_view`, composite for `evals_view`, model for "
    "`models_view`. Those are coarser than each table's own key, so a row can "
    "count as reachable through a sibling row's fact."
)


@dataclass
class Invariant:
    name: str
    ok: bool
    detail: str


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _col(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _delta(before, after) -> str:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if isinstance(before, bool) or isinstance(after, bool):
            return "changed" if before != after else "0"
        diff = after - before
        if isinstance(before, int) and isinstance(after, int):
            return f"{diff:+d}"
        return f"{diff:+.6g}"
    if before == after:
        return "0"
    return "changed"


def _md_table(headers: list[str], rows: list[list]) -> list[str]:
    if not rows:
        return ["(no rows)", ""]
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    out.append("")
    return out


def _flatten(obj, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(obj, dict):
        for key in sorted(obj):
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(obj[key], child))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            flat.update(_flatten(item, f"{prefix}[{idx}]"))
    else:
        flat[prefix] = obj
    return flat


def _load_json(path: Path) -> dict:
    return read_json(path)


def _pins(meta: dict) -> dict[str, str | None]:
    pins: dict[str, str | None] = {}
    upstream = meta.get("upstream_pins") or {}
    for name in sorted(upstream):
        entry = upstream[name]
        pins[name] = entry.get("sha") if isinstance(entry, dict) else None
    for name, key in (
        ("eee_revision", "eee_revision"),
        ("registry_revision", "registry_revision"),
        ("cards_revision", "cards_revision"),
    ):
        if key in meta:
            pins[name] = meta[key]
    return pins


def _register(con, alias: str, snapshot: Path, tables: list[str]) -> None:
    for table in tables:
        path = snapshot / f"{table}.parquet"
        con.execute(
            f"CREATE VIEW {alias}_{table} AS "
            f"SELECT * FROM read_parquet({_lit(str(path))})"
        )


def _columns(con, view: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()]


def _common_columns(con, table: str) -> tuple[list[str], list[str], list[str]]:
    before = _columns(con, f"b_{table}")
    after = _columns(con, f"a_{table}")
    common = [c for c in before if c in set(after)]
    return common, [c for c in before if c not in set(after)], [c for c in after if c not in set(before)]


def _join_on(key: tuple[str, ...]) -> str:
    return " AND ".join(f"b.{_col(k)} IS NOT DISTINCT FROM a.{_col(k)}" for k in key)


def _close(before: str, after: str) -> str:
    return (
        f"(({before} IS NULL AND {after} IS NULL) OR "
        f"abs({before} - {after}) <= {FLOAT_TOLERANCE} * "
        f"greatest(1.0, abs({before}), abs({after})))"
    )


def _differs(column: str) -> str:
    b, a = f"b.{_col(column)}", f"a.{_col(column)}"
    if column == "score_summary":
        floats = " AND ".join(_close(f"{b}.{f}", f"{a}.{f}") for f in ("min", "max", "average"))
        return (
            f"(({b} IS NULL) <> ({a} IS NULL) OR "
            f"{b}.count IS DISTINCT FROM {a}.count OR NOT ({floats}))"
        )
    return f"CAST({b} AS VARCHAR) IS DISTINCT FROM CAST({a} AS VARCHAR)"


def _column_movement(
    con,
    table: str,
    key: tuple[str, ...],
    columns: list[str],
    where: str | None = None,
) -> list[tuple[str, int]]:
    """Rows whose value changed, per column, over key-matched rows."""
    if not columns:
        return []
    selects = ", ".join(
        f"SUM(CASE WHEN {_differs(c)} THEN 1 ELSE 0 END) AS {_col(c)}"
        for c in columns
    )
    clause = f"WHERE {where}" if where else ""
    row = con.execute(
        f"SELECT {selects} FROM b_{table} b "
        f"JOIN a_{table} a ON {_join_on(key)} {clause}"
    ).fetchone()
    return [(c, int(n or 0)) for c, n in zip(columns, row) if int(n or 0) > 0]


def _fmt_moved(moved: list[tuple[str, int]]) -> str:
    return ", ".join(
        f"{column} ({rows} row{'' if rows == 1 else 's'})" for column, rows in moved
    )


def _unmatched(con, table: str, key: tuple[str, ...]) -> tuple[int, int]:
    only_before = con.execute(
        f"SELECT COUNT(*) FROM b_{table} b "
        f"LEFT JOIN a_{table} a ON {_join_on(key)} WHERE a.{_col(key[0])} IS NULL"
    ).fetchone()[0]
    only_after = con.execute(
        f"SELECT COUNT(*) FROM a_{table} a "
        f"LEFT JOIN b_{table} b ON {_join_on(key)} WHERE b.{_col(key[0])} IS NULL"
    ).fetchone()[0]
    return int(only_before), int(only_after)


def _row_multiset_diff(con, table: str, columns: list[str]) -> int:
    """Rows present on one side only, for a table with no unique key."""
    packed = "struct_pack(" + ", ".join(f"{_col(c)} := {_col(c)}" for c in columns) + ")"
    repr_sql = f"CAST({packed} AS VARCHAR)"
    sql = (
        f"SELECT COUNT(*) FROM ("
        f"  (SELECT {repr_sql} AS r FROM b_{table} EXCEPT ALL SELECT {repr_sql} AS r FROM a_{table})"
        f"  UNION ALL"
        f"  (SELECT {repr_sql} AS r FROM a_{table} EXCEPT ALL SELECT {repr_sql} AS r FROM b_{table})"
        f")"
    )
    return int(con.execute(sql).fetchone()[0])


def _scope_predicate(spec: TableSpec, alias: str) -> str | None:
    """SQL that is true for rows attributable to one of the target sources."""
    if spec.scope is None:
        return None
    kind, column = spec.scope
    ref = f"{alias}.{_col(column)}"
    if kind == "source_config":
        return f"COALESCE({ref}, '') IN (SELECT source_config FROM target_source_configs)"
    if kind == "composite":
        return f"COALESCE({ref}, '') IN (SELECT composite_slug FROM target_composites)"
    if kind == "model":
        return f"{ref} IN (SELECT model_key FROM target_model_keys)"
    if kind == "benchmark":
        return f"{ref} IN (SELECT benchmark_key FROM target_benchmark_keys)"
    if kind == "metric":
        return (
            f"({ref} IN (SELECT metric_key FROM target_metric_keys)"
            f" OR {ref} IN (SELECT metric_base_key FROM target_metric_base_keys))"
        )
    raise ValueError(f"unknown scope kind {kind!r}")


def _attributed_predicate(spec: TableSpec, alias: str, source: str) -> str | None:
    """SQL that is true for rows a fact in `source` could have fed.

    Only `fact_results` carries the fact id, so every other table is
    attributed through the keys it does share with a fact row. See
    `ATTRIBUTION_NOTE`.
    """
    if spec.name == "fact_results":
        return f"{alias}.fact_id IN (SELECT fact_id FROM {source})"
    if spec.name == "eval_results_view":
        return (
            f"({alias}.composite_slug, {alias}.model_key, {alias}.benchmark_id) IN "
            f"(SELECT composite_slug, model_aggregation_key, benchmark_id FROM {source})"
        )
    if spec.name == "evals_view":
        return f"{alias}.composite_slug IN (SELECT composite_slug FROM {source})"
    if spec.name == "models_view":
        return f"{alias}.model_key IN (SELECT model_aggregation_key FROM {source})"
    return None


def _spill_predicate(spec: TableSpec, alias: str) -> str | None:
    """SQL that is true for rows fed by another source's facts that share a
    comparability group with a target fact."""
    return _attributed_predicate(spec, alias, "spill_facts")


def _reached_predicate(spec: TableSpec, alias: str) -> str | None:
    """SQL that is true for rows the overlay's fills could have reached."""
    return _attributed_predicate(spec, alias, "reached_facts")


def _build_target_views(con, targets: tuple[str, ...]) -> None:
    values = ", ".join(f"({_lit(t)})" for t in targets)
    con.execute(f"CREATE VIEW target_source_configs(source_config) AS VALUES {values}")
    # Everything the target source configs reach, taken from both sides so a row
    # that only exists on one of them still counts as in scope.
    for view_name, alias, column in (
        ("target_composites", "composite_slug", "composite_slug"),
        ("target_model_keys", "model_key", "model_aggregation_key"),
        ("target_benchmark_keys", "benchmark_key", "benchmark_key"),
        ("target_metric_keys", "metric_key", "metric_key"),
        ("target_metric_base_keys", "metric_base_key", "metric_base_key"),
    ):
        con.execute(
            f"CREATE VIEW {view_name}({alias}) AS "
            f"SELECT DISTINCT {_col(column)} FROM b_fact_results "
            f"WHERE {_col(column)} IS NOT NULL "
            f"  AND COALESCE(source_config, '') IN (SELECT source_config FROM target_source_configs) "
            f"UNION "
            f"SELECT DISTINCT {_col(column)} FROM a_fact_results "
            f"WHERE {_col(column)} IS NOT NULL "
            f"  AND COALESCE(source_config, '') IN (SELECT source_config FROM target_source_configs)"
        )
    con.execute(
        "CREATE VIEW spill_facts AS "
        "SELECT fact_id, composite_slug, model_aggregation_key, benchmark_id FROM ("
        "  SELECT * FROM b_fact_results UNION ALL BY NAME SELECT * FROM a_fact_results"
        ") f "
        "WHERE COALESCE(f.source_config, '') NOT IN (SELECT source_config FROM target_source_configs) "
        "  AND f.comparability_group_id IN ("
        "    SELECT comparability_group_id FROM b_fact_results "
        "    WHERE COALESCE(source_config, '') IN (SELECT source_config FROM target_source_configs) "
        "    UNION "
        "    SELECT comparability_group_id FROM a_fact_results "
        "    WHERE COALESCE(source_config, '') IN (SELECT source_config FROM target_source_configs))"
    )
    # The facts the overlay actually filled, read off the tables rather than
    # off what the after snapshot says it did.
    fills = " OR ".join(
        f"(b.{_col(f)} IS NULL AND a.{_col(f)} IS NOT NULL)"
        for f in DIRECT_INPUT_FIELDS
    )
    con.execute(
        "CREATE VIEW patched_facts AS "
        "SELECT b.fact_id, b.source_config, b.comparability_group_id, "
        "       b.composite_slug, b.model_aggregation_key, b.benchmark_id "
        f"FROM b_fact_results b JOIN a_fact_results a USING (fact_id) WHERE {fills}"
    )
    # Those facts plus the rest of every comparability group one of them is
    # in: a fill changes what the others in the group are compared against.
    con.execute(
        "CREATE VIEW reached_facts AS "
        "SELECT DISTINCT f.fact_id, f.composite_slug, f.model_aggregation_key, "
        "       f.benchmark_id FROM ("
        "  SELECT * FROM b_fact_results UNION ALL BY NAME SELECT * FROM a_fact_results"
        ") f "
        "WHERE f.fact_id IN (SELECT fact_id FROM patched_facts) "
        "   OR (f.comparability_group_id IS NOT NULL "
        "       AND f.comparability_group_id IN ("
        "         SELECT comparability_group_id FROM patched_facts "
        "         WHERE comparability_group_id IS NOT NULL))"
    )


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------


def _section_reproducibility(con, before_headline, after_headline) -> list[str]:
    out = ["## 1. Reproducibility movement", ""]

    out += ["### 1.1 headline.json reproducibility, overall", ""]
    b_flat = _flatten(before_headline.get("reproducibility", {}).get("overall", {}))
    a_flat = _flatten(after_headline.get("reproducibility", {}).get("overall", {}))
    rows = [[k, b_flat.get(k), a_flat.get(k), _delta(b_flat.get(k), a_flat.get(k))]
            for k in sorted(set(b_flat) | set(a_flat))]
    out += _md_table(["field", "before", "after", "delta"], rows)

    out += ["### 1.2 headline.json reproducibility, by tag", ""]
    b_tags = before_headline.get("reproducibility", {}).get("by_tag", {}) or {}
    a_tags = after_headline.get("reproducibility", {}).get("by_tag", {}) or {}
    rows = []
    for tag in sorted(set(b_tags) | set(a_tags)):
        b_tag, a_tag = b_tags.get(tag, {}), a_tags.get(tag, {})
        b_gap = b_tag.get("triples_with_reproducibility_gap")
        a_gap = a_tag.get("triples_with_reproducibility_gap")
        b_temp = (b_tag.get("per_field_missingness", {}).get("temperature", {})
                  .get("missing_count"))
        a_temp = (a_tag.get("per_field_missingness", {}).get("temperature", {})
                  .get("missing_count"))
        b_mt = (b_tag.get("per_field_missingness", {}).get("max_tokens", {})
                .get("missing_count"))
        a_mt = (a_tag.get("per_field_missingness", {}).get("max_tokens", {})
                .get("missing_count"))
        if (b_gap, b_temp, b_mt) == (a_gap, a_temp, a_mt):
            continue
        rows.append([tag, b_tag.get("total_triples"), b_gap, a_gap, _delta(b_gap, a_gap),
                     _delta(b_temp, a_temp), _delta(b_mt, a_mt)])
    if not rows:
        out += ["No tag moved on gap count or per-field missingness.", ""]
    else:
        out += _md_table(
            ["tag", "triples", "gap before", "gap after", "gap delta",
             "missing temperature delta", "missing max_tokens delta"],
            rows,
        )

    out += ["### 1.3 fact-level gaps by source_config", ""]
    rows = con.execute(
        """
        WITH b AS (
            SELECT source_config,
                   COUNT(*)                                            AS rows_n,
                   SUM(CASE WHEN has_reproducibility_gap THEN 1 ELSE 0 END) AS gap_n,
                   SUM(CASE WHEN has_temperature THEN 0 ELSE 1 END)     AS miss_temp,
                   SUM(CASE WHEN has_max_tokens  THEN 0 ELSE 1 END)     AS miss_mt
            FROM b_fact_results GROUP BY 1
        ),
        a AS (
            SELECT source_config,
                   COUNT(*)                                            AS rows_n,
                   SUM(CASE WHEN has_reproducibility_gap THEN 1 ELSE 0 END) AS gap_n,
                   SUM(CASE WHEN has_temperature THEN 0 ELSE 1 END)     AS miss_temp,
                   SUM(CASE WHEN has_max_tokens  THEN 0 ELSE 1 END)     AS miss_mt
            FROM a_fact_results GROUP BY 1
        )
        SELECT COALESCE(b.source_config, a.source_config) AS source_config,
               b.rows_n, a.rows_n, b.gap_n, a.gap_n,
               b.miss_temp, a.miss_temp, b.miss_mt, a.miss_mt
        FROM b FULL OUTER JOIN a ON b.source_config IS NOT DISTINCT FROM a.source_config
        WHERE b.rows_n IS DISTINCT FROM a.rows_n
           OR b.gap_n IS DISTINCT FROM a.gap_n
           OR b.miss_temp IS DISTINCT FROM a.miss_temp
           OR b.miss_mt IS DISTINCT FROM a.miss_mt
           OR COALESCE(b.source_config, a.source_config)
              IN (SELECT source_config FROM target_source_configs)
        ORDER BY 1
        """
    ).fetchall()
    out += _md_table(
        ["source_config", "rows before", "rows after", "gap before", "gap after",
         "missing temperature before", "after", "missing max_tokens before", "after"],
        [list(r) for r in rows],
    )

    out += ["### 1.4 fact-level gaps by benchmark, top movers", ""]
    rows = con.execute(
        f"""
        WITH b AS (
            SELECT source_config, benchmark_key,
                   SUM(CASE WHEN has_reproducibility_gap THEN 1 ELSE 0 END) AS gap_n,
                   COUNT(*) AS rows_n
            FROM b_fact_results GROUP BY 1, 2
        ),
        a AS (
            SELECT source_config, benchmark_key,
                   SUM(CASE WHEN has_reproducibility_gap THEN 1 ELSE 0 END) AS gap_n,
                   COUNT(*) AS rows_n
            FROM a_fact_results GROUP BY 1, 2
        ),
        j AS (
            SELECT COALESCE(b.source_config, a.source_config)   AS source_config,
                   COALESCE(b.benchmark_key, a.benchmark_key)   AS benchmark_key,
                   COALESCE(b.rows_n, 0) AS rows_before, COALESCE(a.rows_n, 0) AS rows_after,
                   COALESCE(b.gap_n, 0)  AS gap_before,  COALESCE(a.gap_n, 0)  AS gap_after
            FROM b FULL OUTER JOIN a
              ON b.source_config IS NOT DISTINCT FROM a.source_config
             AND b.benchmark_key IS NOT DISTINCT FROM a.benchmark_key
        )
        SELECT source_config, benchmark_key, rows_before, rows_after,
               gap_before, gap_after, gap_after - gap_before AS delta
        FROM j
        WHERE gap_before <> gap_after OR rows_before <> rows_after
        ORDER BY abs(gap_after - gap_before) DESC, source_config, benchmark_key
        LIMIT {MOVER_LIMIT}
        """
    ).fetchall()
    moved_total = con.execute(
        """
        WITH b AS (SELECT source_config, benchmark_key,
                          SUM(CASE WHEN has_reproducibility_gap THEN 1 ELSE 0 END) AS g
                   FROM b_fact_results GROUP BY 1, 2),
             a AS (SELECT source_config, benchmark_key,
                          SUM(CASE WHEN has_reproducibility_gap THEN 1 ELSE 0 END) AS g
                   FROM a_fact_results GROUP BY 1, 2)
        SELECT COUNT(*) FROM b FULL OUTER JOIN a
          ON b.source_config IS NOT DISTINCT FROM a.source_config
         AND b.benchmark_key IS NOT DISTINCT FROM a.benchmark_key
        WHERE COALESCE(b.g, 0) IS DISTINCT FROM COALESCE(a.g, 0)
        """
    ).fetchone()[0]
    out += [f"{moved_total} (source_config, benchmark) pairs moved on gap count; "
            f"top {MOVER_LIMIT} by absolute delta:", ""]
    out += _md_table(
        ["source_config", "benchmark_key", "rows before", "rows after",
         "gap before", "gap after", "delta"],
        [list(r) for r in rows],
    )

    out += ["### 1.5 evals_view reproducibility summaries", ""]
    row = con.execute(
        """
        SELECT
            SUM(b.reproducibility_summary.results_total),
            SUM(a.reproducibility_summary.results_total),
            SUM(b.reproducibility_summary.has_reproducibility_gap_count),
            SUM(a.reproducibility_summary.has_reproducibility_gap_count),
            SUM(b.missing_generation_config_count),
            SUM(a.missing_generation_config_count),
            COUNT(*) FILTER (
                WHERE b.reproducibility_summary.has_reproducibility_gap_count
                   IS DISTINCT FROM a.reproducibility_summary.has_reproducibility_gap_count)
        FROM b_evals_view b JOIN a_evals_view a USING (evaluation_id)
        """
    ).fetchone()
    out += _md_table(
        ["measure", "before", "after", "delta"],
        [
            ["results_total (sum)", row[0], row[1], _delta(row[0], row[1])],
            ["has_reproducibility_gap_count (sum)", row[2], row[3], _delta(row[2], row[3])],
            ["missing_generation_config_count (sum)", row[4], row[5], _delta(row[4], row[5])],
            ["evals whose gap count moved", row[6], row[6], "n/a"],
        ],
    )
    rows = con.execute(
        f"""
        SELECT b.evaluation_id, b.composite_slug, b.benchmark_id,
               b.reproducibility_summary.has_reproducibility_gap_count AS gap_before,
               a.reproducibility_summary.has_reproducibility_gap_count AS gap_after,
               b.reproducibility_summary.populated_ratio_avg           AS ratio_before,
               a.reproducibility_summary.populated_ratio_avg           AS ratio_after
        FROM b_evals_view b JOIN a_evals_view a USING (evaluation_id)
        WHERE b.reproducibility_summary.has_reproducibility_gap_count
           IS DISTINCT FROM a.reproducibility_summary.has_reproducibility_gap_count
        ORDER BY abs(COALESCE(a.reproducibility_summary.has_reproducibility_gap_count, 0)
                   - COALESCE(b.reproducibility_summary.has_reproducibility_gap_count, 0)) DESC,
                 b.evaluation_id
        LIMIT {MOVER_LIMIT}
        """
    ).fetchall()
    out += _md_table(
        ["evaluation_id", "composite_slug", "benchmark_id", "gap before",
         "gap after", "populated ratio before", "populated ratio after"],
        [list(r) for r in rows],
    )

    out += ["### 1.6 models_view reproducibility summaries and status bands", ""]
    row = con.execute(
        """
        SELECT
            SUM(b.reproducibility_summary.results_total),
            SUM(a.reproducibility_summary.results_total),
            SUM(b.reproducibility_summary.has_reproducibility_gap_count),
            SUM(a.reproducibility_summary.has_reproducibility_gap_count),
            SUM(b.missing_generation_config_count),
            SUM(a.missing_generation_config_count),
            COUNT(*) FILTER (WHERE b.reproducibility_status
                               IS DISTINCT FROM a.reproducibility_status)
        FROM b_models_view b JOIN a_models_view a USING (model_key)
        """
    ).fetchone()
    out += _md_table(
        ["measure", "before", "after", "delta"],
        [
            ["results_total (sum)", row[0], row[1], _delta(row[0], row[1])],
            ["has_reproducibility_gap_count (sum)", row[2], row[3], _delta(row[2], row[3])],
            ["missing_generation_config_count (sum)", row[4], row[5], _delta(row[4], row[5])],
            ["models whose status band moved", row[6], row[6], "n/a"],
        ],
    )
    out += ["Status-band transition matrix (models_view.reproducibility_status):", ""]
    rows = con.execute(
        """
        SELECT COALESCE(b.reproducibility_status, '(null)') AS before_band,
               COALESCE(a.reproducibility_status, '(null)') AS after_band,
               COUNT(*) AS models
        FROM b_models_view b JOIN a_models_view a USING (model_key)
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()
    out += _md_table(["before", "after", "models"], [list(r) for r in rows])
    out += [MISSING_GEN_CONFIG_NOTE, ""]

    out += ["### 1.7 fact rows the overlay filled, by source_config", ""]
    rows = con.execute(
        "SELECT COALESCE(source_config, '(null)'), COUNT(*) "
        "FROM patched_facts GROUP BY 1 ORDER BY 2 DESC, 1"
    ).fetchall()
    out += _md_table(["source_config", "fact rows filled"], [list(r) for r in rows])
    out += [FACT_FILL_NOTE, ""]

    out += ["### 1.8 fact_results column movement inside the target sources", ""]
    columns = [c for c in _common_columns(con, "fact_results")[0]
               if c not in IGNORED_COLUMNS and c != "fact_id"]
    moved = _column_movement(
        con, "fact_results", ("fact_id",), columns,
        where=_scope_predicate(TABLES[0], "b"),
    )
    if not moved:
        out += ["No fact_results column moved for the target sources.", ""]
    else:
        out += _md_table(["column", "rows changed"], [list(m) for m in moved])
    return out


def _section_variants(con) -> list[str]:
    out = ["## 2. Variant keys", "", "### 2.1 variant_key changes per source_config", ""]
    rows = con.execute(
        """
        SELECT b.source_config,
               COUNT(*)                                                   AS rows_n,
               SUM(CASE WHEN b.variant_key IS DISTINCT FROM a.variant_key
                        THEN 1 ELSE 0 END)                                AS changed,
               COUNT(DISTINCT b.variant_key)                              AS distinct_before,
               COUNT(DISTINCT a.variant_key)                              AS distinct_after
        FROM b_fact_results b JOIN a_fact_results a USING (fact_id)
        GROUP BY 1
        HAVING SUM(CASE WHEN b.variant_key IS DISTINCT FROM a.variant_key THEN 1 ELSE 0 END) > 0
            OR COALESCE(b.source_config, '') IN (SELECT source_config FROM target_source_configs)
        ORDER BY 1
        """
    ).fetchall()
    out += _md_table(
        ["source_config", "fact rows", "variant_key changed",
         "distinct variant_key before", "after"],
        [list(r) for r in rows],
    )

    out += ["### 2.2 variant counts per comparability group", ""]
    rows = con.execute(
        """
        WITH b AS (
            SELECT comparability_group_id AS g, COUNT(DISTINCT variant_key) AS n
            FROM b_fact_results WHERE comparability_group_id IS NOT NULL GROUP BY 1
        ),
        a AS (
            SELECT comparability_group_id AS g, COUNT(DISTINCT variant_key) AS n
            FROM a_fact_results WHERE comparability_group_id IS NOT NULL GROUP BY 1
        )
        SELECT COALESCE(b.n, 0) AS before_n, COALESCE(a.n, 0) AS after_n, COUNT(*) AS groups
        FROM b FULL OUTER JOIN a ON b.g = a.g
        WHERE COALESCE(b.n, 0) IS DISTINCT FROM COALESCE(a.n, 0)
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()
    if not rows:
        out += ["No comparability group changed its distinct variant_key count.", ""]
    else:
        out += _md_table(["variants before", "variants after", "groups"],
                         [list(r) for r in rows])
    row = con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM (
                SELECT comparability_group_id FROM b_fact_results
                WHERE comparability_group_id IS NOT NULL
                GROUP BY 1 HAVING COUNT(DISTINCT variant_key) > 1)),
            (SELECT COUNT(*) FROM (
                SELECT comparability_group_id FROM a_fact_results
                WHERE comparability_group_id IS NOT NULL
                GROUP BY 1 HAVING COUNT(DISTINCT variant_key) > 1))
        """
    ).fetchone()
    out += _md_table(
        ["measure", "before", "after", "delta"],
        [["groups holding more than one variant_key", row[0], row[1], _delta(row[0], row[1])]],
    )

    out += ["### 2.3 models_view.variant_count", ""]
    row = con.execute(
        """
        SELECT SUM(b.variant_count), SUM(a.variant_count),
               COUNT(*) FILTER (WHERE b.variant_count IS DISTINCT FROM a.variant_count)
        FROM b_models_view b JOIN a_models_view a USING (model_key)
        """
    ).fetchone()
    out += _md_table(
        ["measure", "before", "after", "delta"],
        [["variant_count (sum)", row[0], row[1], _delta(row[0], row[1])],
         ["models whose variant_count moved", row[2], row[2], "n/a"]],
    )
    return out


def _section_comparability(con, before_headline, after_headline) -> list[str]:
    out = ["## 3. Comparability and divergence", "",
           "### 3.1 fact-level status and divergence flags", ""]
    rows = con.execute(
        """
        SELECT COALESCE(b.comparability_status, '(null)') AS before_status,
               COALESCE(a.comparability_status, '(null)') AS after_status,
               COUNT(*) AS facts
        FROM b_fact_results b JOIN a_fact_results a USING (fact_id)
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).fetchall()
    out += ["comparability_status transitions:", ""]
    out += _md_table(["before", "after", "fact rows"], [list(r) for r in rows])

    rows = con.execute(
        """
        SELECT 'has_variant_divergence' AS flag,
               COALESCE(CAST(b.has_variant_divergence AS VARCHAR), '(null)') AS before_v,
               COALESCE(CAST(a.has_variant_divergence AS VARCHAR), '(null)') AS after_v,
               COUNT(*) AS facts
        FROM b_fact_results b JOIN a_fact_results a USING (fact_id)
        WHERE b.has_variant_divergence IS DISTINCT FROM a.has_variant_divergence
        GROUP BY 1, 2, 3
        UNION ALL
        SELECT 'has_cross_party_divergence',
               COALESCE(CAST(b.has_cross_party_divergence AS VARCHAR), '(null)'),
               COALESCE(CAST(a.has_cross_party_divergence AS VARCHAR), '(null)'),
               COUNT(*)
        FROM b_fact_results b JOIN a_fact_results a USING (fact_id)
        WHERE b.has_cross_party_divergence IS DISTINCT FROM a.has_cross_party_divergence
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    ).fetchall()
    out += ["Divergence flag flips:", ""]
    out += _md_table(["flag", "before", "after", "fact rows"], [list(r) for r in rows])

    out += ["### 3.2 differing setup fields", ""]
    for column, label in (("variant_differing_fields", "variant"),
                          ("cross_party_differing_fields", "cross-party")):
        changed = con.execute(
            f"""
            SELECT COUNT(*) FROM b_fact_results b JOIN a_fact_results a USING (fact_id)
            WHERE list_sort(list_transform(COALESCE(b.{column}, []), x -> x.field))
               IS DISTINCT FROM
                  list_sort(list_transform(COALESCE(a.{column}, []), x -> x.field))
            """
        ).fetchone()[0]
        rows = con.execute(
            f"""
            WITH j AS (
                SELECT b.source_config,
                       list_sort(list_transform(COALESCE(b.{column}, []), x -> x.field)) AS fb,
                       list_sort(list_transform(COALESCE(a.{column}, []), x -> x.field)) AS fa
                FROM b_fact_results b JOIN a_fact_results a USING (fact_id)
                WHERE list_sort(list_transform(COALESCE(b.{column}, []), x -> x.field))
                   IS DISTINCT FROM
                      list_sort(list_transform(COALESCE(a.{column}, []), x -> x.field))
            ),
            added AS (
                SELECT source_config, unnest(list_filter(fa, x -> NOT list_contains(fb, x))) AS field
                FROM j
            ),
            removed AS (
                SELECT source_config, unnest(list_filter(fb, x -> NOT list_contains(fa, x))) AS field
                FROM j
            )
            SELECT COALESCE(ad.source_config, rm.source_config) AS source_config,
                   COALESCE(ad.field, rm.field)                 AS field,
                   COALESCE(ad.n, 0) AS added_on, COALESCE(rm.n, 0) AS removed_from
            FROM (SELECT source_config, field, COUNT(*) AS n FROM added GROUP BY 1, 2) ad
            FULL OUTER JOIN
                 (SELECT source_config, field, COUNT(*) AS n FROM removed GROUP BY 1, 2) rm
              ON ad.source_config IS NOT DISTINCT FROM rm.source_config
             AND ad.field         IS NOT DISTINCT FROM rm.field
            ORDER BY 1, 2
            """
        ).fetchall()
        out += [f"{changed} fact rows changed their {label} differing-setup field set.", ""]
        out += _md_table(
            ["source_config", "field", "rows where it appeared", "rows where it went away"],
            [list(r) for r in rows],
        )

    out += ["### 3.3 headline.json comparability block", ""]
    b_flat = _flatten(before_headline.get("comparability", {}).get("overall", {}))
    a_flat = _flatten(after_headline.get("comparability", {}).get("overall", {}))
    out += _md_table(
        ["field", "before", "after", "delta"],
        [[k, b_flat.get(k), a_flat.get(k), _delta(b_flat.get(k), a_flat.get(k))]
         for k in sorted(set(b_flat) | set(a_flat))],
    )

    out += ["### 3.4 evals_view and models_view comparability summaries", ""]
    rows = []
    for table, key in (("evals_view", "evaluation_id"), ("models_view", "model_key")):
        row = con.execute(
            f"""
            SELECT SUM(b.comparability_summary.variant_divergent_count),
                   SUM(a.comparability_summary.variant_divergent_count),
                   SUM(b.comparability_summary.cross_party_divergent_count),
                   SUM(a.comparability_summary.cross_party_divergent_count),
                   SUM(b.comparability_summary.groups_with_variant_check),
                   SUM(a.comparability_summary.groups_with_variant_check),
                   COUNT(*) FILTER (WHERE CAST(b.comparability_summary AS VARCHAR)
                                       IS DISTINCT FROM CAST(a.comparability_summary AS VARCHAR))
            FROM b_{table} b JOIN a_{table} a USING ({key})
            """
        ).fetchone()
        rows += [
            [table, "variant_divergent_count (sum)", row[0], row[1], _delta(row[0], row[1])],
            [table, "cross_party_divergent_count (sum)", row[2], row[3], _delta(row[2], row[3])],
            [table, "groups_with_variant_check (sum)", row[4], row[5], _delta(row[4], row[5])],
            [table, "rows whose summary moved", row[6], row[6], "n/a"],
        ]
    out += _md_table(["table", "measure", "before", "after", "delta"], rows)
    return out


def _section_generation_payload(con) -> list[str]:
    out = ["## 4. Generation payload on eval_results_view", ""]
    key_join = _join_on(ERV_KEY)
    rows = []
    for label, path in GENERATION_FIELDS:
        expr_b = "b." + ".".join(_col(p) for p in path.split("."))
        expr_a = "a." + ".".join(_col(p) for p in path.split("."))
        found = con.execute(
            f"""
            SELECT b.composite_slug,
                   SUM(CASE WHEN {expr_b} IS NULL AND {expr_a} IS NOT NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN {expr_b} IS NOT NULL AND {expr_a} IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN {expr_b} IS NOT NULL AND {expr_a} IS NOT NULL
                             AND CAST({expr_b} AS VARCHAR)
                                 IS DISTINCT FROM CAST({expr_a} AS VARCHAR)
                            THEN 1 ELSE 0 END)
            FROM b_eval_results_view b JOIN a_eval_results_view a ON {key_join}
            GROUP BY 1
            HAVING SUM(CASE WHEN CAST({expr_b} AS VARCHAR)
                             IS DISTINCT FROM CAST({expr_a} AS VARCHAR)
                            THEN 1 ELSE 0 END) > 0
            ORDER BY 1
            """
        ).fetchall()
        for composite, filled, cleared, rewritten in found:
            rows.append([label, composite, filled, cleared, rewritten])
    if not rows:
        out += ["No generation-config field moved on eval_results_view.", ""]
    else:
        out += _md_table(
            ["field", "composite_slug", "filled (null to value)",
             "cleared (value to null)", "rewritten"],
            rows,
        )

    out += ["### 4.1 eval_results_view column movement inside the target composites", ""]
    columns = [c for c in _common_columns(con, "eval_results_view")[0]
               if c not in IGNORED_COLUMNS and c not in ERV_KEY]
    moved = _column_movement(
        con, "eval_results_view", ERV_KEY, columns,
        where="COALESCE(b.composite_slug, '') IN (SELECT composite_slug FROM target_composites)",
    )
    if not moved:
        out += ["No eval_results_view column moved for the target composites.", ""]
    else:
        out += _md_table(["column", "rows changed"], [list(m) for m in moved])

    out += ["### 4.2 sample of moved rows", ""]
    rows = con.execute(
        f"""
        SELECT b.composite_slug, b.benchmark_id, b.model_key, b.metric_id,
               b.generation_config.generation_args.temperature,
               a.generation_config.generation_args.temperature,
               b.generation_config.generation_args.max_tokens,
               a.generation_config.generation_args.max_tokens
        FROM b_eval_results_view b JOIN a_eval_results_view a ON {key_join}
        WHERE CAST(b.generation_config AS VARCHAR)
           IS DISTINCT FROM CAST(a.generation_config AS VARCHAR)
        ORDER BY b.composite_slug, b.benchmark_id, b.model_key, b.metric_id,
                 COALESCE(b.protocol_condition, ''), COALESCE(b.judge_condition, ''),
                 COALESCE(b.split, '')
        LIMIT {MOVER_LIMIT}
        """
    ).fetchall()
    out += _md_table(
        ["composite_slug", "benchmark_id", "model_key", "metric_id",
         "temperature before", "after", "max_tokens before", "after"],
        [list(r) for r in rows],
    )
    return out


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------


def _new_fixes(before_meta: dict, after_meta: dict) -> dict[str, dict]:
    """Fixes the after snapshot's overlay has and the before one does not."""
    before_fixes = (before_meta.get("datafix") or {}).get("fixes") or {}
    after_fixes = (after_meta.get("datafix") or {}).get("fixes") or {}
    return {
        fix_id: block or {}
        for fix_id, block in after_fixes.items()
        if fix_id not in before_fixes
    }


def _new_fix_targets(before_meta: dict, after_meta: dict) -> tuple[str, ...]:
    """Source configs the new fixes report having patched."""
    configs: set[str] = set()
    for block in _new_fixes(before_meta, after_meta).values():
        configs.update(block.get("by_source_config") or {})
    return tuple(sorted(configs))


def _invariant_experiment(
    before_meta: dict,
    after_meta: dict,
    expect_fingerprint: str | None,
    before_overlay_fingerprint: str | None,
) -> list[Invariant]:
    """The pair really is the comparison the report claims to describe.

    Read off `snapshot_meta.datafix`, which every bake writes. Without
    `--before-overlay-fingerprint` the before side must carry no overlay and
    the after side one that filled something. With it, the before side must
    carry exactly that overlay, the after side a different one, and every
    fix the after side added must have filled something of its own.
    """
    before_block = before_meta.get("datafix") or {}
    after_block = after_meta.get("datafix") or {}
    fixes = after_block.get("fixes") or {}
    filled = {
        fix_id: (block or {}).get("results_patched") or 0
        for fix_id, block in fixes.items()
    }

    if before_overlay_fingerprint is None:
        raw_before = before_meta.get("datafix")
        before_off = raw_before is None or raw_before.get("enabled") is False
        invariants = [
            Invariant(
                "before snapshot built without the overlay",
                before_off,
                "no datafix block" if raw_before is None
                else f"enabled={raw_before.get('enabled')!r}",
            )
        ]
        patched = {fix_id: n for fix_id, n in filled.items() if n > 0}
        invariants.append(
            Invariant(
                "after snapshot built with the overlay, and it filled something",
                after_block.get("enabled") is True and bool(patched),
                f"enabled={after_block.get('enabled')!r}, results_patched "
                + (", ".join(f"{k}={v}" for k, v in sorted(filled.items()))
                   or "(no fixes)"),
            )
        )
    else:
        found = before_block.get("fingerprint")
        invariants = [
            Invariant(
                "before snapshot built with the known overlay",
                before_block.get("enabled") is True
                and found == before_overlay_fingerprint,
                f"enabled={before_block.get('enabled')!r}, fingerprint {found} "
                f"(expected {before_overlay_fingerprint})",
            )
        ]
        new_fixes = _new_fixes(before_meta, after_meta)
        added = {fix_id: filled.get(fix_id, 0) for fix_id in sorted(new_fixes)}
        invariants.append(
            Invariant(
                "after snapshot built with a later overlay that added fixes",
                after_block.get("enabled") is True
                and after_block.get("fingerprint") != before_overlay_fingerprint
                and bool(added) and all(n > 0 for n in added.values()),
                f"enabled={after_block.get('enabled')!r}, fingerprint "
                f"{after_block.get('fingerprint')}, results_patched "
                + (", ".join(f"{k}={v}" for k, v in added.items()) or "(no new fixes)"),
            )
        )

    if expect_fingerprint is not None:
        found = after_block.get("fingerprint")
        invariants.append(
            Invariant(
                "after snapshot overlay is the expected one",
                found == expect_fingerprint,
                f"fingerprint {found} (expected {expect_fingerprint})",
            )
        )
    return invariants


def _observed_fills(con) -> dict[str, int]:
    """Fact rows per source_config whose temperature or cap went from none."""
    rows = con.execute(
        "SELECT COALESCE(source_config, '') AS source_config, COUNT(*) "
        "FROM patched_facts GROUP BY 1"
    ).fetchall()
    return {config: int(n) for config, n in rows}


def _invariant_observed_movement(
    con,
    before_meta: dict,
    after_meta: dict,
    before_overlay_fingerprint: str | None,
) -> list[Invariant]:
    """What the after snapshot says it filled really moved in the tables.

    A fix's own count is its account of itself; this holds it to the parquet.
    For every source config a fix reports patching, fact rows in that config
    have to have gone from no temperature or output cap to one, and no more
    of them than the fix claims results (see `FACT_FILL_NOTE`).

    Comparing two overlay-enabled snapshots, only the fixes the after side
    adds are held to this: whatever the before side already filled is filled
    on both, so it cannot move again.
    """
    fixes = (after_meta.get("datafix") or {}).get("fixes") or {}
    if before_overlay_fingerprint is not None:
        fixes = _new_fixes(before_meta, after_meta)
    claiming = {
        fix_id: block or {}
        for fix_id, block in fixes.items()
        if ((block or {}).get("results_patched") or 0) > 0
    }
    observed = _observed_fills(con)

    claimed: Counter = Counter()
    for block in claiming.values():
        for config, count in (block.get("by_source_config") or {}).items():
            claimed[config] += count or 0

    invariants: list[Invariant] = []
    for fix_id, block in sorted(claiming.items()):
        configs = block.get("by_source_config") or {}
        if not configs:
            continue
        detail, bad = [], []
        for config in sorted(configs):
            seen = observed.get(config, 0)
            detail.append(f"{config}: {seen} fact row(s) filled, {configs[config]} claimed")
            if seen == 0 or seen > claimed[config]:
                bad.append(config)
        invariants.append(
            Invariant(
                f"{fix_id}: the tables moved where it reports patching",
                not bad,
                "; ".join(detail) + ("" if not bad else f" ; off: {', '.join(bad)}"),
            )
        )

    # An older snapshot's block names no configs. Its claim is then only
    # about the comparison as a whole, so that is what it is held to.
    unattributed = {
        fix_id: block.get("results_patched") or 0
        for fix_id, block in claiming.items()
        if not (block.get("by_source_config") or {})
    }
    if unattributed:
        seen = sum(observed.values())
        cap = sum(unattributed.values())
        invariants.append(
            Invariant(
                "fixes naming no source config: the tables moved",
                0 < seen <= cap,
                f"{seen} fact row(s) filled against {cap} result(s) claimed by "
                + ", ".join(sorted(unattributed)),
            )
        )

    if not invariants:
        invariants.append(
            Invariant(
                "the after snapshot claims a fix filled something",
                False,
                "no fix in the after snapshot's datafix block reports a patched result",
            )
        )
    return invariants


def _invariant_inputs_only_filled(con) -> Invariant:
    """No fact's temperature or output cap was rewritten or cleared.

    The overlay only ever fills what is missing, so a value that was there
    before has to be there after, unchanged.
    """
    changed = " OR ".join(
        f"(b.{_col(f)} IS NOT NULL AND a.{_col(f)} IS DISTINCT FROM b.{_col(f)})"
        for f in DIRECT_INPUT_FIELDS
    )
    rows = con.execute(
        f"SELECT COUNT(*) FROM b_fact_results b JOIN a_fact_results a "
        f"USING (fact_id) WHERE {changed}"
    ).fetchone()[0]
    return Invariant(
        "stated generation inputs neither rewritten nor cleared",
        rows == 0,
        f"{rows} fact row(s) changed a temperature or output cap that was "
        f"already stated",
    )


def _invariant_fact_scope(con) -> list[Invariant]:
    """Inside the target sources too, a fact moves only where a fill reached.

    The direct inputs and everything read off one fact's own payload move
    only on the facts that were filled. The variant key and the comparability
    verdicts compare a fact against its group, so they also move on facts
    that were not filled themselves but share a group with one.
    """
    unfilled = "b.fact_id NOT IN (SELECT fact_id FROM patched_facts)"
    unreached = "b.fact_id NOT IN (SELECT fact_id FROM reached_facts)"
    checks = (
        ("direct generation inputs", DIRECT_INPUT_COLUMNS, unfilled,
         "facts the overlay filled"),
        ("reproducibility columns", DERIVED_REPRO_COLUMNS, unfilled,
         "facts the overlay filled"),
        ("variant and comparability columns", VARIANT_COLUMNS, unreached,
         "facts in a comparability group the overlay filled into"),
    )
    invariants = []
    for label, columns, where, scope in checks:
        moved = _column_movement(con, "fact_results", ("fact_id",), list(columns),
                                 where=where)
        invariants.append(
            Invariant(
                f"fact_results: {label} move only on {scope}",
                not moved,
                "no movement outside that set" if not moved else _fmt_moved(moved),
            )
        )
    return invariants


def _invariant_reached_rows(con, specs: list[TableSpec]) -> list[Invariant]:
    """A view row moves only when a fill could have reached it.

    `_invariant_out_of_scope` holds the rows outside the target sources;
    this holds the rows inside them that no fill reached. Attribution is
    per table, see `ATTRIBUTION_NOTE`.
    """
    invariants = []
    for spec in specs:
        if spec.name == "fact_results" or not spec.key:
            continue
        reached = _reached_predicate(spec, "b")
        if reached is None:
            continue
        columns = [c for c in _common_columns(con, spec.name)[0]
                   if c not in IGNORED_COLUMNS and c not in spec.key]
        moved = _column_movement(con, spec.name, spec.key, columns,
                                 where=f"NOT ({reached})")
        invariants.append(
            Invariant(
                f"{spec.name}: no movement on rows no fill reached",
                not moved,
                "no column moved" if not moved else _fmt_moved(moved),
            )
        )
    return invariants


def _generation_config_rest(alias: str, column: str) -> str:
    """One generation payload with the two fields the overlay writes taken
    out, so what is left can be compared member by member."""
    patch = json.dumps({"generation_args": {f: None for f in DIRECT_INPUT_FIELDS}})
    return (
        f"json_merge_patch(CAST(to_json({alias}.{_col(column)}) AS JSON), "
        f"{_lit(patch)})"
    )


def _invariant_generation_members(con, specs: list[TableSpec]) -> list[Invariant]:
    """Nothing in the emitted generation payload moves but the two fields.

    The payload is one struct, so an allowlist over columns would wave the
    whole of it through. This compares it with `generation_args.temperature`
    and `generation_args.max_tokens` removed: every other member, at every
    depth, has to come through identical.
    """
    invariants = []
    for spec in specs:
        for column in MEMBERWISE_COLUMNS.get(spec.name, ()):
            rows = con.execute(
                f"SELECT COUNT(*) FROM b_{spec.name} b JOIN a_{spec.name} a "
                f"ON {_join_on(spec.key)} WHERE "
                f"{_generation_config_rest('b', column)} IS DISTINCT FROM "
                f"{_generation_config_rest('a', column)}"
            ).fetchone()[0]
            invariants.append(
                Invariant(
                    f"{spec.name}.{column}: only "
                    + " and ".join(
                        f"generation_args.{f}" for f in DIRECT_INPUT_FIELDS
                    )
                    + " move",
                    rows == 0,
                    f"{rows} row(s) moved in another member of the payload",
                )
            )
    return invariants


def _invariant_tables_present(before_dir: Path, after_dir: Path) -> Invariant:
    """Every table this script compares is on both sides.

    A table missing on one side would silently drop out of the comparison
    and take every row it holds with it.
    """
    missing = [
        f"{spec.name} ({side})"
        for spec in TABLES
        for side, directory in (("before", before_dir), ("after", after_dir))
        if not (directory / f"{spec.name}.parquet").is_file()
    ]
    return Invariant(
        "every compared table present on both sides",
        not missing,
        f"{len(TABLES)} tables on both sides" if not missing
        else "missing: " + ", ".join(missing),
    )


def _invariant_keys_usable(con, specs: list[TableSpec]) -> Invariant:
    """Every keyed table's key really identifies a row, on both sides.

    A duplicated key turns the key-matched comparisons below into a fan-out
    and an unexpected null into a silent drop, so movement could hide inside
    either. Key columns a table declares nullable are excluded from the null
    check: they carry "no condition" and the joins are null-safe.
    """
    bad = []
    for spec in specs:
        if not spec.key:
            continue
        packed = "struct_pack(" + ", ".join(f"{_col(k)} := {_col(k)}" for k in spec.key) + ")"
        required = [k for k in spec.key if k not in spec.nullable]
        nulls = " OR ".join(f"{_col(k)} IS NULL" for k in required) or "FALSE"
        for side, alias in (("before", "b"), ("after", "a")):
            rows, distinct, null_rows = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT CAST({packed} AS VARCHAR)), "
                f"SUM(CASE WHEN {nulls} THEN 1 ELSE 0 END) FROM {alias}_{spec.name}"
            ).fetchone()
            if rows != distinct:
                bad.append(f"{spec.name} ({side}): {rows - distinct} duplicate key(s)")
            if null_rows:
                bad.append(f"{spec.name} ({side}): {int(null_rows)} row(s) with a null key")
    return Invariant(
        "keys unique and non-null on both sides",
        not bad,
        "every key identifies one row" if not bad else "; ".join(bad),
    )


def _invariant_only_signal_columns(con, specs: list[TableSpec]) -> Invariant:
    """Nothing outside each table's signal allowlist moves, on any row.

    The scoped invariants below let a row move once it is in scope, and a
    row reaches scope through a model or a composite it merely shares. This
    one holds every key-matched row of every table to the columns the
    overlay can actually reach. Columns compared member by member are held
    by `_invariant_generation_members` instead.
    """
    moved: list[tuple[str, int]] = []
    for spec in specs:
        if not spec.key:
            continue
        common = _common_columns(con, spec.name)[0]
        allowed = set(MOVABLE_COLUMNS.get(spec.name, ()))
        allowed.update(MEMBERWISE_COLUMNS.get(spec.name, ()))
        columns = [c for c in common
                   if c not in IGNORED_COLUMNS and c not in spec.key and c not in allowed]
        moved += [(f"{spec.name}.{column}", rows)
                  for column, rows in _column_movement(con, spec.name, spec.key, columns)]
    return Invariant(
        "only signal columns move",
        not moved,
        "no column outside the per-table signal allowlists moved" if not moved
        else _fmt_moved(moved),
    )


def _invariant_fact_multiset(con) -> Invariant:
    only_before, only_after = _unmatched(con, "fact_results", ("fact_id",))
    dupes = con.execute(
        "SELECT (SELECT COUNT(*) - COUNT(DISTINCT fact_id) FROM b_fact_results), "
        "       (SELECT COUNT(*) - COUNT(DISTINCT fact_id) FROM a_fact_results)"
    ).fetchone()
    ok = only_before == 0 and only_after == 0 and dupes[0] == 0 and dupes[1] == 0
    return Invariant(
        "fact_id multiset equal",
        ok,
        f"{only_before} only in before, {only_after} only in after, "
        f"duplicate fact_ids before/after {dupes[0]}/{dupes[1]}",
    )


def _invariant_row_counts(con, specs: list[TableSpec]) -> Invariant:
    bad = []
    for spec in specs:
        counts = con.execute(
            f"SELECT (SELECT COUNT(*) FROM b_{spec.name}), (SELECT COUNT(*) FROM a_{spec.name})"
        ).fetchone()
        if counts[0] != counts[1]:
            bad.append(f"{spec.name} {counts[0]} to {counts[1]}")
    return Invariant(
        "row counts per table equal",
        not bad,
        "all tables match" if not bad else "; ".join(bad),
    )


def _invariant_group_ids(con) -> Invariant:
    row = con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM b_fact_results b JOIN a_fact_results a USING (fact_id)
             WHERE b.comparability_group_id IS DISTINCT FROM a.comparability_group_id),
            (SELECT COUNT(*) FROM (
                (SELECT DISTINCT comparability_group_id FROM b_fact_results
                 EXCEPT SELECT DISTINCT comparability_group_id FROM a_fact_results)
                UNION ALL
                (SELECT DISTINCT comparability_group_id FROM a_fact_results
                 EXCEPT SELECT DISTINCT comparability_group_id FROM b_fact_results)))
        """
    ).fetchone()
    return Invariant(
        "comparability group ids equal",
        row[0] == 0 and row[1] == 0,
        f"{row[0]} facts reassigned, {row[1]} group ids present on one side only",
    )


def _invariant_representatives(con) -> Invariant:
    """The representative fact id is not emitted, so this checks the inputs and
    the outputs of the pick: the fact rows each triple offers, the columns the
    pick orders on, and the representative-sourced columns on the view."""
    triple_key = ", ".join(_col(k) for k in TRIPLE_KEY)
    membership = con.execute(
        f"""
        WITH b AS (
            SELECT {triple_key}, list_sort(list(fact_id)) AS facts
            FROM b_fact_results GROUP BY {triple_key}
        ),
        a AS (
            SELECT {triple_key}, list_sort(list(fact_id)) AS facts
            FROM a_fact_results GROUP BY {triple_key}
        )
        SELECT COUNT(*) FROM b FULL OUTER JOIN a ON
            {" AND ".join(f"b.{_col(k)} IS NOT DISTINCT FROM a.{_col(k)}" for k in TRIPLE_KEY)}
        WHERE CAST(b.facts AS VARCHAR) IS DISTINCT FROM CAST(a.facts AS VARCHAR)
        """
    ).fetchone()[0]
    order_moved = _column_movement(
        con, "fact_results", ("fact_id",), list(FACT_REPRESENTATIVE_ORDER_COLUMNS)
    )
    view_moved = _column_movement(
        con, "eval_results_view", ERV_KEY, list(ERV_REPRESENTATIVE_COLUMNS)
    )
    ok = membership == 0 and not order_moved and not view_moved
    detail = (
        f"{membership} triples changed membership; "
        f"pick-order columns moved: {_fmt_moved(order_moved) or 'none'}; "
        f"representative-sourced view columns moved: "
        f"{_fmt_moved(view_moved) or 'none'}"
    )
    return Invariant("representative fact ids equal", ok, detail)


def _invariant_headline(con, before_headline, after_headline) -> list[Invariant]:
    fact_flips = con.execute(
        "SELECT COUNT(*) FROM b_fact_results b JOIN a_fact_results a USING (fact_id) "
        "WHERE b.is_headline IS DISTINCT FROM a.is_headline"
    ).fetchone()[0]
    view_flips = con.execute(
        f"SELECT COUNT(*) FROM b_eval_results_view b JOIN a_eval_results_view a "
        f"ON {_join_on(ERV_KEY)} WHERE b.is_headline IS DISTINCT FROM a.is_headline"
    ).fetchone()[0]
    invariants = [
        Invariant(
            "headline mapping equal",
            fact_flips == 0 and view_flips == 0,
            f"{fact_flips} fact rows and {view_flips} view rows flipped is_headline",
        )
    ]

    b_flat = _flatten(before_headline)
    a_flat = _flatten(after_headline)
    ignored = {"generated_at"}
    diffs = []
    for path in sorted(set(b_flat) | set(a_flat)):
        root = path.split(".")[0].split("[")[0]
        if root in ignored or root in HEADLINE_MOVING_BLOCKS:
            continue
        if b_flat.get(path) != a_flat.get(path):
            diffs.append(path)
    invariants.append(
        Invariant(
            "headline.json blocks outside reproducibility and comparability equal",
            not diffs,
            "all equal" if not diffs
            else f"{len(diffs)} paths moved, first: {diffs[:5]}",
        )
    )
    return invariants


def _invariant_scores(con) -> Invariant:
    fact_moved = _column_movement(con, "fact_results", ("fact_id",), list(FACT_SCORE_COLUMNS))
    view_moved = _column_movement(con, "eval_results_view", ERV_KEY, list(ERV_SCORE_COLUMNS))
    evals_moved = _column_movement(
        con, "evals_view", ("evaluation_id",),
        ["avg_score", "avg_score_norm", "top_score", "best_model", "worst_model"],
    )
    models_moved = _column_movement(
        con, "models_view", ("model_key",), ["score_summary", "top_scores"]
    )
    moved = fact_moved + view_moved + evals_moved + models_moved
    return Invariant(
        "scores equal",
        not moved,
        "no score column moved" if not moved else _fmt_moved(moved),
    )


def _invariant_identity(con, specs: list[TableSpec]) -> Invariant:
    fact_moved = _column_movement(con, "fact_results", ("fact_id",), list(FACT_IDENTITY_COLUMNS))
    view_moved = _column_movement(con, "eval_results_view", ERV_KEY, list(ERV_IDENTITY_COLUMNS))
    key_sets = []
    for spec in specs:
        if not spec.key:
            continue
        only_before, only_after = _unmatched(con, spec.name, spec.key)
        if only_before or only_after:
            key_sets.append(f"{spec.name} {only_before}/{only_after} unmatched")
    moved = fact_moved + view_moved
    return Invariant(
        "identity columns equal",
        not moved and not key_sets,
        "no identity column moved and every key matched"
        if not moved and not key_sets
        else f"columns moved: {_fmt_moved(moved) or 'none'}; "
             f"unmatched keys: {'; '.join(key_sets) or 'none'}",
    )


def _invariant_out_of_scope(con, specs: list[TableSpec]) -> list[Invariant]:
    invariants = []
    for spec in specs:
        common, only_before, only_after = _common_columns(con, spec.name)
        if only_before or only_after:
            invariants.append(
                Invariant(
                    f"{spec.name}: schema stable",
                    False,
                    f"columns only in before {only_before}, only in after {only_after}",
                )
            )
        columns = [c for c in common if c not in IGNORED_COLUMNS and c not in spec.key]
        if not spec.key:
            diff = _row_multiset_diff(con, spec.name, [c for c in common if c not in IGNORED_COLUMNS])
            invariants.append(
                Invariant(
                    f"{spec.name}: no movement (no unique key, compared as a row multiset, "
                    f"not attributable to a source_config)",
                    diff == 0,
                    f"{diff} rows present on one side only",
                )
            )
            continue
        in_scope = _scope_predicate(spec, "b")
        spill = _spill_predicate(spec, "b") if in_scope else None
        if in_scope is None:
            moved = _column_movement(con, spec.name, spec.key, columns)
            label = (
                f"{spec.name}: no movement (no source_config attribution available, "
                f"whole table compared)"
            )
        elif spill is None:
            moved = _column_movement(con, spec.name, spec.key, columns,
                                     where=f"NOT ({in_scope})")
            label = f"{spec.name}: zero movement outside the target sources"
        else:
            moved = _column_movement(con, spec.name, spec.key, columns,
                                     where=f"NOT ({in_scope}) AND NOT ({spill})")
            label = (
                f"{spec.name}: zero movement outside the target sources and the "
                f"comparability groups they share"
            )
        invariants.append(
            Invariant(label, not moved,
                      "no column moved" if not moved else _fmt_moved(moved))
        )
        if spill is None:
            continue
        allowed = SPILL_COLUMNS[spec.name]
        where = f"NOT ({in_scope}) AND ({spill})"
        moved = _column_movement(
            con, spec.name, spec.key, [c for c in columns if c not in allowed], where=where
        )
        if spec.name == "eval_results_view":
            parts = " OR ".join(
                f"CAST(b.evalcards_annotations.{part} AS VARCHAR) IS DISTINCT FROM "
                f"CAST(a.evalcards_annotations.{part} AS VARCHAR)"
                for part in ERV_ANNOTATION_STABLE_PARTS
            )
            rows = con.execute(
                f"SELECT COUNT(*) FROM b_{spec.name} b JOIN a_{spec.name} a "
                f"ON {_join_on(spec.key)} WHERE {where} AND ({parts})"
            ).fetchone()[0]
            if rows:
                moved.append(("evalcards_annotations outside comparability", int(rows)))
        shifted = _column_movement(con, spec.name, spec.key, list(allowed), where=where)
        invariants.append(
            Invariant(
                f"{spec.name}: other sources' rows sharing a comparability group with a "
                f"target fact move only in comparability columns",
                not moved,
                (f"comparability movement: {_fmt_moved(shifted) or 'none'}"
                 if not moved else f"unexpected: {_fmt_moved(moved)}"),
            )
        )
    return invariants


# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diff the signals between two warehouse snapshots."
    )
    parser.add_argument("before_dir", type=Path)
    parser.add_argument("after_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="markdown report path; stdout when omitted")
    parser.add_argument(
        "--target-source-config", action="append", dest="targets", default=None,
        help="source_config the overlay may move; repeatable "
             f"(default: {', '.join(DEFAULT_TARGET_SOURCE_CONFIGS)})",
    )
    parser.add_argument(
        "--expect-fingerprint", default=None,
        help="overlay fingerprint the after snapshot must carry; asserted as "
             "an invariant when given",
    )
    parser.add_argument(
        "--before-overlay-fingerprint", default=None,
        help="compare two overlay-enabled snapshots instead of an off/on "
             "pair: the before side must carry exactly this overlay, the "
             "after side a later one whose added fixes each filled something. "
             "Target source configs then default to the ones those fixes "
             "report reaching.",
    )
    return parser.parse_args(argv)


def _emit(
    args: argparse.Namespace,
    report: list[str],
    invariants: list[Invariant],
    tail: list[str] | None = None,
) -> int:
    """Append the invariant section, write the report, return the exit code."""
    failed = [inv for inv in invariants if not inv.ok]
    report = report + ["## Invariants", ""]
    for inv in invariants:
        report.append(f"- [{'PASS' if inv.ok else 'FAIL'}] {inv.name}: {inv.detail}")
    report += ["", f"{len(invariants) - len(failed)} of {len(invariants)} invariants passed.", ""]
    report += tail or []

    text = "\n".join(report) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    for inv in failed:
        print(f"FAIL {inv.name}: {inv.detail}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    for snapshot in (args.before_dir, args.after_dir):
        if not (snapshot / "snapshot_meta.json").is_file():
            print(f"{snapshot} is not a snapshot directory.", file=sys.stderr)
            return 2

    before_meta = _load_json(args.before_dir / "snapshot_meta.json")
    after_meta = _load_json(args.after_dir / "snapshot_meta.json")
    before_pins, after_pins = _pins(before_meta), _pins(after_meta)
    mismatched = [k for k in sorted(set(before_pins) | set(after_pins))
                  if before_pins.get(k) != after_pins.get(k)]
    if mismatched:
        for name in mismatched:
            print(
                f"pin mismatch on {name}: "
                f"{before_pins.get(name)} vs {after_pins.get(name)}",
                file=sys.stderr,
            )
        print("Refusing to diff snapshots built on different pins.", file=sys.stderr)
        return 2

    experiment = _invariant_experiment(
        before_meta, after_meta, args.expect_fingerprint,
        args.before_overlay_fingerprint,
    )

    if args.targets:
        targets = tuple(args.targets)
    elif args.before_overlay_fingerprint:
        targets = _new_fix_targets(before_meta, after_meta)
    else:
        targets = DEFAULT_TARGET_SOURCE_CONFIGS
    if not targets:
        # Everything below is scoped to these, so a report without them
        # would assert nothing about where the overlay was allowed to move.
        return _emit(args, ["# Datafix signal diff", ""], experiment + [
            Invariant(
                "target source configs known",
                False,
                "the after snapshot's datafix block names none for the fixes "
                "it added; pass --target-source-config",
            )
        ])
    presence = _invariant_tables_present(args.before_dir, args.after_dir)
    if not presence.ok:
        # Without every table there is nothing to compare the missing rows
        # against, so the run stops on the invariant rather than reporting
        # on the tables that happen to be there.
        return _emit(
            args, ["# Datafix signal diff", ""], experiment + [presence]
        )

    specs = list(TABLES)
    table_names = [s.name for s in specs]

    con = duckdb.connect()
    _register(con, "b", args.before_dir, table_names)
    _register(con, "a", args.after_dir, table_names)
    _build_target_views(con, targets)

    before_headline = _load_json(args.before_dir / "headline.json")
    after_headline = _load_json(args.after_dir / "headline.json")

    composites = [r[0] for r in con.execute(
        "SELECT composite_slug FROM target_composites ORDER BY 1"
    ).fetchall()]

    report: list[str] = ["# Datafix signal diff", ""]
    report += _md_table(
        ["side", "snapshot_id", "path"],
        [["before", before_meta.get("snapshot_id"), str(args.before_dir)],
         ["after", after_meta.get("snapshot_id"), str(args.after_dir)]],
    )
    report += ["Upstream pins (identical on both sides):", ""]
    report += _md_table(
        ["pin", "sha"],
        [[name, before_pins[name]] for name in sorted(before_pins)],
    )
    report += [
        f"Target source configs: {', '.join(targets)}.",
        "",
        f"Composites they land in: {', '.join(composites) or 'none'}.",
        "",
    ]
    if args.before_overlay_fingerprint:
        added = sorted(_new_fixes(before_meta, after_meta))
        report += [
            f"Comparing two overlay-enabled snapshots. Fixes the after side "
            f"adds: {', '.join(added) or 'none'}.",
            "",
        ]
    for side, meta in (("before", before_meta), ("after", after_meta)):
        datafix = meta.get("datafix")
        if datafix is None:
            report += [
                f"The {side} snapshot carries no `datafix` block in "
                f"snapshot_meta.json.", "",
            ]
        elif side == "after" or args.before_overlay_fingerprint:
            report += [f"Datafix block from the {side} snapshot:", "", "```json",
                       json.dumps(datafix, indent=2, sort_keys=True), "```", ""]

    only_in_before = sorted({p.stem for p in args.before_dir.glob("*.parquet")}
                            - set(table_names))
    only_in_after = sorted({p.stem for p in args.after_dir.glob("*.parquet")}
                           - set(table_names))
    if only_in_before or only_in_after:
        report += [
            f"Tables not compared (this script has no spec for them): "
            f"before {only_in_before or 'none'}, after {only_in_after or 'none'}.",
            "",
        ]

    report += _section_reproducibility(con, before_headline, after_headline)
    report += _section_variants(con)
    report += _section_comparability(con, before_headline, after_headline)
    report += _section_generation_payload(con)

    invariants: list[Invariant] = list(experiment)
    invariants += [
        presence,
        _invariant_keys_usable(con, specs),
        _invariant_fact_multiset(con),
        _invariant_row_counts(con, specs),
        _invariant_group_ids(con),
        _invariant_representatives(con),
    ]
    invariants += _invariant_observed_movement(
        con, before_meta, after_meta, args.before_overlay_fingerprint
    )
    invariants += _invariant_headline(con, before_headline, after_headline)
    invariants += [
        _invariant_scores(con),
        _invariant_identity(con, specs),
        _invariant_only_signal_columns(con, specs),
        _invariant_inputs_only_filled(con),
    ]
    invariants += _invariant_generation_members(con, specs)
    invariants += _invariant_fact_scope(con)
    invariants += _invariant_reached_rows(con, specs)
    invariants += _invariant_out_of_scope(con, specs)

    notes = ["## Notes", "", MISSING_GEN_CONFIG_NOTE, "", FACT_FILL_NOTE, "",
             ATTRIBUTION_NOTE, "",
             "`eval_results_view` does not emit the representative fact id, so the "
             "representative-pick invariant is asserted over the fact rows each triple "
             "offers, the fact columns the pick orders on, and the view columns the "
             "representative sources.", ""]
    return _emit(args, report, invariants, notes)


if __name__ == "__main__":
    sys.exit(main())

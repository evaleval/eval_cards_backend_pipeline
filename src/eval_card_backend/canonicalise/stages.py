"""DuckDB stages for canonicalisation. SQL-heavy; orchestrated by `pipeline.run`.

Each `stage_*` function takes a DuckDB connection and creates one or more
tables on the connection. Tables are wired by name across stages.

Implementation notes:
- EEE records arrive as a typed `pyarrow.Table` from `sources.eee.load_arrow_table`,
  validated against the vendored upstream contract. Stage A registers the
  Arrow table directly with DuckDB (zero-copy) under `eee_raw` — no temp JSONL,
  no schema drift between configs.
- Cards are pre-staged from a Python dict via a temp JSONL.
- `metric_kind` / `metric_unit` / `min_score` / `max_score` / `lower_is_better`
  on each fact row come from the metric-meta resolver UDF, not directly from
  `canonical_metrics` (which is sparse for those fields). Stage A loads the
  registry columns; Stage D's `joined` CTE invokes the UDF once per row,
  and the outer SELECT destructures `_meta.*` into flat columns.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa

from eval_card_backend.signals.reproducibility import (
    AGENTIC_REPRODUCIBILITY_FIELDS,
    BASE_REPRODUCIBILITY_FIELDS,
)
from eval_card_backend.canonicalise import taxonomy
from eval_card_backend.config import EEE_DATASET_REPO
from eval_card_backend.sources import collections as collections_src
from eval_card_backend.sources.registry import read_parquet_arg


# DuckDB's duplicate-column suffix: `a` selected twice becomes `a`, `a_1`.
_AUTO_RENAME_RE = re.compile(r"(.+)_\d+")


def instance_file_path_sql(file_path: str, record_path: str) -> str:
    """SQL resolving an upstream sample-file pointer to one repo-relative path.

    Upstream `detailed_evaluation_results.file_path` arrives in three shapes,
    all of them seen in production:

        <uuid>_samples.jsonl                       bare filename
        ./<uuid>_samples.jsonl                     dot-relative
        data/<config>/<org>/<model>/<uuid>_samples.jsonl    already rooted

    The first two only mean anything next to the record that declared them,
    so they are resolved against that record's directory — `source_record_path`
    is the repo-relative path of the source JSON, carried since Stage A. All
    three then share one shape that addresses the EEE repo from its root, which
    is what `instance_file_url` and every downstream consumer needs.

    Returns NULL when the pointer is absent, or when a relative pointer has no
    usable record path to resolve against — a half-resolved path would 404 just
    as silently as the raw one, so emit nothing rather than guess.

    `file_path` and `record_path` are SQL expressions, not values.
    """
    return f"""CASE
                WHEN {file_path} IS NULL THEN NULL
                WHEN {file_path} LIKE '%/%'
                 AND {file_path} NOT LIKE './%'
                    THEN {file_path}
                WHEN {record_path} IS NULL
                  OR {record_path} NOT LIKE '%/%' THEN NULL
                ELSE regexp_replace({record_path}, '/[^/]*$', '')
                     || '/'
                     || regexp_replace({file_path}, '^\\./', '')
            END"""


def explicit_projection_sql(con, relation: str, alias: str | None = None) -> str:
    """Comma-separated column list of `relation` (a table name or a
    parenthesised subquery), in declared order and with every name quoted.

    `SELECT *` hides a whole class of projection defect: `SELECT t.*, t.*`
    binds without complaint, DuckDB auto-renames the second copy (`a` → `a_1`)
    and the materialised payload silently doubles. Enumerating the columns
    makes the emitted schema an explicit, order-stable statement, and the
    duplicate check below turns that defect into a hard failure instead of a
    memory bill. Binding-only (`DESCRIBE`), so it costs a parse, not a scan.
    """
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]
    auto_renamed = sorted(
        c for c in cols
        if _AUTO_RENAME_RE.fullmatch(c)
        and _AUTO_RENAME_RE.fullmatch(c).group(1) in cols
    )
    if auto_renamed:
        raise RuntimeError(
            f"duplicate projection in {relation}: DuckDB auto-renamed "
            f"{auto_renamed}. A column is being selected twice (a repeated "
            f"`alias.*` is the usual cause); drop the duplicate rather than "
            f"letting the wide payload materialise twice."
        )
    prefix = f"{alias}." if alias else ""
    return ", ".join(f'{prefix}"{c}"' for c in cols)


def protocol_exclusion_sql(col: str = "protocol_condition") -> str:
    """Answer-feedback exclusion predicate, NULL-safe canonical form.

    Used verbatim at every scalar-rollup site. A bare
    `feedback != 'answer_feedback'` would be WRONG: the extraction is NULL
    for every ordinary row and `NULL != x` is NULL, which would drop ALL
    ordinary rows from every pool.
    """
    return (
        f"COALESCE(json_extract_string({col}, '$.feedback'), 'none') "
        f"<> 'answer_feedback'"
    )


def json_array_guard_sql(expr: str) -> str:
    """Predicate: `expr` holds a JSON array.

    `json_valid(x) AND json_type(x) = 'ARRAY'` is NOT safe — DuckDB evaluates
    both sides of the AND and `json_type` raises on a malformed string. Feeding
    `json_type` a NULL instead keeps the validity guard inside one expression.
    """
    return f"json_type(CASE WHEN json_valid({expr}) THEN {expr} END) = 'ARRAY'"


def _build_repro_missing_fields_sql() -> str:
    """Concatenated array-literal expression for `repro_missing_fields`.

    Mirrors the active rule in `signals/reproducibility.py`. Base fields
    fire unconditionally on missing; agentic fields fire only when
    `is_agentic`. Same field names are also used as `has_<field>` flag
    columns in the upstream `base` CTE — keep the two in sync.
    """
    base_clauses = [
        f"(CASE WHEN NOT has_{f} THEN ['{f}'] ELSE []::VARCHAR[] END)"
        for f in BASE_REPRODUCIBILITY_FIELDS
    ]
    agentic_clauses = [
        f"(CASE WHEN is_agentic AND NOT has_{f} THEN ['{f}'] ELSE []::VARCHAR[] END)"
        for f in AGENTIC_REPRODUCIBILITY_FIELDS
    ]
    return "\n                 || ".join(base_clauses + agentic_clauses)


_REPRO_MISSING_FIELDS_SQL = _build_repro_missing_fields_sql()
_REPRO_BASE_COUNT = len(BASE_REPRODUCIBILITY_FIELDS)
_REPRO_AGENTIC_COUNT = _REPRO_BASE_COUNT + len(AGENTIC_REPRODUCIBILITY_FIELDS)


# Stage J view-layer signal-summary STRUCT shapes. Identical across
# `models_view` and `evals_view`; declared here so a shape change is a
# one-line edit.
_REPRODUCIBILITY_SUMMARY_STRUCT = (
    "STRUCT("
    "results_total INTEGER, "
    "has_reproducibility_gap_count INTEGER, "
    "populated_ratio_avg DOUBLE"
    ")"
)
_PROVENANCE_SUMMARY_STRUCT = (
    "STRUCT("
    "total_results INTEGER, total_groups INTEGER, "
    "multi_source_groups INTEGER, first_party_only_groups INTEGER, "
    "source_type_distribution STRUCT("
    "  first_party INTEGER, third_party INTEGER, "
    "  collaborative INTEGER, unspecified INTEGER"
    ")"
    ")"
)
_COMPARABILITY_SUMMARY_STRUCT = (
    "STRUCT("
    "total_groups INTEGER, "
    "groups_with_variant_check INTEGER, "
    "groups_with_cross_party_check INTEGER, "
    "variant_divergent_count INTEGER, "
    "cross_party_divergent_count INTEGER"
    ")"
)


def _source_type_distribution_sql(alias: str) -> str:
    """Emit four SQL aggregate columns for the
    `source_type_distribution` four-way breakdown derived from
    `coverage_cell` + `has_third_party`. Caller must reference the
    output names `pst_first_party`, `pst_third_party`,
    `pst_collaborative`, `pst_unspecified`.
    """
    a = alias
    return (
        f"CAST(SUM(CASE WHEN {a}.coverage_cell = 'self'                              THEN 1 ELSE 0 END) AS INTEGER) AS pst_first_party,\n"
        f"                CAST(SUM(CASE WHEN {a}.coverage_cell = 'third' AND {a}.has_third_party     THEN 1 ELSE 0 END) AS INTEGER) AS pst_third_party,\n"
        f"                CAST(SUM(CASE WHEN {a}.coverage_cell = 'both'                              THEN 1 ELSE 0 END) AS INTEGER) AS pst_collaborative,\n"
        f"                CAST(SUM(CASE WHEN {a}.coverage_cell = 'third' AND NOT {a}.has_third_party THEN 1 ELSE 0 END) AS INTEGER) AS pst_unspecified"
    )


def org_normalize_sql(column_expr: str) -> str:
    r"""Return the SQL expression that lowercases, collapses ASCII
    whitespace runs, trims, and NULLs out the empty string. Mirrors
    `signals/comparability.normalize_org_name` for the same input shape;
    parity is asserted in `tests/test_udf_roundtrip.py`. Use this helper
    everywhere instead of inlining the regex so the two paths stay in
    sync.

    The regex is ASCII-only (`\s` in DuckDB / RE2). Unicode whitespace
    (e.g. NBSP) is left intact in the SQL path; Python's `re.sub` would
    collapse it. Production data has not exhibited this divergence.
    """
    return (
        f"NULLIF(trim(regexp_replace(lower({column_expr}), '\\s+', ' ', 'g')), '')"
    )


def org_display_normalize_sql(column_expr: str) -> str:
    r"""Display-preserving counterpart of `org_normalize_sql`.

    Collapses ASCII whitespace runs, trims, and NULLs out the empty
    string and the literal placeholder `unknown` — but **does not**
    lowercase, so values stay in their original casing for rendering
    (e.g. `Hugging Face`, `Allen Institute for AI`, `LLM Stats`).

    Applied at ingestion (`source_organization_name → org_raw`) so
    downstream aggregations distinct-count cleaned strings. Without
    this, upstream whitespace inconsistencies (e.g. one source writing
    `New York University,  Princeton University …` with a double space
    and another with a single space) inflate the distinct-eval-provider
    count. There's no canonical-org registry behind this — eval-provider
    orgs proliferate freely in the wild and we don't want to build a
    seed/orgs.yaml entry for every new leaderboard — so this is a
    light-touch surface clean rather than alias resolution.
    """
    cleaned = f"trim(regexp_replace({column_expr}, '\\s+', ' ', 'g'))"
    return (
        f"CASE WHEN {cleaned} = '' OR lower({cleaned}) = 'unknown' "
        f"THEN NULL ELSE {cleaned} END"
    )


class StageEStats(NamedTuple):
    """Row-count breakdown for Stage E. Exposed so the orchestrator can
    populate snapshot_meta with each drop reason separately."""
    pre: int                    # rows in fact_results_staging
    n_dropped_no_score: int     # score IS NULL
    n_dropped_sentinel: int     # score = -1 sentinel
    n_dropped_dedup: int        # fact_id collisions
    post: int                   # final fact_results_signaled count

log = logging.getLogger(__name__)


# What a cell made only of PART observations is worth, when the registry does
# not say what the benchmark consists of.
#
# A source that publishes only per-subject rows (35 HELM MMLU subjects, 315
# AIR-Bench categories) never stated a benchmark total. Two answers:
#
#   "pooled_parts" — (current) pool the parts exactly as the pipeline always
#                    pooled them, and label the value so the page says it is a
#                    pooling of parts rather than something the source
#                    published. No page that had a number loses it.
#   "none"         — show nothing at all.
#
# Neither is the same as inventing a total: whole observations always win when
# the cell has any, so a source's own figure is never diluted by the rows
# underneath it, and a complete registry task set still produces a mean
# (`_materialise_slice_parent_rows`) rather than a pool.
AGGREGATE_LESS_CELL_LEVEL = "pooled_parts"


def _cell_level_tail_sql() -> str:
    """The `value_level` branches below `whole`, in precedence order.

    `single` — a cell holding exactly one row, whose value is that row's own
    number and a pooling of nothing — is a statement about ARITHMETIC, so it
    outranks the parts-only label: calling one row a pooling of parts
    describes an average that never happened. It must not outrank the
    parts-only SWITCH when that switch is set to blank, though; one part is
    no more a benchmark total than five are. So the order flips with the
    switch, and `AGGREGATE_LESS_CELL_LEVEL = "none"` keeps blanking every
    parts-only cell including the one-row ones.
    """
    parts_only = (
        "WHEN NOT BOOL_OR(NOT is_part AND score IS NOT NULL) "
        f"THEN '{AGGREGATE_LESS_CELL_LEVEL}'"
    )
    single = "WHEN COUNT(*) = 1 THEN 'single'"
    order = (
        [parts_only, single]
        if AGGREGATE_LESS_CELL_LEVEL == "none"
        else [single, parts_only]
    )
    return "\n                       ".join([*order, "ELSE 'pooled'"])


# How a shown number was computed from its inputs, published on
# `eval_results_view.value_aggregation`:
#
#   NULL            the value is one row's own number, or there is none
#   'median'        several SUBMITTED readings of one quantity at one level —
#                   MMLU's three answer-extraction totals, MATH's two lm-eval
#                   tasks. Repeated measurements of the same thing, so the
#                   middle one represents them.
#   'mean' /        the pipeline combined the PARTS of a benchmark into a
#   'weighted_mean' whole. An average, not a middle: a suite score is the
#                   average of its tasks, weighted by how many samples each
#                   task covers when every one of them says.
AGG_MEDIAN = "median"
AGG_MEAN = "mean"
AGG_WEIGHTED_MEAN = "weighted_mean"


# ---------------------------------------------------------------------------
# MODEL DEVELOPER name-pattern fallback.
#
# Applied in Stage G to orgless model rows — display-name strings that have
# no `org/` slug prefix (e.g. `chatgpt-4o-latest-2025-01-30`,
# `claude-3-5-opus-20240229`, `Qwen2-0.5B-Instruct`). For these we infer the
# developer org by regex on the lowercased model_key. Each tuple is
# `(case-insensitive regex, canonical_orgs.id)`. The org_id MUST exist in
# `seed/orgs.yaml`; misses are silent (the join produces no row and the
# COALESCE chain falls through to the next fallback).
#
# Order does not matter here — the SQL CASE picks the first matching pattern,
# but patterns are intentionally non-overlapping. Keep narrowest patterns
# (e.g. `^o[1-4][-_]`) tight enough not to match unrelated prefixes.
#
# This is intentionally MODEL-DEVELOPER only. Eval-provider org inference
# (HELM, LLM Stats, …) lives in a separate pathway via
# `evals_view.evaluator_names` / `eval_results_view.reporting_orgs`.
# ---------------------------------------------------------------------------
MODEL_DEVELOPER_NAME_PATTERNS: list[tuple[str, str]] = [
    # OpenAI
    (r'^chatgpt[-_]', 'openai'),
    (r'^gpt[-_]', 'openai'),
    (r'^o[1-4][-_]', 'openai'),
    # Anthropic
    (r'^claude', 'anthropic'),
    (r'^anthropic[-_ ]', 'anthropic'),
    # Google
    (r'^gemini', 'google'),
    (r'^gemma', 'google'),
    (r'^palm[-_]', 'google'),
    (r'^bison', 'google'),
    (r'^bard', 'google'),
    # xAI
    (r'^grok', 'xai'),
    # Cohere
    (r'^cohere[ _-]', 'cohere'),
    (r'^command[-_]r', 'cohere'),
    (r'^aya[-_]', 'cohere'),
    # DeepSeek
    (r'^deepseek', 'deepseek'),
    # Alibaba (Qwen)
    (r'^qwen', 'alibaba'),
    # Meta
    (r'^llama[-_ ]?[0-9]', 'meta'),
    (r'^opt[-_][0-9]', 'meta'),
    (r'^galactica', 'meta'),
    # Mistral AI
    (r'^mistral', 'mistralai'),
    (r'^mixtral', 'mistralai'),
    (r'^codestral', 'mistralai'),
    # Microsoft
    (r'^phi[-_]', 'microsoft'),
    (r'^wizardlm', 'microsoft'),
    (r'^wizardcoder', 'microsoft'),
    # IBM Granite
    (r'^granite[-_]', 'ibm-granite'),
    # 01.AI
    (r'^yi[-_][0-9]', '01-ai'),
    # Nous Research
    (r'^openhermes', 'nous-research'),
    (r'^hermes[-_][0-9]', 'nous-research'),
    # Perplexity
    (r'^perplexity', 'perplexity'),
    (r'^sonar[-_]', 'perplexity'),
    # Allen AI
    (r'^olmo', 'allenai'),
    (r'^tulu', 'allenai'),
    # NVIDIA
    (r'^nemotron', 'nvidia'),
    # ZAI / Zhipu
    (r'^chatglm', 'zai'),
    (r'^glm[-_][0-9]', 'zai'),
    # MiniMax
    (r'^minimax', 'minimax'),
    (r'^abab[-_]', 'minimax'),
    # StepFun
    (r'^step[-_][0-9]', 'stepfun'),
    # TII (UAE)
    (r'^falcon[-_]', 'tiiuae'),
    # Inception (Jais)
    (r'^jais', 'inception'),
    # EleutherAI
    (r'^pythia', 'eleutherai'),
    (r'^gpt-neox', 'eleutherai'),
    (r'^gpt-j', 'eleutherai'),
    # Databricks
    (r'^dbrx', 'databricks'),
    # BigScience
    (r'^bloom', 'bigscience'),
    # Upstage
    (r'^solar[-_][0-9]', 'upstage'),
    # Writer
    (r'^palmyra', 'writer'),
    # AI21
    (r'^jamba', 'ai21'),
    (r'^jurassic', 'ai21'),
    # MoonshotAI
    (r'^kimi', 'moonshotai'),
    # Stability AI
    (r'^stablelm', 'stabilityai'),
]


def _model_developer_pattern_case_sql(slug_expr: str) -> str:
    """Compile MODEL_DEVELOPER_NAME_PATTERNS into a SQL CASE expression
    that maps `slug_expr` (e.g. `um.model_key`) to a canonical org_id, or
    NULL when no pattern matches. Used in Stage G's models-dim CTE."""
    if not MODEL_DEVELOPER_NAME_PATTERNS:
        return "CAST(NULL AS VARCHAR)"
    whens = "\n".join(
        # DuckDB `regexp_matches` is case-sensitive; we lower() the input.
        # Pattern strings are SQL-escaped against single quotes (none of
        # ours contain `'`, but be defensive in case future entries do).
        f"            WHEN regexp_matches(lower({slug_expr}), "
        f"'{regex.replace(chr(39), chr(39)*2)}') THEN '{org_id}'"
        for regex, org_id in MODEL_DEVELOPER_NAME_PATTERNS
    )
    return f"CASE\n{whens}\n        END"


# ---------------------------------------------------------------------------
# Stage A drop tracking. The actual counter lives in `sources.eee` (where the
# loader writes to it); these names exist as backward-compat shims for the
# pipeline orchestrator + Stage A test fixtures that pre-date the move.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage A — typed load via pyarrow
# ---------------------------------------------------------------------------


def stage_a_load_eee(con, arrow_table: pa.Table) -> int:
    """Register a typed EEE Arrow table with DuckDB as `eee_raw`.

    Zero-copy: DuckDB reads from the Arrow buffers in place. The caller
    (`pipeline.run`) builds the table via `sources.eee.load_arrow_table`,
    which validates each record against the vendored upstream Pydantic
    models and casts to the schema derived from the JSON Schema.

    Hard-fails on NULL `source_config`: downstream Stage D's
    composite_slug fallback regex evaluates to NULL on NULL input, which
    causes the row to silently disappear from `composites` /
    `benchmarks` / `evals_view`. A loud failure here is better than a
    silent dropout — every EEE record is expected to carry a config
    name; a NULL means upstream contract is broken.
    """
    con.register("eee_raw_arrow", arrow_table)
    con.execute("CREATE TABLE eee_raw AS SELECT * FROM eee_raw_arrow")
    con.unregister("eee_raw_arrow")

    null_cfg_count = con.execute(
        "SELECT COUNT(*) FROM eee_raw WHERE source_config IS NULL"
    ).fetchone()[0]
    if null_cfg_count:
        sample = con.execute(
            "SELECT evaluation_id FROM eee_raw "
            "WHERE source_config IS NULL LIMIT 3"
        ).fetchall()
        raise RuntimeError(
            f"Stage A: {null_cfg_count} EEE record(s) have NULL source_config. "
            f"Downstream stages can't bucket these into a composite/benchmark. "
            f"Sample evaluation_ids: {[r[0] for r in sample]}. Either fix "
            f"upstream EEE to emit a non-null config, or add a coalesce in "
            f"the source loader if a default makes sense."
        )

    return arrow_table.num_rows


def stage_a_load_cards(con, cards: dict) -> int:
    """Stage AutoBenchmarkCards into `cards_raw_in`, then resolve card keys to
    canonical benchmark_ids into `cards_raw`.
    """
    if not cards:
        # Empty placeholder so LEFT JOINs cleanly miss.
        con.execute(
            "CREATE TABLE cards_raw (card_key VARCHAR, card JSON, "
            "benchmark_id VARCHAR, card_resolution_strategy VARCHAR)"
        )
        return 0

    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    with tmp:
        for k, v in cards.items():
            tmp.write(json.dumps({"card_key": k, "card": v}, default=str, ensure_ascii=False))
            tmp.write("\n")

    con.execute(
        f"""
        CREATE TABLE cards_raw_in AS
        SELECT * FROM read_json_auto('{tmp.name}', format = 'newline_delimited',
                                      union_by_name = true,
                                      maximum_object_size = 268435456)
        """
    )

    con.execute(
        """
        CREATE TABLE cards_resolved AS
        SELECT
            card_key,
            card,
            resolve_canonical_id(card_key, 'benchmark', NULL) AS benchmark_id,
            resolve_strategy(card_key, 'benchmark', NULL)     AS card_resolution_strategy
        FROM cards_raw_in
        WHERE card_key IS NOT NULL
        """
    )

    # Dedupe per benchmark_id — multiple card_keys can resolve to the same
    # canonical benchmark (e.g. dataset alias and registered name). Without
    # dedup the LEFT JOIN at Stage D / G fans out fact rows.
    # First-seen-by-card_key wins (deterministic via ORDER BY card_key).
    con.execute(
        """
        CREATE TABLE cards_raw AS
        SELECT card_key, card, benchmark_id, card_resolution_strategy
        FROM (
            SELECT *,
                row_number() OVER (
                    PARTITION BY benchmark_id ORDER BY card_key
                ) AS _rn
            FROM cards_resolved
            WHERE benchmark_id IS NOT NULL
        )
        WHERE _rn = 1

        UNION ALL BY NAME

        -- Cards whose key didn't resolve (benchmark_id IS NULL) are kept as
        -- orphans for the triage path. They never join to fact_results because
        -- benchmark_id IS NULL on both sides.
        SELECT card_key, card, benchmark_id, card_resolution_strategy
        FROM cards_resolved
        WHERE benchmark_id IS NULL
        """
    )

    # Surface the collision count so the operator knows when two card files
    # are competing for the same canonical benchmark (one wins, one is dropped
    # silently from JOINs). Aggregate; per-pair detail available via the
    # `cards_resolved` table for ad-hoc inspection.
    collisions = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT benchmark_id, COUNT(*) AS n
            FROM cards_resolved
            WHERE benchmark_id IS NOT NULL
            GROUP BY benchmark_id
            HAVING n > 1
        )
        """
    ).fetchone()[0]
    if collisions:
        log.warning(
            "Stage A: %d benchmark_id(s) had multiple cards resolve to them; "
            "first-by-card_key wins. Inspect cards_resolved for detail.",
            collisions,
        )

    return con.execute("SELECT count(*) FROM cards_raw").fetchone()[0]


def _table_columns(con, table_path: Path) -> set[str]:
    path = read_parquet_arg(table_path)
    rows = con.execute(
        f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{path}'))"
    ).fetchall()
    return {r[0] for r in rows}


_DIM_SCHEMAS: dict[str, list[tuple[str, str]]] = {
    # canonical_orgs
    "canonical_orgs": [
        ("id", "VARCHAR"),
        ("display_name", "VARCHAR"),
        ("parent_org_id", "VARCHAR"),
        ("website", "VARCHAR"),
        ("logo_url", "VARCHAR"),
        ("hf_org", "VARCHAR"),
        ("kind", "VARCHAR"),
        ("tags", "VARCHAR"),
        ("metadata", "VARCHAR"),
        ("review_status", "VARCHAR"),
    ],
    # canonical_models — mirrors the registry's `canonical_models` table.
    # Lineage is encoded as a typed `parents` JSON list (see decode_parents
    # in eval_entity_resolver.canonical_store) plus scalar `model_group_id`
    # / `lineage_origin_model_org_id`. Stage A derives `parent_model_id` from
    # the first `variant` edge so downstream SQL keeps a flat scalar.
    #
    # Model-resolution-rework end-state names (registry columns, renamed in
    # place, NOT duplicated): `model_group_id` is the always-present GROUP
    # key (membership; self at a group root), `model_family_id` is the
    # STRUCTURAL family-release id (the M3 family walk), `lineage_origin_org_id`
    # -> `lineage_origin_model_org_id`. Other columns: `lineage_origin_model_id`
    # (deepest non-variant ancestor id, null-at-origin), and the resolution provenance
    # enums `resolution_source` / `resolution_granularity`. `_load_dim`
    # NULL-pads any column the published parquet doesn't carry yet, so the
    # load stays backward-safe against an older registry snapshot.
    "canonical_models": [
        ("id", "VARCHAR"),
        ("display_name", "VARCHAR"),
        ("developer", "VARCHAR"),
        ("org_id", "VARCHAR"),
        ("family", "VARCHAR"),
        ("architecture", "VARCHAR"),
        ("params_billions", "DOUBLE"),
        ("parents", "VARCHAR"),
        ("model_group_id", "VARCHAR"),
        ("model_family_id", "VARCHAR"),
        ("lineage_origin_model_id", "VARCHAR"),
        ("lineage_origin_model_org_id", "VARCHAR"),
        ("resolution_source", "VARCHAR"),
        ("resolution_granularity", "VARCHAR"),
        ("open_weights", "BOOLEAN"),
        ("release_date", "VARCHAR"),
        # JSON-encoded list of modality strings (e.g. ["text","image"]) per
        # the registry's canonical_models schema. Surfaced unchanged on
        # `models` dim and unwrapped to VARCHAR[] on `models_view` /
        # `eval_results_view.model_info.modalities`.
        ("input_modalities", "VARCHAR"),
        ("output_modalities", "VARCHAR"),
        ("tags", "VARCHAR"),
        ("metadata", "VARCHAR"),
        ("review_status", "VARCHAR"),
    ],
    # canonical_benchmarks
    "canonical_benchmarks": [
        ("id", "VARCHAR"),
        ("display_name", "VARCHAR"),
        ("description", "VARCHAR"),
        ("dataset_repo", "VARCHAR"),
        ("parent_benchmark_id", "VARCHAR"),
        # Registry-declared merged-view default metric (registry.3.2+);
        # NULL-padded on older snapshots.
        ("preferred_metric_id", "VARCHAR"),
        # Registry-declared "the benchmark's PREFERRED metric is produced by
        # an LLM judge" (registry.3.3+). Gates the judge-condition fallback
        # extraction in Stage D, together with `preferred_metric_id` — the
        # flag says nothing about a benchmark's other channels, which may be
        # keyword detectors or encoder models. NULL-padded on older snapshots.
        ("preferred_metric_llm_judged", "BOOLEAN"),
        ("tags", "VARCHAR"),
        ("metadata", "VARCHAR"),
        ("review_status", "VARCHAR"),
    ],
    # benchmark_metric_folds — curated per-benchmark metric naming folds
    # (registry.3.2+): `from_metric_id` on `benchmark_id` is the same
    # measurement as `to_metric_id` under a generic name. Drives
    # `metric_id_effective`; empty table on older registry snapshots.
    "benchmark_metric_folds": [
        ("benchmark_id", "VARCHAR"),
        ("from_metric_id", "VARCHAR"),
        ("to_metric_id", "VARCHAR"),
        # Curated published-scale -> registry-scale multiplier (0.1 =
        # raw 1-10 onto a [0,1] metric). NULL = detection-based only.
        ("scale_factor", "DOUBLE"),
        # Additive term of the same conversion (registry.3.3+):
        # canonical = published * scale_factor + scale_offset. NULL = 0.
        ("scale_offset", "DOUBLE"),
        # Scope of the row (registry.3.3+). NULL = the benchmark-wide NAMING
        # rule (from -> to). A CONVERSION is only ever valid for the one
        # source whose published scale it describes, so factor/offset live on
        # source-scoped rows; sources without one keep the rename and fall
        # through to the scale classifier.
        ("source_config", "VARCHAR"),
        ("note", "VARCHAR"),
    ],
    # canonical_metrics — registry has score_type/lower_is_better/min/max
    # today; metric_kind / metric_unit are forward-looking. score_type stays
    # as-is (binary/continuous/levels), distinct from metric_kind
    # (accuracy/f1/elo/...). The hotfix UDF synthesises metric_kind /
    # metric_unit per row via a layered chain.
    "canonical_metrics": [
        ("id", "VARCHAR"),
        ("display_name", "VARCHAR"),
        ("metric_kind", "VARCHAR"),
        ("metric_unit", "VARCHAR"),
        ("score_type", "VARCHAR"),
        ("lower_is_better", "BOOLEAN"),
        ("min_score", "DOUBLE"),
        ("max_score", "DOUBLE"),
        ("metadata", "VARCHAR"),
        ("review_status", "VARCHAR"),
    ],
    # eval_harnesses
    "eval_harnesses": [
        ("id", "VARCHAR"),
        ("display_name", "VARCHAR"),
        ("version", "VARCHAR"),
        ("fork_url", "VARCHAR"),
        ("metadata", "VARCHAR"),
        ("review_status", "VARCHAR"),
    ],
    # canonical_families — multi-benchmark / multi-composite groupings.
    # Loaded into the connection so write_hierarchy can read curation
    # for the family-rooted tree.
    # Older registry snapshots that predate this table don't ship it;
    # _load_dim falls back to an empty table with this schema
    # and the hierarchy degrades gracefully (every composite becomes
    # its own singleton family).
    "canonical_families": [
        ("id", "VARCHAR"),
        ("display_name", "VARCHAR"),
        ("category", "VARCHAR"),
        ("benchmark_ids", "VARCHAR"),
        ("primary_benchmark_key", "VARCHAR"),
        ("folder_aliases", "VARCHAR"),
        ("composite_keys", "VARCHAR"),
        ("tags", "VARCHAR"),
        ("metadata", "VARCHAR"),
        ("review_status", "VARCHAR"),
    ],
    # canonical_composites — leaderboard-level groupings. Carries
    # `family_id` (FK to canonical_families.id) so write_hierarchy can
    # bucket composites under their parent family. Same back-compat
    # fallback as canonical_families.
    "canonical_composites": [
        ("id", "VARCHAR"),
        ("display_name", "VARCHAR"),
        ("category", "VARCHAR"),
        ("source_configs", "VARCHAR"),
        ("family_id", "VARCHAR"),
        ("tags", "VARCHAR"),
        ("metadata", "VARCHAR"),
        ("review_status", "VARCHAR"),
    ],
    # canonical_inference_platforms — new dim table introduced by the
    # model-resolution-rework. PK `id` is a models.dev provider slug or
    # EEE host token; rows describe the serving platform an alias maps to.
    # `aliases.inference_platform` is an FK into this table. Loaded so the
    # view layer (and downstream readers) can join platform display
    # metadata; same back-compat fallback as canonical_families — older
    # registry snapshots that predate the table get an empty table.
    # Name matches the registry's published parquet (canonical_* prefix).
    "canonical_inference_platforms": [
        ("id", "VARCHAR"),
        ("display_name", "VARCHAR"),
        ("kind", "VARCHAR"),
        ("aliases", "VARCHAR"),              # JSON-encoded list
        ("canonical_org", "VARCHAR"),
        ("variant_of", "VARCHAR"),
        ("homepage", "VARCHAR"),
        ("created_at", "VARCHAR"),
        ("updated_at", "VARCHAR"),
    ],
}


def _load_aliases_table(con, registry_root: Path | None) -> None:
    """Materialise the registry's aliases table on the DuckDB connection.

    Schema mirrors the registry parquet (raw_value / canonical_id /
    entity_type / status / ...). When the registry doesn't carry an
    aliases parquet (cold-start dev fixture), an empty table is
    created so downstream code can JOIN unconditionally.
    """
    schema = (
        ("raw_value",     "VARCHAR"),
        ("canonical_id",  "VARCHAR"),
        ("entity_type",   "VARCHAR"),
        ("status",        "VARCHAR"),
        ("source_config", "VARCHAR"),
        # Model-resolution-rework: per-alias serving platform (FK ->
        # inference_platforms.id). NULL-safe — the CAST-NULL branch below
        # handles an older aliases parquet that predates this column.
        ("inference_platform", "VARCHAR"),
    )
    ddl = ", ".join(f"{c} {t}" for c, t in schema)

    from eval_card_backend.sources import registry as _registry_src

    path = _registry_src.aliases_path(registry_root) if registry_root else None
    if path is None or not path.exists():
        con.execute(f"CREATE TABLE aliases ({ddl})")
        return

    rp = read_parquet_arg(path)
    present = _table_columns(con, path)
    select_parts = [
        f"CAST({col} AS {t}) AS {col}" if col in present
        else f"CAST(NULL AS {t}) AS {col}"
        for col, t in schema
    ]
    con.execute(
        f"CREATE TABLE aliases AS SELECT {', '.join(select_parts)} "
        f"FROM read_parquet('{rp}')"
    )


def _load_dim(con, name: str, dim_paths: dict) -> None:
    """Load one registry dim table to its spec shape, padding missing
    columns with typed NULLs. When the registry doesn't carry the dim at
    all, create an empty table with the same schema so downstream stages
    can JOIN unconditionally.

    CASTs each present column to the spec'd type so all-NULL columns
    don't poison downstream type inference. Without the cast, an upstream
    parquet with a column of all NULLs lands as INTEGER, then any
    `COALESCE(dim.col, varchar_col)` downstream binds against the wrong
    type.
    """
    schema = _DIM_SCHEMAS[name]
    if name not in dim_paths:
        ddl = ", ".join(f"{c} {t}" for c, t in schema)
        con.execute(f"CREATE TABLE {name} ({ddl})")
        return
    path = read_parquet_arg(dim_paths[name])
    present = _table_columns(con, dim_paths[name])
    select_parts = [
        f"CAST({col} AS {ddl}) AS {col}" if col in present
        else f"CAST(NULL AS {ddl}) AS {col}"
        for col, ddl in schema
    ]
    con.execute(
        f"CREATE TABLE {name} AS SELECT {', '.join(select_parts)} "
        f"FROM read_parquet('{path}')"
    )


def _derive_model_root_id(con) -> None:
    """Overwrite `canonical_models.model_group_id` with the genuine
    transitive root (identity group) for every row.

    Model-resolution-rework: the registry column that carries the
    identity-group root was renamed `root_model_id` -> `model_group_id`
    (renamed in place, NOT duplicated). This walk reads and overwrites
    that renamed column; semantics are unchanged.

    The registry populates `model_group_id` only on identity-preserving
    chains (quantized + version snapshots), where the resolver also
    collapses leaves to the group before reaching the producer. Variant
    chains — e.g. `grok-4-0407` whose `parent_model_id` is `grok-4` —
    are not collapsed by the resolver and stay visible as distinct
    canonical ids on fact rows. Without a transitive walk, signal
    grouping fragments the same identity across its variants.

    The walk alternates two edge kinds until fixed point: the registry's
    incoming `model_group_id` (identity-group root) and `parent_model_id`
    (first `variant` edge, derived earlier in this stage from the typed
    `parents` list). A model with no parent of either kind resolves to
    itself. Cycle-safe: a revisited node terminates the chain.

    The column is overwritten in place so downstream readers get one
    coherent meaning of "root model" without having to choose between
    competing columns.
    """
    rows = con.execute(
        "SELECT id, parent_model_id, model_group_id FROM canonical_models"
    ).fetchall()
    variant_parent: dict[str, str | None] = {row[0]: row[1] for row in rows}
    quant_root: dict[str, str | None] = {row[0]: row[2] for row in rows}

    def walk_to_root(start: str) -> str:
        visited = {start}
        current = start
        while True:
            # Self-edge guard (`!= current`): post model-resolution-rework the
            # registry's `model_group_id` is ALWAYS-PRESENT and equals self at
            # a group root (a self-edge), so the quant-root step must terminate
            # rather than loop. Same guard on the variant-parent step.
            qr = quant_root.get(current)
            if qr and qr != current and qr not in visited:
                visited.add(qr)
                current = qr
                continue
            vp = variant_parent.get(current)
            if vp and vp != current and vp not in visited:
                visited.add(vp)
                current = vp
                continue
            return current

    roots = [(model_id, walk_to_root(model_id)) for model_id in variant_parent]

    if not roots:
        return

    con.execute("DROP TABLE IF EXISTS _model_root_updates")
    con.execute("CREATE TEMP TABLE _model_root_updates (id VARCHAR, root VARCHAR)")
    con.executemany(
        "INSERT INTO _model_root_updates VALUES (?, ?)", roots
    )
    con.execute(
        "UPDATE canonical_models AS cm "
        "SET model_group_id = u.root "
        "FROM _model_root_updates u "
        "WHERE cm.id = u.id"
    )
    con.execute("DROP TABLE _model_root_updates")


def stage_a_backfill_open_weights(
    con,
    *,
    probe=None,
    hf_token: str | None = None,
    cache_path: Path | None = None,
) -> int:
    """Fill `canonical_models.open_weights` where the registry left it NULL,
    using the presence of a Hugging Face model repo as evidence.

    The registry curates the flag for a minority of models (3,932 of 8,815
    set in the 2026-09-20 snapshot); the rest are NULL, which the frontend
    cannot distinguish from "closed" and so cannot filter on. A model whose
    weights are published has a repo on the Hub, so a resolvable id is
    positive evidence of open weights.

    Only ever writes TRUE, and only over a NULL — a curated verdict is never
    overwritten, and a lookup that misses leaves the row NULL rather than
    asserting closed (see `sources.hf_openness` for why a miss is not
    evidence). Returns the number of rows filled.

    Runs after the registry dims are loaded so the backfill flows into the
    `models` dim and every view built from it. Never raises: an unreachable
    Hub leaves every row exactly as the registry had it.
    """
    try:
        pending = [
            row[0]
            for row in con.execute(
                "SELECT id FROM canonical_models "
                "WHERE open_weights IS NULL AND id IS NOT NULL AND id LIKE '%/%'"
            ).fetchall()
        ]
    except Exception:  # noqa: BLE001 - no canonical_models on this connection
        log.debug("open-weights backfill: canonical_models unavailable", exc_info=True)
        return 0

    if not pending:
        return 0

    try:
        if probe is None:
            from eval_card_backend.sources.hf_openness import confirm_open_weights

            confirmed = confirm_open_weights(
                pending, hf_token=hf_token, cache_path=cache_path
            )
        else:
            confirmed = probe.confirm(pending)
    except Exception as exc:  # noqa: BLE001 - probing must never fail the bake
        log.warning(
            "open-weights backfill skipped (%s: %s); %d model(s) keep the "
            "registry's NULL",
            type(exc).__name__, exc, len(pending),
        )
        return 0

    open_ids = [(model_id,) for model_id, is_open in confirmed.items() if is_open]
    if not open_ids:
        log.info("open-weights backfill: no candidates confirmed on the Hub")
        return 0

    con.execute("DROP TABLE IF EXISTS _open_weight_updates")
    con.execute("CREATE TEMP TABLE _open_weight_updates (id VARCHAR)")
    con.executemany("INSERT INTO _open_weight_updates VALUES (?)", open_ids)
    # The NULL guard is belt-and-braces: `pending` was already filtered to
    # NULL rows, but it keeps the statement correct if that ever changes.
    con.execute(
        "UPDATE canonical_models AS cm "
        "SET open_weights = TRUE "
        "FROM _open_weight_updates u "
        "WHERE cm.id = u.id AND cm.open_weights IS NULL"
    )
    filled = con.execute(
        "SELECT count(*) FROM canonical_models cm "
        "JOIN _open_weight_updates u ON cm.id = u.id"
    ).fetchone()[0]
    con.execute("DROP TABLE _open_weight_updates")

    log.info(
        "open-weights backfill: %d of %d unset model(s) confirmed open via a "
        "Hugging Face model repo; the rest stay NULL (unknown, not closed)",
        filled, len(pending),
    )
    return filled


def stage_a_load_registry(
    con,
    dim_paths: dict,
    *,
    registry_root: Path | None = None,
    taxonomy_seed_dir: Path | None = None,
) -> None:
    """Load registry dim tables. Aliases each dim's columns to the spec shape;
    where the registry doesn't carry a column yet, project NULL.

    Also derives `canonical_models.parent_model_id` from the typed
    `parents` JSON list — the registry switched from a scalar
    `parent_model_id` column to a list of typed edges, and downstream
    SQL still wants the flat scalar.

    Loads the composite/family/slice taxonomy seed (`composites.yaml`,
    `families.yaml`, `slice_overrides.yaml`) into three small tables on
    the connection (`composite_config_map`, `family_membership`,
    `slice_promotions`). Applies slice grouping to
    `canonical_benchmarks.parent_benchmark_id` so sibling benchmarks
    (e.g., `gaia` + `gaia-level-1/2/3`) share a slice parent, with the
    promotion set respected so e.g. `bfcl-live` stays a top-level
    benchmark rather than collapsing to a phantom `bfcl` stem.
    """
    from eval_card_backend.canonicalise import taxonomy
    from eval_card_backend.canonicalise.slice_grouping import (
        apply_slice_grouping,
    )

    for name in _DIM_SCHEMAS:
        _load_dim(con, name, dim_paths)

    # Loud presence check: `_load_dim` degrades a missing table/column to an
    # empty table / NULL column, which is correct for pre-registry.3.2
    # snapshots but silently disables the merged-view defaults on a
    # name/pin mismatch. Warn so a bad wiring change can't ship quietly.
    _folds = con.execute("SELECT count(*) FROM benchmark_metric_folds").fetchone()[0]
    _preferred = con.execute(
        "SELECT count(*) FROM canonical_benchmarks WHERE preferred_metric_id IS NOT NULL"
    ).fetchone()[0]
    if _folds == 0 or _preferred == 0:
        log.warning(
            "registry merged-view curation missing: benchmark_metric_folds=%d rows, "
            "preferred_metric_id set on %d benchmarks — expected non-zero on "
            "registry.3.2+; check ENTITY_REGISTRY_REVISION / published table names",
            _folds, _preferred,
        )

    # The benchmark-resolution alias table is registered as a DuckDB-side
    # source so slice_promotion (Stage C) can run its resolver replay
    # in Python without re-reading the parquet. Small table, cheap to materialise.
    _load_aliases_table(con, registry_root)

    con.execute("ALTER TABLE canonical_models ADD COLUMN parent_model_id VARCHAR")
    con.execute(
        "UPDATE canonical_models SET parent_model_id = variant_parent_id_udf(parents)"
    )

    _derive_model_root_id(con)

    _composites, _families, promotions = taxonomy.load_and_materialise(
        con, registry_root, taxonomy_seed_dir,
    )

    apply_slice_grouping(con, promote_to_benchmark=promotions)


# ---------------------------------------------------------------------------
# Stage B — explode evaluation_results[]
# ---------------------------------------------------------------------------


def _eee_raw_columns(con) -> set[str]:
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'eee_raw'"
    ).fetchall()
    return {r[0] for r in rows}


def stage_b_explode_evaluation_results(con) -> int:
    """One row per (evaluation, result_idx). result_idx is 0-based to match the registry.

    EEE arrives as a typed pyarrow Table (validated by `sources.eee.load_arrow_table`
    against the vendored Pydantic models), so every nested field has a stable
    STRUCT type and we can read it with dot notation directly.
    """
    cols = _eee_raw_columns(con)
    if "evaluation_results" not in cols:
        con.execute(
            "CREATE TABLE results_exploded AS SELECT * FROM eee_raw WHERE 0=1"
        )
        return 0

    con.execute(
        f"CREATE TABLE results_exploded AS {explode_select_sql('eee_raw')}"
    )

    con.execute(
        """
        ALTER TABLE results_exploded
        ADD COLUMN evaluation_result_id VARCHAR
        """
    )
    con.execute(
        """
        UPDATE results_exploded
        SET evaluation_result_id = COALESCE(
            evaluation_result_id_raw,
            evaluation_id || '#' || result_idx::VARCHAR
        )
        """
    )
    con.execute("ALTER TABLE results_exploded ADD COLUMN fact_id VARCHAR")
    con.execute(
        "UPDATE results_exploded "
        "SET fact_id = fact_id_udf(evaluation_id, CAST(result_idx AS INTEGER))"
    )

    return con.execute("SELECT count(*) FROM results_exploded").fetchone()[0]


def explode_select_sql(src: str) -> str:
    """The Stage B explode SELECT, parameterised on the source table.

    Shared with the collection extractors
    (`scripts/collections/*.py`), which run the same explode over their
    synthetic EEE-shaped records so the vendored `results.parquet` matches
    `results_exploded` column-for-column (collections spec).
    """
    return f"""
        SELECT
            e.evaluation_id,
            e.retrieved_timestamp,
            -- Top-level evaluation_timestamp on the EEE record:
            -- "Timestamp for when the evaluation was run" per the
            -- vendored Pydantic schema. Distinct from
            -- retrieved_timestamp (snapshot ingestion time). Carried
            -- through here so Stage D can prefer it over the scrape
            -- time when populating fact_results.evaluation_timestamp.
            e.evaluation_timestamp                                AS record_evaluation_timestamp,
            e.source_metadata,
            e.eval_library,
            e.model_info,
            e.detailed_evaluation_results,
            e.source_config,
            -- Repo-relative path of the EEE source JSON this record was
            -- read from (e.g. flat/objects/<s1>/<s2>/<uuid>.json), injected at
            -- Stage A ingestion. Carried through the pipeline so Stage J
            -- can build a deep-link back to the upstream record
            -- (eval_results_view.eee_record_url).
            e._record_path AS source_record_path,
            (idx_1based - 1) AS result_idx,
            e.evaluation_results[idx_1based].evaluation_result_id AS evaluation_result_id_raw,
            e.evaluation_results[idx_1based].evaluation_name      AS evaluation_name,
            e.evaluation_results[idx_1based].source_data          AS source_data,
            e.evaluation_results[idx_1based].metric_config        AS metric_config,
            e.evaluation_results[idx_1based].score_details        AS score_details,
            e.evaluation_results[idx_1based].generation_config    AS generation_config,
            -- Per-result evaluation_timestamp — preferred over the
            -- record-level field when the source disagrees across
            -- evaluation_results[] entries.
            e.evaluation_results[idx_1based].evaluation_timestamp AS result_evaluation_timestamp
        FROM {src} e,
             range(1, len(e.evaluation_results) + 1) AS t(idx_1based)
        WHERE e.evaluation_results IS NOT NULL
          AND len(e.evaluation_results) > 0
    """


def stage_b_count_synth_id_collisions(con) -> int:
    """Count rows whose synthesised `<evaluation_id>#<result_idx>` happens to
    equal a real `evaluation_result_id` from another EEE record.

    The synthesised id feeds `fact_id` via `fact_id_udf`, so a collision
    means two different (evaluation_id, result_idx) tuples could produce
    the same fact_id and the (snapshot_id, fact_id) primary key contract
    silently breaks. The counter surfaces in `snapshot_meta.row_counts`
    so the operator sees it before downstream consumers do; expected to
    be 0 in normal data.
    """
    return con.execute(
        """
        WITH synth AS (
            SELECT evaluation_id || '#' || result_idx::VARCHAR AS synth_id
            FROM results_exploded
            WHERE evaluation_result_id_raw IS NULL
        )
        SELECT COUNT(*) FROM synth s
        WHERE EXISTS (
            SELECT 1 FROM results_exploded r
            WHERE r.evaluation_result_id_raw = s.synth_id
        )
        """
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Stage C — resolve identity
# ---------------------------------------------------------------------------


def stage_c_resolve_identities(con) -> None:
    # Identity inputs come from struct dot notation; the typed Arrow loader
    # in `sources.eee.load_arrow_table` guarantees stable STRUCT shapes so
    # JSON-path extraction isn't needed here.
    org_raw_clean = org_display_normalize_sql('source_metadata.source_organization_name')
    con.execute(
        f"""
        CREATE TABLE results_resolved AS
        WITH raw AS (
            SELECT
                *,
                model_info.id                                                     AS _model_raw,
                -- Structured-name pre-step: for a dotted evaluation_name, the
                -- segments are probed against the benchmark vocabulary and the
                -- most specific surface form the registry knows wins
                -- (bbq.bbq.overall → bbq; MMLU.MMLU-Pro.overall → MMLU-Pro;
                -- vals_ai.mmlu_pro.biology → mmlu pro biology). NULL for every
                -- name where no segment resolves, so those fall through to the
                -- clean_eval_name concatenation unchanged.
                resolve_structured_benchmark_id(evaluation_name, source_config)   AS _benchmark_id_structured,
                COALESCE(
                    resolve_structured_benchmark_raw(evaluation_name, source_config),
                    clean_eval_name_udf(evaluation_name))                         AS _benchmark_raw,
                -- Structured-id pre-step: positionless registry-membership
                -- resolution over the namespaced metric_config.metric_id
                -- segments (lmarena.elo.overall → elo). A registry-flagged
                -- catch-all hit (raw field names like `.score`) and any
                -- ambiguity return NULL, so those rows fall through to the
                -- extract_metric path unchanged.
                resolve_structured_metric_id(metric_config.metric_id, source_config) AS _metric_id_structured,
                -- The tail of that same match: the scoring variant spelled
                -- after the segment that named the metric
                -- (`gpqa.accuracy.diamond`, `squadv2.f1.has_ans`). Aggregate
                -- markers and numeric segments are not tails, so
                -- `lmarena.elo.overall` reports none. Stage D appends it to
                -- the observation key so variants a source publishes side by
                -- side stay separate readings instead of pooling into one
                -- median. NULL for every other resolution path.
                CASE WHEN _metric_id_structured IS NOT NULL
                     THEN resolve_structured_metric_qualifier(
                              metric_config.metric_id, source_config)
                     ELSE NULL
                END                                                               AS _metric_qualifier,
                extract_metric_udf(
                    COALESCE(metric_config.evaluation_description,
                             metric_config.metric_name,
                             evaluation_name))                                    AS _metric_extracted,
                -- Direct pre-step: the record's own metric_name decides the
                -- row when it resolves to a non-catch-all metric and either
                -- the record has no description, extraction found nothing or
                -- a catch-all, or the name refines the extracted keyword
                -- ("Macro Accuracy" vs "Accuracy"). A description naming a
                -- MORE specific metric ("Task success rate" beside
                -- metric_name "Success Rate"; "final_acc" beside "Accuracy")
                -- keeps winning, as it did before the pre-step existed.
                -- (udfs.metric_name_wins; catch-all hits like "Score" never
                -- take a row.)
                metric_name_wins(metric_config.metric_name,
                                 metric_config.evaluation_description,
                                 _metric_extracted, source_config)                AS _metric_name_wins,
                -- metric_raw records the value that actually resolved: the
                -- structured id when that pre-step fired, the record's own
                -- metric_name when it won, else the extraction result. NOTE
                -- resolution_hotfixes.py matches on literal metric_raw values
                -- ('mean', 'score', 'Codegolf v2.2 benchmark'); a row whose
                -- metric_name now resolves no longer carries the extraction
                -- literal there.
                CASE WHEN _metric_id_structured IS NOT NULL
                     THEN trim(metric_config.metric_id)
                     WHEN _metric_name_wins
                     THEN trim(metric_config.metric_name)
                     ELSE _metric_extracted
                END                                                               AS _metric_raw,
                {org_raw_clean}                                                   AS _org_raw,
                -- Concatenate name + version for resolver lookup, but treat
                -- 'unknown'/empty version as no version at all. Upstream EEE
                -- writes 'unknown' verbatim when the version isn't recorded;
                -- feeding 'helm unknown' to the resolver guarantees no_match
                -- (no real registry alias covers the literal 'unknown'
                -- token). Stripping it gives the resolver a fightable string.
                trim(
                    COALESCE(eval_library.name, '')
                    || CASE
                        WHEN eval_library.version IS NULL THEN ''
                        WHEN lower(trim(eval_library.version)) IN ('', 'unknown') THEN ''
                        ELSE ' ' || eval_library.version
                    END
                )                                                                 AS _harness_raw
            FROM results_exploded
        )
        SELECT
            *,
            _model_raw      AS model_raw,
            _benchmark_raw  AS benchmark_raw,
            _metric_raw     AS metric_raw,
            _metric_qualifier AS metric_qualifier,
            _org_raw        AS org_raw,
            NULLIF(_harness_raw, '') AS harness_raw,

            resolve_canonical_id(_model_raw,     'model',     source_config) AS model_id,
            -- `model_leaf_id` is the matched canonical BEFORE any
            -- root-collapse. Identical to `model_id` for non-snapshot
            -- ids; for dated snapshots that collapse to a family
            -- pointer (e.g. `Olmo-3-1125-32B` → root `olmo-3-32b`),
            -- it carries the snapshot canonical so Stage J can read
            -- per-snapshot release_date via leaf-coalesce.
            resolve_leaf_id(_model_raw,          'model',     source_config) AS model_leaf_id,
            -- Model-resolution-rework per-row provenance, threaded from the
            -- resolver's ResolutionResult (in-process path-dep). Each
            -- carries the serving platform / how this id was minted / what
            -- granularity it resolved at, surfaced per warehouse row.
            resolve_inference_platform(_model_raw,     'model', source_config) AS inference_platform,
            resolve_resolution_source(_model_raw,      'model', source_config) AS resolution_source,
            resolve_resolution_granularity(_model_raw, 'model', source_config) AS resolution_granularity,
            COALESCE(_benchmark_id_structured,
                     resolve_canonical_id(_benchmark_raw, 'benchmark', source_config)) AS benchmark_id,
            COALESCE(_metric_id_structured,
                     resolve_canonical_id(_metric_raw, 'metric', source_config)) AS metric_id,
            resolve_canonical_id(_org_raw,       'org',       source_config) AS org_id,
            resolve_canonical_id(NULLIF(_harness_raw, ''), 'harness', source_config) AS harness_id,

            resolve_strategy(_model_raw,     'model',     source_config) AS model_resolution_strategy,
            CASE WHEN _benchmark_id_structured IS NOT NULL THEN 'benchmark_structured'
                 ELSE resolve_strategy(_benchmark_raw, 'benchmark', source_config)
            END                                                          AS benchmark_resolution_strategy,
            CASE WHEN _metric_id_structured IS NOT NULL THEN 'metric_id_structured'
                 WHEN _metric_name_wins THEN 'metric_name_direct'
                 ELSE resolve_strategy(_metric_raw, 'metric', source_config)
            END                                                          AS metric_resolution_strategy,
            resolve_strategy(_org_raw,       'org',       source_config) AS org_resolution_strategy,
            resolve_strategy(NULLIF(_harness_raw, ''), 'harness', source_config) AS harness_resolution_strategy
        FROM raw
        """
    )

    # v2-style slice promotion: for dot-notation aggregator records
    # (llm-stats, artificial-analysis, vals-ai, openeval) whose row-level
    # resolver answer collapses many distinct sub-benchmarks into the
    # source name, replay v2's bucket-then-promote logic and override
    # benchmark_id with the canonical the slice actually denotes.
    from eval_card_backend.canonicalise import slice_promotion
    slice_promotion.apply_overrides(con)

    # Fact-level hot fix: repair HELM composite-aggregate rows (metric name
    # sits in the benchmark field upstream). Runs before _apply_slice_key so
    # the helm_mmlu → `mmlu` mis-resolution never forms a bogus slice.
    from eval_card_backend.canonicalise import resolution_hotfixes
    resolution_hotfixes.fix_helm_composite_aggregates(con)
    # Scorer-wrapper fix must precede the vague-metric fix: it assigns the
    # benchmark_id that "mean" namespacing keys on (l2-bench.mean).
    resolution_hotfixes.fix_scorer_wrapper_benchmarks(con)
    resolution_hotfixes.fix_vague_metric_labels(con)
    # Must precede apply_metric_folds: the (hle, score → accuracy) fold is
    # only safe once hle's mislabelled calibration rows are off `score`.
    resolution_hotfixes.fix_hle_calibration_error(con)
    resolution_hotfixes.fix_scicode_hal_main_rate(con)

    _apply_metric_folds(con)

    _apply_slice_key(con)

    # Collection-synthetic rows are exempt from slice-key minting:
    # they carry the clean benchmark name as evaluation_name, but other
    # sources' raw strings under the same benchmark would otherwise turn
    # them into pseudo-slices. Discriminator: manifest membership — after
    # the Stage B drop, every surviving row with a member evaluation_id is
    # synthetic.
    collections_src.create_collection_tables(con)
    con.execute(
        """
        UPDATE results_resolved
        SET slice_key = NULL, slice_name = NULL
        WHERE evaluation_id IN
              (SELECT evaluation_id FROM collection_member_ids)
        """
    )


def _apply_metric_folds(con) -> None:
    """Apply the registry's curated per-benchmark metric naming folds.

    `metric_id_effective` = fold target when (benchmark_id, metric_id)
    matches the benchmark-wide NAMING row in `benchmark_metric_folds`, else
    `metric_id` unchanged. Source-scoped rows carry a scale conversion for one
    publisher, not a different name, so the rename ignores them. Raw
    `metric_id` is never overwritten — it is what the rename rule and the
    resolution hotfixes key on.

    A registry slice child is the same benchmark's data, so a naming fold
    defined for the benchmark applies to its slices: a row on a child with no
    naming row of its own for that metric inherits the PARENT's. Global-MMLU-
    Lite's 18 language rows resolve to `global-mmlu-lite-<lang>`; without
    this they kept `score` while the totals folded to `accuracy`, and the
    page forked into two metrics. A child's own fold wins; scoped
    (scale-conversion) rows are never inherited; one level only, which is
    all the registry has.
    """
    con.execute(
        "ALTER TABLE results_resolved ADD COLUMN metric_id_effective VARCHAR"
    )
    con.execute("UPDATE results_resolved SET metric_id_effective = metric_id")
    con.execute(
        """
        UPDATE results_resolved AS r
        SET metric_id_effective = f.to_metric_id
        FROM benchmark_metric_folds f
        WHERE r.benchmark_id = f.benchmark_id
          AND r.metric_id = f.from_metric_id
          AND f.source_config IS NULL
        """
    )
    inherited = con.execute(
        f"""
        SELECT cb.parent_benchmark_id, f.from_metric_id, f.to_metric_id,
               COUNT(*) AS n, COUNT(DISTINCT r.benchmark_id) AS n_children
        FROM results_resolved r
        JOIN canonical_benchmarks cb ON cb.id = r.benchmark_id
        JOIN benchmark_metric_folds f
          ON f.benchmark_id = cb.parent_benchmark_id
         AND f.from_metric_id = r.metric_id
         AND f.source_config IS NULL
        WHERE {_INHERITED_FOLD_PREDICATE}
        GROUP BY 1, 2, 3
        ORDER BY n DESC, 1, 2
        """
    ).fetchall()
    con.execute(
        f"""
        UPDATE results_resolved AS r
        SET metric_id_effective = f.to_metric_id
        FROM canonical_benchmarks cb, benchmark_metric_folds f
        WHERE cb.id = r.benchmark_id
          AND f.benchmark_id = cb.parent_benchmark_id
          AND f.from_metric_id = r.metric_id
          AND f.source_config IS NULL
          AND {_INHERITED_FOLD_PREDICATE}
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE metric_id_effective IS DISTINCT FROM metric_id"
    ).fetchone()[0]
    log.info("stage C: metric folds re-keyed %d row(s)", n)
    for parent, from_id, to_id, n_rows, n_children in inherited:
        log.info(
            "stage C: fold %s %s→%s inherited by %d row(s) on %d slice "
            "child(ren)", parent, from_id, to_id, n_rows, n_children,
        )
    _log_metric_fold_source_labels(con)


# A row inherits its parent's naming fold only when it is on a real slice child
# (a parent edge that is not the row itself) that is the parent's own data,
# and the child has no naming row of its own for that metric name. A child
# the registry marks `metadata.role = "diagnostic"` measures a DIFFERENT
# quantity under the parent's name (BFCL's format-sensitivity standard
# deviation is not an accuracy), so the parent's rename of its catch-all
# `score` does not describe it; an `aggregate` child is the same quantity
# rolled up and does inherit. Shared by the count and the UPDATE so the log
# describes exactly the rows that moved.
_INHERITED_FOLD_PREDICATE = """
              cb.parent_benchmark_id IS NOT NULL
          AND cb.parent_benchmark_id <> cb.id
          AND COALESCE(json_extract_string(cb.metadata, '$.role'), '')
              <> 'diagnostic'
          AND NOT EXISTS (
              SELECT 1 FROM benchmark_metric_folds own
              WHERE own.benchmark_id = r.benchmark_id
                AND own.from_metric_id = r.metric_id
                AND own.source_config IS NULL
          )"""


def _log_metric_fold_source_labels(con) -> None:
    """One INFO line per rename rule that fired, listing the source labels it
    swept up and their row counts.

    A rule keyed on the catch-all `score` fires for ANY source channel whose
    own metric name failed to resolve, including channels that land later via
    cron. The 2026-09-13 corpus already renames OpenEval's `haiku-llm-judge`
    (an attack-success-rate judge, inversely correlated with the refusal score)
    and `refusal-strings` (a keyword detector) onto harmbench-refusal-score
    through the harmbench rule. The log is what makes a new label visible
    before it silently joins someone else's page.
    """
    rows = con.execute(
        """
        SELECT benchmark_id, metric_id, metric_id_effective,
               -- the full metric_source_label chain: a channel that only
               -- names itself in metric_id or metric_raw must not be
               -- reported as unlabelled.
               COALESCE(
                   metric_config.additional_details['raw_metric_name'],
                   metric_config.metric_name,
                   metric_config.metric_id,
                   metric_raw,
                   '(unlabelled)'
               )        AS source_label,
               COUNT(*) AS n
        FROM results_resolved
        WHERE metric_id_effective IS DISTINCT FROM metric_id
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 2, 3, n DESC, source_label ASC
        """
    ).fetchall()
    by_rule: dict[tuple[str, str, str], list[tuple[str, int]]] = {}
    for benchmark_id, from_id, to_id, label, n in rows:
        by_rule.setdefault((benchmark_id, from_id, to_id), []).append((label, n))
    for (benchmark_id, from_id, to_id), labels in by_rule.items():
        log.info(
            "stage C: rename %s %s\u2192%s renamed %d row(s); source labels: %s",
            benchmark_id, from_id, to_id, sum(n for _, n in labels),
            ", ".join(f"{label}={n}" for label, n in labels),
        )


def _apply_slice_key(con) -> None:
    """Derive `slice_key` / `slice_name` on `results_resolved`.

    A slice is a within-benchmark subdivision that the registry collapses to
    one canonical: e.g. EEE rows with `evaluation_name` = "Abstract Algebra",
    "Anatomy", "Astronomy" all resolve to canonical_benchmark_id = `mmlu`.
    Without a slice column, downstream signals fold those rows into one
    group keyed on (model, mmlu, accuracy) and treat the natural cross-
    subject score spread as variant divergence — wrong, and headline on
    the divergence-magnitude leaderboard.

    Heuristic: the cleaned `benchmark_raw` is the slice when ≥2 distinct
    cleaned-and-normalised raws map to the same `benchmark_id` within the
    snapshot. Single-raw benchmarks get NULL — there's no slice axis to
    differentiate.

    `slice_key` is the case-insensitive normalised form ("Anatomy" and
    "anatomy" collapse to one slice). `slice_name` keeps the per-row raw
    casing for display; downstream picks a deterministic representative
    per slice_key when rendering.
    """
    con.execute("ALTER TABLE results_resolved ADD COLUMN slice_key VARCHAR")
    con.execute("ALTER TABLE results_resolved ADD COLUMN slice_name VARCHAR")
    con.execute(
        """
        UPDATE results_resolved
        SET slice_key  = LOWER(TRIM(results_resolved.benchmark_raw)),
            slice_name = results_resolved.benchmark_raw
        FROM (
            SELECT benchmark_id
            FROM results_resolved
            WHERE benchmark_id  IS NOT NULL
              AND benchmark_raw IS NOT NULL
            GROUP BY benchmark_id
            HAVING COUNT(DISTINCT LOWER(TRIM(benchmark_raw))) >= 2
        ) AS multi_slice
        WHERE results_resolved.benchmark_id  = multi_slice.benchmark_id
          AND results_resolved.benchmark_raw IS NOT NULL
        """
    )


# ---------------------------------------------------------------------------
# Stage D — flatten + join canonical dims
# ---------------------------------------------------------------------------


def stage_d_join_dims_and_flatten(con, *, strict_collections: bool = False) -> None:
    """Flatten + JOIN.

    Reads typed STRUCT fields directly via dot notation. The metric-meta
    hotfix UDF still takes a JSON string for `metric_config` because its
    internal heuristics walk JSON paths — we `to_json()` at the call site
    rather than rewriting the UDF.

    `additional_details`, `agentic_eval_config`, `eval_plan`, `eval_limits`,
    `sandbox` are emitted as JSON strings to preserve the column shape that
    downstream `fact_results.parquet` consumers expect (the upstream typed
    shapes for these are still in flux). `generation_args_json` is the
    canonical serialised form fed to `variant_key_udf` and divergence UDFs.
    """
    # Curated "was the evaluation submitted by the org that ran it" lookup,
    # keyed by evaluation_id. Materialised as a deduped view (an evaluation_id can
    # recur with a consistent value across records; bool_or collapses it to one
    # row so the LEFT JOIN below can't fan out facts). Missing file -> empty view
    # -> every row defaults to false, so the pipeline still runs without it.
    # Collection tables are Stage B outputs; create empty stand-ins when a
    # pre-collections cache (or a test harness) skipped that step.
    collections_src.create_collection_tables(con)

    validated_path = Path(__file__).resolve().parents[3] / "vendor" / "is_verified_evaluator.parquet"
    if validated_path.exists():
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW is_verified_evaluator AS
            SELECT evaluation_id, bool_or(is_verified_evaluator) AS is_verified_evaluator
            FROM read_parquet('{validated_path.as_posix()}')
            WHERE evaluation_id IS NOT NULL AND evaluation_id <> ''
            GROUP BY evaluation_id
            """
        )
    else:
        con.execute(
            "CREATE OR REPLACE TEMP VIEW is_verified_evaluator AS "
            "SELECT NULL::VARCHAR AS evaluation_id, FALSE AS is_verified_evaluator WHERE FALSE"
        )

    collection_raw_key = collections_src.collection_raw_key_sql(
        "rr.source_metadata.source_organization_name",
        "rr.source_metadata.source_name",
        "rr.eval_library.name",
        "rr.source_config",
    )
    org_token = taxonomy.org_token_sql("rr0.org_id", "rr0.org_raw")
    source_label_slug = collections_src.source_label_slug_sql(
        "rr0.source_metadata.source_name",
        "rr0.eval_library.name",
        "rr0.source_config",
    )
    config_slug = taxonomy.config_slug_sql("rr.source_config")
    _MODELS_JSON = "j.metric_config.additional_details['metric_models_json']"
    staging_body = f"""(
        WITH rr_tok AS (
            -- Composite-partition inputs: org_token is the
            -- partition key; _curated_source_slug is what a curated
            -- `source:` member matches (the source-name half of the raw
            -- collection key). Computed once here so the three
            -- precedence joins below share one definition.
            SELECT
                rr0.*,
                {org_token}          AS org_token,
                {source_label_slug}  AS _curated_source_slug
            FROM results_resolved rr0
        ),
        joined AS (
            -- LEFT JOIN dims, then call the metric-meta hotfix UDF once per row
            -- so its STRUCT result can be destructured cleanly in the outer SELECT
            -- (single UDF invocation per row, not five).
            --
            -- composite_slug joins on the curated map (registry
            -- seed/composites.yaml → canonical_composites), precedence-
            -- resolved over (config, org, source) scopes. Default fallback
            -- for non-curated rows is kebab-case(source_config); display
            -- name falls back to the leaderboard's source_name on EEE
            -- source_metadata (the human-facing label upstream actually
            -- emits) so a brand-new uncurated config still renders.
            -- Multi-org configs are re-keyed per org partition in Stage E,
            -- over the post-supersession population.
            SELECT
                rr.*,
                cb.parent_benchmark_id                                 AS _cb_parent_benchmark_id,
                cb.preferred_metric_llm_judged                         AS _cb_preferred_metric_llm_judged,
                cb.preferred_metric_id                                 AS _cb_preferred_metric_id,
                -- Registry bounds of the EFFECTIVE metric, unmasked (NULL
                -- stays NULL = 'no_bounds'), for the scale classifier below.
                -- An INFINITE bound is no bound for scale placement: a
                -- [0, inf) metric can be neither a fraction nor a percent.
                CASE WHEN isinf(cmet.min_score) THEN NULL ELSE cmet.min_score END AS _eff_min_score,
                CASE WHEN isinf(cmet.max_score) THEN NULL ELSE cmet.max_score END AS _eff_max_score,
                -- Curated published-scale conversion from the registry's
                -- rename rule, joined on the PRE-rename metric id (the rule's
                -- own key) AND the row's source. canonical = published *
                -- factor + offset. A source with no scoped rule gets no
                -- conversion and falls through to the classifier below —
                -- two publishers of one metric rarely share a scale
                -- (OpenEval's WildBench is 1-10, BenchPress's is 33-68).
                bmf.scale_factor                                       AS _scale_factor,
                bmf.scale_offset                                       AS _scale_offset,
                cm_model.parent_model_id                               AS _cm_parent_model_id,
                -- Model-resolution-rework: `model_group_id` is the
                -- always-present GROUP key. Stage A's `_derive_model_root_id`
                -- has already overwritten it with the transitive group root,
                -- so this carries the identity-group key for aggregation.
                cm_model.model_group_id                                AS _cm_model_group_id,
                -- New structural lineage fields (NULL-padded by _load_dim
                -- when an older registry snapshot doesn't ship them).
                cm_model.model_family_id                               AS _cm_model_family_id,
                cm_model.lineage_origin_model_id                       AS _cm_lineage_origin_model_id,
                c.card                                                 AS _card_payload,
                CASE WHEN c.card IS NOT NULL THEN rr.benchmark_id ELSE NULL END AS _benchmark_card_id,
                -- Curated composite claims resolve finest-first:
                -- (config, org, source) > (config, org) > (config). Each
                -- join level matches at most one map row (same-specificity
                -- overlap is a build error in taxonomy validation), so the
                -- fact grain can't fan out. Uncurated rows fall back to
                -- kebab(source_config); Stage E re-keys them when the
                -- config is multi-org over the surviving-row population.
                COALESCE(
                    ccm3.composite_slug,
                    ccm2.composite_slug,
                    ccm.composite_slug,
                    {config_slug}
                )                                                      AS _composite_slug,
                COALESCE(
                    ccm3.composite_display_name,
                    ccm2.composite_display_name,
                    ccm.composite_display_name,
                    -- Skip source_metadata.source_name when it equals
                    -- eval_library.name — that's the upstream harness
                    -- ('inspect_ai', 'helm', ...) bleeding into the
                    -- display field, not a publisher/leaderboard name.
                    -- Aggregator folders (Mercor, Vals.ai, LLM Stats, ...)
                    -- have meaningfully-distinct source_name vs harness
                    -- and survive this guard.
                    CASE
                        WHEN rr.source_metadata.source_name IS NOT NULL
                             AND rr.source_metadata.source_name = rr.eval_library.name
                        THEN NULL
                        ELSE rr.source_metadata.source_name
                    END,
                    rr.source_config
                )                                                      AS _composite_display_name,
                (ccm3.composite_slug IS NOT NULL
                 OR ccm2.composite_slug IS NOT NULL
                 OR ccm.composite_slug IS NOT NULL)                    AS _composite_curated,
                derive_metric_meta_udf(
                    to_json(rr.metric_config),
                    cmet.metric_kind, cmet.metric_unit,
                    cmet.min_score,   cmet.max_score,
                    -- Direction: the registry wins, EXCEPT when the row
                    -- landed on a catch-all bucket (`score`, `mean-score`,
                    -- `overall` — registry rows flagged `catch_all`). Those
                    -- carry a nominal higher-is-better that says nothing
                    -- about the measurement, and it was overriding submitters
                    -- who declared their own metric lower-is-better: a
                    -- toxicity rate read as if more toxicity were better. A
                    -- catch-all direction, like a NULL one, falls through to
                    -- the row's declaration.
                    CASE WHEN COALESCE(
                             CAST(json_extract(cmet.metadata, '$.catch_all')
                                  AS BOOLEAN), FALSE)
                         THEN NULL ELSE cmet.lower_is_better END,
                    rr.metric_config.metric_name,
                    cmet.score_type
                )                                                      AS _meta,
                co_org.display_name                                    AS org_display_canonical,
                -- Raw collection key: slug(org)/slug(source_name)
                -- from the raw upstream fields — registry-free so the id
                -- can't drift across registry pins. Guards: harness bleed
                -- keys on source_config; missing parts fall back to
                -- unknown/unlabeled.
                {collection_raw_key} AS _collection_raw_key
            FROM rr_tok rr
            LEFT JOIN canonical_benchmarks cb       ON cb.id = rr.benchmark_id
            LEFT JOIN canonical_models     cm_model ON cm_model.id = rr.model_id
            -- Registry metric metadata comes from the RENAMED metric
            -- (rename-at-resolution): bounds, direction and score_type all
            -- describe the measurement the row actually reports.
            LEFT JOIN canonical_metrics    cmet     ON cmet.id = COALESCE(rr.metric_id_effective, rr.metric_id)
            LEFT JOIN benchmark_metric_folds bmf
                   ON bmf.benchmark_id   = rr.benchmark_id
                  AND bmf.from_metric_id = rr.metric_id
                  AND bmf.source_config  = rr.source_config
            LEFT JOIN canonical_orgs       co_org   ON co_org.id = rr.org_id
            LEFT JOIN cards_raw            c        ON c.benchmark_id = rr.benchmark_id
            LEFT JOIN composite_config_map ccm
                   ON ccm.source_config = rr.source_config
                  AND ccm.specificity = 1
            LEFT JOIN composite_config_map ccm2
                   ON ccm2.source_config = rr.source_config
                  AND ccm2.specificity = 2
                  AND ccm2.org_token = rr.org_token
            LEFT JOIN composite_config_map ccm3
                   ON ccm3.source_config = rr.source_config
                  AND ccm3.specificity = 3
                  AND ccm3.org_token = rr.org_token
                  AND ccm3.source_slug = rr._curated_source_slug
        ),
        judge_typed AS (
            -- Judge condition, source 1 (D9): the typed `llm_scoring` struct.
            -- `model_info.id` is required by the vendored EEE schema, so a
            -- judge that reaches Stage D always has an id — no name fallback.
            -- Resolve first, THEN dedupe and sort, so two raw spellings of one
            -- judge collapse to a single canonical entry.
            SELECT
                j.fact_id,
                list_sort(list_distinct(list_transform(
                    list_filter(
                        list_transform(j.metric_config.llm_scoring.judges,
                                       jc -> jc.model_info.id),
                        x -> x IS NOT NULL AND trim(x) <> ''
                    ),
                    x -> COALESCE(resolve_canonical_id(x, 'judge_model', j.source_config), x)
                ))) AS judges
            FROM joined j
            WHERE j.metric_config.llm_scoring IS NOT NULL
        ),
        judge_fallback AS (
            -- Judge condition, source 2 (D9): OpenEval's stringified
            -- `metric_models_json`. The registry flag is about the
            -- benchmark's PREFERRED metric only, so the gate is the row's
            -- effective metric being that metric AND the flag being TRUE.
            -- Anything else — an encoder backbone (CNN/DailyMail's DeBERTa),
            -- a keyword detector published beside a judged channel — never
            -- reaches the model resolver and never pollutes its miss
            -- counters. Malformed JSON yields no condition and is counted
            -- (see the diagnostic below), never fatal; `[]` is no judges.
            SELECT
                j.fact_id,
                CASE
                    WHEN {json_array_guard_sql(_MODELS_JSON)}
                    THEN list_sort(list_distinct(list_transform(
                        list_filter(
                            json_extract_string({_MODELS_JSON}, '$[*]'),
                            x -> x IS NOT NULL AND trim(x) <> ''
                        ),
                        x -> COALESCE(resolve_canonical_id(x, 'judge_model', j.source_config), x)
                    )))
                    ELSE NULL
                END AS judges
            FROM joined j
            WHERE {_judge_gate_sql(
                       "j._cb_preferred_metric_llm_judged",
                       "j._cb_preferred_metric_id",
                       "j.metric_id_effective", "j.metric_id")}
              AND {_MODELS_JSON} IS NOT NULL
        ),
        flat AS (
        SELECT
            j.fact_id,
            j.evaluation_id, j.result_idx, j.evaluation_result_id,
            -- Carried into Stage E so the fact_id dedup tie-break can keep the
            -- latest record. Stage F.4 EXCLUDEs it before emitting fact_results.
            j.retrieved_timestamp,

            -- evaluation_timestamp = "when did the eval actually run".
            -- Sourced strictly from EEE's evaluation_timestamp field
            -- (per EEE Pydantic schema): per-result entry first, then
            -- the record-level top-level field. NULL when EEE carries
            -- neither — the snapshot ingestion time
            -- (retrieved_timestamp) is intentionally NOT used as a
            -- fallback because it conflates "when the eval ran" with
            -- "when our pipeline scraped it".
            COALESCE(
                NULLIF(j.result_evaluation_timestamp, ''),
                NULLIF(j.record_evaluation_timestamp, '')
            )                                                                            AS evaluation_timestamp,
            -- benchmark_updated = when the source-of-truth (e.g.
            -- Vals.ai) last refreshed the benchmark itself. Carried on
            -- some EEE records via
            -- source_metadata.additional_details.benchmark_updated;
            -- distinct semantic from evaluation_timestamp (which is
            -- per-eval-run) and retrieved_timestamp (pipeline scrape).
            -- Emitted as its own column so consumers don't have to
            -- destructure the additional_details JSON to reach it.
            NULLIF(j.source_metadata.additional_details['benchmark_updated'], '')
                                                                                          AS benchmark_updated,

            j.model_raw,     j.model_id,     j.model_leaf_id,
            -- Model-resolution-rework per-row provenance (from the resolver
            -- output on Stage C). Carried through to fact_results and the
            -- view layer so each warehouse row records its serving platform
            -- and how the model id was resolved.
            j.inference_platform,
            j.resolution_source,
            j.resolution_granularity,
            j.benchmark_raw, j.benchmark_id,
            j.slice_key,     j.slice_name,
            -- Aggregation level of the submitted number, read off the raw
            -- `evaluation_name`'s last dotted segment. Sources mark their own
            -- rollups: `…overall` is the benchmark-level figure the publisher
            -- stands behind, `…<group>_overall` a group rollup inside it
            -- (global_mmlu's `fr_overall`), anything else one leaf task.
            -- Stage J picks the coarsest level a cell actually has instead of
            -- taking a median over whatever rows happen to be there, so a
            -- page shows the publisher's own aggregate rather than a number
            -- nobody reported. Resolution is untouched by this.
            CASE
                WHEN lower(trim(split_part(
                         COALESCE(j.evaluation_name, ''), '.', -1))) = 'overall'
                    THEN 'root'
                WHEN ends_with(lower(trim(split_part(
                         COALESCE(j.evaluation_name, ''), '.', -1))), '_overall')
                    THEN 'subgroup'
                ELSE 'leaf'
            END                                                                          AS aggregate_level,
            -- WHAT the row measured, relative to the benchmark it resolved
            -- to: the WHOLE benchmark, or one PART inside it.
            --
            -- This comes from RESOLUTION, never from `slice_key`. `slice_key`
            -- is a display axis minted whenever a benchmark sees more than
            -- one raw spelling, so it lands on the total as readily as on the
            -- parts: all 186 Apertus MMLU rows carry one, the three
            -- extraction totals included. Classifying on it would leave MMLU,
            -- MATH, INCLUDE and ACPBench with no whole observation at all.
            --
            -- The structured resolver already reports the subset it found,
            -- and whether that subset is a real part or just a longer way of
            -- spelling the benchmark (an exact curated alias — RealToxicity's
            -- `.small` IS the benchmark; MMLU's `anatomy` is one subject).
            -- A row with no real subset measured the whole thing.
            --
            -- There is no second clause for "resolved to a slice child": a
            -- fact that resolves to a child is IN the child's cell, where it
            -- is that child's own whole observation. Whole-vs-part is a
            -- relation between a fact and its cell's benchmark, not a
            -- property of the fact alone — which is what lets
            -- `bfcl-v3-single-turn` be a whole of its own page and a part of
            -- BFCL-v3.
            resolve_structured_benchmark_subset(
                j.evaluation_name, j.source_config)                                      AS benchmark_subset,
            (resolve_structured_benchmark_subset(
                j.evaluation_name, j.source_config) IS NOT NULL)                         AS is_part,
            -- The same question with the third answer kept: `whole`, `part`,
            -- or `unknown`. `is_part` is the two-way split (`part` vs the
            -- rest) that decides which rows enter a value; this column
            -- records whether resolution actually CHECKED, and is what the
            -- cell's `value_level` label is built from.
            --
            -- `unknown` is every name neither path could verify. The
            -- structured path declines anything flat or spaced, which is
            -- most of HELM; those rows resolve through the plain alias
            -- index, and only its byte-exact answer is a verification: the
            -- registry spelling `MMLU All Subjects` as `mmlu`, as written,
            -- makes the row a whole of `mmlu`, while `Anatomy` scoped to
            -- HELM lands on the slice child `mmlu-anatomy` and is the whole
            -- of THAT cell. A normalized or fuzzy hit says two spellings
            -- probably mean one benchmark and nothing about what the row
            -- measured, so a cell made of those is labelled a pooling. The
            -- exact hit must name the benchmark the row was resolved to
            -- (hotfixes and slice promotion can move a row afterwards).
            resolve_benchmark_observation_role(
                j.evaluation_name, j.source_config, j.benchmark_id)                      AS observation_role,
            -- Which split of the dataset the run scored, as the row states
            -- it. The `flat_split` CTE below fills in a total's split from
            -- its record's part rows (the split is a property of the RUN,
            -- and sources state it on the parts and leave the total bare);
            -- what leaves this stage joins the comparability key (Stage F)
            -- and the cell grain (Stage J), so a run on `test` and a run on
            -- `train` are never compared or pooled as one.
            json_extract_string(j.source_data, '$.hf_split')                             AS split,
            j.metric_raw,    j.metric_id,
            j.org_raw,       j.org_id,
            -- De-aliased eval-provider name. When the registry has a
            -- canonical_orgs row matching `org_id` (e.g. `Ai2` and
            -- `Allen Institute for AI` both resolve to canonical
            -- `allenai`), use the canonical display_name so downstream
            -- aggregations fold the dual-named entries together. When
            -- there's no canonical match (most upstream evaluators —
            -- Vals.ai, LLM Stats, etc. — aren't in seed/orgs.yaml and
            -- aren't expected to be), fall back to the raw upstream
            -- string. Computed once here so view stages can read it
            -- as a column without rejoining canonical_orgs.
            COALESCE(j.org_display_canonical, j.org_raw)                                  AS org_display,
            j.harness_raw,   j.harness_id,

            -- Aggregation keys: addressable-or-canonical-or-raw. Used by
            -- group signals (Stage F) and the view layer (Stage J) so
            -- unresolved rows still pool by raw string and variants
            -- collapse to their transitive root for headline aggregation.
            -- `model_aggregation_key` collapses the variant chain via
            -- `model_group_id` (the always-present GROUP key); `model_id`
            -- and `model_key` stay variant-level for per-row
            -- addressability. Critically this groups via the GROUP id, not
            -- the (now leaf) canonical_id, so post-flip aggregation is
            -- unchanged.
            COALESCE(j._cm_model_group_id, j.model_id, j.model_raw)                      AS model_aggregation_key,
            COALESCE(j.benchmark_id, j.benchmark_raw)                                    AS benchmark_key,
            -- Rename-at-resolution: the registry's per-benchmark rename rule
            -- is already applied (Stage C `metric_id_effective`), so the
            -- grouping/ranking/link key is the renamed metric. Fact-level
            -- `metric_id` above stays PRE-rename — the rename lookup and the
            -- resolution hotfixes key on it.
            -- When the structured match reported a scoring-variant tail, it
            -- joins the key: `gpqa.accuracy.diamond` and
            -- `gpqa.accuracy.extended` are two readings of one metric, and
            -- pooling them would median two different question sets into a
            -- number neither run produced. The un-qualified identity stays
            -- available as `metric_base_key` below, which is what every
            -- registry lookup (bounds, direction, display name, the
            -- benchmark's preferred metric) reads.
            COALESCE(j.metric_id_effective, j.metric_raw)
                || COALESCE('::' || j.metric_qualifier, '')                              AS metric_key,
            -- Retained alias of metric_key so downstream SQL that binds the
            -- fold-aware name keeps working; the two are the same identity.
            COALESCE(j.metric_id_effective, j.metric_raw)
                || COALESCE('::' || j.metric_qualifier, '')                              AS metric_key_effective,
            -- The metric identity without the variant tail: the registry id
            -- to look bounds, direction, display name and the benchmark's
            -- preferred metric up by. Equal to `metric_key` for every row
            -- that carries no qualifier, which is all but the structured ids
            -- that spell one.
            COALESCE(j.metric_id_effective, j.metric_raw)                                AS metric_base_key,
            j.metric_qualifier                                                           AS metric_qualifier,
            -- The source's own label for this published number. Display and
            -- provenance only: never a join key, never aggregated at page
            -- grain (one renamed metric can carry several source labels).
            COALESCE(
                j.metric_config.additional_details['raw_metric_name'],
                j.metric_config.metric_name,
                j.metric_config.metric_id,
                j.metric_raw
            )                                                                            AS metric_source_label,

            j._cb_parent_benchmark_id                                                   AS parent_benchmark_id,
            j._cm_parent_model_id                                                       AS parent_model_id,
            -- Identity-group root from the registry. The always-present GROUP
            -- key on fact_results is `model_aggregation_key` (above); this
            -- registry passthrough is kept only under the legacy
            -- `root_model_id` alias for back-compat. `model_family_id` is the
            -- registry's STRUCTURAL family-release id (the M3 family walk).
            j._cm_model_group_id                                                        AS root_model_id,
            j._cm_model_family_id                                                       AS model_family_id,
            j._cm_lineage_origin_model_id                                               AS lineage_origin_model_id,

            j._composite_slug                                                           AS composite_slug,
            j._composite_display_name                                                   AS composite_display_name,
            j.source_config                                                             AS source_config,
            -- Composite-partition helpers, consumed (and dropped) by
            -- Stage E's org-partition pass.
            j.org_token,
            j._curated_source_slug,
            j._composite_curated,

            j._benchmark_card_id                                                        AS benchmark_card_id,

            j.model_resolution_strategy, j.benchmark_resolution_strategy,
            j.metric_resolution_strategy, j.org_resolution_strategy,
            j.harness_resolution_strategy,

            -- score (typed STRUCT access; uncertainty paths are NULL-safe in
            -- DuckDB when the parent struct is NULL).
            j.score_details.score                                                       AS score,
            j.score_details.uncertainty.standard_error.value                            AS score_se,
            j.score_details.uncertainty.standard_deviation                              AS score_sd,
            j.score_details.uncertainty.confidence_interval.lower                       AS score_ci_lower,
            j.score_details.uncertainty.confidence_interval.upper                       AS score_ci_upper,
            j.score_details.uncertainty.confidence_interval.confidence_level            AS score_ci_level,
            CAST(j.score_details.uncertainty.num_samples AS INTEGER)                    AS n_samples,

            -- source / provenance
            -- The submitter's own declaration is the answer. Every EEE record
            -- carries `source_metadata.evaluator_relationship`, and a lab that
            -- submits its own evaluation says `first_party`; overriding that
            -- relabelled every submitter in the corpus as a third party.
            --
            -- ONE exception, scoped to the config that needs it (ported from
            -- the legacy pipeline's PARTY_OVERRIDE_LLM_STATS_FIX): EEE's
            -- llm-stats config carries the field from the aggregator's
            -- perspective, but the underlying rows are model-maker
            -- self-reports (raw_verified='false') vs aggregator-verified
            -- rescores (raw_verified='true'). Reclassify those on the row, not
            -- in the frontend, so every consumer agrees. Remove once upstream
            -- EEE emits the right value for that config directly.
            CASE
                WHEN j.source_config = 'llm-stats'
                THEN CASE
                        WHEN j.metric_config.additional_details['raw_verified'] = 'false'
                        THEN 'first_party'
                        ELSE 'third_party'
                     END
                ELSE NULLIF(TRIM(j.source_metadata.evaluator_relationship), '')
            END                                                                          AS evaluator_relationship,
            -- Curated provenance flag: was this evaluation submitted by the org
            -- that ran it (vs re-hosted from another leaderboard). Joined from the
            -- vendored lookup on evaluation_id; unmatched rows default to false.
            COALESCE(ev.is_verified_evaluator, FALSE)                                    AS is_verified_evaluator,
            j.source_metadata.source_type                                               AS source_type,
            j.source_metadata.source_organization_url                                   AS source_organization_url,
            j.eval_library.name                                                         AS eval_library_name,
            j.eval_library.version                                                      AS eval_library_version,

            -- metric meta destructured from the resolver UDF struct (see joined CTE).
            -- Layered chain: registry > EEE per-record > heuristic > NULL.
            -- The *_provenance columns surface which step of the chain produced
            -- the value; lets consumers distinguish a real metric_kind='score'
            -- from the catchall and filter rows for registry-side fixes.
            j._meta.metric_kind                                                         AS metric_kind,
            j._meta.metric_unit                                                         AS metric_unit,
            j._meta.lower_is_better                                                     AS lower_is_better,
            j._meta.min_score                                                           AS min_score,
            j._meta.max_score                                                           AS max_score,
            j._meta.metric_kind_provenance                                              AS metric_kind_provenance,
            j._meta.metric_unit_provenance                                              AS metric_unit_provenance,

            -- generation config — typed STRUCT access for scalars; nested
            -- objects (agentic config / eval plan / etc.) emitted as JSON to
            -- match the existing parquet column shape.
            j.generation_config.generation_args.temperature                              AS temperature,
            j.generation_config.generation_args.top_p                                    AS top_p,
            j.generation_config.generation_args.top_k                                    AS top_k,
            CAST(j.generation_config.generation_args.max_tokens AS INTEGER)              AS max_tokens,
            j.generation_config.generation_args.prompt_template                          AS prompt_template,
            j.generation_config.generation_args.reasoning                                AS reasoning,
            CAST(to_json(j.generation_config.generation_args.agentic_eval_config) AS VARCHAR) AS agentic_eval_config,
            CAST(to_json(j.generation_config.generation_args.eval_plan)           AS VARCHAR) AS eval_plan,
            CAST(to_json(j.generation_config.generation_args.eval_limits)         AS VARCHAR) AS eval_limits,
            CAST(to_json(j.generation_config.generation_args.sandbox)             AS VARCHAR) AS sandbox,

            CAST(to_json(j.generation_config.generation_args) AS VARCHAR)                AS generation_args_json,

            CAST(to_json(j.source_metadata.additional_details)   AS VARCHAR) AS source_additional_details,
            CAST(to_json(j.generation_config.additional_details) AS VARCHAR) AS generation_additional_details,
            CAST(to_json(j.metric_config.additional_details)     AS VARCHAR) AS metric_additional_details,

            -- Agent scaffold the run was executed under, when the source
            -- publishes one. Field contract: sources participate by emitting
            -- `model_info.additional_details.agent_name` (terminal_bench_2,
            -- exgentic, cocoabench, researchgym) or `agent_scaffold` (HAL).
            -- Distinct from `harness_raw`/`harness_id` — the scaffold is the
            -- agent loop, not the eval library that ran it.
            COALESCE(
                NULLIF(TRIM(j.model_info.additional_details['agent_name']), ''),
                NULLIF(TRIM(j.model_info.additional_details['agent_scaffold']), '')
            )                                                                            AS agent_scaffold_raw,

            -- upstream EEE record pointer (repo-relative path of the source
            -- JSON this row was exploded from; Stage J builds the HF URL).
            j.source_record_path,

            -- instance pointer, normalised to one repo-relative shape
            -- (see `instance_file_path_sql`); Stage J builds the HF URL.
            {instance_file_path_sql(
                "j.detailed_evaluation_results.file_path", "j.source_record_path"
            )}                                                                           AS instance_file_path,
            j.detailed_evaluation_results.format                                         AS instance_file_format,
            j.detailed_evaluation_results.checksum                                       AS instance_checksum,
            j.detailed_evaluation_results.hash_algorithm                                 AS instance_hash_algorithm,
            CAST(j.detailed_evaluation_results.total_rows AS INTEGER)                    AS instance_rows,

            -- Collection tagging: curated merge folds the raw key
            -- when declared; every row carries a non-NULL collection_id.
            COALESCE(cmm.collection_id, j._collection_raw_key)                           AS collection_id,
            -- Protocol point: canonical sorted-key JSON for
            -- collection-adapter synthetic rows, NULL for ordinary rows.
            cpm.protocol_condition                                                       AS protocol_condition,

            -- LLM-judge identity for this published number (D2): canonical
            -- JSON with a fixed key order and the judge list sorted and
            -- de-duplicated. NULL means "undisclosed", never "no judge".
            -- Source 1 (typed llm_scoring) outranks source 2
            -- (metric_models_json). Orthogonal to protocol_condition.
            CASE
                WHEN len(COALESCE(jt.judges, jf.judges)) > 0
                THEN CAST(to_json({{
                    'judges': COALESCE(jt.judges, jf.judges),
                    'label':  metric_source_label
                }}) AS VARCHAR)
                ELSE NULL
            END                                                                          AS judge_condition,

            j._card_payload AS card_payload,

            -- Scale-classifier inputs; dropped from the emitted table below.
            j._eff_min_score,
            j._eff_max_score,
            j._scale_factor,
            j._scale_offset
        FROM joined j
        LEFT JOIN is_verified_evaluator ev ON ev.evaluation_id = j.evaluation_id
        LEFT JOIN collection_merge_map cmm ON cmm.raw_key = j._collection_raw_key
        LEFT JOIN collection_protocol_map cpm
               ON cpm.evaluation_id = j.evaluation_id
              AND cpm.result_idx    = j.result_idx
        LEFT JOIN judge_typed    jt ON jt.fact_id = j.fact_id
        LEFT JOIN judge_fallback jf ON jf.fact_id = j.fact_id
        ),
        part_splits AS (
            -- The split a record's rows state for one benchmark: the
            -- record's own evidence about which split the run scored. The
            -- split is a property of the RUN, and one record on one
            -- benchmark is one run: Apertus stamps all 171 MMLU subject rows
            -- `validation` and leaves the three extraction totals and the
            -- twelve category rollups bare, and the same run produced all of
            -- them. Rows that resolved to a registry slice child count as
            -- parts of the parent (`_split_evidence_sql`), so a suite total
            -- beside its children's rows reads their split as well.
            {_part_splits_sql("flat")}
        ),
        flat_split AS (
            -- A row that states no split inherits the one split its record's
            -- rows on that benchmark agree on — the bare total from its
            -- parts, a bare group rollup from its siblings. Rows that
            -- disagree, or rows that state nothing, leave it unspecified
            -- (NULL); the disagreeing records are named in the log
            -- (`_log_split_inheritance`). A stated split is never
            -- overridden. `split_source` records which happened.
            SELECT f.* REPLACE (
                CASE WHEN f.split IS NULL AND ps._n_part_splits = 1
                     THEN ps._part_split
                     ELSE f.split
                END AS split
            ),
            CASE WHEN f.split IS NOT NULL THEN 'stated'
                 WHEN ps._n_part_splits = 1 THEN 'inherited'
            END AS split_source
            FROM flat f
            LEFT JOIN part_splits ps
              ON ps.evaluation_id      IS NOT DISTINCT FROM f.evaluation_id
             AND ps.source_record_path IS NOT DISTINCT FROM f.source_record_path
             AND ps.benchmark_key      IS NOT DISTINCT FROM f.benchmark_key
        ),
        scale_grp AS (
            -- Scale-suspect detection is per (source, benchmark, renamed
            -- metric) GROUP; the group max is what tells a percent-scaled
            -- publication apart from genuine fractions. Answer-feedback rows
            -- are excluded from the max so an assisted run can't flip scale
            -- detection for the pool.
            --
            -- On the BASE metric, not the variant-qualified key: a publisher
            -- reports every variant of one metric on one scale, and splitting
            -- the group by variant hides the evidence. SQuAD's no-answer arms
            -- are genuinely 0.0, and alone in a group of their own they read
            -- as fractions while the answerable arms at 44-64 read as
            -- percents.
            SELECT *,
                MAX(score) FILTER (
                    WHERE {protocol_exclusion_sql("protocol_condition")}
                ) OVER (
                    PARTITION BY composite_slug, benchmark_key, metric_base_key
                ) AS _grp_max
            FROM flat_split
        ),
        scale_class AS (
            -- Canonical-scale classification, on facts so every later stage
            -- (comparability grouping included) reads one already-canonical
            -- number. Group-suspect, then per-row-only-where-unambiguous:
            -- mixed groups (Vals.ai AIME: 99.583 percents next to genuine
            -- 0.833 fractions) convert row-by-row; the 1-1.5 band under [0,1]
            -- bounds is ambiguous and flagged, never guessed.
            SELECT *,
                CASE
                    WHEN score IS NULL THEN NULL
                    -- curated affine conversion from the registry's rename
                    -- rule: a known fact, never detected, and range-checked on
                    -- the affine RESULT rather than the multiplied score. The
                    -- registry contract allows either half alone, so a rule
                    -- with an offset and no factor is an identity-factor shift.
                    WHEN _scale_factor IS NOT NULL OR _scale_offset IS NOT NULL THEN
                        CASE
                            WHEN _eff_min_score IS NULL OR _eff_max_score IS NULL
                                THEN 'curated'
                            WHEN score * COALESCE(_scale_factor, 1)
                                 + COALESCE(_scale_offset, 0)
                                 BETWEEN _eff_min_score AND _eff_max_score
                                THEN 'curated'
                            ELSE 'flagged'
                        END
                    WHEN _eff_min_score IS NULL OR _eff_max_score IS NULL
                        THEN 'no_bounds'
                    -- fraction-bounded metric, percent-looking group
                    WHEN _eff_max_score <= 1.5 AND _grp_max > 1.5 THEN
                        CASE
                            WHEN score > 1.5
                             AND score / 100.0
                                 BETWEEN _eff_min_score AND _eff_max_score
                                THEN 'div100'
                            WHEN score
                                 BETWEEN _eff_min_score AND _eff_max_score
                                THEN 'none'
                            ELSE 'flagged'
                        END
                    -- percent-bounded metric, whole group reported as fractions
                    WHEN _eff_min_score = 0 AND _eff_max_score = 100
                     AND _grp_max <= 1.0 THEN
                        CASE
                            WHEN score BETWEEN 0 AND 1.0 THEN 'mul100'
                            ELSE 'flagged'
                        END
                    -- percent-bounded group topping out in (1, 1.5]:
                    -- ambiguous fractions-vs-tiny-percents — never guess
                    WHEN _eff_min_score = 0 AND _eff_max_score = 100
                     AND _grp_max <= 1.5 THEN 'flagged'
                    WHEN score BETWEEN _eff_min_score AND _eff_max_score
                        THEN 'none'
                    ELSE 'flagged'
                END AS scale_conversion
            FROM scale_grp
        ),
        scale_applied AS (
            -- One affine pair per row drives score AND uncertainty:
            -- score/CI endpoints take factor+offset, SE and SD take the
            -- magnitude of the factor only, sample size is untouched.
            -- 'flagged' gets no pair, so every canonical value is NULL.
            SELECT *,
                CASE scale_conversion
                    WHEN 'div100'    THEN 0.01
                    WHEN 'mul100'    THEN 100.0
                    WHEN 'curated'   THEN COALESCE(_scale_factor, 1.0)
                    WHEN 'none'      THEN 1.0
                    WHEN 'no_bounds' THEN 1.0
                    ELSE NULL
                END AS _scale_mult,
                CASE scale_conversion
                    WHEN 'curated' THEN COALESCE(_scale_offset, 0)
                    WHEN 'div100'  THEN 0.0
                    WHEN 'mul100'  THEN 0.0
                    WHEN 'none'    THEN 0.0
                    WHEN 'no_bounds' THEN 0.0
                    ELSE NULL
                END AS _scale_off
            FROM scale_class
        )
        SELECT * EXCLUDE (
            _eff_min_score, _eff_max_score, _scale_factor, _scale_offset,
            _grp_max, _scale_mult, _scale_off
        ),
            score          * _scale_mult + _scale_off AS score_canonical,
            score_se       * abs(_scale_mult)         AS score_se_canonical,
            score_sd       * abs(_scale_mult)         AS score_sd_canonical,
            score_ci_lower * _scale_mult + _scale_off AS score_ci_lower_canonical,
            score_ci_upper * _scale_mult + _scale_off AS score_ci_upper_canonical
        FROM scale_applied
    )"""
    con.execute(
        f"CREATE TABLE fact_results_staging AS "
        f"SELECT {explicit_projection_sql(con, staging_body)} FROM {staging_body}"
    )

    _log_malformed_metric_models(con)
    _log_split_inheritance(con)
    _log_flat_exact_wholes(con)

    _build_collection_keys(con, collection_raw_key, strict_collections)


def _log_flat_exact_wholes(con, top_n: int = 20) -> None:
    """Account for the flat-name rule in `resolve_benchmark_observation_role`:
    how many facts per source are `whole` on the strength of a byte-exact
    alias rather than a structured match, and — the guard — every (source,
    benchmark) whose cells hold two or more DISTINCT such spellings. Two exact
    spellings of the whole benchmark inside one (model, metric) cell are
    either a real re-spelling of the total or a category alias the registry
    still points at the parent instead of a child; the line exists so a
    reader can tell which."""
    rows = con.execute(
        """
        SELECT source_config, COUNT(*) AS n_facts,
               COUNT(DISTINCT benchmark_key) AS n_benchmarks
        FROM fact_results_staging
        WHERE observation_role = 'whole'
          AND benchmark_resolution_strategy = 'exact'
        GROUP BY 1 ORDER BY n_facts DESC, 1
        """
    ).fetchall()
    if not rows:
        return
    log.info(
        "stage D: %d flat-name fact(s) across %d source(s) are whole on a "
        "byte-exact alias of their benchmark",
        sum(r[1] for r in rows), len(rows),
    )
    for source_config, n_facts, n_benchmarks in rows[:top_n]:
        log.info("  flat exact whole: %s — %d fact(s), %d benchmark(s)",
                 source_config, n_facts, n_benchmarks)
    if len(rows) > top_n:
        log.info("  ... and %d more source(s)", len(rows) - top_n)
    multi = con.execute(
        """
        SELECT source_config, benchmark_key,
               COUNT(*) AS n_cells,
               MAX(n_spellings) AS max_spellings,
               arg_max(spellings, n_spellings) AS example
        FROM (
            SELECT source_config, benchmark_key, model_aggregation_key,
                   metric_key, protocol_condition, judge_condition, split,
                   COUNT(DISTINCT benchmark_raw) AS n_spellings,
                   list_sort(list_distinct(list(benchmark_raw))) AS spellings
            FROM fact_results_staging
            WHERE observation_role = 'whole'
              AND benchmark_resolution_strategy = 'exact'
            GROUP BY ALL
        )
        WHERE n_spellings > 1
        GROUP BY 1, 2
        ORDER BY n_cells DESC, 1, 2
        """
    ).fetchall()
    if not multi:
        return
    log.warning(
        "stage D: %d (source, benchmark) group(s) have cells holding 2+ "
        "distinct flat exact-alias spellings classified whole — a leftover "
        "category alias on the parent looks exactly like this",
        len(multi),
    )
    for source_config, benchmark_key, n_cells, max_spellings, example in multi[:top_n]:
        log.warning(
            "  multi-spelling whole: %s / %s — %d cell(s), up to %d "
            "spellings, e.g. %s",
            source_config, benchmark_key, n_cells, max_spellings,
            list(example)[:6],
        )
    if len(multi) > top_n:
        log.warning("  ... and %d more group(s)", len(multi) - top_n)


def _split_evidence_sql(src: str) -> str:
    """The rows of `src` that observed a benchmark or a part of it, keyed by
    that benchmark: every row keyed by its own benchmark (the total, the
    subject rows, the group rollups of one run), and rows that resolved to a
    registry slice child keyed a second time by the child's parent (a fact in
    a child's cell is a part of the parent, which is how TruthfulQA-
    multilingual's 31 language rows relate to its total). `split` is carried
    as stated, NULLs included, so a caller can tell "rows that disagree" from
    "rows that say nothing"."""
    return f"""
            SELECT evaluation_id, source_record_path,
                   benchmark_key AS total_benchmark_key, split, is_part
            FROM {src}
            UNION ALL
            SELECT evaluation_id, source_record_path,
                   parent_benchmark_id AS total_benchmark_key, split, TRUE
            FROM {src}
            WHERE parent_benchmark_id IS NOT NULL
              AND parent_benchmark_id <> benchmark_key"""


def _part_splits_sql(src: str) -> str:
    """Per (record, benchmark): how many distinct splits the record's rows on
    that benchmark state, and the one split when they agree. The `flat_split`
    CTE in Stage D reads `_n_part_splits = 1` as "a bare row inherits
    `_part_split`". `_n_parts` counts the rows that are parts, so a caller
    can tell a record with parts that say nothing from one with no parts."""
    return f"""
            SELECT evaluation_id, source_record_path,
                   total_benchmark_key AS benchmark_key,
                   CAST(COUNT(DISTINCT split) FILTER (WHERE split IS NOT NULL)
                        AS INTEGER)                                AS _n_part_splits,
                   CAST(COUNT(*) FILTER (WHERE is_part) AS INTEGER) AS _n_parts,
                   MAX(split)                                       AS _part_split
            FROM ({_split_evidence_sql(src)})
            GROUP BY 1, 2, 3"""


def _log_split_inheritance(con, top_n: int = 50) -> None:
    """Account for the split-inheritance rule in `flat_split`.

    One INFO line for how many rows took their split from the rest of their
    record; one INFO line per record whose rows DISAGREE about the split (its
    bare rows stay unspecified, and a reader should know which record to
    look at); one summary line for records whose bare rows sit beside part
    rows that state no split at all, which is the corpus norm and not worth
    a line each."""
    n_inherited, n_records = con.execute(
        """
        SELECT COUNT(*),
               COUNT(DISTINCT struct_pack(e := evaluation_id,
                                          p := source_record_path,
                                          b := benchmark_key))
        FROM fact_results_staging
        WHERE split_source = 'inherited'
        """
    ).fetchone()
    if n_inherited:
        log.info(
            "stage D: %d row(s) across %d record(s) inherited the split the "
            "rest of their record states for the benchmark",
            n_inherited, n_records,
        )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _split_bare_totals AS
        SELECT t.evaluation_id, t.benchmark_key, t.n_bare,
               ps._n_part_splits, ps._n_parts,
               list_sort(list_distinct(ev.splits)) AS part_splits
        FROM (
            SELECT evaluation_id, source_record_path, benchmark_key,
                   CAST(COUNT(*) AS INTEGER) AS n_bare
            FROM fact_results_staging
            WHERE split IS NULL
            GROUP BY 1, 2, 3
        ) t
        JOIN ({_part_splits_sql("fact_results_staging")}) ps
          ON ps.evaluation_id      IS NOT DISTINCT FROM t.evaluation_id
         AND ps.source_record_path IS NOT DISTINCT FROM t.source_record_path
         AND ps.benchmark_key      IS NOT DISTINCT FROM t.benchmark_key
        JOIN (
            SELECT evaluation_id, source_record_path, total_benchmark_key,
                   list(split) FILTER (WHERE split IS NOT NULL) AS splits
            FROM ({_split_evidence_sql("fact_results_staging")})
            GROUP BY 1, 2, 3
        ) ev
          ON ev.evaluation_id       IS NOT DISTINCT FROM t.evaluation_id
         AND ev.source_record_path  IS NOT DISTINCT FROM t.source_record_path
         AND ev.total_benchmark_key IS NOT DISTINCT FROM t.benchmark_key
        """
    )
    disagree = con.execute(
        "SELECT evaluation_id, benchmark_key, part_splits, n_bare "
        "FROM _split_bare_totals WHERE _n_part_splits > 1 ORDER BY 1, 2"
    ).fetchall()
    for evaluation_id, benchmark_key, part_splits, n_bare in disagree[:top_n]:
        log.info(
            "stage D: record %s / %s: rows disagree on the split %s; "
            "%d row(s) left with split unspecified",
            evaluation_id, benchmark_key, list(part_splits), n_bare,
        )
    if len(disagree) > top_n:
        log.info("stage D: ... and %d more record(s) with disagreeing "
                 "splits", len(disagree) - top_n)
    n_no_evidence = con.execute(
        "SELECT COUNT(*) FROM _split_bare_totals "
        "WHERE _n_part_splits = 0 AND _n_parts > 0"
    ).fetchone()[0]
    if n_no_evidence:
        log.info(
            "stage D: %d record(s) have bare rows beside part rows that "
            "state no split; those rows stay unspecified", n_no_evidence,
        )
    con.execute("DROP TABLE _split_bare_totals")


def _judge_gate_sql(flag: str, preferred_metric: str,
                    metric_effective: str, metric_id: str) -> str:
    """The source-2 judge gate: a row is eligible only when its effective
    metric IS the benchmark's preferred metric and the registry marks that
    metric LLM-judged. Shared by the extraction CTE and the malformed-JSON
    diagnostic so the count always describes the rows extraction looked at."""
    return (f"({flag} IS TRUE AND "
            f"COALESCE({metric_effective}, {metric_id}) = {preferred_metric})")


def _log_malformed_metric_models(con) -> None:
    """Count the judge-condition fallback rows whose `metric_models_json`
    isn't a JSON array. Those rows get a NULL judge condition rather than
    aborting the run, so the count is the only signal an upstream source has
    started emitting a shape we can't read."""
    models_json = "rr.metric_config.additional_details['metric_models_json']"
    n = con.execute(
        f"""
        SELECT COUNT(*)
        FROM results_resolved rr
        JOIN canonical_benchmarks cb ON cb.id = rr.benchmark_id
        WHERE {_judge_gate_sql("cb.preferred_metric_llm_judged",
                               "cb.preferred_metric_id",
                               "rr.metric_id_effective", "rr.metric_id")}
          AND {models_json} IS NOT NULL
          AND NOT COALESCE({json_array_guard_sql(models_json)}, FALSE)
        """
    ).fetchone()[0]
    if n:
        log.warning(
            "stage D: %d row(s) carry a malformed metric_models_json on a "
            "judged preferred metric; judge_condition left NULL for those "
            "rows", n,
        )


def _build_collection_keys(
    con, collection_raw_key: str, strict_collections: bool
) -> None:
    """Materialise `collection_keys` — one row per observed raw collection
    key with its post-merge collection_id, a representative source_name
    (stub display names, a standing decision), and a record count. Feeds the
    collections.json sidecar and the curated-match assertion.

    Observation universe is RECORD grain (`eee_raw`), not the exploded
    grain: collection membership is a record-level property, and
    sample-carrier records (no evaluation_results — e.g. the one
    Initiative-spelled AISI record) never produce exploded rows but must
    still count as observing their raw key.
    """
    con.execute(
        f"""
        CREATE TABLE collection_keys AS
        WITH keyed AS (
            SELECT
                {collection_raw_key}                     AS raw_key,
                COALESCE(rr.source_metadata.source_name,
                         rr.source_config)               AS src_name
            FROM eee_raw rr
        ),
        name_pick AS (
            -- Most-frequent source_name per raw key, tie-broken
            -- lexicographically, so stub display names are byte-stable.
            SELECT raw_key, src_name
            FROM (
                SELECT raw_key, src_name,
                       ROW_NUMBER() OVER (
                           PARTITION BY raw_key
                           ORDER BY COUNT(*) DESC, src_name ASC
                       ) AS _rk
                FROM keyed
                WHERE src_name IS NOT NULL
                GROUP BY raw_key, src_name
            )
            WHERE _rk = 1
        )
        SELECT
            k.raw_key,
            COALESCE(m.collection_id, k.raw_key) AS collection_id,
            np.src_name                          AS display_source_name,
            COUNT(*)                             AS n_rows
        FROM keyed k
        LEFT JOIN collection_merge_map m ON m.raw_key = k.raw_key
        LEFT JOIN name_pick np           ON np.raw_key = k.raw_key
        GROUP BY 1, 2, 3
        """
    )
    collections_src.assert_curated_keys_observed(
        con, collections_src.load_curated(), strict=strict_collections
    )


# ---------------------------------------------------------------------------
# Stage E — per-row signals (pass 1)
# ---------------------------------------------------------------------------


_SENTINEL_DROP_PREDICATE = """
    score = -1.0
    AND (
        -- (1) Declared scale excludes -1 explicitly: proportion/percent
        --     metrics or anything with a non-negative min_score.
        (metric_unit IS NOT NULL AND metric_unit IN ('proportion', 'percent'))
        OR (min_score IS NOT NULL AND -1.0 < min_score)
        -- (2) Inference fallback: when the metric still has NULL meta on
        --     this row (registry hadn't backfilled metric_unit yet AND
        --     the EEE per-record fields didn't carry it), look for
        --     siblings of the same canonical metric on the same
        --     benchmark that score in [0, 1] — strong indicator the
        --     metric is a proportion. Without this, HELM `-1` rows on
        --     accuracy-shaped metrics with sparse meta survive Stage E
        --     and poison divergence/avg_score downstream.
        OR (
            metric_unit IS NULL
            AND min_score IS NULL
            AND EXISTS (
                SELECT 1 FROM fact_results_staging sib
                WHERE sib.benchmark_key = fact_results_staging.benchmark_key
                  AND sib.metric_key   = fact_results_staging.metric_key
                  AND sib.score IS NOT NULL
                  AND sib.score >= 0.0
                  AND sib.score <= 1.0
            )
        )
    )
"""


def stage_e_per_row_signals(con, *, strict_composites: bool = False) -> StageEStats:
    """Compute per-row signals + apply two drop policies, in this order:

    1. **No-score drop** — `score IS NULL`. The row carries no measurement.
    2. **Sentinel drop** — `score = -1` on a metric whose declared scale
       (`metric_unit ∈ {proportion, percent}` or `min_score > -1`)
       excludes it. HELM emits `-1` as "evaluation failed / not scored";
       without this filter the negative sentinel poisons divergence +
       comparability aggregations. Rows whose declared scale could
       legitimately include `-1` (e.g. a delta or correlation metric)
       pass through untouched.
    3. **fact_id dedup** — multiple records may collide on
       `(snapshot_id, fact_id)`; keep the latest by `retrieved_timestamp`,
       tie-breaking on `evaluation_id` for determinism.

    Per-row signals computed: reproducibility gap, provenance source-type
    collapse, variant_key, score_scale_anomaly, reporting completeness.
    Completeness is per-row (3 of the 28 fields are EEE source_metadata
    that vary across reports); the UDF is invoked once per row in the
    `scored` CTE and destructured in the outer SELECT.

    After the drops, the composite org-partition pass re-keys multi-org
    configs over the surviving-row population (see
    `_apply_composite_partitions`); `strict_composites` mirrors Stage D's
    collections strictness (hard curated-member guards only on full runs).
    """
    staging_cols = {
        r[1] for r in con.execute(
            "PRAGMA table_info('fact_results_staging')"
        ).fetchall()
    }
    if "org_token" not in staging_cols:
        raise RuntimeError(
            "fact_results_staging lacks the composite-partition columns "
            "(org_token / _curated_source_slug / _composite_curated) — it was "
            "restored from a cache written before the composite-partition "
            "schema. Re-run with --from-stage D (or A) to rebuild it."
        )
    pre = con.execute("SELECT count(*) FROM fact_results_staging").fetchone()[0]
    n_dropped_no_score = con.execute(
        "SELECT count(*) FROM fact_results_staging WHERE score IS NULL"
    ).fetchone()[0]
    n_dropped_sentinel = con.execute(
        f"SELECT count(*) FROM fact_results_staging "
        f"WHERE score IS NOT NULL AND ({_SENTINEL_DROP_PREDICATE})"
    ).fetchone()[0]
    con.execute(
        f"""
        CREATE TABLE fact_results_signaled AS
        WITH base AS (
            SELECT
                *,
                temperature           IS NOT NULL  AS has_temperature,
                top_p                 IS NOT NULL  AS has_top_p,
                top_k                 IS NOT NULL  AS has_top_k,
                max_tokens            IS NOT NULL  AS has_max_tokens,
                prompt_template       IS NOT NULL  AS has_prompt_template,
                eval_plan             IS NOT NULL  AS has_eval_plan,
                eval_limits           IS NOT NULL  AS has_eval_limits,
                agentic_eval_config   IS NOT NULL  AS has_agentic_eval_config,
                -- reserved EvalCards fields (registry doesn't carry them today;
                -- defined here so completeness UDF and final fact_results column
                -- read the same source)
                CAST(NULL AS VARCHAR) AS lifecycle_status,
                CAST(NULL AS VARCHAR) AS preregistration_url,
                is_agentic_udf(benchmark_id, to_json(card_payload), generation_args_json) AS is_agentic
            FROM fact_results_staging
            WHERE score IS NOT NULL
              AND NOT ({_SENTINEL_DROP_PREDICATE})
        ),
        scored AS (
            -- One UDF call per row; destructured below. Without the CTE,
            -- DuckDB would invoke the UDF once per dereferenced field.
            -- repro_missing_fields is built here from per-field has_* flags
            -- so the rest of the SELECT can reference it without recomputing.
            SELECT base.*,
                compute_completeness_udf(
                    to_json(card_payload),
                    source_type,
                    org_raw,                         -- source_organization_name
                    evaluator_relationship,
                    lifecycle_status,
                    preregistration_url
                ) AS _completeness,
                ({_REPRO_MISSING_FIELDS_SQL}
                ) AS repro_missing_fields,
                (CASE WHEN is_agentic THEN {_REPRO_AGENTIC_COUNT} ELSE {_REPRO_BASE_COUNT} END) AS repro_required_count
            FROM base
        ),
        signaled AS (
            SELECT
                *,
                len(repro_missing_fields) > 0 AS has_reproducibility_gap,
                (repro_required_count - len(repro_missing_fields)) AS repro_populated_count,

                COALESCE(
                    CASE WHEN evaluator_relationship = 'other' THEN 'unspecified'
                         ELSE evaluator_relationship
                    END, 'unspecified'
                ) AS provenance_source_type,

                variant_key_udf(generation_args_json) AS variant_key,

                -- score_scale_anomaly: row claims a score that contradicts the
            -- metric's declared range. Two cases, OR-ed together:
            --   (1) metric_unit='proportion' but score ∉ [0,1]
            --       (handles registry-missing min/max for proportion metrics).
            --   (2) min_score/max_score declared and score falls outside.
            -- Both clauses are NULL-safe — a NULL declared bound or unit
            -- contributes FALSE, not NULL.
            (
                (metric_unit IS NOT NULL AND metric_unit = 'proportion'
                 AND (score < 0 OR score > 1))
                OR (min_score IS NOT NULL AND score < min_score)
                OR (max_score IS NOT NULL AND score > max_score)
            ) AS score_scale_anomaly,

                -- reporting completeness destructured from the `scored` CTE
                _completeness.completeness_score                   AS completeness_score,
                _completeness.total_fields_evaluated               AS completeness_total_fields_evaluated,
                _completeness.populated_count                      AS completeness_populated_count,
                _completeness.missing_required_fields              AS completeness_missing_required_fields,
                _completeness.partial_fields                       AS completeness_partial_fields
            FROM scored
        ),
        ranked AS (
            -- Dedup on (snapshot_id, fact_id): same fact_id appearing more
            -- than once is real upstream (multi-run reports of one eval);
            -- keep the latest by retrieved_timestamp, break ties on
            -- evaluation_id then evaluation_result_id so the choice is
            -- byte-stable across re-runs.
            --
            -- The evaluation_result_id + source_record_path tiebreaks are
            -- load-bearing: distinct EEE source records can collide on
            -- (evaluation_id, result_idx) — hence on fact_id, which is
            -- sha256(evaluation_id:result_idx) — while carrying identical
            -- retrieved_timestamp AND evaluation_id. Two cases seen in the
            -- corpus:
            --   • LiveBench: two records share one evaluation_id, each with
            --     its own evaluation_results[] array, distinguished by
            --     evaluation_result_id.
            --   • HF-OpenLLM: two near-duplicate record uploads carry the
            --     SAME evaluation_result_id (a synthesised
            --     <eid>#<bench>#<metric> form) and the same timestamp, with
            --     slightly different scores — tied even on
            --     evaluation_result_id.
            -- Without a fully-disambiguating final key the surviving row is
            -- arbitrary and varies run-to-run under multi-threaded scan
            -- order. source_record_path is the repo-relative path of the
            -- EEE source JSON (one file per record), unique per physical
            -- record, so it completes the total order. All tiebreaks sort
            -- after retrieved_timestamp, so latest-by-retrieved_timestamp
            -- semantics are unchanged — only genuine ties are pinned.
            -- CASE pins NULL fact_ids to rank 1 — they can't collide and
            -- shouldn't be silently merged by a NULL-collapsing PARTITION BY.
            SELECT *,
                CASE WHEN fact_id IS NULL THEN 1
                     ELSE ROW_NUMBER() OVER (
                         PARTITION BY fact_id
                         ORDER BY retrieved_timestamp DESC NULLS LAST,
                                  evaluation_id DESC,
                                  evaluation_result_id DESC,
                                  source_record_path DESC
                     )
                END AS _dedup_rank
            FROM signaled
        )
        SELECT * EXCLUDE (_dedup_rank) FROM ranked WHERE _dedup_rank = 1
        """
    )
    _apply_composite_partitions(con, strict=strict_composites)
    post = con.execute("SELECT count(*) FROM fact_results_signaled").fetchone()[0]
    pre_dedup = pre - n_dropped_no_score - n_dropped_sentinel
    n_dropped_dedup = pre_dedup - post
    if n_dropped_sentinel:
        log.warning(
            "Stage E: dropped %d row(s) on the score=-1 sentinel policy "
            "(metric scale excludes -1).",
            n_dropped_sentinel,
        )
    if n_dropped_dedup:
        log.warning(
            "Stage E: dropped %d fact_id collision(s); kept latest by "
            "retrieved_timestamp.",
            n_dropped_dedup,
        )
    return StageEStats(
        pre=pre,
        n_dropped_no_score=n_dropped_no_score,
        n_dropped_sentinel=n_dropped_sentinel,
        n_dropped_dedup=n_dropped_dedup,
        post=post,
    )


def _apply_composite_partitions(con, *, strict: bool) -> None:
    """Composite org-partition pass (notes/composite-partition-spec.md),
    applied to `fact_results_signaled` — the post-supersession
    population, so a superseded row can never split a page whose
    surviving rows are single-org.

    1. **Scoped-member guard**: a curated (config, org[, source]) member
       matching zero surviving rows is a hard error while its config
       still exists in the corpus — a drifted org/source key must not
       silently dump the study's rows back onto the automatic rule. It
       degrades to a warning when the config itself has left the corpus;
       a composite whose scoped members ALL match nothing always fails.
       `strict=False` (a --configs/--config-limit subset run) downgrades
       everything to warnings — a subset legitimately omits configs.
    2. **Multi-org predicate**: a config is multi-org iff its surviving
       rows carry >1 distinct org_token, excluding 'unknown-org' (a junk
       NULL-org row must not re-key an established page).
    3. **Re-key**: every uncurated row of a multi-org config gets
       `<config-slug>--<org_token>` — no partition keeps the bare slug —
       and a partition-scoped display name: the partition's single
       distinct source label when there is exactly one, else
       `<org display> — <source_config>`.

    The Stage D helper columns (org_token, _curated_source_slug,
    _composite_curated) are dropped here, so `fact_results` keeps its
    pre-existing shape.
    """
    # 1. Scoped-member guard.
    member_rows = con.execute(
        """
        SELECT m.composite_slug, m.source_config, m.org_token, m.source_slug,
               m.specificity, count(f.source_config) AS n_matches
        FROM composite_config_map m
        LEFT JOIN fact_results_signaled f
          ON f.source_config = m.source_config
         AND (m.specificity < 2 OR f.org_token = m.org_token)
         AND (m.specificity < 3 OR f._curated_source_slug = m.source_slug)
        GROUP BY 1, 2, 3, 4, 5
        """
    ).fetchall()
    live_configs = {
        r[0] for r in con.execute(
            "SELECT DISTINCT source_config FROM fact_results_signaled"
        ).fetchall()
    }
    problems: list[str] = []
    warnings: list[str] = []
    scoped_by_composite: dict[str, list] = {}
    for slug, cfg, org, source, specificity, n_matches in member_rows:
        if specificity < 2:
            continue
        scoped_by_composite.setdefault(slug, []).append(n_matches)
        if n_matches == 0:
            member = f"(config={cfg!r}, org={org!r}" + (
                f", source={source!r})" if source is not None else ")"
            )
            if cfg in live_configs:
                problems.append(
                    f"composite {slug!r}: scoped member {member} matches zero "
                    f"surviving rows while config {cfg!r} is in the corpus — "
                    f"drifted org/source key?"
                )
            else:
                warnings.append(
                    f"composite {slug!r}: scoped member {member} matches "
                    f"nothing and config {cfg!r} has left the corpus."
                )
    for slug, counts in scoped_by_composite.items():
        if counts and all(n == 0 for n in counts):
            problems.append(
                f"composite {slug!r}: no scoped member matches any surviving "
                f"row — the curated entry is fully detached."
            )
    for msg in warnings:
        log.warning("Stage E composite partitions: %s", msg)
    if problems:
        if strict:
            raise RuntimeError(
                "Stage E composite partitions: curated scoped members failed "
                "the match guard:\n  " + "\n  ".join(problems)
            )
        for msg in problems:
            log.warning(
                "Stage E composite partitions (non-strict subset run): %s", msg
            )

    # 2. Multi-org predicate over the surviving-row population.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _multi_org_configs AS
        SELECT source_config
        FROM fact_results_signaled
        GROUP BY source_config
        HAVING COUNT(DISTINCT org_token)
                   FILTER (WHERE org_token <> 'unknown-org') > 1
        """
    )

    # 3. Partition display names, computed over the automatic (uncurated)
    # rows of each multi-org partition. For uncurated rows
    # composite_display_name IS the guard-adjusted source label
    # (Stage D's COALESCE), so the label census reads it directly.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _partition_names AS
        WITH auto_rows AS (
            SELECT f.source_config, f.org_token,
                   f.composite_display_name, f.org_display
            FROM fact_results_signaled f
            JOIN _multi_org_configs mo USING (source_config)
            WHERE NOT f._composite_curated
        ),
        label_census AS (
            SELECT source_config, org_token,
                   COUNT(DISTINCT composite_display_name) AS n_labels,
                   MAX(composite_display_name)            AS only_label
            FROM auto_rows
            GROUP BY 1, 2
        ),
        org_display_pick AS (
            -- Most-frequent org display per partition, tie-broken
            -- lexicographically so the fallback name is byte-stable.
            SELECT source_config, org_token, org_display
            FROM (
                SELECT source_config, org_token, org_display,
                       ROW_NUMBER() OVER (
                           PARTITION BY source_config, org_token
                           ORDER BY COUNT(*) DESC, org_display ASC
                       ) AS _rk
                FROM auto_rows
                WHERE org_display IS NOT NULL AND org_display <> ''
                GROUP BY source_config, org_token, org_display
            )
            WHERE _rk = 1
        )
        SELECT
            lc.source_config,
            lc.org_token,
            -- Post-split partitions are label-homogeneous in the normal
            -- case and the label IS the source's recognizable name; a
            -- label-heterogeneous partition (one org, several
            -- publications) has no true name in the data, so the
            -- org-prefixed fallback is always correct.
            CASE WHEN lc.n_labels = 1 THEN lc.only_label
                 ELSE COALESCE(odp.org_display, 'Unknown org')
                      || ' — ' || lc.source_config
            END AS partition_display
        FROM label_census lc
        LEFT JOIN org_display_pick odp USING (source_config, org_token)
        """
    )

    config_slug = taxonomy.config_slug_sql("f.source_config")
    con.execute(
        f"""
        CREATE TABLE fact_results_partitioned AS
        SELECT f.* EXCLUDE (org_token, _curated_source_slug, _composite_curated)
               REPLACE (
            CASE WHEN mo.source_config IS NOT NULL AND NOT f._composite_curated
                 THEN {config_slug} || '--' || f.org_token
                 ELSE f.composite_slug
            END AS composite_slug,
            CASE WHEN mo.source_config IS NOT NULL AND NOT f._composite_curated
                 THEN pn.partition_display
                 ELSE f.composite_display_name
            END AS composite_display_name)
        FROM fact_results_signaled f
        LEFT JOIN _multi_org_configs mo ON mo.source_config = f.source_config
        LEFT JOIN _partition_names pn
               ON pn.source_config = f.source_config
              AND pn.org_token     = f.org_token
        """
    )
    con.execute("DROP TABLE fact_results_signaled")
    con.execute(
        "ALTER TABLE fact_results_partitioned RENAME TO fact_results_signaled"
    )

    n_multi = con.execute("SELECT count(*) FROM _multi_org_configs").fetchone()[0]
    if n_multi:
        parts = con.execute(
            "SELECT count(*) FROM _partition_names"
        ).fetchone()[0]
        log.info(
            "Stage E composite partitions: %d multi-org config(s) split into "
            "%d automatic partition(s).",
            n_multi, parts,
        )


# ---------------------------------------------------------------------------
# Stage F — group signals (pass 2)
# ---------------------------------------------------------------------------


def _divergence_rollup_sql(column: str) -> str:
    """Roll a comparability-group divergence boolean up to a coarser grain.

    Plain `BOOL_OR` ignores NULLs, so a group that could not be assessed
    sitting next to one that agreed would surface as FALSE — a claim nobody
    checked. A real divergence still wins; otherwise any contributing group
    that is not `ok` collapses the rollup to NULL.
    """
    return (
        f"CASE WHEN BOOL_OR({column}) THEN TRUE "
        f"WHEN BOOL_OR(comparability_status IS NOT NULL "
        f"AND comparability_status <> 'ok') THEN CAST(NULL AS BOOLEAN) "
        f"ELSE BOOL_OR({column}) END"
    )


def stage_f_group_signals(con, snapshot_id: str) -> int:
    """Group-level signal pass. Two distinct groupings:

      - **Provenance** (F.1) — `(model_aggregation_key, benchmark_key)`.
        Multi-source / first-party-only is a property of the model's
        reporting coverage on a benchmark; orthogonal to which metric
        or which slice. A third-party that reports any metric or slice
        on the pair counts as cross-party verification.
      - **Comparability** (F.2) — `(model_aggregation_key, benchmark_key,
        slice_key, metric_key)`. Divergence asks whether parties /
        setups disagree on the same measurement, so the group key is
        the actual measurement. Slices (e.g. MMLU subjects) are
        different measurements; folding them into one divergence
        calculation conflates natural cross-subject score spread with
        methodological disagreement.

    Both passes group on `*_key` (canonical-or-raw fallback) rather
    than the canonical-only `*_id`. This lets reports with unresolved
    benchmark or metric still pool with each other when their raw
    strings match, and collapses variant chains to a single root for
    aggregation (`grok-4-0407` and `grok-4` reports merge into one
    pool keyed on `grok-4`).

    Returns the count of comparability groups whose rows reported >1
    distinct `metric_unit`. A non-zero count means the per-group divergence
    threshold was computed against a deterministic-but-not-row-matching
    unit, and the operator should backfill the registry's metric_unit
    column for the offending canonical metric.
    """
    # F.1 — provenance, per (model_aggregation_key, benchmark_key).
    #
    # The filter excludes rows where every form of identity is missing
    # — i.e. neither resolved nor raw — which can happen if EEE
    # ships a record with no model name, evaluation_name, or metric_name.
    # Rows that have raw strings but no canonical id still pool here
    # (their raw string acts as the key).
    con.execute(
        f"""
        CREATE TABLE fact_results_grouped AS
        WITH org_normalized AS (
            SELECT *,
                {org_normalize_sql('org_raw')}
                  AS org_normalized_key
            FROM fact_results_signaled
            WHERE model_aggregation_key IS NOT NULL
              AND benchmark_key         IS NOT NULL
              AND metric_key            IS NOT NULL
        ),
        group_orgs AS (
            SELECT
                model_aggregation_key, benchmark_key,
                COUNT(DISTINCT org_normalized_key)
                  FILTER (WHERE org_normalized_key IS NOT NULL)
                  AS distinct_reporting_orgs
            FROM org_normalized
            GROUP BY 1, 2
        )
        SELECT
            o.*,
            go.distinct_reporting_orgs,
            go.distinct_reporting_orgs > 1 AS is_multi_source,
            (o.provenance_source_type = 'first_party' AND go.distinct_reporting_orgs = 1)
              AS first_party_only
        FROM org_normalized o
        JOIN group_orgs go USING (model_aggregation_key, benchmark_key)
        """
    )

    # F.2 — comparability, per (model, benchmark, slice, metric,
    # protocol, judge).
    #
    # The group's scores are `score_canonical` — every row already placed
    # on the renamed metric's scale in Stage D — and a row that could not
    # be placed (NULL canonical) is not assessable and never reaches the
    # UDFs.
    #
    # The bounds the threshold is computed against are the group's, not a
    # per-field MAX across disagreeing rows: the renamed metric's registry
    # bounds when it has a finite ordered pair, else the one record-declared
    # pair the assessable rows agree on. Disagreement (`mixed_scale`) or no
    # pair at all (`no_bounds`) means the group is NOT assessable — the UDFs
    # are not called and every divergence field stays NULL, which is how a
    # consumer tells "we could not compare" from "we compared and they
    # agree".
    #
    # `slice_key` IS NOT DISTINCT FROM in the JOIN treats NULL slice_key
    # (single-raw benchmarks) as equal — a plain `=` would drop those
    # rows since SQL NULL = NULL is unknown. GROUP BY collapses NULL
    # slice_keys to one group automatically, so the JOIN must mirror that;
    # protocol_condition and judge_condition follow the same rule.
    con.execute(
        """
        CREATE TABLE fact_results_grouped_annotated AS
        WITH group_payloads AS (
            -- protocol_condition, judge_condition and split join the
            -- comparability key: a different protocol point, a different LLM
            -- judge or a different dataset split is a different measurement,
            -- displayed unfolded, never averaged into one divergence pool.
            -- NULL (ordinary rows / undisclosed judge / unstated split)
            -- groups as one, exactly as slice_key does.
            SELECT
                model_aggregation_key, benchmark_key, slice_key, metric_key,
                protocol_condition, judge_condition, split,
                -- ORDER BY fact_id is load-bearing for determinism: the
                -- comparability UDFs that consume group_rows record
                -- `differing_setup_fields` in first-seen order and build
                -- `scores_by_organization` in row-encounter order, so an
                -- unordered array_agg (DuckDB scan order varies run-to-run)
                -- would shuffle those array elements / dict keys between
                -- runs. The divergence magnitudes/booleans are order-free,
                -- but a stable input order makes the whole pass byte-stable.
                array_agg(struct_pack(
                    fact_id                  := fact_id,
                    evaluation_id            := evaluation_id,
                    score                    := score_canonical,
                    generation_args          := generation_args_json,
                    evaluator_relationship   := evaluator_relationship,
                    source_organization_name := org_raw
                ) ORDER BY fact_id)
                  FILTER (WHERE score_canonical IS NOT NULL) AS group_rows,
                MAX(metric_kind) FILTER (WHERE metric_kind IS NOT NULL)
                  AS _metric_kind,
                -- Guard for the registry-bounds join below: an unresolved
                -- metric's `metric_key` is its raw string, which must not
                -- match a canonical id by coincidence.
                BOOL_OR(metric_id IS NOT NULL) AS _metric_resolved,
                -- The registry id behind the key, which a scoring-variant
                -- qualifier would otherwise hide from the bounds join. It is
                -- functionally determined by `metric_key`, so MAX picks the
                -- one value the group has.
                MAX(metric_base_key) AS _metric_base_key,
                -- Record-declared bounds over the ASSESSABLE rows only. A
                -- usable pair is present, finite and ordered; the COALESCE is
                -- load-bearing, because `isfinite(NULL)` is NULL and an
                -- unguarded `NOT (...)` would leave a bare row counted as
                -- neither usable nor missing. `_n_bounds_missing` counts the
                -- assessable rows that have no usable pair, so one shared pair
                -- plus a bare row still reads as partial. The status tree
                -- states its own finiteness requirement rather than trusting
                -- Stage D's sanitisation or the registry loader's validation.
                COUNT(DISTINCT struct_pack(lo := min_score, hi := max_score))
                  FILTER (WHERE score_canonical IS NOT NULL
                            AND COALESCE(isfinite(min_score), FALSE)
                            AND COALESCE(isfinite(max_score), FALSE)
                            AND max_score > min_score)     AS _n_record_bounds,
                COUNT(*) FILTER (WHERE score_canonical IS NOT NULL
                             AND NOT (COALESCE(isfinite(min_score), FALSE)
                                      AND COALESCE(isfinite(max_score), FALSE)
                                      AND max_score > min_score)) AS _n_bounds_missing,
                MAX(min_score) FILTER (WHERE score_canonical IS NOT NULL
                            AND COALESCE(isfinite(min_score), FALSE)
                            AND COALESCE(isfinite(max_score), FALSE)
                            AND max_score > min_score)     AS _record_min,
                MAX(max_score) FILTER (WHERE score_canonical IS NOT NULL
                            AND COALESCE(isfinite(min_score), FALSE)
                            AND COALESCE(isfinite(max_score), FALSE)
                            AND max_score > min_score)     AS _record_max
            FROM fact_results_grouped
            GROUP BY 1, 2, 3, 4, 5, 6, 7
        ),
        group_status AS (
            -- Registry bounds of the renamed metric win outright;
            -- otherwise the assessable rows must agree
            -- on one record-declared pair. `min_score`/`max_score` on a fact
            -- row is registry-then-record, so once the registry contributes
            -- no usable bound the row's pair IS the record's own.
            SELECT p.*,
                CASE
                    WHEN _reg_min IS NOT NULL AND _reg_max IS NOT NULL
                        THEN 'ok'
                    WHEN _n_record_bounds = 1 AND _n_bounds_missing = 0
                        THEN 'ok'
                    WHEN _n_record_bounds >= 1 THEN 'mixed_scale'
                    ELSE 'no_bounds'
                END AS comparability_status,
                CASE WHEN _reg_min IS NOT NULL AND _reg_max IS NOT NULL
                     THEN _reg_min
                     WHEN _n_record_bounds = 1 AND _n_bounds_missing = 0
                     THEN _record_min
                END AS _bound_min,
                CASE WHEN _reg_min IS NOT NULL AND _reg_max IS NOT NULL
                     THEN _reg_max
                     WHEN _n_record_bounds = 1 AND _n_bounds_missing = 0
                     THEN _record_max
                END AS _bound_max,
                -- Threshold config, built from the CHOSEN bounds alone:
                -- [0,1] is a proportion and [0,100] a percentage by
                -- construction, so thresholds.py takes its two unit-keyed
                -- bases from the interval itself. The record's own
                -- `metric_unit` label is neither read nor rewritten.
                struct_pack(
                    metric_kind := _metric_kind,
                    metric_unit := CASE
                        WHEN _bound_min = 0 AND _bound_max = 1   THEN 'proportion'
                        WHEN _bound_min = 0 AND _bound_max = 100 THEN 'percent'
                    END,
                    min_score   := _bound_min,
                    max_score   := _bound_max
                ) AS metric_config
            FROM (
                SELECT gp.*,
                    -- The renamed metric's registry bounds: `metric_key` IS
                    -- the effective metric id, so this is the same row Stage
                    -- D read when it placed `score_canonical` on scale. A
                    -- missing, non-finite (NaN or infinite) or inverted
                    -- bound is no bound — a cached dimension table never
                    -- validated by the current loader can still carry one.
                    CASE WHEN COALESCE(isfinite(cmet.min_score), FALSE)
                              AND COALESCE(isfinite(cmet.max_score), FALSE)
                              AND cmet.max_score > cmet.min_score
                         THEN cmet.min_score END AS _reg_min,
                    CASE WHEN COALESCE(isfinite(cmet.min_score), FALSE)
                              AND COALESCE(isfinite(cmet.max_score), FALSE)
                              AND cmet.max_score > cmet.min_score
                         THEN cmet.max_score END AS _reg_max
                FROM group_payloads gp
                LEFT JOIN canonical_metrics cmet
                       ON cmet.id = gp._metric_base_key AND gp._metric_resolved
            ) p
        ),
        group_annotations AS (
            -- Only an `ok` group is compared; the rest keep NULL flags,
            -- magnitudes and thresholds.
            SELECT
                model_aggregation_key, benchmark_key, slice_key, metric_key,
                protocol_condition, judge_condition, split, comparability_status,
                CASE WHEN comparability_status = 'ok' THEN
                    compute_variant_divergence_udf(group_rows, metric_config)
                END AS variant,
                CASE WHEN comparability_status = 'ok' THEN
                    compute_cross_party_divergence_udf(group_rows, metric_config)
                END AS cross_party
            FROM group_status
        )
        SELECT
            fr.*,
            -- Hash each key separately before concatenation so a `|`
            -- character inside a raw fallback string can't collide
            -- with the separator. Each inner md5 produces a fixed-
            -- width hex digest, making the concatenation unambiguous.
            md5(md5(fr.model_aggregation_key)
                || md5(fr.benchmark_key)
                || md5(COALESCE(fr.slice_key, ''))
                || md5(fr.metric_key)
                || md5(COALESCE(fr.protocol_condition, ''))
                || md5(COALESCE(fr.judge_condition, ''))
                || md5(COALESCE(fr.split, '')))
              AS comparability_group_id,
            ga.comparability_status                 AS comparability_status,
            ga.variant.has_variant_divergence       AS has_variant_divergence,
            ga.variant.divergence_magnitude         AS variant_divergence_magnitude,
            ga.variant.threshold_used               AS variant_divergence_threshold,
            ga.variant.threshold_basis              AS variant_threshold_basis,
            ga.variant.differing_setup_fields       AS variant_differing_fields,

            ga.cross_party.has_cross_party_divergence  AS has_cross_party_divergence,
            ga.cross_party.divergence_magnitude        AS cross_party_divergence_magnitude,
            ga.cross_party.threshold_used              AS cross_party_divergence_threshold,
            ga.cross_party.threshold_basis             AS cross_party_threshold_basis,
            ga.cross_party.differing_setup_fields      AS cross_party_differing_fields,
            ga.cross_party.organization_count          AS cross_party_org_count,
            ga.cross_party.scores_by_organization      AS scores_by_organization
        FROM fact_results_grouped fr
        LEFT JOIN group_annotations ga
          ON ga.model_aggregation_key = fr.model_aggregation_key
         AND ga.benchmark_key         = fr.benchmark_key
         AND ga.slice_key             IS NOT DISTINCT FROM fr.slice_key
         AND ga.metric_key            = fr.metric_key
         AND ga.protocol_condition    IS NOT DISTINCT FROM fr.protocol_condition
         AND ga.judge_condition       IS NOT DISTINCT FROM fr.judge_condition
         AND ga.split                 IS NOT DISTINCT FROM fr.split
        """
    )

    # F.4 — final fact_results: union resolved-with-group-signals + unresolved passthrough
    #
    # `model_key = COALESCE(model_id, model_raw)` is the row's
    # variant-level addressable identifier (URLs, per-variant fact
    # rows). `model_aggregation_key = COALESCE(root_model_id, model_id,
    # model_raw)` is the root-collapsed grouping key (already on the
    # row from Stage D); the view layer uses it for one-row-per-root
    # rollups while fact_results retains the variant grain.
    #
    # The unresolved passthrough now only fires when the row lacks
    # raw identity entirely (model_aggregation_key / benchmark_key /
    # metric_key all NULL) — rare, since Stage C always extracts the
    # raw strings when the source carries them. Group signals are NULL
    # on these rows because there is no identity to pool against.
    fact_body = f"""(
        SELECT
            TIMESTAMP '{snapshot_id_to_sql(snapshot_id)}' AS snapshot_id,
            * EXCLUDE (card_payload, org_normalized_key, generation_args_json,
                       _completeness),
            COALESCE(model_id, model_raw) AS model_key
        FROM fact_results_grouped_annotated

        UNION ALL BY NAME

        SELECT
            TIMESTAMP '{snapshot_id_to_sql(snapshot_id)}' AS snapshot_id,
            fr.* EXCLUDE (card_payload, generation_args_json, _completeness),
            COALESCE(fr.model_id, fr.model_raw)                AS model_key,
            CAST(NULL AS INTEGER)                              AS distinct_reporting_orgs,
            CAST(NULL AS VARCHAR)                              AS comparability_group_id,
            CAST(NULL AS BOOLEAN)                              AS is_multi_source,
            CAST(NULL AS BOOLEAN)                              AS first_party_only,
            CAST(NULL AS VARCHAR)                              AS comparability_status,
            CAST(NULL AS BOOLEAN)                              AS has_variant_divergence,
            CAST(NULL AS DOUBLE)                               AS variant_divergence_magnitude,
            CAST(NULL AS DOUBLE)                               AS variant_divergence_threshold,
            CAST(NULL AS VARCHAR)                              AS variant_threshold_basis,
            CAST(NULL AS STRUCT(field VARCHAR, "values" JSON)[]) AS variant_differing_fields,
            CAST(NULL AS BOOLEAN)                              AS has_cross_party_divergence,
            CAST(NULL AS DOUBLE)                               AS cross_party_divergence_magnitude,
            CAST(NULL AS DOUBLE)                               AS cross_party_divergence_threshold,
            CAST(NULL AS VARCHAR)                              AS cross_party_threshold_basis,
            CAST(NULL AS STRUCT(field VARCHAR, "values" JSON)[]) AS cross_party_differing_fields,
            CAST(NULL AS INTEGER)                              AS cross_party_org_count,
            CAST(NULL AS MAP(VARCHAR, DOUBLE))                 AS scores_by_organization
        FROM fact_results_signaled fr
        WHERE fr.model_aggregation_key IS NULL
           OR fr.benchmark_key         IS NULL
           OR fr.metric_key            IS NULL
    )"""
    con.execute(
        f"CREATE TABLE fact_results AS "
        f"SELECT {explicit_projection_sql(con, fact_body)} FROM {fact_body}"
    )

    # Defensive sanity check: the two UNION BY NAME arms above MUST have
    # identical column sets. UNION BY NAME silently fills NULL when one
    # arm has a column the other doesn't — that drops signal data on the
    # floor when a future field gets added to `fact_results_signaled`
    # without a matching `CAST(NULL AS …)` line in the unresolved-row
    # passthrough. Re-run the two SELECTs in isolation, compare column
    # name sets, and fail fast on drift.
    _resolved_cols = {
        r[0] for r in con.execute(
            f"DESCRIBE SELECT TIMESTAMP '{snapshot_id_to_sql(snapshot_id)}' AS snapshot_id, "
            "* EXCLUDE (card_payload, org_normalized_key, generation_args_json, _completeness), "
            "COALESCE(model_id, model_raw) AS model_key "
            "FROM fact_results_grouped_annotated"
        ).fetchall()
    }
    _passthrough_cols = {
        r[0] for r in con.execute(
            "DESCRIBE SELECT * FROM fact_results LIMIT 0"
        ).fetchall()
    }
    _drift = _resolved_cols.symmetric_difference(_passthrough_cols)
    if _drift:
        raise RuntimeError(
            f"Stage F.4 column drift between resolved and unresolved "
            f"UNION arms: {sorted(_drift)}. Add a matching "
            f"`CAST(NULL AS …) AS <col>` to the passthrough SELECT, or "
            f"add the column to the EXCLUDE list on the resolved SELECT. "
            f"UNION ALL BY NAME would otherwise silently NULL these out."
        )

    _log_comparability_status(con)

    # Operator-visible counter, at the real comparability grain: groups whose
    # rows disagree about metric_unit. The label no longer feeds the
    # threshold (that comes from the group's chosen bounds), but a group that
    # cannot agree on its own unit is a registry-backfill signal.
    n_unit_inconsistent = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT model_aggregation_key, benchmark_key, slice_key, metric_key,
                   protocol_condition, judge_condition, split
            FROM fact_results_grouped
            GROUP BY 1, 2, 3, 4, 5, 6, 7
            HAVING COUNT(DISTINCT metric_unit)
                   FILTER (WHERE metric_unit IS NOT NULL) > 1
        )
        """
    ).fetchone()[0]
    if n_unit_inconsistent:
        log.warning(
            "Stage F: %d comparability group(s) had >1 distinct metric_unit "
            "across rows. Backfill the registry's metric_unit for the "
            "offending canonical metric to silence.",
            n_unit_inconsistent,
        )
    return n_unit_inconsistent


def _log_comparability_status(con) -> None:
    """Report the groups the divergence pass could not assess, and why.

    `mixed_scale` means the assessable rows declared more than one bounds
    pair (or only some of them declared one) on a metric the registry does
    not bound; `no_bounds` means nobody declared one. Both leave every
    divergence field NULL. The examples name a stable group id and the
    distinct bounds pairs the operator has to reconcile — usually by giving
    the renamed metric registry bounds. The example names the full grouping
    grain (model, benchmark, slice, metric, protocol, judge), not just the
    benchmark/metric pair the count is taken over.
    """
    rows = con.execute(
        """
        SELECT comparability_status,
               COUNT(DISTINCT comparability_group_id)                AS n_groups,
               COUNT(*)                                              AS n_rows,
               MIN(comparability_group_id)                           AS example_group,
               arg_min(model_aggregation_key
                       || ' on ' || benchmark_key
                       || '/' || COALESCE(slice_key, '-')
                       || '/' || metric_key
                       || ' protocol=' || COALESCE(protocol_condition, '-')
                       || ' judge=' || COALESCE(judge_condition, '-')
                       || ' split=' || COALESCE(split, '-'),
                       comparability_group_id)                       AS example_pair
        FROM fact_results_grouped_annotated
        WHERE comparability_status IN ('mixed_scale', 'no_bounds')
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    for status, n_groups, n_rows, example_group, example_pair in rows:
        bounds = con.execute(
            """
            SELECT DISTINCT min_score, max_score
            FROM fact_results_grouped_annotated
            WHERE comparability_group_id = ?
            ORDER BY 1 NULLS FIRST, 2 NULLS FIRST
            """,
            [example_group],
        ).fetchall()
        log.warning(
            "Stage F: %d comparability group(s) (%d row(s)) are not "
            "assessable — %s. Example %s on %s, declared bounds: %s.",
            n_groups, n_rows, status, example_group, example_pair,
            ", ".join(f"[{lo}, {hi}]" for lo, hi in bounds),
        )


def snapshot_id_to_sql(snapshot_id: str) -> str:
    """DuckDB's TIMESTAMP literal doesn't accept the trailing 'Z'. Strip it
    and the parser does the right thing.
    """
    return snapshot_id[:-1] if snapshot_id.endswith("Z") else snapshot_id


def ts_cast_sql(column_expr: str) -> str:
    """Return the SQL expression that coerces a `retrieved_timestamp`
    value to a TIMESTAMP. EEE's schema declares the field as a
    `format: date-time` string, but in practice upstream sources also
    emit numeric Unix-epoch values (e.g. `1775549757.575894`). A bare
    `TRY_CAST(x AS TIMESTAMP)` only parses ISO strings — epoch numerics
    fail and silently land as NULL, which is what produces the
    "Updated: Unknown" rendering on every evaluation page.

    The COALESCE chain tries ISO first, then falls back to interpreting
    the value as a Unix-epoch double via `to_timestamp(...)`. NULL in
    both branches preserves NULL.
    """
    return (
        f"COALESCE("
        f"TRY_CAST({column_expr} AS TIMESTAMP), "
        f"TRY_CAST(to_timestamp(TRY_CAST({column_expr} AS DOUBLE)) AS TIMESTAMP)"
        f")"
    )


# ---------------------------------------------------------------------------
# Stage G — dim tables (benchmarks, composites, families, models)
# ---------------------------------------------------------------------------


def _synthesise_phantom_benchmarks(con) -> int:
    """Add canonical_benchmarks rows for stems referenced as
    parent_benchmark_id but missing as id.

    When ≥2 siblings share a stem and no bare-stem canonical exists
    (e.g. `caparena-*` with no bare `caparena`), the slice-grouping
    pass sets `parent_benchmark_id = caparena` on every sibling. This
    helper materialises a synthetic row with that key, display name
    derived from the longest common prefix of slice display names
    (falling back to a title-cased stem), and tags as the union of
    slice tags. Inserted into canonical_benchmarks so the rest of
    Stage G doesn't need a UNION-with-phantoms code path.

    Returns the number of synthetic rows inserted.
    """
    rows = con.execute(
        """
        WITH phantom_stems AS (
            SELECT DISTINCT cb.parent_benchmark_id AS stem
            FROM canonical_benchmarks cb
            WHERE cb.parent_benchmark_id IS NOT NULL
              AND cb.parent_benchmark_id != cb.id
              AND NOT EXISTS (
                  SELECT 1 FROM canonical_benchmarks cb2
                  WHERE cb2.id = cb.parent_benchmark_id
              )
        )
        SELECT
            ps.stem,
            ARRAY_AGG(cb.display_name ORDER BY cb.id)
                FILTER (WHERE cb.display_name IS NOT NULL) AS member_names,
            ARRAY_AGG(cb.tags ORDER BY cb.id)
                FILTER (WHERE cb.tags IS NOT NULL)         AS member_tags
        FROM phantom_stems ps
        JOIN canonical_benchmarks cb ON cb.parent_benchmark_id = ps.stem
        GROUP BY ps.stem
        """
    ).fetchall()
    if not rows:
        return 0

    # Late import to avoid a hard dep on sidecars from stages.
    from eval_card_backend.canonicalise.sidecars import (
        _common_prefix,
        _title_case_stem,
    )

    insert_rows: list[tuple[str, str, str | None]] = []
    for stem, names, tag_jsons in rows:
        names = names or []
        display = ""
        if len(names) >= 2:
            display = _common_prefix(names) or ""
        if not display or len(display) < 2:
            display = _title_case_stem(stem)
        # Union of tags across slices. Each slice's tags column is a
        # JSON array string; parse, union, re-serialise.
        merged: set[str] = set()
        for tj in (tag_jsons or []):
            try:
                import json as _j
                items = _j.loads(tj) if tj else []
                if isinstance(items, list):
                    merged.update(str(x) for x in items)
            except Exception:
                continue
        tags_json = None
        if merged:
            import json as _j
            tags_json = _j.dumps(sorted(merged))
        insert_rows.append((stem, display, tags_json))

    con.executemany(
        "INSERT INTO canonical_benchmarks (id, display_name, tags) "
        "VALUES (?, ?, ?)",
        insert_rows,
    )
    log.info(
        "Stage G: synthesised %d phantom-stem benchmark(s) for siblings "
        "missing a bare-stem canonical row.", len(insert_rows),
    )
    return len(insert_rows)


def _synthesise_singleton_families(con) -> None:
    """Ensure family_membership has a row for every *root* benchmark.

    A singleton family is `{family_id == benchmark_id, display_name ==
    benchmark.display_name}`. The curated YAML only carries multi-
    benchmark families; this helper fills in the long tail.

    Slice rows (parent_benchmark_id != id) are excluded — their family
    comes from the root benchmark, looked up at dim materialisation
    time. Without the filter, slice ids like `gaia-level-1` would land
    as their own singleton families and clutter the families[] index.
    """
    con.execute(
        """
        INSERT INTO family_membership (family_id, family_display_name, benchmark_id)
        SELECT cb.id, COALESCE(cb.display_name, cb.id), cb.id
        FROM canonical_benchmarks cb
        WHERE cb.id IS NOT NULL
          AND (cb.parent_benchmark_id IS NULL OR cb.parent_benchmark_id = cb.id)
          AND cb.id NOT IN (SELECT benchmark_id FROM family_membership)
        """
    )


def stage_g_materialise_dim_tables(con, snapshot_id: str) -> None:
    """Materialise four dim tables:
      - `benchmarks` keyed on (composite_slug, benchmark_id) — one row per
        (composite, benchmark) appearance, plus is_slice rows for slice
        cuts (gaia-level-1 etc.).
      - `composites` — one row per composite_slug with display name +
        config list.
      - `families` — one row per family_id with display name + member
        list.
      - `models` — unchanged from the legacy shape.
    """
    sid = snapshot_id_to_sql(snapshot_id)

    _synthesise_phantom_benchmarks(con)
    _synthesise_singleton_families(con)

    # ---- benchmarks dim ---------------------------------------------------
    # Per-(composite_slug, benchmark_id). Card columns and registry meta
    # are per-benchmark (independent of composite) so they're duplicated
    # across each (composite, benchmark) row. Slice rows materialise inline
    # with is_slice=TRUE; for slices `benchmark_id` is the slice's own id
    # and `parent_benchmark_id` points at the root (e.g. gaia-level-1
    # carries parent_benchmark_id='gaia').
    con.execute(
        f"""
        CREATE TABLE benchmarks AS
        WITH cards_json AS (
            SELECT card_key, benchmark_id, to_json(card) AS card_j FROM cards_raw
        ),
        card_missing_per_benchmark AS (
            SELECT
                benchmark_key                              AS benchmark_id,
                MAX(len(list_filter(
                    completeness_missing_required_fields,
                    x -> starts_with(x, 'autobenchmarkcard.')
                ))) AS card_missing_count
            FROM fact_results
            WHERE benchmark_key IS NOT NULL
            GROUP BY benchmark_key
        ),
        base_pairs AS (
            -- Distinct (composite, benchmark_key) pairs from fact_results.
            -- Keying on `benchmark_key` (canonical-or-raw) keeps
            -- unresolved benchmarks from being silently dropped at the
            -- dim layer; the LEFT JOIN to canonical_benchmarks below
            -- harmlessly produces NULLs for raw-only entries and the
            -- SELECT falls back to the raw string for display.
            SELECT DISTINCT composite_slug, benchmark_key AS benchmark_id
            FROM fact_results
            WHERE composite_slug IS NOT NULL
              AND benchmark_key  IS NOT NULL
        ),
        composite_displays AS (
            -- Most-frequent display name per composite, tie-broken
            -- lexicographically. Multi-record composites (hle) carry
            -- several names; ANY_VALUE flipped between them run-to-run.
            SELECT composite_slug, composite_display_name
            FROM (
                SELECT composite_slug, composite_display_name,
                       ROW_NUMBER() OVER (
                           PARTITION BY composite_slug
                           ORDER BY COUNT(*) DESC, composite_display_name ASC
                       ) AS _rk
                FROM fact_results
                WHERE composite_slug IS NOT NULL
                  AND composite_display_name IS NOT NULL
                GROUP BY composite_slug, composite_display_name
            )
            WHERE _rk = 1
        ),
        phantom_root_pairs AS (
            -- Phantom stems (e.g. arc-agi, caparena) referenced as
            -- parent_benchmark_id by ≥2 children but with no own fact
            -- rows. Stage G has already inserted them into
            -- canonical_benchmarks via _synthesise_phantom_benchmarks,
            -- but they don't appear in `base_pairs` because they have
            -- no fact rows themselves. Without this UNION the children
            -- would be orphan slices in the hierarchy (no root).
            SELECT DISTINCT bp.composite_slug, cb.parent_benchmark_id AS benchmark_id
            FROM base_pairs bp
            JOIN canonical_benchmarks cb ON cb.id = bp.benchmark_id
            WHERE cb.parent_benchmark_id IS NOT NULL
              AND cb.parent_benchmark_id != cb.id
              AND NOT EXISTS (
                  SELECT 1 FROM base_pairs bp2
                  WHERE bp2.composite_slug = bp.composite_slug
                    AND bp2.benchmark_id = cb.parent_benchmark_id
              )
        ),
        composite_benchmark_pairs AS (
            SELECT
                ap.composite_slug,
                cd.composite_display_name,
                ap.benchmark_id
            FROM (
                SELECT * FROM base_pairs
                UNION
                SELECT * FROM phantom_root_pairs
            ) ap
            LEFT JOIN composite_displays cd USING (composite_slug)
        ),
        is_slice_flag AS (
            -- A canonical id is a slice when its parent points at a
            -- *different* benchmark id. Self-parented bare stems (e.g.
            -- gaia → gaia) stay non-slices: they're the root benchmark
            -- with a phantom-or-explicit stem reference.
            SELECT
                cb.id AS benchmark_id,
                (cb.parent_benchmark_id IS NOT NULL
                 AND cb.parent_benchmark_id != cb.id) AS is_slice
            FROM canonical_benchmarks cb
        ),
        family_lookup AS (
            -- For non-slice rows, look up family directly. For slice
            -- rows (parent != self), inherit the root's family so
            -- gaia-level-1 lands in the same family as gaia. The
            -- COALESCE picks the slice's parent if set; otherwise the
            -- benchmark's own id (which family_membership has a
            -- singleton entry for after _synthesise_singleton_families).
            SELECT
                cb.id AS benchmark_id,
                COALESCE(fm.family_id,         cb.id)                AS family_id,
                COALESCE(fm.family_display_name,
                         cb.display_name, cb.id)                     AS family_display_name
            FROM canonical_benchmarks cb
            LEFT JOIN family_membership fm
              ON fm.benchmark_id = CASE
                  WHEN cb.parent_benchmark_id IS NOT NULL
                       AND cb.parent_benchmark_id != cb.id
                  THEN cb.parent_benchmark_id
                  ELSE cb.id
                 END
        )
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            cbp.composite_slug,
            cbp.composite_display_name,
            cbp.benchmark_id,

            COALESCE(cb.display_name, cbp.benchmark_id)          AS display_name,
            COALESCE(cb.display_name, cbp.benchmark_id)          AS benchmark_display_name,
            cb.description,
            cb.dataset_repo,
            cb.parent_benchmark_id,
            fl.family_id,
            fl.family_display_name,
            COALESCE(isf.is_slice, FALSE)                        AS is_slice,
            TRY_CAST(from_json(cb.tags, '["VARCHAR"]') AS VARCHAR[]) AS registry_tags,
            TRY_CAST(cb.metadata AS JSON) AS registry_metadata,
            cb.review_status,

            json_extract_string(c.card_j, '$.benchmark_details.name')      AS card_name,
            json_extract_string(c.card_j, '$.benchmark_details.overview')  AS overview,
            json_extract_string(c.card_j, '$.benchmark_details.data_type') AS data_type,
            TRY_CAST(from_json(json_extract(c.card_j, '$.benchmark_details.domains'),     '["VARCHAR"]') AS VARCHAR[]) AS domains,
            TRY_CAST(from_json(json_extract(c.card_j, '$.benchmark_details.languages'),   '["VARCHAR"]') AS VARCHAR[]) AS languages,
            TRY_CAST(from_json(json_extract(c.card_j, '$.benchmark_details.similar_benchmarks'), '["VARCHAR"]') AS VARCHAR[]) AS similar_benchmarks,
            TRY_CAST(from_json(json_extract(c.card_j, '$.benchmark_details.resources'),   '["VARCHAR"]') AS VARCHAR[]) AS resources,

            json_extract_string(c.card_j, '$.purpose_and_intended_users.goal')       AS goal,
            TRY_CAST(from_json(json_extract(c.card_j, '$.purpose_and_intended_users.audience'), '["VARCHAR"]') AS VARCHAR[]) AS audience,
            TRY_CAST(from_json(json_extract(c.card_j, '$.purpose_and_intended_users.tasks'),    '["VARCHAR"]') AS VARCHAR[]) AS tasks,
            json_extract_string(c.card_j, '$.purpose_and_intended_users.limitations') AS limitations,
            TRY_CAST(from_json(json_extract(c.card_j, '$.purpose_and_intended_users.out_of_scope_uses'), '["VARCHAR"]') AS VARCHAR[]) AS out_of_scope_uses,

            json_extract_string(c.card_j, '$.data.source')     AS data_source,
            json_extract_string(c.card_j, '$.data.size')       AS data_size,
            json_extract_string(c.card_j, '$.data.format')     AS data_format,
            json_extract_string(c.card_j, '$.data.annotation') AS data_annotation,

            TRY_CAST(from_json(json_extract(c.card_j, '$.methodology.methods'), '["VARCHAR"]') AS VARCHAR[]) AS methods,
            TRY_CAST(from_json(json_extract(c.card_j, '$.methodology.metrics'), '["VARCHAR"]') AS VARCHAR[]) AS card_metrics,
            json_extract_string(c.card_j, '$.methodology.calculation')      AS calculation,
            json_extract_string(c.card_j, '$.methodology.interpretation')   AS interpretation,
            json_extract_string(c.card_j, '$.methodology.baseline_results') AS baseline_results,
            json_extract_string(c.card_j, '$.methodology.validation')       AS validation,

            json_extract_string(c.card_j, '$.ethical_and_legal_considerations.privacy_and_anonymity')        AS privacy_and_anonymity,
            json_extract_string(c.card_j, '$.ethical_and_legal_considerations.data_licensing')               AS data_licensing,
            json_extract_string(c.card_j, '$.ethical_and_legal_considerations.consent_procedures')           AS consent_procedures,
            json_extract_string(c.card_j, '$.ethical_and_legal_considerations.compliance_with_regulations')  AS compliance_with_regulations,

            -- possible_risks: typed STRUCT array. Upstream cards populate
            -- only category, description, url; description is always a LIST
            -- of strings, never a scalar. TRY_CAST returns NULL when the
            -- card omits the field entirely.
            TRY_CAST(from_json(
                json_extract(c.card_j, '$.possible_risks'),
                '[{{"category": "VARCHAR", "description": "VARCHAR[]", "url": "VARCHAR"}}]'
            ) AS STRUCT(category VARCHAR, description VARCHAR[], url VARCHAR)[]) AS possible_risks,
            json_extract(c.card_j, '$.flagged_fields') AS flagged_fields,

            (c.card_j IS NOT NULL) AS card_present,
            json_extract_string(c.card_j, '$._generated_by') AS card_generated_by,
            COALESCE(len(json_keys(json_extract(c.card_j, '$.flagged_fields'))), 0) AS card_flagged_count,
            cmpb.card_missing_count

        FROM composite_benchmark_pairs cbp
        LEFT JOIN canonical_benchmarks cb         ON cb.id = cbp.benchmark_id
        LEFT JOIN cards_json c                    ON c.benchmark_id = cbp.benchmark_id
        LEFT JOIN card_missing_per_benchmark cmpb ON cmpb.benchmark_id = cbp.benchmark_id
        LEFT JOIN family_lookup fl                ON fl.benchmark_id = cbp.benchmark_id
        LEFT JOIN is_slice_flag isf               ON isf.benchmark_id = cbp.benchmark_id
        """
    )

    # ---- composites dim --------------------------------------------------
    # One row per composite_slug that has at least one resolved
    # benchmark in the benchmarks dim. Composites whose fact rows are
    # entirely unresolved (registry resolver gap) are excluded so the
    # frontend doesn't render hollow tiles. configs[] is the list of EEE
    # source_configs that funnel into this composite. evals_count = sum
    # of distinct (benchmark, metric) triples across the composite.
    con.execute(
        f"""
        CREATE TABLE composites AS
        WITH display_pick AS (
            -- Most-frequent display name per composite (lexicographic
            -- tie-break) — multi-record composites carry several names
            -- and ANY_VALUE flipped between runs.
            SELECT composite_slug, composite_display_name
            FROM (
                SELECT composite_slug, composite_display_name,
                       ROW_NUMBER() OVER (
                           PARTITION BY composite_slug
                           ORDER BY COUNT(*) DESC, composite_display_name ASC
                       ) AS _rk
                FROM fact_results
                WHERE composite_slug IS NOT NULL
                  AND composite_display_name IS NOT NULL
                GROUP BY composite_slug, composite_display_name
            )
            WHERE _rk = 1
        ),
        composite_configs AS (
            SELECT
                fr.composite_slug,
                MAX(dp.composite_display_name)       AS composite_display_name,
                ARRAY_AGG(DISTINCT fr.source_config ORDER BY fr.source_config)
                    FILTER (WHERE fr.source_config IS NOT NULL) AS source_configs,
                COUNT(DISTINCT (fr.benchmark_key, fr.metric_key))
                    FILTER (WHERE fr.benchmark_key IS NOT NULL
                            AND fr.metric_key      IS NOT NULL) AS evals_count
            FROM fact_results fr
            LEFT JOIN display_pick dp USING (composite_slug)
            WHERE fr.composite_slug IS NOT NULL
            GROUP BY fr.composite_slug
        ),
        live_composites AS (
            SELECT DISTINCT composite_slug FROM benchmarks
        )
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            cc.composite_slug,
            cc.composite_display_name,
            cc.source_configs,
            cc.evals_count
        FROM composite_configs cc
        JOIN live_composites lc USING (composite_slug)
        ORDER BY cc.composite_slug
        """
    )

    # ---- families dim ----------------------------------------------------
    # One row per family_id (curated multi-benchmark families + singleton
    # default families). member_benchmark_keys lists every benchmark id in
    # the family that's actually represented in fact_results for this
    # snapshot — a curated family member that hasn't shown up yet drops
    # off the list rather than appearing with zero data behind it.
    con.execute(
        f"""
        CREATE TABLE families AS
        WITH used_benchmarks AS (
            SELECT DISTINCT benchmark_id FROM fact_results
            WHERE benchmark_id IS NOT NULL
        ),
        family_members AS (
            SELECT
                fm.family_id,
                ANY_VALUE(fm.family_display_name) AS family_display_name,
                ARRAY_AGG(DISTINCT fm.benchmark_id ORDER BY fm.benchmark_id)
                    AS member_benchmark_keys
            FROM family_membership fm
            JOIN used_benchmarks ub ON ub.benchmark_id = fm.benchmark_id
            GROUP BY fm.family_id
        )
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            family_id,
            family_display_name,
            member_benchmark_keys
        FROM family_members
        ORDER BY family_id
        """
    )

    # Stage G models-dim: precompute the MODEL DEVELOPER name-pattern CASE
    # so the SQL stays readable. See module-scope MODEL_DEVELOPER_NAME_PATTERNS.
    dev_pattern_case = _model_developer_pattern_case_sql("um.model_key")

    # models.parquet — root grain. One row per `model_aggregation_key`
    # (= transitive variant root for resolved rows, raw string for
    # unresolved). Variants of the same identity collapse into one row;
    # `raw_model_ids` and `variant_keys` expose the per-variant strings
    # and ids that fed into it. `model_id` is the root canonical id (or
    # NULL when unresolved). Registry display fields are looked up by
    # joining canonical_models on the root id; `display_name` falls back
    # to a raw source name; `review_status` is 'unresolved' for rows
    # that don't match canonical so consumers can flag them.
    con.execute(
        f"""
        CREATE TABLE models AS
        WITH used_models AS (
            SELECT
                model_aggregation_key                             AS model_key,
                ANY_VALUE(model_raw)                              AS model_raw_sample,
                ARRAY_AGG(DISTINCT model_raw ORDER BY model_raw)
                    FILTER (WHERE model_raw IS NOT NULL)          AS raw_model_ids,
                ARRAY_AGG(DISTINCT model_key ORDER BY model_key)
                    FILTER (WHERE model_key IS NOT NULL)          AS variant_keys,
                ARRAY_AGG(DISTINCT model_id ORDER BY model_id)
                    FILTER (WHERE model_id IS NOT NULL)           AS leaf_model_ids,
                ARRAY_AGG(DISTINCT model_leaf_id ORDER BY model_leaf_id)
                    FILTER (WHERE model_leaf_id IS NOT NULL)      AS resolved_leaf_ids
            FROM fact_results
            WHERE model_aggregation_key IS NOT NULL
            GROUP BY model_aggregation_key
        ),
        -- Earliest snapshot release_date across the resolved leaves
        -- that aggregate into this model_key. Lets Stage J's view
        -- surface a per-snapshot release date even when the family
        -- pointer canonical's `release_date` is NULL — the common
        -- shape post-Gap-A where dated snapshots become first-class
        -- canonicals with their own `release_date` and the family
        -- pointer is a moving label without one. MIN picks the
        -- model's earliest known snapshot, which is "when the family
        -- first shipped."
        leaf_release AS (
            SELECT
                um.model_key,
                MIN(leaf_cm.release_date) AS leaf_release_date
            FROM used_models um,
                 UNNEST(um.resolved_leaf_ids) AS t(leaf_id)
            LEFT JOIN canonical_models leaf_cm ON leaf_cm.id = t.leaf_id
            GROUP BY um.model_key
        )
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            um.model_key,
            cm.id                                            AS model_id,
            um.raw_model_ids,
            um.variant_keys,
            um.leaf_model_ids,
            um.resolved_leaf_ids,

            COALESCE(
                cm.display_name,
                -- For unresolved HF-shaped raws (`org/name`), drop the
                -- org prefix so display matches the resolved-row
                -- convention: name carries the model, developer carries
                -- the org separately. The full raw id is preserved in
                -- `raw_model_ids` / `model_key` for callers needing the
                -- original string.
                CASE
                    WHEN um.model_raw_sample LIKE '%/%'
                         AND length(split_part(um.model_raw_sample, '/', 2)) > 0
                    THEN split_part(um.model_raw_sample, '/', 2)
                    ELSE um.model_raw_sample
                END
            )                                                AS display_name,
            cm.developer,
            cm.org_id,
            cm.family,
            cm.architecture,
            cm.params_billions,
            cm.parent_model_id,
            -- Model-resolution-rework end-state names (registry-side, in
            -- place): `root_model_id` -> `model_group_id` (the always-present
            -- group key; self at root), `model_family_id` is the STRUCTURAL
            -- family-release id (the M3 family walk), `lineage_origin_org_id`
            -- -> `lineage_origin_model_org_id`. Emit the new canonical names
            -- plus the legacy `root_model_id` alias for back-compat.
            cm.model_group_id,
            cm.model_group_id                                AS root_model_id,
            cm.model_family_id                               AS model_family_id,
            cm.lineage_origin_model_id,
            cm.lineage_origin_model_org_id,
            cm.lineage_origin_model_org_id                   AS lineage_origin_org_id,
            cm.resolution_source,
            cm.resolution_granularity,
            cm.open_weights,
            -- Prefer the leaf-aggregated date over the family pointer's
            -- own release_date. NULL on both sides yields NULL, which
            -- the frontend renders as "—" via formatDateShort.
            COALESCE(lr.leaf_release_date, cm.release_date)  AS release_date,
            -- Modalities surfaced as VARCHAR[] for the views; on `models`
            -- dim we keep the JSON-encoded form to round-trip cleanly via
            -- parquet readers that don't support nested arrays in joins.
            TRY_CAST(from_json(cm.input_modalities, '["VARCHAR"]') AS VARCHAR[])  AS input_modalities,
            TRY_CAST(from_json(cm.output_modalities, '["VARCHAR"]') AS VARCHAR[]) AS output_modalities,
            cm.parents                                       AS lineage_parents,
            TRY_CAST(from_json(cm.tags, '["VARCHAR"]') AS VARCHAR[]) AS registry_tags,
            TRY_CAST(cm.metadata AS JSON)                    AS registry_metadata,
            COALESCE(cm.review_status, 'unresolved')         AS review_status,

            -- Developer/org backfill priority. "Model developer" includes
            -- individuals (HF community uploaders) — not just labs — so the
            -- raw slug prefix is a legitimate developer identity even when
            -- the registry has no canonical org for it.
            --   1. canonical_models.org_id → co.*  (registry-truthful)
            --   2. canonical_models.lineage_origin_model_org_id → co_lineage.*
            --      (registry hit but org_id missing, e.g. xiaomi/mimo-v2)
            --   3. split_part(model_key, '/', 1) → co_slug.*
            --      (registry miss, slug carries an org we know — e.g.
            --       `openai/gpt-5-...` matches canonical orgs.id='openai',
            --       so we render polished "OpenAI" rather than the slug)
            --   4. name-pattern → co_pattern.*
            --      Orgless display names like `chatgpt-4o-latest`,
            --      `claude-3-5-opus`, `Qwen2-72B-Instruct` map to a
            --      canonical org via MODEL_DEVELOPER_NAME_PATTERNS.
            --   5. raw slug prefix as-is — `jaspionjader/Llama-…-merged` etc.
            --      Filters the literal `unknown` placeholder so it isn't
            --      treated as a developer. Casing reflects what the source
            --      uploaded (slug-style for community, polished for known
            --      orgs via step 3).
            COALESCE(
                co.display_name,
                co_lineage.display_name,
                co_slug.display_name,
                co_pattern.display_name,
                CASE
                    WHEN um.model_key LIKE '%/%'
                         AND length(split_part(um.model_key, '/', 1)) > 0
                         AND split_part(um.model_key, '/', 1) != 'unknown'
                    THEN split_part(um.model_key, '/', 1)
                END
            )                                                                            AS org_display_name,
            COALESCE(co.website,       co_lineage.website,       co_slug.website,       co_pattern.website)       AS org_website,
            COALESCE(co.hf_org,        co_lineage.hf_org,        co_slug.hf_org,        co_pattern.hf_org)        AS org_hf_org,
            COALESCE(co.kind,          co_lineage.kind,          co_slug.kind,          co_pattern.kind)          AS org_kind,
            COALESCE(co.parent_org_id, co_lineage.parent_org_id, co_slug.parent_org_id, co_pattern.parent_org_id) AS org_parent_id

        FROM used_models um
        LEFT JOIN canonical_models cm ON cm.id = um.model_key
        LEFT JOIN leaf_release lr     ON lr.model_key = um.model_key
        LEFT JOIN canonical_orgs co         ON co.id         = cm.org_id
        LEFT JOIN canonical_orgs co_lineage ON co_lineage.id = cm.lineage_origin_model_org_id
        LEFT JOIN canonical_orgs co_slug    ON co_slug.id    = split_part(um.model_key, '/', 1)
        LEFT JOIN canonical_orgs co_pattern ON co_pattern.id = ({dev_pattern_case})
        """
    )

    # Surface registry-staleness signals on the models dim. A model_key
    # that didn't match `canonical_models.id` indicates either (a) the
    # registry hasn't synced this model yet — operator must run
    # `eval-card-registry sync` and push to entity-registry-data — or
    # (b) the producer's join key (model_aggregation_key) carries the
    # raw HF id rather than the registry slug. Either way the row
    # surfaces with NULL metadata in the warehouse, so consumers
    # deserve a heads-up; HF-shaped misses are particularly noteworthy
    # because the registry's auto-create path would normally have
    # populated those.
    # When the canonical lookup misses, `model_aggregation_key` falls
    # back to the raw `model_raw` (the Stage E COALESCE), so on
    # the `models` dim that raw value lands in `model_key` itself —
    # detecting HF-shape there avoids a re-join to fact_results.
    miss_total, miss_hf_shaped = con.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE model_id IS NULL),
            COUNT(*) FILTER (
                WHERE model_id IS NULL
                  AND model_key LIKE '%/%'
                  AND length(split_part(model_key, '/', 1)) > 0
            )
        FROM models
        """
    ).fetchone() or (0, 0)
    if miss_total:
        log.warning(
            "Stage G: %d models row(s) had no canonical_models match "
            "(%d look like HF ids — stale registry snapshot, or new "
            "upstream models not yet seeded in the registry). "
            "Consumers will see NULL metadata for these models; "
            "view-layer falls developer back to the raw org prefix.",
            miss_total, miss_hf_shaped,
        )

    # Developer coverage on the models dim. "Model developer" is intentionally
    # broad — labs, academic groups, and individual HF uploaders all count.
    # The breakdown distinguishes registry-rich (canonical org) from raw-slug
    # fallback (community uploaders) so operators can see both kinds of
    # coverage and decide which gaps are worth investing in (canonicalising
    # a high-traffic uploader vs adding a name-pattern rule).
    cov_pattern_case = _model_developer_pattern_case_sql("m.model_key")
    cov = con.execute(
        f"""
        SELECT
            COUNT(*)                                                AS total,
            -- Rich resolution: registry knew the org, the slug prefix
            -- matched a canonical org, or a name pattern matched.
            COUNT(*) FILTER (
                WHERE COALESCE(
                    co_match.id, co_lineage_match.id,
                    co_slug_match.id, co_pattern_match.id
                ) IS NOT NULL
            )                                                       AS resolved_canonical,
            -- Raw-slug fallback: developer is the HF user/org slug because
            -- no canonical_orgs row matched. Each distinct prefix here is a
            -- candidate for promotion to seed/orgs.yaml if it's a real lab.
            COUNT(*) FILTER (
                WHERE org_display_name IS NOT NULL
                  AND COALESCE(
                      co_match.id, co_lineage_match.id,
                      co_slug_match.id, co_pattern_match.id
                  ) IS NULL
            )                                                       AS resolved_raw_slug,
            -- Orgless display names that didn't match any pattern. These
            -- are Fix-2 candidates: extend MODEL_DEVELOPER_NAME_PATTERNS
            -- once a pattern is identified.
            COUNT(*) FILTER (WHERE org_display_name IS NULL)        AS unresolved_orgless
        FROM models m
        LEFT JOIN canonical_models cm_match ON cm_match.id = m.model_key
        LEFT JOIN canonical_orgs co_match
            ON co_match.id = cm_match.org_id
        LEFT JOIN canonical_orgs co_lineage_match
            ON co_lineage_match.id = cm_match.lineage_origin_model_org_id
        LEFT JOIN canonical_orgs co_slug_match
            ON co_slug_match.id = split_part(m.model_key, '/', 1)
        LEFT JOIN canonical_orgs co_pattern_match
            ON co_pattern_match.id = ({cov_pattern_case})
        """
    ).fetchone() or (0, 0, 0, 0)
    cov_total, cov_canonical, cov_raw_slug, cov_orgless = cov
    cov_distinct_devs = con.execute(
        "SELECT COUNT(DISTINCT org_display_name) FROM models WHERE org_display_name IS NOT NULL"
    ).fetchone()[0]
    cov_rate = ((cov_canonical + cov_raw_slug) / cov_total) if cov_total else 0.0
    log.info(
        "Stage G developer coverage: %d/%d models (%.1f%%) have a developer; "
        "%d distinct developers in total. Breakdown — %d via canonical org "
        "(registry/slug match), %d via raw slug fallback (community "
        "uploaders), %d orgless display names (needs name→org pattern "
        "table).",
        cov_canonical + cov_raw_slug, cov_total, cov_rate * 100,
        cov_distinct_devs, cov_canonical, cov_raw_slug, cov_orgless,
    )


# ---------------------------------------------------------------------------
# Stage H removed — completeness is per-row, computed in Stage E.
# Stage G derives benchmarks.card_missing_count inline from fact_results.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage I — emit Parquet
# ---------------------------------------------------------------------------


# fact_id makes the sort total — the five-column prefix ties for
# multi-slice/multi-record triples and left row order (and thus the emitted
# bytes) run-to-run unstable. Clustered on the EFFECTIVE metric identity
# (metric_key), which is what every consumer groups and filters by. Shared
# with Stage J's re-emit so the two writes agree on physical order.
FACT_RESULTS_SORT_KEY = (
    "(composite_slug, model_key, benchmark_id, metric_key, slice_key, fact_id)"
)


def _diagnostic_role_sql(metadata_expr: str) -> str:
    """1 when the registry marks this metric's role `diagnostic`, else 0.

    A diagnostic metric describes how a run went — cost, latency, response
    length, degeneration, invalid-rate, rank, uncertainty — rather than how
    well the model did the task. Ordering on this demotes them when a page's
    headline metric is chosen automatically."""
    return (
        "CASE WHEN json_extract_string("
        f"{metadata_expr}, '$.role') = 'diagnostic' THEN 1 ELSE 0 END"
    )


def _pooled_value_aggregates_sql(col: str, prefix: str) -> str:
    """Every pooled statistic for one score column, over a cell's value rows.

    Median AND mean are computed for both layers — the first-party value rows
    and all of them — because which one the cell publishes depends on its
    aggregation level and on whether any first-party row scored, and an
    aggregate cannot branch on its own results. The outer SELECT picks.

    `*_wmean_ok` is the weighting precondition: every value row carries an
    `n_samples`, and they sum to something. A suite score weighted by sample
    count is only meaningful when no part is missing its count — one absent
    weight and the average silently becomes a different quantity.
    """
    parts: list[str] = []
    for tag, flt in (
        (
            "fp",
            f"_is_value_row AND evaluator_relationship = 'first_party' "
            f"AND {col} IS NOT NULL",
        ),
        ("all", f"_is_value_row AND {col} IS NOT NULL"),
    ):
        parts += [
            f"MEDIAN({col}) FILTER (WHERE {flt}) AS {prefix}_median_{tag}",
            f"AVG({col}) FILTER (WHERE {flt}) AS {prefix}_mean_{tag}",
            f"(SUM({col} * n_samples) FILTER (WHERE {flt} AND n_samples IS NOT NULL)"
            f" / NULLIF(SUM(n_samples) FILTER (WHERE {flt} AND n_samples IS NOT NULL), 0))"
            f" AS {prefix}_wmean_{tag}",
            f"(COUNT(*) FILTER (WHERE {flt}) > 0"
            f" AND COUNT(*) FILTER (WHERE {flt})"
            f"   = COUNT(*) FILTER (WHERE {flt} AND n_samples IS NOT NULL)"
            f" AND COALESCE(SUM(n_samples) FILTER (WHERE {flt}), 0) > 0)"
            f" AS {prefix}_wmean_ok_{tag}",
        ]
    return ",\n                ".join(parts)


def _pooled_value_pick_sql(prefix: str, alias: str = "ta") -> str:
    """The number a cell publishes, out of `_pooled_value_aggregates_sql`.

    A `derived` cell is the pipeline combining parts into a whole and takes
    the mean; every other level is several submitted readings of one quantity
    and takes the median. The first-party layer is preferred at both, by
    falling through when it produced nothing.
    """
    fp_mean = (
        f"CASE WHEN {alias}.{prefix}_wmean_ok_fp THEN {alias}.{prefix}_wmean_fp "
        f"ELSE {alias}.{prefix}_mean_fp END"
    )
    all_mean = (
        f"CASE WHEN {alias}.{prefix}_wmean_ok_all THEN {alias}.{prefix}_wmean_all "
        f"ELSE {alias}.{prefix}_mean_all END"
    )
    return (
        f"CASE WHEN {alias}._value_level = 'derived' "
        f"     THEN COALESCE({fp_mean}, {all_mean}) "
        f"     ELSE COALESCE({alias}.{prefix}_median_fp, {alias}.{prefix}_median_all) "
        f"END"
    )


def _pooled_aggregation_pick_sql(prefix: str, alias: str = "ta") -> str:
    """How that number was computed, for `value_aggregation`. NULL when only
    one row went in — nothing was aggregated."""
    use_fp = f"{alias}.{prefix}_mean_fp IS NOT NULL"
    weighted = (
        f"CASE WHEN {use_fp} THEN {alias}.{prefix}_wmean_ok_fp "
        f"ELSE {alias}.{prefix}_wmean_ok_all END"
    )
    return (
        f"CASE WHEN {alias}._n_value_rows <= 1 THEN NULL "
        f"     WHEN {alias}._value_level = 'derived' "
        f"     THEN CASE WHEN {weighted} THEN '{AGG_WEIGHTED_MEAN}' "
        f"               ELSE '{AGG_MEAN}' END "
        f"     ELSE '{AGG_MEDIAN}' END"
    )


def _qualify_sort_key(sort_key: str, alias: str) -> str:
    """Prefix every bare column in a sort key with a table alias, for the
    emit that joins fact_results to the headline map."""
    return re.sub(r"(?<![\w.])(\w+)", rf"{alias}.\1", sort_key)


def stage_i_emit_warehouse_parquets(con, out_dir: Path, snapshot_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = snapshot_id_to_sql(snapshot_id)
    for table, sort_key in [
        ("fact_results", FACT_RESULTS_SORT_KEY),
        ("benchmarks", "(composite_slug, benchmark_id)"),
        ("composites", "(composite_slug)"),
        ("families", "(family_id)"),
        ("models", "(model_key)"),
    ]:
        path = out_dir / f"{table}.parquet"
        con.execute(
            f"""
            COPY (SELECT {explicit_projection_sql(con, table)} FROM {table}
                  ORDER BY {sort_key} NULLS LAST)
            TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    # canonical_metrics is COPYed straight from the registry; inject
    # snapshot_id so it satisfies the append-only contract every other
    # warehouse table follows.
    path = out_dir / "canonical_metrics.parquet"
    con.execute(
        f"""
        COPY (
            SELECT TIMESTAMP '{sid}' AS snapshot_id, *
            FROM canonical_metrics
            ORDER BY id NULLS LAST
        )
        TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # collection_trajectories.parquet: vendored trajectories joined
    # to resolved ids — the collection dashboard's data source. Only emitted
    # when a collection adapter actually loaded trajectories this run.
    collections_src.create_collection_tables(con)
    n_traj = con.execute(
        "SELECT count(*) FROM collection_trajectories_raw"
    ).fetchone()[0]
    if n_traj:
        path = out_dir / "collection_trajectories.parquet"
        # Tie-break digest over the row's own columns, in declared order and
        # NULL-safe. `CAST(t AS VARCHAR)` would digest DuckDB's struct
        # serialisation, an implementation detail that can respell between
        # versions and silently reorder the file for identical data.
        traj_digest = "md5(" + " || ".join(
            f'md5(COALESCE(CAST(t."{c}" AS VARCHAR), \'\'))'
            for c in (r[0] for r in con.execute(
                "DESCRIBE SELECT * FROM collection_trajectories_raw").fetchall())
        ) + ")"
        con.execute(
            f"""
            COPY (
                SELECT
                    TIMESTAMP '{sid}' AS snapshot_id,
                    t.*,
                    ids.model_id,
                    ids.benchmark_id,
                    -- canonical-or-raw keys, same convention as the rest
                    -- of the warehouse: never NULL, fall back to the raw
                    -- string when the registry has no canonical entry.
                    COALESCE(ids.model_key, t.model_raw)         AS model_key,
                    COALESCE(ids.benchmark_key, t.benchmark_raw) AS benchmark_key
                FROM collection_trajectories_raw t
                LEFT JOIN (
                    -- (collection, model_raw, config) → resolved ids, read
                    -- off the collection's synthetic fact rows so the
                    -- trajectories inherit exactly the resolution the
                    -- warehouse shipped. benchmark_raw on trajectories is
                    -- the benchmark config name by extractor contract.
                    SELECT collection_id, model_raw, source_config,
                           MAX(model_id)      AS model_id,
                           MAX(benchmark_id)  AS benchmark_id,
                           MAX(model_key)     AS model_key,
                           MAX(benchmark_key) AS benchmark_key
                    FROM fact_results
                    WHERE evaluation_id IN
                          (SELECT evaluation_id FROM collection_member_ids)
                    GROUP BY 1, 2, 3
                ) ids
                  ON ids.collection_id = t.collection_id
                 AND ids.model_raw     = t.model_raw
                 AND ids.source_config = t.benchmark_raw
                -- Total order: the natural key is not unique (a task can
                -- carry several unstitched pieces under one trajectory_idx,
                -- and idx is NULL for some extractors), so the whole row is
                -- the final tie-break. Without it two identical runs emit
                -- the same rows in a different order.
                ORDER BY t.collection_id, t.benchmark_raw, t.model_raw,
                         t.protocol_condition, t.task_id,
                         t.trajectory_idx NULLS LAST,
                         t.source_record_uuids, {traj_digest}
            )
            TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        log.info("Stage I: emitted collection_trajectories.parquet (%d rows)", n_traj)


# ---------------------------------------------------------------------------
# Stage J — view-layer materialisation
# ---------------------------------------------------------------------------


def _ensure_merged_view_inputs(con) -> None:
    """Stage J can run on a connection rebuilt from emitted parquets
    (tests' view-materialise helpers, `--from-stage J` over a pre-fold
    cache) that lacks the Stage A registry tables the fold/merged logic
    reads. Create empty stand-ins: folds simply don't apply and the
    merged view degrades to empty rather than raising CatalogException.
    """
    for table in ("benchmark_metric_folds", "canonical_benchmarks"):
        ddl = ", ".join(f"{c} {t}" for c, t in _DIM_SCHEMAS[table])
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({ddl})")


def stage_j_eval_results_view(con, snapshot_id: str, eee_revision: str | None = None) -> None:
    """Materialise `eval_results_view` — one row per (composite, benchmark,
    metric, model, protocol condition, judge condition, split). Foundation
    view: models_view + evals_view fan out from this.

    The view is denormalised so the frontend's `ModelResultForBenchmark`
    cast is a no-op spread. JOINs onto `models`, `benchmarks`, and
    `canonical_metrics` happen here so the read side never JOINs.

    **Representative score rule** — a triple may have multiple fact rows
    (different orgs, setup variants). The view collapses to one row per
    triple. Score is the median over fact rows, layered: prefer first-party
    scores when any exist; else all rows. NULL when every row's score is
    NULL. Per-row context (timestamps, source metadata, instance pointer,
    eval library) comes from a representative row chosen by:
    `(score IS NOT NULL DESC, evaluator_relationship='first_party' DESC,
      evaluation_id ASC)`.

    **Position / total / percentile** — per `(benchmark_id, metric_id)`
    partition, HEADLINE rows are ranked honouring `lower_is_better`.
    Non-headline rows (extra judge conditions, extra protocol arms) and
    NULL-score rows survive in the view for coverage purposes but are
    excluded from `position` / `total`. `percentile` =
    `1 - (position-1) / (total-1)`.

    **Headline** — `fact_headline` (materialised here) maps every fact row
    to the one condition row that represents its (composite, benchmark,
    metric, model) cell. Every page-level rollup, here and in the sidecars,
    filters on it.
    """
    sid = snapshot_id_to_sql(snapshot_id)
    _ensure_merged_view_inputs(con)

    # Repo + revision for building eee_record_url deep-links back to the
    # raw EEE source records. `main` is a safe default — records are
    # addressed by a stable repo-relative file path, so /resolve/main/<path>
    # resolves without pinning. Pass the resolved commit SHA in for
    # immutable links.
    eee_repo = EEE_DATASET_REPO
    eee_rev = eee_revision or "main"

    eval_annotation_struct_type = (
        "STRUCT("
        "reproducibility_gap STRUCT("
        "  missing_fields VARCHAR[],"
        "  populated_count INTEGER,"
        "  required_count INTEGER"
        "),"
        "provenance STRUCT("
        "  source_type VARCHAR,"
        "  evaluator_relationship VARCHAR,"
        "  organization_name VARCHAR"
        "),"
        # `comparability_status` is the group-level verdict behind both
        # divergence blocks; `has_divergence` is NULL whenever it is not
        # `ok`, and NULL there means "not assessable", never "no divergence".
        "comparability_status VARCHAR,"
        "variant_divergence STRUCT("
        "  has_divergence BOOLEAN,"
        "  magnitude DOUBLE,"
        "  threshold DOUBLE,"
        "  basis VARCHAR,"
        '  differing_fields STRUCT(field VARCHAR, "values" JSON)[]'
        "),"
        "cross_party_divergence STRUCT("
        "  has_divergence BOOLEAN,"
        "  magnitude DOUBLE,"
        "  threshold DOUBLE,"
        "  basis VARCHAR,"
        '  differing_fields STRUCT(field VARCHAR, "values" JSON)[],'
        "  organization_count INTEGER"
        ")"
        ")"
    )

    aggregate_components_type = (
        "STRUCT("
        "evaluation_id VARCHAR,"
        "composite_slug VARCHAR,"
        "composite_display_name VARCHAR,"
        "score DOUBLE,"
        "normalized_score DOUBLE,"
        "evaluation_timestamp TIMESTAMP,"
        "source_name VARCHAR,"
        "source_type VARCHAR,"
        "source_organization_name VARCHAR,"
        "evaluator_relationship VARCHAR"
        ")[]"
    )

    # D6 scale-copy dedupe. A source that publishes one measurement twice —
    # its own scale and an already-rescaled copy — leaves two fact rows that
    # are one number. They collapse only when every other thing about the
    # measurement matches (same source record, identity, slice, protocol,
    # judge set, response count) and the converted copy lands on the
    # unconverted one within 1e-6; the row that needed no conversion is the
    # survivor. Anything looser would delete independent runs that merely
    # agree (1023-vs-1024 responses, 509-vs-1024).
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _erv_scale_copies AS
        WITH cand AS (
            SELECT fact_id, source_record_path, composite_slug, benchmark_key,
                   metric_key, model_aggregation_key, slice_key,
                   protocol_condition, split, scale_conversion, score_canonical,
                   -- the judge SET, not the condition: the two copies carry
                   -- the source's two labels for the same judged run
                   CAST(json_extract(judge_condition, '$.judges') AS VARCHAR)
                       AS _judges,
                   json_extract_string(metric_additional_details,
                                       '$.response_count') AS _responses
            FROM fact_results
            WHERE source_record_path IS NOT NULL
              AND score_canonical IS NOT NULL
              AND scale_conversion IN ('curated', 'none')
        )
        SELECT DISTINCT c.fact_id
        FROM cand c
        JOIN cand k
          ON k.source_record_path    = c.source_record_path
         AND k.composite_slug        = c.composite_slug
         AND k.benchmark_key         = c.benchmark_key
         AND k.metric_key            = c.metric_key
         AND k.model_aggregation_key = c.model_aggregation_key
         AND k.slice_key             IS NOT DISTINCT FROM c.slice_key
         AND k.protocol_condition    IS NOT DISTINCT FROM c.protocol_condition
         AND k.split                 IS NOT DISTINCT FROM c.split
         AND k._judges               IS NOT DISTINCT FROM c._judges
         -- both absent = one measurement reported twice; exactly one
         -- absent = we cannot tell the runs apart, so keep both
         AND k._responses            IS NOT DISTINCT FROM c._responses
         AND k.scale_conversion = 'none'
         AND c.scale_conversion = 'curated'
         AND abs(k.score_canonical - c.score_canonical) <= 1e-6
        """
    )
    n_scale_copies = con.execute(
        "SELECT count(*) FROM _erv_scale_copies"
    ).fetchone()[0]
    if n_scale_copies:
        log.info(
            "stage J: dedupe dropped %d converted scale-copy fact row(s) in "
            "favour of the source's own canonical-scale row", n_scale_copies,
        )

    # Condition-grain rows — one per (composite, model, benchmark, metric,
    # protocol condition, judge condition, split), with the representative fact
    # row attached. Its own table so the headline mapping below and the
    # view itself read exactly the same rows.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _erv_tri AS
        WITH
        tris AS (
            -- Triples are at root grain: keyed on
            -- (composite_slug, model_aggregation_key, benchmark_key,
            -- metric_key). Variants of the same identity collapse into
            -- one triple; rows whose model, benchmark, or metric failed
            -- to resolve still flow through via the raw fallback baked
            -- into each `*_key`. `fact_results.org_display` carries the
            -- de-aliased eval-provider name (canonical when registered,
            -- raw otherwise — see Stage D's `joined` CTE).
            SELECT *
            FROM fact_results
            WHERE fact_id NOT IN (SELECT fact_id FROM _erv_scale_copies)
              AND model_aggregation_key IS NOT NULL
              AND benchmark_key         IS NOT NULL
              AND metric_key            IS NOT NULL
              AND composite_slug        IS NOT NULL
        ),
        cell_level AS (
            -- Which submitted aggregation level speaks for this cell.
            --
            -- A page shows one number per (model, benchmark, metric,
            -- conditions), and until now that number was a median over every
            -- fact in the cell. Where a source publishes its own benchmark
            -- total alongside the per-subject rows it was computed from, that
            -- median is a number nobody reported: MMLU's three answer-
            -- extraction totals plus 183 subject rows medianed to 0.750 where
            -- the publisher reported 0.704.
            --
            -- The cell takes its WHOLE observations if it has any: rows that
            -- measured the benchmark itself rather than a part of it. Several
            -- of them are repeated measurements of one quantity — reruns,
            -- repeated leaderboard records, MMLU's three answer-extraction
            -- filters, MATH's two lm-eval tasks, GPQA's few-shot and
            -- zero-shot arms — and pool by median.
            --
            -- Parts NEVER enter a whole's value. That is the defect this rule
            -- exists for: MMLU's 183 subject rows used to median in with its
            -- three totals and publish 0.750 where the source reported 0.704.
            --
            -- With no whole at all, the parts can still make one, but only
            -- when the registry says what the benchmark consists of and every
            -- one of those atomic children has a value here — then the suite
            -- value is their mean (`_materialise_slice_parent_rows`).
            -- Otherwise the rows pool as the pipeline always pooled them,
            -- labelled `pooled_parts` so the page says the number is a
            -- pooling of parts rather than anything the source published, and
            -- the cell is named in the log.
            --
            -- `whole` is the LABEL, and it is claimed only where resolution
            -- verified one: `observation_role = 'whole'`, not merely
            -- `NOT is_part`. Most of HELM is neither — a flat or spaced name
            -- the structured path declines to read resolves through the plain
            -- alias index, which says nothing about what the row measured.
            -- Those cells pool exactly as they always have and are labelled
            -- `pooled` (or `single`, for a cell holding one row), so the page
            -- stops asserting that 34 subjects are 34 readings of the whole.
            --
            -- `split` is part of the cell: a run on `test` and a run on
            -- `validation` are two measurements, so they never pool. Stage D
            -- has already given a bare total the split its own parts state,
            -- so a source's total and the parts it was computed from land in
            -- one cell; NULL (unstated) groups as one value like the other
            -- condition columns.
            SELECT composite_slug, model_aggregation_key, benchmark_key,
                   metric_key, protocol_condition, judge_condition, split,
                   CASE
                       WHEN BOOL_OR(observation_role = 'whole'
                                    AND score IS NOT NULL)
                            THEN 'whole'
                       {_cell_level_tail_sql()}
                   END AS _value_level
            FROM tris
            GROUP BY 1, 2, 3, 4, 5, 6, 7
        ),
        tris_levelled AS (
            SELECT t.*, cl._value_level,
                   CASE cl._value_level
                       -- the whole observations speak; the parts stay out
                       WHEN 'whole'        THEN NOT t.is_part
                       -- one row, which is its own value
                       WHEN 'single'       THEN TRUE
                       -- Nothing verified, so nothing is preferred: the
                       -- unread rows pool as they always did. Parts stay out
                       -- for the same reason they stay out of a whole — a
                       -- subject row is known to be narrower than the rows
                       -- beside it even when those rows are unread.
                       WHEN 'pooled'       THEN NOT t.is_part
                       -- no whole: every row pools, or none does
                       WHEN 'pooled_parts' THEN TRUE
                       ELSE FALSE
                   END AS _is_value_row
            FROM tris t
            JOIN cell_level cl
              ON cl.composite_slug        = t.composite_slug
             AND cl.model_aggregation_key = t.model_aggregation_key
             AND cl.benchmark_key         = t.benchmark_key
             AND cl.metric_key            = t.metric_key
             AND cl.protocol_condition    IS NOT DISTINCT FROM t.protocol_condition
             AND cl.judge_condition       IS NOT DISTINCT FROM t.judge_condition
             AND cl.split                 IS NOT DISTINCT FROM t.split
        ),
        tri_agg AS (
            -- protocol_condition, judge_condition and split join the
            -- grouping key: one view row per (protocol point, judge
            -- condition, dataset split). NULL-condition rows (all ordinary
            -- sources) group exactly as before — DuckDB GROUP BY treats
            -- NULLs as one group. Reruns carrying the identical condition
            -- still pool (median below).
            SELECT
                composite_slug, model_aggregation_key, benchmark_key, metric_key,
                protocol_condition, judge_condition, split,
                -- Functionally determined by metric_key; carried so every
                -- registry lookup downstream (bounds, display name, the
                -- benchmark's preferred metric) reads the un-qualified id.
                MAX(metric_base_key) AS metric_base_key,
                MAX(metric_qualifier) AS metric_qualifier,
                -- How many facts are in the cell at all, kept for the
                -- diagnostic log; the view publishes the number of facts the
                -- VALUE came from (see `_value_input_count`), which is what
                -- `aggregate_components` lists.
                CAST(COUNT(*) AS INTEGER) AS cell_fact_count,
                _value_level,
                -- How many facts the shown value was computed from, and
                -- whether it is one fact's own number. Per-fact context
                -- (uncertainty, generation config, record pointer) is only
                -- true of the cell when it is: a median over several runs has
                -- no single standard error or temperature.
                CAST(COALESCE(
                    COUNT(*) FILTER (
                        WHERE _is_value_row
                          AND evaluator_relationship = 'first_party'
                          AND score IS NOT NULL
                    ),
                    0
                ) AS INTEGER) AS _n_value_first_party,
                CAST(COUNT(*) FILTER (
                    WHERE _is_value_row AND score IS NOT NULL
                ) AS INTEGER) AS _n_value_rows,
                -- Pooled statistics over the value rows, on the PUBLISHED
                -- scale. Only meaningful when every value row is on the same
                -- scale — see `_n_value_scale_classes` below and the
                -- `rep_score` that the outer SELECT derives from the two.
                {_pooled_value_aggregates_sql("score", "_pub")},
                -- How many conversion classes the value rows span. A win rate
                -- published as 1 by one source and as 100 by another is one
                -- measurement on two scales: their published median (50.5) is
                -- a number on no scale at all. More than one class here means
                -- the cell has no single published scale, and the canonical
                -- one is the only scale its rows share.
                -- Counted over the same rows the value came from, so the
                -- first-party preference below applies to the scale question
                -- too: 0 first-party rows means the fallback set decides.
                COALESCE(
                    NULLIF(CAST(COUNT(DISTINCT scale_conversion) FILTER (
                        WHERE _is_value_row
                          AND evaluator_relationship = 'first_party'
                          AND score IS NOT NULL
                    ) AS INTEGER), 0),
                    CAST(COUNT(DISTINCT scale_conversion) FILTER (
                        WHERE _is_value_row AND score IS NOT NULL
                    ) AS INTEGER)
                ) AS _n_value_scale_classes,
                -- Canonical-scale twin, pooled the same way. Stage D already
                -- placed every row on the renamed metric's registry scale, so
                -- this view never re-derives a conversion.
                {_pooled_value_aggregates_sql("score_canonical", "_can")},
                -- The source's own label for the shown number, and how many
                -- distinct labels went into it. A median over rows the source
                -- named differently has no one label, so the view shows none.
                arg_min(metric_source_label, fact_id)
                    FILTER (WHERE _is_value_row AND metric_source_label IS NOT NULL)
                                                         AS _value_metric_source_label,
                CAST(COUNT(DISTINCT metric_source_label) FILTER (
                    WHERE _is_value_row AND metric_source_label IS NOT NULL
                ) AS INTEGER)                            AS _n_value_metric_source_labels,
                -- The facts behind a derived value, for the row's
                -- `aggregate_components`. Ordered by fact_id so the emitted
                -- parquet is byte-stable.
                ARRAY_AGG(struct_pack(
                    evaluation_id          := evaluation_id,
                    composite_slug         := composite_slug,
                    composite_display_name := composite_display_name,
                    score                  := score,
                    normalized_score       := CAST(NULL AS DOUBLE),
                    evaluation_timestamp   := {ts_cast_sql("evaluation_timestamp")},
                    source_name            := org_display,
                    source_type            := source_type,
                    source_organization_name := org_raw,
                    evaluator_relationship := evaluator_relationship
                ) ORDER BY fact_id)
                    FILTER (WHERE _is_value_row AND score IS NOT NULL)
                                                         AS _value_components,
                -- The triple's conversion class: the class the value rows
                -- actually converted under, preferring a converted row over a
                -- flagged one; arg_min on fact_id keeps the pick stable. The
                -- first-party layer comes first for the same reason the value
                -- prefers it — the label has to describe the number shown.
                COALESCE(
                    arg_min(scale_conversion, fact_id) FILTER (
                        WHERE _is_value_row
                          AND evaluator_relationship = 'first_party'
                          AND score_canonical IS NOT NULL),
                    arg_min(scale_conversion, fact_id)
                        FILTER (WHERE _is_value_row AND score_canonical IS NOT NULL),
                    arg_min(scale_conversion, fact_id)
                        FILTER (WHERE _is_value_row AND score IS NOT NULL),
                    arg_min(scale_conversion, fact_id)
                ) AS _scale_conversion_rep,
                BOOL_OR(evaluator_relationship = 'first_party') AS has_first_party,
                BOOL_OR(evaluator_relationship = 'third_party') AS has_third_party,
                -- ORDER BY the distinct expr so the array element order is
                -- run-to-run stable (set content is already deterministic;
                -- only the ordering varied under unordered aggregation).
                ARRAY_AGG(DISTINCT evaluator_relationship ORDER BY evaluator_relationship)
                    FILTER (WHERE evaluator_relationship IS NOT NULL)
                    AS evaluator_relationships,
                ARRAY_AGG(DISTINCT org_display ORDER BY org_display)
                    FILTER (WHERE org_display IS NOT NULL)
                    AS reporting_orgs,
                -- Provenance signals (`is_multi_source`, `first_party_only`)
                -- are computed at the (model, benchmark) level in Stage F.1
                -- and so are constant across all (slice, metric) rows in this
                -- triple — ANY_VALUE is exact.
                --
                -- Comparability signals (variant + cross-party divergence)
                -- are computed at (model, benchmark, slice, metric) in
                -- Stage F.2, so different slices on the same metric carry
                -- different values. The triple-level rollup uses BOOL_OR
                -- for booleans ("does any slice diverge?") and MAX for
                -- magnitudes / org counts ("worst slice"). Threshold + basis
                -- are derived from per-metric metric_config and stay
                -- constant; differing_fields and scores_by_organization
                -- vary per-slice, so the representative is pinned to the
                -- lowest fact_id (arg_min) — ANY_VALUE made the pick
                -- run-to-run unstable and the emitted parquet
                -- byte-nondeterministic. AVG is rounded to 12 decimals to
                -- squash float summation-order noise (~1e-16 flips).
                arg_min(scores_by_organization, fact_id)
                    FILTER (WHERE scores_by_organization IS NOT NULL)
                                                         AS scores_by_organization,
                MAX(is_multi_source)                     AS is_multi_source,
                MAX(first_party_only)                    AS first_party_only,
                -- A triple can span several comparability groups (slices,
                -- which are not part of this grain).
                -- Precedence mixed_scale > no_bounds > ok: the triple is
                -- only as comparable as its least comparable group.
                CASE
                    WHEN BOOL_OR(comparability_status = 'mixed_scale') THEN 'mixed_scale'
                    WHEN BOOL_OR(comparability_status = 'no_bounds')   THEN 'no_bounds'
                    WHEN BOOL_OR(comparability_status = 'ok')          THEN 'ok'
                END                                      AS comparability_status,
                -- BOOL_OR alone drops NULLs, so a group that could not be
                -- assessed next to one that agreed would read FALSE — "we
                -- compared and they agree" about rows nobody compared. A
                -- real divergence still wins; otherwise any non-`ok`
                -- contributor collapses the rollup to NULL.
                {_divergence_rollup_sql("has_variant_divergence")}
                                                         AS has_variant_divergence,
                {_divergence_rollup_sql("has_cross_party_divergence")}
                                                         AS has_cross_party_divergence,
                MAX(variant_divergence_magnitude)        AS variant_divergence_magnitude,
                MAX(variant_divergence_threshold)        AS variant_divergence_threshold,
                MAX(variant_threshold_basis)             AS variant_threshold_basis,
                arg_min(variant_differing_fields, fact_id)
                    FILTER (WHERE variant_differing_fields IS NOT NULL)
                                                         AS variant_differing_fields,
                MAX(cross_party_divergence_magnitude)    AS cross_party_divergence_magnitude,
                MAX(cross_party_divergence_threshold)    AS cross_party_divergence_threshold,
                MAX(cross_party_threshold_basis)         AS cross_party_threshold_basis,
                arg_min(cross_party_differing_fields, fact_id)
                    FILTER (WHERE cross_party_differing_fields IS NOT NULL)
                                                         AS cross_party_differing_fields,
                MAX(cross_party_org_count)                  AS cross_party_org_count,
                BOOL_OR(has_reproducibility_gap)         AS triple_has_repro_gap,
                ROUND(AVG(completeness_score), 12)       AS triple_avg_completeness
            FROM tris_levelled
            GROUP BY composite_slug, model_aggregation_key, benchmark_key,
                     metric_key, protocol_condition, judge_condition, split,
                     _value_level
        ),
        tri_rep_ranked AS (
            -- Pick one representative fact row per triple.
            -- Order: scored rows first → first-party first → lowest
            -- evaluation_id → lowest fact_id. fact_id is the final total
            -- tiebreak: a triple can hold several fact rows sharing the
            -- same evaluation_id (different result_idx) or several
            -- first-party scored rows, leaving evaluation_id ASC tied and
            -- the representative — which sources rep_score and every other
            -- rep_* scalar surfaced on eval_results_view — arbitrary and
            -- run-to-run unstable. fact_id is unique per fact row.
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY composite_slug, model_aggregation_key,
                                 benchmark_key, metric_key, protocol_condition,
                                 judge_condition, split
                    ORDER BY
                        -- A row that did not contribute to the shown value
                        -- cannot represent it; a cell with no value at all
                        -- still keeps a representative for coverage.
                        CASE WHEN _is_value_row THEN 0 ELSE 1 END ASC,
                        CASE WHEN score IS NULL THEN 1 ELSE 0 END ASC,
                        CASE WHEN evaluator_relationship = 'first_party' THEN 0 ELSE 1 END ASC,
                        evaluation_id ASC,
                        fact_id ASC
                ) AS _rep_rank
            FROM tris_levelled
        ),
        tri_rep AS (
            SELECT * FROM tri_rep_ranked WHERE _rep_rank = 1
        )
        SELECT
                ta.*,
                {_pooled_value_pick_sql("_can")} AS _score_canonical,
                {_pooled_aggregation_pick_sql("_pub")} AS _value_aggregation,
                -- The shown number and the scale it is on, decided together.
                -- With one conversion class the cell publishes the source's
                -- own number, as it always has. With several, no published
                -- scale is shared by the rows, so the cell publishes the
                -- canonical-scale value and says `mixed`; `score_published`
                -- goes NULL further down, because no single number was ever
                -- published for this cell.
                CASE WHEN ta._n_value_scale_classes > 1
                     THEN {_pooled_value_pick_sql("_can")}
                     ELSE {_pooled_value_pick_sql("_pub")}
                END                           AS rep_score,
                CASE WHEN ta._n_value_scale_classes > 1
                     THEN 'mixed'
                     ELSE ta._scale_conversion_rep
                END                           AS scale_conversion,
                -- How many facts the shown value came from, after the
                -- first-party preference inside the winning level.
                CASE WHEN ta._n_value_first_party > 0
                     THEN ta._n_value_first_party
                     ELSE ta._n_value_rows
                END                           AS fact_row_count,
                (CASE WHEN ta._n_value_first_party > 0
                      THEN ta._n_value_first_party
                      ELSE ta._n_value_rows
                 END) = 1                     AS _value_single_fact,
                tr.collection_id              AS rep_collection_id,
                tr.evaluation_id              AS rep_evaluation_id,
                tr.fact_id                    AS rep_fact_id,
                tr.retrieved_timestamp        AS rep_retrieved_timestamp,
                tr.evaluation_timestamp       AS rep_evaluation_timestamp,
                tr.benchmark_updated          AS rep_benchmark_updated,
                tr.evaluator_relationship     AS rep_evaluator_relationship,
                tr.is_verified_evaluator      AS rep_is_verified_evaluator,
                tr.provenance_source_type     AS rep_provenance_source_type,
                tr.org_raw                    AS rep_org_raw,
                -- De-aliased evaluator org per row (same name space as
                -- evaluator_names). Consumed by evals_view to build
                -- verified_evaluator_names; surfaced on the view as an
                -- intermediate provenance column.
                tr.org_display                AS rep_org_display,
                tr.source_type                AS rep_source_type,
                tr.source_organization_url    AS rep_source_org_url,
                tr.eval_library_name          AS rep_eval_library_name,
                tr.eval_library_version       AS rep_eval_library_version,
                tr.score_se                   AS rep_score_se,
                tr.score_sd                   AS rep_score_sd,
                tr.score_ci_lower             AS rep_ci_lower,
                tr.score_ci_upper             AS rep_ci_upper,
                tr.score_ci_level             AS rep_ci_level,
                -- Uncertainty on the canonical scale (Stage D): CI endpoints
                -- take the affine conversion, SE and SD its magnitude.
                tr.score_se_canonical         AS rep_score_se_canonical,
                tr.score_sd_canonical         AS rep_score_sd_canonical,
                tr.score_ci_lower_canonical   AS rep_ci_lower_canonical,
                tr.score_ci_upper_canonical   AS rep_ci_upper_canonical,
                -- Source label of the representative published number. Kept
                -- at row grain on purpose: one renamed metric can carry
                -- several source labels, and there is no page-level answer.
                tr.metric_source_label        AS rep_metric_source_label,
                tr.n_samples                  AS rep_n_samples,
                tr.lower_is_better            AS rep_lower_is_better,
                tr.metric_unit                AS rep_metric_unit,
                tr.parent_benchmark_id        AS rep_parent_benchmark_id,
                tr.model_raw                  AS rep_model_raw,
                -- Model-resolution-rework per-row provenance from the
                -- representative fact row (resolver output, Stage C).
                tr.inference_platform         AS rep_inference_platform,
                tr.resolution_source          AS rep_resolution_source,
                tr.resolution_granularity     AS rep_resolution_granularity,
                tr.repro_missing_fields       AS rep_repro_missing_fields,
                tr.repro_populated_count      AS rep_repro_populated_count,
                tr.repro_required_count       AS rep_repro_required_count,
                tr.instance_file_path         AS rep_instance_file_path,
                tr.instance_file_format       AS rep_instance_file_format,
                tr.instance_rows              AS rep_instance_rows,
                tr.source_record_path         AS rep_source_record_path,
                -- Generation config from the representative fact row.
                -- Re-assembled into a STRUCT below to round-trip the shape
                -- the EEE source carried + the frontend's GenerationConfig
                -- TS interface expects.
                tr.temperature                AS rep_temperature,
                tr.top_p                      AS rep_top_p,
                tr.top_k                      AS rep_top_k,
                tr.max_tokens                 AS rep_max_tokens,
                tr.prompt_template            AS rep_prompt_template,
                tr.reasoning                  AS rep_reasoning,
                tr.generation_additional_details AS rep_generation_additional_details
            FROM tri_agg ta
            -- Explicit ON (not USING): protocol_condition is NULL for all
            -- ordinary rows and USING-equality would drop them; IS NOT
            -- DISTINCT FROM matches NULLs.
            JOIN tri_rep tr
              ON tr.composite_slug        = ta.composite_slug
             AND tr.model_aggregation_key = ta.model_aggregation_key
             AND tr.benchmark_key         = ta.benchmark_key
             AND tr.metric_key            = ta.metric_key
             AND tr.protocol_condition    IS NOT DISTINCT FROM ta.protocol_condition
             AND tr.judge_condition       IS NOT DISTINCT FROM ta.judge_condition
             AND tr.split                 IS NOT DISTINCT FROM ta.split
        """
    )

    _log_cells_without_aggregate(con)

    # Headline mapping: exactly one row per (composite, benchmark,
    # metric_key, model) is the page's summary reading, chosen across BOTH
    # condition axes and materialised at fact grain so every direct-fact
    # consumer (subtasks, hierarchy slices, collection context, benchmark
    # dominant conversion, peer ranks) reads the same pick instead of
    # re-deriving it. Computed after condition-grain aggregation and before
    # ranking; non-headline rows stay in the view, unranked.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE fact_headline AS
        WITH eligible AS (
            -- Answer-feedback protocol arms are never headline (the
            -- exclusion the ranking pool has always applied). Unscored rows
            -- stay in: a cell whose only rows are unscored still needs one
            -- representative row so coverage counts keep seeing it. Scored
            -- rows outrank unscored ones in the pick below, so the choice
            -- among scored rows is unaffected.
            SELECT
                composite_slug, benchmark_key, metric_key,
                model_aggregation_key, protocol_condition, judge_condition,
                split,
                rep_score, rep_lower_is_better, scale_conversion, rep_fact_id,
                -- The arm contest below is decided by comparing scores, so
                -- it is decided on the canonical scale. `rep_score` keeps its
                -- job of saying WHETHER an arm was scored at all — a row
                -- whose scale could not be placed (`flagged`) still counts as
                -- read, it just cannot be ranked.
                _score_canonical,
                -- 0 for an undisclosed judge. Only the panel/non-panel
                -- split is ordered on (see the pick below); a single
                -- disclosed judge and an undisclosed row tie here.
                COALESCE(json_array_length(
                    json_extract(judge_condition, '$.judges')), 0) AS _judge_n,
                -- The judges alone, without the source's label for them. Two
                -- labels over one judge set are one judge's coverage of the
                -- page, not two; counting the label with it lets a raw
                -- channel outvote the source's own rescaled copy of itself.
                CAST(json_extract(judge_condition, '$.judges') AS VARCHAR)
                    AS _judge_set
            FROM _erv_tri
            WHERE {protocol_exclusion_sql("protocol_condition")}
        ),
        coverage AS (
            -- How many distinct models each judge SET has a score for on
            -- this page.
            SELECT composite_slug, benchmark_key, metric_key, _judge_set,
                   CAST(COUNT(DISTINCT model_aggregation_key) AS INTEGER)
                       AS _judge_models
            FROM eligible
            WHERE rep_score IS NOT NULL
            GROUP BY 1, 2, 3, 4
        ),
        armed AS (
            SELECT e.*, c._judge_models,
                -- The protocol axis is settled per ARM, not per row: the
                -- existing representative rule ranks a model's protocol
                -- points by score, and the judge rules then choose inside
                -- the winning arm. Comparing raw rows would let the
                -- highest-scoring judge silently win the protocol contest.
                MAX(CASE WHEN COALESCE(e.rep_lower_is_better, FALSE)
                         THEN -e._score_canonical ELSE e._score_canonical END) OVER (
                    PARTITION BY e.composite_slug, e.benchmark_key,
                                 e.metric_key, e.model_aggregation_key,
                                 e.protocol_condition
                ) AS _arm_best,
                -- an arm nobody scored can still be a cell's only arm
                CASE WHEN e.rep_score IS NULL THEN 1 ELSE 0 END AS _unscored
            FROM eligible e
            LEFT JOIN coverage c
                   ON c.composite_slug   = e.composite_slug
                  AND c.benchmark_key    = e.benchmark_key
                  AND c.metric_key       = e.metric_key
                  AND c._judge_set       IS NOT DISTINCT FROM e._judge_set
        ),
        picked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY composite_slug, benchmark_key, metric_key,
                                 model_aggregation_key
                    ORDER BY
                        -- (0) a scored row always represents the cell
                        _unscored ASC,
                        -- (1) protocol axis, existing representative rule:
                        -- an arm we could not read sorts after a known-clean
                        -- one, then the best-scoring arm wins.
                        CASE WHEN COALESCE(json_extract_string(
                                 protocol_condition, '$.feedback'), 'none')
                             = 'unknown' THEN 1 ELSE 0 END ASC,
                        _arm_best DESC NULLS LAST,
                        COALESCE(protocol_condition, '') ASC,
                        -- (2) judge cardinality is a preference for PANELS
                        -- only: a disclosed panel beats its members. A single
                        -- disclosed judge and an undisclosed condition are
                        -- equal here, so (3) coverage decides between them.
                        CASE WHEN _judge_n > 1 THEN 0 ELSE 1 END ASC,
                        CASE WHEN _judge_n > 1 THEN _judge_n ELSE 0 END DESC,
                        -- (3) widest distinct-model coverage on the page,
                        -- per judge set
                        _judge_models DESC,
                        -- (4) the source's own canonical-scale number over a
                        -- converted one
                        CASE WHEN scale_conversion = 'none' THEN 0 ELSE 1 END ASC,
                        -- (5) / (6) deterministic finishers. Split has no
                        -- preference of its own: two cells that differ only
                        -- in split are two measurements, and which one heads
                        -- the page is settled here, deterministically
                        -- (unstated first), not by score.
                        COALESCE(judge_condition, '') ASC,
                        COALESCE(split, '') ASC,
                        rep_fact_id ASC
                ) AS _hl_rank
            FROM armed
        ),
        winners AS (
            SELECT composite_slug, benchmark_key, metric_key,
                   model_aggregation_key, protocol_condition, judge_condition,
                   split
            FROM picked WHERE _hl_rank = 1
        )
        SELECT
            f.fact_id,
            CAST(w.composite_slug IS NOT NULL AND d.fact_id IS NULL AS BOOLEAN)
                AS is_headline
        FROM fact_results f
        LEFT JOIN _erv_scale_copies d ON d.fact_id = f.fact_id
        LEFT JOIN winners w
               ON w.composite_slug        = f.composite_slug
              AND w.benchmark_key         = f.benchmark_key
              AND w.metric_key            = f.metric_key
              AND w.model_aggregation_key = f.model_aggregation_key
              AND w.protocol_condition    IS NOT DISTINCT FROM f.protocol_condition
              AND w.judge_condition       IS NOT DISTINCT FROM f.judge_condition
              AND w.split                 IS NOT DISTINCT FROM f.split
        """
    )

    con.execute(
        f"""
        CREATE TABLE eval_results_view AS
        WITH benchmark_tags AS (
            SELECT
                composite_slug, benchmark_id,
                resolve_benchmark_tags_udf(display_name, benchmark_id) AS derived_tags
            FROM benchmarks
        ),
        joined AS (
            SELECT
                ta.*,
                COALESCE(fh.is_headline, FALSE) AS is_headline,
                m.model_id                    AS m_model_id,
                m.display_name                AS m_display_name,
                m.developer                   AS m_developer,
                m.org_display_name            AS m_org_display_name,
                m.architecture                AS m_architecture,
                m.params_billions             AS m_params_billions,
                m.release_date                AS m_release_date,
                m.open_weights                AS m_open_weights,
                m.input_modalities            AS m_input_modalities,
                m.output_modalities           AS m_output_modalities,
                -- The variant tail joins the display name too, so three
                -- readings of one metric are distinguishable on the page.
                cmet.display_name
                    || COALESCE(' (' || ta.metric_qualifier || ')', '')
                                              AS metric_display_name,
                -- Registry bounds behind the view's min_score / max_score /
                -- score_normalized. An infinite side (the registry's
                -- "unbounded by definition") is folded to NULL here so
                -- those columns read exactly as they did for a NULL bound
                -- ([0, 1] defaults) instead of a 0/inf range that would
                -- normalise every score to 0.
                CASE WHEN isinf(cmet.min_score) THEN NULL ELSE cmet.min_score END AS cmet_min_score,
                CASE WHEN isinf(cmet.max_score) THEN NULL ELSE cmet.max_score END AS cmet_max_score,
                b.parent_benchmark_id         AS b_parent_benchmark_id,
                b.composite_display_name      AS b_composite_display_name,
                b.family_id                   AS b_family_id,
                b.family_display_name         AS b_family_display_name,
                b.is_slice                    AS b_is_slice,
                bt.derived_tags                AS b_derived_tags,
                -- Pulled through so eval_results_view.source_data uses the
                -- same fill rule as evals_view.source_data (was previously
                -- hard-coded NULL on this view, causing schema drift).
                b.display_name                AS b_display_name,
                b.dataset_repo                AS b_dataset_repo,
                b.data_format                 AS b_data_format,
                b.resources                   AS b_resources,
                -- Parent benchmark's display name (dim self-join below).
                pb.display_name               AS pb_display_name
            FROM _erv_tri ta
            -- Join keys are root-grain. `models.model_key` is the
            -- transitive root id; `benchmarks.benchmark_id` and
            -- `canonical_metrics.id` are canonical ids — the LEFT JOIN
            -- harmlessly returns NULLs for raw-only rows that have no
            -- canonical dim entry, and per-row display falls back to
            -- the raw string baked into the key.
            LEFT JOIN models m              ON m.model_key    = ta.model_aggregation_key
            LEFT JOIN benchmarks b          ON b.composite_slug = ta.composite_slug
                                            AND b.benchmark_id  = ta.benchmark_key
            LEFT JOIN benchmark_tags bt ON bt.composite_slug = ta.composite_slug
                                      AND bt.benchmark_id  = ta.benchmark_key
            -- Self-join the benchmarks dim on the row's parent so slice
            -- rows can surface the parent's actual display name (the dim
            -- already carries the parent row per composite — phantom
            -- roots included).
            LEFT JOIN benchmarks pb         ON pb.composite_slug = ta.composite_slug
                                            AND pb.benchmark_id  = b.parent_benchmark_id
            -- `metric_key` is already the renamed identity (Stage C/D), so
            -- this IS the effective metric's registry row — no fold
            -- re-derivation, and no second metrics join.
            LEFT JOIN canonical_metrics cmet ON cmet.id       = ta.metric_base_key
            -- The headline pick is constant across a condition row's fact
            -- rows, so the representative row's flag IS the row's flag.
            LEFT JOIN fact_headline fh ON fh.fact_id = ta.rep_fact_id
        ),
        contexted AS (
            -- Per-fact context only survives when the cell's value IS one
            -- fact's number. A median over MMLU's three extraction filters
            -- has no single standard error, no one temperature and no one
            -- source record: showing the representative fact's would present
            -- one run's setup as the setup behind a pooled figure. Those
            -- fields go NULL and `aggregate_components` lists the inputs
            -- instead. The label follows the same rule with one relaxation:
            -- a pool whose facts all carry the SAME source label keeps it.
            SELECT * REPLACE (
                CASE WHEN _value_single_fact THEN rep_score_se END
                    AS rep_score_se,
                CASE WHEN _value_single_fact THEN rep_score_sd END
                    AS rep_score_sd,
                CASE WHEN _value_single_fact THEN rep_ci_lower END
                    AS rep_ci_lower,
                CASE WHEN _value_single_fact THEN rep_ci_upper END
                    AS rep_ci_upper,
                CASE WHEN _value_single_fact THEN rep_ci_level END
                    AS rep_ci_level,
                CASE WHEN _value_single_fact THEN rep_score_se_canonical END
                    AS rep_score_se_canonical,
                CASE WHEN _value_single_fact THEN rep_score_sd_canonical END
                    AS rep_score_sd_canonical,
                CASE WHEN _value_single_fact THEN rep_ci_lower_canonical END
                    AS rep_ci_lower_canonical,
                CASE WHEN _value_single_fact THEN rep_ci_upper_canonical END
                    AS rep_ci_upper_canonical,
                CASE WHEN _value_single_fact THEN rep_n_samples END
                    AS rep_n_samples,
                CASE WHEN _value_single_fact THEN rep_temperature END
                    AS rep_temperature,
                CASE WHEN _value_single_fact THEN rep_top_p END
                    AS rep_top_p,
                CASE WHEN _value_single_fact THEN rep_top_k END
                    AS rep_top_k,
                CASE WHEN _value_single_fact THEN rep_max_tokens END
                    AS rep_max_tokens,
                CASE WHEN _value_single_fact THEN rep_prompt_template END
                    AS rep_prompt_template,
                CASE WHEN _value_single_fact THEN rep_reasoning END
                    AS rep_reasoning,
                CASE WHEN _value_single_fact
                     THEN rep_generation_additional_details END
                    AS rep_generation_additional_details,
                CASE WHEN _value_single_fact THEN rep_source_record_path END
                    AS rep_source_record_path,
                CASE WHEN _value_single_fact THEN rep_instance_file_path END
                    AS rep_instance_file_path,
                CASE WHEN _value_single_fact THEN rep_instance_file_format END
                    AS rep_instance_file_format,
                CASE WHEN _value_single_fact THEN rep_instance_rows END
                    AS rep_instance_rows,
                CASE WHEN _n_value_metric_source_labels = 1
                     THEN _value_metric_source_label END
                    AS rep_metric_source_label
            )
            FROM joined
        ),
        ranked AS (
            -- Rank the headline pool within (composite_slug, benchmark_key,
            -- metric_key), honouring lower_is_better, ON THE CANONICAL
            -- SCALE: sources within one partition can publish mixed units
            -- (Scale SEAL hle percents next to AISI fractions) and a
            -- raw-score rank would order a 0.625 below a 2.72(%).
            -- `is_headline` already carries the answer-feedback exclusion
            -- and the one-row-per-model pick across both condition axes; an
            -- unscored headline row (a cell nobody scored keeps one, so
            -- coverage counts still see it) and rows whose scale is 'flagged'
            -- (unplaceable on the canonical scale) leave the pool on top
            -- of that. Rows outside the pool get position=NULL; `total`
            -- counts pool rows only.
            SELECT *,
                CASE
                    WHEN NOT (is_headline AND rep_score IS NOT NULL
                              AND scale_conversion IS DISTINCT FROM 'flagged')
                        THEN NULL
                    ELSE CAST(ROW_NUMBER() OVER (
                        PARTITION BY composite_slug, benchmark_key, metric_key,
                                     (is_headline AND rep_score IS NOT NULL
                                      AND scale_conversion IS DISTINCT FROM 'flagged')
                        ORDER BY
                            CASE WHEN _score_canonical IS NULL THEN 1 ELSE 0 END ASC,
                            CASE WHEN COALESCE(rep_lower_is_better, FALSE)
                                 THEN _score_canonical
                                 ELSE -_score_canonical
                            END ASC,
                            model_aggregation_key ASC
                    ) AS INTEGER)
                END AS position,
                CAST(SUM(CASE WHEN is_headline AND rep_score IS NOT NULL
                              AND scale_conversion IS DISTINCT FROM 'flagged'
                         THEN 1 ELSE 0 END)
                     OVER (
                    PARTITION BY composite_slug, benchmark_key, metric_key
                ) AS INTEGER) AS total
            FROM contexted
        )
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            url_encode_udf(composite_slug || '/' || benchmark_key) AS evaluation_id,
            metric_summary_id_udf(benchmark_key, metric_key)       AS metric_summary_id,
            composite_slug,
            b_composite_display_name                             AS composite_display_name,
            benchmark_key                                        AS benchmark_id,
            b_family_id                                          AS family_id,
            b_family_display_name                                AS family_display_name,
            COALESCE(b_is_slice, FALSE)                          AS is_slice,
            -- parent_benchmark_id mirrors the contract on
            -- the comparison-index sidecar:
            -- null for roots, the parent benchmark id for slices.
            -- The dim sometimes stores parent_benchmark_id == benchmark_id
            -- for roots, so gate on is_slice rather than trusting the raw
            -- column value.
            CASE WHEN COALESCE(b_is_slice, FALSE)
                 THEN b_parent_benchmark_id
                 ELSE NULL END                                   AS parent_benchmark_id,
            -- The parent benchmark's own display name (NOT the composite
            -- label) — slice-fold titles read this so cross-benchmark
            -- suites don't title groups with the suite name. Same
            -- fallback-to-id rule as the dim's display_name.
            CASE WHEN COALESCE(b_is_slice, FALSE)
                 THEN COALESCE(pb_display_name, b_parent_benchmark_id)
                 ELSE NULL END                                   AS parent_benchmark_display_name,
            metric_key                                           AS metric_id,
            -- The registry id behind `metric_id` once a scoring-variant tail
            -- is appended, plus the tail itself. Equal to `metric_id` /
            -- NULL on every row that carries no qualifier.
            metric_base_key                                      AS metric_base_id,
            metric_qualifier,
            -- Which submitted aggregation level produced this row's value:
            -- `root` the source's own benchmark total, `subgroup` its group
            -- rollups, `single` a lone observation, `derived` a rollup this
            -- pipeline computed from slice children, `none` a cell left
            -- without a value because only task-level rows exist.
            _value_level                                         AS value_level,
            -- How the shown number was computed from its inputs, and — on a
            -- suite row rolled up from registry task children — how much of
            -- the expected task set went into it. NULL on an ordinary row:
            -- there is no task set to be complete about.
            _value_aggregation                                   AS value_aggregation,
            CAST(NULL AS INTEGER)                                AS children_present,
            CAST(NULL AS INTEGER)                                AS children_expected,
            model_aggregation_key                                AS model_key,
            m_model_id                                           AS model_id,
            url_encode_udf(model_aggregation_key)                AS model_route_id,

            -- model_info: denormalised display context. `id` reflects the
            -- canonical id when known and falls back to the raw source name
            -- so unresolved models still expose a stable identifier.
            -- `developer` falls through canonical org → free-text developer
            -- → raw HF id prefix. The third tier is a safety net for rows
            -- whose model_aggregation_key didn't match canonical_models
            -- (stale registry snapshot, casing-mismatched alias, brand-new
            -- model not yet synced) — without it those rows show NULL
            -- developer even when the raw value clearly carries an org.
            -- `review_status='unresolved'` on `models` still distinguishes
            -- these from canonically-resolved rows.
            CAST({{
                'name':              COALESCE(m_display_name, rep_model_raw),
                'id':                COALESCE(model_id, rep_model_raw),
                'developer':         COALESCE(
                    m_org_display_name,
                    m_developer,
                    CASE
                        WHEN rep_model_raw LIKE '%/%'
                             AND length(split_part(rep_model_raw, '/', 1)) > 0
                        THEN split_part(rep_model_raw, '/', 1)
                        ELSE NULL
                    END
                ),
                -- Per-run serving platform from the resolver output
                -- (model-resolution-rework). NULL when the resolution
                -- carried no platform signal.
                'inference_platform': rep_inference_platform,
                'inference_engine':   NULL,
                'model_version':     NULL,
                'architecture':      m_architecture,
                -- parameter_count: prefer the canonical's params, else
                -- best-effort regex from the raw HF id (e.g.
                -- 'Llama-3-OffsetBias-RM-8B' → '8B', 'Mixtral-8x7B' →
                -- '8x7B'). Anchored to delimiters so digits inside
                -- other tokens don't match. `K` is excluded to avoid
                -- context-length false positives ('phi-3-mini-4k').
                'parameter_count':   COALESCE(
                    CASE WHEN m_params_billions IS NOT NULL
                         THEN CAST(m_params_billions AS VARCHAR) || 'B'
                         ELSE NULL END,
                    NULLIF(upper(regexp_extract(
                        rep_model_raw,
                        '(?:^|[/_\\- ])((?:\\d+x)?\\d+(?:\\.\\d+)?[BbMm])(?:[/_\\- ]|$)',
                        1
                    )), '')
                ),
                -- release_date: prefer the canonical's date, else
                -- best-effort regex on the raw HF id's snapshot suffix.
                --   trailing `-YYYY-MM-DD`            → that date
                --   trailing `-YYYYMMDD` (compact)    → reformat as YYYY-MM-DD
                --   trailing `-YYYY-MM`               → year-month
                -- Bare 4-digit MMDD codes ('kimi-k2-0905') are skipped —
                -- no year context to ground them. Only fires when the
                -- canonical date is NULL, never overrides registry data.
                'release_date':      COALESCE(
                    m_release_date,
                    CASE
                        WHEN regexp_matches(rep_model_raw, '-20\\d{{2}}-\\d{{2}}-\\d{{2}}$')
                            THEN regexp_extract(rep_model_raw, '-(20\\d{{2}}-\\d{{2}}-\\d{{2}})$', 1)
                        WHEN regexp_matches(rep_model_raw, '-20\\d{{6}}$')
                            THEN regexp_replace(
                                rep_model_raw,
                                '.*-(20\\d{{2}})(\\d{{2}})(\\d{{2}})$',
                                '\\1-\\2-\\3'
                            )
                        WHEN regexp_matches(rep_model_raw, '-20\\d{{2}}-\\d{{2}}$')
                            THEN regexp_extract(rep_model_raw, '-(20\\d{{2}}-\\d{{2}})$', 1)
                        ELSE NULL
                    END
                ),
                'model_url':         NULL,
                'open_weights':      m_open_weights,
                'modalities':        {{
                    'input':  m_input_modalities,
                    'output': m_output_modalities
                }}
            }} AS STRUCT(
                name VARCHAR, id VARCHAR, developer VARCHAR,
                inference_platform VARCHAR, inference_engine VARCHAR,
                model_version VARCHAR, architecture VARCHAR,
                parameter_count VARCHAR, release_date VARCHAR,
                model_url VARCHAR,
                open_weights BOOLEAN,
                modalities STRUCT(input VARCHAR[], output VARCHAR[])
            )) AS model_info,

            -- generation_config: re-assembled from the representative fact
            -- row's exploded fields. Mirrors `lib/benchmark-schema.ts`'s
            -- `GenerationConfig` interface — `generation_args` carries the
            -- decoder knobs (temperature/top_p/top_k/max_tokens/reasoning),
            -- `prompt_template` is the surface form, `additional_details`
            -- is the EEE catch-all JSON. `num_few_shot` isn't tracked in
            -- the producer's flattening (Stage D drops it); reserved as
            -- NULL until upstream EEE rows surface it.
            CAST({{
                'num_few_shot':       NULL,
                'generation_args':    {{
                    'temperature': rep_temperature,
                    'top_p':       rep_top_p,
                    'top_k':       rep_top_k,
                    'max_tokens':  rep_max_tokens,
                    'reasoning':   rep_reasoning
                }},
                'additional_details': rep_generation_additional_details,
                'prompt_template':    rep_prompt_template
            }} AS STRUCT(
                num_few_shot       INTEGER,
                generation_args    STRUCT(
                    temperature DOUBLE,
                    top_p       DOUBLE,
                    top_k       DOUBLE,
                    max_tokens  INTEGER,
                    reasoning   BOOLEAN
                ),
                additional_details VARCHAR,
                prompt_template    VARCHAR
            )) AS generation_config,

            metric_display_name,
            rep_metric_unit                                       AS metric_unit,
            rep_lower_is_better                                   AS lower_is_better,
            b_derived_tags                                        AS derived_tags,
            COALESCE(cmet_min_score, 0)                           AS min_score,
            COALESCE(cmet_max_score, 1)                           AS max_score,
            -- Normalised against the renamed metric's registry bounds, on
            -- the CANONICAL score: normalising a published 1-10 WildBench
            -- number against [0, 1] would clamp every model to 1.
            CASE
                -- A cell with no value has no normalised value either. Without
                -- this guard the clamp turns a NULL score into 1.0 — a cell the
                -- pipeline declined to score would read as a perfect one.
                WHEN _score_canonical IS NULL THEN NULL
                -- No registry bounds (absent, or infinite by definition) means
                -- no scale to normalise against, and there is no default one:
                -- clamping an Arena Elo of 945-1620 into [0, 1] reported every
                -- Arena model as a perfect 1.0. NULL says "not normalisable",
                -- and AVG skips it, so the page average covers only the rows
                -- that really have a scale.
                WHEN cmet_min_score IS NULL OR cmet_max_score IS NULL THEN NULL
                WHEN (cmet_max_score - cmet_min_score) <= 0 THEN 0
                WHEN COALESCE(rep_lower_is_better, FALSE)
                    THEN GREATEST(0, LEAST(1,
                        1.0 - (_score_canonical - cmet_min_score)
                              / (cmet_max_score - cmet_min_score)))
                ELSE GREATEST(0, LEAST(1,
                    (_score_canonical - cmet_min_score)
                    / (cmet_max_score - cmet_min_score)))
            END                                                   AS score_normalized,
            regexp_replace(
                regexp_replace(metric_summary_id_udf(benchmark_key, metric_key),
                    '_(stderr|std_err|standard_error)$', '', 'i'),
                '_(acc|accuracy|score|value|result)$', '', 'i'
            )                                                     AS metric_pair_key,

            -- Displayed score. A `curated` conversion is a registry-stated
            -- fact about the source's scale (WildBench's 1-10 rating IS
            -- wb-score on [0, 1]), so the canonical number is the one to
            -- show and the published one moves to `score_published`. Every
            -- other class — including a detected div100/mul100 — keeps
            -- publishing the source's own number, as it always has.
            -- Uncertainty follows the same rule, from the canonical columns.
            -- A `mixed` cell already carries the canonical value in
            -- `rep_score` (its rows share no published scale), so `score` is
            -- that value and `score_published` is NULL — there is no one
            -- number the sources published for this cell.
            CASE WHEN scale_conversion = 'curated'
                 THEN _score_canonical ELSE rep_score END       AS score,
            CASE WHEN scale_conversion = 'mixed'
                 THEN NULL ELSE rep_score END                   AS score_published,
            CAST({{
                'score':             CASE WHEN scale_conversion = 'curated'
                                          THEN _score_canonical ELSE rep_score END,
                'standard_error':    CASE WHEN scale_conversion = 'curated'
                                          THEN rep_score_se_canonical
                                          ELSE rep_score_se END,
                'standard_deviation': CASE WHEN scale_conversion = 'curated'
                                          THEN rep_score_sd_canonical
                                          ELSE rep_score_sd END,
                'sample_size':       rep_n_samples,
                'confidence_interval': {{
                    'lower':             CASE WHEN scale_conversion = 'curated'
                                              THEN rep_ci_lower_canonical
                                              ELSE rep_ci_lower END,
                    'upper':             CASE WHEN scale_conversion = 'curated'
                                              THEN rep_ci_upper_canonical
                                              ELSE rep_ci_upper END,
                    'confidence_level':  rep_ci_level
                }}
            }} AS STRUCT(
                score DOUBLE, standard_error DOUBLE, standard_deviation DOUBLE,
                sample_size INTEGER,
                confidence_interval STRUCT(
                    lower DOUBLE, upper DOUBLE, confidence_level DOUBLE
                )
            )) AS score_details,
            fact_row_count,

            position,
            total,
            CASE
                WHEN total IS NULL OR total <= 1 OR position IS NULL THEN NULL
                ELSE 1.0 - (position - 1.0) / (total - 1.0)
            END AS percentile,

            -- evaluation_timestamp is the actual eval-run time
            -- (sourced from EEE evaluation_timestamp, NOT
            -- retrieved_timestamp which is just our scrape time).
            -- ts_cast_sql handles both ISO date-time strings and
            -- Unix-epoch numerics — upstream EEE sources emit both
            -- forms. NULL when the source carries no eval-run timestamp.
            {ts_cast_sql("rep_evaluation_timestamp")}             AS evaluation_timestamp,
            -- benchmark_updated: when the source last refreshed the
            -- benchmark itself (vs when this specific eval ran).
            -- Carried on a subset of EEE records; NULL elsewhere.
            {ts_cast_sql("rep_benchmark_updated")}                AS benchmark_updated,
            -- retrieved_timestamp preserved for diagnostics — when the
            -- snapshot pipeline scraped this record. Frontend should
            -- prefer evaluation_timestamp for "Updated"/eval-date UX.
            {ts_cast_sql("rep_retrieved_timestamp")}              AS retrieved_timestamp,

            CAST({{
                'source_name':              rep_org_raw,
                'source_type':              rep_source_type,
                'source_organization_name': rep_org_raw,
                'source_organization_url':  rep_source_org_url,
                'evaluator_relationship':   rep_evaluator_relationship,
                'source_url':               NULL,
                'publication_date':         NULL
            }} AS STRUCT(
                source_name VARCHAR, source_type VARCHAR,
                source_organization_name VARCHAR, source_organization_url VARCHAR,
                evaluator_relationship VARCHAR, source_url VARCHAR,
                publication_date DATE
            )) AS source_metadata,

            -- source_data: populated from the benchmark dim (same fill
            -- rule as evals_view.source_data). Was previously hard-coded
            -- NULL here, which was schema drift across the two views.
            -- The per-row EEE evaluation_results[].source_data isn't yet
            -- threaded through fact_results, so we surface the
            -- benchmark-level dataset metadata as a substitute.
            CAST({{
                'dataset_name':    b_display_name,
                'source_type':     b_data_format,
                'hf_repo':         b_dataset_repo,
                'hf_split':        NULL,
                'samples_number':  NULL,
                'url':             b_resources,
                'dataset_url':     NULL,
                'dataset_version': NULL
            }} AS STRUCT(
                dataset_name VARCHAR, source_type VARCHAR, hf_repo VARCHAR,
                hf_split VARCHAR, samples_number INTEGER, url VARCHAR[],
                dataset_url VARCHAR, dataset_version VARCHAR
            )) AS source_data,

            -- Legacy slot: under v1 this pointed at a processed
            -- card_backend record JSON. v2 emits Parquet, not per-row
            -- JSON, so there is no such artifact — left NULL. The raw
            -- upstream record is surfaced as eee_record_url below.
            CAST(NULL AS VARCHAR) AS source_record_url,

            -- Deep-link back to the raw EEE source record this triple's
            -- representative row was derived from. Built from the
            -- repo-relative path carried since Stage A; NULL when the
            -- representative row has no recorded path.
            CASE
                WHEN rep_source_record_path IS NOT NULL
                THEN 'https://huggingface.co/datasets/{eee_repo}/resolve/{eee_rev}/'
                     || rep_source_record_path
                ELSE NULL
            END AS eee_record_url,

            CAST({{
                'name':    rep_eval_library_name,
                'version': rep_eval_library_version,
                'fork':    NULL
            }} AS STRUCT(name VARCHAR, version VARCHAR, fork VARCHAR)) AS eval_library,

            evaluator_relationships,
            has_first_party,
            has_third_party,
            -- Curated badge flag for this triple's displayed (representative)
            -- row: was the evaluation submitted by the org that ran it. Carried
            -- from fact_results via the representative fact row.
            rep_is_verified_evaluator AS is_verified_evaluator,
            -- Per-row de-aliased evaluator org; building block for
            -- evals_view.verified_evaluator_names (not read directly by the UI).
            rep_org_display AS evaluator_display_name,
            CASE
                WHEN has_first_party AND has_third_party THEN 'both'
                WHEN has_first_party                     THEN 'self'
                ELSE                                          'third'
            END AS coverage_cell,
            reporting_orgs,
            scores_by_organization,

            is_summary_score_udf(metric_id, rep_parent_benchmark_id, benchmark_id)
                AS is_summary_score,
            rep_parent_benchmark_id AS summary_score_for,
            -- The facts the shown value was computed from, listed whenever it
            -- is a median over more than one of them. A one-fact value needs
            -- no component list: the row's own context IS the fact's.
            CASE WHEN _value_single_fact
                 THEN CAST(NULL AS {aggregate_components_type})
                 ELSE CAST(_value_components AS {aggregate_components_type})
            END AS aggregate_components,

            triple_has_repro_gap        AS has_reproducibility_gap,
            triple_avg_completeness     AS completeness_score,
            is_multi_source,
            first_party_only,
            comparability_status,
            has_variant_divergence,
            has_cross_party_divergence,

            CAST({{
                'reproducibility_gap': {{
                    'missing_fields':  rep_repro_missing_fields,
                    'populated_count': rep_repro_populated_count,
                    'required_count':  rep_repro_required_count
                }},
                'provenance': {{
                    'source_type':            rep_provenance_source_type,
                    'evaluator_relationship': rep_evaluator_relationship,
                    'organization_name':      rep_org_raw
                }},
                'comparability_status': comparability_status,
                'variant_divergence': {{
                    'has_divergence':   has_variant_divergence,
                    'magnitude':        variant_divergence_magnitude,
                    'threshold':        variant_divergence_threshold,
                    'basis':            variant_threshold_basis,
                    'differing_fields': variant_differing_fields
                }},
                'cross_party_divergence': {{
                    'has_divergence':     has_cross_party_divergence,
                    'magnitude':          cross_party_divergence_magnitude,
                    'threshold':          cross_party_divergence_threshold,
                    'basis':              cross_party_threshold_basis,
                    'differing_fields':   cross_party_differing_fields,
                    'organization_count': cross_party_org_count
                }}
            }} AS {eval_annotation_struct_type}) AS evalcards_annotations,

            rep_instance_file_path   AS instance_file_path,
            rep_instance_file_format AS instance_file_format,
            rep_instance_rows        AS instance_rows,

            -- Fetchable URL for the per-instance sample file, built from the
            -- repo-relative path Stage A normalised. Same repo + revision the
            -- `eee_record_url` deep-link uses, so a snapshot's record links
            -- and its sample links always address the same upstream state.
            CASE
                WHEN rep_instance_file_path IS NOT NULL
                THEN 'https://huggingface.co/datasets/{eee_repo}/resolve/{eee_rev}/'
                     || rep_instance_file_path
                ELSE NULL
            END AS instance_file_url,

            -- Merged-view columns (spec P2/P3). `score_canonical` is on the
            -- renamed metric's registry scale; raw `score` is never
            -- overwritten. Flagged rows get NULL (never guessed);
            -- no_bounds rows pass through unconverted.
            -- `metric_id_effective` is retained as an alias of the view's
            -- `metric_id`: rename-at-resolution made them one identity.
            metric_key               AS metric_id_effective,
            scale_conversion,
            _score_canonical         AS score_canonical,
            rep_score_se_canonical   AS score_se_canonical,
            rep_score_sd_canonical   AS score_sd_canonical,
            rep_ci_lower_canonical   AS score_ci_lower_canonical,
            rep_ci_upper_canonical   AS score_ci_upper_canonical,
            -- The source's own label for the representative published
            -- number, and the LLM-judge identity behind it. Both are row
            -- grain: a renamed metric can carry several of each.
            rep_metric_source_label  AS metric_source_label,
            judge_condition,
            -- Summary eligibility: exactly one TRUE per (composite,
            -- benchmark, metric_id, model). Non-headline rows keep NULL
            -- position/total/percentile and are excluded from every
            -- page-level rollup; they stay in the view so a judge or
            -- protocol arm is still readable next to the headline reading.
            is_headline,

            -- Collections (collections spec): submission-channel tag
            -- (representative fact row's; never NULL on fact rows) and this
            -- row's protocol point — NULL for ordinary rows, canonical
            -- sorted-key JSON for collection-adapter rows. One view row per
            -- protocol point.
            rep_collection_id        AS collection_id,
            protocol_condition,
            -- The dataset split this row's facts scored (Stage D, stated or
            -- inherited from the record's parts); the third condition column
            -- of the row grain. NULL when no fact in the cell stated one.
            split
        FROM ranked
        ORDER BY metric_summary_id, model_key, protocol_condition,
                 judge_condition, split, is_headline DESC, rep_fact_id
        """
    )

    _materialise_slice_parent_rows(con, snapshot_id, aggregate_components_type)


def _materialise_slice_parent_rows(
    con, snapshot_id: str, aggregate_components_type: str
) -> None:
    """Give a benchmark whose task variants are registry slice children a
    row of its own, averaged over those children — but only when they are all
    there.

    When the registry models a suite's variants as separate canonical
    benchmarks carrying `parent_benchmark_id` (BFCL-v3's 14 categories,
    RealGuardrails' three sub-benchmarks), every fact resolves to a child and
    the parent has none. `evals_view` drops fact-less shells, so the suite
    disappears from the product entirely — 14 category pages and no BFCL-v3.

    What makes a suite number sayable here, and not for a loose pile of task
    rows, is that the registry states the expected task set: the parent's
    children ARE the benchmark. So the row is emitted per (composite, model,
    metric, protocol condition, judge condition) only when EVERY registry
    child of that parent has a value in that cell. `children_expected` and
    `children_present` are published on the row so a reader sees the
    denominator, and a cell that falls short is logged as partial coverage and
    gets no row: a mean over 13 of 14 categories is a different quantity from
    BFCL-v3, and nothing on the page would say so.

    The value is a MEAN, not a median. These are the parts of one benchmark
    being combined into a whole, not repeated readings of one quantity, and
    the average of the parts is what a suite score is. It is weighted by
    `n_samples` when every child carries one — a 50-item category should not
    count as much as a 2,000-item one — and unweighted otherwise;
    `value_aggregation` says which. `aggregate_components` lists the children,
    `fact_row_count` is how many there were, and every per-fact context field
    is NULL: no fact of the parent's own exists to describe.

    A parent that DOES publish this cell itself keeps its own number and gains
    nothing here (TruthfulQA-multilingual submits a multilingual total next to
    its 31 language children), so no child is ever counted twice into a number
    the source already reported. That suppression is per condition — a parent
    row under one judge says nothing about the same cell under another.
    """
    sid = snapshot_id_to_sql(snapshot_id)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _erv_parent_candidates AS
        WITH child_of AS (
            -- The expected task set, straight from the registry: every direct
            -- child of the parent, whether or not this composite happens to
            -- carry data for it. Reading the composite-scoped `benchmarks`
            -- dim instead would make the denominator whatever arrived, which
            -- is exactly the question the coverage gate asks.
            --
            -- Membership role. A child the registry marks
            -- `metadata.role = "aggregate"` is not a part of the benchmark —
            -- it is the source's own rollup OVER other children, so averaging
            -- it with them counts those results twice. BFCL-v3 is the case:
            -- `bfcl-v3-single-turn` is Swiss AI's aggregate over the 13
            -- single-turn categories, published next to them. It keeps its
            -- own child row and its own page; it just does not enter the
            -- parent's expected set, its coverage, or its mean. A
            -- `diagnostic` child is excluded for the opposite reason: it
            -- measures a different quantity under the parent's name (BFCL's
            -- format-sensitivity spread), so averaging it with the parts
            -- would mix units.
            SELECT id            AS child_benchmark_id,
                   parent_benchmark_id
            FROM canonical_benchmarks
            WHERE parent_benchmark_id IS NOT NULL
              AND parent_benchmark_id <> id
              AND COALESCE(json_extract_string(metadata, '$.role'), '')
                  NOT IN ('aggregate', 'diagnostic')
        ),
        expected AS (
            SELECT parent_benchmark_id,
                   CAST(COUNT(*) AS INTEGER) AS children_expected
            FROM child_of
            GROUP BY 1
        ),
        child_cells AS (
            SELECT erv.composite_slug, c.parent_benchmark_id,
                   erv.model_key, erv.metric_id, erv.metric_base_id,
                   erv.metric_qualifier,
                   erv.protocol_condition, erv.judge_condition, erv.split,
                   erv.benchmark_id, erv.evaluation_id, erv.model_info,
                   erv.score, erv.score_canonical, erv.score_normalized,
                   erv.score_details.sample_size AS n_samples,
                   erv.lower_is_better, erv.metric_unit, erv.metric_display_name,
                   erv.min_score, erv.max_score, erv.scale_conversion,
                   erv.composite_display_name, erv.evaluation_timestamp,
                   erv.evaluator_relationships, erv.has_first_party,
                   erv.has_third_party, erv.reporting_orgs,
                   erv.evaluator_display_name, erv.is_verified_evaluator,
                   erv.collection_id, erv.source_metadata
            FROM eval_results_view erv
            JOIN child_of c
              ON c.child_benchmark_id = erv.benchmark_id
            -- Every condition row of the child, not only its headline one:
            -- the parent is built per condition, so the children that count
            -- towards a judge are the ones that published under that judge.
            WHERE erv.score IS NOT NULL
        ),
        rolled AS (
            -- One parent row per (composite, model, metric, protocol
            -- condition, judge condition, split) — the same grain every other
            -- row on this view has. The conditions are part of the
            -- measurement, not decoration: MT-Bench's children judged by
            -- three different judges are three readings, and medianing them
            -- into one row under a blank judge presents a number no judge
            -- produced and feeds it to the merged best-result and the
            -- comparison index as though it were comparable. Children run on
            -- different splits likewise never average into one suite value.
            SELECT
                composite_slug, parent_benchmark_id, model_key, metric_id,
                protocol_condition, judge_condition, split,
                MAX(metric_base_id)       AS metric_base_id,
                MAX(metric_qualifier)     AS metric_qualifier,
                -- Same scale rule as the cell rollup: children published on
                -- different scales share only the canonical one, and an
                -- average over their published numbers lands on no scale.
                CAST(COUNT(DISTINCT scale_conversion) AS INTEGER)
                                          AS _n_scale_classes,
                -- The suite value: the mean of its parts, sample-weighted
                -- when every part says how many samples it covers.
                (COUNT(*) = COUNT(n_samples) AND COALESCE(SUM(n_samples), 0) > 0)
                                          AS _weighted,
                AVG(score)                AS _score_published_mean,
                SUM(score * n_samples) / NULLIF(SUM(n_samples), 0)
                                          AS _score_published_wmean,
                AVG(score_canonical)      AS _score_canonical_mean,
                SUM(score_canonical * n_samples) / NULLIF(SUM(n_samples), 0)
                                          AS _score_canonical_wmean,
                AVG(score_normalized)     AS _score_normalized_mean,
                SUM(score_normalized * n_samples) / NULLIF(SUM(n_samples), 0)
                                          AS _score_normalized_wmean,
                CAST(COUNT(DISTINCT benchmark_id) AS INTEGER) AS children_present,
                CAST(COUNT(*) AS INTEGER) AS fact_row_count,
                BOOL_OR(COALESCE(lower_is_better, FALSE)) AS lower_is_better,
                MAX(metric_unit)          AS metric_unit,
                MAX(metric_display_name)  AS metric_display_name,
                MAX(min_score)            AS min_score,
                MAX(max_score)            AS max_score,
                MAX(composite_display_name) AS composite_display_name,
                arg_min(model_info, benchmark_id)       AS model_info,
                arg_min(scale_conversion, benchmark_id) AS _scale_conversion_rep,
                arg_min(evaluation_timestamp, benchmark_id) AS evaluation_timestamp,
                arg_min(evaluator_relationships, benchmark_id) AS evaluator_relationships,
                arg_min(reporting_orgs, benchmark_id) AS reporting_orgs,
                BOOL_OR(has_first_party)  AS has_first_party,
                BOOL_OR(has_third_party)  AS has_third_party,
                arg_min(evaluator_display_name, benchmark_id) AS evaluator_display_name,
                BOOL_OR(is_verified_evaluator) AS is_verified_evaluator,
                arg_min(collection_id, benchmark_id) AS collection_id,
                arg_min(source_metadata, benchmark_id) AS source_metadata,
                ARRAY_AGG(struct_pack(
                    evaluation_id          := evaluation_id,
                    composite_slug         := composite_slug,
                    composite_display_name := composite_display_name,
                    score                  := score,
                    normalized_score       := score_normalized,
                    evaluation_timestamp   := evaluation_timestamp,
                    source_name            := evaluator_display_name,
                    source_type            := source_metadata.source_type,
                    source_organization_name := source_metadata.source_organization_name,
                    evaluator_relationship := source_metadata.evaluator_relationship
                ) ORDER BY benchmark_id) AS aggregate_components
            FROM child_cells
            GROUP BY 1, 2, 3, 4, 5, 6, 7
            HAVING COUNT(*) > 0
        )
        SELECT r.* EXCLUDE (_score_published_mean, _score_published_wmean,
                            _score_canonical_mean, _score_canonical_wmean,
                            _score_normalized_mean, _score_normalized_wmean,
                            _n_scale_classes, _scale_conversion_rep, _weighted),
            e.children_expected,
            CASE WHEN r._weighted THEN '{AGG_WEIGHTED_MEAN}'
                 ELSE '{AGG_MEAN}' END AS value_aggregation,
            (CASE WHEN r._weighted THEN r._score_canonical_wmean
                  ELSE r._score_canonical_mean END) AS score_canonical,
            (CASE WHEN r._weighted THEN r._score_normalized_wmean
                  ELSE r._score_normalized_mean END) AS score_normalized,
            CASE WHEN r._n_scale_classes > 1
                 THEN (CASE WHEN r._weighted THEN r._score_canonical_wmean
                            ELSE r._score_canonical_mean END)
                 ELSE (CASE WHEN r._weighted THEN r._score_published_wmean
                            ELSE r._score_published_mean END)
            END AS score,
            CASE WHEN r._n_scale_classes > 1
                 THEN 'mixed' ELSE r._scale_conversion_rep
            END AS scale_conversion,
            -- Exactly one headline row per (composite, parent, metric, model),
            -- the invariant every page-level rollup relies on. Among a
            -- parent's condition rows the one built from the most children
            -- speaks for the suite; the rest stay readable next to it,
            -- unranked, like any other condition row. A parent that reports
            -- the cell itself under ANY condition already has its headline
            -- row (`fact_headline`), so no derived row may claim a second
            -- one: derived rows under the other conditions stay readable,
            -- unranked.
            ROW_NUMBER() OVER (
                PARTITION BY r.composite_slug, r.parent_benchmark_id,
                             r.model_key, r.metric_id
                ORDER BY r.fact_row_count DESC,
                         COALESCE(r.protocol_condition, '') ASC,
                         COALESCE(r.judge_condition, '') ASC,
                         COALESCE(r.split, '') ASC
            ) = 1
            AND NOT EXISTS (
                SELECT 1 FROM eval_results_view erv
                WHERE erv.composite_slug = r.composite_slug
                  AND erv.benchmark_id   = r.parent_benchmark_id
                  AND erv.model_key      = r.model_key
                  AND erv.metric_id      = r.metric_id
            ) AS _is_headline
        FROM rolled r
        JOIN expected e ON e.parent_benchmark_id = r.parent_benchmark_id
        -- A parent that reports this cell itself keeps its own number — but
        -- only for the condition it published it under. A parent row under
        -- one judge says nothing about the same cell under another, so the
        -- suppression is per condition, with NULL-safe equality so ordinary
        -- condition-less rows still match each other.
        --
        -- Split is the one condition where NULL is weaker than a value: an
        -- unstated split on the source's own total is no evidence that the
        -- total and its children ran on different data (TruthfulQA-
        -- multilingual's total sits in its own record, its 31 language rows
        -- in theirs, all stamped `val`), and a derived mean beside it would
        -- count the same results twice. So a bare total suppresses the
        -- derived row under every split; a total that STATES a split
        -- suppresses only its own.
        WHERE NOT EXISTS (
            SELECT 1 FROM eval_results_view erv
            WHERE erv.composite_slug = r.composite_slug
              AND erv.benchmark_id   = r.parent_benchmark_id
              AND erv.model_key      = r.model_key
              AND erv.metric_id      = r.metric_id
              AND erv.protocol_condition IS NOT DISTINCT FROM r.protocol_condition
              AND erv.judge_condition    IS NOT DISTINCT FROM r.judge_condition
              AND (erv.split IS NULL OR erv.split IS NOT DISTINCT FROM r.split)
        )
        """
    )
    # The coverage gate. A mean over 13 of BFCL-v3's 14 categories is a
    # different quantity from BFCL-v3, and nothing on the page would say so,
    # so the short cell gets no row at all — and a line naming it, because a
    # missing category is usually a data or seed question worth answering.
    _log_partial_parent_coverage(con)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _erv_parent_rows AS
        SELECT * EXCLUDE (children_present, children_expected),
               children_present, children_expected
        FROM _erv_parent_candidates
        WHERE children_present = children_expected
        """
    )
    n_parent = con.execute("SELECT count(*) FROM _erv_parent_rows").fetchone()[0]
    if not n_parent:
        return
    con.execute(
        f"""
        INSERT INTO eval_results_view BY NAME
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            url_encode_udf(r.composite_slug || '/' || r.parent_benchmark_id)
                                                        AS evaluation_id,
            metric_summary_id_udf(r.parent_benchmark_id, r.metric_id)
                                                        AS metric_summary_id,
            r.composite_slug,
            r.composite_display_name,
            r.parent_benchmark_id                       AS benchmark_id,
            b.family_id, b.family_display_name,
            FALSE                                       AS is_slice,
            CAST(NULL AS VARCHAR)                       AS parent_benchmark_id,
            CAST(NULL AS VARCHAR)                       AS parent_benchmark_display_name,
            r.metric_id,
            r.metric_base_id,
            r.metric_qualifier,
            'derived'                                   AS value_level,
            r.model_key,
            r.model_info.id                             AS model_id,
            url_encode_udf(r.model_key)                 AS model_route_id,
            r.model_info,
            r.metric_display_name,
            r.metric_unit,
            r.lower_is_better,
            resolve_benchmark_tags_udf(b.display_name, b.benchmark_id) AS derived_tags,
            r.min_score, r.max_score,
            r.score_normalized,
            regexp_replace(
                regexp_replace(
                    metric_summary_id_udf(r.parent_benchmark_id, r.metric_id),
                    '_(stderr|std_err|standard_error)$', '', 'i'),
                '_(acc|accuracy|score|value|result)$', '', 'i'
            )                                           AS metric_pair_key,
            r.score,
            CASE WHEN r.scale_conversion = 'mixed'
                 THEN NULL ELSE r.score END             AS score_published,
            CAST({{
                'score':             r.score,
                'standard_error':    NULL,
                'standard_deviation': NULL,
                'sample_size':       NULL,
                'confidence_interval': {{
                    'lower': NULL, 'upper': NULL, 'confidence_level': NULL
                }}
            }} AS STRUCT(
                score DOUBLE, standard_error DOUBLE, standard_deviation DOUBLE,
                sample_size INTEGER,
                confidence_interval STRUCT(
                    lower DOUBLE, upper DOUBLE, confidence_level DOUBLE
                )
            ))                                          AS score_details,
            r.fact_row_count,
            r.evaluation_timestamp,
            r.source_metadata,
            CAST({{
                'dataset_name':    b.display_name,
                'source_type':     b.data_format,
                'hf_repo':         b.dataset_repo,
                'hf_split':        NULL,
                'samples_number':  NULL,
                'url':             b.resources,
                'dataset_url':     NULL,
                'dataset_version': NULL
            }} AS STRUCT(
                dataset_name VARCHAR, source_type VARCHAR, hf_repo VARCHAR,
                hf_split VARCHAR, samples_number INTEGER, url VARCHAR[],
                dataset_url VARCHAR, dataset_version VARCHAR
            ))                                          AS source_data,
            r.evaluator_relationships,
            r.has_first_party,
            r.has_third_party,
            r.is_verified_evaluator,
            r.evaluator_display_name,
            CASE
                WHEN r.has_first_party AND r.has_third_party THEN 'both'
                WHEN r.has_first_party                       THEN 'self'
                ELSE                                              'third'
            END                                         AS coverage_cell,
            r.reporting_orgs,
            FALSE                                       AS is_summary_score,
            CAST(r.aggregate_components AS {aggregate_components_type})
                                                        AS aggregate_components,
            r.metric_id                                 AS metric_id_effective,
            r.value_aggregation,
            r.children_present,
            r.children_expected,
            r.scale_conversion,
            r.score_canonical,
            CAST(NULL AS VARCHAR)                       AS metric_source_label,
            r.judge_condition,
            r._is_headline                              AS is_headline,
            r.collection_id,
            r.protocol_condition,
            r.split
        FROM _erv_parent_rows r
        LEFT JOIN benchmarks b ON b.composite_slug = r.composite_slug
                              AND b.benchmark_id   = r.parent_benchmark_id
        """
    )
    n_weighted = con.execute(
        f"SELECT count(*) FROM _erv_parent_rows "
        f"WHERE value_aggregation = '{AGG_WEIGHTED_MEAN}'"
    ).fetchone()[0]
    log.info(
        "stage J: materialised %d slice-parent row(s) as the MEAN of their "
        "registry children's cell values, complete task sets only "
        "(%d sample-size weighted, %d unweighted)",
        n_parent, n_weighted, n_parent - n_weighted,
    )


def _log_partial_parent_coverage(con, top_n: int = 20) -> None:
    """WARN for every suite cell that was dropped because some of the
    parent's registry children had no value in it.

    A partial mean is a different quantity from the benchmark it is named
    after, so no row is emitted. What a reader of this log decides is whether
    the child really has no data or whether the registry's child set is wrong
    — an over-broad parent makes every cell partial, and a set that omits a
    real task makes the means that DO pass too narrow."""
    rows = con.execute(
        """
        SELECT composite_slug, parent_benchmark_id, metric_id,
               COUNT(*) AS n_cells,
               MIN(children_present) AS min_present,
               MAX(children_present) AS max_present,
               MAX(children_expected) AS expected
        FROM _erv_parent_candidates
        -- Candidates the parent already reports itself were dropped by the
        -- own-row anti-join, not by the coverage gate: the page has its
        -- number and nothing is missing from it.
        WHERE children_present < children_expected
        GROUP BY 1, 2, 3
        ORDER BY n_cells DESC, 1, 2, 3
        """
    ).fetchall()
    if not rows:
        return
    total = sum(r[3] for r in rows)
    log.warning(
        "stage J: %d suite cell(s) across %d (composite, benchmark, metric) "
        "group(s) cover only part of the parent's registry task set — no "
        "derived row emitted (a mean over some of the tasks is not the "
        "benchmark)", total, len(rows),
    )
    for slug, parent, metric, n_cells, lo, hi, expected in rows[:top_n]:
        span = f"{lo}" if lo == hi else f"{lo}-{hi}"
        log.warning(
            "  partial coverage %s/%d: %s / %s / %s — %d model cell(s)",
            span, expected, slug, parent, metric, n_cells,
        )
    if len(rows) > top_n:
        log.warning("  ... and %d more group(s)", len(rows) - top_n)


def _log_cells_without_aggregate(con, top_n: int = 20) -> None:
    """WARN for every cell that holds only PART observations — no row that
    measured the benchmark itself, and no complete registry task set.

    These are not errors in the data: they are benchmarks where the source
    reported per-task numbers only. What the cell then shows is
    `AGGREGATE_LESS_CELL_LEVEL`'s business — no value at all (`none`) or a
    pipeline-derived mean (`derived`) — but either way a reviewer wants them
    named, to decide whether the source really has no aggregate or whether its
    whole row simply failed to resolve as one. Cells that have a whole
    observation never appear here."""
    rows = con.execute(
        f"""
        SELECT composite_slug, benchmark_key, metric_key,
               COUNT(*) AS n_cells, MAX(cell_fact_count) AS max_facts
        FROM _erv_tri
        WHERE _value_level = '{AGGREGATE_LESS_CELL_LEVEL}'
        GROUP BY 1, 2, 3
        ORDER BY n_cells DESC, 1, 2, 3
        """
    ).fetchall()
    if not rows:
        return
    total = sum(r[3] for r in rows)
    log.warning(
        "stage J: %d cell(s) across %d (composite, benchmark, metric) group(s) "
        "hold only PART observations with no whole and no complete registry "
        "task set — value is %s",
        total, len(rows),
        "a pooling of those parts, labelled `pooled_parts`"
        if AGGREGATE_LESS_CELL_LEVEL == "pooled_parts"
        else "left NULL",
    )
    for composite_slug, benchmark_key, metric_key, n_cells, max_facts in rows[:top_n]:
        log.warning(
            "  parts-only cell: %s / %s / %s — %d model cell(s), "
            "up to %d fact(s) each",
            composite_slug, benchmark_key, metric_key, n_cells, max_facts,
        )
    if len(rows) > top_n:
        log.warning("  ... and %d more group(s)", len(rows) - top_n)


def stage_j_models_view(con, snapshot_id: str) -> None:
    """Materialise `models_view` — one row per model.

    Aggregates the model's fact rows (evidence_count, variant_count,
    timestamps, evaluator/source breakdowns) and per-triple data from
    `eval_results_view` (evaluations_count, signal rollups, category
    breakdown, top scores). Joins onto `models` for display fields.

    Depends on `eval_results_view` already being materialised on the
    connection by `stage_j_eval_results_view`.

    `variants[]` is single-self for v1 (one entry per row, the row's own
    model). Family-scoped variant rollup is a follow-up — today's
    registry doesn't carry the variant metadata (qualifier/version_date)
    that would let us populate other family members usefully.
    """
    sid = snapshot_id_to_sql(snapshot_id)

    con.execute(
        f"""
        CREATE TABLE models_view AS
        WITH fact_aggs AS (
            -- Per-fact-row rollups at root grain: evidence_count,
            -- variant_count_setup, generation-config gaps, latest
            -- timestamp, evaluator names. Aggregating by
            -- `model_aggregation_key` collapses variants of the same
            -- identity into one row and matches the grain of
            -- `triple_aggs` (which reads from eval_results_view).
            SELECT
                model_aggregation_key                            AS model_key,
                CAST(COUNT(*) AS BIGINT)                    AS evidence_count,
                CAST(COUNT(DISTINCT variant_key) AS INTEGER) AS variant_count,
                CAST(COUNT(*) FILTER (
                    WHERE NOT (has_temperature AND has_top_p AND has_max_tokens)
                ) AS INTEGER)                                AS missing_generation_config_count,
                -- latest_timestamp = latest of the model's actual
                -- eval-run timestamps (NOT scrape times). NULL when
                -- none of this model's evaluations carry timestamps.
                MAX({ts_cast_sql("evaluation_timestamp")}) AS latest_timestamp,
                arg_max(org_raw,
                        struct_pack(t := {ts_cast_sql("evaluation_timestamp")}, n := org_raw))
                    FILTER (WHERE org_raw IS NOT NULL)        AS latest_source_name,
                -- evaluator_count uses the de-aliased identity (`org_display`)
                -- so models evaluated by both `Ai2` and `Allen Institute for
                -- AI` rows count as one evaluator, and unresolved-org rows
                -- still contribute (counting raw string identity) instead
                -- of being filtered out by an `org_id IS NOT NULL` predicate.
                CAST(COUNT(DISTINCT org_display) FILTER (WHERE org_display IS NOT NULL) AS BIGINT)
                                                              AS evaluator_count,
                ARRAY_AGG(DISTINCT org_display ORDER BY org_display)
                    FILTER (WHERE org_display IS NOT NULL)    AS evaluator_names,
                CAST(COUNT(DISTINCT provenance_source_type)
                     FILTER (WHERE provenance_source_type IS NOT NULL) AS INTEGER)
                                                              AS source_type_count,
                ARRAY_AGG(DISTINCT provenance_source_type ORDER BY provenance_source_type)
                    FILTER (WHERE provenance_source_type IS NOT NULL) AS source_types,
                ARRAY_AGG(DISTINCT model_raw ORDER BY model_raw)
                    FILTER (WHERE model_raw IS NOT NULL)      AS raw_model_ids,
                ARRAY_AGG(DISTINCT struct_pack(
                    "name"    := eval_library_name,
                    "version" := eval_library_version,
                    fork      := CAST(NULL AS VARCHAR)
                ) ORDER BY struct_pack(
                    "name"    := eval_library_name,
                    "version" := eval_library_version,
                    fork      := CAST(NULL AS VARCHAR)
                )) FILTER (
                    WHERE eval_library_name IS NOT NULL
                       OR eval_library_version IS NOT NULL
                )                                             AS eval_libraries
            FROM fact_results
            WHERE model_aggregation_key IS NOT NULL
            GROUP BY 1
        ),
        triple_aggs AS (
            -- Per-triple rollups read from eval_results_view (one row per
            -- triple already). Counts of (benchmark_id, metric_id) cells,
            -- third-party coverage, signal flags, score summary, category
            -- breakdown. LEFT JOINs benchmarks dim only to read is_slice;
            -- the join is 1:1 on (composite_slug, benchmark_id) so the
            -- other counts are unaffected. benchmarks_count excludes slice
            -- rows so the per-model count matches the snapshot-level
            -- benchmark_count denominator (both parents-only).
            SELECT
                erv.model_key,
                CAST(COUNT(*) AS BIGINT)                                  AS evaluations_count,
                CAST(COUNT(DISTINCT erv.benchmark_id)
                    FILTER (WHERE NOT COALESCE(b.is_slice, FALSE)) AS BIGINT)
                                                                          AS benchmarks_count,
                CAST(COUNT(*) FILTER (WHERE coverage_cell IN ('third', 'both')) AS BIGINT)
                                                                          AS third_party_eval_count,
                ROUND(AVG(CASE WHEN has_reproducibility_gap THEN 1.0 ELSE 0.0 END), 12)
                                                                          AS gap_rate,
                CAST(SUM(CASE WHEN has_reproducibility_gap THEN 1 ELSE 0 END) AS INTEGER)
                                                                          AS gap_count,
                ROUND(AVG(completeness_score), 12)                        AS completeness_avg,
                list_sort(list_distinct(flatten(
                    list(from_json(derived_tags, '["VARCHAR"]'))
                    FILTER (WHERE derived_tags IS NOT NULL)
                )))                                                       AS derived_tags_union,
                -- answer-feedback rows are excluded from the score
                -- aggregates only (FILTER, not a pool-level WHERE — counts
                -- and signal rates still see every row).
                --
                -- On `score_canonical`, like every other arithmetic in the
                -- warehouse. A model's summary spans every benchmark it was
                -- run on, so its rows are guaranteed to disagree about scale:
                -- summarising published numbers reported `aristotle/aristotle`
                -- as min = max = avg 86.0 for a metric bounded at 1. The
                -- count goes with them — a row with no canonical value has
                -- no place on the scale these three describe, and counting
                -- it would put a denominator under numbers it never entered.
                CAST(COUNT(score_canonical) FILTER (
                    WHERE {protocol_exclusion_sql("erv.protocol_condition")}
                ) AS INTEGER)                                             AS score_count,
                MIN(score_canonical) FILTER (
                    WHERE {protocol_exclusion_sql("erv.protocol_condition")}
                )                                                         AS score_min,
                MAX(score_canonical) FILTER (
                    WHERE {protocol_exclusion_sql("erv.protocol_condition")}
                )                                                         AS score_max,
                ROUND(AVG(score_canonical) FILTER (
                    WHERE {protocol_exclusion_sql("erv.protocol_condition")}
                ), 12)                                                    AS score_avg,
                CAST(SUM(CASE WHEN is_multi_source THEN 1 ELSE 0 END) AS INTEGER)
                                                                          AS multi_source_groups,
                CAST(SUM(CASE WHEN first_party_only THEN 1 ELSE 0 END) AS INTEGER)
                                                                          AS first_party_only_groups,
                {_source_type_distribution_sql("erv")}
            FROM eval_results_view erv
            LEFT JOIN benchmarks b
              ON b.composite_slug = erv.composite_slug
             AND b.benchmark_id   = erv.benchmark_id
            -- One row per (benchmark, metric) cell: the headline reading.
            -- Extra judge conditions and protocol arms would otherwise
            -- count the same cell several times.
            WHERE erv.is_headline
            GROUP BY 1
        ),
        model_comparability AS (
            -- Divergence is a comparability-GROUP signal at the slice-aware
            -- grain (comparability_group_id includes slice_key), so it cannot
            -- be counted off eval_results_view (which collapses slices). We
            -- source it from fact_results, counting each comparability_group_id
            -- once per model (the flag is constant within a group).
            -- See sensitivity/docs/divergence-count-grain.md.
            SELECT
                model_aggregation_key,
                CAST(COUNT(DISTINCT comparability_group_id)
                     FILTER (WHERE has_variant_divergence) AS INTEGER)
                                                                          AS variant_divergent_count,
                CAST(COUNT(DISTINCT comparability_group_id)
                     FILTER (WHERE has_cross_party_divergence) AS INTEGER)
                                                                          AS cross_party_divergent_count,
                CAST(COUNT(DISTINCT comparability_group_id)
                     FILTER (WHERE has_variant_divergence IS NOT NULL) AS INTEGER)
                                                                          AS groups_with_variant_check,
                CAST(COUNT(DISTINCT comparability_group_id)
                     FILTER (WHERE has_cross_party_divergence IS NOT NULL) AS INTEGER)
                                                                          AS groups_with_cross_party_check
            FROM fact_results
            WHERE comparability_group_id IS NOT NULL
            GROUP BY model_aggregation_key
        ),
        benchmark_names AS (
            -- Excludes slice display names so the array length matches the
            -- per-model `benchmarks_count` (parents-only). Downstream
            -- consumers: developer rollup in headline.json, search filter
            -- on /models, "X benchmarks" tags on model cards.
            SELECT
                erv.model_key,
                ARRAY_AGG(DISTINCT b.display_name ORDER BY b.display_name)
                    FILTER (WHERE b.display_name IS NOT NULL
                            AND NOT COALESCE(b.is_slice, FALSE)) AS benchmark_names
            FROM eval_results_view erv
            LEFT JOIN benchmarks b
              ON b.composite_slug = erv.composite_slug
             AND b.benchmark_id   = erv.benchmark_id
            WHERE erv.is_headline
            GROUP BY 1
        ),
        erv_with_display AS (
            SELECT
                erv.model_key,
                erv.derived_tags,
                erv.benchmark_id                               AS raw_benchmark_id,
                COALESCE(b.display_name, erv.benchmark_id)     AS benchmark_display,
                erv.evaluation_id                              AS benchmark_key,
                -- The canonical number, ranked on and shown: a model's "top
                -- score per tag" pool spans benchmarks and therefore spans
                -- scales, so a published-score rank puts a 100.0 percent
                -- above every fraction on the page and then prints it.
                erv.score_canonical                            AS score,
                erv.metric_display_name,
                erv.lower_is_better
            FROM eval_results_view erv
            LEFT JOIN benchmarks b
              ON b.composite_slug = erv.composite_slug
             AND b.benchmark_id   = erv.benchmark_id
            WHERE erv.score_canonical IS NOT NULL
              AND erv.derived_tags IS NOT NULL
              -- best-style rollup: only the cell's headline reading, and
              -- (redundantly, since answer-feedback rows are never
              -- headline) never an answer-feedback row
              AND erv.is_headline
              AND {protocol_exclusion_sql("erv.protocol_condition")}
        ),
        ranked_for_top AS (
            SELECT
                e.model_key,
                tag.t                                          AS tag,
                e.benchmark_display,
                e.benchmark_key,
                e.score,
                e.metric_display_name,
                ROW_NUMBER() OVER (
                    PARTITION BY e.model_key, tag.t
                    ORDER BY
                        benchmark_priority_udf(e.raw_benchmark_id) DESC,
                        CASE WHEN COALESCE(e.lower_is_better, FALSE)
                             THEN e.score ELSE -e.score
                        END ASC,
                        e.benchmark_key ASC,
                        -- Final total tiebreak: one (model_key, tag) can hold
                        -- several metrics on the same benchmark_key (same
                        -- benchmark, different metric) with equal scores,
                        -- leaving benchmark_key ASC tied. metric_display_name
                        -- pins which metric's row becomes the tag's top score.
                        e.metric_display_name ASC
                ) AS _rk
            FROM erv_with_display e,
                 UNNEST(from_json(e.derived_tags, '["VARCHAR"]')) AS tag(t)
        ),
        top_scores AS (
            SELECT
                model_key,
                ARRAY_AGG(struct_pack(
                    benchmark    := benchmark_display,
                    benchmarkKey := benchmark_key,
                    score        := score,
                    metric       := metric_display_name,
                    tag          := tag
                ) ORDER BY tag) AS top_scores
            FROM ranked_for_top
            WHERE _rk = 1
            GROUP BY 1
        ),
        link_rollups AS (
            SELECT
                model_key,
                ARRAY_AGG(DISTINCT source_metadata.source_organization_url
                          ORDER BY source_metadata.source_organization_url)
                    FILTER (WHERE source_metadata.source_organization_url IS NOT NULL)
                    AS source_urls
            FROM eval_results_view
            GROUP BY 1
        ),
        tag_counts AS (
            SELECT
                erv.model_key,
                tag.t                                          AS tag,
                CAST(COUNT(*) AS INTEGER)                      AS cnt
            FROM eval_results_view erv,
                 UNNEST(from_json(erv.derived_tags, '["VARCHAR"]')) AS tag(t)
            WHERE erv.is_headline
            GROUP BY 1, 2
        ),
        tag_stats_agg AS (
            SELECT
                model_key,
                to_json(MAP(LIST(tag ORDER BY tag), LIST(cnt ORDER BY tag))) AS tag_stats
            FROM tag_counts
            GROUP BY 1
        )
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            m.model_key,
            m.model_id,
            m.model_key                                 AS id,
            url_encode_udf(m.model_key)                 AS route_id,
            url_encode_udf(m.model_key)                 AS model_route_id,
            -- model_group_id is the always-present GROUP key (the
            -- model_aggregation_key). Because `models.model_key` already
            -- collapses to the group root via Stage A's
            -- `_derive_model_root_id`, the group key is simply `model_key`
            -- here — non-null for every model (self at root), no parent-walk
            -- needed. `model_route_id` (above) is url_encode of this key.
            m.model_key                                 AS model_group_id,

            m.display_name                              AS model_name,
            m.display_name                              AS canonical_model_name,
            m.family                                    AS model_family_name,
            COALESCE(m.org_display_name, m.developer)   AS developer,

            m.release_date                              AS release_date,
            CAST(NULL AS VARCHAR)                       AS model_url,
            m.architecture,
            CAST(NULL AS VARCHAR)                       AS params,
            COALESCE(m.params_billions,
                     extract_params_billions_udf(m.display_name))
                                                        AS params_billions,
            m.open_weights                              AS open_weights,
            -- Modalities pulled through from canonical_models. NULL when
            -- registry has no data; frontend treats NULL the same as []
            -- (no modality affordance shown).
            m.input_modalities                          AS input_modalities,
            m.output_modalities                         AS output_modalities,
            -- Model-resolution-rework end-state names. `model_group_id`
            -- (the always-present GROUP key) is emitted ONCE above as
            -- `m.model_key AS model_group_id` — `model_route_id` is its
            -- url_encode. `model_family_id` here is the registry's STRUCTURAL
            -- family-release id (the M3 family walk; nullable, distinct
            -- concept from the group key). `root_model_id` is the legacy
            -- back-compat alias of the group key.
            m.model_group_id                            AS root_model_id,
            m.model_family_id                           AS model_family_id,
            m.lineage_origin_model_id                   AS lineage_origin_model_id,
            m.lineage_origin_model_org_id               AS lineage_origin_model_org_id,
            m.lineage_origin_model_org_id               AS lineage_origin_org_id,
            m.resolution_source                         AS resolution_source,
            m.resolution_granularity                    AS resolution_granularity,
            CAST(NULL AS VARCHAR)                       AS inference_engine,
            -- inference_platform is a PER-RUN fact (a model can be served by
            -- many platforms), so it's populated per-row on
            -- eval_results_view.model_info, not at this model-grain view.
            CAST(NULL AS VARCHAR)                       AS inference_platform,

            COALESCE(ta.evaluations_count, 0)           AS evaluations_count,
            COALESCE(ta.benchmarks_count,  0)           AS benchmarks_count,
            COALESCE(ta.benchmarks_count,  0)           AS benchmark_coverage_count,
            COALESCE(fa.variant_count,     0)           AS variant_count,
            COALESCE(fa.evaluator_count,   0)           AS evaluator_count,
            fa.evaluator_names,
            COALESCE(fa.source_type_count, 0)           AS source_type_count,
            fa.source_types,
            COALESCE(ta.third_party_eval_count, 0)      AS third_party_eval_count,
            CASE
                WHEN COALESCE(ta.evaluations_count, 0) > 0
                THEN CAST(ta.third_party_eval_count AS DOUBLE) / ta.evaluations_count
                ELSE NULL
            END                                          AS independent_verification_ratio,
            COALESCE(fa.evidence_count, 0)               AS evidence_count,
            COALESCE(fa.missing_generation_config_count, 0) AS missing_generation_config_count,
            fa.latest_timestamp,
            fa.latest_source_name,
            bn.benchmark_names,

            ta.derived_tags_union                        AS derived_tags,
            tsa.tag_stats,

            -- reproducibility band rule (legacy: 0/1/0<x<1 → complete/missing/partial)
            CASE
                WHEN ta.gap_rate IS NULL THEN NULL
                WHEN ta.gap_rate = 0     THEN 'complete'
                WHEN ta.gap_rate = 1     THEN 'missing'
                ELSE                          'partial'
            END                                          AS reproducibility_status,
            CAST({{
                'results_total':                CAST(COALESCE(ta.evaluations_count, 0) AS INTEGER),
                'has_reproducibility_gap_count': COALESCE(ta.gap_count, 0),
                'populated_ratio_avg':           ta.completeness_avg
            }} AS {_REPRODUCIBILITY_SUMMARY_STRUCT}) AS reproducibility_summary,

            CAST({{
                'total_results':           CAST(COALESCE(fa.evidence_count, 0) AS INTEGER),
                'total_groups':            CAST(COALESCE(ta.evaluations_count, 0) AS INTEGER),
                'multi_source_groups':     COALESCE(ta.multi_source_groups, 0),
                'first_party_only_groups': COALESCE(ta.first_party_only_groups, 0),
                'source_type_distribution': {{
                    'first_party':   COALESCE(ta.pst_first_party,   0),
                    'third_party':   COALESCE(ta.pst_third_party,   0),
                    'collaborative': COALESCE(ta.pst_collaborative, 0),
                    'unspecified':   COALESCE(ta.pst_unspecified,   0)
                }}
            }} AS {_PROVENANCE_SUMMARY_STRUCT}) AS provenance_summary,

            CAST({{
                'total_groups':                  CAST(COALESCE(ta.evaluations_count, 0) AS INTEGER),
                'groups_with_variant_check':     COALESCE(mc.groups_with_variant_check, 0),
                'groups_with_cross_party_check': COALESCE(mc.groups_with_cross_party_check, 0),
                'variant_divergent_count':       COALESCE(mc.variant_divergent_count, 0),
                'cross_party_divergent_count':   COALESCE(mc.cross_party_divergent_count, 0)
            }} AS {_COMPARABILITY_SUMMARY_STRUCT}) AS comparability_summary,

            fa.eval_libraries,

            CAST({{
                'count':   COALESCE(ta.score_count, 0),
                'min':     ta.score_min,
                'max':     ta.score_max,
                'average': ta.score_avg
            }} AS STRUCT(
                "count" INTEGER, "min" DOUBLE, "max" DOUBLE, average DOUBLE
            )) AS score_summary,

            ts.top_scores,

            lr.source_urls,
            CAST([] AS VARCHAR[])                        AS detail_urls,

            -- variants[]: single self-entry for v1 — see function docstring.
            [CAST({{
                'variant_id':           m.model_key,
                'variant_key':          url_encode_udf(m.model_key),
                'variant_label':        m.display_name,
                'variant_display_name': m.display_name,
                'raw_model_ids':        fa.raw_model_ids,
                'family_id':            m.model_key,
                'family_name':          m.family,
                'version_date':         CAST(NULL AS VARCHAR),
                'version_qualifier':    CAST(NULL AS VARCHAR),
                'total_evaluations':    CAST(COALESCE(ta.evaluations_count, 0) AS INTEGER),
                'last_updated':         fa.latest_timestamp,
                'tags_covered':         ta.derived_tags_union
            }} AS STRUCT(
                variant_id VARCHAR, variant_key VARCHAR,
                variant_label VARCHAR, variant_display_name VARCHAR,
                raw_model_ids VARCHAR[], family_id VARCHAR, family_name VARCHAR,
                version_date VARCHAR, version_qualifier VARCHAR,
                total_evaluations INTEGER, last_updated TIMESTAMP,
                tags_covered VARCHAR[]
            ))]                                          AS variants,

            fa.raw_model_ids
        FROM models m
        LEFT JOIN fact_aggs    fa ON fa.model_key = m.model_key
        LEFT JOIN triple_aggs  ta ON ta.model_key = m.model_key
        LEFT JOIN model_comparability mc ON mc.model_aggregation_key = m.model_key
        LEFT JOIN benchmark_names bn ON bn.model_key = m.model_key
        LEFT JOIN top_scores   ts ON ts.model_key = m.model_key
        LEFT JOIN link_rollups lr ON lr.model_key = m.model_key
        LEFT JOIN tag_stats_agg tsa ON tsa.model_key = m.model_key
        ORDER BY m.model_key
        """
    )


def stage_j_evals_view(con, snapshot_id: str) -> None:
    """Materialise `evals_view` — one row per benchmark.

    Carries the primary metric's config + scalars plus the multi-metric
    pre-pivoted leaderboard (`leaderboard_metrics[]` columns, one
    `leaderboard_rows[]` entry per model with a `values` MAP keyed by
    metric `column_key`). The frontend's eval detail page renders multi-
    metric directly off these arrays — no per-page GROUP BY.

    `primary_metric_id`: the registry's preferred metric when it has a
    headline row on this page, else the metric covering the most distinct
    models, then the most headline rows, then metric_id ASC — the same rule
    the merged page and the hierarchy apply. The benchmark-level scalars
    (`avg_score`, `top_score`, `best_model`) are scoped to that primary
    metric.

    Depends on `eval_results_view` already being materialised on the
    connection.

    `subtasks[]` rolls up per-slice metric aggregations from
    `fact_results` directly (eval_results_view's triple-grouping doesn't
    carry slice_key — see Stage C `_apply_slice_key`). One subtask per
    distinct `(benchmark_id, slice_key)` with non-null slice_key; each
    subtask's `metrics[]` mirrors the root benchmark's `root_metrics[]`
    shape (display, models_count, top_score, etc.) so the frontend's
    subtask breakdown panel renders the same way as the root listing.

    `aggregate_sources[]` (suite rollup) is not yet tracked, and
    `is_aggregated` is always false.
    """
    sid = snapshot_id_to_sql(snapshot_id)

    benchmark_card_struct_type = (
        "STRUCT("
        "benchmark_details STRUCT("
        '  "name" VARCHAR, overview VARCHAR, data_type VARCHAR,'
        "  domains VARCHAR[], languages VARCHAR[],"
        "  similar_benchmarks VARCHAR[], resources VARCHAR[]"
        "),"
        "purpose_and_intended_users STRUCT("
        "  goal VARCHAR, audience VARCHAR[], tasks VARCHAR[],"
        "  limitations VARCHAR, out_of_scope_uses VARCHAR[]"
        "),"
        "data STRUCT(source VARCHAR, size VARCHAR, format VARCHAR, annotation VARCHAR),"
        "methodology STRUCT("
        "  methods VARCHAR[], metrics VARCHAR[], calculation VARCHAR,"
        "  interpretation VARCHAR, baseline_results VARCHAR, validation VARCHAR"
        "),"
        "ethical_and_legal_considerations STRUCT("
        "  privacy_and_anonymity VARCHAR, data_licensing VARCHAR,"
        "  consent_procedures VARCHAR, compliance_with_regulations VARCHAR"
        "),"
        "possible_risks STRUCT(category VARCHAR, description VARCHAR[], url VARCHAR)[],"
        "flagged_fields JSON,"
        "missing_fields VARCHAR[],"
        "card_info STRUCT(created_at VARCHAR, llm VARCHAR)"
        ")"
    )

    leaderboard_metric_struct_type = (
        "STRUCT("
        "column_key VARCHAR, metric_summary_id VARCHAR,"
        "metric_id VARCHAR, metric_name VARCHAR, display_name VARCHAR,"
        "canonical_display_name VARCHAR, lower_is_better BOOLEAN,"
        "unit VARCHAR, scope VARCHAR, subtask_key VARCHAR, subtask_name VARCHAR"
        ")"
    )

    source_data_struct_type = (
        "STRUCT("
        "dataset_name VARCHAR, source_type VARCHAR, hf_repo VARCHAR,"
        "hf_split VARCHAR, samples_number INTEGER, url VARCHAR[],"
        "dataset_url VARCHAR, dataset_version VARCHAR"
        ")"
    )

    con.execute(
        f"""
        CREATE TABLE evals_view AS
        WITH per_metric AS (
            -- One row per (composite_slug, benchmark_id, metric_id).
            -- Carries metric meta + counts + lower_is_better-aware top
            -- score in a single scan of eval_results_view.
            SELECT
                erv.composite_slug,
                erv.benchmark_id,
                erv.metric_id,
                ANY_VALUE(erv.metric_base_id)      AS metric_base_id,
                ANY_VALUE(erv.metric_display_name) AS metric_display_name,
                ANY_VALUE(erv.metric_unit)         AS metric_unit,
                ANY_VALUE(erv.lower_is_better)     AS lower_is_better,
                COUNT(DISTINCT erv.model_key)      AS metric_models_count,
                CAST(COUNT(*) AS BIGINT)           AS metric_rows_count,
                -- top_score is a best-style rollup: answer-feedback rows
                -- are excluded from the score aggregate only.
                --
                -- On score_canonical, like every other comparison on this
                -- view. `score` is what each source published, and one page
                -- can hold fractions beside percentages (a detected div100
                -- row publishes 88.5 where its canonical twin is 0.885, and a
                -- `mixed` cell publishes the canonical number outright), so a
                -- max over it returns whichever row happened to be on the
                -- larger scale.
                CASE WHEN COALESCE(ANY_VALUE(erv.lower_is_better), FALSE)
                     THEN MIN(erv.score_canonical) FILTER (
                         WHERE {protocol_exclusion_sql("erv.protocol_condition")})
                     ELSE MAX(erv.score_canonical) FILTER (
                         WHERE {protocol_exclusion_sql("erv.protocol_condition")})
                END AS top_score
            FROM eval_results_view erv
            -- One row per (benchmark, metric, model) cell: the headline
            -- reading. A metric's coverage is how many models it reads for,
            -- not how many judge channels published it.
            WHERE erv.is_headline
            GROUP BY 1, 2, 3
        ),
        primary_metric AS (
            -- Default metric, the same rule the merged page applies:
            -- the registry's preferred metric when it has at least one
            -- headline row here; else a metric that MEASURES the thing the
            -- benchmark is for, ahead of one that only describes the run;
            -- then widest distinct-model coverage, most headline rows,
            -- metric_id.
            --
            -- The diagnostic step exists because the coverage/alphabetical
            -- tiebreak was headlining cost per task, response length,
            -- degeneration and invalid-rate — every model on the page
            -- publishes those, so they win on coverage, and `average-word-count`
            -- sorts before `length-controlled-win-rate`. The registry marks
            -- such metrics `metadata.role = diagnostic`; they stay on the
            -- page, they just no longer speak for it. A page whose metrics
            -- are ALL diagnostic still headlines one, because the ordering
            -- only demotes.
            SELECT composite_slug, benchmark_id, metric_id, metric_base_id,
                   metric_display_name, metric_unit, lower_is_better, top_score
            FROM (
                SELECT pm.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY pm.composite_slug, pm.benchmark_id
                           ORDER BY
                               COALESCE(pm.metric_base_id = cb.preferred_metric_id,
                                        FALSE) DESC,
                               {_diagnostic_role_sql("cmet.metadata")} ASC,
                               pm.metric_models_count DESC,
                               pm.metric_rows_count DESC,
                               pm.metric_id ASC
                       ) AS _rk
                FROM per_metric pm
                LEFT JOIN canonical_benchmarks cb ON cb.id = pm.benchmark_id
                LEFT JOIN canonical_metrics cmet  ON cmet.id = pm.metric_base_id
            )
            WHERE _rk = 1
        ),
        primary_triples AS (
            -- One row per triple on the primary metric. The
            -- `scoring_score` flips sign for lower-is-better metrics so
            -- arg_max/arg_min pick the right model in primary_facts.
            --
            -- SCORE CONTRACT: every average, extremum, rank and sort on this
            -- view reads `score_canonical`, the one scale all of a page's
            -- rows share. `score` and `score_published` are display and
            -- provenance — they carry the source's own number, which is a
            -- fraction on one row and a percentage on the next, so comparing
            -- or adding them across models is arithmetic on mixed units.
            SELECT
                erv.*,
                CASE WHEN COALESCE(pm.lower_is_better, FALSE)
                     THEN -erv.score_canonical ELSE erv.score_canonical
                END AS scoring_score
            FROM eval_results_view erv
            JOIN primary_metric pm
              ON pm.composite_slug = erv.composite_slug
             AND pm.benchmark_id   = erv.benchmark_id
             AND pm.metric_id      = erv.metric_id
            WHERE erv.is_headline
        ),
        evaluator_names_agg AS (
            -- Distinct org names across primary-metric triples for this
            -- (composite, benchmark). Done in a separate CTE so the
            -- unnest doesn't inflate the per-triple aggregations.
            SELECT pt.composite_slug, pt.benchmark_id,
                   ARRAY_AGG(DISTINCT u ORDER BY u) FILTER (WHERE u IS NOT NULL) AS evaluator_names,
                   -- The subset of evaluators that are validated submitters
                   -- (de-aliased, same name space as evaluator_names) — powers
                   -- the verified badge on the /evals list "Reported by" row.
                   -- DISTINCT collapses the unnest-inflated rows.
                   ARRAY_AGG(DISTINCT pt.evaluator_display_name
                             ORDER BY pt.evaluator_display_name)
                       FILTER (WHERE pt.is_verified_evaluator
                               AND pt.evaluator_display_name IS NOT NULL)
                       AS verified_evaluator_names
            FROM primary_triples pt,
                 UNNEST(COALESCE(pt.reporting_orgs, [])) AS u_t(u)
            GROUP BY 1, 2
        ),
        source_types_agg AS (
            SELECT pt.composite_slug, pt.benchmark_id,
                   ARRAY_AGG(DISTINCT t ORDER BY t) FILTER (WHERE t IS NOT NULL) AS source_types
            FROM primary_triples pt,
                 UNNEST(COALESCE(pt.evaluator_relationships, [])) AS t_t(t)
            GROUP BY 1, 2
        ),
        primary_facts AS (
            -- Per-(composite, benchmark) scalars over the primary
            -- metric's triples. One row per triple — no cross-join
            -- unnest here, so SUMs and COUNTs are accurate.
            SELECT
                pt.composite_slug, pt.benchmark_id,
                CAST(COUNT(DISTINCT pt.model_key) AS BIGINT)           AS models_count,
                arg_max(pt.source_metadata.source_organization_name,
                        struct_pack(t := pt.evaluation_timestamp,
                                    n := pt.source_metadata.source_organization_name))
                                                                       AS latest_source_name,
                ROUND(AVG(CASE WHEN pt.coverage_cell IN ('third', 'both')
                         THEN 1.0 ELSE 0.0 END), 12)                   AS third_party_ratio,
                CAST(SUM(CASE
                    WHEN pt.evalcards_annotations.reproducibility_gap.populated_count
                       < pt.evalcards_annotations.reproducibility_gap.required_count
                    THEN 1 ELSE 0 END) AS INTEGER)                     AS missing_generation_config_count,
                -- exclusion predicate as FILTER on the score
                -- aggregates only — NOT a pool-level WHERE, which would
                -- also change models_count / evaluator_names / gap rates.
                ROUND(AVG(pt.score_canonical) FILTER (
                    WHERE {protocol_exclusion_sql("pt.protocol_condition")}
                ), 12)                                                 AS avg_score,
                -- The denominator behind that average. `models_count` is
                -- every model with a headline cell on the primary metric,
                -- scored or not; AVG silently skips the unscored ones, so a
                -- page could report 317 models over a mean of 216. This says
                -- how many actually contributed.
                CAST(COUNT(DISTINCT pt.model_key) FILTER (
                    WHERE pt.score_canonical IS NOT NULL
                      AND {protocol_exclusion_sql("pt.protocol_condition")}
                ) AS BIGINT)                                           AS scored_models_count,
                -- The list card's normalised figure is the mean of the
                -- per-model normalised scores this page already shows, not a
                -- second normalisation of `avg_score`. Deriving it again from
                -- the average published number skipped the canonical-scale
                -- conversion and the lower-is-better inversion, so the list
                -- card and the result rows disagreed: SQuAD read 22.22 on the
                -- card and 0.33 on the rows, HarmBench 0.26 against 0.74.
                ROUND(AVG(pt.score_normalized) FILTER (
                    WHERE {protocol_exclusion_sql("pt.protocol_condition")}
                ), 12)                                                 AS avg_score_normalized,
                MIN(pt.score_canonical) FILTER (
                    WHERE {protocol_exclusion_sql("pt.protocol_condition")}
                )                                                      AS min_score_seen,
                MAX(pt.score_canonical) FILTER (
                    WHERE {protocol_exclusion_sql("pt.protocol_condition")}
                )                                                      AS max_score_seen,
                -- top/bottom are addressable identifiers — use model_key so
                -- unresolved models can also occupy these slots and the
                -- downstream JOIN to `models` resolves their display name.
                -- The ordering value is a (score, model_key) struct so score
                -- ties break deterministically on model_key (rather than
                -- arg_max/arg_min picking an arbitrary tied row, which
                -- varied run-to-run); the primary key stays scoring_score.
                arg_max(pt.model_key,
                        struct_pack(s := pt.scoring_score, k := pt.model_key))
                    FILTER (WHERE {protocol_exclusion_sql("pt.protocol_condition")})
                                                                       AS top_model_id,
                arg_min(pt.model_key,
                        struct_pack(s := pt.scoring_score, k := pt.model_key))
                    FILTER (WHERE {protocol_exclusion_sql("pt.protocol_condition")})
                                                                       AS bottom_model_id,
                ROUND(AVG(CASE WHEN pt.has_reproducibility_gap THEN 1.0 ELSE 0.0 END), 12)
                                                                       AS gap_rate,
                CAST(SUM(CASE WHEN pt.has_reproducibility_gap THEN 1 ELSE 0 END) AS INTEGER)
                                                                       AS gap_count,
                ROUND(AVG(pt.completeness_score), 12)                  AS completeness_avg,
                CAST(COUNT(*) AS INTEGER)                              AS gprov_total_groups,
                CAST(SUM(CASE WHEN pt.is_multi_source THEN 1 ELSE 0 END) AS INTEGER)
                                                                       AS multi_source_groups,
                CAST(SUM(CASE WHEN pt.first_party_only THEN 1 ELSE 0 END) AS INTEGER)
                                                                       AS first_party_only_groups,
                {_source_type_distribution_sql("pt")}
            FROM primary_triples pt
            GROUP BY pt.composite_slug, pt.benchmark_id
        ),
        primary_comparability AS (
            -- Divergence is a comparability-GROUP signal at the slice-aware
            -- grain (comparability_group_id includes slice_key), so it cannot
            -- be counted off primary_triples (erv.*, which collapses slices).
            -- We source it from fact_results restricted to the primary metric
            -- (JOIN primary_metric on composite_slug, benchmark, metric),
            -- counting each comparability_group_id once per (composite,
            -- benchmark). See sensitivity/docs/divergence-count-grain.md.
            SELECT
                fr.composite_slug,
                fr.benchmark_key                                       AS benchmark_id,
                CAST(COUNT(DISTINCT fr.comparability_group_id)
                     FILTER (WHERE fr.has_variant_divergence) AS INTEGER)
                                                                       AS variant_divergent_count,
                CAST(COUNT(DISTINCT fr.comparability_group_id)
                     FILTER (WHERE fr.has_cross_party_divergence) AS INTEGER)
                                                                       AS cross_party_divergent_count,
                CAST(COUNT(DISTINCT fr.comparability_group_id)
                     FILTER (WHERE fr.has_variant_divergence IS NOT NULL) AS INTEGER)
                                                                       AS groups_with_variant_check,
                CAST(COUNT(DISTINCT fr.comparability_group_id)
                     FILTER (WHERE fr.has_cross_party_divergence IS NOT NULL) AS INTEGER)
                                                                       AS groups_with_cross_party_check
            FROM fact_results fr
            JOIN primary_metric pm
              ON pm.composite_slug = fr.composite_slug
             AND pm.benchmark_id   = fr.benchmark_key
             AND pm.metric_id      = fr.metric_key
            WHERE fr.comparability_group_id IS NOT NULL
            GROUP BY fr.composite_slug, fr.benchmark_key
        ),
        leaderboard_metrics_agg AS (
            SELECT
                pm.composite_slug,
                pm.benchmark_id,
                CAST(COUNT(*) AS INTEGER) AS metrics_count,
                ARRAY_AGG(pm.metric_display_name ORDER BY pm.metric_id)
                    AS metric_names,
                ARRAY_AGG(struct_pack(
                    column_key             := pm.metric_id,
                    metric_summary_id      := metric_summary_id_udf(
                                                  pm.benchmark_id, pm.metric_id),
                    metric_id              := pm.metric_id,
                    metric_name            := pm.metric_display_name,
                    display_name           := pm.metric_display_name,
                    canonical_display_name := pm.metric_display_name,
                    lower_is_better        := pm.lower_is_better,
                    unit                   := pm.metric_unit,
                    scope                  := 'root',
                    subtask_key            := CAST(NULL AS VARCHAR),
                    subtask_name           := CAST(NULL AS VARCHAR)
                ) ORDER BY pm.metric_id) AS leaderboard_metrics,
                ARRAY_AGG(struct_pack(
                    metric_summary_id      := metric_summary_id_udf(
                                                  pm.benchmark_id, pm.metric_id),
                    metric_name            := pm.metric_display_name,
                    display_name           := pm.metric_display_name,
                    canonical_display_name := pm.metric_display_name,
                    metric_key             := pm.metric_id,
                    lower_is_better        := pm.lower_is_better,
                    models_count           := CAST(pm.metric_models_count AS INTEGER),
                    top_score              := pm.top_score,
                    unit                   := pm.metric_unit
                ) ORDER BY pm.metric_id) AS root_metrics
            FROM per_metric pm
            GROUP BY pm.composite_slug, pm.benchmark_id
        ),
        leaderboard_one_per_metric AS (
            -- Collapse condition points to one row per (composite,
            -- benchmark, model, metric) for the pre-pivoted leaderboard:
            -- the values MAP is keyed by metric_id and would raise on
            -- duplicate keys. Headline rows only, like every other
            -- page-level rollup — a cell whose rows are all answer-feedback
            -- arms has no headline and contributes nothing, rather than
            -- publishing a score that is excluded from the rankings,
            -- summaries and comparison index it sits beside.
            SELECT erv.* FROM eval_results_view erv WHERE erv.is_headline
        ),
        leaderboard_per_model AS (
            -- One row per (composite_slug, benchmark_id, model_key)
            -- carrying its values map across all metrics on that
            -- (composite, benchmark) pair.
            SELECT
                erv.composite_slug,
                erv.benchmark_id,
                erv.model_key,
                -- arg_min not ANY_VALUE: a model scored on several metrics
                -- contributes one row per metric to this group, and those
                -- rows can disagree on these columns (per-metric records
                -- carrying different upload timestamps or reporting orgs).
                -- ANY_VALUE returned whichever row the hash aggregate saw
                -- first, so the emitted leaderboard struct flipped
                -- run-to-run. metric_id is unique within the group (the
                -- CTE above keeps one row per metric), so it is a total
                -- order; FILTER keeps ANY_VALUE's "first non-null" reach.
                arg_min(erv.model_route_id, erv.metric_id)
                    FILTER (WHERE erv.model_route_id IS NOT NULL)
                                                               AS model_route_id,
                arg_min(erv.model_info, erv.metric_id)
                    FILTER (WHERE erv.model_info IS NOT NULL)  AS model_info,
                arg_min(erv.evaluation_timestamp, erv.metric_id)
                    FILTER (WHERE erv.evaluation_timestamp IS NOT NULL)
                                                               AS evaluation_timestamp,
                arg_min(erv.source_metadata, erv.metric_id)
                    FILTER (WHERE erv.source_metadata IS NOT NULL)
                                                               AS source_metadata,
                arg_min(erv.source_data, erv.metric_id)
                    FILTER (WHERE erv.source_data IS NOT NULL) AS source_data,
                MAP(
                    ARRAY_AGG(erv.metric_id ORDER BY erv.metric_id),
                    ARRAY_AGG(erv.score     ORDER BY erv.metric_id)
                )                                              AS values_map,
                CAST(COUNT(erv.score) AS INTEGER)              AS metrics_present
            FROM leaderboard_one_per_metric erv
            GROUP BY 1, 2, 3
        ),
        leaderboard_rows_agg AS (
            SELECT
                composite_slug,
                benchmark_id,
                ARRAY_AGG(struct_pack(
                    model_info           := model_info,
                    model_route_id       := model_route_id,
                    evaluation_timestamp := evaluation_timestamp,
                    source_metadata      := source_metadata,
                    source_data          := source_data,
                    "values"             := values_map,
                    metrics_present      := metrics_present
                ) ORDER BY model_key) AS leaderboard_rows
            FROM leaderboard_per_model
            GROUP BY 1, 2
        ),
        instance_summary AS (
            SELECT
                erv.composite_slug,
                erv.benchmark_id,
                CAST(COUNT(DISTINCT erv.instance_file_path)
                     FILTER (WHERE erv.instance_file_path IS NOT NULL) AS BIGINT)
                    AS url_count,
                -- `instance_data.sample_urls` is consumed as a link list,
                -- so aggregate the fetchable URL rather than the
                -- repo-relative path it is built from.
                ARRAY_AGG(DISTINCT erv.instance_file_url
                          ORDER BY erv.instance_file_url)
                    FILTER (WHERE erv.instance_file_url IS NOT NULL)
                    AS sample_urls_full,
                CAST(COUNT(DISTINCT erv.model_key)
                     FILTER (WHERE erv.instance_file_path IS NOT NULL) AS INTEGER)
                    AS models_with_loaded_instances
            FROM eval_results_view erv
            GROUP BY 1, 2
        ),
        per_slice_metric AS (
            -- One row per (composite_slug, benchmark_key, slice_key,
            -- metric_key). Reads from fact_results because
            -- eval_results_view collapses to one row per (composite,
            -- model, benchmark, metric) and doesn't carry slice_key.
            -- Keys (canonical-or-raw) are used so unresolved benchmarks
            -- and metrics still surface their slices, mirroring the
            -- root listing's `root_metrics` field-for-field.
            -- `metric_models_count` is at root grain so it matches
            -- root_metrics, which sources from eval_results_view (also
            -- root grain).
            SELECT
                fr.composite_slug,
                fr.benchmark_key                   AS benchmark_id,
                fr.slice_key,
                fr.metric_key                      AS metric_id,
                MIN(fr.slice_name)                 AS slice_name_rep,
                -- MAX not ANY_VALUE: rows within a slice-metric group can
                -- disagree on these (mixed source metadata) and ANY_VALUE
                -- made the emitted structs run-to-run unstable.
                MAX(cmet.display_name)             AS metric_display_name,
                MAX(fr.metric_unit)                AS metric_unit,
                MAX(fr.lower_is_better)            AS lower_is_better,
                CAST(COUNT(DISTINCT fr.model_aggregation_key) AS INTEGER)
                                                   AS metric_models_count,
                -- The subtask's best reading, on `score_canonical`. This is a
                -- MIN/MAX across models, which is arithmetic over a set of
                -- rows that need not share a published scale, and the
                -- carve-out for `curated` only covered the one class the
                -- registry had restated: LiveBench's Zebra Puzzle reported a
                -- top of 100.0 under a metric the same row declares to be a
                -- proportion. The canonical scale is the only one every row
                -- here shares, so the extreme is taken on it.
                CASE WHEN COALESCE(MAX(fr.lower_is_better), FALSE)
                     THEN MIN(fr.score_canonical)
                     ELSE MAX(fr.score_canonical)
                END AS top_score
            FROM fact_results fr
            LEFT JOIN canonical_metrics cmet ON cmet.id = fr.metric_base_key
            -- Headline fact rows only: the condition grain sits above the
            -- slice grain, so this drops the losing judge/protocol
            -- conditions without dropping any slice.
            WHERE fr.fact_id IN (SELECT fact_id FROM fact_headline
                                 WHERE is_headline)
              AND fr.composite_slug         IS NOT NULL
              AND fr.benchmark_key          IS NOT NULL
              AND fr.slice_key              IS NOT NULL
              AND fr.metric_key             IS NOT NULL
              AND fr.model_aggregation_key  IS NOT NULL
            GROUP BY 1, 2, 3, 4
        ),
        slice_metrics_agg AS (
            -- One row per (composite_slug, benchmark_id, slice_key) —
            -- metrics rolled into a struct array. Deterministic
            -- ordering by metric_id.
            SELECT
                composite_slug,
                benchmark_id,
                slice_key,
                MIN(slice_name_rep) AS slice_name_rep,
                ARRAY_AGG(struct_pack(
                    metric_summary_id      := metric_summary_id_udf(
                                                  benchmark_id, metric_id),
                    metric_name            := metric_display_name,
                    display_name           := metric_display_name,
                    canonical_display_name := metric_display_name,
                    metric_key             := metric_id,
                    lower_is_better        := lower_is_better,
                    models_count           := metric_models_count,
                    top_score              := top_score,
                    unit                   := metric_unit
                ) ORDER BY metric_id) AS metrics
            FROM per_slice_metric
            GROUP BY composite_slug, benchmark_id, slice_key
        ),
        subtasks_agg AS (
            -- One row per (composite_slug, benchmark_id) — slices rolled
            -- into a struct array.
            SELECT
                composite_slug,
                benchmark_id,
                ARRAY_AGG(struct_pack(
                    subtask_key            := slice_key,
                    subtask_name           := slice_name_rep,
                    display_name           := slice_name_rep,
                    canonical_display_name := slice_name_rep,
                    metrics                := metrics
                ) ORDER BY slice_key) AS subtasks,
                CAST(COUNT(*) AS INTEGER) AS subtasks_count
            FROM slice_metrics_agg
            GROUP BY composite_slug, benchmark_id
        )
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            url_encode_udf(b.composite_slug || '/' || b.benchmark_id) AS evaluation_id,
            b.composite_slug,
            b.composite_display_name,
            b.benchmark_id,
            b.family_id,
            b.family_display_name,
            b.is_slice,
            -- parent_benchmark_id mirrors the contract on
            -- the comparison-index sidecar:
            -- null for roots, the parent benchmark id for slices.
            -- The dim sometimes stores parent_benchmark_id == benchmark_id
            -- for roots, so gate on is_slice rather than trusting the raw
            -- column value.
            CASE WHEN b.is_slice THEN b.parent_benchmark_id ELSE NULL END
                                                        AS parent_benchmark_id,
            -- The parent benchmark's own display name (NOT the composite
            -- label) — slice-fold titles read this so cross-benchmark
            -- suites don't title groups with the suite name. Same
            -- fallback-to-id rule as the dim's display_name.
            CASE WHEN b.is_slice
                 THEN COALESCE(pb.display_name, b.parent_benchmark_id)
                 ELSE NULL END                          AS parent_benchmark_display_name,
            pm.metric_id                                AS primary_metric_id,

            b.display_name                              AS evaluation_name,
            b.display_name                              AS canonical_display_name,
            resolve_benchmark_tags_udf(b.display_name, b.benchmark_id) AS derived_tags,
            lookup_known_issues_udf(b.benchmark_id, b.display_name)  AS known_issues,

            CAST(struct_pack(
                evaluation_description := pm.metric_display_name,
                lower_is_better        := pm.lower_is_better,
                score_type             := CAST(NULL AS VARCHAR),
                min_score              := cmet.min_score,
                max_score              := cmet.max_score,
                unit                   := pm.metric_unit
            ) AS STRUCT(
                evaluation_description VARCHAR, lower_is_better BOOLEAN,
                score_type VARCHAR, min_score DOUBLE, max_score DOUBLE,
                unit VARCHAR
            )) AS metric_config,

            COALESCE(pf.models_count, 0)                AS models_count,
            -- How many of those models the page average actually covers.
            COALESCE(pf.scored_models_count, 0)         AS scored_models_count,
            ena.evaluator_names,
            ena.verified_evaluator_names,
            sta.source_types,
            pf.latest_source_name,
            pf.third_party_ratio,
            pf.missing_generation_config_count,
            CAST(struct_pack(
                "name" := COALESCE(top_m.display_name, pf.top_model_id),
                score  := pm.top_score
            ) AS STRUCT("name" VARCHAR, score DOUBLE)) AS best_model,
            CAST(struct_pack(
                "name" := COALESCE(bot_m.display_name, pf.bottom_model_id),
                score  := CASE WHEN COALESCE(pm.lower_is_better, FALSE)
                               THEN pf.max_score_seen ELSE pf.min_score_seen END
            ) AS STRUCT("name" VARCHAR, score DOUBLE)) AS worst_model,
            pf.avg_score,
            pf.avg_score_normalized                      AS avg_score_norm,
            pm.top_score                                 AS top_score,

            COALESCE(b.card_present, FALSE)              AS has_card,
            -- All VARCHAR[] fields normalise NULL → [] at the boundary so the
            -- consumer-facing TS contract (`string[]`, non-nullable) holds.
            -- Upstream JSON extraction returns NULL when the source omits a
            -- field; without these COALESCE shims a missing `methodology.metrics`
            -- crashes `methodology.metrics.length` in the frontend.
            CAST(struct_pack(
                benchmark_details := struct_pack(
                    "name"    := b.card_name,
                    overview  := b.overview,
                    data_type := b.data_type,
                    domains            := COALESCE(b.domains,            CAST([] AS VARCHAR[])),
                    languages          := COALESCE(b.languages,          CAST([] AS VARCHAR[])),
                    similar_benchmarks := COALESCE(b.similar_benchmarks, CAST([] AS VARCHAR[])),
                    resources          := COALESCE(b.resources,          CAST([] AS VARCHAR[]))
                ),
                purpose_and_intended_users := struct_pack(
                    goal              := b.goal,
                    audience          := COALESCE(b.audience,          CAST([] AS VARCHAR[])),
                    tasks             := COALESCE(b.tasks,             CAST([] AS VARCHAR[])),
                    limitations       := b.limitations,
                    out_of_scope_uses := COALESCE(b.out_of_scope_uses, CAST([] AS VARCHAR[]))
                ),
                data := struct_pack(
                    source     := b.data_source,
                    size       := b.data_size,
                    format     := b.data_format,
                    annotation := b.data_annotation
                ),
                methodology := struct_pack(
                    methods          := COALESCE(b.methods,      CAST([] AS VARCHAR[])),
                    metrics          := COALESCE(b.card_metrics, CAST([] AS VARCHAR[])),
                    calculation      := b.calculation,
                    interpretation   := b.interpretation,
                    baseline_results := b.baseline_results,
                    validation       := b.validation
                ),
                ethical_and_legal_considerations := struct_pack(
                    privacy_and_anonymity        := b.privacy_and_anonymity,
                    data_licensing               := b.data_licensing,
                    consent_procedures           := b.consent_procedures,
                    compliance_with_regulations  := b.compliance_with_regulations
                ),
                possible_risks := COALESCE(
                    b.possible_risks,
                    CAST([] AS STRUCT(category VARCHAR, description VARCHAR[], url VARCHAR)[])
                ),
                flagged_fields := b.flagged_fields,
                missing_fields := CAST([] AS VARCHAR[]),
                card_info := struct_pack(
                    created_at := CAST(NULL AS VARCHAR),
                    llm        := b.card_generated_by
                )
            ) AS {benchmark_card_struct_type})            AS benchmark_card,

            FALSE                                         AS is_aggregated,
            CAST(NULL AS STRUCT(
                evaluation_id VARCHAR,
                composite_slug VARCHAR,
                composite_display_name VARCHAR,
                models_count INTEGER,
                avg_score_norm DOUBLE
            )[])                                          AS aggregate_sources,
            is_summary_score_udf(
                pm.metric_id, b.parent_benchmark_id, b.benchmark_id
            )                                             AS is_summary_score,

            CAST(struct_pack(
                domains   := b.domains,
                languages := b.languages,
                tasks     := b.tasks
            ) AS STRUCT(
                domains VARCHAR[], languages VARCHAR[], tasks VARCHAR[]
            )) AS tags,
            CAST(struct_pack(
                dataset_name    := b.display_name,
                source_type     := b.data_format,
                hf_repo         := b.dataset_repo,
                hf_split        := CAST(NULL AS VARCHAR),
                samples_number  := CAST(NULL AS INTEGER),
                url             := b.resources,
                dataset_url     := CAST(NULL AS VARCHAR),
                dataset_version := CAST(NULL AS VARCHAR)
            ) AS {source_data_struct_type})              AS source_data,

            CAST(struct_pack(
                results_total                := COALESCE(pf.gprov_total_groups, 0),
                has_reproducibility_gap_count := COALESCE(pf.gap_count, 0),
                populated_ratio_avg          := pf.completeness_avg
            ) AS {_REPRODUCIBILITY_SUMMARY_STRUCT}) AS reproducibility_summary,

            CAST(struct_pack(
                total_results            := COALESCE(pf.gprov_total_groups, 0),
                total_groups             := COALESCE(pf.gprov_total_groups, 0),
                multi_source_groups      := COALESCE(pf.multi_source_groups, 0),
                first_party_only_groups  := COALESCE(pf.first_party_only_groups, 0),
                source_type_distribution := struct_pack(
                    first_party   := COALESCE(pf.pst_first_party, 0),
                    third_party   := COALESCE(pf.pst_third_party, 0),
                    collaborative := COALESCE(pf.pst_collaborative, 0),
                    unspecified   := COALESCE(pf.pst_unspecified, 0)
                )
            ) AS {_PROVENANCE_SUMMARY_STRUCT}) AS provenance_summary,

            CAST(struct_pack(
                total_groups                  := COALESCE(pf.gprov_total_groups, 0),
                groups_with_variant_check     := COALESCE(pcmp.groups_with_variant_check, 0),
                groups_with_cross_party_check := COALESCE(pcmp.groups_with_cross_party_check, 0),
                variant_divergent_count       := COALESCE(pcmp.variant_divergent_count, 0),
                cross_party_divergent_count   := COALESCE(pcmp.cross_party_divergent_count, 0)
            ) AS {_COMPARABILITY_SUMMARY_STRUCT}) AS comparability_summary,

            CAST(struct_pack(
                available                    := COALESCE(ins.url_count, 0) > 0,
                url_count                    := COALESCE(ins.url_count, 0),
                sample_urls                  := COALESCE(ins.sample_urls_full[1:5],
                                                          CAST([] AS VARCHAR[])),
                models_with_loaded_instances := COALESCE(ins.models_with_loaded_instances, 0)
            ) AS STRUCT(
                available BOOLEAN, url_count BIGINT,
                sample_urls VARCHAR[], models_with_loaded_instances INTEGER
            )) AS instance_data,

            COALESCE(lma.metrics_count, 0)               AS metrics_count,
            lma.metric_names,
            CAST(COALESCE(
                lma.leaderboard_metrics,
                CAST([] AS {leaderboard_metric_struct_type}[])
            ) AS {leaderboard_metric_struct_type}[]) AS leaderboard_metrics,
            lra.leaderboard_rows,

            lma.root_metrics,

            CAST(COALESCE(
                sub.subtasks,
                CAST([] AS STRUCT(
                    subtask_key VARCHAR, subtask_name VARCHAR, display_name VARCHAR,
                    canonical_display_name VARCHAR,
                    metrics STRUCT(
                        metric_summary_id VARCHAR, metric_name VARCHAR,
                        display_name VARCHAR, canonical_display_name VARCHAR,
                        metric_key VARCHAR, lower_is_better BOOLEAN,
                        models_count INTEGER, top_score DOUBLE, unit VARCHAR
                    )[]
                )[])
            ) AS STRUCT(
                subtask_key VARCHAR, subtask_name VARCHAR, display_name VARCHAR,
                canonical_display_name VARCHAR,
                metrics STRUCT(
                    metric_summary_id VARCHAR, metric_name VARCHAR,
                    display_name VARCHAR, canonical_display_name VARCHAR,
                    metric_key VARCHAR, lower_is_better BOOLEAN,
                    models_count INTEGER, top_score DOUBLE, unit VARCHAR
                )[]
            )[])                                         AS subtasks,
            COALESCE(sub.subtasks_count, 0)              AS subtasks_count
        FROM benchmarks b
        -- Self-join the dim on the row's parent so slice rows can surface
        -- the parent's actual display name (the dim already carries the
        -- parent row per composite — phantom roots included).
        LEFT JOIN benchmarks pb         ON pb.composite_slug = b.composite_slug
                                        AND pb.benchmark_id  = b.parent_benchmark_id
        LEFT JOIN primary_metric pm     ON pm.composite_slug = b.composite_slug
                                        AND pm.benchmark_id  = b.benchmark_id
        LEFT JOIN canonical_metrics cmet ON cmet.id = pm.metric_base_id
        LEFT JOIN primary_facts pf      ON pf.composite_slug = b.composite_slug
                                        AND pf.benchmark_id  = b.benchmark_id
        LEFT JOIN primary_comparability pcmp ON pcmp.composite_slug = b.composite_slug
                                             AND pcmp.benchmark_id  = b.benchmark_id
        LEFT JOIN evaluator_names_agg ena ON ena.composite_slug = b.composite_slug
                                          AND ena.benchmark_id  = b.benchmark_id
        LEFT JOIN source_types_agg    sta ON sta.composite_slug = b.composite_slug
                                          AND sta.benchmark_id  = b.benchmark_id
        LEFT JOIN models top_m          ON top_m.model_key = pf.top_model_id
        LEFT JOIN models bot_m          ON bot_m.model_key = pf.bottom_model_id
        LEFT JOIN leaderboard_metrics_agg lma ON lma.composite_slug = b.composite_slug
                                              AND lma.benchmark_id  = b.benchmark_id
        LEFT JOIN leaderboard_rows_agg    lra ON lra.composite_slug = b.composite_slug
                                              AND lra.benchmark_id  = b.benchmark_id
        LEFT JOIN instance_summary        ins ON ins.composite_slug = b.composite_slug
                                              AND ins.benchmark_id  = b.benchmark_id
        LEFT JOIN subtasks_agg            sub ON sub.composite_slug = b.composite_slug
                                              AND sub.benchmark_id  = b.benchmark_id
        -- Drop fact-less parent shells. The benchmarks dim deliberately
        -- includes parent benchmarks that have no own fact rows (so the
        -- hierarchy graph is complete — see _synthesise_singleton_families
        -- and the parent-only DISTINCT branch). But evals_view is the
        -- user-facing eval list: a row that no model has reported on
        -- isn't an eval. Aligning with comparison-index, which is built
        -- from per-(eval, metric) buckets and therefore already excludes
        -- these shells.
        WHERE (EXISTS (
            -- benchmarks dim's `benchmark_id` is the canonical-or-raw
            -- key, so match it against fr.benchmark_key (not the
            -- canonical-only fr.benchmark_id) — otherwise raw-only
            -- benchmark rows in the dim never find their fact rows
            -- and get dropped from evals_view.
            SELECT 1 FROM fact_results fr
            WHERE fr.composite_slug = b.composite_slug
              AND fr.benchmark_key  = b.benchmark_id
        ) OR EXISTS (
            -- A suite whose task variants are registry slice children has
            -- no facts of its own but does have a value, rolled up from
            -- those children in `eval_results_view`. Retaining it on that
            -- basis is what keeps BFCL-v3 and RealGuardrails in the eval
            -- list next to their categories, instead of showing 14 category
            -- pages and no suite.
            SELECT 1 FROM eval_results_view erv
            WHERE erv.composite_slug = b.composite_slug
              AND erv.benchmark_id   = b.benchmark_id
        ))
        -- Drop leaderboard rollup metrics that EEE ships as if they
        -- were benchmark names. HELM family ("Mean win rate", "Mean
        -- score"), BFCL ("overall"), facts-grounding ("score"), etc.
        -- These are composite-level aggregate scores; surfacing them
        -- as standalone evals is misleading. The hierarchy build
        -- surfaces them as composite rollup metrics in hierarchy.json.
        -- Match case-insensitively because EEE's
        -- casing is inconsistent ("Mean win rate" vs "Mean score").
        AND LOWER(b.benchmark_id) NOT IN (
            'mean win rate', 'mean score', 'overall', 'overall score',
            'score', 'aggregate', 'aggregate score', 'mean', 'total',
            'total score', 'all', 'rank', 'elo', 'average'
        )
        ORDER BY b.composite_slug, b.benchmark_id
        """
    )


def stage_j_merged_evals_view(con, snapshot_id: str) -> None:
    """One summary row per resolved canonical benchmark for the merged
    all-sources detail page (merged-benchmark-view spec P5).

    Separate artifact by design — NOT rows in `evals_view` (the frontend
    eval list, evaluator grouping, and `benchmark_index` writer have no
    `is_aggregated` filters, and a dormant legacy branch would route
    aggregated rows to the composite-card page). The reserved
    `is_aggregated`/`aggregate_sources` columns in `evals_view` stay
    untouched.

    Grain rules (spec design pt 7): a benchmark with >=1 top-level
    observation gets a benchmark-grain row; a canonical benchmark whose
    only observations are slice-level gets a slice-grain row (slice
    selector; same query at slice grain). Sources reporting only
    slice-level data for a benchmark-grain page are disclosed in
    `aggregate_sources` with `slice_only = TRUE`, never silently omitted.

    `evaluation_id = url_encode(benchmark_id)` is single-segment and
    cannot collide with per-source ids (those all contain %2F; the
    registry seed guard keeps '/' out of benchmark ids).

    The default metric is the registry `preferred_metric_id` when at
    least one source reports it (post-fold), else the fallback: widest
    distinct-model coverage, ties by observation count then lexicographic.
    Coverage leads because a metric five sources publish for two models
    describes the page worse than one three sources publish for three.
    `best_result` follows Q10: directional on the effective metric,
    flagged rows excluded; when the default metric has no registry
    bounds (generic `score` pages) the unconverted pool is used as a
    fallback rather than surfacing no best result at all.

    No result rows are duplicated — the merged leaderboard remains a
    query over `eval_results_view` by `benchmark_id` + `metric_id_effective`.
    """
    sid = snapshot_id_to_sql(snapshot_id)
    _ensure_merged_view_inputs(con)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE merged_evals_view AS
        WITH tl AS (
            -- Each protocol point is its own observation row: rows
            -- arrive pre-split from eval_results_view's protocol grain and
            -- carry protocol_condition through for the best_result pool.
            SELECT r.benchmark_id, r.evaluation_id, r.composite_slug,
                   r.composite_display_name, r.family_id, r.family_display_name,
                   r.metric_id_effective, r.metric_base_id,
                   r.model_key, r.model_info,
                   r.score, r.score_canonical, r.scale_conversion,
                   r.protocol_condition,
                   CAST(NULL AS VARCHAR) AS slice_id,
                   CAST(NULL AS VARCHAR) AS slice_display_name
            FROM eval_results_view r
            JOIN canonical_benchmarks cb ON cb.id = r.benchmark_id
            -- Headline rows only: one observation per (source, model) cell.
            WHERE NOT r.is_slice AND r.score IS NOT NULL AND r.is_headline
        ),
        sl AS (
            SELECT r.parent_benchmark_id AS benchmark_id,
                   CAST(NULL AS VARCHAR) AS evaluation_id,
                   r.composite_slug, r.composite_display_name,
                   r.family_id, r.family_display_name,
                   r.metric_id_effective, r.metric_base_id,
                   r.model_key, r.model_info,
                   r.score, r.score_canonical, r.scale_conversion,
                   r.protocol_condition,
                   r.benchmark_id AS slice_id,
                   COALESCE(cbs.display_name, r.benchmark_id)
                       AS slice_display_name
            FROM eval_results_view r
            JOIN canonical_benchmarks cb ON cb.id = r.parent_benchmark_id
            LEFT JOIN canonical_benchmarks cbs ON cbs.id = r.benchmark_id
            WHERE r.is_slice AND r.score IS NOT NULL AND r.is_headline
        ),
        universe AS (
            SELECT benchmark_id, 'benchmark' AS grain
            FROM tl GROUP BY benchmark_id
            UNION ALL
            SELECT benchmark_id, 'slice' AS grain
            FROM sl
            WHERE benchmark_id NOT IN (SELECT DISTINCT benchmark_id FROM tl)
            GROUP BY benchmark_id
        ),
        page_rows AS (
            SELECT u.grain, t.* FROM universe u JOIN tl t USING (benchmark_id)
            WHERE u.grain = 'benchmark'
            UNION ALL
            SELECT u.grain, s.* FROM universe u JOIN sl s USING (benchmark_id)
            WHERE u.grain = 'slice'
        ),
        metric_stats AS (
            SELECT benchmark_id, metric_id_effective,
                   MAX(metric_base_id)             AS metric_base_id,
                   COUNT(*)                        AS results_count,
                   COUNT(DISTINCT model_key)       AS models_count,
                   -- the denominator behind any average over this metric:
                   -- models whose cell actually carries a value
                   COUNT(DISTINCT model_key) FILTER (
                       WHERE score_canonical IS NOT NULL) AS scored_models_count,
                   COUNT(DISTINCT composite_slug)  AS sources_count
            FROM page_rows
            GROUP BY 1, 2
        ),
        chosen AS (
            -- Same ordering as the per-source page: registry preference
            -- first, then a metric that measures the task ahead of one that
            -- only describes the run (registry `metadata.role = diagnostic`),
            -- then coverage, row count, id.
            SELECT ms.*,
                   (ms.metric_base_id = cb.preferred_metric_id) AS is_registry_preferred,
                   ROW_NUMBER() OVER (
                       PARTITION BY ms.benchmark_id
                       ORDER BY
                           COALESCE(ms.metric_base_id = cb.preferred_metric_id, FALSE) DESC,
                           {_diagnostic_role_sql("cmet.metadata")} ASC,
                           ms.models_count DESC,
                           ms.results_count DESC,
                           ms.metric_id_effective ASC
                   ) AS rk
            FROM metric_stats ms
            LEFT JOIN canonical_benchmarks cb ON cb.id = ms.benchmark_id
            LEFT JOIN canonical_metrics cmet  ON cmet.id = ms.metric_base_id
        ),
        default_metric AS (
            SELECT benchmark_id,
                   metric_id_effective AS default_metric_id,
                   -- the registry id behind it, for the metric-meta joins
                   metric_base_id      AS default_metric_base_id,
                   COALESCE(is_registry_preferred, FALSE) AS preferred_from_registry,
                   results_count, models_count, scored_models_count, sources_count
            FROM chosen WHERE rk = 1
        ),
        best AS (
            SELECT p.benchmark_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.benchmark_id
                       ORDER BY
                           -- converted pool first; unconverted no_bounds
                           -- pool as fallback; flagged rows are excluded
                           CASE WHEN p.scale_conversion = 'no_bounds' THEN 1 ELSE 0 END ASC,
                           CASE WHEN COALESCE(cm.lower_is_better, FALSE)
                                THEN p.score_canonical
                                ELSE -p.score_canonical
                           END ASC,
                           -- full tiebreak: same (score, model) can appear
                           -- under two composites (exgentic pairs) — without
                           -- the composite term the winning source flips
                           -- with input order between builds
                           p.model_key ASC,
                           p.composite_slug ASC,
                           COALESCE(p.protocol_condition, '') ASC
                   ) AS rk,
                   p.model_info."name" AS model_name,
                   p.model_key, p.score, p.score_canonical,
                   p.composite_slug, p.evaluation_id
            FROM page_rows p
            JOIN default_metric dm
              ON dm.benchmark_id = p.benchmark_id
             AND p.metric_id_effective = dm.default_metric_id
            LEFT JOIN canonical_metrics cm ON cm.id = dm.default_metric_base_id
            WHERE p.scale_conversion != 'flagged'
              -- slice-grain pages get NO best_result: a best across
              -- different slices compares incomparables (same rule as the
              -- comparison-index merged entries)
              AND p.grain = 'benchmark'
              -- answer-feedback rows never win best_result
              AND {protocol_exclusion_sql("p.protocol_condition")}
        ),
        src_page AS (
            -- MAX not ANY_VALUE: upstream carries per-row variance in
            -- display names (multi-record composites), and ANY_VALUE
            -- would make this view build-nondeterministic on top of it.
            SELECT p.benchmark_id, p.composite_slug,
                   MAX(p.composite_display_name)       AS composite_display_name,
                   MAX(p.evaluation_id)                AS evaluation_id,
                   COUNT(*)                            AS results_count,
                   COUNT(DISTINCT p.model_key)         AS models_count,
                   BOOL_OR(p.metric_id_effective = dm.default_metric_id)
                                                       AS reports_preferred,
                   FALSE                               AS slice_only
            FROM page_rows p
            JOIN default_metric dm ON dm.benchmark_id = p.benchmark_id
            GROUP BY 1, 2
        ),
        src_slice_only AS (
            -- 7(a) disclosure: sources with ONLY slice-level data for a
            -- benchmark-grain page.
            SELECT s.benchmark_id, s.composite_slug,
                   MAX(s.composite_display_name)       AS composite_display_name,
                   CAST(NULL AS VARCHAR)               AS evaluation_id,
                   COUNT(*)                            AS results_count,
                   COUNT(DISTINCT s.model_key)         AS models_count,
                   FALSE                               AS reports_preferred,
                   TRUE                                AS slice_only
            FROM sl s
            JOIN universe u ON u.benchmark_id = s.benchmark_id AND u.grain = 'benchmark'
            WHERE (s.benchmark_id, s.composite_slug) NOT IN (
                SELECT benchmark_id, composite_slug FROM tl
            )
            GROUP BY 1, 2
        ),
        sources AS (
            SELECT benchmark_id,
                   LIST({{
                       'evaluation_id': evaluation_id,
                       'composite_slug': composite_slug,
                       'composite_display_name': composite_display_name,
                       'models_count': CAST(models_count AS INTEGER),
                       'results_count': CAST(results_count AS INTEGER),
                       'reports_preferred': reports_preferred,
                       'slice_only': slice_only
                   }} ORDER BY slice_only ASC, results_count DESC, composite_slug ASC)
                       AS aggregate_sources,
                   COUNT(*) AS all_sources_count
            FROM (SELECT * FROM src_page UNION ALL SELECT * FROM src_slice_only)
            GROUP BY benchmark_id
        ),
        metrics_list AS (
            SELECT ms.benchmark_id,
                   LIST({{
                       'metric_id': ms.metric_id_effective,
                       'display_name': COALESCE(cm.display_name, ms.metric_id_effective),
                       'results_count': CAST(ms.results_count AS INTEGER),
                       'models_count': CAST(ms.models_count AS INTEGER),
                       'sources_count': CAST(ms.sources_count AS INTEGER),
                       'lower_is_better': cm.lower_is_better
                   }} ORDER BY ms.results_count DESC, ms.metric_id_effective ASC)
                       AS metrics
            FROM metric_stats ms
            LEFT JOIN canonical_metrics cm ON cm.id = ms.metric_base_id
            GROUP BY ms.benchmark_id
        ),
        slices_list AS (
            SELECT benchmark_id,
                   LIST({{
                       'slice_id': slice_id,
                       'display_name': slice_display_name
                   }} ORDER BY slice_id) AS slices
            FROM (
                SELECT DISTINCT s.benchmark_id, s.slice_id, s.slice_display_name
                FROM sl s
                JOIN universe u ON u.benchmark_id = s.benchmark_id
                              AND u.grain = 'slice'
            )
            GROUP BY benchmark_id
        )
        SELECT
            TIMESTAMP '{sid}'                       AS snapshot_id,
            url_encode_udf(u.benchmark_id)          AS evaluation_id,
            u.benchmark_id,
            COALESCE(cb.display_name, u.benchmark_id) AS display_name,
            fam.family_id,
            fam.family_display_name,
            u.grain,
            dm.default_metric_id                    AS preferred_metric_id,
            COALESCE(cm.display_name, dm.default_metric_id)
                                                    AS preferred_metric_display_name,
            dm.preferred_from_registry,
            COALESCE(cm.lower_is_better, FALSE)     AS lower_is_better,
            CAST(dm.sources_count  AS INTEGER)      AS sources_count,
            CAST(so.all_sources_count AS INTEGER)   AS all_sources_count,
            CAST(dm.results_count  AS INTEGER)      AS results_count,
            CAST(dm.models_count   AS INTEGER)      AS models_count,
            CAST(dm.scored_models_count AS INTEGER) AS scored_models_count,
            CAST({{
                'model_name':     b.model_name,
                'model_key':      b.model_key,
                'score':          b.score,
                'score_canonical': b.score_canonical,
                'composite_slug': b.composite_slug,
                'evaluation_id':  b.evaluation_id
            }} AS STRUCT(
                model_name VARCHAR, model_key VARCHAR, score DOUBLE,
                score_canonical DOUBLE, composite_slug VARCHAR,
                evaluation_id VARCHAR
            ))                                      AS best_result,
            so.aggregate_sources,
            ml.metrics,
            sll.slices
        FROM universe u
        LEFT JOIN canonical_benchmarks cb ON cb.id = u.benchmark_id
        LEFT JOIN default_metric dm       ON dm.benchmark_id = u.benchmark_id
        LEFT JOIN canonical_metrics cm    ON cm.id = dm.default_metric_base_id
        LEFT JOIN best b                  ON b.benchmark_id = u.benchmark_id AND b.rk = 1
        LEFT JOIN sources so              ON so.benchmark_id = u.benchmark_id
        LEFT JOIN metrics_list ml         ON ml.benchmark_id = u.benchmark_id
        LEFT JOIN slices_list sll         ON sll.benchmark_id = u.benchmark_id
        LEFT JOIN (
            SELECT benchmark_id,
                   MAX(family_id) AS family_id,
                   MAX(family_display_name) AS family_display_name
            FROM page_rows GROUP BY benchmark_id
        ) fam ON fam.benchmark_id = u.benchmark_id
        ORDER BY u.benchmark_id
        """
    )
    n = con.execute("SELECT count(*) FROM merged_evals_view").fetchone()[0]
    n_slice = con.execute(
        "SELECT count(*) FROM merged_evals_view WHERE grain = 'slice'"
    ).fetchone()[0]
    log.info(
        "stage J: merged_evals_view — %d merged benchmark row(s), %d slice-grain",
        n, n_slice,
    )


def stage_j_emit_view_parquets(con, out_dir: Path, snapshot_id: str) -> None:
    """Emit the view-layer parquets to the warehouse snapshot dir.

    Companion to `stage_i_emit_warehouse_parquets`. Stage J creates the
    view tables on the connection (via the per-view materialiser
    functions); this function writes them to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for table, sort_key in [
        # protocol_condition, judge_condition and split complete the sort:
        # one view row per (protocol point, judge condition, split) means
        # (composite, metric, model) alone is no longer total.
        ("eval_results_view",
         "(composite_slug, metric_summary_id, model_key, protocol_condition, "
         "judge_condition, split)"),
        ("models_view",       "(model_key)"),
        ("evals_view",        "(evaluation_id)"),
        ("merged_evals_view", "(evaluation_id)"),
    ]:
        path = out_dir / f"{table}.parquet"
        con.execute(
            f"""
            COPY (SELECT {explicit_projection_sql(con, table)} FROM {table}
                  ORDER BY {sort_key} NULLS LAST)
            TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    # Re-emit fact_results with the headline flag attached. Stage I writes the
    # facts before the mapping exists, and a consumer reading the facts
    # directly (the frontend's build-time matrix) has no way to re-derive the
    # pick. `fact_headline` stays the cached source of truth; this is a
    # denormalised copy of it on the grain the consumer already reads.
    path = out_dir / "fact_results.parquet"
    con.execute(
        f"""
        COPY (
            SELECT {explicit_projection_sql(con, "fact_results", "f")},
                   CAST(COALESCE(h.is_headline, FALSE) AS BOOLEAN) AS is_headline
            FROM fact_results f
            LEFT JOIN fact_headline h ON h.fact_id = f.fact_id
            ORDER BY {_qualify_sort_key(FACT_RESULTS_SORT_KEY, "f")} NULLS LAST
        )
        TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

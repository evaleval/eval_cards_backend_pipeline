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
import tempfile
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa

from eval_card_backend.signals.reproducibility import (
    AGENTIC_REPRODUCIBILITY_FIELDS,
    BASE_REPRODUCIBILITY_FIELDS,
)
from eval_card_backend.config import EEE_DATASET_REPO
from eval_card_backend.sources.registry import read_parquet_arg


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
        """
        CREATE TABLE results_exploded AS
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
        FROM eee_raw e,
             range(1, len(e.evaluation_results) + 1) AS t(idx_1based)
        WHERE e.evaluation_results IS NOT NULL
          AND len(e.evaluation_results) > 0
        """
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
                clean_eval_name_udf(evaluation_name)                              AS _benchmark_raw,
                extract_metric_udf(
                    COALESCE(metric_config.evaluation_description,
                             metric_config.metric_name,
                             evaluation_name))                                    AS _metric_raw,
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
            resolve_canonical_id(_benchmark_raw, 'benchmark', source_config) AS benchmark_id,
            resolve_canonical_id(_metric_raw,    'metric',    source_config) AS metric_id,
            resolve_canonical_id(_org_raw,       'org',       source_config) AS org_id,
            resolve_canonical_id(NULLIF(_harness_raw, ''), 'harness', source_config) AS harness_id,

            resolve_strategy(_model_raw,     'model',     source_config) AS model_resolution_strategy,
            resolve_strategy(_benchmark_raw, 'benchmark', source_config) AS benchmark_resolution_strategy,
            resolve_strategy(_metric_raw,    'metric',    source_config) AS metric_resolution_strategy,
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


def _apply_metric_folds(con) -> None:
    """Apply the registry's curated per-benchmark metric naming folds.

    `metric_id_effective` = fold target when (benchmark_id, metric_id)
    matches a `benchmark_metric_folds` row, else `metric_id` unchanged.
    Raw `metric_id` is never overwritten — the merged view groups on the
    effective id; per-source views keep reporting the source's own label.
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
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE metric_id_effective IS DISTINCT FROM metric_id"
    ).fetchone()[0]
    log.info("stage C: metric folds re-keyed %d row(s)", n)


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


def stage_d_join_dims_and_flatten(con) -> None:
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

    con.execute(
        """
        CREATE TABLE fact_results_staging AS
        WITH joined AS (
            -- LEFT JOIN dims, then call the metric-meta hotfix UDF once per row
            -- so its STRUCT result can be destructured cleanly in the outer SELECT
            -- (single UDF invocation per row, not five).
            --
            -- composite_slug joins on the source_config curated map (per
            -- evalcard-registry/seed/composites.yaml). Default fallback for
            -- non-curated configs is kebab-case(source_config); display
            -- name falls back to the leaderboard's source_name on EEE
            -- source_metadata (the human-facing label upstream actually
            -- emits) so a brand-new uncurated config still renders.
            SELECT
                rr.*,
                cb.parent_benchmark_id                                 AS _cb_parent_benchmark_id,
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
                COALESCE(
                    ccm.composite_slug,
                    trim(both '-' from regexp_replace(
                        regexp_replace(lower(rr.source_config), '_', '-', 'g'),
                        '[^a-z0-9-]+', '-', 'g'
                    ))
                )                                                      AS _composite_slug,
                COALESCE(
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
                derive_metric_meta_udf(
                    to_json(rr.metric_config),
                    cmet.metric_kind, cmet.metric_unit,
                    cmet.min_score,   cmet.max_score, cmet.lower_is_better,
                    rr.metric_config.metric_name,
                    cmet.score_type
                )                                                      AS _meta,
                co_org.display_name                                    AS org_display_canonical
            FROM results_resolved rr
            LEFT JOIN canonical_benchmarks cb       ON cb.id = rr.benchmark_id
            LEFT JOIN canonical_models     cm_model ON cm_model.id = rr.model_id
            LEFT JOIN canonical_metrics    cmet     ON cmet.id = rr.metric_id
            LEFT JOIN canonical_orgs       co_org   ON co_org.id = rr.org_id
            LEFT JOIN cards_raw            c        ON c.benchmark_id = rr.benchmark_id
            LEFT JOIN composite_config_map ccm      ON ccm.source_config = rr.source_config
        )
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
            COALESCE(j.metric_id, j.metric_raw)                                          AS metric_key,
            -- Fold-aware twin of metric_key: registry naming folds applied
            -- (Stage C metric_id_effective); the merged view groups on this.
            COALESCE(j.metric_id_effective, j.metric_raw)                                AS metric_key_effective,

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

            j._benchmark_card_id                                                        AS benchmark_card_id,

            j.model_resolution_strategy, j.benchmark_resolution_strategy,
            j.metric_resolution_strategy, j.org_resolution_strategy,
            j.harness_resolution_strategy,

            -- score (typed STRUCT access; uncertainty paths are NULL-safe in
            -- DuckDB when the parent struct is NULL).
            j.score_details.score                                                       AS score,
            j.score_details.uncertainty.standard_error.value                            AS score_se,
            j.score_details.uncertainty.confidence_interval.lower                       AS score_ci_lower,
            j.score_details.uncertainty.confidence_interval.upper                       AS score_ci_upper,
            j.score_details.uncertainty.confidence_interval.confidence_level            AS score_ci_level,
            CAST(j.score_details.uncertainty.num_samples AS INTEGER)                    AS n_samples,

            -- source / provenance
            -- TEMPORARY upstream-data-quality fix (ported from the legacy
            -- pipeline's PARTY_OVERRIDE_LLM_STATS_FIX): EEE's llm-stats config
            -- carries `evaluator_relationship` from the aggregator's
            -- perspective, but the underlying rows are model-maker self-reports
            -- (raw_verified='false') vs aggregator-verified rescores
            -- (raw_verified='true'). Reclassify on the row, not in the
            -- frontend, so every consumer agrees. Remove once upstream EEE
            -- emits the right value directly.
            CASE
                WHEN j.source_config = 'llm-stats'
                     AND j.metric_config.additional_details['raw_verified'] = 'false'
                THEN 'first_party'
                ELSE 'third_party'
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

            -- upstream EEE record pointer (repo-relative path of the source
            -- JSON this row was exploded from; Stage J builds the HF URL).
            j.source_record_path,

            -- instance pointer
            j.detailed_evaluation_results.file_path                                      AS instance_file_path,
            j.detailed_evaluation_results.format                                         AS instance_file_format,
            j.detailed_evaluation_results.checksum                                       AS instance_checksum,
            j.detailed_evaluation_results.hash_algorithm                                 AS instance_hash_algorithm,
            CAST(j.detailed_evaluation_results.total_rows AS INTEGER)                    AS instance_rows,

            j._card_payload AS card_payload
        FROM joined j
        LEFT JOIN is_verified_evaluator ev ON ev.evaluation_id = j.evaluation_id
        """
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


def stage_e_per_row_signals(con) -> StageEStats:
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
    """
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


# ---------------------------------------------------------------------------
# Stage F — group signals (pass 2)
# ---------------------------------------------------------------------------


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

    # F.2 — comparability, per (model, benchmark, slice, metric).
    #
    # Per-group metric_config used by the divergence threshold MUST be
    # deterministic across re-runs and consistent across all rows in the
    # group. MAX FILTER picks the same value every time (vs `any_value`
    # which is order-dependent). When the registry is sparse, the hotfix
    # may produce different metric_unit values across rows in the same
    # canonical metric — `n_metric_unit_distinct` surfaces those groups
    # so the operator can target a registry-alias backfill at the right
    # canonical metric.
    #
    # `slice_key` IS NOT DISTINCT FROM in the JOIN treats NULL slice_key
    # (single-raw benchmarks) as equal — a plain `=` would drop those
    # rows since SQL NULL = NULL is unknown. GROUP BY collapses NULL
    # slice_keys to one group automatically, so the JOIN must mirror that.
    con.execute(
        """
        CREATE TABLE fact_results_grouped_annotated AS
        WITH group_payloads AS (
            SELECT
                model_aggregation_key, benchmark_key, slice_key, metric_key,
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
                    score                    := score,
                    generation_args          := generation_args_json,
                    evaluator_relationship   := evaluator_relationship,
                    source_organization_name := org_raw
                ) ORDER BY fact_id) AS group_rows,
                struct_pack(
                    metric_kind := MAX(metric_kind) FILTER (WHERE metric_kind IS NOT NULL),
                    metric_unit := MAX(metric_unit) FILTER (WHERE metric_unit IS NOT NULL),
                    min_score   := MAX(min_score)   FILTER (WHERE min_score   IS NOT NULL),
                    max_score   := MAX(max_score)   FILTER (WHERE max_score   IS NOT NULL)
                ) AS metric_config
            FROM fact_results_grouped
            GROUP BY 1, 2, 3, 4
        ),
        group_annotations AS (
            SELECT
                model_aggregation_key, benchmark_key, slice_key, metric_key,
                compute_variant_divergence_udf(group_rows, metric_config)      AS variant,
                compute_cross_party_divergence_udf(group_rows, metric_config)  AS cross_party
            FROM group_payloads
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
                || md5(fr.metric_key))
              AS comparability_group_id,
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
    con.execute(
        f"""
        CREATE TABLE fact_results AS
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
        """
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

    # Operator-visible counter: which root-collapsed (model, benchmark,
    # metric) groups had rows reporting more than one distinct
    # metric_unit. Each such group's variant_threshold_basis was
    # computed against the deterministic-but-not-row-matching unit
    # picked by the F.2 MAX FILTER aggregation.
    n_unit_inconsistent = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT model_aggregation_key, benchmark_key, metric_key
            FROM fact_results_grouped
            GROUP BY 1, 2, 3
            HAVING COUNT(DISTINCT metric_unit)
                   FILTER (WHERE metric_unit IS NOT NULL) > 1
        )
        """
    ).fetchone()[0]
    if n_unit_inconsistent:
        log.warning(
            "Stage F: %d comparability group(s) had >1 distinct metric_unit "
            "across rows; the per-group threshold basis label may not match "
            "every row's own unit. Backfill the registry's metric_unit for "
            "the offending canonical metric to silence.",
            n_unit_inconsistent,
        )
    return n_unit_inconsistent


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
        WITH composite_configs AS (
            SELECT
                fr.composite_slug,
                ANY_VALUE(fr.composite_display_name) AS composite_display_name,
                ARRAY_AGG(DISTINCT fr.source_config ORDER BY fr.source_config)
                    FILTER (WHERE fr.source_config IS NOT NULL) AS source_configs,
                COUNT(DISTINCT (fr.benchmark_key, fr.metric_key))
                    FILTER (WHERE fr.benchmark_key IS NOT NULL
                            AND fr.metric_key      IS NOT NULL) AS evals_count
            FROM fact_results fr
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


def stage_i_emit_warehouse_parquets(con, out_dir: Path, snapshot_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = snapshot_id_to_sql(snapshot_id)
    for table, sort_key in [
        ("fact_results", "(composite_slug, model_key, benchmark_id, metric_id)"),
        ("benchmarks", "(composite_slug, benchmark_id)"),
        ("composites", "(composite_slug)"),
        ("families", "(family_id)"),
        ("models", "(model_key)"),
    ]:
        path = out_dir / f"{table}.parquet"
        con.execute(
            f"""
            COPY (SELECT * FROM {table} ORDER BY {sort_key} NULLS LAST)
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
    """Materialise `eval_results_view` — one row per (benchmark, metric, model)
    triple. Foundation view: models_view + evals_view fan out from this.

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
    partition, rows are ranked honouring `lower_is_better`. NULL-score
    rows survive in the view (for coverage purposes) but are excluded
    from `position` / `total`. `percentile` = `1 - (position-1) / (total-1)`.
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
        "variant_divergence STRUCT("
        "  magnitude DOUBLE,"
        "  threshold DOUBLE,"
        "  basis VARCHAR,"
        '  differing_fields STRUCT(field VARCHAR, "values" JSON)[]'
        "),"
        "cross_party_divergence STRUCT("
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

    con.execute(
        f"""
        CREATE TABLE eval_results_view AS
        WITH benchmark_tags AS (
            SELECT
                composite_slug, benchmark_id,
                resolve_benchmark_tags_udf(display_name, benchmark_id) AS derived_tags
            FROM benchmarks
        ),
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
            WHERE model_aggregation_key IS NOT NULL
              AND benchmark_key         IS NOT NULL
              AND metric_key            IS NOT NULL
              AND composite_slug        IS NOT NULL
        ),
        tri_agg AS (
            SELECT
                composite_slug, model_aggregation_key, benchmark_key, metric_key,
                CAST(COUNT(*) AS INTEGER) AS fact_row_count,
                -- Median rule: prefer first-party scores; fall back to all rows.
                COALESCE(
                    MEDIAN(score) FILTER (
                        WHERE evaluator_relationship = 'first_party'
                          AND score IS NOT NULL
                    ),
                    MEDIAN(score) FILTER (WHERE score IS NOT NULL)
                ) AS rep_score,
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
                BOOL_OR(has_variant_divergence)          AS has_variant_divergence,
                BOOL_OR(has_cross_party_divergence)      AS has_cross_party_divergence,
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
            FROM tris
            GROUP BY composite_slug, model_aggregation_key, benchmark_key, metric_key
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
                    PARTITION BY composite_slug, model_aggregation_key, benchmark_key, metric_key
                    ORDER BY
                        CASE WHEN score IS NULL THEN 1 ELSE 0 END ASC,
                        CASE WHEN evaluator_relationship = 'first_party' THEN 0 ELSE 1 END ASC,
                        evaluation_id ASC,
                        fact_id ASC
                ) AS _rep_rank
            FROM tris
        ),
        tri_rep AS (
            SELECT * FROM tri_rep_ranked WHERE _rep_rank = 1
        ),
        joined AS (
            SELECT
                ta.*,
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
                tr.score_ci_lower             AS rep_ci_lower,
                tr.score_ci_upper             AS rep_ci_upper,
                tr.score_ci_level             AS rep_ci_level,
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
                tr.generation_additional_details AS rep_generation_additional_details,
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
                cmet.display_name             AS metric_display_name,
                cmet.min_score                AS cmet_min_score,
                cmet.max_score                AS cmet_max_score,
                -- Fold-aware effective metric (same derivation as Stage C's
                -- metric_id_effective — the fold map is deterministic on
                -- (benchmark_key, metric_key)) + its registry bounds for
                -- canonical-scale conversion. Unmasked on purpose: NULL
                -- bounds must stay NULL ('no_bounds'), not become [0,1].
                COALESCE(bmf.to_metric_id, ta.metric_key) AS metric_key_effective,
                bmf.scale_factor              AS eff_scale_factor,
                cmet_eff.lower_is_better      AS eff_lower_is_better,
                cmet_eff.min_score            AS eff_min_score,
                cmet_eff.max_score            AS eff_max_score,
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
            FROM tri_agg ta
            JOIN tri_rep tr USING (composite_slug, model_aggregation_key, benchmark_key, metric_key)
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
            LEFT JOIN canonical_metrics cmet ON cmet.id       = ta.metric_key
            LEFT JOIN benchmark_metric_folds bmf
                   ON bmf.benchmark_id   = ta.benchmark_key
                  AND bmf.from_metric_id = ta.metric_key
            LEFT JOIN canonical_metrics cmet_eff
                   ON cmet_eff.id = COALESCE(bmf.to_metric_id, ta.metric_key)
        ),
        ranked AS (
            -- Rank by score within (composite_slug, benchmark_key, metric_key),
            -- honouring lower_is_better. NULL scores sort last and get
            -- position=NULL. COUNT(rep_score) over the partition counts
            -- non-NULL scores.
            SELECT *,
                CASE
                    WHEN rep_score IS NULL THEN NULL
                    ELSE CAST(ROW_NUMBER() OVER (
                        PARTITION BY composite_slug, benchmark_key, metric_key
                        ORDER BY
                            CASE WHEN rep_score IS NULL THEN 1 ELSE 0 END ASC,
                            CASE WHEN COALESCE(rep_lower_is_better, FALSE)
                                 THEN rep_score
                                 ELSE -rep_score
                            END ASC,
                            model_aggregation_key ASC
                    ) AS INTEGER)
                END AS position,
                CAST(COUNT(rep_score) OVER (
                    PARTITION BY composite_slug, benchmark_key, metric_key
                ) AS INTEGER) AS total,
                -- Scale-suspect detection is per (source, benchmark,
                -- effective-metric) GROUP; the group max is what tells a
                -- percent-scaled publication apart from genuine fractions.
                MAX(rep_score) OVER (
                    PARTITION BY composite_slug, benchmark_key, metric_key_effective
                ) AS eff_grp_max
            FROM joined
        ),
        conv AS (
            -- Canonical-scale conversion (merged-view spec P3, design pt 5).
            -- Group-suspect, then per-row-only-where-unambiguous: mixed
            -- groups (Vals.ai AIME: 99.583 percents next to genuine 0.833
            -- fractions) convert row-by-row; the 1–1.5 band under [0,1]
            -- bounds is ambiguous and flagged, never guessed.
            SELECT *,
                CASE
                    WHEN rep_score IS NULL THEN NULL
                    -- curated published-scale factor on the fold: a known
                    -- fact, never detected — overrides detection entirely
                    WHEN eff_scale_factor IS NOT NULL THEN
                        CASE
                            WHEN eff_min_score IS NULL OR eff_max_score IS NULL
                                THEN 'curated'
                            WHEN rep_score * eff_scale_factor
                                 BETWEEN eff_min_score AND eff_max_score
                                THEN 'curated'
                            ELSE 'flagged'
                        END
                    WHEN eff_min_score IS NULL OR eff_max_score IS NULL
                        THEN 'no_bounds'
                    -- fraction-bounded metric, percent-looking group
                    WHEN eff_max_score <= 1.5 AND eff_grp_max > 1.5 THEN
                        CASE
                            WHEN rep_score > 1.5
                             AND rep_score / 100.0
                                 BETWEEN eff_min_score AND eff_max_score
                                THEN 'div100'
                            WHEN rep_score
                                 BETWEEN eff_min_score AND eff_max_score
                                THEN 'none'
                            ELSE 'flagged'
                        END
                    -- percent-bounded metric, whole group reported as fractions
                    WHEN eff_min_score = 0 AND eff_max_score = 100
                     AND eff_grp_max <= 1.0 THEN
                        CASE
                            WHEN rep_score BETWEEN 0 AND 1.0 THEN 'mul100'
                            ELSE 'flagged'
                        END
                    -- percent-bounded group topping out in (1, 1.5]:
                    -- ambiguous fractions-vs-tiny-percents — never guess
                    WHEN eff_min_score = 0 AND eff_max_score = 100
                     AND eff_grp_max <= 1.5 THEN 'flagged'
                    WHEN rep_score BETWEEN eff_min_score AND eff_max_score
                        THEN 'none'
                    ELSE 'flagged'
                END AS scale_conversion
            FROM ranked
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
            CASE
                WHEN (COALESCE(cmet_max_score, 1) - COALESCE(cmet_min_score, 0)) <= 0 THEN 0
                WHEN COALESCE(rep_lower_is_better, FALSE)
                    THEN GREATEST(0, LEAST(1,
                        1.0 - (rep_score - COALESCE(cmet_min_score, 0))
                              / (COALESCE(cmet_max_score, 1) - COALESCE(cmet_min_score, 0))))
                ELSE GREATEST(0, LEAST(1,
                    (rep_score - COALESCE(cmet_min_score, 0))
                    / (COALESCE(cmet_max_score, 1) - COALESCE(cmet_min_score, 0))))
            END                                                   AS score_normalized,
            regexp_replace(
                regexp_replace(metric_summary_id_udf(benchmark_key, metric_key),
                    '_(stderr|std_err|standard_error)$', '', 'i'),
                '_(acc|accuracy|score|value|result)$', '', 'i'
            )                                                     AS metric_pair_key,

            rep_score                                             AS score,
            CAST({{
                'score':             rep_score,
                'standard_error':    rep_score_se,
                'sample_size':       rep_n_samples,
                'confidence_interval': {{
                    'lower':             rep_ci_lower,
                    'upper':             rep_ci_upper,
                    'confidence_level':  rep_ci_level
                }}
            }} AS STRUCT(
                score DOUBLE, standard_error DOUBLE, sample_size INTEGER,
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
            CAST(NULL AS {aggregate_components_type}) AS aggregate_components,

            triple_has_repro_gap        AS has_reproducibility_gap,
            triple_avg_completeness     AS completeness_score,
            is_multi_source,
            first_party_only,
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
                'variant_divergence': {{
                    'magnitude':        variant_divergence_magnitude,
                    'threshold':        variant_divergence_threshold,
                    'basis':            variant_threshold_basis,
                    'differing_fields': variant_differing_fields
                }},
                'cross_party_divergence': {{
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

            -- Merged-view columns (spec P2/P3). `score_canonical` is on the
            -- effective metric's registry scale; raw `score` is never
            -- overwritten. Flagged rows get NULL (never guessed);
            -- no_bounds rows pass through unconverted.
            metric_key_effective     AS metric_id_effective,
            scale_conversion,
            CASE scale_conversion
                WHEN 'div100'    THEN rep_score / 100.0
                WHEN 'mul100'    THEN rep_score * 100.0
                WHEN 'curated'   THEN rep_score * eff_scale_factor
                WHEN 'none'      THEN rep_score
                WHEN 'no_bounds' THEN rep_score
                ELSE NULL
            END                      AS score_canonical
        FROM conv
        ORDER BY metric_summary_id, model_key
        """
    )


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
                CAST(COUNT(score) AS INTEGER)                             AS score_count,
                MIN(score)                                                AS score_min,
                MAX(score)                                                AS score_max,
                ROUND(AVG(score), 12)                                     AS score_avg,
                CAST(SUM(CASE WHEN is_multi_source THEN 1 ELSE 0 END) AS INTEGER)
                                                                          AS multi_source_groups,
                CAST(SUM(CASE WHEN first_party_only THEN 1 ELSE 0 END) AS INTEGER)
                                                                          AS first_party_only_groups,
                {_source_type_distribution_sql("erv")}
            FROM eval_results_view erv
            LEFT JOIN benchmarks b
              ON b.composite_slug = erv.composite_slug
             AND b.benchmark_id   = erv.benchmark_id
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
            GROUP BY 1
        ),
        erv_with_display AS (
            SELECT
                erv.model_key,
                erv.derived_tags,
                erv.benchmark_id                               AS raw_benchmark_id,
                COALESCE(b.display_name, erv.benchmark_id)     AS benchmark_display,
                erv.evaluation_id                              AS benchmark_key,
                erv.score,
                erv.metric_display_name,
                erv.lower_is_better
            FROM eval_results_view erv
            LEFT JOIN benchmarks b
              ON b.composite_slug = erv.composite_slug
             AND b.benchmark_id   = erv.benchmark_id
            WHERE erv.score IS NOT NULL
              AND erv.derived_tags IS NOT NULL
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

    `primary_metric_id` heuristic: metric with the most distinct models;
    tie-break on metric_id ASC. The benchmark-level scalars (`avg_score`,
    `top_score`, `best_model`) are scoped to that primary metric.

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
                ANY_VALUE(erv.metric_display_name) AS metric_display_name,
                ANY_VALUE(erv.metric_unit)         AS metric_unit,
                ANY_VALUE(erv.lower_is_better)     AS lower_is_better,
                COUNT(DISTINCT erv.model_key)      AS metric_models_count,
                CASE WHEN COALESCE(ANY_VALUE(erv.lower_is_better), FALSE)
                     THEN MIN(erv.score) ELSE MAX(erv.score) END AS top_score
            FROM eval_results_view erv
            GROUP BY 1, 2, 3
        ),
        primary_metric AS (
            -- Pick one metric per (composite, benchmark): most-covered
            -- (tie-break on metric_id).
            SELECT composite_slug, benchmark_id, metric_id, metric_display_name,
                   metric_unit, lower_is_better, top_score
            FROM (
                SELECT pm.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY composite_slug, benchmark_id
                           ORDER BY metric_models_count DESC, metric_id ASC
                       ) AS _rk
                FROM per_metric pm
            )
            WHERE _rk = 1
        ),
        primary_triples AS (
            -- One row per triple on the primary metric. The
            -- `scoring_score` flips sign for lower-is-better metrics so
            -- arg_max/arg_min pick the right model in primary_facts.
            SELECT
                erv.*,
                CASE WHEN COALESCE(pm.lower_is_better, FALSE)
                     THEN -erv.score ELSE erv.score
                END AS scoring_score
            FROM eval_results_view erv
            JOIN primary_metric pm
              ON pm.composite_slug = erv.composite_slug
             AND pm.benchmark_id   = erv.benchmark_id
             AND pm.metric_id      = erv.metric_id
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
                ROUND(AVG(pt.score), 12)                               AS avg_score,
                MIN(pt.score)                                          AS min_score_seen,
                MAX(pt.score)                                          AS max_score_seen,
                -- top/bottom are addressable identifiers — use model_key so
                -- unresolved models can also occupy these slots and the
                -- downstream JOIN to `models` resolves their display name.
                -- The ordering value is a (score, model_key) struct so score
                -- ties break deterministically on model_key (rather than
                -- arg_max/arg_min picking an arbitrary tied row, which
                -- varied run-to-run); the primary key stays scoring_score.
                arg_max(pt.model_key,
                        struct_pack(s := pt.scoring_score, k := pt.model_key)) AS top_model_id,
                arg_min(pt.model_key,
                        struct_pack(s := pt.scoring_score, k := pt.model_key)) AS bottom_model_id,
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
        leaderboard_per_model AS (
            -- One row per (composite_slug, benchmark_id, model_key)
            -- carrying its values map across all metrics on that
            -- (composite, benchmark) pair.
            SELECT
                erv.composite_slug,
                erv.benchmark_id,
                erv.model_key,
                ANY_VALUE(erv.model_route_id)                  AS model_route_id,
                ANY_VALUE(erv.model_info)                      AS model_info,
                ANY_VALUE(erv.evaluation_timestamp)            AS evaluation_timestamp,
                ANY_VALUE(erv.source_metadata)                 AS source_metadata,
                ANY_VALUE(erv.source_data)                     AS source_data,
                MAP(
                    ARRAY_AGG(erv.metric_id ORDER BY erv.metric_id),
                    ARRAY_AGG(erv.score     ORDER BY erv.metric_id)
                )                                              AS values_map,
                CAST(COUNT(erv.score) AS INTEGER)              AS metrics_present
            FROM eval_results_view erv
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
                ARRAY_AGG(DISTINCT erv.instance_file_path
                          ORDER BY erv.instance_file_path)
                    FILTER (WHERE erv.instance_file_path IS NOT NULL)
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
                ANY_VALUE(cmet.display_name)       AS metric_display_name,
                ANY_VALUE(fr.metric_unit)          AS metric_unit,
                ANY_VALUE(fr.lower_is_better)      AS lower_is_better,
                CAST(COUNT(DISTINCT fr.model_aggregation_key) AS INTEGER)
                                                   AS metric_models_count,
                CASE WHEN COALESCE(ANY_VALUE(fr.lower_is_better), FALSE)
                     THEN MIN(fr.score) ELSE MAX(fr.score) END AS top_score
            FROM fact_results fr
            LEFT JOIN canonical_metrics cmet ON cmet.id = fr.metric_key
            WHERE fr.composite_slug         IS NOT NULL
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
            CASE
                WHEN cmet.min_score IS NULL OR cmet.max_score IS NULL
                  OR cmet.max_score = cmet.min_score THEN NULL
                ELSE (pf.avg_score - cmet.min_score) / (cmet.max_score - cmet.min_score)
            END                                          AS avg_score_norm,
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
        LEFT JOIN canonical_metrics cmet ON cmet.id = pm.metric_id
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
        WHERE EXISTS (
            -- benchmarks dim's `benchmark_id` is the canonical-or-raw
            -- key, so match it against fr.benchmark_key (not the
            -- canonical-only fr.benchmark_id) — otherwise raw-only
            -- benchmark rows in the dim never find their fact rows
            -- and get dropped from evals_view.
            SELECT 1 FROM fact_results fr
            WHERE fr.composite_slug = b.composite_slug
              AND fr.benchmark_key  = b.benchmark_id
        )
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
    least one source reports it (post-fold), else the Q1 fallback:
    most observations, ties by distinct models then lexicographic.
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
            SELECT r.benchmark_id, r.evaluation_id, r.composite_slug,
                   r.composite_display_name, r.family_id, r.family_display_name,
                   r.metric_id_effective, r.model_key, r.model_info,
                   r.score, r.score_canonical, r.scale_conversion,
                   CAST(NULL AS VARCHAR) AS slice_id,
                   CAST(NULL AS VARCHAR) AS slice_display_name
            FROM eval_results_view r
            JOIN canonical_benchmarks cb ON cb.id = r.benchmark_id
            WHERE NOT r.is_slice AND r.score IS NOT NULL
        ),
        sl AS (
            SELECT r.parent_benchmark_id AS benchmark_id,
                   CAST(NULL AS VARCHAR) AS evaluation_id,
                   r.composite_slug, r.composite_display_name,
                   r.family_id, r.family_display_name,
                   r.metric_id_effective, r.model_key, r.model_info,
                   r.score, r.score_canonical, r.scale_conversion,
                   r.benchmark_id AS slice_id,
                   COALESCE(cbs.display_name, r.benchmark_id)
                       AS slice_display_name
            FROM eval_results_view r
            JOIN canonical_benchmarks cb ON cb.id = r.parent_benchmark_id
            LEFT JOIN canonical_benchmarks cbs ON cbs.id = r.benchmark_id
            WHERE r.is_slice AND r.score IS NOT NULL
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
                   COUNT(*)                        AS results_count,
                   COUNT(DISTINCT model_key)       AS models_count,
                   COUNT(DISTINCT composite_slug)  AS sources_count
            FROM page_rows
            GROUP BY 1, 2
        ),
        chosen AS (
            SELECT ms.*,
                   (ms.metric_id_effective = cb.preferred_metric_id) AS is_registry_preferred,
                   ROW_NUMBER() OVER (
                       PARTITION BY ms.benchmark_id
                       ORDER BY
                           COALESCE(ms.metric_id_effective = cb.preferred_metric_id, FALSE) DESC,
                           ms.results_count DESC,
                           ms.models_count DESC,
                           ms.metric_id_effective ASC
                   ) AS rk
            FROM metric_stats ms
            LEFT JOIN canonical_benchmarks cb ON cb.id = ms.benchmark_id
        ),
        default_metric AS (
            SELECT benchmark_id,
                   metric_id_effective AS default_metric_id,
                   COALESCE(is_registry_preferred, FALSE) AS preferred_from_registry,
                   results_count, models_count, sources_count
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
                           p.composite_slug ASC
                   ) AS rk,
                   p.model_info."name" AS model_name,
                   p.model_key, p.score, p.score_canonical,
                   p.composite_slug, p.evaluation_id
            FROM page_rows p
            JOIN default_metric dm
              ON dm.benchmark_id = p.benchmark_id
             AND p.metric_id_effective = dm.default_metric_id
            LEFT JOIN canonical_metrics cm ON cm.id = dm.default_metric_id
            WHERE p.scale_conversion != 'flagged'
              -- slice-grain pages get NO best_result: a best across
              -- different slices compares incomparables (same rule as the
              -- comparison-index merged entries)
              AND p.grain = 'benchmark'
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
            LEFT JOIN canonical_metrics cm ON cm.id = ms.metric_id_effective
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
        LEFT JOIN canonical_metrics cm    ON cm.id = dm.default_metric_id
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
        ("eval_results_view", "(composite_slug, metric_summary_id, model_key)"),
        ("models_view",       "(model_key)"),
        ("evals_view",        "(evaluation_id)"),
        ("merged_evals_view", "(evaluation_id)"),
    ]:
        path = out_dir / f"{table}.parquet"
        con.execute(
            f"""
            COPY (SELECT * FROM {table} ORDER BY {sort_key} NULLS LAST)
            TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

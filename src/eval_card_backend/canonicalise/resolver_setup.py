"""Register the resolver and all helper UDFs against a DuckDB connection.

Single entry point: `register_udfs(con, resolver)`. Idempotent — re-registering
a UDF on an existing connection raises, so callers should use a fresh
connection per pipeline run.
"""
from __future__ import annotations

from eval_entity_resolver.eee import clean_eval_name, extract_metric

from eval_card_backend import categorisation, slugs
from eval_card_backend.canonicalise import udfs


def register_udfs(con, resolver) -> None:
    resolve_canonical_id_py, resolve_strategy_py = udfs.make_resolver_udfs(resolver)

    con.create_function(
        "resolve_canonical_id", resolve_canonical_id_py,
        ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "resolve_strategy", resolve_strategy_py,
        ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "clean_eval_name_udf",
        lambda v: clean_eval_name(v) if isinstance(v, str) else None,
        ["VARCHAR"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "extract_metric_udf",
        lambda v: extract_metric(v) if isinstance(v, str) else None,
        ["VARCHAR"], "VARCHAR",
        null_handling="special",
    )

    con.create_function(
        "fact_id_udf", udfs.fact_id_py,
        ["VARCHAR", "INTEGER"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "variant_key_udf", udfs.variant_key_py,
        ["JSON"], "VARCHAR",
        null_handling="special",
    )
    # `canonical_models.parents` is a JSON list of typed edges
    # (`{id, relationship}`). Take first `variant` edge and flatten.
    # TODO this will break for merged models
    con.create_function(
        "variant_parent_id_udf", udfs.variant_parent_id_py,
        ["VARCHAR"], "VARCHAR",
        null_handling="special",
    )

    con.create_function(
        "is_agentic_udf", udfs.is_agentic_py,
        ["VARCHAR", "JSON", "JSON"], "BOOLEAN",
        null_handling="special",
    )
    # JSON-typed UDF params arrive from DuckDB as serialised strings; parse
    # before canonical_json json.dumps's the input, otherwise the result is
    # doubly encoded.
    from eval_card_backend.signals.setup import _coerce_json

    con.create_function(
        "canonical_json_udf",
        lambda v: udfs.canonical_json(_coerce_json(v, caller="canonical_json_udf")),
        ["JSON"], "VARCHAR",
        null_handling="special",
    )

    completeness_struct_type = (
        "STRUCT(completeness_score DOUBLE, total_fields_evaluated INTEGER, "
        "populated_count DOUBLE, missing_required_fields VARCHAR[], "
        'partial_fields STRUCT(field_path VARCHAR, score DOUBLE, '
        "populated_subitems INTEGER, total_subitems INTEGER)[])"
    )
    # Per-row completeness UDF. Card is benchmark-level (JOINed at Stage D);
    # the next three args are the row's source_metadata; the last two are
    # reserved evalcards fields (NULL today, columns reserved for future use).
    con.create_function(
        "compute_completeness_udf", udfs.compute_completeness_py,
        ["JSON", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR"],
        completeness_struct_type,
        null_handling="special",
    )

    # Group-level signal UDFs. `differing_setup_fields` is a list of
    # STRUCT(field, "values"); the UDF body produces those structs directly.
    group_row_type = (
        'STRUCT(fact_id VARCHAR, evaluation_id VARCHAR, score DOUBLE, '
        "generation_args VARCHAR, evaluator_relationship VARCHAR, "
        "source_organization_name VARCHAR)"
    )
    metric_cfg_type = (
        "STRUCT(metric_kind VARCHAR, metric_unit VARCHAR, "
        "min_score DOUBLE, max_score DOUBLE)"
    )
    variant_out_type = (
        "STRUCT(has_variant_divergence BOOLEAN, divergence_magnitude DOUBLE, "
        "threshold_used DOUBLE, threshold_basis VARCHAR, "
        'differing_setup_fields STRUCT(field VARCHAR, "values" JSON)[])'
    )
    cross_out_type = (
        "STRUCT(has_cross_party_divergence BOOLEAN, divergence_magnitude DOUBLE, "
        "threshold_used DOUBLE, threshold_basis VARCHAR, "
        'differing_setup_fields STRUCT(field VARCHAR, "values" JSON)[], '
        "organization_count INTEGER, "
        "scores_by_organization MAP(VARCHAR, DOUBLE))"
    )
    con.create_function(
        "compute_variant_divergence_udf",
        udfs.compute_variant_divergence_udf_body,
        [f"{group_row_type}[]", metric_cfg_type], variant_out_type,
        null_handling="special",
    )
    con.create_function(
        "compute_cross_party_divergence_udf",
        udfs.compute_cross_party_divergence_udf_body,
        [f"{group_row_type}[]", metric_cfg_type], cross_out_type,
        null_handling="special",
    )

    # Per-row metric-meta resolver. Both `canonical_metrics` (registry) and
    # the per-record `metric_config` (EEE) are sparse for kind/unit, so
    # Stage D can't read either directly. The UDF runs a layered chain
    # (registry → EEE → heuristic → default) and returns the five resolved
    # fields plus per-row provenance for kind and unit.
    metric_meta_out_type = (
        "STRUCT(metric_kind VARCHAR, metric_unit VARCHAR, "
        "min_score DOUBLE, max_score DOUBLE, lower_is_better BOOLEAN, "
        "metric_kind_provenance VARCHAR, metric_unit_provenance VARCHAR)"
    )
    con.create_function(
        "derive_metric_meta_udf",
        udfs.derive_metric_meta_udf_body,
        [
            "JSON",      # eee_metric_config (per-record blob)
            "VARCHAR",   # registry_metric_kind   (today: NULL)
            "VARCHAR",   # registry_metric_unit   (today: NULL)
            "DOUBLE",    # registry_min_score
            "DOUBLE",    # registry_max_score
            "BOOLEAN",   # registry_lower_is_better
            "VARCHAR",   # metric_name (drives the regex heuristic)
            "VARCHAR",   # registry_score_type (binary|continuous|levels — feeds proportion-shape heuristic)
        ],
        metric_meta_out_type,
        null_handling="special",
    )

    # View-layer slug UDFs. Producer-owned, RFC 3986 percent-encoded.
    con.create_function(
        "url_encode_udf", slugs.url_encode,
        ["VARCHAR"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "metric_summary_id_udf", slugs.metric_summary_id,
        ["VARCHAR", "VARCHAR"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "is_summary_score_udf", slugs.is_summary_score,
        ["VARCHAR", "VARCHAR", "VARCHAR"], "BOOLEAN",
        null_handling="special",
    )

    # Producer-owned benchmark categorisation. Output is constrained to
    # the typed CategoryType enum; default 'General' on no-match.
    con.create_function(
        "categorise_benchmark_udf", categorisation.classify_benchmark,
        ["VARCHAR[]", "VARCHAR[]", "VARCHAR[]"], "VARCHAR",
        null_handling="special",
    )

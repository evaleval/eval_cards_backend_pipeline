"""Flat row-level analytics parquet for ad-hoc data exploration.

Emits ``output/duckdb/v1/evaluations.parquet`` — one row per
``(eval_summary, metric_summary, evaluation_result row)`` triple,
with already-resolved registry-aware columns (canonical_benchmark_id,
source_organization_normalized) so DuckDB queries don't need to
reimplement any normalization.

Strictly additive: not consumed by the frontend, doesn't replace any
existing output. Intended for analyst / developer use:

    duckdb> SELECT canonical_benchmark_id, model_route_id,
    ...        COUNT(DISTINCT source_organization_normalized) AS orgs
    ... FROM 'output/duckdb/v1/evaluations.parquet'
    ... GROUP BY 1, 2 HAVING orgs > 1;

Schema deliberately excludes group-level signals (multi_source,
cross_party_divergence_magnitude). Those depend on the chosen grouping
rule and should be recomputed via SQL when needed — having
precomputed columns would lock in a specific rule.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.signals import normalize_org_name


PARQUET_RELATIVE_PATH = Path("duckdb") / "v1" / "evaluations.parquet"


def _summary_grouping_key(summary: dict) -> str:
    """Same fallback chain as `pipeline._summary_canonical_grouping_key`
    but resolved to a single string column for SQL convenience.
    Mirrored here to avoid a cross-module call from the emitter."""
    canonical = summary.get("canonical_benchmark_id") or ""
    if canonical:
        return canonical
    eval_summary_id = summary.get("eval_summary_id") or ""
    if eval_summary_id:
        return f"summary:{eval_summary_id}"
    family = summary.get("benchmark_family_key") or ""
    if family:
        return f"family:{family}"
    return ""


def _iter_rows(eval_summaries: list[dict]):
    """Walk eval_summaries → root metrics + subtask metrics →
    model_results. Yields (summary, metric, row) tuples."""
    for summary in eval_summaries:
        for metric in summary.get("metrics") or []:
            for row in metric.get("model_results") or []:
                yield summary, metric, row
        for subtask in summary.get("subtasks") or []:
            for metric in subtask.get("metrics") or []:
                for row in metric.get("model_results") or []:
                    yield summary, metric, row


def _flatten(summary: dict, metric: dict, row: dict) -> dict:
    """Build one flat record. All columns nullable; types chosen to
    survive parquet round-trip cleanly."""
    sm = row.get("source_metadata")
    if not isinstance(sm, dict):
        sm = {}
    annotations = ((row.get("evalcards") or {}).get("annotations") or {})
    repro = annotations.get("reproducibility_gap")
    repro_gap = bool(repro.get("has_reproducibility_gap")) if isinstance(repro, dict) else None
    repro_missing_count = (
        len(repro.get("missing_fields") or []) if isinstance(repro, dict) else None
    )
    repro_required_count = (
        repro.get("required_field_count") if isinstance(repro, dict) else None
    )
    provenance = annotations.get("provenance")
    if isinstance(provenance, dict):
        prov_source_type = provenance.get("source_type")
        prov_first_party_only = bool(provenance.get("first_party_only"))
    else:
        prov_source_type = None
        prov_first_party_only = None

    # generation_args: lives on the private `_generation_args` row
    # field (stashed for variant_divergence comparison). Stripped
    # before serialization, but we emit before strip so it's still
    # present here.
    gen_args = row.get("_generation_args")
    if gen_args is not None:
        try:
            gen_args_str: str | None = json.dumps(gen_args, sort_keys=True)
        except (TypeError, ValueError):
            gen_args_str = None
    else:
        gen_args_str = None

    org_raw = sm.get("source_organization_name")
    org_normalized = normalize_org_name(org_raw) if org_raw else None

    return {
        # identity
        "eval_summary_id": summary.get("eval_summary_id"),
        "metric_summary_id": metric.get("metric_summary_id"),
        "evaluation_id": row.get("evaluation_id"),
        # benchmark
        "canonical_benchmark_id": summary.get("canonical_benchmark_id"),
        "benchmark_grouping_key": _summary_grouping_key(summary),
        "benchmark_family_key": summary.get("benchmark_family_key"),
        "benchmark_leaf_key": summary.get("benchmark_leaf_key"),
        "benchmark_leaf_name": summary.get("benchmark_leaf_name"),
        "metric_key": metric.get("metric_key"),
        "metric_name": metric.get("metric_name"),
        "lower_is_better": metric.get("lower_is_better"),
        # model
        "model_route_id": row.get("model_route_id"),
        "model_family_id": row.get("model_id"),
        "model_developer": row.get("developer"),
        # score
        "score": row.get("score"),
        "variant_key": row.get("variant_key"),
        # source / provenance (raw + normalized so SQL doesn't reimplement)
        "source_organization_raw": org_raw,
        "source_organization_normalized": org_normalized,
        "evaluator_relationship": sm.get("evaluator_relationship"),
        "source_type": prov_source_type,
        # eval_library_name/version were considered but the data lives
        # on the original EEE record (top-level eval_library dict) and
        # isn't carried through into the row dicts by the legacy build
        # path. Skip rather than emit a column of all-nulls. Add it to
        # the flatten path if a query needs harness-level distinct
        # counts.
        # row-level reproducibility facts
        "reproducibility_has_gap": repro_gap,
        "reproducibility_missing_field_count": repro_missing_count,
        "reproducibility_required_field_count": repro_required_count,
        "provenance_first_party_only": prov_first_party_only,
        # categorization
        "category": summary.get("category"),
        "is_summary_score": bool(summary.get("is_summary_score")),
        # generation args (kept as JSON string — variable shape per source)
        "generation_args_json": gen_args_str,
    }


def write_evaluations_parquet(eval_summaries: list[dict], output_dir: Path) -> Path:
    """Build and write the analytics parquet. Returns the path written.

    Called from ``pipeline.main()`` AFTER ``attach_canonical_signals``
    (so per-row provenance / reproducibility annotations are present)
    but BEFORE ``_strip_signals_internals`` (so private
    ``_generation_args`` rows are still available)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [_flatten(s, m, r) for s, m, r in _iter_rows(eval_summaries)]

    if not rows:
        # Empty pipeline run; still emit the file with the right schema
        # so downstream tooling can rely on the path being present.
        rows = [
            {k: None for k in _flatten({}, {}, {}).keys()}
        ][:0]

    target = Path(output_dir) / PARQUET_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, target)
    print(
        f"[pipeline] {json.dumps({'event': 'evaluations_parquet.write', 'path': str(target), 'rows': len(rows)})}"
    )
    return target

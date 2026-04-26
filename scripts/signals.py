"""EvalCards interpretive signals.

Implements functions for:
- Signal 1 (Reproducibility Gap)
- Signal 2 (Reporting Completeness)
- Signal 3 (Provenance)
- Signal 4 (Comparability — variant + cross-party divergence)

Plus per-record summarizers (`summarize_*`) and corpus-level aggregators
(`aggregate_*`) with `stratify` orchestrators for `by_category` breakdowns.
"""

import json
import re
import statistics
from pathlib import Path
from typing import Any

from scripts.helpers.benchmark_identity import is_agentic_benchmark_name

SIGNAL_VERSION = "1.0"

# Original spec specifies 4 main reproducibility fields.
# Kept here verbatim for reference;
# `compute_reproducibility_gap` accepts a `base_fields=` override so
# callers / tests can run against this set when needed.
SPEC_BASE_REPRODUCIBILITY_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "prompt_template",
)

# Actual current fields used to illustrate reproducibility gaps
BASE_REPRODUCIBILITY_FIELDS = (
    "temperature",
    # "top_p",            # disabled 2026-04-26
    "max_tokens",
    # "prompt_template",  # disabled 2026-04-26
)
AGENTIC_REPRODUCIBILITY_FIELDS = ("eval_plan", "eval_limits")

AGENTIC_TASK_TOKENS = frozenset({"agentic", "tool_use", "multi_step_agent"})


# Operationalizes spec §3.4 "category tags suggest agentic" using shared helper
# constants so Signal B stays consistent across scripts.

COMPLETENESS_FIELDS_PATH = (
    Path(__file__).resolve().parent.parent / "registry" / "completeness_fields.json"
)


def load_completeness_field_set(path: Path = COMPLETENESS_FIELDS_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("fields") or [])


COMPLETENESS_FIELD_SET: list[dict] = load_completeness_field_set()


def is_populated(obj: Any, field: str) -> bool:
    """Spec §2.3: a field is populated when present and non-null.

    Empty string for `prompt_template` is explicitly populated per §3.4 and
    falls out of this rule naturally (an empty string is not None).
    """
    if not isinstance(obj, dict):
        return False
    if field not in obj:
        return False
    return obj[field] is not None


def _resolve_path(record: Any, path: str) -> Any:
    current = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def is_agentic(
    benchmark_name: str | None,
    benchmark_card: dict | None,
    generation_args: dict | None,
) -> bool:
    """Spec tasks-literal ∪ Signal A ∪ Signal B.

    Spec tasks-literal (§3.1): benchmark card's
    `purpose_and_intended_users.tasks` includes any of the literal tokens
    `agentic`, `tool_use`, `multi_step_agent`.

    Signal A (§3.1): EEE `generation_args.agentic_eval_config` is present
    and non-null.

    Signal B (§3.4 operationalization): benchmark name matches the same
    regex `infer_category_from_benchmark` uses to label "agentic".
    """
    if isinstance(benchmark_card, dict):
        purpose = benchmark_card.get("purpose_and_intended_users") or {}
        tasks = purpose.get("tasks") or []
        if isinstance(tasks, list):
            for task in tasks:
                if (
                    isinstance(task, str)
                    and task.strip().lower() in AGENTIC_TASK_TOKENS
                ):
                    return True

    if (
        isinstance(generation_args, dict)
        and generation_args.get("agentic_eval_config") is not None
    ):
        return True

    if is_agentic_benchmark_name(benchmark_name):
        return True

    return False


def compute_reproducibility_gap(
    generation_args: dict | None,
    is_agentic_benchmark: bool,
    base_fields: tuple[str, ...] | None = None,
    agentic_fields: tuple[str, ...] | None = None,
) -> dict | None:
    """Spec §3.2.

    Returns None reserves the spec's "no EEE record at all" case (§3.4); in
    this pipeline that case is structurally impossible (we iterate EEE
    records to build outputs), but we keep the return type spec-faithful.

    `missing_fields` uses bare field names (matches spec §3.3 example) since
    all required fields come from one nested location (`generation_args`).

    `base_fields` / `agentic_fields` default to `BASE_REPRODUCIBILITY_FIELDS`
    / `AGENTIC_REPRODUCIBILITY_FIELDS` (the active runtime subsets). Callers
    / tests can pass `SPEC_BASE_REPRODUCIBILITY_FIELDS` explicitly to verify
    spec-literal behavior.
    """
    base = tuple(
        base_fields if base_fields is not None else BASE_REPRODUCIBILITY_FIELDS
    )
    agentic = tuple(
        agentic_fields if agentic_fields is not None else AGENTIC_REPRODUCIBILITY_FIELDS
    )
    required: list[str] = list(base)
    if is_agentic_benchmark:
        required.extend(agentic)

    missing = [field for field in required if not is_populated(generation_args, field)]

    return {
        "has_reproducibility_gap": len(missing) > 0,
        "missing_fields": missing,
        "required_field_count": len(required),
        "populated_field_count": len(required) - len(missing),
        "signal_version": SIGNAL_VERSION,
    }


def compute_reporting_completeness(
    joined_record: dict,
    field_set: list[dict] | None = None,
) -> dict:
    """Spec §4.2.

    `joined_record` is a nested dict keyed by source — typically:
        {
            "autobenchmarkcard": <ABC card dict>,
            "eee_eval": {"source_metadata": <merged source_metadata dict>},
            "evalcards": {...},
        }
    Each field path in `field_set` is resolved against this structure with
    dotted-path traversal; absent intermediate keys are treated as missing.

    `missing_required_fields` uses full dotted paths (matches spec §4.3
    example); `field_scores[].field_path` uses the same convention.
    """
    fields = field_set if field_set is not None else COMPLETENESS_FIELD_SET

    field_scores: list[dict] = []
    for field in fields:
        path = field["path"]
        coverage = field["coverage"]

        if coverage in ("full", "reserved"):
            value = _resolve_path(joined_record, path)
            score = 1.0 if value is not None else 0.0
            field_scores.append(
                {"field_path": path, "coverage_type": coverage, "score": score}
            )
        elif coverage == "partial":
            subitem_paths: list[str] = list(field.get("subitem_paths") or [])
            total = len(subitem_paths)
            populated = sum(
                1
                for sp in subitem_paths
                if _resolve_path(joined_record, sp) is not None
            )
            score = (populated / total) if total else 0.0
            field_scores.append(
                {"field_path": path, "coverage_type": coverage, "score": score}
            )
        else:
            raise ValueError(f"Unknown coverage type: {coverage!r} for field {path!r}")

    total_fields = len(field_scores)
    completeness_score = (
        sum(fs["score"] for fs in field_scores) / total_fields if total_fields else 0.0
    )

    missing_required_fields = [
        fs["field_path"] for fs in field_scores if fs["score"] == 0
    ]

    partial_fields: list[dict] = []
    for fs, field in zip(field_scores, fields):
        if fs["coverage_type"] != "partial" or not (0 < fs["score"] < 1):
            continue
        subitem_paths = list(field.get("subitem_paths") or [])
        total = len(subitem_paths)
        populated = sum(
            1 for sp in subitem_paths if _resolve_path(joined_record, sp) is not None
        )
        partial_fields.append(
            {
                "field_path": fs["field_path"],
                "score": fs["score"],
                "populated_subitems": populated,
                "total_subitems": total,
            }
        )

    return {
        "completeness_score": completeness_score,
        "total_fields_evaluated": total_fields,
        "missing_required_fields": missing_required_fields,
        "partial_fields": partial_fields,
        "field_scores": field_scores,
        "signal_version": SIGNAL_VERSION,
    }


# ---------------------------------------------------------------------------
# Signal 3: Provenance (spec §5)
# Signal 4: Comparability — variant + cross-party divergence (spec §6)
# ---------------------------------------------------------------------------


# Spec §6.1.2 verbatim. agentic_eval_config is unconditionally compared:
# for non-agentic groups every row's value is None, the canonical value
# set has size 1, and the field never enters differing_setup_fields.
GENERATION_ARGS_COMPARISON_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "prompt_template",
    "reasoning",
    "agentic_eval_config",
)

PROVENANCE_BUCKETS: tuple[str, ...] = (
    "first_party",
    "third_party",
    "collaborative",
    "unspecified",
)

_WHITESPACE_REGEX = re.compile(r"\s+")


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_org_name(name: Any) -> str | None:
    """Whitespace + case normalization for source_organization_name set membership.

    Strips and collapses internal whitespace, lowercases. Returns None for
    None / non-strings / empty results so the caller can exclude trivially.
    Display casing is preserved separately by the caller.
    """
    if not isinstance(name, str):
        return None
    cleaned = _WHITESPACE_REGEX.sub(" ", name).strip()
    if not cleaned:
        return None
    return cleaned.lower()


def canonicalize_setup_value(value: Any) -> str:
    """Stable JSON canonical form for a single setup field's value.

    Used for set-membership comparison across rows. Order-independent for
    dicts (sort_keys=True). Non-JSON-native values fall through `default=str`.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def compute_threshold(metric_config: Any) -> tuple[float, str]:
    """Spec §6.1.2. Returns (threshold, basis).

    `basis` records which rule produced the threshold so the annotation can
    surface it (extension beyond spec example outputs). Values:
    `proportion_or_continuous_normalized`, `percent`, `range_5pct`,
    `fallback_default`.
    """
    if isinstance(metric_config, dict):
        metric_unit = metric_config.get("metric_unit")
        metric_kind = metric_config.get("metric_kind")
        if metric_unit == "proportion" or metric_kind == "continuous_normalized":
            return 0.05, "proportion_or_continuous_normalized"
        if metric_unit == "percent":
            return 5.0, "percent"
        min_score = metric_config.get("min_score")
        max_score = metric_config.get("max_score")
        if _is_real_number(min_score) and _is_real_number(max_score):
            score_range = max_score - min_score
            if score_range > 0:
                return 0.05 * score_range, "range_5pct"
    return 0.05, "fallback_default"


def aggregated_setup(rows_for_org: list[dict]) -> dict | None:
    """Spec §6.2.2 — deterministic lower-median rule.

    Sorts rows by (score, evaluation_id) and returns the generation_args of
    the row at index `(n - 1) // 2`. For odd n this is the median row; for
    even n this is the lower of the two middle rows. None scores sort to
    the end (treated as +inf) so they never become the representative when
    real scores exist.
    """
    if not rows_for_org:
        return None
    if len(rows_for_org) == 1:
        return rows_for_org[0].get("generation_args")
    sorted_rows = sorted(
        rows_for_org,
        key=lambda r: (
            r["score"] if _is_real_number(r.get("score")) else float("inf"),
            str(r.get("evaluation_id") or ""),
        ),
    )
    n = len(sorted_rows)
    representative = sorted_rows[(n - 1) // 2]
    return representative.get("generation_args")


def _group_variant_breakdown(group_rows: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for row in group_rows:
        vk = row.get("variant_key") or "default"
        counts[vk] = counts.get(vk, 0) + 1
    return [{"variant_key": k, "row_count": v} for k, v in sorted(counts.items())]


def _differing_setup_fields(setups: list[Any]) -> list[dict]:
    """Returns spec §6.1.2 / §6.2.2 differing-fields list with original values.

    For each comparison field, canonicalizes the value across all setups; if
    the canonical-value set has size > 1, records the field with the unique
    original values (in first-seen order, deduped by canonical form).
    """
    differing: list[dict] = []
    for field in GENERATION_ARGS_COMPARISON_FIELDS:
        seen_canon: set[str] = set()
        original_values: list[Any] = []
        for setup in setups:
            value = setup.get(field) if isinstance(setup, dict) else None
            canon = canonicalize_setup_value(value)
            if canon not in seen_canon:
                seen_canon.add(canon)
                original_values.append(value)
        if len(seen_canon) > 1:
            differing.append({"field": field, "values": original_values})
    return differing


def compute_provenance(group_rows: list[dict]) -> list[dict]:
    """Spec §5.2 — one annotation dict per row in the group.

    Each row's input is expected to carry:
      - source_metadata: dict | None  (with evaluator_relationship,
        source_organization_name)

    Returns a list of dicts, in the same order as `group_rows`. Each dict has
    a per-row `source_type` (other / null collapsed to "unspecified") and the
    group-level `is_multi_source`, `first_party_only`, and
    `distinct_reporting_organizations` fields. `first_party_only` requires
    both `source_type == first_party` AND exactly one distinct named org in
    the group (§5.2 pseudocode literal).
    """
    distinct_orgs: set[str] = set()
    for row in group_rows:
        sm = row.get("source_metadata")
        if isinstance(sm, dict):
            normalized = normalize_org_name(sm.get("source_organization_name"))
            if normalized:
                distinct_orgs.add(normalized)

    distinct_count = len(distinct_orgs)
    is_multi_source = distinct_count > 1

    annotations: list[dict] = []
    for row in group_rows:
        sm = (
            row.get("source_metadata")
            if isinstance(row.get("source_metadata"), dict)
            else {}
        )
        relationship = (
            sm.get("evaluator_relationship") if isinstance(sm, dict) else None
        )
        if relationship is None or relationship == "other":
            source_type = "unspecified"
        else:
            source_type = relationship

        first_party_only = source_type == "first_party" and distinct_count == 1

        annotations.append(
            {
                "source_type": source_type,
                "is_multi_source": is_multi_source,
                "first_party_only": first_party_only,
                "distinct_reporting_organizations": distinct_count,
                "signal_version": SIGNAL_VERSION,
            }
        )
    return annotations


def compute_variant_divergence(
    group_rows: list[dict],
    metric_config: Any,
    group_id: str | None = None,
) -> dict | None:
    """Spec §6.1.2 — group-level dict, or None when not applicable.

    Returns None when:
      (a) <2 rows in the group;
      (b) all rows have identical setups across the comparison fields;
      (c) <2 rows have non-null scores after exclusions.

    The returned dict carries the group-level fields. The pipeline injects
    a per-row `this_triple_score` (§6.1.3 example) when copying onto rows.
    """
    if len(group_rows) < 2:
        return None

    setups = [row.get("generation_args") for row in group_rows]
    differing_fields = _differing_setup_fields(setups)
    if not differing_fields:
        return None

    rows_with_score = [r for r in group_rows if _is_real_number(r.get("score"))]
    if len(rows_with_score) < 2:
        return None

    scores = [r["score"] for r in rows_with_score]
    divergence = max(scores) - min(scores)
    threshold, threshold_basis = compute_threshold(metric_config)
    has_variant_divergence = divergence > threshold

    metric_unit = (
        metric_config.get("metric_unit") if isinstance(metric_config, dict) else None
    )
    score_scale_anomaly = bool(
        metric_unit == "proportion" and any((s < 0 or s > 1) for s in scores)
    )

    return {
        "has_variant_divergence": has_variant_divergence,
        "group_id": group_id,
        "divergence_magnitude": divergence,
        "threshold_used": threshold,
        "threshold_basis": threshold_basis,
        "differing_setup_fields": differing_fields,
        "scores_in_group": scores,
        "triple_count_in_group": len(rows_with_score),
        "score_scale_anomaly": score_scale_anomaly,
        "group_variant_breakdown": _group_variant_breakdown(group_rows),
        "signal_version": SIGNAL_VERSION,
    }


def compute_cross_party_divergence(
    group_rows: list[dict],
    metric_config: Any,
    group_id: str | None = None,
) -> dict | None:
    """Spec §6.2.2 — group-level dict, or None when <2 distinct named orgs.

    Per §6.2.4: rows with null `source_organization_name` are excluded from
    the org set; if fewer than 2 named orgs remain, returns None. Per-org
    score is the standard median; per-org representative setup is the
    lower-median triple's setup (`aggregated_setup`). The two median rules
    are intentionally asymmetric (§6.2.2).
    """
    by_org_normalized: dict[str, list[dict]] = {}
    org_display: dict[str, str] = {}

    for row in group_rows:
        if not _is_real_number(row.get("score")):
            continue
        sm = row.get("source_metadata")
        if not isinstance(sm, dict):
            continue
        raw_org = sm.get("source_organization_name")
        normalized = normalize_org_name(raw_org)
        if not normalized:
            continue
        by_org_normalized.setdefault(normalized, []).append(row)
        if normalized not in org_display:
            display = (
                _WHITESPACE_REGEX.sub(" ", raw_org).strip()
                if isinstance(raw_org, str)
                else ""
            )
            org_display[normalized] = display or normalized

    if len(by_org_normalized) < 2:
        return None

    org_scores: dict[str, float] = {}
    org_setups: dict[str, dict | None] = {}
    for normalized, rows in by_org_normalized.items():
        org_scores[normalized] = statistics.median(r["score"] for r in rows)
        org_setups[normalized] = aggregated_setup(rows)

    divergence = max(org_scores.values()) - min(org_scores.values())
    threshold, threshold_basis = compute_threshold(metric_config)
    has_cross_party_divergence = divergence > threshold

    differing_fields = _differing_setup_fields(list(org_setups.values()))

    scores_by_organization = {
        org_display[normalized]: score for normalized, score in org_scores.items()
    }

    return {
        "has_cross_party_divergence": has_cross_party_divergence,
        "group_id": group_id,
        "divergence_magnitude": divergence,
        "threshold_used": threshold,
        "threshold_basis": threshold_basis,
        "scores_by_organization": scores_by_organization,
        "differing_setup_fields": differing_fields,
        "organization_count": len(by_org_normalized),
        "group_variant_breakdown": _group_variant_breakdown(group_rows),
        "signal_version": SIGNAL_VERSION,
    }


# ---------------------------------------------------------------------------
# Rollups (deviation from spec §7's query-time aggregation)
# ---------------------------------------------------------------------------


def summarize_provenance(
    per_row_provenance: list[dict],
    per_group_signals: list[dict],
) -> dict:
    """Per-eval / per-model provenance rollup.

    `per_row_provenance` — one provenance annotation per row in scope.
    Drives `total_results` and `source_type_distribution`.

    `per_group_signals` — one record per group in scope, carrying
    `is_multi_source` and `first_party_only_in_group` flags. Drives
    `total_groups`, `multi_source_groups`, `first_party_only_groups`. The
    eligible-groups counts are essential because per-group facts cannot be
    counted from per-row data without double-counting.
    """
    total_results = len(per_row_provenance)
    total_groups = len(per_group_signals)
    multi_source_groups = sum(1 for g in per_group_signals if g.get("is_multi_source"))
    first_party_only_groups = sum(
        1 for g in per_group_signals if g.get("first_party_only_in_group")
    )

    distribution = {bucket: 0 for bucket in PROVENANCE_BUCKETS}
    for annotation in per_row_provenance:
        bucket = annotation.get("source_type") if isinstance(annotation, dict) else None
        if bucket not in distribution:
            bucket = "unspecified"
        distribution[bucket] += 1

    return {
        "total_results": total_results,
        "total_groups": total_groups,
        "multi_source_groups": multi_source_groups,
        "first_party_only_groups": first_party_only_groups,
        "source_type_distribution": distribution,
    }


def summarize_comparability(per_group_signals: list[dict]) -> dict:
    """Per-eval / per-model comparability rollup.

    `per_group_signals` — list of {variant_divergence, cross_party_divergence}
    per group in scope (each may be None). Eligibility (`groups_with_*_check`)
    is the non-null count; divergent counts are the subset where the
    matching `has_*_divergence` flag is true.
    """
    total_groups = len(per_group_signals)

    groups_with_variant_check = 0
    variant_divergent_count = 0
    groups_with_cross_party_check = 0
    cross_party_divergent_count = 0
    for entry in per_group_signals:
        variant = entry.get("variant_divergence") if isinstance(entry, dict) else None
        if isinstance(variant, dict):
            groups_with_variant_check += 1
            if variant.get("has_variant_divergence"):
                variant_divergent_count += 1
        cross = entry.get("cross_party_divergence") if isinstance(entry, dict) else None
        if isinstance(cross, dict):
            groups_with_cross_party_check += 1
            if cross.get("has_cross_party_divergence"):
                cross_party_divergent_count += 1

    return {
        "total_groups": total_groups,
        "groups_with_variant_check": groups_with_variant_check,
        "groups_with_cross_party_check": groups_with_cross_party_check,
        "variant_divergent_count": variant_divergent_count,
        "cross_party_divergent_count": cross_party_divergent_count,
    }


def summarize_reproducibility(annotations: list[dict]) -> dict:
    """Aggregate per-row reproducibility_gap annotations.

    Shape mirrors the repo's existing `score_summary` minimalism; surfaced
    on per-eval and per-model rollups as a documented deviation from spec
    §7 (which says aggregation is query-time).
    """
    rows = [a for a in annotations if isinstance(a, dict)]
    total = len(rows)
    has_reproducibility_gap_count = sum(
        1 for a in rows if a.get("has_reproducibility_gap")
    )
    if total:
        ratios = []
        for a in rows:
            required = a.get("required_field_count") or 0
            populated = a.get("populated_field_count") or 0
            if required:
                ratios.append(populated / required)
        populated_ratio_avg = (sum(ratios) / len(ratios)) if ratios else 0.0
    else:
        populated_ratio_avg = None
    return {
        "results_total": total,
        "has_reproducibility_gap_count": has_reproducibility_gap_count,
        "populated_ratio_avg": populated_ratio_avg,
    }


# ---------------------------------------------------------------------------
# Corpus-level aggregates (spec §8 — DEPRIORITIZE in spec but worth shipping
# for paper-side analysis). Each aggregator is a pure function over already-
# computed per-row / per-group / per-eval signal outputs. The pipeline glue
# collects the inputs, calls these, and writes `corpus-aggregates.json`.
#
# Stratification: callers wrap an aggregator with `stratify` (single-input)
# or `stratify_provenance` (dual-input) to produce
#     {"overall": {...}, "by_category": {category: {...}, ...}}
# blocks. The aggregators themselves are stratification-agnostic.
# ---------------------------------------------------------------------------


def aggregate_reproducibility(rows: list[dict]) -> dict:
    """Spec §8.3.1 + §8.3.2.

    `rows` is a list of dicts each shaped:
        {"annotation": <reproducibility_gap dict>, "is_agentic": bool}
    The `is_agentic` flag is needed for §8.3.2's agentic-aware denominator
    on `eval_plan` / `eval_limits` — those fields are only required for
    agentic triples, so the missingness rate uses an agentic-only denominator.

    Per-field missingness reports the active-runtime field set
    (`BASE_REPRODUCIBILITY_FIELDS` + `AGENTIC_REPRODUCIBILITY_FIELDS`).
    Spec-disabled fields (`top_p`, `prompt_template` under the active subset)
    don't appear in `missing_fields` of any annotation, so they aren't
    counted; documented in CLAUDE.md.
    """
    total_triples = len(rows)
    triples_with_gap = sum(
        1 for r in rows if (r.get("annotation") or {}).get("has_reproducibility_gap")
    )
    agentic_count = sum(1 for r in rows if r.get("is_agentic"))

    base_fields = list(BASE_REPRODUCIBILITY_FIELDS)
    agentic_fields = list(AGENTIC_REPRODUCIBILITY_FIELDS)
    field_missing_counts: dict[str, int] = {
        f: 0 for f in (*base_fields, *agentic_fields)
    }
    for r in rows:
        for field in (r.get("annotation") or {}).get("missing_fields") or []:
            if field in field_missing_counts:
                field_missing_counts[field] += 1

    per_field_missingness: dict[str, dict] = {}
    for field, count in field_missing_counts.items():
        if field in agentic_fields:
            denom = agentic_count
            denom_label = "agentic_only"
        else:
            denom = total_triples
            denom_label = "all_triples"
        per_field_missingness[field] = {
            "missing_count": count,
            "missing_rate": (count / denom) if denom else None,
            "denominator": denom_label,
            "denominator_count": denom,
        }

    return {
        "total_triples": total_triples,
        "triples_with_reproducibility_gap": triples_with_gap,
        "reproducibility_gap_rate": (triples_with_gap / total_triples)
        if total_triples
        else None,
        "agentic_triples": agentic_count,
        "per_field_missingness": per_field_missingness,
    }


def aggregate_completeness(completeness_per_eval: list[dict]) -> dict:
    """Spec §8.4.1 — per-field population rate across benchmarks.

    `completeness_per_eval` is a list of `reporting_completeness` annotation
    dicts (one per benchmark / eval_summary). Aggregates `field_scores`
    across them.

    Output:
      - `total_benchmarks`
      - `completeness_score_mean` / `completeness_score_median` over the
        per-benchmark scores
      - `per_field_population` keyed by `field_path`:
        - `mean_score` — average of per-benchmark field_score values
        - `populated_rate` — fraction of benchmarks with score > 0
        - `fully_populated_rate` — fraction with score == 1.0
    """
    total_benchmarks = len(completeness_per_eval)
    if not total_benchmarks:
        return {
            "total_benchmarks": 0,
            "completeness_score_mean": None,
            "completeness_score_median": None,
            "per_field_population": {},
        }

    scores = [
        c.get("completeness_score") or 0.0
        for c in completeness_per_eval
        if isinstance(c, dict)
    ]
    completeness_score_mean = sum(scores) / len(scores) if scores else None
    completeness_score_median = statistics.median(scores) if scores else None

    field_scores: dict[str, list[float]] = {}
    for c in completeness_per_eval:
        for fs in (c or {}).get("field_scores") or []:
            field_scores.setdefault(fs["field_path"], []).append(float(fs["score"]))

    per_field_population: dict[str, dict] = {}
    for path, vals in sorted(field_scores.items()):
        n = len(vals)
        per_field_population[path] = {
            "mean_score": sum(vals) / n if n else 0.0,
            "populated_rate": sum(1 for v in vals if v > 0) / n if n else 0.0,
            "fully_populated_rate": sum(1 for v in vals if v == 1.0) / n if n else 0.0,
            "benchmark_count": n,
        }

    return {
        "total_benchmarks": total_benchmarks,
        "completeness_score_mean": completeness_score_mean,
        "completeness_score_median": completeness_score_median,
        "per_field_population": per_field_population,
    }


def aggregate_provenance(
    per_row_provenance: list[dict],
    per_group_signals: list[dict],
) -> dict:
    """§8 implicit (provenance corpus aggregates).

    `per_row_provenance` — list of provenance annotation dicts (one per row).
        Drives `total_triples` and `source_type_distribution`.
    `per_group_signals` — list of group entries shaped like the
        `_signal_groups` items: {is_multi_source, first_party_only_in_group, ...}.
        Drives `total_groups`, `multi_source_*`, `first_party_only_*`.

    Reports always include all 4 source_type buckets per the standardized-
    fields principle (`first_party`, `third_party`, `collaborative`,
    `unspecified`).
    """
    total_triples = len(per_row_provenance)
    total_groups = len(per_group_signals)
    multi_source_groups = sum(1 for g in per_group_signals if g.get("is_multi_source"))
    first_party_only_groups = sum(
        1 for g in per_group_signals if g.get("first_party_only_in_group")
    )

    distribution = {bucket: 0 for bucket in PROVENANCE_BUCKETS}
    for annotation in per_row_provenance:
        bucket = annotation.get("source_type") if isinstance(annotation, dict) else None
        if bucket not in distribution:
            bucket = "unspecified"
        distribution[bucket] += 1

    return {
        "total_triples": total_triples,
        "total_groups": total_groups,
        "multi_source_groups": multi_source_groups,
        "multi_source_rate": (multi_source_groups / total_groups)
        if total_groups
        else None,
        "first_party_only_groups": first_party_only_groups,
        "first_party_only_rate": (first_party_only_groups / total_groups)
        if total_groups
        else None,
        "source_type_distribution": distribution,
    }


def aggregate_comparability(per_group_signals: list[dict]) -> dict:
    """§8 implicit (comparability corpus aggregates).

    Eligibility-aware denominators: only groups where the matching signal
    is non-null contribute to the rate denominator. Reporting against
    `total_groups` would conflate "ineligible" with "tested-and-passed."

    On current-corpus data, `cross_party_eligible_groups` is typically 0
    (no genuinely multi-org groups), so `cross_party_divergence_rate`
    will be `null`. Documented per the audit in
    `notes/signals-3-4-planning.md`.
    """
    total_groups = len(per_group_signals)
    variant_eligible = 0
    variant_divergent = 0
    cross_party_eligible = 0
    cross_party_divergent = 0
    for entry in per_group_signals:
        if not isinstance(entry, dict):
            continue
        variant = entry.get("variant_divergence")
        if isinstance(variant, dict):
            variant_eligible += 1
            if variant.get("has_variant_divergence"):
                variant_divergent += 1
        cross = entry.get("cross_party_divergence")
        if isinstance(cross, dict):
            cross_party_eligible += 1
            if cross.get("has_cross_party_divergence"):
                cross_party_divergent += 1

    return {
        "total_groups": total_groups,
        "variant_eligible_groups": variant_eligible,
        "variant_divergent_groups": variant_divergent,
        "variant_divergence_rate": (variant_divergent / variant_eligible)
        if variant_eligible
        else None,
        "cross_party_eligible_groups": cross_party_eligible,
        "cross_party_divergent_groups": cross_party_divergent,
        "cross_party_divergence_rate": (cross_party_divergent / cross_party_eligible)
        if cross_party_eligible
        else None,
    }


def stratify(items_with_category: list[tuple], aggregator) -> dict:
    """Generic single-input stratifier.

    `items_with_category`: list of `(item, category)` tuples. `category=None`
        excludes the item from the `by_category` block but still includes it
        in `overall`.
    `aggregator`: callable `list[item] -> dict`.

    Returns `{"overall": {...}, "by_category": {category: {...}, ...}}`.
    Categories are sorted for deterministic output.
    """
    items = [it for it, _ in items_with_category]
    overall = aggregator(items)
    by_cat: dict[str, list] = {}
    for item, cat in items_with_category:
        if cat:
            by_cat.setdefault(cat, []).append(item)
    return {
        "overall": overall,
        "by_category": {c: aggregator(by_cat[c]) for c in sorted(by_cat)},
    }


def stratify_provenance(
    rows_with_category: list[tuple],
    groups_with_category: list[tuple],
) -> dict:
    """Dual-input stratifier for `aggregate_provenance` (which takes both
    per-row and per-group inputs).

    Both lists may have items with `category=None`, which are still counted
    in `overall` but excluded from `by_category`. Categories are the union
    of those seen in either input.
    """
    rows = [r for r, _ in rows_with_category]
    groups = [g for g, _ in groups_with_category]
    overall = aggregate_provenance(rows, groups)
    rows_by_cat: dict[str, list] = {}
    groups_by_cat: dict[str, list] = {}
    for r, c in rows_with_category:
        if c:
            rows_by_cat.setdefault(c, []).append(r)
    for g, c in groups_with_category:
        if c:
            groups_by_cat.setdefault(c, []).append(g)
    cats = sorted(set(rows_by_cat) | set(groups_by_cat))
    return {
        "overall": overall,
        "by_category": {
            c: aggregate_provenance(rows_by_cat.get(c, []), groups_by_cat.get(c, []))
            for c in cats
        },
    }

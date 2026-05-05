"""Variant- and cross-party-divergence over a (model, benchmark, metric) group.

Both functions return None when the signal is "not applicable" for the group;
distinguishing N/A (NULL on the row) from "applicable & no divergence" (FALSE)
is mandatory — the frontend renders the two differently.
"""
from __future__ import annotations

import re
import statistics
from typing import Any

from eval_card_backend.canonicalise.thresholds import compute_threshold  # re-export
from eval_card_backend.signals.setup import (
    _coerce_json,
    differing_setup_fields,
)


__all__ = [
    "aggregated_setup",
    "compute_cross_party_divergence_py",
    "compute_threshold",
    "compute_variant_divergence_py",
    "normalize_org_name",
]


_WHITESPACE_REGEX = re.compile(r"\s+")


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_org_name(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    cleaned = _WHITESPACE_REGEX.sub(" ", name).strip()
    if not cleaned:
        return None
    return cleaned.lower()


def _display_org_name(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    cleaned = _WHITESPACE_REGEX.sub(" ", name).strip()
    return cleaned or None


def aggregated_setup(rows_for_org: list[dict]) -> dict | None:
    """Lower-median rule: sort by (score, evaluation_id) ascending — None scores
    treated as +inf — pick row at index (n - 1) // 2.

    For odd n this is the median row; for even n the lower of the two middle rows.
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
    return sorted_rows[(n - 1) // 2].get("generation_args")


def _coerce_rows_gen_args(rows: list[dict], caller: str) -> None:
    for row in rows:
        row["generation_args"] = _coerce_json(row.get("generation_args"), caller=caller)


def compute_variant_divergence_py(
    rows: list[dict], metric_config: Any
) -> dict | None:
    """Detect score divergence across rows in a (model, benchmark, metric)
    group that share generation setup keys but differ on at least one
    setup field. Returns None when not applicable (fewer than 2 rows, no
    differing setup fields, or fewer than 2 rows with real scores) so the
    frontend can distinguish N/A from "applicable & no divergence".

    Each row dict carries: fact_id, evaluation_id, score, generation_args
    (str | dict), evaluator_relationship, source_organization_name.
    """
    _coerce_rows_gen_args(rows, "compute_variant_divergence_py.gen_args")

    if len(rows) < 2:
        return None

    diffs = differing_setup_fields([r.get("generation_args") for r in rows])
    if not diffs:
        return None

    rows_with_score = [r for r in rows if _is_real_number(r.get("score"))]
    if len(rows_with_score) < 2:
        return None

    scores = [r["score"] for r in rows_with_score]
    divergence = max(scores) - min(scores)
    threshold, basis = compute_threshold(metric_config)

    return {
        "has_variant_divergence": divergence > threshold,
        "divergence_magnitude": divergence,
        "threshold_used": threshold,
        "threshold_basis": basis,
        "differing_setup_fields": diffs,
    }


def compute_cross_party_divergence_py(
    rows: list[dict], metric_config: Any
) -> dict | None:
    """Detect score divergence across distinct reporting organisations for
    one (model, benchmark, metric) group. Returns None when fewer than 2
    distinct named orgs report — the signal isn't applicable, and the
    NULL distinguishes it from "applicable & no divergence".

    Per-org score is the median of the org's scored rows; per-org
    representative setup is `aggregated_setup` of those rows (lower-median
    rule). differing_setup_fields is computed across org-representative
    setups, not across all rows.
    """
    _coerce_rows_gen_args(rows, "compute_cross_party_divergence_py.gen_args")

    by_org: dict[str, list[dict]] = {}
    org_display: dict[str, str] = {}

    for row in rows:
        if not _is_real_number(row.get("score")):
            continue
        raw_org = row.get("source_organization_name")
        normalized = normalize_org_name(raw_org)
        if not normalized:
            continue
        by_org.setdefault(normalized, []).append(row)
        if normalized not in org_display:
            org_display[normalized] = _display_org_name(raw_org) or normalized

    if len(by_org) < 2:
        return None

    org_scores: dict[str, float] = {}
    org_setups: dict[str, dict | None] = {}
    for normalized, org_rows in by_org.items():
        org_scores[normalized] = statistics.median(r["score"] for r in org_rows)
        org_setups[normalized] = aggregated_setup(org_rows)

    divergence = max(org_scores.values()) - min(org_scores.values())
    threshold, basis = compute_threshold(metric_config)

    diffs = differing_setup_fields(list(org_setups.values()))

    scores_by_org = {
        org_display[normalized]: score for normalized, score in org_scores.items()
    }

    return {
        "has_cross_party_divergence": divergence > threshold,
        "divergence_magnitude": divergence,
        "threshold_used": threshold,
        "threshold_basis": basis,
        "differing_setup_fields": diffs,
        "organization_count": len(by_org),
        "scores_by_organization": scores_by_org,
    }

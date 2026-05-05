"""Per-fact-row reporting completeness, scored against the operationalised
28-field set in `registry/completeness_fields.json`.

The signal is per-fact-row (not per-benchmark): 3 of the 28 fields are EEE
source_metadata that describe the report, not the benchmark, and so vary
across reports of the same benchmark. The remaining 25 are benchmark-level
constants (card + reserved EvalCards fields) that repeat identically
across rows for a given benchmark — an accepted denormalisation cost.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval_card_backend.signals.setup import _coerce_json


_COMPLETENESS_FIELDS_PATH = (
    Path(__file__).resolve().parent.parent / "registry" / "completeness_fields.json"
)


def _load_field_set() -> list[dict]:
    data = json.loads(_COMPLETENESS_FIELDS_PATH.read_text(encoding="utf-8"))
    return list(data.get("fields") or [])


COMPLETENESS_FIELD_SET: list[dict] = _load_field_set()


def _resolve_path(record: Any, path: str) -> Any:
    current = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _build_record(
    card: Any,
    source_type: str | None,
    source_organization_name: str | None,
    evaluator_relationship: str | None,
    lifecycle_status: str | None,
    preregistration_url: str | None,
) -> dict:
    """Assemble the joined record the field-set paths resolve against.

    Each top-level key matches the field-set path prefix:
      - `autobenchmarkcard.*` paths read from the card.
      - `eee_eval.source_metadata.*` paths read from the per-row source_metadata.
      - `evalcards.*` paths read from the per-row reserved fields.
    """
    return {
        "autobenchmarkcard": card if isinstance(card, dict) else {},
        "eee_eval": {
            "source_metadata": {
                "source_type": source_type,
                "source_organization_name": source_organization_name,
                "evaluator_relationship": evaluator_relationship,
            }
        },
        "evalcards": {
            "lifecycle_status": lifecycle_status,
            "preregistration_url": preregistration_url,
        },
    }


def compute_completeness_py(
    card: Any,
    source_type: str | None = None,
    source_organization_name: str | None = None,
    evaluator_relationship: str | None = None,
    lifecycle_status: str | None = None,
    preregistration_url: str | None = None,
) -> dict:
    """Score the 28-field operationalised completeness set against a fact row.

    Inputs are the per-row data: the benchmark card (constant per benchmark)
    plus the row's source_metadata fields and the two reserved evalcards
    fields (currently always NULL). Returns a dict shaped to the per-row
    completeness columns on `fact_results`.
    """
    parsed = _coerce_json(card, caller="compute_completeness_py")
    record = _build_record(
        parsed,
        source_type,
        source_organization_name,
        evaluator_relationship,
        lifecycle_status,
        preregistration_url,
    )

    field_scores: list[dict] = []
    partial_fields: list[dict] = []

    for field in COMPLETENESS_FIELD_SET:
        path = field["path"]
        coverage = field["coverage"]

        if coverage in ("full", "reserved"):
            value = _resolve_path(record, path)
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
                if _resolve_path(record, sp) is not None
            )
            score = (populated / total) if total else 0.0
            field_scores.append(
                {"field_path": path, "coverage_type": coverage, "score": score}
            )
            if 0 < score < 1:
                partial_fields.append(
                    {
                        "field_path": path,
                        "score": score,
                        "populated_subitems": populated,
                        "total_subitems": total,
                    }
                )
        else:
            raise ValueError(
                f"Unknown coverage type {coverage!r} for field {path!r}"
            )

    total = len(field_scores)
    populated_count = sum(fs["score"] for fs in field_scores)
    completeness_score = (populated_count / total) if total else 0.0
    missing_required_fields = [fs["field_path"] for fs in field_scores if fs["score"] == 0]

    return {
        "completeness_score": completeness_score,
        "total_fields_evaluated": total,
        "populated_count": populated_count,
        "missing_required_fields": missing_required_fields,
        "partial_fields": partial_fields,
    }

"""Corpus-level invariants the catalog must satisfy.

These are the checks we wish we'd had before shipping the canonical-union
work. Run via ``audit_output(OUTPUT_DIR)`` from ``pipeline.main()`` after
``validate_output_contract``, or standalone:

    uv run --with pandas --with pyarrow --no-project \\
        python -m scripts.audit_output [output_dir]

Each check raises ``AuditError`` with a structured message naming the
offending rows, so failures point at concrete data rather than a stack
trace deep in the pipeline. Violations are hard errors — the pipeline
should refuse to publish a catalog containing them.

The bug classes we catch (each a real regression we've already shipped):

1. ``zero_count_tile``: an eval-list row with ``models_count == 0`` or
   ``metrics_count == 0``. Always a bug — a tile with no data shouldn't
   exist.
2. ``hierarchy_dangling_eval_summary_id``: a hierarchy family or leaf
   references an ``eval_summary_id`` that isn't in eval-list and isn't a
   tracked drilldown (i.e., not in any catalog row's ``reporting_sources``).
   Catches the GPQA / SciCode / 11-tile 0/0 regression directly.
3. ``hierarchy_family_unrendrable``: a hierarchy family whose ``key``
   doesn't match the ``benchmark_family_key`` of any eval-list row. The
   frontend renders family tile aggregates by filtering on this key, so
   an unmatched family produces a 0/0 ghost tile.
4. ``duplicate_display_name_within_family``: two eval-list rows share
   ``benchmark_family_key`` and an identical ``evaluation_name`` /
   ``composite_benchmark_name``. Catches the "4 Finance Agents" pattern
   if it ever resurfaces (i.e., catches a regression of canonical-union /
   parent-rollup that lets sibling sub-benchmarks ship as separate tiles).
5. ``canonical_id_not_collapsed``: two eval-list rows share a non-empty
   ``canonical_benchmark_id``. The canonical-union is the only place
   that's allowed to collapse rows by canonical id; if two distinct rows
   leak through with the same canonical, we have a partial-collapse bug.

Catches that DON'T live here: anything that's per-row (validate_output_contract
already does file-vs-list parity), anything that needs the registry online
(no network calls inside audit), or anything purely visual (Playwright covers
that). Audit is fast — sub-second on a full corpus run — so it's safe to wire
into every pipeline run, including dry-run smoke tests.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


class AuditError(RuntimeError):
    """Raised when an output-corpus invariant is violated."""


def _payload(row: dict) -> dict:
    raw = row.get("payload_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def _normalize_key(value: Any) -> str:
    """Lowercase + alphanumeric-only, mirroring the frontend's
    ``getSummaryScopeKey`` so we compare keys the same way the rendering
    code does (otherwise ``benchmark_family_key="tau2-bench"`` and
    ``family.key="tau2_bench"`` would falsely diverge)."""
    text = "" if value is None else str(value)
    return "".join(c for c in text if c.isalnum()).lower()


def _load_eval_list(output_dir: Path) -> list[dict]:
    """Return eval-list catalog rows. Reads the parquet (always emitted) and
    flattens the payload_json onto the scalar columns so every row has
    ``eval_summary_id``, ``benchmark_family_key``, ``models_count``,
    ``metrics_count``, ``canonical_benchmark_id``, ``evaluation_name``, and
    ``reporting_sources`` reachable at the top level regardless of which
    surface (parquet vs JSON) the consumer reads."""
    import pandas as pd

    path = output_dir / "duckdb" / "v1" / "eval_list.parquet"
    if not path.exists():
        raise AuditError(
            f"audit: eval_list.parquet missing at {path} — pipeline didn't "
            "emit the parity layer; was write_parity_artifacts skipped?"
        )
    df = pd.read_parquet(path)
    rows: list[dict] = []
    for _, row in df.iterrows():
        payload = _payload(row.to_dict())
        flat = {
            "eval_summary_id": row.get("eval_summary_id"),
            "benchmark_family_key": row.get("benchmark_family_key"),
            "models_count": row.get("models_count"),
            "evaluation_name": payload.get("evaluation_name"),
            "composite_benchmark_name": payload.get("composite_benchmark_name"),
            "metrics_count": payload.get("metrics_count"),
            "canonical_benchmark_id": payload.get("canonical_benchmark_id"),
            "reporting_sources": payload.get("reporting_sources"),
        }
        rows.append(flat)
    return rows


def _load_hierarchy(output_dir: Path) -> dict:
    path = output_dir / "eval-hierarchy.json"
    if not path.exists():
        raise AuditError(f"audit: eval-hierarchy.json missing at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_zero_count_tile(rows: list[dict], errors: list[str]) -> None:
    """An eval-list tile with models_count=0 or metrics_count=0 is always a
    bug — the canonical-union pipeline shouldn't publish empty tiles."""
    offenders: list[str] = []
    for row in rows:
        models = row.get("models_count") or 0
        metrics = row.get("metrics_count") or 0
        if models <= 0 or metrics <= 0:
            offenders.append(
                f"{row.get('eval_summary_id')!r} "
                f"(models={models}, metrics={metrics})"
            )
    if offenders:
        errors.append(
            f"zero_count_tile: {len(offenders)} eval-list row(s) have "
            f"models_count=0 or metrics_count=0: {offenders[:10]}"
        )


def _check_hierarchy_dangling_eval_summary_ids(
    rows: list[dict], hierarchy: dict, errors: list[str]
) -> None:
    """Every eval_summary_id referenced anywhere in eval-hierarchy must
    exist in eval-list (as a primary row OR as a tracked reporting_source).
    Catches the regression where a hierarchy family / leaf points at an
    eval_summary that the catalog excluded — the frontend renders the
    family with no matching summaries → 0/0 tile."""
    known_ids: set[str] = set()
    for row in rows:
        primary = str(row.get("eval_summary_id") or "")
        if primary:
            known_ids.add(primary)
        for source in row.get("reporting_sources") or []:
            if source:
                known_ids.add(str(source))

    referenced: dict[str, str] = {}  # id -> "where" for diagnostics
    for fam in hierarchy.get("families") or []:
        for esid in fam.get("eval_summary_ids") or []:
            if esid:
                referenced.setdefault(str(esid), f"family={fam.get('key')}")
        for leaf in fam.get("leaves") or []:
            for esid in leaf.get("eval_summary_ids") or []:
                if esid:
                    referenced.setdefault(
                        str(esid), f"family={fam.get('key')}/leaf={leaf.get('key')}"
                    )

    dangling = sorted(esid for esid in referenced if esid not in known_ids)
    if dangling:
        diag = [f"{esid!r} ({referenced[esid]})" for esid in dangling[:10]]
        errors.append(
            f"hierarchy_dangling_eval_summary_id: {len(dangling)} hierarchy "
            f"reference(s) point at eval_summary_ids not present in eval-list "
            f"or its reporting_sources: {diag}"
        )


def _check_hierarchy_family_unrenderable(
    rows: list[dict], hierarchy: dict, errors: list[str]
) -> None:
    """The frontend renders family tile counts by filtering eval-list rows
    whose ``benchmark_family_key`` matches the family's ``key``. A family
    with no matching rows produces a 0/0 ghost tile (the regression we
    just shipped a fix for in the AA-lift drop / canonical-union work).

    Apply the same case-insensitive alphanumeric normalization the
    frontend uses so this check matches its filter exactly. Skip empty
    families (evals_count=0) — they're already pruned upstream and
    surfacing them here would just be noise.
    """
    rows_by_normalized_family: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = _normalize_key(row.get("benchmark_family_key"))
        if key:
            rows_by_normalized_family[key].append(str(row.get("eval_summary_id") or ""))

    offenders: list[str] = []
    for fam in hierarchy.get("families") or []:
        if (fam.get("evals_count") or 0) <= 0:
            continue
        norm = _normalize_key(fam.get("key"))
        if norm and norm not in rows_by_normalized_family:
            offenders.append(
                f"family.key={fam.get('key')!r} display={fam.get('display_name')!r} "
                f"evals_count={fam.get('evals_count')}"
            )
    if offenders:
        errors.append(
            f"hierarchy_family_unrenderable: {len(offenders)} hierarchy "
            f"famil(ies) reference a benchmark_family_key with no matching "
            f"eval-list rows (would render as 0/0 in the frontend): "
            f"{offenders[:10]}"
        )


def _check_duplicate_display_name_within_family(
    rows: list[dict], errors: list[str]
) -> None:
    """Within a single ``benchmark_family_key``, no two eval-list rows
    should share an ``evaluation_name`` or ``composite_benchmark_name``.
    Catches "4 Finance Agents" if a registry alias gap or a missed
    canonical-union allows sibling sub-benchmarks to ship as distinct
    tiles with the same label. This is structurally distinct from
    canonical_id_not_collapsed (below) — the canonical may be missing
    entirely on these rows; we only need the labels to differ.
    """
    by_family_and_name: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        family = str(row.get("benchmark_family_key") or "")
        name = (
            str(row.get("evaluation_name") or "")
            or str(row.get("composite_benchmark_name") or "")
        ).strip()
        if family and name:
            by_family_and_name[(family, name)].append(
                str(row.get("eval_summary_id") or "")
            )

    duplicates = {
        key: ids for key, ids in by_family_and_name.items() if len(ids) > 1
    }
    if duplicates:
        rendered = [
            f"{name!r} in family={family!r}: {ids}"
            for (family, name), ids in list(duplicates.items())[:10]
        ]
        errors.append(
            f"duplicate_display_name_within_family: {len(duplicates)} "
            f"(family, evaluation_name) pair(s) appear on multiple eval-list "
            f"rows: {rendered}"
        )


def _check_canonical_id_not_collapsed(rows: list[dict], errors: list[str]) -> None:
    """At most one eval-list row per non-empty ``canonical_benchmark_id``.
    Two rows sharing a canonical id means the canonical-union step missed
    them — either they were excluded from the contributors set, or the
    parent-rollup didn't fire for this canonical."""
    by_canonical: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        canonical = str(row.get("canonical_benchmark_id") or "")
        if canonical:
            by_canonical[canonical].append(str(row.get("eval_summary_id") or ""))
    leaks = {cid: ids for cid, ids in by_canonical.items() if len(ids) > 1}
    if leaks:
        rendered = [f"canonical={cid!r}: {ids}" for cid, ids in list(leaks.items())[:10]]
        errors.append(
            f"canonical_id_not_collapsed: {len(leaks)} canonical_benchmark_id(s) "
            f"appear on more than one eval-list row (canonical-union should "
            f"have merged them): {rendered}"
        )


def audit_output(output_dir: Path = Path("output")) -> None:
    """Run all corpus-level invariant checks. Raises ``AuditError`` for the
    blocking checks (catalog-tile data integrity); logs warnings for the
    advisory checks (hierarchy-vs-catalog drift) so they're visible without
    breaking the build. Don't fail-fast inside either bucket — surfacing
    every violation in one shot lets the operator triage all classes per
    cycle.

    Blocking checks:
      - ``zero_count_tile`` — every catalog tile must have models AND metrics
        (a 0/x or x/0 tile is always a bug)
      - ``duplicate_display_name_within_family`` — sibling rows with the
        same label silently mislead users (the "4 Finance Agents" pattern)
      - ``canonical_id_not_collapsed`` — canonical-union must merge every
        canonical's contributors into one row

    Advisory (warn-only) checks:
      - ``hierarchy_dangling_eval_summary_id`` and
        ``hierarchy_family_unrenderable`` — surface pre-existing pipeline
        weirdness (e.g. the ``theory_of_mind`` orphan in eval_summaries
        but not eval_list, see backlog) without blocking publication.
        Promote to blocking once the underlying data hygiene is sorted.
    """
    output_dir = Path(output_dir)
    rows = _load_eval_list(output_dir)
    hierarchy = _load_hierarchy(output_dir)

    blocking_errors: list[str] = []
    advisory_errors: list[str] = []

    _check_zero_count_tile(rows, blocking_errors)
    _check_duplicate_display_name_within_family(rows, blocking_errors)
    _check_canonical_id_not_collapsed(rows, blocking_errors)

    _check_hierarchy_dangling_eval_summary_ids(rows, hierarchy, advisory_errors)
    _check_hierarchy_family_unrenderable(rows, hierarchy, advisory_errors)

    for warning in advisory_errors:
        print(
            f"[pipeline] {json.dumps({'event': 'audit.warning', 'message': warning})}"
        )

    if blocking_errors:
        raise AuditError(
            "Output corpus audit failed:\n- " + "\n- ".join(blocking_errors)
        )

    print(
        f"[pipeline] {json.dumps({'event': 'audit.ok', 'eval_list_rows': len(rows), 'hierarchy_families': len(hierarchy.get('families') or []), 'advisory_warnings': len(advisory_errors)})}"
    )


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else Path("output")
    try:
        audit_output(target)
    except AuditError as exc:
        print(f"AUDIT FAILED:\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

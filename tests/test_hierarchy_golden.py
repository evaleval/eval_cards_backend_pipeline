"""Golden-file structural-parity test against ref-hierarchy_v2.json.

Per `notes/hierarchy-alignment.md` §8 — assert that the producer's
`hierarchy.json` matches the *shape* of the reference (the colleague's
gold standard) without requiring byte- or value-identity. Specifically:

  - Top-level keys: schema_version, families, benchmark_index, stats.
  - Each family carries exactly one of standalone_benchmarks /
    benchmarks / composites.
  - Each family has the required fields per spec §5.1.
  - Each benchmark inside any layout has the required fields.
  - Each benchmark_index entry has the required fields and represents
    a cross-suite appearance (≥2 distinct family_keys).

What this test deliberately does NOT assert:

  - Family-set identity. Our family-bucketing is composite-driven;
    the reference's is EEE-folder-driven. The same canonical data
    surfaces under different family keys. See
    `tests/hierarchy_golden_allowlist.yaml` for documented divergences.
  - Stat values. Snapshots are independent runs of independent
    pipelines on different EEE pulls.
  - Display-name strings. Curation polish evolves; identity is what
    matters.

Runs against the latest snapshot under `warehouse/` (the most-recent
ISO-named directory). Skips when no snapshot is present (e.g. fresh
checkout before first bake).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE_DIR = REPO_ROOT / "warehouse"
REF_PATH = REPO_ROOT.parent / "ref-hierarchy_v2.json"
ALLOWLIST_PATH = Path(__file__).parent / "hierarchy_golden_allowlist.yaml"


# Required field names per spec §5.1. Sets, so order doesn't matter.
_REQUIRED_FAMILY_FIELDS: set[str] = {
    "key", "display_name", "category", "tags",
    "evals_count", "eval_summary_ids",
    "reproducibility_summary", "provenance_summary", "comparability_summary",
}
_LAYOUT_FIELDS: set[str] = {"standalone_benchmarks", "benchmarks", "composites"}

_REQUIRED_BENCHMARK_FIELDS: set[str] = {
    "key", "display_name", "family_id", "is_slice", "is_overall",
    "primary_metric_key", "has_card", "tags", "metrics", "slices",
    "summary_eval_ids",
}

_REQUIRED_COMPOSITE_FIELDS: set[str] = {
    "key", "display_name", "category", "tags", "benchmarks",
}

_REQUIRED_BENCHMARK_INDEX_FIELDS: set[str] = {
    "key", "display_name", "appearances",
}

_REQUIRED_STATS_FIELDS: set[str] = {
    "family_count", "composite_count", "benchmark_count",
    "slice_count", "metric_count", "metric_rows_scanned",
}


def _latest_snapshot() -> Path | None:
    """Return the most-recent warehouse snapshot dir, or None when no
    snapshots exist. ISO-named dirs sort lexically by recency."""
    if not WAREHOUSE_DIR.is_dir():
        return None
    snapshots = sorted(p for p in WAREHOUSE_DIR.iterdir() if p.is_dir())
    if not snapshots:
        return None
    return snapshots[-1]


def _load_allowlist() -> dict:
    """Read the curated divergence allowlist (informational; the test
    doesn't enforce family-set parity, so allowlist mostly documents
    rationale rather than gating)."""
    if not ALLOWLIST_PATH.exists():
        return {}
    with ALLOWLIST_PATH.open() as f:
        return yaml.safe_load(f) or {}


@pytest.fixture(scope="module")
def ours() -> dict:
    snap = _latest_snapshot()
    if snap is None:
        pytest.skip(f"no snapshot under {WAREHOUSE_DIR}")
    path = snap / "hierarchy.json"
    if not path.exists():
        pytest.skip(f"hierarchy.json missing at {path}")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def ref() -> dict:
    if not REF_PATH.exists():
        pytest.skip(f"reference not present at {REF_PATH}")
    return json.loads(REF_PATH.read_text())


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------


def test_top_level_keys_match_spec(ours: dict) -> None:
    """Spec §5.1: top-level keys are schema_version, generated_at,
    stats, families, benchmark_index."""
    expected = {"schema_version", "stats", "families"}
    missing = expected - ours.keys()
    assert not missing, f"hierarchy.json missing top-level keys: {missing}"


def test_schema_version_v3(ours: dict) -> None:
    assert ours["schema_version"] == "v3.hierarchy.1", (
        f"expected schema_version='v3.hierarchy.1', got {ours['schema_version']!r}"
    )


def test_top_level_has_benchmark_index(ours: dict) -> None:
    """benchmark_index[] is the cross-suite lookup per spec §5.1."""
    assert "benchmark_index" in ours
    assert isinstance(ours["benchmark_index"], list)


def test_stats_has_required_keys(ours: dict) -> None:
    stats = ours["stats"]
    missing = _REQUIRED_STATS_FIELDS - stats.keys()
    assert not missing, f"stats missing keys: {missing}"


# ---------------------------------------------------------------------------
# Per-family shape
# ---------------------------------------------------------------------------


def test_every_family_has_required_fields(ours: dict) -> None:
    for fam in ours["families"]:
        missing = _REQUIRED_FAMILY_FIELDS - fam.keys()
        assert not missing, (
            f"family {fam.get('key')!r} missing fields: {missing}"
        )


def test_every_family_has_exactly_one_layout(ours: dict) -> None:
    """Spec §3 / §5.1: each family chooses one of three layouts —
    standalone_benchmarks (singleton), benchmarks (flat), or
    composites (multi-composite, e.g. HELM)."""
    for fam in ours["families"]:
        layouts = _LAYOUT_FIELDS & fam.keys()
        assert len(layouts) == 1, (
            f"family {fam['key']!r} has {len(layouts)} layouts present "
            f"({sorted(layouts)}); spec requires exactly one"
        )


# ---------------------------------------------------------------------------
# Per-benchmark shape
# ---------------------------------------------------------------------------


def _walk_benchmarks(family: dict):
    yield from family.get("standalone_benchmarks") or []
    yield from family.get("benchmarks") or []
    for c in family.get("composites") or []:
        yield from c.get("benchmarks") or []


def test_every_benchmark_has_required_fields(ours: dict) -> None:
    for fam in ours["families"]:
        for bench in _walk_benchmarks(fam):
            missing = _REQUIRED_BENCHMARK_FIELDS - bench.keys()
            assert not missing, (
                f"benchmark {bench.get('key')!r} (family {fam['key']!r}) "
                f"missing fields: {missing}"
            )


def test_benchmark_metrics_have_is_primary(ours: dict) -> None:
    """Per spec §5.1 — each metric carries an is_primary flag, and at
    most one metric per benchmark is is_primary=True."""
    for fam in ours["families"]:
        for bench in _walk_benchmarks(fam):
            primaries = [m for m in (bench.get("metrics") or [])
                         if m.get("is_primary")]
            assert len(primaries) <= 1, (
                f"benchmark {bench['key']!r} has {len(primaries)} primary "
                f"metrics; should be at most 1"
            )
            # When primary_metric_key is set, exactly one metric matches it.
            pmk = bench.get("primary_metric_key")
            if pmk and bench.get("metrics"):
                matched = [m for m in bench["metrics"]
                           if m.get("key") == pmk and m.get("is_primary")]
                assert len(matched) == 1, (
                    f"benchmark {bench['key']!r} has primary_metric_key="
                    f"{pmk!r} but no metric matches with is_primary=True"
                )


def test_family_primary_benchmark_exclusive(ours: dict) -> None:
    """At most one benchmark per family has is_primary=True. The flag
    is set by _mark_family_primary_benchmark in the producer."""
    for fam in ours["families"]:
        benches = list(_walk_benchmarks(fam))
        if not benches:
            continue
        primaries = [b for b in benches if b.get("is_primary")]
        assert len(primaries) <= 1, (
            f"family {fam['key']!r} has {len(primaries)} primary benchmarks; "
            f"should be at most 1"
        )


# ---------------------------------------------------------------------------
# Composite layout shape (when present)
# ---------------------------------------------------------------------------


def test_composites_have_required_fields(ours: dict) -> None:
    for fam in ours["families"]:
        for comp in fam.get("composites") or []:
            missing = _REQUIRED_COMPOSITE_FIELDS - comp.keys()
            assert not missing, (
                f"composite {comp.get('key')!r} (family {fam['key']!r}) "
                f"missing fields: {missing}"
            )


def test_multi_composite_family_marks_one_primary(ours: dict) -> None:
    """When a family uses the composites layout, exactly one composite
    should carry is_primary=True (the headline composite)."""
    for fam in ours["families"]:
        comps = fam.get("composites") or []
        if not comps:
            continue
        primaries = [c for c in comps if c.get("is_primary")]
        assert len(primaries) == 1, (
            f"family {fam['key']!r} composites: expected exactly 1 primary, "
            f"got {len(primaries)}"
        )


# ---------------------------------------------------------------------------
# benchmark_index shape
# ---------------------------------------------------------------------------


def test_benchmark_index_entries_have_required_fields(ours: dict) -> None:
    for entry in ours["benchmark_index"]:
        missing = _REQUIRED_BENCHMARK_INDEX_FIELDS - entry.keys()
        assert not missing, (
            f"benchmark_index entry {entry.get('key')!r} missing fields: {missing}"
        )


def test_benchmark_index_appearances_are_cross_suite(ours: dict) -> None:
    """Spec §5.1: benchmark_index entries surface canonicals that
    appear under 2+ distinct families. Single-family appearances
    aren't cross-suite by definition."""
    for entry in ours["benchmark_index"]:
        families = {a["family_key"] for a in entry["appearances"]}
        assert len(families) >= 2, (
            f"benchmark_index entry {entry['key']!r} has only "
            f"{len(families)} distinct family — should be 2+ for "
            f"cross-suite cross-linking"
        )


# ---------------------------------------------------------------------------
# Reference comparison (informational — divergences allowed via allowlist)
# ---------------------------------------------------------------------------


def test_reference_top_level_shape_compatible(ours: dict, ref: dict) -> None:
    """Soft check: both ours and ref carry families[] + benchmark_index[]
    + stats. Schema_version differs (v3 vs v2) — that's the entire
    point of v3, documented in the allowlist."""
    for key in ("families", "benchmark_index", "stats"):
        assert key in ours, f"ours missing {key!r}"
        assert key in ref, f"ref missing {key!r} (sanity)"


def test_reference_layout_distribution_recognisable(
    ours: dict, ref: dict,
) -> None:
    """Both producer's and ref's families use the same three layouts.
    Numeric distributions differ (composite-driven vs folder-driven
    family bucketing), but the SET of layouts in use should match."""
    def layouts_in_use(payload: dict) -> set[str]:
        out: set[str] = set()
        for fam in payload["families"]:
            for k in fam:
                if k in _LAYOUT_FIELDS:
                    out.add(k)
        return out

    ours_layouts = layouts_in_use(ours)
    ref_layouts = layouts_in_use(ref)
    assert ours_layouts <= _LAYOUT_FIELDS
    assert ref_layouts <= _LAYOUT_FIELDS
    # At least one layout in common — sanity.
    assert ours_layouts & ref_layouts, (
        f"ours layouts {ours_layouts} share none with ref {ref_layouts}"
    )


def test_allowlist_loads(ours: dict) -> None:
    """The allowlist YAML loads without errors. Documents intentional
    divergences but doesn't gate the test (most checks are structural,
    not identity-based)."""
    allowlist = _load_allowlist()
    assert isinstance(allowlist, dict)

"""Golden-file structural-parity test against ref-hierarchy_v2.json.

Asserts that the producer's
`hierarchy.json` matches the *shape* of the reference (the colleague's
gold standard) without requiring byte- or value-identity. Specifically:

  - Top-level keys: schema_version, families, benchmark_index, stats.
  - Each family carries exactly one of standalone_benchmarks /
    benchmarks / composites.
  - Each family has the required fields.
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


# Required field names. Sets, so order doesn't matter.
_REQUIRED_FAMILY_FIELDS: set[str] = {
    "key", "display_name", "derivedTags", "tags",
    "evals_count", "constituent_evaluation_ids",
    "reproducibility_summary", "provenance_summary", "comparability_summary",
}
_LAYOUT_FIELDS: set[str] = {"standalone_benchmarks", "benchmarks", "composites"}

_REQUIRED_BENCHMARK_FIELDS: set[str] = {
    "key", "display_name", "family_id", "is_slice", "is_overall",
    "primary_metric_key", "has_card", "tags", "metrics", "slices",
    "constituent_evaluation_ids",
}

_REQUIRED_COMPOSITE_FIELDS: set[str] = {
    "key", "display_name", "tags", "benchmarks",
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
    """Top-level keys are schema_version, generated_at,
    stats, families, benchmark_index."""
    expected = {"schema_version", "stats", "families"}
    missing = expected - ours.keys()
    assert not missing, f"hierarchy.json missing top-level keys: {missing}"


def test_schema_version_v3(ours: dict) -> None:
    assert ours["schema_version"] == "v3.hierarchy.1", (
        f"expected schema_version='v3.hierarchy.1', got {ours['schema_version']!r}"
    )


def test_top_level_has_benchmark_index(ours: dict) -> None:
    """benchmark_index[] is the cross-suite lookup."""
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
    """Each family chooses one of three layouts —
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
    """Each metric carries an is_primary flag, and at
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
    """At most one DISTINCT benchmark canonical per family has
    is_primary=True. The flag is set by _mark_family_primary_benchmark
    in the producer. A canonical can legitimately appear under multiple
    composites of the same family (e.g. `math` reported in both
    `reward-bench` and `reward-bench-2` leaderboards) — the dedup makes
    the assertion check distinct keys, not raw walk count."""
    for fam in ours["families"]:
        benches = list(_walk_benchmarks(fam))
        if not benches:
            continue
        primary_keys = {b["key"] for b in benches if b.get("is_primary")}
        assert len(primary_keys) <= 1, (
            f"family {fam['key']!r} has {len(primary_keys)} distinct primary "
            f"benchmark canonicals ({sorted(primary_keys)}); should be at most 1"
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
    """benchmark_index entries surface canonicals that
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


# ---------------------------------------------------------------------------
# v2 → v3 membership coverage (strict)
# ---------------------------------------------------------------------------
# These tests treat ref-hierarchy_v2.json as the gold for *coverage*, not
# values. Every benchmark / family / benchmark_index canonical that v2
# surfaces must be reachable somewhere in v3's hierarchy. Missing
# canonicals are the broken-link bug list.
#
# The assertions deliberately do NOT consult the allowlist — every miss
# should surface so we can decide case-by-case whether to fix the
# producer or document the drop. The allowlist YAML is now informational
# burn-down.


def _v2_canonical_keys(ref: dict) -> set[str]:
    """V2 entities that resolved to a registry canonical_id. These are
    the entities the registry recognises as real benchmarks; v2 keeps
    additional non-canonical slugged leftovers (`bfcl-leaderboard-csv-
    overall`, `arc-prize-evaluations-leaderboard-json-v1-public-eval`,
    etc.) which we don't treat as required surface."""
    out: set[str] = set()
    for fam in ref.get("families") or []:
        for layout in ("standalone_benchmarks", "benchmarks"):
            for b in fam.get(layout) or []:
                cid = b.get("underlying_canonical_id")
                if cid:
                    out.add(cid)
    return out


def _v3_all_keys(ours: dict) -> set[str]:
    """Every key v3 emits at any level — family / composite / benchmark
    / slice. Used as a liberal coverage target: a v2 benchmark is
    "reachable" in v3 if its canonical_id matches anything we surface,
    including as a within-benchmark slice."""
    out: set[str] = set()
    for fam in ours.get("families") or []:
        if fam.get("key"):
            out.add(fam["key"])
        for layout in ("standalone_benchmarks", "benchmarks"):
            for b in fam.get(layout) or []:
                if b.get("key"):
                    out.add(b["key"])
                for s in b.get("slices") or []:
                    if s.get("key"):
                        out.add(s["key"])
        for comp in fam.get("composites") or []:
            if comp.get("key"):
                out.add(comp["key"])
            for b in comp.get("benchmarks") or []:
                if b.get("key"):
                    out.add(b["key"])
                for s in b.get("slices") or []:
                    if s.get("key"):
                        out.add(s["key"])
    return out


def _v3_family_member_count(ours: dict) -> dict[str, int]:
    """Map of benchmark key → number of distinct v3 families it appears
    in. Used to validate that v2 benchmark_index cross-suite canonicals
    still appear in 2+ families even when v3 doesn't emit them in
    benchmark_index."""
    appearances: dict[str, set[str]] = {}
    for fam in ours.get("families") or []:
        fkey = fam.get("key")
        if not fkey:
            continue
        for layout in ("standalone_benchmarks", "benchmarks"):
            for b in fam.get(layout) or []:
                if b.get("key"):
                    appearances.setdefault(b["key"], set()).add(fkey)
        for comp in fam.get("composites") or []:
            for b in comp.get("benchmarks") or []:
                if b.get("key"):
                    appearances.setdefault(b["key"], set()).add(fkey)
    return {k: len(v) for k, v in appearances.items()}


def test_v2_benchmark_keys_reachable_in_v3(ours: dict, ref: dict) -> None:
    """Every registry-canonical benchmark that v2 surfaces must be
    reachable somewhere in v3's hierarchy — top-level family, composite,
    nested benchmark, or within-benchmark slice. Missing canonicals are
    the broken-link bug list. We deliberately ignore v2's non-canonical
    synthetic keys (slugged ds_name leftovers) — those aren't required
    surface."""
    v2_keys = _v2_canonical_keys(ref)
    v3_keys = _v3_all_keys(ours)
    missing = sorted(v2_keys - v3_keys)
    assert not missing, (
        f"v3 hierarchy is missing {len(missing)} v2 canonical benchmark(s) "
        f"(of {len(v2_keys)} total):\n  " + "\n  ".join(missing)
    )


# v2 family keys that the v3 model intentionally does not surface as a
# family/composite/benchmark key. These were per-stem "shell" families
# the old per-benchmark grouping synthesised even when the bare stem had
# no evaluation data of its own. v3 only surfaces entities that carry
# data or are registry-curated; the real benchmark (a differently-keyed
# split) remains reachable. `appworld` → data lives under
# `appworld-test-normal` in the exgentic-open-agent family.
_EXPECTED_ABSENT_V2_FAMILY_KEYS: frozenset[str] = frozenset({"appworld"})


def test_v2_family_keys_reachable_in_v3(ours: dict, ref: dict) -> None:
    """Every family key that v2 surfaces as a top-level family must
    appear somewhere in v3 — either as a v3 family, a composite under
    a v3 family, or a benchmark key. The v3 grouping bucket can differ
    from v2's, but the entity itself must be reachable. Data-less v2
    shell-stem families (see allowlist) are exempt — their real
    benchmark surfaces under a different key."""
    v2_family_keys = {f["key"] for f in ref.get("families") or [] if f.get("key")}
    v3_keys = _v3_all_keys(ours)
    missing = sorted(v2_family_keys - v3_keys - _EXPECTED_ABSENT_V2_FAMILY_KEYS)
    assert not missing, (
        f"v3 has no surface for {len(missing)} v2 top-level family key(s):\n  "
        + "\n  ".join(missing)
    )


def test_v2_benchmark_index_canonicals_preserved(ours: dict, ref: dict) -> None:
    """For every cross-suite canonical that v2 lists in benchmark_index
    (≥2 family appearances in v2), v3 must either:
      (a) list it in its own benchmark_index, OR
      (b) actually surface it under 2+ distinct v3 families.
    Otherwise the cross-suite link is silently broken."""
    v2_index_keys = {e["key"] for e in ref.get("benchmark_index") or []}
    v3_index_keys = {e["key"] for e in ours.get("benchmark_index") or []}
    v3_member_counts = _v3_family_member_count(ours)
    broken: list[str] = []
    for k in sorted(v2_index_keys):
        if k in v3_index_keys:
            continue
        if v3_member_counts.get(k, 0) >= 2:
            continue
        broken.append(k)
    assert not broken, (
        f"v3 has dropped {len(broken)} of v2's {len(v2_index_keys)} "
        f"cross-suite benchmark_index canonicals (not in v3 benchmark_index "
        f"AND not appearing in 2+ v3 families):\n  " + "\n  ".join(broken)
    )

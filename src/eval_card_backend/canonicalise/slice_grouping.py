"""Slice grouping for the hierarchy view.

A *slice* is a within-benchmark cut: GAIA Level 1/2/3, MMLU subjects,
CapArena vs-X comparisons. Slices live per-(composite, benchmark) — the
registry resolves only the canonical benchmark id; this module assigns
each canonical id to its slice parent when the relationship is
explicitly declared.

Two sources of parent_benchmark_id, both explicit:

  1. **Registry-authored** — `parent_benchmark_id` set directly in
     `benchmarks.yaml`. Covers BFCL sub-benchmarks, AIME yearly
     editions, ARC-AGI splits, RewardBench subscales, AIR-Bench
     categories, etc. These are never overridden.

  2. **Alias map** — `_ALIAS_MAP` below maps raw EEE names to their
     canonical stem when the resolver's alias table doesn't cover
     them. Groups benchmarks that share a stem and have ≥2 siblings.

No suffix-stripping heuristic is applied. Benchmarks like
`big-bench-hard`, `math-level-5`, or `livecodebench-pro` are
independent evaluations that happen to share a name prefix with
another benchmark; making them slices would be incorrect. If a
benchmark genuinely is a sub-score of another, the relationship
must be authored in `benchmarks.yaml`.
"""
from __future__ import annotations

import re
from collections import defaultdict


# Explicit aliases — for variants whose raw EEE names don't match the
# resolver's alias table. Glob suffix `*` matches any tail.
_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "gaia": (
        "hal_gaia",
        "hal_gaia_level_1",
        "hal_gaia_level_2",
        "hal_gaia_level_3",
    ),
    "hf-open-llm-v2": (
        "hfopenllm_v2_bbh",
        "hfopenllm_v2_gpqa",
        "hfopenllm_v2_ifeval",
        "hfopenllm_v2_math_level_5",
        "hfopenllm_v2_mmlu_pro",
        "hfopenllm_v2_musr",
    ),
    "helm-classic": ("helm_classic_*",),
    "helm-lite": ("helm_lite_*",),
    "mt-bench": ("mt_bench", "mtbench"),
    "videomme": ("videomme-w-sub", "videomme-w-o-sub"),
}


def normalize_stem(s: str) -> str:
    """Lowercase, replace `_`/whitespace with `-`, collapse repeats, trim."""
    s = s.lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _alias_lookup(benchmark_id: str) -> str | None:
    """Match against the alias map, supporting trailing `*` glob entries."""
    for stem, members in _ALIAS_MAP.items():
        for member in members:
            if member.endswith("*"):
                if benchmark_id.startswith(member[:-1]):
                    return stem
            elif member == benchmark_id:
                return stem
    return None


def compute_slice_stem(benchmark_id: str) -> str:
    """Return the canonical slice stem for a benchmark id.

    Alias-map lookups return the mapped stem; all other benchmarks
    return their own normalised id (i.e. they are their own stem and
    will not be grouped by the heuristic).
    """
    aliased = _alias_lookup(benchmark_id)
    if aliased is not None:
        return normalize_stem(aliased)
    return normalize_stem(benchmark_id)


def group_benchmarks(benchmark_ids: list[str]) -> dict[str, list[str]]:
    """Bucket ids by computed slice stem. Useful for tests."""
    out: dict[str, list[str]] = defaultdict(list)
    for bid in benchmark_ids:
        out[compute_slice_stem(bid)].append(bid)
    return dict(out)


def apply_slice_grouping(
    con,
    *,
    promote_to_benchmark: set[str] | None = None,
) -> int:
    """Mutate `canonical_benchmarks.parent_benchmark_id` in place.

    Only fills in parent edges the registry left NULL — the registry's
    hand-curated edges are never overridden. Grouping is driven by the
    alias map only (no suffix heuristic).

    Self-parents the bare-stem row when ≥1 sibling exists so it shows up
    as a benchmark inside the composite (GAIA's composite includes the
    bare `gaia` row as the suite's "Overall").

    Singleton stems are left alone (they stay standalone with
    `family.key == benchmark.key`).

    `promote_to_benchmark`: ids whose registry-authored
    `parent_benchmark_id` should be reset to NULL — the benchmark is
    a sibling, not a slice. Used for GPQA-Diamond and MMLU-Pro.

    Returns the number of rows whose parent was changed.
    """
    promote_to_benchmark = promote_to_benchmark or set()
    rows = con.execute(
        "SELECT id, parent_benchmark_id FROM canonical_benchmarks "
        "WHERE id IS NOT NULL"
    ).fetchall()
    parents: dict[str, str | None] = {bid: parent for bid, parent in rows}

    stem_members: dict[str, list[str]] = defaultdict(list)
    for bid in parents:
        stem_members[compute_slice_stem(bid)].append(bid)

    updates: list[tuple[str, str | None]] = []
    for stem, members in stem_members.items():
        if len(members) < 2:
            continue
        # Don't override the registry's existing edges.
        non_stem_registry_parents = {
            parents[m] for m in members
            if parents[m] is not None and parents[m] != stem
        }
        if non_stem_registry_parents:
            continue
        for member in members:
            if parents[member] is None:
                updates.append((member, stem))

    for bid in promote_to_benchmark:
        if parents.get(bid) is not None:
            updates.append((bid, None))

    if not updates:
        return 0

    con.execute("DROP TABLE IF EXISTS _slice_grouping_updates")
    con.execute(
        "CREATE TEMP TABLE _slice_grouping_updates "
        "(id VARCHAR, parent VARCHAR)"
    )
    con.executemany(
        "INSERT INTO _slice_grouping_updates VALUES (?, ?)", updates
    )
    con.execute(
        """
        UPDATE canonical_benchmarks AS cb
        SET parent_benchmark_id = u.parent
        FROM _slice_grouping_updates AS u
        WHERE cb.id = u.id
          AND cb.parent_benchmark_id IS DISTINCT FROM u.parent
        """
    )
    con.execute("DROP TABLE _slice_grouping_updates")
    return len(updates)

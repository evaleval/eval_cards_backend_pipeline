"""Slice grouping for the hierarchy view.

A *slice* is a within-benchmark cut: GAIA Level 1/2/3, MMLU subjects,
CapArena vs-X comparisons. Slices live per-(composite, benchmark) — the
registry resolves only the canonical benchmark id; this module assigns
each canonical id to its slice parent (or leaves it as a top-level
benchmark when no siblings share its stem).

The grouping pass mutates `canonical_benchmarks.parent_benchmark_id` in
place so downstream stages (Stage G dim materialisation, Stage I
evals_view, Stage J hierarchy.json) see slice edges with no further
changes.

Stem rule:
  1. Alias-map override (explicit cases the suffix rules can't model).
  2. Iterative suffix stripping (`-level-\\d+`, `-vN`,
     `-(diamond|lite|...)`, `-vs-...`, `-(auto-avg|caption-length)`,
     `-(zero-shot|few-shot|cot)`, `_Nshot`).
  3. Normalisation (lowercase, _/space → -, collapse repeats, strip).

Grouping rule:
  - A stem with ≥2 distinct benchmark ids sharing it forms a slice
    bucket. Every sibling's `parent_benchmark_id` is set to the stem
    (the bare-stem row, if it exists, becomes self-parented so it shows
    up alongside its variants in the composite).
  - A singleton stem leaves `parent_benchmark_id` untouched — the
    benchmark stays standalone with its own id as the family key.

Slice promotions:
  - Benchmarks listed in `slice_overrides.yaml::promote_to_benchmark`
    are *not* slices of their stem parent; their
    `parent_benchmark_id` is reset to NULL even when the heuristic
    would set it. Used for cases like `bfcl-live` /
    `bfcl-multi-turn` / `bfcl-non-live` / `bfcl-web-search` which
    are sibling *benchmarks* in the BFCL family rather than slices
    of a phantom `bfcl` stem.
"""
from __future__ import annotations

import re
from collections import defaultdict


_SUFFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-level-\d+$"),
    re.compile(r"-l\d+$"),
    re.compile(r"-v\d+(?:\.\d+)?$"),
    re.compile(r"-(?:diamond|lite|mini|hard|extra|easy|pro)$"),
    re.compile(r"-vs-[a-z0-9-]+$"),
    re.compile(r"-(?:auto-avg|caption-length)$"),
    re.compile(r"-(?:zero-shot|few-shot|cot)$"),
    re.compile(r"_\d+shot$"),
)


# Explicit aliases — for variants whose canonical ids don't share a common
# suffix that the rules above can strip. Glob suffix `*` matches any tail.
# Alias-map wins over suffix rules when both apply.
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
    # Variants that don't share a recognisable suffix.
    "videomme": ("videomme-w-sub", "videomme-w-o-sub"),
}


def normalize_stem(s: str) -> str:
    """Lowercase, replace `_`/whitespace with `-`, collapse repeats, trim."""
    s = s.lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _strip_suffixes(s: str) -> str:
    """Apply suffix rules iteratively until nothing matches."""
    while True:
        prev = s
        for pattern in _SUFFIX_PATTERNS:
            stripped = pattern.sub("", s)
            if stripped != s and stripped:
                s = stripped
                break
        if s == prev:
            return s


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

    Alias-map overrides win; otherwise iterative suffix stripping
    followed by normalisation. Always returns a non-empty string.
    """
    aliased = _alias_lookup(benchmark_id)
    if aliased is not None:
        return normalize_stem(aliased)
    return normalize_stem(_strip_suffixes(benchmark_id))


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
    hand-curated edges win over the suffix heuristic. So
    `rewardbench-chat-hard` (registry: parent=rewardbench) stays in the
    rewardbench composite, even though `compute_slice_stem` would strip
    `-hard` and place it in a `rewardbench-chat` family.

    Self-parents the bare-stem row when ≥1 sibling exists so it shows up
    as a benchmark inside the composite (GAIA's composite includes the
    bare `gaia` row as the suite's "Overall").

    Singleton stems are left alone (they stay standalone with
    `family.key == benchmark.key`).

    `promote_to_benchmark`: ids in this set are reset to
    `parent_benchmark_id = NULL` after the heuristic runs — they are
    sibling benchmarks in the family rather than slices of a phantom
    stem. Used for BFCL's four sub-benchmarks, MMLU-Pro vs. MMLU, and
    similar cases where the heuristic over-groups.

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

    # Promotion happens before grouping: remove promoted ids from each
    # sibling group up front so a stem with a single non-promoted member
    # stops being a "group" at all (no self-parenting on the bare stem
    # when its only siblings have been promoted to standalone benchmarks).
    updates: list[tuple[str, str | None]] = []
    for stem, members in stem_members.items():
        eligible = [m for m in members if m not in promote_to_benchmark]
        if len(eligible) < 2:
            continue
        # Don't override the registry's existing edges: if any eligible
        # member is already wired to a parent that isn't this stem, the
        # family is registry-authored — fill in NULLs only when the rest
        # agrees.
        non_stem_registry_parents = {
            parents[m] for m in eligible
            if parents[m] is not None and parents[m] != stem
        }
        if non_stem_registry_parents:
            continue
        for member in eligible:
            if parents[member] is None:
                updates.append((member, stem))

    # Reset any registry-set parents on promoted ids: a promoted id is
    # always parentless, even if the registry (or a prior heuristic
    # run) wired it to its stem.
    for bid in promote_to_benchmark:
        if parents.get(bid) is not None:
            updates.append((bid, None))

    if not updates:
        return 0

    # Stash updates in a temp table for one set-based UPDATE rather than N
    # round-trips. Faster and keeps the mutation deterministic.
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

"""Temporary hierarchy hot fixes.

Each function compensates for an upstream data issue that should be
fixed at the source (registry seed, resolver aliases, or EEE data).
Every function has a lifecycle annotation: what removes the need for it.

All functions mutate the families list in place.
"""

from __future__ import annotations

import re


def _walk_family_benchmarks(family: dict) -> list[dict]:
    out: list[dict] = []
    for layout in ("standalone_benchmarks", "benchmarks"):
        out.extend(family.get(layout) or [])
    for c in family.get("composites") or []:
        out.extend(c.get("benchmarks") or [])
    return out


def _family_benchmark_count(fam: dict) -> int:
    total = len(fam.get("benchmarks") or [])
    total += len(fam.get("standalone_benchmarks") or [])
    for c in fam.get("composites") or []:
        total += len(c.get("benchmarks") or [])
    return total


# ── 1. Consolidate AIR-Bench under HELM ──────────────────────────────
# Remove when: registry models AIR-Bench 2024 with proper
# family/composite structure and scoped aliases prevent
# cross-family contamination.


def _is_air_bench_eval_id(eid: str) -> bool:
    return bool(re.search(r"(?:^|%2F)air-bench-2024(?:[-%]|$)", eid, re.IGNORECASE))


def _is_air_bench_benchmark_key(key: str) -> bool:
    return bool(re.match(r"^air-bench-2024(?:[-_]|$)", key, re.IGNORECASE))


# Real eval slugs are lowercase, hyphen/dot/underscore separated. AIR-Bench's
# source data also carries many fine-grained subtask slices whose keys are raw
# display names ("airbench 2024 - #1.1: network intrusion") — those are not
# real evaluations (absent from the comparison-index) and must not be turned
# into constituent eval ids.
_SLICE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def consolidate_air_bench(families: list[dict]) -> None:
    """Consolidate all AIR-Bench appearances under the HELM family.

    AIR-Bench 2024 is a single safety benchmark with a multi-tier
    taxonomy. The warehouse surfaces it in:
      1. helm > helm-air-bench composite (canonical)
      2. agentharm family (misresolved rows: fraud, harassment)
      3. standalone air-bench-2024 family (the same rows)

    This function:
      - Collects all AIR-Bench eval ids across the hierarchy
      - Drops the standalone air-bench-2024 family
      - Strips AIR-Bench rows from non-HELM families
      - Plants all AIR-Bench eval ids under helm > helm-air-bench
    """
    air_bench_eval_ids: set[str] = set()

    for fam in families:
        for eid in fam.get("constituent_evaluation_ids") or []:
            if _is_air_bench_eval_id(eid):
                air_bench_eval_ids.add(eid)
        for composite in fam.get("composites") or []:
            for bench in composite.get("benchmarks") or []:
                if not _is_air_bench_benchmark_key(bench["key"]):
                    continue
                source_prefixes: set[str] = set()
                for eid in bench.get("constituent_evaluation_ids") or []:
                    if "%2F" in eid:
                        source_prefixes.add(eid.split("%2F")[0])
                if not source_prefixes:
                    source_prefixes.add(composite["key"])
                for sl in bench.get("slices") or []:
                    key = sl.get("key") or ""
                    if not _SLICE_SLUG_RE.match(key):
                        continue
                    for prefix in source_prefixes:
                        air_bench_eval_ids.add(f"{prefix}%2F{key}")

    # 1. Drop standalone air-bench-2024 family.
    families[:] = [f for f in families if f["key"] != "air-bench-2024"]

    # 2. Strip AIR-Bench from non-HELM families.
    for fam in families:
        if fam["key"] == "helm":
            continue
        if fam.get("constituent_evaluation_ids"):
            fam["constituent_evaluation_ids"] = [
                eid for eid in fam["constituent_evaluation_ids"]
                if not _is_air_bench_eval_id(eid)
            ]
        if fam.get("benchmarks"):
            fam["benchmarks"] = [
                b for b in fam["benchmarks"]
                if not _is_air_bench_benchmark_key(b["key"])
            ]
        if fam.get("standalone_benchmarks"):
            fam["standalone_benchmarks"] = [
                b for b in fam["standalone_benchmarks"]
                if not _is_air_bench_benchmark_key(b["key"])
            ]
        for c in fam.get("composites") or []:
            if c.get("benchmarks"):
                c["benchmarks"] = [
                    b for b in c["benchmarks"]
                    if not _is_air_bench_benchmark_key(b["key"])
                ]

    # 3. Plant all AIR-Bench eval ids under helm > helm-air-bench.
    helm = next((f for f in families if f["key"] == "helm"), None)
    if helm:
        helm_ids = set(helm.get("constituent_evaluation_ids") or [])
        helm_ids.update(air_bench_eval_ids)
        helm["constituent_evaluation_ids"] = sorted(helm_ids)

        composite = next(
            (c for c in (helm.get("composites") or []) if c["key"] == "helm-air-bench"),
            None,
        )
        if composite:
            bench = next(
                (b for b in (composite.get("benchmarks") or [])
                 if b["key"] == "air-bench-2024"),
                None,
            ) or (composite.get("benchmarks") or [None])[0]
            if bench:
                bench_ids = set(bench.get("constituent_evaluation_ids") or [])
                bench_ids.update(air_bench_eval_ids)
                bench["constituent_evaluation_ids"] = sorted(bench_ids)


# ── 2. Dedup "vals ai" aliases within vals-ai family ─────────────────
# Remove when: vals.ai aliases are fully covered in the registry
# alias store, so the resolver maps "vals ai X" → canonical X
# before the hierarchy is built.


def dedup_vals_ai_aliases(families: list[dict]) -> None:
    """Drop "vals ai X" benchmarks when a canonical sibling (X) already
    exists in the vals-ai family.

    The upstream feed publishes some benchmarks twice: once with a
    canonical key (e.g. "mgsm", "gpqa-overall") and once with a
    "vals ai <suffix>" alias. These are pure surface duplicates.

    Aliases without a non-aliased sibling are preserved. The
    vals_ai.swebench.<bucket> time-buckets use a different naming
    scheme (dot/underscore) and are untouched.
    """
    ALIAS_PREFIX = "vals ai "

    def _tokens(key: str) -> set[str]:
        return set(
            t for t in re.split(r"[^a-z0-9]+", key.lower()) if t
        )

    for fam in families:
        if fam["key"] != "vals-ai":
            continue

        benches = list(fam.get("benchmarks") or []) + list(
            fam.get("standalone_benchmarks") or []
        )
        sibling_tokens = [
            {"key": b["key"], "tokens": _tokens(b["key"])}
            for b in benches
            if not b["key"].startswith(ALIAS_PREFIX)
        ]

        def is_aliased_duplicate(b: dict) -> bool:
            if not b["key"].startswith(ALIAS_PREFIX):
                return False
            suffix_tokens = _tokens(b["key"][len(ALIAS_PREFIX):])
            if not suffix_tokens:
                return False
            return any(
                all(t in sib["tokens"] for t in suffix_tokens)
                for sib in sibling_tokens
            )

        if fam.get("benchmarks"):
            fam["benchmarks"] = [
                b for b in fam["benchmarks"] if not is_aliased_duplicate(b)
            ]
        if fam.get("standalone_benchmarks"):
            fam["standalone_benchmarks"] = [
                b for b in fam["standalone_benchmarks"]
                if not is_aliased_duplicate(b)
            ]
        for c in fam.get("composites") or []:
            if c.get("benchmarks"):
                c["benchmarks"] = [
                    b for b in c["benchmarks"] if not is_aliased_duplicate(b)
                ]


# ── 3. Group single-bench families sharing a benchmark key ───────────
# Remove when: a pipeline flag (`is_thin_family`) or frontend visual
# de-emphasis replaces this merge. The underlying model is correct
# (each family is an independent evaluation campaign); this is a
# presentation preference.


def group_same_bench_across_sources(
    families: list[dict],
    protected_keys: frozenset[str] = frozenset(),
) -> None:
    """Merge single-bench families that publish the same benchmark from
    different upstream sources into one merged family card.

    Triggered when >= 2 single-bench families share a bench key but
    their bench rows have non-overlapping constituent_evaluation_ids (independent
    sources). The richest family (most models) keeps its slot; others
    contribute their bench as siblings with display names suffixed by
    " · <Source>".
    """
    def _source_label(bench: dict, fallback: str) -> str:
        sources = []
        for m in bench.get("metrics") or []:
            sources.extend(m.get("sources") or [])
        for s in sources:
            trimmed = str(s or "").strip()
            if trimmed:
                return trimmed
        return fallback

    def _slugify_short(s: str) -> str:
        return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", s.lower()))

    def _collect_benches(fam: dict) -> list[dict]:
        return (
            list(fam.get("benchmarks") or [])
            + list(fam.get("standalone_benchmarks") or [])
            + [
                b
                for c in (fam.get("composites") or [])
                for b in (c.get("benchmarks") or [])
            ]
        )

    candidates_by_key: dict[str, list[tuple[dict, dict]]] = {}
    for fam in families:
        if fam["key"] in protected_keys:
            continue
        benches = _collect_benches(fam)
        if len(benches) != 1:
            continue
        sole = benches[0]
        candidates_by_key.setdefault(sole["key"], []).append((fam, sole))

    dropped: set[int] = set()

    for group in candidates_by_key.values():
        if len(group) < 2:
            continue

        # eval_ids must be disjoint.
        seen_ids: set[str] = set()
        disjoint = True
        for _, bench in group:
            for eid in bench.get("constituent_evaluation_ids") or []:
                if eid in seen_ids:
                    disjoint = False
                    break
                seen_ids.add(eid)
            if not disjoint:
                break
        if not disjoint:
            continue

        sorted_group = sorted(
            group,
            key=lambda x: (
                -(x[1].get("metrics") or [{}])[0].get("models_count", 0)
                if (x[1].get("metrics") or [])
                else 0,
                x[0]["key"],
            ),
        )
        survivor_fam, survivor_bench = sorted_group[0]
        base_display = (
            (survivor_bench.get("display_name") or "").strip()
            or (survivor_fam.get("display_name") or "").strip()
            or survivor_bench["key"]
        )

        # Promote standalone_benchmarks → benchmarks before merging so the
        # family doesn't end up with two layout keys.
        if survivor_fam.get("standalone_benchmarks") and not survivor_fam.get("benchmarks"):
            survivor_fam["benchmarks"] = survivor_fam.pop("standalone_benchmarks")

        for fam, bench in sorted_group:
            src = _source_label(bench, fam.get("display_name") or fam["key"])
            bench["display_name"] = f"{base_display} · {src}"
            if fam is not survivor_fam:
                slug = _slugify_short(src) or _slugify_short(fam["key"])
                bench["key"] = f"{survivor_bench['key']}__{slug}"
                bench["is_primary"] = False
                bench["is_overall"] = False
                if survivor_fam.get("benchmarks") is None:
                    survivor_fam["benchmarks"] = []
                survivor_fam["benchmarks"].append(bench)
                dropped.add(id(fam))
        survivor_fam["display_name"] = base_display

    if not dropped:
        return
    families[:] = [f for f in families if id(f) not in dropped]

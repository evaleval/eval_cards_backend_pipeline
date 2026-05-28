"""Systematic dedup for the hierarchy tree.

Two passes:

1. `consolidate_dedicated_home_benchmarks` — drop strictly-poorer
   duplicate appearances of the same benchmark across families when
   they share constituent_evaluation_ids (physical duplicates).

2. `dedup_aggregator_benchmarks` — drop aggregator-family benchmarks
   (llm-stats) when their scores literally match a non-aggregator
   family's benchmark within 1e-9 tolerance.

Both mutate the families list in place.
"""

from __future__ import annotations

import re
from typing import Any

PROTECTED_LEADERBOARD_FAMILIES: frozenset[str] = frozenset({"hf-open-llm-v2"})
AGGREGATOR_FAMILY_KEYS: frozenset[str] = frozenset({"llm-stats"})

# Families explicitly defined in canonical_families (registry-curated)
# are exempt from dedup removal. Set by write_hierarchy before dedup runs.
_curated_family_keys: frozenset[str] = frozenset()


def set_curated_families(keys: frozenset[str]) -> None:
    global _curated_family_keys
    _curated_family_keys = keys


def _walk_family_benchmarks(family: dict) -> list[dict]:
    out: list[dict] = []
    for layout in ("standalone_benchmarks", "benchmarks"):
        out.extend(family.get(layout) or [])
    for c in family.get("composites") or []:
        out.extend(c.get("benchmarks") or [])
    return out


def _richness(b: dict) -> int:
    slices = b.get("slices") or []
    metric_count = sum(len(s.get("metrics") or []) for s in slices)
    return len(slices) + metric_count


def _family_benchmark_count(fam: dict) -> int:
    total = len(fam.get("benchmarks") or [])
    total += len(fam.get("standalone_benchmarks") or [])
    for c in fam.get("composites") or []:
        total += len(c.get("benchmarks") or [])
    return total


def _slugify(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def _family_keys_related(a_key: str, b_key: str) -> bool:
    a_slug = _slugify(a_key)
    b_slug = _slugify(b_key)
    if not a_slug or not b_slug:
        return False
    return a_slug == b_slug or a_slug in b_slug or b_slug in a_slug


def _flat_key(s: str) -> str:
    return re.sub(r"[_\s-]+", "", (s or "").lower())


def _filter_benchmarks(fam: dict, drop_ids: set[int]) -> None:
    """Remove benchmarks whose id() is in drop_ids."""
    if fam.get("benchmarks"):
        fam["benchmarks"] = [b for b in fam["benchmarks"] if id(b) not in drop_ids]
    if fam.get("standalone_benchmarks"):
        fam["standalone_benchmarks"] = [
            b for b in fam["standalone_benchmarks"] if id(b) not in drop_ids
        ]
    for c in fam.get("composites") or []:
        if c.get("benchmarks"):
            c["benchmarks"] = [b for b in c["benchmarks"] if id(b) not in drop_ids]
    if fam.get("composites"):
        fam["composites"] = [
            c for c in fam["composites"] if (c.get("benchmarks") or [])
        ]


def _drop_empty_families(families: list[dict]) -> None:
    families[:] = [f for f in families if _family_benchmark_count(f) > 0]


# ── Score-based redundancy ───────────────────────────────────────────
# Two appearances of the same benchmark are redundant ONLY when their
# scores match — one is an echo of the other. Independent evaluations
# (different methodology, different scores) are NOT redundant and must
# both be kept, even when they carry the same canonical benchmark_id.

_SCORE_TOLERANCE = 1e-9
_MIN_SHARED_MODELS = 3


def _make_score_map_fn(con):
    """Return a cached `eval_id -> {model_key: score}` builder.

    Uses one metric per eval (first non-stderr metric by id order) so
    two appearances are compared on the same axis. Returns None for an
    eval with no usable scores. `con` exposes `eval_results_view`.
    """
    cache: dict[str, dict[str, float] | None] = {}

    def build(eval_id: str) -> dict[str, float] | None:
        if eval_id in cache:
            return cache[eval_id]
        try:
            rows = con.execute(
                """
                WITH usable AS (
                    SELECT metric_id, model_key, score
                    FROM eval_results_view
                    WHERE evaluation_id = ?
                      AND score IS NOT NULL
                      AND metric_id NOT LIKE '%%stderr%%'
                      AND metric_id NOT LIKE '%%std_err%%'
                      AND metric_id NOT LIKE '%%standard_error%%'
                ),
                first_metric AS (SELECT MIN(metric_id) AS metric_id FROM usable)
                SELECT u.model_key, u.score
                FROM usable u JOIN first_metric fm ON u.metric_id = fm.metric_id
                """,
                [eval_id],
            ).fetchall()
        except Exception:
            cache[eval_id] = None
            return None
        result = {r[0]: float(r[1]) for r in rows if r[0] and r[1] is not None}
        cache[eval_id] = result or None
        return cache[eval_id]

    return build


def _benches_redundant(build_score_map, bench_a: dict, bench_b: dict) -> bool:
    """True iff bench_a and bench_b are echoes of each other.

    - Sharing an evaluation_id literally means the same source row →
      redundant.
    - Otherwise, scores must match within tolerance across at least
      `_MIN_SHARED_MODELS` shared models for one pair of their evals.
    - When neither holds (scores diverge, or too few shared models to
      confirm), they are treated as INDEPENDENT and kept.
    """
    a_ids = bench_a.get("constituent_evaluation_ids") or []
    b_ids = bench_b.get("constituent_evaluation_ids") or []
    if not a_ids or not b_ids:
        return False
    b_id_set = set(b_ids)
    for a_id in a_ids:
        if a_id in b_id_set:
            return True  # same physical eval row
    for a_id in a_ids:
        a_map = build_score_map(a_id)
        if not a_map:
            continue
        for b_id in b_ids:
            b_map = build_score_map(b_id)
            if not b_map:
                continue
            shared = [
                (a_map[m], b_map[m]) for m in a_map if m in b_map
            ]
            if len(shared) < _MIN_SHARED_MODELS:
                continue
            if all(abs(x - y) <= _SCORE_TOLERANCE for x, y in shared):
                return True
    return False


# ── 1. Consolidate dedicated home benchmarks ────────────────────────


def consolidate_dedicated_home_benchmarks(families: list[dict], con) -> None:
    """Drop strictly-poorer *redundant* appearances of the same benchmark
    across families. A poorer copy is dropped only when a richer peer is
    an echo of it — scores match (`_benches_redundant`). Two
    appearances with the same benchmark_id but divergent scores are
    independent evaluations and both stay.

    Protected leaderboard families (hf-open-llm-v2) and registry-curated
    families are exempt. After individual benchmark drops, runs several
    wrapper-elimination passes to remove families that are strict
    subsets, self-wrappers, leaderboard wrappers, sole-bench ties, or
    pure eval-id aliases — each gated on score-redundancy.
    """
    build_score_map = _make_score_map_fn(con)

    instances_by_key: dict[str, list[dict]] = {}
    for fam in families:
        for b in _walk_family_benchmarks(fam):
            instances_by_key.setdefault(b["key"], []).append(b)

    def is_poorer_duplicate(b: dict) -> bool:
        peers = instances_by_key.get(b["key"], [])
        r = _richness(b)
        return any(
            peer is not b and _richness(peer) > r
            and _benches_redundant(build_score_map, peer, b)
            for peer in peers
        )

    protected = PROTECTED_LEADERBOARD_FAMILIES | _curated_family_keys

    drop_set: set = set()
    for fam in families:
        if fam["key"] in protected:
            continue
        for b in _walk_family_benchmarks(fam):
            if is_poorer_duplicate(b):
                drop_set.add(id(b))

    for fam in families:
        if fam["key"] in protected:
            continue
        _filter_benchmarks(fam, drop_set)

    _drop_empty_families(families)

    # --- Wrapper-elimination passes ---

    all_families = list(families)
    benches_by_fam: dict[int, list[tuple[dict, dict]]] = {}
    for fam in all_families:
        handles: list[tuple[dict, dict]] = []
        for b in _walk_family_benchmarks(fam):
            handles.append((fam, b))
        benches_by_fam[id(fam)] = handles

    dropped: set[int] = set()

    # (a) Strict-subset wrapper: family A's benchmark set ⊂ family B's,
    # gated by textually related family keys.
    for a in all_families:
        if id(a) in dropped or a["key"] in protected:
            continue
        a_handles = benches_by_fam[id(a)]
        if not a_handles:
            continue
        for b in all_families:
            if a is b or id(b) in dropped:
                continue
            if not _family_keys_related(a["key"], b["key"]):
                continue
            b_handles = benches_by_fam[id(b)]
            if len(b_handles) <= len(a_handles):
                continue
            b_by_key = {h[1]["key"]: h[1] for h in b_handles}
            is_subset = all(
                h[1]["key"] in b_by_key
                and _benches_redundant(build_score_map, b_by_key[h[1]["key"]], h[1])
                and _richness(b_by_key[h[1]["key"]]) >= _richness(h[1])
                for h in a_handles
            )
            if is_subset:
                dropped.add(id(a))
                break

    # (b) Self-wrapper tie: family.key == its sole benchmark.key AND
    # another family carries the same benchmark at >= richness.
    for a in all_families:
        if id(a) in dropped or a["key"] in protected:
            continue
        a_handles = benches_by_fam[id(a)]
        if len(a_handles) != 1:
            continue
        sole = a_handles[0][1]
        if sole["key"] != a["key"]:
            continue
        elsewhere = any(
            id(b) not in dropped
            and b is not a
            and any(
                h[1]["key"] == sole["key"]
                and _richness(h[1]) >= _richness(sole)
                and _benches_redundant(build_score_map, h[1], sole)
                for h in benches_by_fam[id(b)]
            )
            for b in all_families
        )
        if elsewhere:
            dropped.add(id(a))

    # (b2) Leaderboard-wrapper merge: *-leaderboard family with 1 bench
    # whose key matches family minus suffix → merge eval ids into target.
    for a in all_families:
        if id(a) in dropped or not a["key"].endswith("-leaderboard"):
            continue
        a_handles = benches_by_fam[id(a)]
        if len(a_handles) != 1:
            continue
        sole = a_handles[0][1]
        stripped = a["key"].removesuffix("-leaderboard")
        if sole["key"] != stripped:
            continue
        sole_rich = _richness(sole)
        candidates = [
            b
            for b in all_families
            if a is not b
            and id(b) not in dropped
            and any(
                h[1]["key"] == sole["key"] and _richness(h[1]) >= sole_rich
                for h in benches_by_fam[id(b)]
            )
        ]
        candidates.sort(key=lambda x: (1 if x["key"] in AGGREGATOR_FAMILY_KEYS else 0))
        if not candidates:
            continue
        target = candidates[0]
        peer_bench = next(
            (h[1] for h in benches_by_fam[id(target)] if h[1]["key"] == sole["key"]),
            None,
        )
        if not peer_bench:
            continue
        # Only fold the leaderboard family into the target when it's a
        # redundant echo (matching scores). An independent
        # leaderboard — e.g. swe-bench-verified-leaderboard reports
        # different scores than llm-stats's swe-bench-verified — is a
        # genuine cross-suite appearance and must stay separate.
        if not _benches_redundant(build_score_map, peer_bench, sole):
            continue
        merged = set(peer_bench.get("constituent_evaluation_ids") or [])
        merged.update(sole.get("constituent_evaluation_ids") or [])
        peer_bench["constituent_evaluation_ids"] = sorted(merged)
        dropped.add(id(a))

    # (c) Sole-bench / shared-eval-id tie. Two single-bench families
    # with identical constituent_evaluation_ids → drop one, preferring the family
    # whose key best matches the benchmark key.
    for a in all_families:
        if id(a) in dropped or a["key"] in protected:
            continue
        a_handles = benches_by_fam[id(a)]
        if len(a_handles) != 1:
            continue
        a_sole = a_handles[0][1]
        a_ids = set(a_sole.get("constituent_evaluation_ids") or [])
        if not a_ids:
            continue
        for b in all_families:
            if a is b or id(b) in dropped:
                continue
            b_handles = benches_by_fam[id(b)]
            if len(b_handles) != 1:
                continue
            b_sole = b_handles[0][1]
            b_ids = set(b_sole.get("constituent_evaluation_ids") or [])
            if b_ids != a_ids:
                continue
            a_match = _flat_key(a["key"]) == _flat_key(a_sole["key"])
            b_match = _flat_key(b["key"]) == _flat_key(b_sole["key"])
            if a_match and not b_match:
                loser_id = id(b)
            elif b_match and not a_match:
                loser_id = id(a)
            else:
                loser_id = id(b) if a["key"] < b["key"] else id(a)
            dropped.add(loser_id)
            if loser_id == id(a):
                break

    # (d) Drop single-bench families that are pure aliases — every
    # evaluation_id already covered by another surviving family, and
    # the bench key already has its own canonical family.
    ids_covered_by_family: dict[int, set[str]] = {}
    for fam in all_families:
        if id(fam) in dropped:
            continue
        ids = set()
        for _, bench in benches_by_fam[id(fam)]:
            ids.update(bench.get("constituent_evaluation_ids") or [])
        ids_covered_by_family[id(fam)] = ids

    alive_family_keys = {fam["key"] for fam in all_families if id(fam) not in dropped}

    for fam in all_families:
        if id(fam) in dropped or fam["key"] in protected:
            continue
        handles = benches_by_fam[id(fam)]
        if len(handles) != 1:
            continue
        sole_bench = handles[0][1]
        if sole_bench["key"] == fam["key"]:
            continue
        if sole_bench["key"] not in alive_family_keys:
            continue
        sole_ids = sole_bench.get("constituent_evaluation_ids") or []
        if not sole_ids:
            continue
        all_covered = all(
            any(
                eid in other_ids
                for other_fam_id, other_ids in ids_covered_by_family.items()
                if other_fam_id != id(fam) and other_fam_id not in dropped
            )
            for eid in sole_ids
        )
        if all_covered:
            dropped.add(id(fam))

    families[:] = [f for f in all_families if id(f) not in dropped]


# ── 2. Dedup aggregator benchmarks by score comparison ───────────────


def dedup_aggregator_benchmarks(
    families: list[dict], con: Any
) -> None:
    """Drop aggregator-family benchmarks when their scores match a
    non-aggregator benchmark for the same key, within 1e-9 tolerance
    across >= 3 shared models.

    Requires a DuckDB connection `con` with `eval_results_view` available.
    """
    _TOLERANCE = 1e-9
    _MIN_SHARED = 3

    all_handles: list[tuple[dict, dict, list[dict]]] = []
    for fam in families:
        if fam.get("benchmarks"):
            for b in fam["benchmarks"]:
                all_handles.append((fam, b, fam["benchmarks"]))
        if fam.get("standalone_benchmarks"):
            for b in fam["standalone_benchmarks"]:
                all_handles.append((fam, b, fam["standalone_benchmarks"]))
        for c in fam.get("composites") or []:
            if c.get("benchmarks"):
                for b in c["benchmarks"]:
                    all_handles.append((fam, b, c["benchmarks"]))

    by_key: dict[str, list[BenchHandle]] = {}
    for h in all_handles:
        by_key.setdefault(h[1]["key"], []).append(h)

    score_cache: dict[str, dict[str, float] | None] = {}

    def _build_score_map(eval_id: str) -> dict[str, float] | None:
        """Build model→score map for a single eval, using exactly one
        metric (the first non-stderr metric by id order). Matches the
        TS behaviour of picking `usableMetrics[0]`.
        """
        if eval_id in score_cache:
            return score_cache[eval_id]
        try:
            rows = con.execute(
                """
                WITH usable AS (
                    SELECT metric_id, model_key, score
                    FROM eval_results_view
                    WHERE evaluation_id = ?
                      AND score IS NOT NULL
                      AND metric_id NOT LIKE '%%stderr%%'
                      AND metric_id NOT LIKE '%%std_err%%'
                      AND metric_id NOT LIKE '%%standard_error%%'
                ),
                first_metric AS (
                    SELECT MIN(metric_id) AS metric_id FROM usable
                )
                SELECT u.model_key, u.score
                FROM usable u
                JOIN first_metric fm ON u.metric_id = fm.metric_id
                ORDER BY u.model_key
                """,
                [eval_id],
            ).fetchall()
        except Exception:
            score_cache[eval_id] = None
            return None
        if not rows:
            score_cache[eval_id] = None
            return None
        result = {r[0]: float(r[1]) for r in rows if r[0] and r[1] is not None}
        score_cache[eval_id] = result if result else None
        return score_cache[eval_id]

    drops: set[int] = set()

    for handles in by_key.values():
        if len(handles) < 2:
            continue
        agg_handles = [h for h in handles if h[0]["key"] in AGGREGATOR_FAMILY_KEYS]
        non_agg = [h for h in handles if h[0]["key"] not in AGGREGATOR_FAMILY_KEYS]
        if not agg_handles or not non_agg:
            continue

        for agg_fam, agg_bench, _ in agg_handles:
            agg_ids = agg_bench.get("constituent_evaluation_ids") or []
            if not agg_ids:
                continue
            matched = False
            for peer_fam, peer_bench, _ in non_agg:
                peer_ids = peer_bench.get("constituent_evaluation_ids") or []
                if not peer_ids:
                    continue
                for a_id in agg_ids:
                    a_map = _build_score_map(a_id)
                    if not a_map:
                        continue
                    for p_id in peer_ids:
                        p_map = _build_score_map(p_id)
                        if not p_map:
                            continue
                        shared = [
                            (a_map[model], p_map[model])
                            for model in a_map
                            if model in p_map
                        ]
                        if len(shared) < _MIN_SHARED:
                            continue
                        if all(abs(a - p) <= _TOLERANCE for a, p in shared):
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                drops.add(id(agg_bench))

    if not drops:
        return

    for fam in families:
        if fam.get("benchmarks"):
            fam["benchmarks"] = [b for b in fam["benchmarks"] if id(b) not in drops]
        if fam.get("standalone_benchmarks"):
            fam["standalone_benchmarks"] = [
                b for b in fam["standalone_benchmarks"] if id(b) not in drops
            ]
        for c in fam.get("composites") or []:
            if c.get("benchmarks"):
                c["benchmarks"] = [b for b in c["benchmarks"] if id(b) not in drops]
        if fam.get("composites"):
            fam["composites"] = [
                c for c in fam["composites"] if (c.get("benchmarks") or [])
            ]

    _drop_empty_families(families)

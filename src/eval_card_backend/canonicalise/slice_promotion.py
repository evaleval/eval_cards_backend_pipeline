"""Slice-promotion overrides for benchmark resolution.

The default Stage C resolver derives `benchmark_id` per row from
`clean_eval_name(evaluation_name)`. That over-collapses dot-notation
aggregator records ("llm_stats.aime-2024" → "llm stats" → mismatched
canonical) and drops 400+ benchmark canonicals that v2's reference
pipeline surfaces.

This module replays v2's `pass1` (bucket by dataset_name) plus
`maybe_promote_slices` (split a candidate when 2+ of its accumulated
slice names resolve to canonicals) in Python and writes a
`slice_promotion_overrides` temp table. Stage C consults this table
after creating `results_resolved` and overrides `benchmark_id` for
the rows v2 would have placed elsewhere.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Iterable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slice-promotion literals (suffix/family/composite maps).
# ---------------------------------------------------------------------------

SUFFIX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"-level-\d+$"), ""),
    (re.compile(r"-l\d+$"), ""),
    (re.compile(r"-v\d+(\.\d+)?$"), ""),
    (re.compile(r"-(diamond|lite|mini|hard|extra|easy|pro)$"), ""),
    (re.compile(r"-vs-[a-z0-9-]+$"), ""),
    (re.compile(r"-(auto-avg|caption-length)$"), ""),
    (re.compile(r"-(zero-shot|few-shot|cot)$"), ""),
    (re.compile(r"-\d+shot$"), ""),
    (re.compile(r"-(airline|retail|telecom|banking)$"), ""),
    (re.compile(r"-(multiple-choice|open-ended)$"), ""),
    (re.compile(r"^(fibble)\d+(?=-)"), r"\1"),
]

EXPLICIT_FAMILY_MAP: dict[str, str] = {
    "hal_gaia": "gaia", "hal_gaia_level_1": "gaia",
    "hal_gaia_level_2": "gaia", "hal_gaia_level_3": "gaia",
    "hfopenllm_v2": "hf-open-llm-v2",
    "helm_classic": "helm", "helm_lite": "helm", "helm_capabilities": "helm",
    "helm_instruct": "helm", "helm_mmlu": "helm",
    "mt_bench": "mt-bench", "mtbench": "mt-bench",
    "appworld_test_normal": "appworld",
    "tau-bench-2_airline": "tau-bench", "tau-bench-2_retail": "tau-bench",
    "tau-bench-2_telecom": "tau-bench", "tau-bench-2_banking": "tau-bench",
    "fibble_arena": "fibble-arena", "fibble1_arena": "fibble-arena",
    "fibble2_arena": "fibble-arena", "fibble3_arena": "fibble-arena",
    "fibble4_arena": "fibble-arena", "fibble5_arena": "fibble-arena",
    "wordle_arena": "wordle-arena",
    "MMMU-Multiple-Choice": "mmmu", "MMMU-Open-Ended": "mmmu",
    "MMLU-Pro": "mmlu-pro", "global-mmlu-lite": "global-mmlu-lite",
    "GAIA": "gaia", "IFEval": "ifeval", "MathVista": "mathvista",
    "swe-bench": "swe-bench", "terminal-bench-2.0": "terminal-bench-2",
    "livecodebenchpro": "livecodebench-pro",
    "reward-bench": "reward-bench", "reward-bench-2": "reward-bench",
    "rewardbench": "reward-bench", "rewardbench-2": "reward-bench",
    "arc-agi": "arc-agi", "apex-agents": "apex-agents",
    "apex-v1": "apex-v1", "agentharm": "agentharm", "ace": "ace",
    "bfcl": "bfcl", "browsecompplus": "browsecompplus",
    "la_leaderboard": "la-leaderboard", "sciarena": "sciarena",
    "theory_of_mind": "theory-of-mind",
    "artificial-analysis-llms": "artificial-analysis",
    "big_bench_hard": "bbh", "caparena-auto": "caparena",
    "cocoabench": "cocoabench", "commonsense_qa": "commonsense-qa",
    "cvebench": "cvebench", "cybench": "cybench",
    "cyse2_interpreter_abuse": "cyse2", "cyse2_prompt_injection": "cyse2",
    "cyse2_vulnerability_exploit": "cyse2",
    "facts-grounding": "facts-grounding",
    "gpqa-diamond": "gpqa", "gpqa_diamond": "gpqa",
    "gdm_intercode_ctf": "gdm-intercode-ctf",
    "helm_air_bench": "helm", "helm_safety": "helm",
    "hal-gaia": "hal", "hal-assistantbench": "hal",
    "hal-corebench-hard": "hal", "hal-online-mind2web": "hal",
    "hal-scicode": "hal", "hal-scienceagentbench": "hal",
    "hal-swebench-verified-mini": "hal", "hal-taubench-airline": "hal",
    "hal-usaco": "hal",
}

EXPLICIT_COMPOSITE_MAP: dict[str, str] = {
    "helm_classic": "helm-classic", "helm_lite": "helm-lite",
    "helm_capabilities": "helm-capabilities", "helm_instruct": "helm-instruct",
    "helm_mmlu": "helm-mmlu", "helm_air_bench": "helm-air-bench",
    "helm_safety": "helm-safety",
}

_GENERIC_WORDS = {
    "score", "overall", "main", "all", "total", "sum", "mean", "avg",
    "best", "first", "last", "rank", "default",
}

EXCLUDED_FOLDERS = {"alphaxiv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slugify(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"[^a-z0-9-/.]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def family_stem(
    folder: str,
    canonical_set: set[str],
    alias_to_canonical: dict[str, str],
    parent: dict[str, str],
    scoped_alias_to_canonical: dict[tuple[str, str], str] | None = None,
) -> str:
    if folder in EXPLICIT_FAMILY_MAP:
        return EXPLICIT_FAMILY_MAP[folder]
    # The folder IS the source_config, so resolve it config-scoped-first.
    cid = resolve_canonical(
        folder,
        canonical_set,
        alias_to_canonical,
        scoped_alias_to_canonical=scoped_alias_to_canonical,
        source_config=folder,
    )
    if cid:
        cur = cid
        seen: set[str] = set()
        while cur in parent and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
        return slugify(cur)
    stem = slugify(folder)
    while True:
        prev = stem
        for pat, rep in SUFFIX_PATTERNS:
            stem = pat.sub(rep, stem)
        if stem == prev:
            return stem


def resolve_canonical(
    raw: str | None,
    canonical_set: set[str],
    alias_to_canonical: dict[str, str],
    scoped_alias_to_canonical: dict[tuple[str, str], str] | None = None,
    source_config: str | None = None,
) -> str | None:
    if not raw:
        return None
    keys = (raw, raw.lower(), slugify(raw))
    # Config-scoped first (matches AliasStore.lookup precedence).
    if source_config and scoped_alias_to_canonical:
        for k in keys:
            scoped = scoped_alias_to_canonical.get((source_config, k))
            if scoped is not None:
                return scoped
    for k in keys:
        if k in alias_to_canonical:
            return alias_to_canonical[k]
    if raw in canonical_set:
        return raw
    if slugify(raw) in canonical_set:
        return slugify(raw)
    return None


def resolve_canonical_strict(
    raw: str | None,
    canonical_set: set[str],
    alias_to_canonical: dict[str, str] | None = None,
    scoped_alias_to_canonical: dict[tuple[str, str], str] | None = None,
    source_config: str | None = None,
) -> str | None:
    """Strict resolution for slice→benchmark promotion.

    Accepts a slice name as a canonical reference when it either:
      (1) slug-equals an existing canonical_id, OR
      (2) hits an alias (e.g. registry alias `LCR` → `aa-lcr`,
          `artificial_analysis.lcr` → `aa-lcr`). The alias path is gated
          by the generic-words blacklist so suite-scoped tokens
          ("Overall", "Score") don't accidentally promote.

    Alias resolution is CONFIG-AWARE, mirroring the resolver's own
    `AliasStore.lookup` precedence (config-scoped before global): when a
    `source_config` is given and `scoped_alias_to_canonical` carries a
    scoped alias for that config, the scoped canonical wins; otherwise we
    fall back to the global/unscoped `alias_to_canonical`. This prevents
    a config-blind global map from picking the wrong canonical for a
    surface form scoped to different canonicals in different EEE folders
    (e.g. "Investment Banking" → apex-agents under the apex-agents folder
    but apex-v1 under the apex-v1 folder).
    """
    if not raw:
        return None
    s = slugify(raw)
    if not s or s in _GENERIC_WORDS:
        return None
    if s in canonical_set:
        return s
    keys = (raw, raw.lower(), s)
    # Config-scoped first (matches AliasStore.lookup precedence).
    if source_config and scoped_alias_to_canonical:
        for k in keys:
            scoped = scoped_alias_to_canonical.get((source_config, k))
            if scoped is not None:
                return scoped
    # Global / unscoped fallback.
    if alias_to_canonical:
        for k in keys:
            if k in alias_to_canonical:
                return alias_to_canonical[k]
    return None


def extract_slice_candidate(family: str, ds_name: str, eval_name: str) -> str | None:
    """Pull a slice-name candidate out of `evaluation_name`. Returns the
    slice name, or None when the eval_name reduces to the dataset/family
    root.
    """
    if not eval_name:
        return None
    ev = eval_name

    # Strip "{ds_name} - X" / "{family} - X" prefixes
    if " - " in ev:
        head, rest = ev.split(" - ", 1)
        if slugify(head) in (family, slugify(ds_name)):
            ev = rest.strip()

    if "." in ev:
        parts = ev.split(".")
        if len(parts) >= 2 and slugify(parts[0]) == family:
            parts = parts[1:]
        if len(parts) >= 2:
            return parts[0]
        if len(parts) == 1:
            return parts[0]
        return None

    # No dots: eval_name is the candidate unless it equals the root
    root_keys = {slugify(ds_name), family}
    if slugify(ev) in root_keys or not ev:
        return None
    return ev


# ---------------------------------------------------------------------------
# Override computation
# ---------------------------------------------------------------------------


def _load_registry_maps(
    con,
) -> tuple[set[str], dict[str, str], dict[tuple[str, str], str], dict[str, str]]:
    """Build {canonical_set, alias_to_canonical, scoped_alias_to_canonical,
    parent} from the canonical_benchmarks + aliases tables already loaded
    on the connection.

    `alias_to_canonical` holds GLOBAL (unscoped, `source_config IS NULL`)
    benchmark aliases. `scoped_alias_to_canonical` holds config-scoped
    aliases keyed by `(source_config, alias_key)`, where `alias_key` is one
    of the raw/lower/slug spellings — mirroring how the resolver scopes
    aliases per EEE folder. Slice-promotion resolution prefers a scoped
    entry for the bucket's folder before falling back to the global map.
    """
    canonical_set: set[str] = set()
    parent: dict[str, str] = {}
    for row in con.execute(
        "SELECT id, parent_benchmark_id FROM canonical_benchmarks"
    ).fetchall():
        cid, pid = row
        if cid:
            canonical_set.add(cid)
            if pid:
                parent[cid] = pid

    alias_to_canonical: dict[str, str] = {}
    scoped_alias_to_canonical: dict[tuple[str, str], str] = {}
    rows = con.execute(
        "SELECT raw_value, canonical_id, source_config FROM aliases "
        "WHERE entity_type='benchmark' AND status IN ('confirmed','auto')"
    ).fetchall()
    for raw, cid, source_config in rows:
        if not raw or not cid:
            continue
        keys = (raw, raw.lower(), slugify(raw))
        if source_config:
            for k in keys:
                scoped_alias_to_canonical[(source_config, k)] = cid
        else:
            for k in keys:
                alias_to_canonical[k] = cid
    return canonical_set, alias_to_canonical, scoped_alias_to_canonical, parent


def _iter_exploded_rows(con) -> Iterable[tuple[str, int, str, str, str, str]]:
    """Yield (evaluation_id, result_idx, source_config, ds_name, eval_name,
    evaluation_result_id) for every row in `results_exploded`.

    The `ORDER BY` is load-bearing for determinism: `compute_overrides`
    consumes this stream with first-seen-wins accumulation (`setdefault`
    for the bucket's `ds_canonical`), so an unordered scan — whose row
    order varies run-to-run under DuckDB's multi-threaded execution —
    would let the bucket's resolved canonical flip between runs. A total
    order on (evaluation_id, result_idx, evaluation_result_id) pins it.
    `evaluation_result_id` is the per-physical-record disambiguator:
    distinct EEE source records can collide on (evaluation_id,
    result_idx) but never on evaluation_result_id.
    """
    # `source_data` materialises as VARCHAR (JSON-serialised) on
    # `results_exploded`, not a STRUCT, so dataset_name is extracted via
    # json_extract_string. Falls back to source_config when dataset_name
    # is absent / null.
    rows = con.execute(
        """
        SELECT
            evaluation_id,
            result_idx,
            source_config,
            COALESCE(
                json_extract_string(source_data, '$.dataset_name'),
                source_config
            ) AS ds_name,
            evaluation_name,
            evaluation_result_id
        FROM results_exploded
        ORDER BY evaluation_id, result_idx, evaluation_result_id
        """
    ).fetchall()
    for r in rows:
        yield r


def compute_overrides(con) -> int:
    """Replay v2's pass1+pass2 logic over the rows in `results_exploded`,
    write `slice_promotion_overrides(evaluation_result_id, canonical_id)`
    into the connection. Returns row count.

    The override is non-None only for rows where v2's logic produced an
    answer that v3's row-level resolver would not (the "promoted slice"
    case). All other rows are no-ops at apply time.

    Overrides are keyed by `evaluation_result_id`, NOT (evaluation_id,
    result_idx): distinct EEE source records can collide on the latter
    (e.g. two LiveBench records that share one evaluation_id, each with
    its own evaluation_results[] array, land on the same result_idx).
    Keying by (evaluation_id, result_idx) would emit two override rows
    for the same key with different canonicals and let the apply-time
    UPDATE pick whichever the scan reached last — a run-to-run flip.
    `evaluation_result_id` is unique per physical row, so each record
    keeps its own correct slice canonical and the UPDATE join is 1:1.
    """
    canonical_set, alias_to_canonical, scoped_alias_to_canonical, parent = (
        _load_registry_maps(con)
    )
    if not canonical_set:
        log.warning(
            "slice_promotion: canonical_benchmarks is empty; "
            "skipping override computation"
        )
        con.execute(
            "CREATE OR REPLACE TEMP TABLE slice_promotion_overrides "
            "(evaluation_result_id VARCHAR, canonical_id VARCHAR)"
        )
        return 0

    # Pass 1: bucket rows by (folder, family, composite, ds_slug),
    #         accumulate distinct slice candidate names per bucket. The
    #         bucket also records its candidate canonical (resolved from
    #         dataset_name) for the non-promoted path.
    candidates: dict[tuple, dict] = {}
    # (evaluation_result_id, bucket_key, slice_candidate)
    row_records: list[tuple] = []
    for (
        evaluation_id, result_idx, folder, ds_name, eval_name, evaluation_result_id
    ) in _iter_exploded_rows(con):
        if folder in EXCLUDED_FOLDERS:
            continue
        family = family_stem(
            folder, canonical_set, alias_to_canonical, parent,
            scoped_alias_to_canonical=scoped_alias_to_canonical,
        )
        composite = EXPLICIT_COMPOSITE_MAP.get(folder)
        ds_slug = slugify(ds_name) or slugify(folder)
        bucket_key = (folder, family, composite, ds_slug)

        cand = candidates.setdefault(bucket_key, {
            "ds_canonical": resolve_canonical(
                ds_name, canonical_set, alias_to_canonical,
                scoped_alias_to_canonical=scoped_alias_to_canonical,
                source_config=folder,
            ),
            "slice_names":      set(),
            "slice_canonicals": set(),
        })
        slice_candidate = extract_slice_candidate(family, ds_name, eval_name or "")
        if slice_candidate:
            cand["slice_names"].add(slice_candidate)
            sc = resolve_canonical_strict(
                slice_candidate, canonical_set, alias_to_canonical,
                scoped_alias_to_canonical=scoped_alias_to_canonical,
                source_config=folder,
            )
            if sc:
                cand["slice_canonicals"].add(sc)

        row_records.append((evaluation_result_id, bucket_key, slice_candidate))

    # Pass 2: per bucket, decide promotion (≥2 distinct slice canonicals).
    promoted_buckets: set[tuple] = {
        bk for bk, cand in candidates.items() if len(cand["slice_canonicals"]) >= 2
    }

    # Pass 3: per row, emit an override.
    #   (a) Promoted bucket + this row's slice resolves → use the slice canonical
    #       (the artificial-analysis / vals-ai / openeval case).
    #   (b) Otherwise, if the bucket's dataset_name resolves to a canonical
    #       → use that canonical (the llm-stats / per-record-distinct-ds_name
    #       case where v3's clean_eval_name path mishandles the source-prefixed
    #       evaluation_name).
    overrides: list[tuple[str, str]] = []
    for evaluation_result_id, bucket_key, slice_candidate in row_records:
        cand = candidates[bucket_key]
        folder = bucket_key[0]
        override: str | None = None
        if bucket_key in promoted_buckets and slice_candidate:
            # Resolve the slice candidate config-scoped-first so a surface
            # form scoped to different canonicals across EEE folders picks
            # this bucket's folder canonical, not whichever the (former
            # config-blind) global map happened to hold last.
            override = resolve_canonical_strict(
                slice_candidate, canonical_set, alias_to_canonical,
                scoped_alias_to_canonical=scoped_alias_to_canonical,
                source_config=folder,
            )
        if override is None and cand["ds_canonical"]:
            override = cand["ds_canonical"]
        if override and evaluation_result_id is not None:
            overrides.append((evaluation_result_id, override))

    con.execute(
        "CREATE OR REPLACE TEMP TABLE slice_promotion_overrides "
        "(evaluation_result_id VARCHAR, canonical_id VARCHAR)"
    )
    if overrides:
        con.executemany(
            "INSERT INTO slice_promotion_overrides VALUES (?, ?)",
            overrides,
        )
    log.info(
        "slice_promotion: %d candidate buckets, %d promoted, %d row overrides",
        len(candidates), len(promoted_buckets), len(overrides),
    )
    return len(overrides)


def apply_overrides(con) -> int:
    """Compute and apply slice-promotion overrides to `results_resolved`.

    Must be called AFTER `results_resolved` has been created. Updates
    `benchmark_id` in place for matched rows. Returns the number of
    rows actually updated.
    """
    n_overrides = compute_overrides(con)
    if n_overrides == 0:
        return 0

    # Stamp provenance: resolve_strategy logged 'no_match' against the full
    # source-prefixed raw before this override resolved the real sub-benchmark.
    # Marking the rows 'slice_promotion' makes them auditable — query this
    # strategy to find every row relying on the vendored heuristic rather than
    # a registry alias, and decide which to promote into the registry.
    n_updated = con.execute(
        """
        UPDATE results_resolved AS rr
        SET benchmark_id = ovr.canonical_id,
            benchmark_resolution_strategy = 'slice_promotion'
        FROM slice_promotion_overrides AS ovr
        WHERE rr.evaluation_result_id = ovr.evaluation_result_id
          AND ovr.canonical_id IS NOT NULL
          AND rr.benchmark_id IS DISTINCT FROM ovr.canonical_id
        RETURNING 1
        """
    ).fetchall()
    log.info(
        "slice_promotion: updated benchmark_id (+stamped strategy) on %d rows",
        len(n_updated),
    )
    return len(n_updated)

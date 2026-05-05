"""Simulate v2's pass1 (dataset_name bucketing) + pass2 (slice promotion)
in-memory against the cached EEE data, to estimate how many canonicals
would be recovered if v3 ported maybe_promote_slices.

Reads:
  - EEE records from DEPRECATED_eval_card_backend/.cache/eee_datastore/
  - Registry aliases from DEPRECATED_eval_card_backend/.cache/entity_registry/
  - ref-hierarchy_v2.json (gold reference)
  - warehouse/<latest>/hierarchy.json (current v3 output, via symlink)

Reports:
  - simulation set of canonical ids
  - v2 ref set
  - v3 current set
  - three-way Venn

Run: uv run python scripts/simulate_v2_pipeline.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb


REPO = Path(__file__).resolve().parents[1]
EEE_DIR = Path("/Users/jchim/projects/evaleval/DEPRECATED_eval_card_backend/.cache/eee_datastore/data")
REG_DIR = Path("/Users/jchim/projects/evaleval/DEPRECATED_eval_card_backend/.cache/entity_registry")
REF_PATH = REPO.parent / "ref-hierarchy_v2.json"
V3_PATH = REPO / "warehouse" / "2026-05-05T00-42-42Z" / "hierarchy.json"

EXCLUDED_FOLDERS = {"alphaxiv"}

# Vendored verbatim from ref-build_hierarchy.py:39-149
SUFFIX_PATTERNS: list[tuple[re.Pattern, str]] = [
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


def slugify(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"[^a-z0-9-/.]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def family_stem(folder: str, canonical_set: set[str], alias_to_canonical: dict[str, str], parent: dict[str, str]) -> str:
    if folder in EXPLICIT_FAMILY_MAP:
        return EXPLICIT_FAMILY_MAP[folder]
    # registry lookup → walk parent chain
    cid = resolve_canonical(folder, canonical_set, alias_to_canonical)
    if cid:
        cur = cid
        seen: set[str] = set()
        while cur in parent and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
        return slugify(cur)
    # suffix-stripping fallback
    stem = slugify(folder)
    while True:
        prev = stem
        for pat, rep in SUFFIX_PATTERNS:
            stem = pat.sub(rep, stem)
        if stem == prev:
            break
    return stem


def resolve_canonical(raw: str | None, canonical_set: set[str], alias_to_canonical: dict[str, str]) -> str | None:
    if not raw:
        return None
    for k in (raw, raw.lower(), slugify(raw)):
        if k in alias_to_canonical:
            return alias_to_canonical[k]
    if raw in canonical_set:
        return raw
    if slugify(raw) in canonical_set:
        return slugify(raw)
    return None


def resolve_canonical_strict(raw: str | None, canonical_set: set[str]) -> str | None:
    if not raw:
        return None
    s = slugify(raw)
    if not s or s in _GENERIC_WORDS:
        return None
    if s in canonical_set:
        return s
    return None


def load_registry() -> tuple[set[str], dict[str, str], dict[str, str]]:
    con = duckdb.connect()
    cb = con.execute(
        f"SELECT id, parent_benchmark_id FROM '{REG_DIR}/canonical_benchmarks/part-0.parquet'"
    ).fetchall()
    canonical_set: set[str] = {r[0] for r in cb}
    parent: dict[str, str] = {r[0]: r[1] for r in cb if r[1]}

    al = con.execute(
        f"SELECT raw_value, canonical_id FROM '{REG_DIR}/aliases/part-0.parquet' "
        f"WHERE entity_type='benchmark' AND status IN ('confirmed','auto')"
    ).fetchall()
    alias_to_canonical: dict[str, str] = {}
    for raw, cid in al:
        alias_to_canonical[raw] = cid
        alias_to_canonical[raw.lower()] = cid
        alias_to_canonical[slugify(raw)] = cid
    return canonical_set, alias_to_canonical, parent


def iter_eee_records():
    for folder in sorted(EEE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in EXCLUDED_FOLDERS:
            continue
        for f in folder.rglob("*.json"):
            try:
                yield folder.name, json.loads(f.read_text())
            except Exception:
                continue


def simulate(canonical_set, alias_to_canonical, parent):
    """Replay v2's pass1 + pass2.

    Pass 1: bucket records by (folder, family, composite, bench_slug) with
    bench_slug derived from dataset_name, accumulating slice names from
    evaluation_name dot-notation parsing.

    Pass 2: for each candidate, if 2+ accumulated slice names resolve to
    canonicals via resolve_canonical_strict, promote each resolvable slice
    to its own benchmark. The output set of canonical ids is what the
    final hierarchy would surface.
    """
    # candidates[(folder, family, composite, bench_slug)] = {
    #     "canonical_id": cid_or_None,
    #     "slice_names": set[str],
    # }
    candidates: dict[tuple, dict] = {}

    for folder, rec in iter_eee_records():
        family = family_stem(folder, canonical_set, alias_to_canonical, parent)
        composite = EXPLICIT_COMPOSITE_MAP.get(folder)

        for er in rec.get("evaluation_results", []) or []:
            sd = er.get("source_data") or {}
            ds_name = sd.get("dataset_name") or folder
            eval_name = er.get("evaluation_name") or ""

            cid = resolve_canonical(ds_name, canonical_set, alias_to_canonical)
            bench_slug = cid if cid else slugify(ds_name)
            if not bench_slug:
                bench_slug = slugify(folder)

            key = (folder, family, composite, bench_slug)
            cand = candidates.setdefault(key, {
                "canonical_id": cid,
                "slice_names": set(),
            })

            # v2's slice extraction (lines 610-668):
            # If eval_name has dots, peel off family prefix when matching;
            # the remaining first segment becomes the slice name.
            slice_name: str | None = None
            ev = eval_name

            # Strip "GAIA - X" style prefixes
            if " - " in ev:
                head, rest = ev.split(" - ", 1)
                if slugify(head) in (family, slugify(ds_name)):
                    ev = rest.strip()

            if "." in ev:
                parts = ev.split(".")
                if len(parts) >= 2 and slugify(parts[0]) == family:
                    parts = parts[1:]
                if len(parts) >= 2:
                    slice_name = parts[0]
                elif len(parts) == 1:
                    # whole eval_name becomes the slice candidate
                    slice_name = parts[0]
            else:
                # No dots: eval_name itself is the candidate slice
                # unless it equals the dataset/family root
                root_keys = {slugify(ds_name), family}
                if slugify(ev) not in root_keys and ev:
                    slice_name = ev

            if slice_name:
                cand["slice_names"].add(slice_name)

    # Pass 2: maybe_promote_slices simulation
    surfaced_canonicals: set[str] = set()
    surfaced_keys: set[str] = set()  # keys (canonical or synthetic) the simulation surfaces
    for key, cand in candidates.items():
        folder, family, composite, bench_slug = key
        slice_names = cand["slice_names"]
        canonical_slice_count = sum(
            1 for s in slice_names
            if resolve_canonical_strict(s, canonical_set) is not None
        )
        promote = canonical_slice_count >= 2

        if promote:
            for s in slice_names:
                cid = resolve_canonical_strict(s, canonical_set)
                if cid:
                    surfaced_canonicals.add(cid)
                    surfaced_keys.add(cid)
                else:
                    # synthetic key (price_1m_blended, etc.) — still surfaces
                    surfaced_keys.add(slugify(f"{bench_slug}-{s}"))
        else:
            # No promotion; the candidate itself is the benchmark
            if cand["canonical_id"]:
                surfaced_canonicals.add(cand["canonical_id"])
            surfaced_keys.add(bench_slug)

    return surfaced_canonicals, surfaced_keys, candidates


def v2_canonicals(ref: dict) -> set[str]:
    out: set[str] = set()
    for fam in ref.get("families") or []:
        for layout in ("standalone_benchmarks", "benchmarks"):
            for b in fam.get(layout) or []:
                cid = b.get("underlying_canonical_id")
                if cid:
                    out.add(cid)
    return out


def v2_keys(ref: dict) -> set[str]:
    out: set[str] = set()
    for fam in ref.get("families") or []:
        for layout in ("standalone_benchmarks", "benchmarks"):
            for b in fam.get(layout) or []:
                if b.get("key"):
                    out.add(b["key"])
                if b.get("underlying_canonical_id"):
                    out.add(b["underlying_canonical_id"])
    return out


def v3_keys(h: dict) -> set[str]:
    out: set[str] = set()
    for fam in h.get("families") or []:
        if fam.get("key"):
            out.add(fam["key"])
        for layout in ("standalone_benchmarks", "benchmarks"):
            for b in fam.get(layout) or []:
                if b.get("key"):
                    out.add(b["key"])
        for c in fam.get("composites") or []:
            if c.get("key"):
                out.add(c["key"])
            for b in c.get("benchmarks") or []:
                if b.get("key"):
                    out.add(b["key"])
    return out


def main() -> int:
    print("Loading registry...", file=sys.stderr)
    canonical_set, alias_to_canonical, parent = load_registry()
    print(f"  canonical benchmarks: {len(canonical_set)}", file=sys.stderr)
    print(f"  aliases: {len(alias_to_canonical)}", file=sys.stderr)

    print("Running simulation...", file=sys.stderr)
    sim_canonicals, sim_keys, candidates = simulate(
        canonical_set, alias_to_canonical, parent
    )
    print(f"  candidates produced: {len(candidates)}", file=sys.stderr)

    print("Loading reference + current...", file=sys.stderr)
    ref = json.loads(REF_PATH.read_text())
    cur = json.loads(V3_PATH.read_text())
    v2_can = v2_canonicals(ref)
    v2_k = v2_keys(ref)
    v3_k = v3_keys(cur)

    print()
    print("=" * 70)
    print("CANONICAL-ID RECOVERY (entities the registry recognises)")
    print("=" * 70)
    print(f"v2 ref canonicals (with underlying_canonical_id): {len(v2_can)}")
    print(f"v3 current keys (any level):                       {len(v3_k)}")
    print(f"Simulation canonical ids surfaced:                {len(sim_canonicals)}")
    print()
    print(f"v2 ∩ v3 current:        {len(v2_can & v3_k)}")
    print(f"v2 ∩ simulation:        {len(v2_can & sim_canonicals)}")
    print(f"v2 ∖ v3 current (gap):  {len(v2_can - v3_k)}")
    print(f"v2 ∖ simulation (residual): {len(v2_can - sim_canonicals)}")
    print()
    recovered = (v2_can & sim_canonicals) - v3_k
    print(f"RECOVERED by simulation (in v2, not in v3, would be reachable): {len(recovered)}")

    print()
    print("=" * 70)
    print("ALL-KEY COVERAGE (including v2's synthetic keys)")
    print("=" * 70)
    print(f"v2 ref keys (any):        {len(v2_k)}")
    print(f"Simulation keys surfaced: {len(sim_keys)}")
    print(f"v2 ∩ simulation: {len(v2_k & sim_keys)}")
    print(f"v2 ∖ simulation: {len(v2_k - sim_keys)} (these are the residual broken links the fix wouldn't catch)")

    # Show samples of each bucket for inspection
    print()
    print("=" * 70)
    print("SAMPLES")
    print("=" * 70)
    print()
    print("Sample of v2 canonicals NOT in v3 BUT recovered by simulation (first 25):")
    for k in sorted(recovered)[:25]:
        print(f"  {k}")
    print()
    print("Sample of v2 canonicals NOT in simulation (residual, first 25):")
    for k in sorted(v2_can - sim_canonicals)[:25]:
        print(f"  {k}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

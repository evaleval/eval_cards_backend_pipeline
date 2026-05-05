"""
Pipeline-side v2 hierarchy emitter.

Builds the hierarchy.json sidecar directly from the freshly-pulled EEE
datastore, applying the v2 spec rules: family-stem grouping (spec §1–§7),
slice→benchmark promotion, dot-namespace handling, prefix stripping
("GAIA - GAIA Level 1"), eval_name parsing, raw_verified-driven 1P/3P,
and cross-suite benchmark_index at benchmark + metric levels (spec §8).

Two adaptations vs the standalone `scripts/build_hierarchy_v2.py`:
  - paths come from the caller via `write_hierarchy_v2(...)` so Stage J
    can point at the same caches Stage A populates.
  - `model_results[]` is NOT emitted per metric — the frontend's
    hierarchy.json is a navigation/structure layer only; per-model
    rows live in the warehouse parquets.

Curation lives in the in-code constants below (EXPLICIT_FAMILY_MAP,
EXPLICIT_COMPOSITE_MAP, FAMILY_PRIMARY_OVERRIDE, ACRONYMS). Only
`categorized.json` (curator-supplied benchmark→categories[] mapping,
~555 entries) is loaded from `temp_registry_override/` since
that's impractical to inline.

The legacy registry-driven emit (`write_hierarchy` in sidecars.py)
stays in the codebase but is no longer wired into Stage J. Delete it
once the v2 path is the only one in production.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Defaults — used when callers don't override. The pipeline's Stage J
# call to `write_hierarchy_v2` passes explicit paths.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EEE_DIR = _DEFAULT_ROOT / ".cache/eee_datastore/data"
_DEFAULT_REGISTRY_DIR = _DEFAULT_ROOT / ".cache/entity_registry"
_DEFAULT_OVERRIDE_DIR = _DEFAULT_ROOT / "temp_registry_override"

EEE_DIR = _DEFAULT_EEE_DIR
REGISTRY_DIR = _DEFAULT_REGISTRY_DIR
OVERRIDE_DIR = _DEFAULT_OVERRIDE_DIR

EXCLUDED_FOLDERS = {"alphaxiv"}


def _configure_paths(*, eee_dir: Path | None = None,
                     registry_dir: Path | None = None,
                     override_dir: Path | None = None) -> None:
    """Override module-level paths before `build()` runs. Resets the lazy
    categorized.json index so the next lookup re-reads from the new
    override_dir."""
    global EEE_DIR, REGISTRY_DIR, OVERRIDE_DIR
    global _CATEGORIZED_RAW, _CATEGORIZED
    if eee_dir is not None:
        EEE_DIR = Path(eee_dir)
    if registry_dir is not None:
        REGISTRY_DIR = Path(registry_dir)
    if override_dir is not None:
        OVERRIDE_DIR = Path(override_dir)
    _CATEGORIZED_RAW = _load_categorized_raw()
    _CATEGORIZED = {}


def _load_categorized_raw() -> dict:
    path = OVERRIDE_DIR / "categorized.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"[hierarchy_v2] failed to read categorized.json: {exc}",
              file=sys.stderr)
        return {}


_CATEGORIZED_RAW = _load_categorized_raw()
_CATEGORIZED: dict[str, list[str]] = {}  # populated lazily on first lookup


def _ensure_categorized_index() -> dict[str, list[str]]:
    """Index categorized.json by lowercased display name only. The curated
    file ships display-name keys ("AIME 2025", "APEX Agents", …); slugs and
    canonical ids do NOT match its keyspace."""
    global _CATEGORIZED
    if _CATEGORIZED or not _CATEGORIZED_RAW:
        return _CATEGORIZED
    for k, v in _CATEGORIZED_RAW.items():
        cats = [str(c) for c in v] if isinstance(v, list) else []
        if not cats:
            continue
        _CATEGORIZED[k.strip().lower()] = cats
    print(f"[override] loaded {len(_CATEGORIZED_RAW)} categorized entries",
          file=sys.stderr)
    return _CATEGORIZED


def lookup_categories(*display_names: str | None) -> list[str]:
    """Look up curated categories by display name (case-insensitive exact
    match). Pass multiple candidates (e.g. raw_display_name then key) and
    the first hit wins."""
    idx = _ensure_categorized_index()
    if not idx:
        return []
    for c in display_names:
        if not c:
            continue
        cats = idx.get(str(c).strip().lower())
        if cats:
            return list(cats)
    return []


def primary_category(cats: list[str]) -> str:
    return cats[0] if cats else "other"


# Spec §3.1 — suffix rules (applied iteratively to a normalised slug)
SUFFIX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-level-\d+$"), ""),
    (re.compile(r"-l\d+$"), ""),
    (re.compile(r"-v\d+(\.\d+)?$"), ""),
    (re.compile(r"-(diamond|lite|mini|hard|extra|easy|pro)$"), ""),
    (re.compile(r"-vs-[a-z0-9-]+$"), ""),
    (re.compile(r"-(auto-avg|caption-length)$"), ""),
    (re.compile(r"-(zero-shot|few-shot|cot)$"), ""),
    (re.compile(r"-\d+shot$"), ""),
    # tau-bench-2_airline → tau-bench-2; fibble3_arena → fibble_arena (handled
    # via underscore→dash earlier)
    (re.compile(r"-(airline|retail|telecom|banking)$"), ""),
    # MMMU-Multiple-Choice, MMMU-Open-Ended → mmmu
    (re.compile(r"-(multiple-choice|open-ended)$"), ""),
    # fibble1, fibble2 → fibble; agentharm-foo → agentharm; etc.
    (re.compile(r"^(fibble)\d+(?=-)"), r"\1"),
]

# Spec §3.2 — explicit alias map: folder/raw → family stem
EXPLICIT_FAMILY_MAP: dict[str, str] = {
    "hal_gaia": "gaia",
    "hal_gaia_level_1": "gaia",
    "hal_gaia_level_2": "gaia",
    "hal_gaia_level_3": "gaia",
    "hfopenllm_v2": "hf-open-llm-v2",
    # HELM: every flavour rolls up under one HELM family
    "helm_classic": "helm",
    "helm_lite": "helm",
    "helm_capabilities": "helm",
    "helm_instruct": "helm",
    "helm_mmlu": "helm",
    "mt_bench": "mt-bench",
    "mtbench": "mt-bench",
    "appworld_test_normal": "appworld",
    "tau-bench-2_airline": "tau-bench",
    "tau-bench-2_retail": "tau-bench",
    "tau-bench-2_telecom": "tau-bench",
    "tau-bench-2_banking": "tau-bench",
    "fibble_arena": "fibble-arena",
    "fibble1_arena": "fibble-arena",
    "fibble2_arena": "fibble-arena",
    "fibble3_arena": "fibble-arena",
    "fibble4_arena": "fibble-arena",
    "fibble5_arena": "fibble-arena",
    "wordle_arena": "wordle-arena",
    "MMMU-Multiple-Choice": "mmmu",
    "MMMU-Open-Ended": "mmmu",
    "MMLU-Pro": "mmlu-pro",
    "global-mmlu-lite": "global-mmlu-lite",
    "GAIA": "gaia",
    "IFEval": "ifeval",
    "MathVista": "mathvista",
    "swe-bench": "swe-bench",
    "terminal-bench-2.0": "terminal-bench-2",
    "livecodebenchpro": "livecodebench-pro",
    "reward-bench": "reward-bench",
    "reward-bench-2": "reward-bench",
    "rewardbench": "reward-bench",
    "rewardbench-2": "reward-bench",
    "arc-agi": "arc-agi",
    "apex-agents": "apex-agents",
    "apex-v1": "apex-v1",
    "agentharm": "agentharm",
    "ace": "ace",
    "bfcl": "bfcl",
    "browsecompplus": "browsecompplus",
    "la_leaderboard": "la-leaderboard",
    "sciarena": "sciarena",
    "theory_of_mind": "theory-of-mind",
    # Newer EEE folders
    "artificial-analysis-llms": "artificial-analysis",
    "big_bench_hard": "bbh",
    "caparena-auto": "caparena",
    "cocoabench": "cocoabench",
    "commonsense_qa": "commonsense-qa",
    "cvebench": "cvebench",
    "cybench": "cybench",
    "cyse2_interpreter_abuse": "cyse2",
    "cyse2_prompt_injection": "cyse2",
    "cyse2_vulnerability_exploit": "cyse2",
    "facts-grounding": "facts-grounding",
    "gpqa-diamond": "gpqa",
    "gpqa_diamond": "gpqa",
    "gdm_intercode_ctf": "gdm-intercode-ctf",
    # HELM extras
    "helm_air_bench": "helm",
    "helm_safety": "helm",
    # HAL (Holistic Agent Leaderboard): every hal-* folder rolls under one
    # HAL family, each folder becoming its own benchmark inside.
    "hal-gaia": "hal",
    "hal-assistantbench": "hal",
    "hal-corebench-hard": "hal",
    "hal-online-mind2web": "hal",
    "hal-scicode": "hal",
    "hal-scienceagentbench": "hal",
    "hal-swebench-verified-mini": "hal",
    "hal-taubench-airline": "hal",
    "hal-usaco": "hal",
}

# Folder → composite slug, when the folder belongs to a multi-folder family
# and should occupy its own composite layer between family and benchmarks.
EXPLICIT_COMPOSITE_MAP: dict[str, str] = {
    "helm_classic": "helm-classic",
    "helm_lite": "helm-lite",
    "helm_capabilities": "helm-capabilities",
    "helm_instruct": "helm-instruct",
    "helm_mmlu": "helm-mmlu",
    "helm_air_bench": "helm-air-bench",
    "helm_safety": "helm-safety",
}

# Known acronyms preserved verbatim in titlecase
ACRONYMS = {"gaia", "gpqa", "mmlu", "mmmu", "bfcl", "helm", "ifeval", "bbh",
            "mmlu-pro", "musr", "math", "gsm8k", "arc-agi", "swe-bench",
            "boolq", "imdb", "raft", "quac", "hellaswag", "openbookqa",
            "narrativeqa", "truthfulqa", "civilcomments", "wildbench",
            "legalbench", "medqa", "swe", "ace", "hle", "aime", "cve",
            "ctf", "cot", "rlhf", "api", "csv", "json", "ai", "ml",
            "lcr", "ifbench", "scicode", "hal", "usaco"}

# Hard-coded category guesses by family stem; "other" is the fallback.
FAMILY_CATEGORY: dict[str, str] = {
    "gaia": "agentic", "swe-bench": "agentic", "appworld": "agentic",
    "terminal-bench-2": "agentic", "tau-bench-2": "agentic",
    "tau-bench": "agentic",
    "browsecompplus": "agentic", "agentharm": "safety", "arc-agi": "reasoning",
    "bfcl": "tool-use", "apex-agents": "agentic", "apex-v1": "agentic",
    "mmlu": "knowledge", "mmlu-pro": "knowledge", "mmmu": "multimodal",
    "global-mmlu-lite": "knowledge",
    "helm": "general", "helm-classic": "general",
    "helm-lite": "general", "helm-capabilities": "general",
    "helm-instruct": "general", "helm-mmlu": "knowledge",
    "hf-open-llm-v2": "general", "ifeval": "instruction-following",
    "ace": "general", "reward-bench": "reward-modelling",
    "fibble-arena": "reasoning", "wordle-arena": "reasoning",
    "livecodebench-pro": "code", "sciarena": "general",
    "mathvista": "math", "theory-of-mind": "reasoning",
    "la-leaderboard": "general",
    "artificial-analysis": "general", "bbh": "reasoning",
    "caparena": "multimodal", "cocoabench": "multimodal",
    "commonsense-qa": "reasoning", "cvebench": "security",
    "cybench": "security", "cyse2": "security",
    "facts-grounding": "factuality",
    "hal": "agentic",
}


# ──────────────────────────────────────────────────────────────────────────
# Slug + display helpers
# ──────────────────────────────────────────────────────────────────────────

def slugify(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"[^a-z0-9-/.]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def titleize(stem: str, reg: dict[str, Any] | None = None) -> str:
    """Convert a slug into a human-readable display name. Prefers the
    registry's display_name when the slug matches a canonical id."""
    if reg is not None:
        canon = reg.get("display_name", {}).get(stem)
        if canon:
            return canon
    if stem in ACRONYMS:
        return stem.upper()
    return prettify_display(stem)


_ACRONYM_UPPER = {a.upper() for a in ACRONYMS}


def _title_segment(seg: str) -> str:
    """Title-case a single segment, preserving known acronyms; numeric-only
    tokens (1m, v1, 2024) stay as-is."""
    if not seg:
        return seg
    if seg in ACRONYMS or seg.upper() in _ACRONYM_UPPER:
        return seg.upper()
    # Pure number / version-like → leave alone
    if seg.replace(".", "").isdigit() or re.fullmatch(r"v\d+(\.\d+)*", seg):
        return seg
    if seg[0].isalpha() and seg[0].islower():
        return seg[:1].upper() + seg[1:]
    return seg


def prettify_display(name: str) -> str:
    """Cleanup an arbitrary slice/eval label into a tidy human title.

    - Replaces underscores with spaces.
    - When the input has no uppercase characters (looks like a slug), title-
      cases each hyphen-segment of each word (so "caparena-auto" →
      "Caparena-Auto", "swe-bench" → "Swe-Bench").
    - Preserves known acronyms uppercase.
    - Mixed-case inputs are left alone — registries already supply nice
      display names like "MMLU-Pro", "Humanity's Last Exam"."""
    if not name:
        return name
    cleaned = name.replace("_", " ").strip()
    if any(c.isupper() for c in cleaned):
        return cleaned
    words = cleaned.split()
    out = []
    for w in words:
        if "-" in w:
            out.append("-".join(_title_segment(s) for s in w.split("-")))
        else:
            out.append(_title_segment(w))
    return " ".join(out)


# Metric-tail detector. When evaluation_name = "{slice} {metric}" without an
# explicit metric_config.metric_name, peel off the trailing metric token so
# eval_names like "Overall Pass@1", "Overall Mean Score" collapse into one
# slice "Overall" carrying both Pass@1 and Mean Score metrics.
_METRIC_TAIL_RE = re.compile(
    r"\s+("
    r"(?:mean\s+)?(?:score|accuracy|f1|em|loss)|"
    r"pass@\d+|pass@k|"
    r"standard\s+error|win\s+rate|"
    r"avg(?:\s+(?:attempts|latency(?:_ms)?))?|"
    r"average\s+attempts|"
    r"mean|rank|elo"
    r")$",
    re.IGNORECASE,
)


# Metric names that, when present, mark a metric as the primary readout for
# its benchmark. Order matters — earlier wins. Compared case-insensitively.
PRIMARY_METRIC_PREFERENCE = [
    "overall", "mean win rate", "mean score", "score", "accuracy",
    "exact match", "exact_match", "win rate", "elo", "rank",
    "pass@1", "f1", "mean",
]


# Explicit overrides for which benchmark is the primary readout of a family
# or composite. Keyed by family/composite slug; value is the benchmark key
# (post-resolution). Add entries as the user calls them out.
FAMILY_PRIMARY_OVERRIDE: dict[str, str] = {
    "artificial-analysis": "artificial-analysis-intelligence-index",
}


def pick_primary_metric(metrics: list[dict]) -> str | None:
    """Return the metric_key of the primary metric for a benchmark, or None."""
    if not metrics:
        return None
    by_name = {m["metric_name"].strip().lower(): m for m in metrics}
    for pref in PRIMARY_METRIC_PREFERENCE:
        if pref in by_name:
            return by_name[pref]["metric_key"]
    # Fallback: most-reported metric (highest models_count, then alphabetical)
    best = max(metrics, key=lambda m: (m.get("models_count", 0), -ord(m["metric_key"][:1] or " ")))
    return best["metric_key"]


def parse_eval_name(eval_name: str) -> tuple[str, str | None]:
    """Returns (slice_name, metric_label_or_None). metric_label is None when
    we couldn't detect a tail — treat the whole eval_name as the slice."""
    m = _METRIC_TAIL_RE.search(eval_name)
    if m and m.start() > 0:
        return eval_name[:m.start()].rstrip(), eval_name[m.start():].strip()
    return eval_name, None


def strip_family_prefix(family: str, name: str) -> str:
    """For benchmark.display_name — strip redundant family prefix."""
    fl = family.lower()
    nl = name.lower()
    if nl.startswith(fl):
        rest = name[len(family):].lstrip(" -:_/")
        return rest or name
    return name


# ──────────────────────────────────────────────────────────────────────────
# Entity-registry loaders
# ──────────────────────────────────────────────────────────────────────────

def load_registry() -> dict[str, Any]:
    out: dict[str, Any] = {"alias_to_canonical": {}, "canonical": {},
                           "benchmark_parent": {}, "display_name": {}}
    cb_path = REGISTRY_DIR / "canonical_benchmarks/part-0.parquet"
    al_path = REGISTRY_DIR / "aliases/part-0.parquet"
    if not cb_path.exists():
        print(f"WARNING: {cb_path} missing — registry features disabled", file=sys.stderr)
        return out

    cb = pd.read_parquet(cb_path)
    for _, r in cb.iterrows():
        cid = r["id"]
        out["canonical"][cid] = {
            "id": cid,
            "display_name": r["display_name"],
            "parent_benchmark_id": r.get("parent_benchmark_id"),
            "tags": r.get("tags"),
        }
        out["display_name"][cid] = r["display_name"]
        if pd.notna(r.get("parent_benchmark_id")):
            out["benchmark_parent"][cid] = r["parent_benchmark_id"]

    al = pd.read_parquet(al_path)
    for _, r in al.iterrows():
        if r["entity_type"] != "benchmark":
            continue
        if r["status"] not in ("confirmed", "auto"):
            continue
        raw = str(r["raw_value"])
        out["alias_to_canonical"][raw] = r["canonical_id"]
        out["alias_to_canonical"][raw.lower()] = r["canonical_id"]
        out["alias_to_canonical"][slugify(raw)] = r["canonical_id"]
    return out


_GENERIC_WORDS = {
    "score", "overall", "main", "all", "total", "sum", "mean", "avg",
    "best", "first", "last", "rank", "default",
}


def resolve_canonical_strict(reg: dict[str, Any], raw: str) -> str | None:
    """Stricter alias resolution: only when slugify(raw) directly matches a
    canonical id, AND that id isn't a single generic word. Used for slice→
    benchmark promotion so suite-scoped aliases like 'Score' (SCoRE) or
    'Overall' (apex-v1) don't pollute cross-suite matches, but real bench
    names like 'aime', 'gpqa', 'mmlu_pro', 'livecodebench' still resolve."""
    if not raw:
        return None
    s = slugify(raw)
    if not s or s in _GENERIC_WORDS:
        return None
    if s in reg["canonical"]:
        return s
    return None


def resolve_canonical(reg: dict[str, Any], raw: str) -> str | None:
    if not raw:
        return None
    for k in (raw, raw.lower(), slugify(raw)):
        if k in reg["alias_to_canonical"]:
            return reg["alias_to_canonical"][k]
    if raw in reg["canonical"]:
        return raw
    if slugify(raw) in reg["canonical"]:
        return slugify(raw)
    return None


def walk_to_root(reg: dict[str, Any], cid: str) -> str:
    seen = set()
    cur = cid
    while cur in reg["benchmark_parent"] and cur not in seen:
        seen.add(cur)
        cur = reg["benchmark_parent"][cur]
    return cur


# ──────────────────────────────────────────────────────────────────────────
# Family-stem computation (spec §3)
# ──────────────────────────────────────────────────────────────────────────

def family_stem(folder: str, reg: dict[str, Any]) -> str:
    # 1. Explicit in-code alias map — covers the curated multi-folder
    #    families (HELM, HAL, tau-bench, fibble, mmmu, reward-bench, …).
    if folder in EXPLICIT_FAMILY_MAP:
        return EXPLICIT_FAMILY_MAP[folder]

    # 2. Try entity-registry: alias → canonical → walk parent chain
    cid = resolve_canonical(reg, folder)
    if cid:
        root = walk_to_root(reg, cid)
        if root:
            return slugify(root)

    # 3. Suffix-stripping rules
    stem = slugify(folder)
    while True:
        prev = stem
        for pat, rep in SUFFIX_PATTERNS:
            stem = pat.sub(rep, stem)
            stem = re.sub(r"-+", "-", stem).strip("-")
            if not stem:
                stem = prev
                break
        if stem == prev:
            break
    return stem or slugify(folder)


# ──────────────────────────────────────────────────────────────────────────
# 1st/3rd-party rule (per user instructions, May 2026)
# ──────────────────────────────────────────────────────────────────────────

def is_llm_stats_source(source_name: str | None) -> bool:
    if not source_name:
        return False
    s = source_name.lower()
    return "llm-stats" in s or "llmstats" in s or "llm stats" in s or "llm_stats" in s


def party_label(source_metadata: dict, metric_config: dict) -> str:
    """Override existing labels. Only llm-stats records with raw_verified==false
    are first_party; everything else is third_party."""
    src = source_metadata.get("source_name") if source_metadata else None
    if is_llm_stats_source(src):
        ad = (metric_config or {}).get("additional_details") or {}
        rv = ad.get("raw_verified")
        if isinstance(rv, str):
            rv_norm = rv.strip().lower()
            if rv_norm == "false":
                return "first_party"
        elif rv is False:
            return "first_party"
    return "third_party"


# ──────────────────────────────────────────────────────────────────────────
# EEE record extraction
# ──────────────────────────────────────────────────────────────────────────

def iter_eee_records(eee_dir: Path) -> Iterable[tuple[str, dict]]:
    for jsonp in eee_dir.rglob("*.json"):
        if jsonp.name.endswith("_samples.jsonl"):
            continue
        parts = jsonp.relative_to(eee_dir).parts
        if not parts:
            continue
        folder = parts[0]
        if folder in EXCLUDED_FOLDERS:
            continue
        try:
            with jsonp.open() as f:
                rec = json.load(f)
        except Exception as exc:
            print(f"skip {jsonp}: {exc}", file=sys.stderr)
            continue
        yield folder, rec


def normalize_score(sd: dict | None) -> float | None:
    if not sd:
        return None
    v = sd.get("score")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────
# Main build
# ──────────────────────────────────────────────────────────────────────────

def build() -> dict:
    print("Loading entity registry…", file=sys.stderr)
    reg = load_registry()
    print(f"  {len(reg['canonical'])} canonical benchmarks, "
          f"{len(reg['alias_to_canonical'])} aliases", file=sys.stderr)

    # Pass 1: read every record, emit one "row" per (folder, dataset_name,
    # evaluation_name, model). Track tags + provenance per benchmark.
    print("Scanning EEE records…", file=sys.stderr)

    # benchmarks[(family, composite_key, benchmark_slug)] = {...}
    # composite_key is None when the benchmark sits directly under the family.
    benchmarks: dict[tuple[str, str | None, str], dict] = {}
    folder_count = Counter()
    record_count = 0

    for folder, rec in iter_eee_records(EEE_DIR):
        folder_count[folder] += 1
        record_count += 1

        sm = rec.get("source_metadata", {}) or {}
        mi = rec.get("model_info", {}) or {}
        model_id = mi.get("id") or mi.get("name") or "unknown"
        model_name = mi.get("name") or model_id
        developer = mi.get("developer")

        family = family_stem(folder, reg)
        # A composite layer is only emitted when the folder belongs to a
        # multi-composite family (HELM, HAL, tau-bench, …). Singleton
        # families collapse the composite into the family directly.
        composite = EXPLICIT_COMPOSITE_MAP.get(folder)

        for er in rec.get("evaluation_results", []) or []:
            sd = er.get("source_data") or {}
            ds_name = sd.get("dataset_name") or folder
            mc = er.get("metric_config") or {}
            score_obj = er.get("score_details") or {}
            value = normalize_score(score_obj)
            if value is None:
                continue

            # Benchmark-within-family key. Use canonical alias if available,
            # else slug of dataset_name.
            cid = resolve_canonical(reg, ds_name)
            bench_slug = cid if cid else slugify(ds_name)
            if not bench_slug:
                bench_slug = slugify(folder)

            # is_overall = the row IS the level-stem (composite or family).
            # For composite layer (e.g. HELM Capabilities), it's the row whose
            # ds_name equals the folder. For family-direct, ds_name==folder
            # AND folder==family.
            level_key = composite or family
            is_overall = slugify(ds_name) == slugify(folder) and (
                composite is not None or slugify(folder) == family
            )
            if is_overall:
                bench_slug = level_key

            bkey = (family, composite, bench_slug)
            b = benchmarks.setdefault(bkey, {
                "key": bench_slug,
                "family": family,
                "display_name": reg["display_name"].get(cid) or prettify_display(ds_name),
                "dataset_name": ds_name,
                "underlying_canonical_id": cid,
                "is_overall": is_overall,
                "domains": set(),
                "languages": set(),
                "tasks": set(),
                "model_ids": set(),
                "summary_eval_ids": set(),
                "slices": {},
                # Provenance counters (for §4 roll-up):
                "provenance": {
                    "total_results": 0,
                    "first_party": 0,
                    "third_party": 0,
                    "by_source": Counter(),
                    "groups": defaultdict(set),  # group_key -> set(party)
                },
                "metric_keys": set(),
            })

            # Tags
            tags = (sd.get("tags") or {}) if isinstance(sd.get("tags"), dict) else {}
            for tk, dest in (("domains", "domains"), ("languages", "languages"),
                             ("tasks", "tasks")):
                vs = tags.get(tk) or []
                if isinstance(vs, list):
                    for v in vs:
                        if isinstance(v, str) and v.strip() and v.strip().lower() != "not specified":
                            b[dest].add(v.strip())

            # Slice / metric routing
            eval_name = er.get("evaluation_name") or mc.get("metric_name") or "score"
            metric_id = mc.get("metric_id") or slugify(mc.get("metric_name") or "score")
            metric_name = mc.get("metric_name") or eval_name
            lower_better = bool(mc.get("lower_is_better"))

            # When metric_config didn't carry an explicit metric_name, try to
            # peel a metric tail off the eval_name. Lets "Overall Pass@1",
            # "Overall Mean Score" collapse into slice "Overall" with two
            # metrics, instead of two siblings each with a single metric.
            if not mc.get("metric_name") and "." not in eval_name:
                parsed_slice, parsed_metric = parse_eval_name(eval_name)
                if parsed_metric:
                    eval_name = parsed_slice
                    metric_name = parsed_metric
                    metric_id = slugify(parsed_metric)

            # Dot-namespaced eval_names (e.g. "bfcl.live.live_accuracy") split
            # into slice = "live", metric_id = "live_accuracy". Strip a leading
            # family-stem token if present. Matches existing-pipeline grouping
            # for BFCL/HELM-style sources.
            slice_key = None
            slice_name = None

            # Strip leading "GAIA - " / "BFCL - " / "MMLU - " prefixes —
            # eval_name "GAIA - GAIA Level 1" becomes "GAIA Level 1" so the
            # canonical id (gaia-level-1) resolves and the level appears as
            # its own benchmark inside the family.
            if " - " in eval_name:
                head, rest = eval_name.split(" - ", 1)
                if slugify(head) in (family, slugify(ds_name),
                                      slugify(re.sub(r"\s*\([^)]*\)\s*$", "", ds_name))):
                    eval_name = rest.strip()

            if "." in eval_name:
                parts = eval_name.split(".")
                if len(parts) >= 2 and slugify(parts[0]) == family:
                    parts = parts[1:]  # strip family prefix (bfcl., artificial_analysis.)
                    eval_name = ".".join(parts) or eval_name  # rewrite for downstream
                if len(parts) >= 2:
                    slice_key = slugify(parts[0])
                    slice_name = parts[0]
                    metric_id = ".".join(parts[1:]) if not mc.get("metric_id") else metric_id
                    if not mc.get("metric_name"):
                        metric_name = parts[-1].replace("_", " ").strip()

            # Detect "X (metric_label)" eval_names where the parenthetical is
            # actually the metric (HLE: "Humanity's Last Exam (accuracy)" with
            # metric_name="Accuracy"). Strip the parens for slice-routing.
            m_paren = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", eval_name)
            if m_paren and mc.get("metric_name"):
                if slugify(m_paren.group(2)) == slugify(metric_name):
                    eval_name = m_paren.group(1).strip()
            # ds_name with any trailing parenthetical removed (helps HLE's
            # "Humanity's Last Exam (Scale SEAL leaderboard)" match)
            ds_bare = re.sub(r"\s*\([^)]*\)\s*$", "", ds_name).strip()

            # No slice when eval_name IS the root: it matches ds_name (with or
            # without parens) OR the family stem.
            # Also collapse when eval_name == metric_name AND eval_name doesn't
            # look like a real benchmark (no canonical resolution) — that
            # catches caparena's "CapArena-Auto Score (vs X)" variants without
            # collapsing AA's "aime"/"gpqa"/"mmlu_pro" (which DO resolve to
            # canonicals and should stay as separate benchmarks).
            if slice_key is None:
                root_keys = {slugify(ds_name), slugify(ds_bare), family}
                slug_eval = slugify(eval_name)
                eval_is_canonical = resolve_canonical_strict(reg, eval_name) is not None
                if (slug_eval in root_keys
                        or (slug_eval == slugify(metric_name)
                            and not eval_is_canonical)):
                    slice_key = None
                    slice_name = None
                else:
                    slice_key = slug_eval
                    slice_name = eval_name

            slice_dict = b["slices"].setdefault(slice_key or "__root__", {
                "slice_key": slice_key,
                "slice_name": slice_name,
                "metrics": {},
            })
            mkey = metric_id
            metric = slice_dict["metrics"].setdefault(mkey, {
                "metric_key": metric_id,
                "metric_name": metric_name,
                "lower_is_better": lower_better,
                "model_results": [],
            })

            party = party_label(sm, mc)
            b["provenance"]["total_results"] += 1
            b["provenance"][party] += 1
            b["provenance"]["by_source"][sm.get("source_name") or "unknown"] += 1
            grp_key = (model_id, ds_name, eval_name, metric_id)
            b["provenance"]["groups"][grp_key].add(party)

            metric["model_results"].append({
                "model_id": model_id,
                "model_name": model_name,
                "developer": developer,
                "score": value,
                "party": party,
                "source_name": sm.get("source_name"),
                "evaluation_id": rec.get("evaluation_id"),
                "retrieved_timestamp": rec.get("retrieved_timestamp"),
            })
            b["model_ids"].add(model_id)
            b["summary_eval_ids"].add(bench_slug)
            b["metric_keys"].add(metric_id)

    print(f"  scanned {record_count} records across {len(folder_count)} folders",
          file=sys.stderr)

    # Pass 2: assemble per-benchmark payloads, grouped by (family, composite)
    print("Assembling family hierarchy…", file=sys.stderr)

    def serialize_slices(slices: dict) -> list[dict]:
        out: list[dict] = []
        for sk, sd in sorted(slices.items(),
                             key=lambda kv: (kv[0] != "__root__", kv[0])):
            metrics_out: list[dict] = []
            for mid, m in sorted(sd["metrics"].items()):
                # Navigation-layer only: counts + top_score for previews.
                # Per-model rows live in models.parquet — frontend joins
                # on (benchmark_key, metric_key) when it needs them.
                vals = [r["score"] for r in m["model_results"]]
                metrics_out.append({
                    "metric_key": m["metric_key"],
                    "metric_name": prettify_display(m["metric_name"]),
                    "lower_is_better": m["lower_is_better"],
                    "models_count": len({r["model_id"] for r in m["model_results"]}),
                    "results_count": len(m["model_results"]),
                    "top_score": (min(vals) if m["lower_is_better"] else max(vals)) if vals else None,
                })
            out.append({
                "slice_key": sd["slice_key"],
                "slice_name": prettify_display(sd["slice_name"]) if sd["slice_name"] else None,
                "metrics": metrics_out,
            })
        return out

    def make_bench_payload(b: dict, family_key: str) -> dict:
        slices_out = serialize_slices(b["slices"])
        # Primary metric: prefer a root-scope metric, else the first slice's
        # primary, else None.
        primary_metric_key = None
        for sl in slices_out:
            if sl["slice_key"] in (None, "__root__"):
                primary_metric_key = pick_primary_metric(sl["metrics"])
                if primary_metric_key:
                    break
        if primary_metric_key is None and slices_out:
            primary_metric_key = pick_primary_metric(slices_out[0]["metrics"])
        # Mark each metric as primary or not for the viewer
        for sl in slices_out:
            for m in sl["metrics"]:
                m["is_primary"] = (m["metric_key"] == primary_metric_key
                                   and sl["slice_key"] in (None, "__root__"))
        prov = b["provenance"]
        groups = prov["groups"]
        first_only = sum(1 for v in groups.values() if v == {"first_party"})
        third_only = sum(1 for v in groups.values() if v == {"third_party"})
        multi = sum(1 for v in groups.values() if len(v) > 1)
        # categorized.json keys on display name only — id/slug won't match
        bench_cats = lookup_categories(b["display_name"])
        payload = {
            "key": b["key"],
            "display_name": strip_family_prefix(family_key, b["display_name"]) or "Overall",
            "raw_display_name": b["display_name"],
            "dataset_name": b["dataset_name"],
            "underlying_canonical_id": b["underlying_canonical_id"],
            "is_overall": b["is_overall"],
            "category": primary_category(bench_cats) or "other",
            "categories": bench_cats,
            "tags": {
                "domains": sorted(b["domains"]),
                "languages": sorted(b["languages"]),
                "tasks": sorted(b["tasks"]),
            },
            "models_count": len(b["model_ids"]),
            "results_count": prov["total_results"],
            "summary_eval_ids": sorted(b["summary_eval_ids"]),
            "slices": slices_out,
            "metric_keys": sorted(b["metric_keys"]),
            "primary_metric_key": primary_metric_key,
            "provenance_summary": {
                "total_results": prov["total_results"],
                "first_party_results": prov["first_party"],
                "third_party_results": prov["third_party"],
                "total_groups": len(groups),
                "first_party_only_groups": first_only,
                "third_party_only_groups": third_only,
                "multi_source_groups": multi,
                "source_distribution": dict(prov["by_source"].most_common()),
            },
        }
        cid = b["underlying_canonical_id"]
        if cid and cid != family_key and cid != b["key"]:
            payload["underlying_benchmark"] = {
                "key": cid,
                "display_name": reg["display_name"].get(cid, titleize(cid)),
            }
        return payload

    # Promote slices to benchmarks when the benchmark has no overall row.
    # ARC-AGI: dataset "ARC Prize evaluations leaderboard JSON" with slices
    # v1_Public_Eval, v1_Semi_Private, ... gets exploded into 6 sibling
    # benchmarks. Triggers when no `__root__` slice exists AND there's
    # >1 slice (single-slice benchmarks stay as-is so we don't lose them).
    def maybe_promote_slices(b: dict, family_key: str) -> list[dict]:
        slices = b["slices"]
        has_root = "__root__" in slices
        # Promote when there's no root (slices are independent benchmarks)
        # OR when 2+ non-root slices resolve to known canonical benchmarks
        # (AA case: a mixed bag of metric-like roots + canonical sub-benchmarks
        # — the canonical sub-benchmarks should bubble up as their own
        # benchmarks even though some root metrics also exist).
        canonical_slice_count = sum(
            1 for sk, sd in slices.items()
            if sk != "__root__"
            and resolve_canonical_strict(reg, sd["slice_name"] or sk) is not None
        )
        promote = (not has_root and len(slices) > 1) or canonical_slice_count >= 2
        if not promote:
            return [make_bench_payload(b, family_key)]
        # Promote each slice to a sibling benchmark
        out = []
        for sk, sd in slices.items():
            # __root__ slice → an "Overall" sibling that keeps the parent's
            # display name and canonical id. Other slices → their own bench.
            is_root_slice = (sk == "__root__")
            slice_label = sd["slice_name"] or sk
            slice_slug = slugify(slice_label)
            inherited_cid: str | None
            if is_root_slice:
                inherited_cid = b["underlying_canonical_id"]
                slice_label = b["display_name"]
                slice_slug = slugify(slice_label) or "overall"
            elif slice_slug == "overall":
                # Slice literally named "Overall" → inherit parent canonical
                inherited_cid = b["underlying_canonical_id"]
            else:
                inherited_cid = resolve_canonical_strict(reg, slice_label)
            # Use the registry's display_name when we resolved a canonical;
            # otherwise prettify the slice label (underscores → spaces).
            if inherited_cid:
                pretty_display = reg["display_name"].get(inherited_cid, prettify_display(slice_label))
            else:
                pretty_display = prettify_display(slice_label)
            child = {
                "key": inherited_cid or (b["key"] if is_root_slice
                                          else (slugify(b["key"] + "-" + sk) if b["key"] else sk)),
                "family": b["family"],
                "display_name": pretty_display,
                "dataset_name": b["dataset_name"],
                "underlying_canonical_id": inherited_cid,
                "is_overall": False,
                "domains": b["domains"], "languages": b["languages"], "tasks": b["tasks"],
                "model_ids": {r["model_id"] for m in sd["metrics"].values()
                              for r in m["model_results"]},
                "summary_eval_ids": {slugify(b["key"] + "-" + sk)},
                "slices": {"__root__": {"slice_key": None, "slice_name": None,
                                         "metrics": sd["metrics"]}},
                "provenance": {
                    "total_results": sum(len(m["model_results"]) for m in sd["metrics"].values()),
                    "first_party": sum(1 for m in sd["metrics"].values()
                                       for r in m["model_results"] if r["party"] == "first_party"),
                    "third_party": sum(1 for m in sd["metrics"].values()
                                       for r in m["model_results"] if r["party"] == "third_party"),
                    "by_source": Counter(r["source_name"] for m in sd["metrics"].values()
                                         for r in m["model_results"] if r.get("source_name")),
                    "groups": defaultdict(set),
                },
                "metric_keys": {m["metric_key"] for m in sd["metrics"].values()},
            }
            out.append(make_bench_payload(child, family_key))
        return out

    # Group benchmarks by (family, composite)
    by_family: dict[str, dict[str | None, list]] = defaultdict(lambda: defaultdict(list))
    for (family_key, composite, bench_slug), b in benchmarks.items():
        payloads = maybe_promote_slices(b, family_key)
        by_family[family_key][composite].extend(payloads)

    # Pass 3: shape each family
    print("Choosing composites vs standalone…", file=sys.stderr)
    out_families: list[dict] = []

    def prov_sum(items, field):
        return sum(b["provenance_summary"][field] for b in items)

    for fkey in sorted(by_family):
        comp_groups = by_family[fkey]  # composite_key (None or str) → [bench]
        # Sort each group: overall first
        for benches in comp_groups.values():
            benches.sort(key=lambda b: (not b["is_overall"], b["key"]))

        all_benches = [b for benches in comp_groups.values() for b in benches]

        domains = sorted({d for b in all_benches for d in b["tags"]["domains"]})
        langs = sorted({l for b in all_benches for l in b["tags"]["languages"]})
        tasks = sorted({t for b in all_benches for t in b["tags"]["tasks"]})
        all_eval_ids = sorted({eid for b in all_benches for eid in b["summary_eval_ids"]})
        evals_count = sum(b["models_count"] for b in all_benches)

        fam_prov = {
            "total_results": prov_sum(all_benches, "total_results"),
            "first_party_results": prov_sum(all_benches, "first_party_results"),
            "third_party_results": prov_sum(all_benches, "third_party_results"),
            "total_groups": prov_sum(all_benches, "total_groups"),
            "first_party_only_groups": prov_sum(all_benches, "first_party_only_groups"),
            "third_party_only_groups": prov_sum(all_benches, "third_party_only_groups"),
            "multi_source_groups": prov_sum(all_benches, "multi_source_groups"),
            "source_distribution": dict(sum(
                (Counter(b["provenance_summary"]["source_distribution"]) for b in all_benches),
                Counter())),
        }

        # Category: derived from the constituent benchmarks' curated
        # categories via mode-most-common. Direct family-slug lookup is
        # unsafe — slugs collide with unrelated curated entries (e.g. "HAL"
        # hits the hallucination entry, but our `hal` family is the
        # Holistic Agent Leaderboard). Falls back to the in-code
        # FAMILY_CATEGORY map, then "other".
        family_display = titleize(fkey, reg)
        # A family's `categories` is the UNION of its members' categories
        # (deduplicated, sorted). `category` (singular) keeps the most-common
        # one as a backward-compat scalar for consumers that expect one
        # string. Slug lookup is the fallback when the family has no
        # member-derived categories at all.
        bench_cat_lists = [b.get("categories") or [] for b in all_benches]
        flat_cats = [c for lst in bench_cat_lists for c in lst]
        union_cats = sorted(set(flat_cats))
        if union_cats:
            ordered = [c for c, _ in Counter(flat_cats).most_common()]
        else:
            ordered = list(lookup_categories(family_display))
            union_cats = sorted(set(ordered))
        family_category = (
            primary_category(ordered)
            or FAMILY_CATEGORY.get(fkey, "other")
        )
        family_payload = {
            "key": fkey,
            "display_name": family_display,
            "category": family_category,
            "categories": union_cats or [family_category],
            "tags": {"domains": domains, "languages": langs, "tasks": tasks},
            "evals_count": evals_count,
            "eval_summary_ids": all_eval_ids,
            "provenance_summary": fam_prov,
        }

        # Build composite blocks for any composite_key != None
        composites_out = []
        for ckey, benches in sorted(comp_groups.items(),
                                    key=lambda kv: (kv[0] is None, kv[0] or "")):
            if ckey is None:
                continue
            primary_bench_key = next((b["key"] for b in benches if b["is_overall"]), None)
            for b in benches:
                b["is_primary"] = (b["key"] == primary_bench_key)
            comp_display = titleize(ckey, reg)
            # Composite categories: union of member-benchmark categories
            # (matching the family-level rule). Mode is used as the scalar
            # `category` for backward compat.
            comp_flat = [c for b in benches for c in (b.get("categories") or [])]
            if comp_flat:
                comp_ordered = [c for c, _ in Counter(comp_flat).most_common()]
                comp_union = sorted(set(comp_flat))
            else:
                comp_ordered = lookup_categories(comp_display)
                comp_union = sorted(set(comp_ordered))
            comp_category = (primary_category(comp_ordered)
                             or FAMILY_CATEGORY.get(ckey, family_payload["category"]))
            composites_out.append({
                "key": ckey,
                "display_name": comp_display,
                "category": comp_category,
                "categories": comp_union or [comp_category],
                "tags": {
                    "domains": sorted({d for b in benches for d in b["tags"]["domains"]}),
                    "languages": sorted({l for b in benches for l in b["tags"]["languages"]}),
                    "tasks": sorted({t for b in benches for t in b["tags"]["tasks"]}),
                },
                "summary_eval_ids": [b["key"] for b in benches if b["is_overall"]],
                "primary_benchmark_key": primary_bench_key,
                "benchmarks": benches,
            })
        direct_benches = comp_groups.get(None, [])

        # Decide layout for the family-direct benches (those not in any composite)
        if composites_out and not direct_benches:
            family_payload["composites"] = composites_out
        elif composites_out and direct_benches:
            family_payload["composites"] = composites_out
            # Has overall? → also a composite. No overall → benchmarks[]
            if any(b["is_overall"] for b in direct_benches):
                primary_key = next((b["key"] for b in direct_benches if b["is_overall"]), None)
                for b in direct_benches:
                    b["is_primary"] = (b["key"] == primary_key)
                family_payload["composites"].insert(0, {
                    "key": fkey,
                    "display_name": titleize(fkey, reg),
                    "category": family_payload["category"],
                    "tags": family_payload["tags"],
                    "summary_eval_ids": [b["key"] for b in direct_benches if b["is_overall"]],
                    "primary_benchmark_key": primary_key,
                    "benchmarks": direct_benches,
                })
            else:
                # Apply explicit primary override even without an "overall"
                override = FAMILY_PRIMARY_OVERRIDE.get(fkey)
                if override:
                    for b in direct_benches:
                        b["is_primary"] = (b["key"] == override)
                    family_payload["primary_benchmark_key"] = override
                family_payload["benchmarks"] = direct_benches
        else:
            # No explicit composites — pure direct family
            n = len(direct_benches)
            if n == 1 and direct_benches[0]["is_overall"]:
                direct_benches[0]["is_primary"] = True
                family_payload["standalone_benchmarks"] = direct_benches
            elif any(b["is_overall"] for b in direct_benches):
                primary_key = next((b["key"] for b in direct_benches if b["is_overall"]), None)
                for b in direct_benches:
                    b["is_primary"] = (b["key"] == primary_key)
                family_payload["composites"] = [{
                    "key": fkey,
                    "display_name": titleize(fkey, reg),
                    "category": family_payload["category"],
                    "tags": family_payload["tags"],
                    "summary_eval_ids": [b["key"] for b in direct_benches if b["is_overall"]],
                    "primary_benchmark_key": primary_key,
                    "benchmarks": direct_benches,
                }]
            else:
                # No overall → flat benchmarks[]. Apply primary override.
                override = FAMILY_PRIMARY_OVERRIDE.get(fkey)
                if override:
                    for b in direct_benches:
                        b["is_primary"] = (b["key"] == override)
                    family_payload["primary_benchmark_key"] = override
                family_payload["benchmarks"] = direct_benches

        out_families.append(family_payload)

    # Re-build a flat families view for downstream use (benchmark_index)
    families: dict[str, dict] = {f["key"]: {"key": f["key"], "benchmarks": []}
                                 for f in out_families}
    for fp in out_families:
        all_b = list(fp.get("standalone_benchmarks", []))
        all_b.extend(fp.get("benchmarks", []))
        for c in fp.get("composites", []):
            all_b.extend(c["benchmarks"])
        families[fp["key"]]["benchmarks"] = all_b

    # Pass 4: benchmark_index (spec §8.4–8.5)
    print("Building benchmark_index…", file=sys.stderr)
    by_canonical: dict[str, list] = defaultdict(list)
    seen_keys: set[tuple[str, str, str]] = set()  # (cid, family, bench) dedup

    def add_appearance(cid: str, fkey: str, b: dict, level: str,
                       metric_key: str | None = None) -> None:
        key = (cid, fkey, b["key"], level, metric_key or "")
        if key in seen_keys:
            return
        seen_keys.add(key)
        by_canonical[cid].append((b, {
            "family_key": fkey,
            "benchmark_key": b["key"],
            "level": level,  # "benchmark" or "metric"
            "metric_key": metric_key,
            "eval_summary_ids": b["summary_eval_ids"],
            "models_count": b["models_count"],
            "results_count": b["results_count"],
            "is_canonical_home": (cid == fkey or b["is_overall"] and cid == b["key"]),
        }))

    for fkey, fam in families.items():
        for b in fam["benchmarks"]:
            # Benchmark-level: own canonical id (or its key)
            cid = b["underlying_canonical_id"] or b["key"]
            if cid and resolve_canonical_strict(reg, cid):
                add_appearance(cid, fkey, b, "benchmark")
            # Metric-level: a metric whose name slugs to a canonical benchmark
            # (and isn't a generic word) — surfaces e.g. AIME 2025 as a metric
            # under AA's "AIME" benchmark cross-linking to llm-stats's AIME 2025.
            for sl in b["slices"]:
                for m in sl["metrics"]:
                    mc_cid = resolve_canonical_strict(reg, m["metric_name"])
                    if mc_cid and mc_cid != cid:
                        add_appearance(mc_cid, fkey, b, "metric", m["metric_key"])

    benchmark_index = []
    for cid, entries in sorted(by_canonical.items()):
        # Cross-suite means appearing in 2+ DISTINCT families. Same-family
        # splits (e.g. apex-agents corporate-law / corporate-lawyer / etc.)
        # are not cross-suite — skip them.
        distinct_families = {a["family_key"] for _, a in entries}
        if len(distinct_families) < 2:
            continue
        appearances = [a for _, a in entries]
        benchmark_index.append({
            "key": cid,
            "display_name": reg["display_name"].get(cid, titleize(cid)),
            "appearances": appearances,
        })

    return {
        "schema_version": "v2.hierarchy.1",
        "generated_from": "evaleval/EEE_datastore",
        "stats": {
            "family_count": len(out_families),
            "benchmark_count": sum(len(f.get("standalone_benchmarks", []))
                                   + sum(len(c["benchmarks"]) for c in f.get("composites", []))
                                   for f in out_families),
            "record_count": record_count,
            "folder_count": len(folder_count),
        },
        "families": out_families,
        "benchmark_index": benchmark_index,
    }


def _assign_evaluation_ids(payload: dict) -> None:
    """Stamp each leaf benchmark with `evaluation_id = url_encode(
    composite_slug + '/' + bench_key)` and use that as its
    `summary_eval_ids[0]`. Same id flows into `hierarchy_evals.parquet`
    so the frontend has a guaranteed match by construction — no
    dependency on warehouse evals_view.parquet identity scheme.

    The same id propagates into:
      - benchmark.summary_eval_ids
      - family.eval_summary_ids (union)
      - benchmark_index[*].appearances[*].eval_summary_ids
    """
    from urllib.parse import quote

    def _eid(composite_slug: str, bench_key: str) -> str:
        return quote(f"{composite_slug}/{bench_key}", safe="")

    def _process(composite_slug: str, benches: list[dict]) -> None:
        for b in benches:
            eid = _eid(composite_slug, b["key"])
            b["evaluation_id"] = eid
            b["composite_slug"] = composite_slug
            b["summary_eval_ids"] = [eid]

    for fam in payload.get("families", []):
        for b in (fam.get("standalone_benchmarks") or []):
            _process(fam["key"], [b])
        for b in (fam.get("benchmarks") or []):
            _process(fam["key"], [b])
        for c in fam.get("composites") or []:
            _process(c["key"], c.get("benchmarks") or [])

    # Re-roll family.eval_summary_ids
    for fam in payload.get("families", []):
        all_ids: set[str] = set()
        for stream in ("standalone_benchmarks", "benchmarks"):
            for b in fam.get(stream) or []:
                all_ids.update(b.get("summary_eval_ids") or [])
        for c in fam.get("composites") or []:
            for b in c.get("benchmarks") or []:
                all_ids.update(b.get("summary_eval_ids") or [])
        fam["eval_summary_ids"] = sorted(all_ids)

    # Re-roll benchmark_index appearances
    for entry in payload.get("benchmark_index", []):
        for app in entry.get("appearances", []):
            fam_key = app["family_key"]
            bench_key = app["benchmark_key"]
            fam = next((f for f in payload["families"]
                        if f["key"] == fam_key), None)
            if fam is None:
                continue
            for stream in ("standalone_benchmarks", "benchmarks"):
                for b in fam.get(stream) or []:
                    if b["key"] == bench_key:
                        app["eval_summary_ids"] = list(b.get("summary_eval_ids") or [])
                        break
            for c in fam.get("composites") or []:
                for b in c.get("benchmarks") or []:
                    if b["key"] == bench_key:
                        app["eval_summary_ids"] = list(b.get("summary_eval_ids") or [])
                        break


def _flatten_for_parquet(payload: dict) -> list[dict]:
    """Yield one row per leaf benchmark with the navigation/structure
    fields the frontend needs. Joins on `evaluation_id` map back to
    hierarchy.json by construction.

    Per-model leaderboard rows live in fact_results.parquet — not
    duplicated here. This file is a navigation index, not a fact table."""
    rows: list[dict] = []
    for fam in payload.get("families", []):
        family_key = fam["key"]
        family_display = fam.get("display_name") or family_key
        family_category = fam.get("category")
        family_categories = list(fam.get("categories") or [])

        def _walk(composite_slug: str, composite_display: str,
                  benches: list[dict]) -> None:
            for b in benches:
                metric_names: list[str] = []
                for sl in b.get("slices") or []:
                    for m in sl.get("metrics") or []:
                        nm = m.get("metric_name") or m.get("metric_key")
                        if nm and nm not in metric_names:
                            metric_names.append(nm)
                rows.append({
                    "evaluation_id":          b.get("evaluation_id"),
                    "composite_slug":         composite_slug,
                    "composite_display_name": composite_display,
                    "benchmark_id":           b["key"],
                    "benchmark_display_name": b.get("raw_display_name") or b["key"],
                    "family_id":              family_key,
                    "family_display_name":    family_display,
                    "category":               b.get("category") or family_category,
                    "categories":             list(b.get("categories") or family_categories),
                    "is_overall":             bool(b.get("is_overall")),
                    "is_primary":             bool(b.get("is_primary")),
                    "is_slice":               False,
                    "primary_metric_key":     b.get("primary_metric_key"),
                    "underlying_canonical_id": b.get("underlying_canonical_id"),
                    "models_count":           int(b.get("models_count") or 0),
                    "results_count":          int(b.get("results_count") or 0),
                    "metric_keys":            list(b.get("metric_keys") or []),
                    "metric_names":           metric_names,
                    "tag_domains":            list((b.get("tags") or {}).get("domains") or []),
                    "tag_languages":          list((b.get("tags") or {}).get("languages") or []),
                    "tag_tasks":              list((b.get("tags") or {}).get("tasks") or []),
                    "summary_eval_ids":       list(b.get("summary_eval_ids") or []),
                    "first_party_results":    int((b.get("provenance_summary") or {})
                                                  .get("first_party_results") or 0),
                    "third_party_results":    int((b.get("provenance_summary") or {})
                                                  .get("third_party_results") or 0),
                    "total_results":          int((b.get("provenance_summary") or {})
                                                  .get("total_results") or 0),
                })

        for b in (fam.get("standalone_benchmarks") or []):
            _walk(family_key, family_display, [b])
        for b in (fam.get("benchmarks") or []):
            _walk(family_key, family_display, [b])
        for c in fam.get("composites") or []:
            _walk(c["key"], c.get("display_name") or c["key"],
                  c.get("benchmarks") or [])
    return rows


def _write_hierarchy_evals_parquet(payload: dict, out_dir: Path) -> Path:
    """Write `hierarchy_evals.parquet` — navigation-grain rows keyed on
    `evaluation_id`. Matches the ids in hierarchy.json by construction."""
    rows = _flatten_for_parquet(payload)
    df = pd.DataFrame(rows)
    path = out_dir / "hierarchy_evals.parquet"
    df.to_parquet(path, compression="zstd", index=False)
    return path


def write_hierarchy_v2(
    out_dir: Path,
    snapshot_meta: dict,
    *,
    con: Any | None = None,
    eee_dir: Path | None = None,
    registry_dir: Path | None = None,
    override_dir: Path | None = None,
) -> Path:
    """Stage J entry point. Builds the hierarchy from the EEE datastore
    using the v2 spec and writes:
      - `<out_dir>/hierarchy.json` — multi-level navigation tree
      - `<out_dir>/hierarchy_evals.parquet` — one row per leaf benchmark,
        keyed by `evaluation_id` (= url_encode(composite_slug/bench_key)).
        Same id appears in hierarchy.json's `summary_eval_ids` so the
        frontend has a guaranteed match by construction.

    `con` is accepted for API symmetry with the warehouse-driven sidecar
    emitters but is unused — v2 is self-contained, doesn't depend on
    warehouse evals_view.

    `eee_dir` / `registry_dir` / `override_dir` default to the same
    `.cache/...` paths the standalone build uses."""
    del con  # v2 is self-contained; warehouse evals_view is no longer the
             # source of truth for hierarchy ids.
    _configure_paths(eee_dir=eee_dir, registry_dir=registry_dir,
                     override_dir=override_dir)
    payload = build()
    payload["generated_at"] = snapshot_meta.get("snapshot_id")
    _assign_evaluation_ids(payload)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hierarchy.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    parquet_path = _write_hierarchy_evals_parquet(payload, out_dir)
    print(f"hierarchy_v2: wrote {parquet_path} "
          f"({parquet_path.stat().st_size // 1024} KB, "
          f"{len(_flatten_for_parquet(payload))} rows)", file=sys.stderr)
    size_kb = path.stat().st_size // 1024
    print(f"hierarchy_v2: wrote {path} ({size_kb} KB) — "
          f"families={payload['stats']['family_count']}  "
          f"benchmarks={payload['stats']['benchmark_count']}  "
          f"records={payload['stats']['record_count']}", file=sys.stderr)
    return path


def main() -> int:
    """Standalone entry point — preserved for ad-hoc local runs."""
    out_dir = _DEFAULT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "hierarchy_v2.json"
    payload = build()
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    size_kb = out_json.stat().st_size // 1024
    print(f"Wrote {out_json} ({size_kb} KB)", file=sys.stderr)
    print(f"  families={payload['stats']['family_count']}  "
          f"benchmarks={payload['stats']['benchmark_count']}  "
          f"records={payload['stats']['record_count']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

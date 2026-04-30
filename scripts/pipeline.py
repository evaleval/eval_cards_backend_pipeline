import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, snapshot_download

from scripts import registry, signals
from scripts.helpers.benchmark_identity import (
    canonical_benchmark_family_key,
    is_agentic_benchmark_name,
    normalize_benchmark_key as shared_normalize_benchmark_key,
)


PRODUCTION_DATASET_REPO = "evaleval/card_backend"
# `CARD_BACKEND_OUTPUT_REPO` lets local/CI runs target a different upload
# destination. The `resolve_upload_target` guard refuses unintended writes
# to `evaleval/card_backend` from local shells (where `HF_TOKEN` is often
# auto-loaded from a profile); `CARD_BACKEND_ALLOW_PRODUCTION=1` is the
# explicit opt-in for an intentional manual local prod push. CI runs
# (`GITHUB_ACTIONS=true`) deploy to production by default — the gate
# there is owner PR review at merge time, not an env flag.
#
# Both `DATASET_REPO` and `DATASET_RESOLVE_BASE` are resolved at *import*
# time. Tests that need to flip the env after import should call
# `reload_dataset_target()` to re-bind the module-level constants.
EEE_DATASET_REPO = "evaleval/EEE_datastore"
BENCHMARK_METADATA_DATASET_REPO = "evaleval/auto-benchmarkcards"
EEE_DATASET_RAW_BASE = f"https://huggingface.co/datasets/{EEE_DATASET_REPO}/raw/main"


def _resolve_dataset_repo() -> str:
    return os.environ.get("CARD_BACKEND_OUTPUT_REPO") or PRODUCTION_DATASET_REPO


DATASET_REPO = _resolve_dataset_repo()
DATASET_RESOLVE_BASE = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"


def reload_dataset_target() -> None:
    """Re-bind module-level dataset constants from the current env.

    Used by tests + CLI flags that change ``CARD_BACKEND_OUTPUT_REPO``
    after import. Without this, validators / README generators continue
    to reference the import-time value and silently produce stale URLs.
    """
    global DATASET_REPO, DATASET_RESOLVE_BASE
    DATASET_REPO = _resolve_dataset_repo()
    DATASET_RESOLVE_BASE = (
        f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"
    )
CONFIG_VERSION = 1
OUTPUT_DIR = Path("output")
DEFAULT_LOCAL_DATASET_DIR = ".cache/eee_datastore"
DEFAULT_LOCAL_BENCHMARK_METADATA_DIR = ".cache/auto_benchmarkcards"
DEFAULT_METRIC_REGISTRY_PATH = Path("registry/metric_looking_strings.json")
FILE_READ_MAX_RETRIES = 5
FILE_READ_RETRY_DELAY_SEC = 1.5
VERSION_SUFFIX_REGEX = re.compile(
    r"^(.*?)-((?:19|20)\d{6}|(?:19|20)\d{2}-\d{2}-\d{2})(?:-(.+))?$"
)
PASS_AT_REGEX = re.compile(r"pass\s*@?\s*(\d+)", flags=re.IGNORECASE)
PASS_AT_EXACT_REGEX = re.compile(r"^\s*pass\s*@?\s*(\d+)\s*$", flags=re.IGNORECASE)
EVAL_DESCRIPTION_METRIC_REGEX = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 @%+./_-]*?)\s+on\s+(.+?)\s*$"
)
BENCHMARK_DEFAULT_METRICS = {
    "global_mmlu_lite": ("Accuracy", "accuracy"),
}
# Leaf-key values that indicate a score is a summary across all sub-benchmarks
# rather than an independent benchmark of its own.
SUMMARY_SCORE_LEAF_KEYS = {"overall", "aggregate", "total", "all"}
COMMON_LANGUAGE_SUBSET_KEYS = {
    "albanian",
    "arabic",
    "ar",
    "bengali",
    "bn",
    "burmese",
    "chinese",
    "cy",
    "english",
    "en",
    "french",
    "fr",
    "german",
    "de",
    "hindi",
    "hi",
    "id",
    "indonesian",
    "italian",
    "it",
    "japanese",
    "ja",
    "korean",
    "ko",
    "my",
    "portuguese",
    "pt",
    "spanish",
    "es",
    "sq",
    "sw",
    "swahili",
    "welsh",
    "yo",
    "yoruba",
    "zh",
}
BUILTIN_METRIC_DISPLAY_MAP = {
    "accuracy": "Accuracy",
    "exact_match": "Exact Match",
    "win_rate": "Win Rate",
    "mean_win_rate": "Mean Win Rate",
    "average_attempts": "Average Attempts",
    "average_latency_ms": "Average Latency (ms)",
    "latency_mean": "Latency Mean",
    "latency_std": "Latency Standard Deviation",
    "latency_p95": "Latency 95th Percentile",
    "rank": "Rank",
    "overall_accuracy": "Overall Accuracy",
    "total_cost": "Total Cost",
    "cost_per_task": "Cost per Task",
    "cost_per_100_calls": "Cost per 100 Calls",
    "elo": "Elo Rating",
    "score": "Score",
    "arc_score": "ARC Score",
    "mean_score": "Mean Score",
    "format_sensitivity_stddev": "Format Sensitivity Standard Deviation",
    "format_sensitivity_max_delta": "Format Sensitivity Max Delta",
}
PREFERRED_BENCHMARK_DISPLAY_NAMES = {
    "ace": "ACE",
    "apex": "APEX",
    "apex_agents": "APEX Agents",
    "apex_v1": "APEX v1",
    # SWE-PolyBench / Multi-SWE-bench: hardcoded because removing them
    # caused regressions in three separate code paths (family_name,
    # parent_name, AND aggregator-row dataset_name normalization that
    # happens to land on the same normalized key). The fully-clean
    # refactor would route family/parent name derivation through the
    # registry's canonical display_name when canonical_benchmark_id
    # resolves — but that requires per-row registry calls during
    # normalization, which is bigger architectural work. Keeping these
    # explicit until that lands; documented in CLAUDE.md.
    "swe_polybench": "SWE-PolyBench",
    "multi_swe_bench": "Multi-SWE-bench",
}
# Adding ``helm_capabilities`` / ``helm_instruct`` here is unsafe:
# ``canonical_benchmark_display_name`` walks every candidate and fires
# PREFERRED on any match, so an entry for ``helm_capabilities`` would
# trigger whenever that key appears as ANY candidate, including
# sub-benchmark rows whose family_key happens to be ``helm_capabilities``
# (e.g. ``helm_capabilities_mmlu_pro``). Override clobbers leaf_name
# for every child eval_summary in those families. For SWE-PolyBench /
# Multi-SWE-bench the same over-firing is desirable (all children
# should display the parent name); for HELM each child has its own
# distinct name and must be preserved. The HELM-duplicate display
# issue requires a more surgical mechanism (or the deferred
# family-collapse work) — left visible until then.
METRIC_REGISTRY_ALIAS_LOOKUP: dict[str, str] = {}
METRIC_REGISTRY_ENTRIES: dict[str, dict] = {}
METRIC_SUFFIX_ALIAS_CANDIDATES: list[str] = []
KNOWN_TOP_LEVEL_KEYS = {
    "schema_version",
    "evaluation_id",
    "retrieved_timestamp",
    "source_metadata",
    "eval_library",
    "model_info",
    "evaluation_results",
    "detailed_evaluation_results",
}


def as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_benchmark_key(value: Any) -> str:
    return shared_normalize_benchmark_key(value)


# Tokens that lose distinguishing characters when fed straight through
# normalize_benchmark_key. ``c++`` and ``c`` both collapse to ``c`` because
# normalize strips ``+``; same hazard for ``c#`` vs ``c``. Pre-mapping these
# unambiguous canonical forms lets distinct slices stay distinct without
# changing the global benchmark-key normalizer (which has many callers).
LANGUAGE_TOKEN_ALIASES = {
    "c++": "cpp",
    "c#": "csharp",
    "f#": "fsharp",
    ".net": "dotnet",
    "objective-c": "objective_c",
}


def normalize_subset_key(value: Any) -> str:
    """``normalize_benchmark_key`` for slice / subset tokens, with a
    punctuation-aware pre-pass so c++ ≠ c at the key level. Display names
    still flow through ``humanize_token_key`` against the original token,
    so the user sees ``C++`` even though the lookup key is ``cpp``."""
    text = as_string(value).strip().lower()
    if not text:
        return ""
    if text in LANGUAGE_TOKEN_ALIASES:
        text = LANGUAGE_TOKEN_ALIASES[text]
    return normalize_benchmark_key(text)


def slugify(value: Any) -> str:
    return normalize_benchmark_key(value)


def sanitize_slug_input(value: Any) -> str:
    text = as_string(value)
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r"\\x[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\\u[0-9a-fA-F]{4}", "", text)
    return text


def ensure_safe_slug_segment(value: Any) -> str:
    cleaned = as_string(value).strip()
    if not cleaned:
        return "unknown"
    if re.match(r"^x00", cleaned, flags=re.IGNORECASE):
        trimmed = re.sub(r"^x0+", "", cleaned, flags=re.IGNORECASE)
        return f"safe-{trimmed or 'unknown'}"
    return cleaned


def slugify_developer(value: Any) -> str:
    cleaned = sanitize_slug_input(value).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return ensure_safe_slug_segment(cleaned)


def slugify_model_segment(value: Any) -> str:
    cleaned = sanitize_slug_input(value).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return ensure_safe_slug_segment(cleaned)


_V_VERSION_TOKEN_RE = re.compile(r"^v\d", re.IGNORECASE)


def humanize_slug(value: Any) -> str:
    parts = [p for p in re.split(r"[-_/]+", as_string(value)) if p]
    out = []
    for part in parts:
        # v/V version tokens (e.g. `v3`, `v0.1`) — keep `v` lowercase per
        # spec 01 Group B. This rule predates the digit-uppercase rule
        # below; without it, model names like `Deepseek v3` regress to
        # `Deepseek V3` (1,253 cards in the production corpus).
        if _V_VERSION_TOKEN_RE.match(part):
            out.append("v" + part[1:])
            continue
        if len(part) <= 3 and any(c.isdigit() for c in part):
            out.append(part.upper())
        else:
            out.append(part[:1].upper() + part[1:])
    return " ".join(out)


def normalize_setup_alias_qualifier(value: Any) -> str:
    return as_string(value).strip().lower().replace("_", "-").replace(" ", "-")


def is_setup_alias_qualifier(value: Any) -> bool:
    normalized = normalize_setup_alias_qualifier(value)
    return bool(
        normalized
        in {
            "prompt",
            "fc",
            "function-calling",
            "prompt-thinking",
            "fc-thinking",
            "function-calling-thinking",
            "thinking",
        }
        or normalized.startswith("thinking-")
        or normalized.startswith("prompt-thinking-")
        or normalized.startswith("fc-thinking-")
        or normalized.startswith("function-calling-thinking-")
    )


def strip_setup_alias_suffix(value: Any) -> tuple[str, str] | None:
    normalized = slugify_model_segment(value)
    patterns = [
        re.compile(
            r"^(.*?)-((?:prompt|fc|function-calling)-thinking(?:-[a-z0-9.]+)*)$"
        ),
        re.compile(r"^(.*?)-(thinking(?:-[a-z0-9.]+)*)$"),
        re.compile(r"^(.*?)-((?:prompt|fc|function-calling))$"),
    ]

    for pattern in patterns:
        match = pattern.match(normalized)
        if match and match.group(1):
            return match.group(1), match.group(2)

    return None


def parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = as_string(value).strip()
    if not text:
        return None

    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_params_billions_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if numeric > 0 else None

    text = as_string(value).strip().lower()
    if not text:
        return None

    text = text.replace(",", "")
    scale = 1.0
    if "trillion" in text or re.search(r"\bt\b", text):
        scale = 1000.0
    elif "million" in text or re.search(r"\bm\b", text):
        scale = 0.001
    elif "thousand" in text or re.search(r"\bk\b", text):
        scale = 0.000001
    elif "billion" in text or re.search(r"\bb\b", text):
        scale = 1.0

    number_match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not number_match:
        return None
    try:
        numeric = float(number_match.group(0)) * scale
    except Exception:
        return None
    return numeric if numeric > 0 else None


def infer_params_billions_from_name(*values: Any) -> float | None:
    patterns = [
        re.compile(
            r"(\d+(?:\.\d+)?)\s*[x*]\s*(\d+(?:\.\d+)?)\s*b", flags=re.IGNORECASE
        ),
        re.compile(r"(\d+(?:\.\d+)?)\s*b", flags=re.IGNORECASE),
        re.compile(r"(\d+(?:\.\d+)?)\s*m(?:\b|illion)", flags=re.IGNORECASE),
    ]

    for value in values:
        text = as_string(value)
        if not text:
            continue

        mo = patterns[0].search(text)
        if mo:
            left = parse_float(mo.group(1))
            right = parse_float(mo.group(2))
            if left and right:
                inferred = left * right
                if inferred > 0:
                    return inferred

        mo = patterns[1].search(text)
        if mo:
            inferred = parse_float(mo.group(1))
            if inferred and inferred > 0:
                return inferred

        mo = patterns[2].search(text)
        if mo:
            inferred_m = parse_float(mo.group(1))
            if inferred_m and inferred_m > 0:
                return inferred_m / 1000.0

    return None


def derive_model_params_billions(model_info: dict) -> float | None:
    additional = model_info.get("additional_details")
    additional_details = additional if isinstance(additional, dict) else {}

    candidate_paths = [
        model_info.get("params_billions"),
        model_info.get("parameters_billions"),
        model_info.get("parameter_count_billions"),
        model_info.get("parameter_count"),
        model_info.get("num_parameters"),
        additional_details.get("params_billions"),
        additional_details.get("parameters_billions"),
        additional_details.get("parameter_count_billions"),
        additional_details.get("parameter_count"),
        additional_details.get("num_parameters"),
        additional_details.get("parameters"),
        additional_details.get("model_size"),
    ]

    for candidate in candidate_paths:
        parsed = parse_params_billions_value(candidate)
        if parsed is not None:
            return parsed

    return infer_params_billions_from_name(model_info.get("name"), model_info.get("id"))


def iso_from_epoch_string(value: Any) -> str | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return (
        datetime.fromtimestamp(numeric, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def max_iso(left: str | None, right: str | None) -> str | None:
    if not left:
        return right
    if not right:
        return left
    return left if left > right else right


def load_benchmark_metadata(
    metadata_cache_dir: str,
) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    return load_benchmark_metadata_from_dir(Path(metadata_cache_dir))


def compact_benchmark_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", as_string(value).lower())


def load_benchmark_metadata_from_dir(
    root_dir: Path,
) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    cards = []
    lookup: dict[str, dict] = {}
    flat_map: dict[str, dict] = {}

    if not root_dir.exists():
        return cards, lookup, flat_map

    flat_metadata_path = root_dir / "benchmark-metadata.json"
    if flat_metadata_path.exists():
        parsed = json.loads(flat_metadata_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            for raw_key, raw_card in parsed.items():
                if not isinstance(raw_card, dict) or not isinstance(
                    raw_card.get("benchmark_details"), dict
                ):
                    continue
                card_id = normalize_benchmark_key(raw_key)
                if card_id:
                    flat_map[card_id] = raw_card

    cards_dir = root_dir / "cards"
    if cards_dir.exists():
        for file_path in sorted(cards_dir.glob("*.json")):
            parsed = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict) and isinstance(
                parsed.get("benchmark_card"), dict
            ):
                card = parsed["benchmark_card"]
                base_name = file_path.stem.replace("benchmark_card_", "")
            elif isinstance(parsed, dict) and isinstance(
                parsed.get("benchmark_details"), dict
            ):
                card = parsed
                base_name = file_path.stem
            else:
                continue
            card_id = normalize_benchmark_key(base_name)
            if card_id and card_id not in flat_map:
                flat_map[card_id] = card

    for card_id, card in sorted(flat_map.items()):
        keys = candidate_benchmark_keys(
            card_id, card.get("benchmark_details", {}).get("name")
        )
        cards.append(
            {
                "file_name": f"{card_id}.json",
                "base_name": card_id,
                "card": card,
                "keys": keys,
            }
        )
        for key in keys:
            lookup[key] = card

    return cards, lookup, flat_map


def has_cached_benchmark_metadata(cards_dir: Path, flat_metadata_path: Path) -> bool:
    return flat_metadata_path.exists() or (
        cards_dir.exists() and any(cards_dir.glob("*.json"))
    )


def ensure_local_benchmark_metadata_snapshot(
    local_metadata_dir: str, hf_token: str | None, force_refresh: bool
) -> str | None:
    target_dir = Path(local_metadata_dir).resolve()
    cards_dir = target_dir / "cards"
    flat_metadata_path = target_dir / "benchmark-metadata.json"

    if force_refresh and target_dir.exists():
        shutil.rmtree(target_dir)

    if has_cached_benchmark_metadata(cards_dir, flat_metadata_path):
        return str(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=BENCHMARK_METADATA_DATASET_REPO,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=["benchmark-metadata.json", "cards/**"],
            token=hf_token,
        )
    except Exception:
        if has_cached_benchmark_metadata(cards_dir, flat_metadata_path):
            return str(target_dir)
        return None

    return str(target_dir)


def candidate_benchmark_keys(*values: Any) -> list[str]:
    keys = set()
    for value in values:
        text = as_string(value)
        if not text:
            continue
        normalized = normalize_benchmark_key(text)
        stripped = normalize_benchmark_key(
            re.sub(r"^benchmark_card_", "", text, flags=re.IGNORECASE)
        )
        separated = normalize_benchmark_key(re.sub(r"[_-]+", " ", text))
        compact = compact_benchmark_key(text)
        keys.add(normalized)
        keys.add(stripped)
        keys.add(separated)
        keys.add(compact)
        family_key = canonical_benchmark_family_key(text)
        if family_key:
            keys.add(family_key)
    return [k for k in keys if k]


def lookup_benchmark_card(
    metadata_lookup: dict[str, dict], *values: Any
) -> dict | None:
    for key in candidate_benchmark_keys(*values):
        if key in metadata_lookup:
            return metadata_lookup[key]
    return None


def iter_matching_benchmark_cards(
    metadata_lookup: dict[str, dict], *values: Any
) -> list[dict]:
    matches: list[dict] = []
    seen: set[str] = set()
    for key in candidate_benchmark_keys(*values):
        card = metadata_lookup.get(key)
        if not card:
            continue
        details = card.get("benchmark_details") if isinstance(card, dict) else {}
        card_id = normalize_benchmark_key((details or {}).get("name")) or key
        if card_id in seen:
            continue
        seen.add(card_id)
        matches.append(card)
    return matches


def lookup_benchmark_card_for_parent(
    metadata_lookup: dict[str, dict],
    *values: Any,
    aux_values: tuple[Any, ...] = (),
    parent_values: tuple[Any, ...] = (),
) -> dict | None:
    """Resolve the benchmark card that best describes the requested benchmark.

    `*values` are the **primary** identifiers of the benchmark we're looking
    up (its leaf name and key). `aux_values` are auxiliary identifiers used
    to cast a wider net when matching (e.g. the parent suite name, family
    key, dataset_name on a per-record source) — they help us *find*
    candidates but on their own they don't make a candidate the right answer.
    `parent_values` carry the same role as before: they describe the
    enclosing parent and are used to pick the most compatible child variant
    when a card has multiple `appears_in` entries.

    A returned card is the one whose own name matches a primary identifier
    ("self"). If none qualifies, we fall back to a child card whose
    `appears_in` matches one of our search keys (this is what allows
    leaf-benchmark lookups to find variant-specific cards). We deliberately
    do **not** fall back to a card that matched only via `aux_values`
    without also being a recorded child of the search — returning such a
    card would mean handing the requester their *parent's* card (e.g. the
    helm_classic suite card when they asked for XSUM).
    """

    # Phase 1: direct lookup by the primary identifiers. metadata_lookup is
    # keyed by normalized card identifiers (file basename / card name), so a
    # match here means the card *is* the thing we asked for. This is the most
    # reliable signal — when it succeeds, return it immediately without
    # consulting aux_values, which would otherwise drag in parent/family cards.
    direct = lookup_benchmark_card(metadata_lookup, *values)
    if direct is not None:
        return direct

    # Phase 2: cast a wider net via aux_values. The candidates picked up here
    # were matched via family or dataset identifiers, so they're typically
    # variant siblings/children. We accept them only when their `appears_in`
    # explicitly lists one of our search keys — otherwise it's an
    # ancestor of the leaf and returning it would mislead.
    all_search_values = (*values, *aux_values)
    candidates = iter_matching_benchmark_cards(metadata_lookup, *all_search_values)
    if not candidates:
        return None

    search_keys = set(candidate_benchmark_keys(*all_search_values))
    parent_keys = set(candidate_benchmark_keys(*parent_values))

    child_cards: list[tuple[dict, set[str]]] = []
    for card in candidates:
        details = card.get("benchmark_details") if isinstance(card, dict) else {}
        appears_in = as_string_list((details or {}).get("appears_in"))
        if not appears_in:
            continue  # skip standalone cards picked up only via aux/parent keys
        appears_in_keys = set(candidate_benchmark_keys(*appears_in))
        if appears_in_keys & search_keys:
            child_cards.append((card, appears_in_keys))

    if not parent_keys:
        return child_cards[0][0] if child_cards else None

    compatible = [c for c, appears_in_keys in child_cards if parent_keys & appears_in_keys]
    if compatible:
        return compatible[0]
    if child_cards:
        return child_cards[0][0]
    return None


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [as_string(item) for item in value if as_string(item)]
    text = as_string(value)
    return [text] if text else []


def extract_benchmark_tags(benchmark_card: dict | None) -> dict:
    """Extract structured tags from a benchmark card for frontend filtering."""
    if not benchmark_card:
        return {"domains": [], "languages": [], "tasks": []}
    details = benchmark_card.get("benchmark_details") or {}
    purpose = benchmark_card.get("purpose_and_intended_users") or {}
    return {
        "domains": as_string_list(details.get("domains")),
        "languages": as_string_list(details.get("languages")),
        "tasks": as_string_list(purpose.get("tasks")),
    }


def humanize_token_key(value: Any) -> str:
    text = re.sub(r"[._/]+", " ", as_string(value))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return humanize_slug(text)


def canonical_benchmark_display_name(*values: Any, fallback: Any = None) -> str:
    candidates = [
        as_string(value).strip() for value in values if as_string(value).strip()
    ]

    for candidate in candidates:
        preferred = PREFERRED_BENCHMARK_DISPLAY_NAMES.get(
            normalize_benchmark_key(candidate)
        )
        if preferred:
            return preferred

    for candidate in candidates:
        if any(char.isupper() for char in candidate) or " " in candidate:
            return candidate

    fallback_text = as_string(fallback).strip()
    if fallback_text:
        preferred = PREFERRED_BENCHMARK_DISPLAY_NAMES.get(
            normalize_benchmark_key(fallback_text)
        )
        if preferred:
            return preferred
        return humanize_token_key(fallback_text)

    if candidates:
        return humanize_token_key(candidates[0])
    return ""


def join_display_name_parts(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        text = as_string(value).strip()
        if not text:
            continue
        if parts and parts[-1] == text:
            continue
        parts.append(text)
    return " / ".join(parts)


def load_metric_registry(path: Path = DEFAULT_METRIC_REGISTRY_PATH) -> None:
    global METRIC_REGISTRY_ALIAS_LOOKUP
    global METRIC_REGISTRY_ENTRIES
    global METRIC_SUFFIX_ALIAS_CANDIDATES

    METRIC_REGISTRY_ALIAS_LOOKUP = {}
    METRIC_REGISTRY_ENTRIES = {}
    METRIC_SUFFIX_ALIAS_CANDIDATES = []

    if not path.exists():
        return

    parsed = json.loads(path.read_text(encoding="utf-8"))
    entries = parsed.get("entries") if isinstance(parsed, dict) else None
    alias_map = parsed.get("alias_to_normalized") if isinstance(parsed, dict) else None

    if isinstance(entries, list):
        for entry in entries:
            normalized = as_string((entry or {}).get("normalized")).strip()
            if normalized:
                METRIC_REGISTRY_ENTRIES[normalized] = entry

    if isinstance(alias_map, dict):
        for key, value in alias_map.items():
            norm_key = as_string(key).strip()
            norm_value = as_string(value).strip()
            if norm_key and norm_value:
                METRIC_REGISTRY_ALIAS_LOOKUP[norm_key] = norm_value

    candidate_set = set()
    for raw_alias in METRIC_REGISTRY_ALIAS_LOOKUP:
        normalized_alias = normalize_benchmark_key(raw_alias)
        if normalized_alias:
            candidate_set.add(normalized_alias)
    for canonical_key in METRIC_REGISTRY_ENTRIES:
        normalized_alias = normalize_benchmark_key(canonical_key)
        if normalized_alias:
            candidate_set.add(normalized_alias)
    METRIC_SUFFIX_ALIAS_CANDIDATES = sorted(
        candidate_set, key=lambda value: (-len(value.split("_")), -len(value), value)
    )


def humanize_metric_key(value: Any) -> str:
    text = normalize_benchmark_key(value)
    if not text:
        return ""
    pass_match = re.match(r"pass_at_(\d+)$", text)
    if pass_match:
        return f"Pass@{pass_match.group(1)}"

    special = {
        "ast": "AST",
        "kv": "KV",
        "ndcg": "NDCG",
        "arc": "ARC",
        "ifeval": "IFEval",
        "cot": "CoT",
        "bleu": "BLEU",
        "rouge": "ROUGE",
        "elo": "Elo",
        "ms": "(ms)",
        "p95": "95th Percentile",
    }
    parts = []
    for part in text.split("_"):
        if part in special:
            parts.append(special[part])
        elif part.isdigit():
            parts.append(part)
        else:
            parts.append(part[:1].upper() + part[1:])
    label = " ".join(parts).replace(" (ms)", " (ms)")
    return re.sub(r"\s+", " ", label).strip()


def canonicalize_metric_key(value: Any) -> str:
    raw = as_string(value).strip()
    if not raw:
        return ""
    pass_match = PASS_AT_EXACT_REGEX.match(raw)
    if pass_match:
        return f"pass_at_{pass_match.group(1)}"

    candidates = [
        raw,
        normalize_benchmark_key(raw),
        normalize_benchmark_key(raw.split(".")[-1]),
    ]
    for candidate in candidates:
        if candidate and candidate in METRIC_REGISTRY_ALIAS_LOOKUP:
            return METRIC_REGISTRY_ALIAS_LOOKUP[candidate]

    return normalize_benchmark_key(raw.split(".")[-1]) or normalize_benchmark_key(raw)


def strict_metric_alias_lookup(value: Any) -> str:
    raw = as_string(value).strip()
    if not raw:
        return ""
    pass_match = PASS_AT_EXACT_REGEX.match(raw)
    if pass_match:
        return f"pass_at_{pass_match.group(1)}"

    candidates = [
        raw,
        normalize_benchmark_key(raw),
        normalize_benchmark_key(raw.split(".")[-1]),
    ]
    for candidate in candidates:
        if candidate and candidate in METRIC_REGISTRY_ALIAS_LOOKUP:
            return METRIC_REGISTRY_ALIAS_LOOKUP[candidate]
    return ""


def preferred_metric_display(metric_key: str, raw_label: Any = None) -> str:
    if metric_key in METRIC_REGISTRY_ENTRIES:
        display = as_string(
            METRIC_REGISTRY_ENTRIES[metric_key].get("display_name")
        ).strip()
        if display:
            return display
    if metric_key in BUILTIN_METRIC_DISPLAY_MAP:
        return BUILTIN_METRIC_DISPLAY_MAP[metric_key]
    if (
        raw_label
        and canonicalize_metric_key(raw_label) == metric_key
        and normalize_benchmark_key(raw_label) == metric_key
    ):
        return as_string(raw_label).strip()
    return humanize_metric_key(metric_key)


def infer_metric_from_value(
    metric_name: Any = None, metric_id: Any = None
) -> dict | None:
    explicit_id = as_string(metric_id).strip()
    explicit_name = as_string(metric_name).strip()

    if explicit_id:
        metric_key = canonicalize_metric_key(explicit_id) or slugify(explicit_id)
        display = preferred_metric_display(
            metric_key, explicit_name or explicit_id.split(".")[-1]
        )
        return {
            "metric_name": display,
            "metric_id": explicit_id,
            "metric_key": metric_key or "score",
        }

    raw = explicit_name
    if not raw:
        return None

    metric_key = canonicalize_metric_key(raw) or slugify(raw) or "score"
    display = preferred_metric_display(metric_key, explicit_name)
    return {
        "metric_name": display,
        "metric_id": metric_key,
        "metric_key": metric_key,
    }


def infer_metric_from_score_details(result: dict) -> dict | None:
    details = (
        ((result.get("score_details") or {}).get("details") or {})
        if isinstance(result, dict)
        else {}
    )
    if not isinstance(details, dict):
        return None
    tab = as_string(details.get("tab")).strip()
    if not tab:
        return None
    return infer_metric_from_value(metric_name=tab)


def infer_metric_from_benchmark_card(card: dict | None) -> dict | None:
    metrics = (
        (((card or {}).get("methodology") or {}).get("metrics") or [])
        if isinstance(card, dict)
        else []
    )
    if isinstance(metrics, list) and metrics:
        return infer_metric_from_value(metric_name=metrics[0])
    return None


def infer_metric_from_benchmark_defaults(benchmark_key: str) -> dict | None:
    default = BENCHMARK_DEFAULT_METRICS.get(normalize_benchmark_key(benchmark_key))
    if not default:
        return None
    metric_name, metric_id_value = default
    return {
        "metric_name": metric_name,
        "metric_id": metric_id_value,
        "metric_key": normalize_benchmark_key(metric_id_value),
    }


def metric_namespace_component(
    metric_id: str, benchmark_family_key: str
) -> tuple[str | None, str | None]:
    parts = [part for part in re.split(r"[./]+", as_string(metric_id)) if part]
    if len(parts) < 3:
        return None, None
    if normalize_benchmark_key(parts[0]) != normalize_benchmark_key(
        benchmark_family_key
    ):
        return None, None
    component_parts = parts[1:-1]
    if not component_parts:
        return None, None
    component_key = normalize_benchmark_key("_".join(component_parts))
    return humanize_token_key(" ".join(component_parts)), component_key


def split_metric_from_evaluation_description(description: Any) -> dict | None:
    text = as_string(description).strip()
    if not text:
        return None
    match = EVAL_DESCRIPTION_METRIC_REGEX.match(text)
    if not match:
        return None
    return infer_metric_from_value(metric_name=match.group(1))


def split_metric_from_evaluation_name(
    raw_name: Any, benchmark_keys: list[str]
) -> dict | None:
    name = as_string(raw_name).strip()
    if not name:
        return None

    normalized_name = normalize_benchmark_key(name)
    for benchmark_key in benchmark_keys:
        if benchmark_key and normalized_name.startswith(f"{benchmark_key}_"):
            suffix = normalized_name[len(benchmark_key) + 1 :]
            if strict_metric_alias_lookup(suffix):
                maybe_metric = infer_metric_from_value(metric_name=suffix)
                if maybe_metric:
                    return {
                        "component_name": None,
                        "component_key": None,
                        "metric": maybe_metric,
                        "metric_source": "evaluation_name_suffix",
                    }

    raw_tokens = [token for token in re.split(r"[.\s_-]+", name) if token]
    for split_index in range(1, len(raw_tokens)):
        prefix_raw = " ".join(raw_tokens[:split_index]).strip()
        suffix_raw = " ".join(raw_tokens[split_index:]).strip()
        if not suffix_raw:
            continue
        if not strict_metric_alias_lookup(suffix_raw):
            continue
        metric = infer_metric_from_value(metric_name=suffix_raw)
        if not metric:
            continue
        component_key = normalize_benchmark_key(prefix_raw) if prefix_raw else None
        if component_key and component_key in benchmark_keys:
            prefix_raw = ""
            component_key = None
        return {
            "component_name": humanize_token_key(prefix_raw) if prefix_raw else None,
            "component_key": component_key,
            "metric": metric,
            "metric_source": "evaluation_name_suffix",
        }

    direct_metric_key = strict_metric_alias_lookup(name)
    if direct_metric_key:
        metric = infer_metric_from_value(metric_name=name)
        if metric:
            return {
                "component_name": None,
                "component_key": None,
                "metric": metric,
                "metric_source": "evaluation_name",
            }

    for alias_candidate in METRIC_SUFFIX_ALIAS_CANDIDATES:
        if not alias_candidate or not normalized_name.endswith(f"_{alias_candidate}"):
            continue
        prefix = normalized_name[: -(len(alias_candidate) + 1)]
        if not prefix:
            continue
        if not strict_metric_alias_lookup(alias_candidate):
            continue
        metric = infer_metric_from_value(metric_name=alias_candidate)
        if not metric:
            continue
        component_key = normalize_benchmark_key(prefix)
        if component_key in benchmark_keys:
            component_key = None
            component_name = None
        else:
            component_name = humanize_token_key(prefix)
        return {
            "component_name": component_name,
            "component_key": component_key,
            "metric": metric,
            "metric_source": "evaluation_name_suffix",
        }

    return None


def infer_top_level_benchmark_name(benchmark: Any, benchmark_family_name: str) -> str:
    benchmark_key = normalize_benchmark_key(benchmark)
    if benchmark_key.startswith("helm_"):
        suffix = benchmark_key.split("_", 1)[1]
        return canonical_benchmark_display_name(suffix, fallback=suffix)
    if (
        benchmark_family_name
        and normalize_benchmark_key(benchmark_family_name) == benchmark_key
    ):
        return benchmark_family_name
    return canonical_benchmark_display_name(
        benchmark, fallback=benchmark or benchmark_family_name
    )


def infer_subset_slice_from_name(
    name: Any, benchmark: Any
) -> tuple[str | None, str | None]:
    text = as_string(name).strip()
    benchmark_key = normalize_benchmark_key(benchmark)
    if not text or not benchmark_key or "/" not in text:
        return None, None

    # Only treat direct benchmark/subset pairs as subtasks. Deeper paths often
    # encode run storage layout rather than semantic benchmark subdivisions.
    if text.count("/") != 1:
        return None, None

    prefix_raw, suffix_raw = text.split("/", 1)
    if normalize_benchmark_key(prefix_raw) != benchmark_key:
        return None, None

    suffix_text = suffix_raw.strip(" /")
    if not suffix_text:
        return None, None

    return normalize_benchmark_key(suffix_text), humanize_token_key(suffix_text)


def benchmark_card_language_keys(benchmark_card: dict | None) -> set[str]:
    if not benchmark_card:
        return set()

    details = (
        benchmark_card.get("benchmark_details")
        if isinstance(benchmark_card, dict)
        else {}
    )
    languages = as_string_list((details or {}).get("languages"))
    keys = set()
    for language in languages:
        keys.update(candidate_benchmark_keys(language))
        keys.add(compact_benchmark_key(language))
    return {key for key in keys if key}


def is_language_subset_name(name: Any, benchmark_card: dict | None) -> bool:
    if not benchmark_card:
        return False

    normalized = normalize_benchmark_key(name)
    compact = compact_benchmark_key(name)
    if not normalized and not compact:
        return False

    language_keys = benchmark_card_language_keys(benchmark_card)
    if normalized in language_keys or compact in language_keys:
        return True
    return (
        normalized in COMMON_LANGUAGE_SUBSET_KEYS
        or compact in COMMON_LANGUAGE_SUBSET_KEYS
    )


def top_level_benchmark_owns_slices(
    benchmark: Any, benchmark_card: dict | None
) -> bool:
    benchmark_key = normalize_benchmark_key(benchmark)
    if benchmark_card:
        return True
    if benchmark_key in {
        normalize_benchmark_key(key) for key in BENCHMARK_DEFAULT_METRICS
    }:
        return True
    if benchmark_key.startswith("helm_"):
        return True
    return False


def infer_benchmark_leaf_and_slice(
    evaluation: dict,
    result: dict,
    benchmark_family_key: str,
    benchmark_family_name: str,
    component_key: str | None,
    component_name: str | None,
    benchmark_card: dict | None,
) -> tuple[str, str, str | None, str | None]:
    benchmark = as_string(evaluation.get("benchmark"))
    source_data = (
        result.get("source_data") if isinstance(result.get("source_data"), dict) else {}
    )
    dataset_name = as_string((source_data or {}).get("dataset_name"))
    raw_name = as_string(result.get("evaluation_name")).strip()
    raw_name_key = normalize_benchmark_key(raw_name)
    dataset_key = normalize_benchmark_key(dataset_name)
    top_level_key = normalize_benchmark_key(benchmark or dataset_name)
    top_level_name = infer_top_level_benchmark_name(
        benchmark or dataset_name, benchmark_family_name
    )

    subset_key, subset_name = infer_subset_slice_from_name(
        dataset_name or raw_name, benchmark or dataset_name
    )
    if subset_key and subset_name:
        return top_level_key, top_level_name, subset_key, subset_name

    # Paren-form subset: ``<prefix> (<suffix>)`` in dataset_name. Sibling
    # to the slash-form parser above (``tau-bench-2/airline``). The
    # prefix-must-extend-benchmark-identity guard keeps aggregator records
    # like ``MMLU (CoT)`` under ``llm-stats`` or ``NaturalQuestions
    # (closed-book)`` under ``helm_classic`` from being collapsed —
    # their prefix doesn't share an identity with the EEE config name.
    #
    # Unlike the slash form (which returns ``top_level_key`` as the leaf),
    # this returns the prefix as the leaf so ``SWE-PolyBench Verified``
    # and ``SWE-PolyBench`` stay distinct leaves with their own subtask
    # families, instead of collapsing into one.
    paren_match = re.match(r"^(.+?)\s*\(([^()]+)\)\s*$", dataset_name or raw_name or "")
    if paren_match:
        paren_prefix_raw = paren_match.group(1).strip()
        paren_suffix_raw = paren_match.group(2).strip()
        paren_prefix_key = normalize_benchmark_key(paren_prefix_raw)
        if (
            paren_prefix_key
            and paren_suffix_raw
            and top_level_key
            and (
                paren_prefix_key == top_level_key
                or paren_prefix_key.startswith(top_level_key + "_")
                or top_level_key.startswith(paren_prefix_key + "_")
            )
        ):
            slice_key = normalize_subset_key(paren_suffix_raw)
            slice_name = humanize_token_key(paren_suffix_raw)
            if slice_key:
                leaf_name = canonical_benchmark_display_name(
                    paren_prefix_raw,
                    fallback=paren_prefix_raw,
                )
                return paren_prefix_key, leaf_name, slice_key, slice_name

    if raw_name and raw_name_key and dataset_key and raw_name_key == dataset_key:
        canonical_raw_name = canonical_benchmark_display_name(
            raw_name,
            benchmark or dataset_name,
            benchmark_family_name,
            fallback=top_level_name,
        )
        if (
            is_language_subset_name(raw_name, benchmark_card)
            and raw_name_key != top_level_key
        ):
            return top_level_key, top_level_name, raw_name_key, raw_name
        return raw_name_key, canonical_raw_name, None, None

    if component_name:
        component_summary_key = normalize_benchmark_key(component_name)
        if component_summary_key in SUMMARY_SCORE_LEAF_KEYS:
            return top_level_key, top_level_name, None, None
        if top_level_benchmark_owns_slices(
            benchmark or dataset_name, benchmark_card
        ) or is_language_subset_name(component_name, benchmark_card):
            if (
                component_key == top_level_key
                or normalize_benchmark_key(component_name) == top_level_key
            ):
                return top_level_key, top_level_name, None, None
            return top_level_key, top_level_name, component_key, component_name
        return (
            component_key or normalize_benchmark_key(component_name),
            component_name,
            None,
            None,
        )

    return top_level_key, top_level_name, None, None


def classify_evaluation_result(
    evaluation: dict, result: dict, benchmark_card: dict | None
) -> dict:
    benchmark = as_string(evaluation.get("benchmark"))
    source_data = (
        result.get("source_data") if isinstance(result.get("source_data"), dict) else {}
    )
    dataset_name = as_string((source_data or {}).get("dataset_name"))
    benchmark_family_key = canonical_benchmark_family_key(benchmark or dataset_name)
    benchmark_card_name = as_string(
        ((benchmark_card or {}).get("benchmark_details") or {}).get("name")
    )
    benchmark_family_name = (
        canonical_benchmark_display_name(
            benchmark_family_key,
            benchmark_card_name,
            benchmark,
            dataset_name,
            fallback=benchmark_family_key or benchmark or dataset_name,
        )
        or "Unknown Benchmark"
    )
    raw_name = as_string(result.get("evaluation_name")).strip()
    benchmark_keys = [
        candidate
        for candidate in {
            normalize_benchmark_key(benchmark),
            normalize_benchmark_key(dataset_name),
            benchmark_family_key,
        }
        if candidate
    ]

    metric_config = (
        result.get("metric_config")
        if isinstance(result.get("metric_config"), dict)
        else {}
    )
    metric = None
    metric_source = "unknown"
    component_name = None
    component_key = None
    raw_name_consumed_as_metric = False

    explicit_metric = infer_metric_from_value(
        metric_name=metric_config.get("metric_name"),
        metric_id=metric_config.get("metric_id"),
    )
    if explicit_metric:
        metric = explicit_metric
        metric_source = "metric_config"
        component_name, component_key = metric_namespace_component(
            metric["metric_id"], benchmark_family_key
        )
        split_metric = split_metric_from_evaluation_name(raw_name, benchmark_keys)
        if (
            split_metric
            and split_metric["metric"]["metric_key"] == metric["metric_key"]
        ):
            if not component_name and not component_key:
                component_name = split_metric["component_name"]
                component_key = split_metric["component_key"]
            raw_name_consumed_as_metric = (
                split_metric["component_name"] is None
                and split_metric["component_key"] is None
            )

    if metric is None:
        split_metric = split_metric_from_evaluation_name(raw_name, benchmark_keys)
        if split_metric:
            metric = split_metric["metric"]
            metric_source = split_metric["metric_source"]
            component_name = split_metric["component_name"]
            component_key = split_metric["component_key"]
            raw_name_consumed_as_metric = component_name is None

    if metric is None:
        metric = split_metric_from_evaluation_description(
            metric_config.get("evaluation_description")
        )
        if metric:
            metric_source = "evaluation_description"

    if metric is None:
        metric = infer_metric_from_benchmark_card(benchmark_card)
        if metric:
            metric_source = "benchmark_card"

    if metric is None:
        metric = infer_metric_from_benchmark_defaults(benchmark_family_key)
        if metric:
            metric_source = "benchmark_default"

    if metric is None:
        metric = infer_metric_from_score_details(result)
        if metric:
            metric_source = "score_details"

    if metric is None:
        metric = {
            "metric_name": "Score",
            "metric_id": "score",
            "metric_key": "score",
        }
        metric_source = "fallback"

    raw_name_key = normalize_benchmark_key(raw_name)
    if (
        raw_name
        and not component_name
        and not raw_name_consumed_as_metric
        and raw_name_key
        and raw_name_key not in benchmark_keys
        and raw_name_key != metric["metric_key"]
    ):
        component_name = raw_name
        component_key = raw_name_key

    if component_name and not component_key:
        component_key = normalize_benchmark_key(component_name)

    benchmark_leaf_key, benchmark_leaf_name, slice_key, slice_name = (
        infer_benchmark_leaf_and_slice(
            evaluation,
            result,
            benchmark_family_key or normalize_benchmark_key(benchmark or dataset_name),
            benchmark_family_name,
            component_key,
            component_name,
            benchmark_card,
        )
    )

    # A leaf key that is a generic summary word (e.g. "overall") and differs
    # from the parent benchmark key means this score aggregates all the real
    # sub-benchmarks rather than representing an independent benchmark.
    parent_key = normalize_benchmark_key(benchmark or dataset_name)
    is_summary_score = (
        bool(benchmark_leaf_key)
        and benchmark_leaf_key in SUMMARY_SCORE_LEAF_KEYS
        and benchmark_leaf_key != parent_key
    )
    benchmark_parent_name = canonical_benchmark_display_name(
        parent_key,
        benchmark,
        dataset_name,
        benchmark_card_name,
        fallback=benchmark or dataset_name,
    )
    display_name = join_display_name_parts(component_name, metric["metric_name"])
    if not display_name:
        display_name = (
            benchmark_leaf_name or benchmark_parent_name or benchmark_family_name
        )
    canonical_display_name = join_display_name_parts(
        benchmark_leaf_name or benchmark_parent_name or benchmark_family_name,
        slice_name,
        metric["metric_name"],
    )
    if not canonical_display_name:
        canonical_display_name = display_name

    return {
        "benchmark_family_key": benchmark_family_key or parent_key,
        "benchmark_family_name": benchmark_family_name,
        "benchmark_parent_key": parent_key,
        "benchmark_parent_name": benchmark_parent_name,
        "benchmark_component_key": component_key,
        "benchmark_component_name": component_name,
        "benchmark_leaf_key": benchmark_leaf_key,
        "benchmark_leaf_name": benchmark_leaf_name,
        "slice_key": slice_key,
        "slice_name": slice_name,
        "metric_name": metric["metric_name"],
        "metric_id": metric["metric_id"],
        "metric_key": metric["metric_key"],
        "metric_source": metric_source,
        "display_name": display_name,
        "canonical_display_name": canonical_display_name,
        "raw_evaluation_name": raw_name or None,
        "is_summary_score": is_summary_score,
    }


def ensure_local_dataset_snapshot(
    local_dataset_dir: str, hf_token: str | None, force_refresh: bool
) -> str:
    target_dir = Path(local_dataset_dir).resolve()
    data_dir = target_dir / "data"
    target_dir.mkdir(parents=True, exist_ok=True)

    if force_refresh and target_dir.exists():
        shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        data_dir = target_dir / "data"

    if data_dir.exists() and any(data_dir.iterdir()):
        return str(target_dir)

    snapshot_download(
        repo_id=EEE_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=["data/**"],
        token=hf_token,
    )

    return str(target_dir)


def discover_configs(local_dataset_dir: str | None, hf_token: str | None) -> list[str]:
    if local_dataset_dir:
        data_root = Path(local_dataset_dir) / "data"
        configs = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
        return configs

    fs = HfFileSystem(token=hf_token)
    entries = fs.ls(f"datasets/{EEE_DATASET_REPO}/data", detail=True)
    configs = []
    for entry in entries:
        name = entry.get("name", "")
        config = name.split("/")[-1]
        if config:
            configs.append(config)
    return sorted(set(configs))


def list_json_files_for_config(
    config: str, local_dataset_dir: str | None, hf_token: str | None
) -> list[str]:
    if local_dataset_dir:
        root = Path(local_dataset_dir) / "data" / config
        return sorted(
            str(p.relative_to(local_dataset_dir)).replace(os.sep, "/")
            for p in root.rglob("*.json")
            if p.is_file() and not p.name.endswith(".jsonl")
        )

    fs = HfFileSystem(token=hf_token)
    pattern = f"datasets/{EEE_DATASET_REPO}/data/{config}/**/*.json"
    paths = [p for p in fs.glob(pattern) if not p.endswith(".jsonl")]
    prefix = f"datasets/{EEE_DATASET_REPO}/"
    return sorted(p[len(prefix) :] for p in paths)


def read_dataset_json(
    dataset_path: str, local_dataset_dir: str | None, hf_token: str | None
) -> dict:
    if local_dataset_dir:
        local_path = Path(local_dataset_dir) / dataset_path
        return json.loads(local_path.read_text(encoding="utf-8"))

    local_path = hf_hub_download(
        repo_id=EEE_DATASET_REPO,
        filename=dataset_path,
        repo_type="dataset",
        token=hf_token,
    )
    return json.loads(Path(local_path).read_text(encoding="utf-8"))


def raw_url_for_dataset_path(dataset_path: str) -> str:
    return f"{EEE_DATASET_RAW_BASE}/{dataset_path.lstrip('/')}"


def normalize_detailed_eval_meta(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if isinstance(value.get("entries"), dict):
            return value["entries"]
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        file_path_match = re.search(
            r"file_path'?:\s*'([^']+)'", value, flags=re.IGNORECASE
        ) or re.search(r'"file_path"\s*:\s*"([^"]+)"', value)
        format_match = re.search(
            r"format'?:\s*'([^']+)'", value, flags=re.IGNORECASE
        ) or re.search(r'"format"\s*:\s*"([^"]+)"', value)
        rows_match = re.search(
            r"total_rows'?:\s*(\d+)", value, flags=re.IGNORECASE
        ) or re.search(r'"total_rows"\s*:\s*(\d+)', value)
        if file_path_match or format_match or rows_match:
            return {
                "file_path": file_path_match.group(1) if file_path_match else None,
                "format": format_match.group(1) if format_match else None,
                "total_rows": int(rows_match.group(1)) if rows_match else None,
            }
    return None


def resolve_detailed_results_url(record: dict, source_record_url: str) -> str | None:
    value = record.get("detailed_evaluation_results")
    if isinstance(value, str) and value:
        if value.startswith("http://") or value.startswith("https://"):
            return value
        cleaned = value.lstrip("/")
        if cleaned.startswith("data/"):
            return raw_url_for_dataset_path(cleaned)
    if isinstance(value, dict):
        file_path = as_string(
            value.get("file_path") or value.get("path") or value.get("url")
        )
        if file_path:
            if file_path.startswith("http://") or file_path.startswith("https://"):
                return file_path
            if file_path.startswith("data/"):
                return raw_url_for_dataset_path(file_path)
            if source_record_url:
                base = source_record_url[: source_record_url.rfind("/") + 1]
                return f"{base}{file_path.lstrip('/')}"
    if source_record_url.endswith(".json"):
        return f"{source_record_url[:-5]}_samples.jsonl"
    return None


def infer_interaction_type(instances: list[dict]) -> str:
    if not instances:
        return "unknown"
    first = instances[0]
    if isinstance(first, dict):
        if "interactions" in first or "messages" in first:
            return "multi_turn"
        if "tool_calls" in first or "tool_use" in first:
            return "agentic"
        if "input" in first and "output" in first and "evaluation" in first:
            return "single_turn"
    return "unknown"


def maybe_load_instance_data(
    record: dict, local_dataset_dir: str | None, hf_token: str | None
) -> dict | None:
    candidates: list[str] = []
    explicit = as_string(record.get("detailed_evaluation_results"))
    source_record_url = as_string(record.get("source_record_url"))

    if explicit:
        candidates.append(explicit)
    if source_record_url.endswith(".json"):
        base = source_record_url[:-5]
        candidates.append(f"{base}_samples.jsonl")
        candidates.append(f"{base}.jsonl")

    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    for url in deduped:
        dataset_path = ""
        if url.startswith(f"{EEE_DATASET_RAW_BASE}/"):
            dataset_path = url[len(EEE_DATASET_RAW_BASE) + 1 :]

        try:
            if local_dataset_dir and dataset_path:
                text = (Path(local_dataset_dir) / dataset_path).read_text(
                    encoding="utf-8"
                )
            elif dataset_path:
                local_path = hf_hub_download(
                    repo_id=EEE_DATASET_REPO,
                    filename=dataset_path,
                    repo_type="dataset",
                    token=hf_token,
                )
                text = Path(local_path).read_text(encoding="utf-8")
            else:
                continue
        except Exception:
            continue

        lines = [line for line in text.splitlines() if line.strip()]
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

        if rows:
            examples = rows[:] if len(rows) <= 5 else random.sample(rows, 5)
            return {
                "interaction_type": infer_interaction_type(rows),
                "instance_count": len(rows),
                "source_url": url,
                "instance_examples": examples,
            }

    return None


def read_text_from_dataset_url(
    url: str, local_dataset_dir: str | None, hf_token: str | None
) -> str | None:
    dataset_path = ""
    if url.startswith(f"{EEE_DATASET_RAW_BASE}/"):
        dataset_path = url[len(EEE_DATASET_RAW_BASE) + 1 :]

    try:
        if local_dataset_dir and dataset_path:
            return (Path(local_dataset_dir) / dataset_path).read_text(encoding="utf-8")
        if dataset_path:
            local_path = hf_hub_download(
                repo_id=EEE_DATASET_REPO,
                filename=dataset_path,
                repo_type="dataset",
                token=hf_token,
            )
            return Path(local_path).read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def dataset_resolve_url(relative_path: str) -> str:
    return f"{DATASET_RESOLVE_BASE}/{relative_path.lstrip('/')}"


def build_instance_artifact_relative_path(evaluation: dict) -> str:
    route_id = as_string(
        (evaluation.get("model_info") or {}).get("model_route_id") or "unknown"
    )
    evaluation_key = slugify(
        evaluation.get("evaluation_id")
        or evaluation.get("source_record_url")
        or "instance"
    )
    return f"instances/{route_id}/{evaluation_key}.jsonl"


def build_record_artifact_relative_path(evaluation: dict) -> str:
    route_id = as_string(
        (evaluation.get("model_info") or {}).get("model_route_id") or "unknown"
    )
    evaluation_key = slugify(evaluation.get("evaluation_id") or "record")
    return f"records/{route_id}/{evaluation_key}.json"


def build_evaluation_hierarchy_payload(evaluation: dict) -> dict:
    category = infer_category_from_benchmark(
        as_string(evaluation.get("benchmark")), evaluation.get("benchmark_card")
    )
    model_info = evaluation.get("model_info") or {}
    return {
        "category": category,
        "benchmark": evaluation.get("benchmark"),
        "model_family_id": as_string(model_info.get("family_id")),
        "model_route_id": as_string(model_info.get("model_route_id")),
        "eval_summary_ids": evaluation.get("eval_summary_ids", []),
    }


def build_result_hierarchy_payload(evaluation: dict, result: dict) -> dict:
    normalized = result.get("normalized_result") or {}
    return {
        **build_evaluation_hierarchy_payload(evaluation),
        "eval_summary_id": get_eval_group_id(evaluation, result),
        "metric_summary_id": get_metric_summary_id(evaluation, result),
        "benchmark_family_key": normalized.get("benchmark_family_key"),
        "benchmark_family_name": normalized.get("benchmark_family_name"),
        "benchmark_parent_key": normalized.get("benchmark_parent_key"),
        "benchmark_parent_name": normalized.get("benchmark_parent_name"),
        "benchmark_leaf_key": normalized.get("benchmark_leaf_key"),
        "benchmark_leaf_name": normalized.get("benchmark_leaf_name"),
        "benchmark_component_key": normalized.get("benchmark_component_key"),
        "benchmark_component_name": normalized.get("benchmark_component_name"),
        "slice_key": normalized.get("slice_key"),
        "slice_name": normalized.get("slice_name"),
        "metric_key": normalized.get("metric_key"),
        "metric_name": normalized.get("metric_name"),
        "metric_source": normalized.get("metric_source"),
        "display_name": normalized.get("display_name"),
        "canonical_display_name": normalized.get("canonical_display_name"),
        "is_summary_score": bool(normalized.get("is_summary_score")),
    }


def find_matching_result_for_instance_row(evaluation: dict, row: dict) -> dict | None:
    results = evaluation.get("evaluation_results") or []
    if not results:
        return None

    evaluation_result_id = as_string(row.get("evaluation_result_id")).strip()
    if evaluation_result_id:
        for result in results:
            if (
                as_string(result.get("evaluation_result_id")).strip()
                == evaluation_result_id
            ):
                return result

    evaluation_name = as_string(row.get("evaluation_name")).strip()
    if evaluation_name:
        matches = []
        for result in results:
            normalized = result.get("normalized_result") or {}
            candidate_names = {
                as_string(result.get("evaluation_name")).strip(),
                as_string(normalized.get("raw_evaluation_name")).strip(),
                as_string(normalized.get("display_name")).strip(),
                as_string(normalized.get("canonical_display_name")).strip(),
                as_string(normalized.get("benchmark_leaf_name")).strip(),
            } - {""}
            if evaluation_name in candidate_names:
                matches.append(result)
        if len(matches) == 1:
            return matches[0]

    if len(results) == 1:
        return results[0]
    return None


def annotate_instance_row(evaluation: dict, row: dict) -> dict:
    annotated = dict(row)
    matched_result = find_matching_result_for_instance_row(evaluation, annotated)
    if matched_result is not None:
        annotated["hierarchy"] = build_result_hierarchy_payload(
            evaluation, matched_result
        )
    else:
        annotated["hierarchy"] = build_evaluation_hierarchy_payload(evaluation)
        if len(annotated["hierarchy"].get("eval_summary_ids", [])) == 1:
            annotated["hierarchy"]["eval_summary_id"] = annotated["hierarchy"][
                "eval_summary_ids"
            ][0]
    return annotated


def transform_instance_artifact_text(evaluation: dict, artifact_text: str) -> str:
    transformed_lines = []
    for line in artifact_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except Exception:
            transformed_lines.append(stripped)
            continue
        transformed_lines.append(
            json.dumps(annotate_instance_row(evaluation, row), ensure_ascii=False)
        )
    return "\n".join(transformed_lines) + ("\n" if transformed_lines else "")


def normalize_model_info(model_info: dict) -> dict:
    raw_id = as_string(
        model_info.get("id") or model_info.get("name") or "unknown/unknown"
    )
    fallback_developer = as_string(
        model_info.get("developer") or raw_id.split("/")[0] or "unknown"
    )
    if "/" in raw_id:
        parts = raw_id.split("/")
    else:
        parts = [slugify_developer(fallback_developer), raw_id]
    raw_developer = (
        parts[0] if len(parts) > 1 else slugify_developer(fallback_developer)
    )
    raw_model_name = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
    match = VERSION_SUFFIX_REGEX.match(raw_model_name)
    base_slug = match.group(1) if match else raw_model_name
    version_date = match.group(2) if match else None
    qualifier = match.group(3) if match else None

    return {
        "raw_id": raw_id,
        "developer": as_string(
            model_info.get("developer") or humanize_slug(raw_developer)
        ),
        "developer_slug": slugify_developer(raw_developer),
        "model_name": as_string(model_info.get("name") or humanize_slug(base_slug)),
        "raw_model_name": raw_model_name,
        "family_slug": slugify_model_segment(base_slug),
        "version_date": version_date,
        "qualifier": qualifier,
    }


def canonical_model_identity(model_info: dict) -> dict:
    normalized = normalize_model_info(model_info)
    family_id = f"{normalized['developer_slug']}/{normalized['family_slug']}"
    normalized_id = (
        f"{normalized['developer_slug']}/{normalized['raw_model_name']}"
        if "/" in normalized["raw_id"]
        else f"{normalized['developer_slug']}/{normalized['raw_id']}"
    )
    variant_parts = [
        p for p in [normalized["version_date"], normalized["qualifier"]] if p
    ]
    variant_key = "-".join(variant_parts) if variant_parts else "default"
    variant_label = " ".join(variant_parts) if variant_parts else "Default"

    return {
        "normalized_id": normalized_id,
        "family_id": family_id,
        "family_slug": normalized["family_slug"],
        "family_name": normalized["model_name"],
        "model_route_id": family_id.replace("/", "__"),
        "variant_key": variant_key,
        "variant_label": variant_label,
    }


def is_setup_alias_mode(model_info: dict) -> bool:
    additional_details = model_info.get("additional_details") or {}
    raw_mode = as_string(additional_details.get("mode")).strip()
    if not raw_mode:
        return False

    return is_setup_alias_qualifier(raw_mode)


def aggregated_display_identity(model_info: dict) -> dict:
    identity = canonical_model_identity(model_info)
    if not is_setup_alias_mode(model_info):
        return {
            **identity,
            "merged_setup_alias": False,
        }

    normalized = normalize_model_info(model_info)
    family_slug = normalized["family_slug"]
    stripped_alias = strip_setup_alias_suffix(family_slug)
    if stripped_alias:
        family_slug = stripped_alias[0]

    family_id = f"{normalized['developer_slug']}/{family_slug}"
    family_name = humanize_slug(family_slug)
    if normalized["version_date"]:
        variant_key = normalized["version_date"]
        variant_label = normalized["version_date"]
    else:
        variant_key = "default"
        variant_label = "Default"

    return {
        **identity,
        "family_id": family_id,
        "family_slug": family_slug,
        "family_name": family_name,
        "model_route_id": family_id.replace("/", "__"),
        "variant_key": variant_key,
        "variant_label": variant_label,
        "merged_setup_alias": True,
    }


# Domain keywords → high-level category mapping
_DOMAIN_CATEGORY_MAP = {
    "safety": "safety",
    "toxic": "safety",
    "bias": "safety",
    "fairness": "safety",
    "harmful": "safety",
    "ethics": "safety",
    "math": "reasoning",
    "mathematics": "reasoning",
    "reasoning": "reasoning",
    "commonsense reasoning": "reasoning",
    "planning": "reasoning",
    "logic": "reasoning",
    "olympiad": "reasoning",
    "coding": "coding",
    "code generation": "coding",
    "software engineering": "coding",
    "programming": "coding",
    "instruction following": "instruction_following",
    "summarization": "language_understanding",
    "reading comprehension": "language_understanding",
    "natural language understanding": "language_understanding",
    "natural language inference": "language_understanding",
    "question answering": "knowledge",
    "open domain qa": "knowledge",
    "multiple choice qa": "knowledge",
    "medical knowledge": "knowledge",
    "legal": "knowledge",
    "STEM": "knowledge",
    "humanities": "knowledge",
    "social sciences": "knowledge",
    "dialogue modeling": "language_understanding",
    "text generation": "language_understanding",
    "text classification": "language_understanding",
}


# Lowercase tokens that map to a backend-key category. Mirrors the TS
# `inferCategoryFromBenchmark` regex (`lib/benchmark-schema.ts:182`) so a
# benchmark with no card domains still gets the same canonical bucket the
# frontend would have computed at request time.
_FRONTEND_REGEX_CATEGORY_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "safety",
        (
            "safety",
            "harmful",
            "toxic",
            "truthful",
            "unsafe",
            "civilcomments",
            "civil_comments",
            "jailbreak",
            "red-team",
            "adversarial",
        ),
    ),
    (
        "agentic",
        (
            "agent",
            "swe-bench",
            "swe_bench",
            "terminal-bench",
            "tau-bench",
            "tau_bench",
            "appworld",
            "browsecomp",
        ),
    ),
    (
        "reasoning",
        (
            "reasoning",
            "bbh",
            "math",
            "gsm",
            "gpqa",
            "musr",
            "code",
            "humaneval",
            "livecodebench",
        ),
    ),
    (
        "knowledge",
        (
            "mmlu",
            "knowledge",
            "trivia",
            "medqa",
            "legalbench",
            "theory_of_mind",
        ),
    ),
)


def _frontend_regex_category(benchmark_name: str) -> str | None:
    if not benchmark_name:
        return None
    text = str(benchmark_name).lower()
    for category, tokens in _FRONTEND_REGEX_CATEGORY_TOKENS:
        if any(token in text for token in tokens):
            return category
    return None


def infer_category_from_benchmark(
    benchmark_name: str, benchmark_card: dict | None = None
) -> str:
    """Derive a high-level category, preferring benchmark card domains over regex.

    Layered fallback (most-specific → most-general):
      1. ABC card `benchmark_details.domains` (exact + substring match)
      2. Pipeline's own keyword regexes (helm, mmlu, etc.)
      3. The frontend's `inferCategoryFromBenchmark` regex tokens —
         catches Safety / Agentic / Reasoning / Knowledge cases the
         pipeline's narrower regex would have left as `"other"`.

    Without layer 3, the corpus emitted `category: "other"` on ~84% of
    evals (per migration audit 2026-04-27); downstream JSON consumers
    saw bogus catch-all buckets.
    """
    # Try card domains first
    if benchmark_card:
        domains = (benchmark_card.get("benchmark_details") or {}).get("domains") or []
        for domain in domains:
            domain_lower = domain.lower()
            if domain_lower in _DOMAIN_CATEGORY_MAP:
                return _DOMAIN_CATEGORY_MAP[domain_lower]
            # Partial matching for compound domains
            for keyword, category in _DOMAIN_CATEGORY_MAP.items():
                if keyword in domain_lower:
                    return category

    # Fallback to benchmark name regex
    key = normalize_benchmark_key(benchmark_name)
    if not key:
        return "other"
    if is_agentic_benchmark_name(key):
        return "agentic"
    if re.search(r"(global_mmlu_lite|boolq|medqa|legalbench|quac|cnn_dailymail)", key):
        return "knowledge"
    if re.search(r"(reward_bench)", key):
        return "general"
    if re.search(r"(math|gsm|gpqa|mmlu|hellaswag|musr)", key):
        return "reasoning"
    if re.search(r"(ifeval)", key):
        return "instruction_following"
    if re.search(r"(hfopenllm|helm)", key):
        return "general"

    frontend = _frontend_regex_category(benchmark_name)
    if frontend is not None:
        return frontend
    return "other"


def extract_score(result: dict) -> float | None:
    score_details = result.get("score_details") if isinstance(result, dict) else None
    if not isinstance(score_details, dict):
        return None
    score = score_details.get("score")
    try:
        return float(score)
    except Exception:
        return None


def get_eval_group_id(evaluation: dict, result: dict) -> str:
    normalized = result.get("normalized_result") if isinstance(result, dict) else None
    source_data = result.get("source_data") if isinstance(result, dict) else {}
    parent_key = (
        (normalized or {}).get("benchmark_parent_key")
        or evaluation.get("benchmark")
        or (source_data or {}).get("dataset_name")
    )
    benchmark_key = (
        (normalized or {}).get("benchmark_leaf_key")
        or (normalized or {}).get("benchmark_family_key")
        or evaluation.get("benchmark")
        or (source_data or {}).get("dataset_name")
        or result.get("evaluation_name")
    )
    pieces = []
    if as_string(parent_key):
        pieces.append(parent_key)
    if as_string(benchmark_key) and as_string(benchmark_key) != as_string(parent_key):
        pieces.append(benchmark_key)
    return slugify("__".join(as_string(piece) for piece in pieces if as_string(piece)))


def get_metric_summary_id(evaluation: dict, result: dict) -> str:
    normalized = result.get("normalized_result") if isinstance(result, dict) else None
    metric_key = (normalized or {}).get("metric_key")
    pieces = [get_eval_group_id(evaluation, result)]
    slice_key = (normalized or {}).get("slice_key")
    if slice_key:
        pieces.append(slice_key)
    if metric_key:
        pieces.append(metric_key)
    return slugify("__".join(as_string(piece) for piece in pieces if as_string(piece)))


def clean_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "evals").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "developers").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "instances").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "records").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_output_relative_files(root_dir: Path = OUTPUT_DIR) -> list[str]:
    if not root_dir.exists():
        return []
    return sorted(
        str(path.relative_to(root_dir)).replace(os.sep, "/")
        for path in root_dir.rglob("*")
        if path.is_file()
    )


def build_lightweight_model_cards(model_cards: list[dict]) -> list[dict]:
    lite_cards: list[dict] = []

    for card in model_cards:
        lite_cards.append(
            {
                "model_family_id": card.get("model_family_id"),
                "model_route_id": card.get("model_route_id"),
                "model_family_name": card.get("model_family_name"),
                "developer": card.get("developer"),
                "params_billions": card.get("params_billions"),
                "total_evaluations": card.get("total_evaluations"),
                "benchmark_count": card.get("benchmark_count"),
                "benchmark_family_count": card.get("benchmark_family_count"),
                "categories_covered": card.get("categories_covered") or [],
                "last_updated": card.get("last_updated"),
                "variants": [
                    {
                        "variant_key": variant.get("variant_key"),
                        "variant_label": variant.get("variant_label"),
                        "evaluation_count": variant.get("evaluation_count"),
                        "raw_model_ids": [],
                        "last_updated": variant.get("last_updated"),
                    }
                    for variant in (card.get("variants") or [])
                ],
                "score_summary": card.get("score_summary") or {},
                "reproducibility_summary": card.get("reproducibility_summary"),
                "provenance_summary": card.get("provenance_summary"),
                "comparability_summary": card.get("comparability_summary"),
                "benchmark_names": (card.get("benchmark_names") or [])[:8],
                "top_benchmark_scores": (card.get("top_benchmark_scores") or [])[:6],
            }
        )

    return lite_cards


def build_lightweight_eval_list(eval_list: dict) -> dict:
    lite_evals: list[dict] = []

    for summary in eval_list.get("evals") or []:
        instance_data = summary.get("instance_data") or {}
        lite_evals.append(
            {
                "eval_summary_id": summary.get("eval_summary_id"),
                "benchmark": summary.get("benchmark"),
                "benchmark_family_key": summary.get("benchmark_family_key"),
                "benchmark_family_name": summary.get("benchmark_family_name"),
                "benchmark_parent_key": summary.get("benchmark_parent_key"),
                "benchmark_parent_name": summary.get("benchmark_parent_name"),
                "benchmark_leaf_key": summary.get("benchmark_leaf_key"),
                "benchmark_leaf_name": summary.get("benchmark_leaf_name"),
                "benchmark_component_key": summary.get("benchmark_component_key"),
                "benchmark_component_name": summary.get("benchmark_component_name"),
                "evaluation_name": summary.get("evaluation_name"),
                "display_name": summary.get("display_name"),
                "canonical_display_name": summary.get("canonical_display_name"),
                "is_summary_score": summary.get("is_summary_score", False),
                "summary_score_for": summary.get("summary_score_for"),
                "summary_score_for_name": summary.get("summary_score_for_name"),
                "summary_eval_ids": summary.get("summary_eval_ids") or [],
                "category": summary.get("category", "other"),
                "models_count": summary.get("models_count", 0),
                "metrics_count": summary.get("metrics_count"),
                "subtasks_count": summary.get("subtasks_count"),
                "metric_names": summary.get("metric_names") or [],
                "primary_metric_name": summary.get("primary_metric_name"),
                "tags": summary.get("tags")
                or {"domains": [], "languages": [], "tasks": []},
                "source_data": summary.get("source_data"),
                "metrics": summary.get("metrics") or [],
                "top_score": summary.get("top_score"),
                "instance_data": {
                    "available": bool(instance_data.get("available", False)),
                    "url_count": instance_data.get("url_count", 0),
                    "sample_urls": (instance_data.get("sample_urls") or [])[:1],
                    "models_with_loaded_instances": instance_data.get(
                        "models_with_loaded_instances", 0
                    ),
                },
                "reproducibility_summary": summary.get("reproducibility_summary"),
                "provenance_summary": summary.get("provenance_summary"),
                "comparability_summary": summary.get("comparability_summary"),
            }
        )

    return {"evals": lite_evals}


def collect_artifact_sizes(root_dir: Path = OUTPUT_DIR) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for relative_path in iter_output_relative_files(root_dir):
        file_path = root_dir / relative_path
        artifacts.append(
            {
                "path": relative_path,
                "bytes": file_path.stat().st_size,
            }
        )
    return artifacts


# Uniform grouping for every metric the pipeline emits. The group label is
# attached to each metric in comparison-index.json so the frontend can render
# tabs in a consistent order ("Capability" before "Cost" before "Latency"
# etc.) across every benchmark, and is also used internally to choose a
# primary metric for eval-list.json's ``primary_metric_name``.
#
# Rules were derived empirically from the full set of 207 metric occurrences
# across 89 evals (see commit message). ``metric_kind`` from upstream EEE
# takes precedence; for the ~78% of metrics where ``metric_kind`` is absent
# we fall back to name regexes that cover every observed metric name.
_METRIC_KIND_TO_GROUP = {
    "accuracy": "capability",
    "elo": "capability",
    "score": "capability",
    "pass": "capability",
    "f1": "capability",
    "win_rate": "capability",
    "winrate": "capability",
    "cost": "cost",
    "latency": "latency",
    "throughput": "latency",
    "time": "latency",
    "rank": "rank",
    "difference": "robustness",
}

# Order matters: first matching pattern wins. Listed most-specific first so
# e.g. "Latency Standard Deviation" lands in latency rather than robustness.
_METRIC_NAME_GROUP_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (
        "cost",
        re.compile(r"\b(?:cost|usd|dollar|price)\b", re.IGNORECASE),
    ),
    (
        "latency",
        re.compile(
            r"\b(?:latency|throughput|elapsed|wall[\s_]?time|"
            r"tokens?[\s_/]?(?:per|sec|s)\b|p\d{2,3}|percentile)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rank",
        re.compile(r"\brank\b", re.IGNORECASE),
    ),
    (
        "robustness",
        re.compile(
            r"\b(?:sensitivity|delta|stddev|standard[\s_]?deviation|"
            r"variance|robustness)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "efficiency",
        re.compile(r"\b(?:attempts|retries|tries)\b", re.IGNORECASE),
    ),
    (
        "capability",
        re.compile(
            r"\b(?:accuracy|acc|elo|score|pass@\d+|win[\s_]?rate|f1|"
            r"exact[\s_]?match|em|bleu|rouge(?:-\d+)?|recall|precision|"
            r"mrr|ndcg|coverage|correct|harmlessness)\b",
            re.IGNORECASE,
        ),
    ),
)

# Tab order in the histogram UI. Capability surfaces first (the actual
# task score), followed by capability-adjacent groups, then instrumental
# groups, with "other" as a fallback bucket.
METRIC_GROUP_ORDER = (
    "capability",
    "robustness",
    "efficiency",
    "cost",
    "latency",
    "rank",
    "other",
)
_METRIC_GROUP_INDEX = {group: i for i, group in enumerate(METRIC_GROUP_ORDER)}


def metric_group(metric: dict) -> str:
    """Classify a metric into one of ``METRIC_GROUP_ORDER``.

    ``metric_kind`` from upstream EEE is authoritative when present (e.g.
    ``kind=cost`` always means ``cost`` regardless of name). Otherwise we
    match the metric name against the rules above. Defaults to ``other``.
    """
    config = metric.get("metric_config") or {}
    kind = (as_string(config.get("metric_kind")) or "").lower()
    if kind in _METRIC_KIND_TO_GROUP:
        return _METRIC_KIND_TO_GROUP[kind]
    name = as_string(metric.get("metric_name"))
    if name:
        for group, pattern in _METRIC_NAME_GROUP_RULES:
            if pattern.search(name):
                return group
    return "other"


def metric_group_order_index(group: str) -> int:
    """Sort key for a metric group. Lower = surfaced first in the UI."""
    return _METRIC_GROUP_INDEX.get(group, _METRIC_GROUP_INDEX["other"])


# Tier mapping derived from the group taxonomy: capability wins, then
# capability-adjacent (robustness / efficiency / other), then instrumental.
# Used only for eval-list.json's ``primary_metric_name`` — comparison-index
# emits all metrics so doesn't pick a primary.
_METRIC_TIER_BY_GROUP = {
    "capability": 0,
    "robustness": 1,
    "efficiency": 1,
    "other": 1,
    "cost": 2,
    "latency": 2,
    "rank": 2,
}


def metric_priority_tier(metric: dict) -> int:
    """Rank a metric for ``primary_metric_name`` selection. Derived from
    ``metric_group`` so the picker and the comparison-index tab order stay
    in lockstep.
    """
    return _METRIC_TIER_BY_GROUP.get(metric_group(metric), 1)


def pick_primary_metric(metrics: list[dict]) -> dict | None:
    """Pick one canonical metric from a list, preferring capability scores
    over instrumental ones. Stable secondary sort matches the existing
    alphabetical order so within-tier choices stay deterministic and align
    with the order metrics appear in eval-list.json. Used for eval-list's
    ``primary_metric_name`` only — the comparison index emits every metric
    via a tabbed UI and no longer needs a single pick.
    """
    if not metrics:
        return None
    return min(
        metrics,
        key=lambda m: (
            metric_priority_tier(m),
            as_string(m.get("metric_name")),
            as_string(m.get("metric_summary_id")),
        ),
    )


# Submission "axes" — the kind of thing that differentiates one row from
# another for the same (model_route_id, eval, metric). Surfaced per row so
# the frontend can label the bar correctly: "Harness: droid" vs "Variant:
# thinking-8k" vs "Re-run: 2026-03-17".
RUN_KIND_HARNESS = "harness"  # different agent / scaffold (terminal_bench, swe_bench)
RUN_KIND_VARIANT = "variant"  # raw_model_id varies (reasoning budget, snapshot)
RUN_KIND_RERUN = "rerun"  # same setup, re-evaluated later
RUN_KIND_DEFAULT = "default"  # the only submission for this peer


def extract_run_descriptor(row: dict) -> tuple[str, str]:
    """Return ``(run_kind, run_label)`` for a single ``model_results`` row.

    These rows are *submissions*, not subtasks. Multiple rows for the same
    ``model_route_id`` on a benchmark generally mean one of three things:

      * **harness**: same model run by different agent scaffolding —
        terminal_bench's droid / letta-code / mux / openhands / claude-code
        agents, browsecompplus's smolagents-code / openai-solo / claude-
        code-cli, swe_bench's openai-solo / claude-code-cli / smolagents-
        code. The harness is encoded as the ``<harness>__<model>`` segment
        of ``evaluation_id``.
      * **variant**: same family but a different ``raw_model_id`` —
        reasoning-budget variants (claude-haiku-4-5 + ``-thinking-1k`` /
        ``-thinking-8k``) or snapshot dates (claude-3-5-sonnet
        ``-20240620`` / ``-20241022``) that the canonical model identity
        collapses to one family.
      * **rerun**: identical model and setup, evaluated at a later date —
        differentiated only by ``retrieved_timestamp`` /
        ``evaluation_timestamp``.

    Returned ``run_kind`` is one of ``RUN_KIND_*``; ``run_label`` is a
    short string the frontend can display ("droid", "thinking-8k",
    "2026-03-17"). Single-submission peers receive
    ``(RUN_KIND_DEFAULT, "")`` from the caller — this function only runs
    when there is something to differentiate.
    """
    eval_id = as_string(row.get("evaluation_id"))
    if eval_id:
        parts = eval_id.split("/")
        if len(parts) >= 2 and "__" in parts[1]:
            harness = parts[1].split("__", 1)[0].strip()
            if harness:
                return RUN_KIND_HARNESS, harness

    raw = as_string(row.get("raw_model_id"))
    family = as_string(row.get("model_id"))
    if raw and family and raw != family:
        if raw.lower().startswith(family.lower()):
            tail = raw[len(family) :].lstrip("-_/").strip()
            if tail:
                return RUN_KIND_VARIANT, tail
        return RUN_KIND_VARIANT, raw

    pt = row.get("passthrough_top_level_fields") or {}
    eval_ts = as_string(pt.get("evaluation_timestamp"))
    if eval_ts:
        return RUN_KIND_RERUN, eval_ts

    retrieved = as_string(row.get("retrieved_timestamp"))
    if retrieved:
        try:
            iso = (
                datetime.fromtimestamp(float(retrieved), tz=timezone.utc)
                .date()
                .isoformat()
            )
            return RUN_KIND_RERUN, iso
        except (TypeError, ValueError):
            return RUN_KIND_RERUN, retrieved
    return RUN_KIND_RERUN, "submission"


def build_comparison_index(eval_summaries: list[dict], generated_at: str) -> dict:
    """Build an exhaustive per-eval, per-metric comparison index.

    For each eval_summary, emits every metric the eval reports — the frontend
    renders one tab per metric in its histogram view, so there is no
    ``primary metric`` here. Each metric carries its full ``scores`` list:
    one entry per scoring model_route_id, ranked best-first respecting
    ``lower_is_better``. Also emits an inverse ``by_model`` index keyed by
    (model_route_id, eval_summary_id, metric_summary_id) so a model detail
    page can look up its peer comparisons in O(1) per benchmark+metric.

    Metrics within an eval are ordered by ``metric_group`` (capability tabs
    first, then robustness / efficiency / cost / latency / rank / other), so
    the tab strip has the same shape across every benchmark.
    """

    evals_out: dict[str, dict] = {}
    by_model: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for summary in eval_summaries:
        eval_summary_id = as_string(summary.get("eval_summary_id"))
        if not eval_summary_id:
            continue

        # Use root metrics if the eval has any; otherwise fall back to the
        # first subtask's metrics. Mirrors the same fallback that
        # ``primary_metric_name`` uses in eval-list.json.
        root_metrics = summary.get("metrics") or []
        subtasks = summary.get("subtasks") or []
        candidate_metrics = (
            root_metrics or (subtasks[0].get("metrics") if subtasks else None) or []
        )
        if not candidate_metrics:
            continue

        # Capability tabs surface first across every eval. Within-group
        # ordering is alphabetical to keep the artifact deterministic.
        ordered_metrics = sorted(
            candidate_metrics,
            key=lambda m: (
                metric_group_order_index(metric_group(m)),
                as_string(m.get("metric_name")),
                as_string(m.get("metric_summary_id")),
            ),
        )

        metrics_out: list[dict] = []
        for metric in ordered_metrics:
            metric_summary_id = as_string(metric.get("metric_summary_id"))
            if not metric_summary_id:
                continue
            lower_is_better = bool(metric.get("lower_is_better"))
            group = metric_group(metric)

            # Group rows by model_route_id. Multiple rows per route are
            # *submissions* (different agent harnesses, reasoning-budget
            # variants, model snapshots, or simple re-runs) — see
            # `extract_run_descriptor`. The headline bar uses the best
            # submission's score; the full submission list is kept so the
            # frontend can drill in.
            rows_by_route: dict[str, list[dict]] = defaultdict(list)
            for row in metric.get("model_results") or []:
                route = as_string(row.get("model_route_id"))
                if not route:
                    continue
                if row.get("score") is None:
                    continue
                rows_by_route[route].append(row)

            # Build a per-route headline entry + its submission tail.
            route_entries: list[tuple[str, dict, list[dict]]] = []
            for route, rows in rows_by_route.items():
                # Sort submissions best-first within the route.
                rows = sorted(
                    rows,
                    key=lambda r: r["score"],
                    reverse=not lower_is_better,
                )
                headline = rows[0]
                # Only label submissions when more than one exists for this
                # route — single-submission peers don't need a setup label.
                if len(rows) > 1:
                    submissions = []
                    for sub in rows:
                        run_kind, run_label = extract_run_descriptor(sub)
                        submissions.append(
                            {
                                "score": sub["score"],
                                "run_kind": run_kind,
                                "run_label": run_label,
                                "raw_model_id": as_string(sub.get("raw_model_id"))
                                or None,
                            }
                        )
                else:
                    submissions = []
                route_entries.append((route, headline, submissions))

            # Two-pass stable sort across routes: route id ascending
            # tiebreak, then headline score in the metric's preferred
            # direction.
            route_entries.sort(key=lambda e: e[0])
            route_entries.sort(key=lambda e: e[1]["score"], reverse=not lower_is_better)

            total = len(route_entries)
            scores_out: list[dict] = []
            position = 0
            previous_score = None
            for idx, (route_id, headline, submissions) in enumerate(
                route_entries, start=1
            ):
                row_score = headline["score"]
                if previous_score is None or row_score != previous_score:
                    position = idx
                    previous_score = row_score
                # Detect the submission axis for this peer (used by the UI to
                # decide how to caption the drill-in: "8 harnesses" vs
                # "3 reasoning budgets" vs "2 re-runs").
                if submissions:
                    kinds = {s["run_kind"] for s in submissions}
                    submission_axis = next(iter(kinds)) if len(kinds) == 1 else "mixed"
                    headline_kind, headline_label = extract_run_descriptor(headline)
                else:
                    submission_axis = RUN_KIND_DEFAULT
                    headline_kind, headline_label = RUN_KIND_DEFAULT, ""

                entry: dict = {
                    "model_route_id": route_id,
                    "model_family_id": as_string(headline.get("model_id")),
                    "model_family_name": as_string(headline.get("model_name")),
                    "developer": as_string(headline.get("developer")),
                    "variant_key": as_string(headline.get("variant_key")) or "default",
                    "score": row_score,
                    "rank": position,
                    "total": total,
                    "submission_count": len(submissions) if submissions else 1,
                    "submission_axis": submission_axis,
                }
                if submissions:
                    entry["headline_run_kind"] = headline_kind
                    entry["headline_run_label"] = headline_label
                    entry["submissions"] = submissions
                scores_out.append(entry)

                by_model[route_id][eval_summary_id][metric_summary_id] = {
                    "score": row_score,
                    "rank": position,
                    "total": total,
                    "submission_count": entry["submission_count"],
                    "submission_axis": submission_axis,
                }

            metric_config = metric.get("metric_config") or {}
            unit = (
                as_string(metric_config.get("unit"))
                or as_string(metric_config.get("metric_unit"))
                or None
            )

            metrics_out.append(
                {
                    "metric_summary_id": metric_summary_id,
                    "metric_name": as_string(metric.get("metric_name")),
                    "metric_id": as_string(metric.get("metric_id")),
                    "metric_key": as_string(metric.get("metric_key")),
                    "group": group,
                    "group_order": metric_group_order_index(group),
                    "lower_is_better": lower_is_better,
                    "unit": unit,
                    "scores": scores_out,
                }
            )

        if not metrics_out:
            continue

        evals_out[eval_summary_id] = {
            "eval_summary_id": eval_summary_id,
            "canonical_benchmark_id": summary.get("canonical_benchmark_id"),
            "benchmark_family_key": summary.get("benchmark_family_key"),
            "benchmark_family_name": summary.get("benchmark_family_name"),
            "benchmark_parent_key": summary.get("benchmark_parent_key"),
            "benchmark_parent_name": summary.get("benchmark_parent_name"),
            "benchmark_leaf_key": summary.get("benchmark_leaf_key"),
            "benchmark_leaf_name": summary.get("benchmark_leaf_name"),
            "display_name": summary.get("display_name"),
            "category": summary.get("category", "other"),
            "is_summary_score": bool(summary.get("is_summary_score")),
            "summary_score_for": summary.get("summary_score_for"),
            "summary_eval_ids": summary.get("summary_eval_ids", []),
            "metrics": metrics_out,
        }

    return {
        "generated_at": generated_at,
        "config_version": CONFIG_VERSION,
        "metric_group_order": list(METRIC_GROUP_ORDER),
        "evals": evals_out,
        "by_model": {
            route: {eid: dict(metric_map) for eid, metric_map in eval_map.items()}
            for route, eval_map in by_model.items()
        },
    }


def validate_output_contract(output_dir: Path = OUTPUT_DIR) -> None:
    errors: list[str] = []

    eval_list_path = output_dir / "eval-list.json"
    model_cards_path = output_dir / "model-cards.json"
    evals_dir = output_dir / "evals"
    models_dir = output_dir / "models"

    eval_summary_ids: set[str] = set()
    if eval_list_path.exists():
        eval_list = json.loads(eval_list_path.read_text(encoding="utf-8"))
        eval_summary_ids = {
            as_string(item.get("eval_summary_id"))
            for item in (eval_list.get("evals") or [])
            if as_string(item.get("eval_summary_id"))
        }

    published_eval_files = {
        path.stem for path in evals_dir.glob("*.json") if path.is_file()
    }
    if eval_summary_ids and published_eval_files != eval_summary_ids:
        missing_files = sorted(eval_summary_ids - published_eval_files)
        extra_files = sorted(published_eval_files - eval_summary_ids)
        if missing_files:
            errors.append(
                f"Missing eval files for eval-list entries: {missing_files[:10]}"
            )
        if extra_files:
            errors.append(
                f"Extra eval files not present in eval-list: {extra_files[:10]}"
            )

    required_eval_keys = [
        "benchmark_family_key",
        "benchmark_family_name",
        "benchmark_parent_key",
        "benchmark_parent_name",
        "benchmark_leaf_key",
        "benchmark_leaf_name",
        "canonical_display_name",
    ]

    for eval_path in sorted(evals_dir.glob("*.json")):
        parsed = json.loads(eval_path.read_text(encoding="utf-8"))
        missing_keys = [key for key in required_eval_keys if not parsed.get(key)]
        if missing_keys:
            errors.append(
                f"{eval_path.name} missing top-level hierarchy keys: {missing_keys}"
            )

        for metric in parsed.get("metrics", []):
            for row in metric.get("model_results", []):
                record_url = as_string(row.get("source_record_url"))
                if record_url and not record_url.startswith(
                    f"{DATASET_RESOLVE_BASE}/records/"
                ):
                    errors.append(
                        f"{eval_path.name} has non-pipeline source_record_url: {record_url}"
                    )
                detailed_url = as_string(row.get("detailed_evaluation_results"))
                if detailed_url and not detailed_url.startswith(
                    f"{DATASET_RESOLVE_BASE}/instances/"
                ):
                    errors.append(
                        f"{eval_path.name} has non-pipeline detailed_evaluation_results URL: {detailed_url}"
                    )
                instance_data = row.get("instance_level_data") or {}
                source_url = as_string(instance_data.get("source_url"))
                if source_url and not source_url.startswith(
                    f"{DATASET_RESOLVE_BASE}/instances/"
                ):
                    errors.append(
                        f"{eval_path.name} has non-pipeline instance_level_data.source_url: {source_url}"
                    )

        for subtask in parsed.get("subtasks", []):
            for metric in subtask.get("metrics", []):
                for row in metric.get("model_results", []):
                    record_url = as_string(row.get("source_record_url"))
                    if record_url and not record_url.startswith(
                        f"{DATASET_RESOLVE_BASE}/records/"
                    ):
                        errors.append(
                            f"{eval_path.name} has non-pipeline source_record_url: {record_url}"
                        )
                    detailed_url = as_string(row.get("detailed_evaluation_results"))
                    if detailed_url and not detailed_url.startswith(
                        f"{DATASET_RESOLVE_BASE}/instances/"
                    ):
                        errors.append(
                            f"{eval_path.name} has non-pipeline detailed_evaluation_results URL: {detailed_url}"
                        )
                    instance_data = row.get("instance_level_data") or {}
                    source_url = as_string(instance_data.get("source_url"))
                    if source_url and not source_url.startswith(
                        f"{DATASET_RESOLVE_BASE}/instances/"
                    ):
                        errors.append(
                            f"{eval_path.name} has non-pipeline instance_level_data.source_url: {source_url}"
                        )

    for model_path in sorted(models_dir.glob("*.json")):
        parsed = json.loads(model_path.read_text(encoding="utf-8"))
        if "hierarchy_by_category" not in parsed:
            errors.append(f"{model_path.name} missing hierarchy_by_category")

    model_cards_by_route_id: dict[str, dict] = {}
    if model_cards_path.exists():
        parsed_model_cards = json.loads(model_cards_path.read_text(encoding="utf-8"))
        if isinstance(parsed_model_cards, list):
            model_cards_by_route_id = {
                as_string(card.get("model_route_id")): card
                for card in parsed_model_cards
                if as_string(card.get("model_route_id"))
            }

    for model_path in sorted(models_dir.glob("*.json")):
        parsed = json.loads(model_path.read_text(encoding="utf-8"))
        hierarchy_by_category = parsed.get("hierarchy_by_category") or {}
        hierarchy_categories = sorted(
            category
            for category, summaries in hierarchy_by_category.items()
            if summaries
        )
        actual_eval_summary_ids = {
            as_string(summary.get("eval_summary_id"))
            for summaries in hierarchy_by_category.values()
            for summary in summaries or []
            if as_string(summary.get("eval_summary_id"))
        }

        declared_categories = sorted(
            as_string(category)
            for category in parsed.get("categories_covered") or []
            if as_string(category)
        )
        if declared_categories != hierarchy_categories:
            errors.append(
                f"{model_path.name} categories_covered mismatch: declared={declared_categories} actual={hierarchy_categories}"
            )

        route_id = as_string(parsed.get("model_route_id")) or model_path.stem
        model_card = model_cards_by_route_id.get(route_id)
        if not model_card:
            continue

        expected_eval_summary_ids = {
            as_string(entry.get("benchmarkKey"))
            for entry in model_card.get("top_benchmark_scores") or []
            if as_string(entry.get("benchmarkKey")) in eval_summary_ids
        }
        missing_eval_summary_ids = sorted(
            expected_eval_summary_ids - actual_eval_summary_ids
        )
        if missing_eval_summary_ids:
            errors.append(
                f"{model_path.name} missing hierarchy nodes for model-card top_benchmark_scores: {missing_eval_summary_ids[:10]}"
            )

        card_categories = sorted(
            as_string(category)
            for category in model_card.get("categories_covered") or []
            if as_string(category)
        )
        if card_categories != hierarchy_categories:
            errors.append(
                f"{model_path.name} model-card categories_covered mismatch: card={card_categories} actual={hierarchy_categories}"
            )

    for relative_path in iter_output_relative_files(output_dir):
        text = (output_dir / relative_path).read_text(encoding="utf-8")
        if EEE_DATASET_REPO in text:
            errors.append(f"{relative_path} still contains {EEE_DATASET_REPO}")

    if errors:
        raise RuntimeError(
            "Output contract validation failed:\n- " + "\n- ".join(errors[:50])
        )


def delete_stale_remote_files(
    api: HfApi,
    token: str,
    output_dir: Path = OUTPUT_DIR,
    repo_id: str | None = None,
) -> None:
    target = repo_id or DATASET_REPO
    local_files = set(iter_output_relative_files(output_dir))
    remote_files = set(
        api.list_repo_files(target, repo_type="dataset", token=token)
    )
    stale_files = sorted(remote_files - local_files)
    if not stale_files:
        return

    chunk_size = 200
    for index in range(0, len(stale_files), chunk_size):
        chunk = stale_files[index : index + chunk_size]
        api.delete_files(
            repo_id=target,
            repo_type="dataset",
            token=token,
            delete_patterns=chunk,
            commit_message=f"Remove stale pipeline artifacts ({index + 1}-{index + len(chunk)})",
        )


def attach_comparability_signals(metric_summary: dict) -> None:
    """Compute and attach comparability and provenance signals for one metric summary."""
    metric_summary_id = as_string(metric_summary.get("metric_summary_id"))
    metric_config = metric_summary.get("metric_config")

    rows = metric_summary.get("model_results") or []
    grouped: dict[str, list[dict]] = defaultdict(list)
    group_order: list[str] = []
    for row in rows:
        route = as_string(row.get("model_route_id"))
        if route not in grouped:
            group_order.append(route)
        grouped[route].append(row)

    # Store by group so that per-eval and per-model aggregates can count groups (not rows)
    signal_groups: list[dict] = []
    for route in group_order:
        group_rows = grouped[route]
        group_id = f"{route}__{metric_summary_id}"

        projected = [
            {
                "score": row.get("score"),
                "evaluation_id": row.get("evaluation_id"),
                "source_metadata": row.get("source_metadata"),
                "generation_args": row.get("_generation_args"),
                "variant_key": row.get("variant_key"),
                "model_route_id": row.get("model_route_id"),
            }
            for row in group_rows
        ]

        provenance_per_row = signals.compute_provenance(projected)
        variant_signal = signals.compute_variant_divergence(
            projected, metric_config, group_id=group_id
        )
        cross_party_signal = signals.compute_cross_party_divergence(
            projected, metric_config, group_id=group_id
        )

        first_party_only_in_group = False
        for row, prov in zip(group_rows, provenance_per_row):
            annotations = row.setdefault("evalcards", {}).setdefault("annotations", {})
            annotations["provenance"] = prov
            if variant_signal is None:
                annotations["variant_divergence"] = None
            else:
                row_variant = dict(variant_signal)
                row_variant["this_triple_score"] = row.get("score")
                annotations["variant_divergence"] = row_variant
            annotations["cross_party_divergence"] = cross_party_signal
            if prov.get("first_party_only"):
                first_party_only_in_group = True

        is_multi_source = (
            bool(provenance_per_row[0]["is_multi_source"])
            if provenance_per_row
            else False
        )

        signal_groups.append(
            {
                "group_id": group_id,
                "model_route_id": route,
                "is_multi_source": is_multi_source,
                "first_party_only_in_group": first_party_only_in_group,
                "variant_divergence": variant_signal,
                "cross_party_divergence": cross_party_signal,
            }
        )

    metric_summary["_signal_groups"] = signal_groups


def _iter_summary_metrics(summary: dict):
    """Yield root metrics + subtask metrics for one eval_summary."""
    for metric in summary.get("metrics") or []:
        yield metric
    for subtask in summary.get("subtasks") or []:
        for metric in subtask.get("metrics") or []:
            yield metric


def attach_canonical_cross_party_signals(eval_summaries: list[dict]) -> None:
    """Sidecar Signal 4 grouped by (canonical_benchmark_id, metric_key).

    The legacy ``attach_comparability_signals`` groups rows within a
    single ``metric_summary_id`` (always suite-prefixed), so MMLU-on-GPT-4
    rows from ``helm_classic_mmlu`` and ``helm_lite_mmlu`` never land in
    the same cross-party group. This pass re-buckets metric_summaries
    across all eval_summaries by their registry-resolved canonical
    benchmark id and re-runs ``compute_cross_party_divergence`` on the
    union of rows, attaching the result as a sidecar
    ``cross_party_divergence_canonical`` annotation.

    Field shape contract:
      - Every row gets ``cross_party_divergence_canonical`` set (uniform
        presence; None when no cross-suite duplicates exist).
      - Each metric_summary gets ``_signal_groups_canonical`` (filtered
        to model_route_ids that have rows in that specific
        metric_summary, so per-eval and per-model rollups don't
        double-count groups across summaries that share a canonical id).
      - Buckets that span only 1 eval_summary are skipped — there is no
        cross-suite comparison to make. Their rows keep
        ``cross_party_divergence_canonical = None``.
    """
    # Bucket every metric_summary by (canonical_benchmark_id, metric_key).
    buckets: dict[tuple[str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for summary in eval_summaries:
        canonical_id = as_string(summary.get("canonical_benchmark_id"))
        if not canonical_id:
            continue
        for metric in _iter_summary_metrics(summary):
            metric_key = as_string(metric.get("metric_key"))
            if not metric_key:
                continue
            buckets[(canonical_id, metric_key)].append((summary, metric))

    # Initialize annotation + signal_groups to defaults so the field
    # shape is uniform across the corpus, regardless of canonical match.
    for summary in eval_summaries:
        for metric in _iter_summary_metrics(summary):
            metric["_signal_groups_canonical"] = []
            for row in metric.get("model_results") or []:
                annotations = row.setdefault("evalcards", {}).setdefault(
                    "annotations", {}
                )
                annotations.setdefault("cross_party_divergence_canonical", None)

    canonical_groups_emitted = 0
    canonical_buckets_with_divergence = 0
    cross_suite_buckets_count = 0

    for (canonical_id, metric_key), entries in buckets.items():
        # Identity (not key) comparison — two summaries with identical
        # eval_summary_id can't exist post-construction, so id() is
        # safe and avoids re-hashing the dict.
        if len({id(s) for s, _ in entries}) < 2:
            continue
        cross_suite_buckets_count += 1

        # The metric_config of the first entry drives lower_is_better /
        # threshold. By construction (same canonical_id + metric_key),
        # all entries SHOULD agree on these; pick arbitrarily.
        metric_config = entries[0][1].get("metric_config")

        rows_by_route: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        route_order: list[str] = []
        for _summary, metric in entries:
            for row in metric.get("model_results") or []:
                route = as_string(row.get("model_route_id"))
                if not route:
                    continue
                if route not in rows_by_route:
                    route_order.append(route)
                rows_by_route[route].append((metric, row))

        for route in route_order:
            group_rows = rows_by_route[route]
            group_id = (
                f"{route}__canonical__{canonical_id}__{metric_key}"
            )
            projected = [
                {
                    "score": row.get("score"),
                    "evaluation_id": row.get("evaluation_id"),
                    "source_metadata": row.get("source_metadata"),
                    "generation_args": row.get("_generation_args"),
                    "variant_key": row.get("variant_key"),
                    "model_route_id": row.get("model_route_id"),
                }
                for _metric, row in group_rows
            ]
            cross_party_signal = signals.compute_cross_party_divergence(
                projected, metric_config, group_id=group_id
            )

            for _metric, row in group_rows:
                annotations = row.setdefault("evalcards", {}).setdefault(
                    "annotations", {}
                )
                annotations["cross_party_divergence_canonical"] = (
                    cross_party_signal
                )

            # Stash a signal_group entry on each contributing
            # metric_summary, but only once per metric_summary per route
            # so per-eval rollups count groups correctly.
            seen_metric_ids: set[int] = set()
            for metric, _row in group_rows:
                if id(metric) in seen_metric_ids:
                    continue
                seen_metric_ids.add(id(metric))
                metric["_signal_groups_canonical"].append(
                    {
                        "group_id": group_id,
                        "model_route_id": route,
                        "canonical_benchmark_id": canonical_id,
                        "metric_key": metric_key,
                        "cross_party_divergence": cross_party_signal,
                    }
                )

            canonical_groups_emitted += 1
            if cross_party_signal and cross_party_signal.get(
                "has_cross_party_divergence"
            ):
                canonical_buckets_with_divergence += 1

    print(
        f"[pipeline] {json.dumps({'event': 'registry.canonical_signal_4', 'cross_suite_buckets': cross_suite_buckets_count, 'eligible_groups': canonical_groups_emitted, 'divergent_groups': canonical_buckets_with_divergence})}"
    )


def collect_signal_rollup_inputs(
    metrics: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Walk metrics and aggregate row + group annotations.

    Returns ``(row_repro, row_provenance, signal_groups, signal_groups_canonical)``.
    The canonical list is the cross-suite Signal 4 groups stashed by
    ``attach_canonical_cross_party_signals``; entries are filtered per
    metric_summary so each (canonical bucket, model_route_id) appears at
    most once across the metrics passed in.
    """
    row_repro_annotations: list[dict] = []
    row_provenance_annotations: list[dict] = []
    signal_groups: list[dict] = []
    signal_groups_canonical: list[dict] = []
    for metric in metrics:
        signal_groups.extend(metric.get("_signal_groups") or [])
        signal_groups_canonical.extend(metric.get("_signal_groups_canonical") or [])
        for row in metric.get("model_results", []):
            annotations = (row.get("evalcards") or {}).get("annotations") or {}
            repro = annotations.get("reproducibility_gap")
            if repro is not None:
                row_repro_annotations.append(repro)
            prov = annotations.get("provenance")
            if prov is not None:
                row_provenance_annotations.append(prov)
    return (
        row_repro_annotations,
        row_provenance_annotations,
        signal_groups,
        signal_groups_canonical,
    )


def build_benchmark_comparability(signal_groups: list[dict]) -> dict:
    variant_divergence_groups = [
        {
            "group_id": g["group_id"],
            "model_route_id": g["model_route_id"],
            "divergence_magnitude": g["variant_divergence"]["divergence_magnitude"],
            "threshold_used": g["variant_divergence"]["threshold_used"],
            "threshold_basis": g["variant_divergence"].get("threshold_basis"),
            "differing_setup_fields": g["variant_divergence"]["differing_setup_fields"],
        }
        for g in signal_groups
        if g.get("variant_divergence")
        and g["variant_divergence"].get("has_variant_divergence")
    ]
    cross_party_divergence_groups = [
        {
            "group_id": g["group_id"],
            "model_route_id": g["model_route_id"],
            "divergence_magnitude": g["cross_party_divergence"]["divergence_magnitude"],
            "threshold_used": g["cross_party_divergence"]["threshold_used"],
            "threshold_basis": g["cross_party_divergence"].get("threshold_basis"),
            "scores_by_organization": g["cross_party_divergence"][
                "scores_by_organization"
            ],
            "differing_setup_fields": g["cross_party_divergence"][
                "differing_setup_fields"
            ],
        }
        for g in signal_groups
        if g.get("cross_party_divergence")
        and g["cross_party_divergence"].get("has_cross_party_divergence")
    ]
    return {
        "variant_divergence_groups": variant_divergence_groups,
        "cross_party_divergence_groups": cross_party_divergence_groups,
    }


def build_benchmark_comparability_canonical(
    signal_groups_canonical: list[dict],
) -> dict:
    """Cross-suite view of Signal 4 only (no variant_divergence channel).

    Sidecar to ``build_benchmark_comparability``; the canonical signal
    groups by definition span eval_summaries with the same
    ``canonical_benchmark_id`` + ``metric_key``, so each entry carries
    those identity fields alongside the divergence dict for traceability.
    """
    cross_party_divergence_groups = [
        {
            "group_id": g["group_id"],
            "model_route_id": g["model_route_id"],
            "canonical_benchmark_id": g.get("canonical_benchmark_id"),
            "metric_key": g.get("metric_key"),
            "divergence_magnitude": g["cross_party_divergence"][
                "divergence_magnitude"
            ],
            "threshold_used": g["cross_party_divergence"]["threshold_used"],
            "threshold_basis": g["cross_party_divergence"].get("threshold_basis"),
            "scores_by_organization": g["cross_party_divergence"][
                "scores_by_organization"
            ],
            "differing_setup_fields": g["cross_party_divergence"][
                "differing_setup_fields"
            ],
        }
        for g in signal_groups_canonical
        if g.get("cross_party_divergence")
        and g["cross_party_divergence"].get("has_cross_party_divergence")
    ]
    return {"cross_party_divergence_groups": cross_party_divergence_groups}


def summarize_comparability_canonical(
    signal_groups_canonical: list[dict],
) -> dict:
    """Cross-suite-only Signal 4 rollup. Mirrors the cross-party fields
    from ``signals.summarize_comparability`` but skips variant since
    canonical groups don't carry it."""
    total_groups = len(signal_groups_canonical)
    groups_with_cross_party_check = 0
    cross_party_divergent_count = 0
    for entry in signal_groups_canonical:
        cross = entry.get("cross_party_divergence") if isinstance(entry, dict) else None
        if isinstance(cross, dict):
            groups_with_cross_party_check += 1
            if cross.get("has_cross_party_divergence"):
                cross_party_divergent_count += 1
    return {
        "total_groups": total_groups,
        "groups_with_cross_party_check": groups_with_cross_party_check,
        "cross_party_divergent_count": cross_party_divergent_count,
    }


def filter_metric_summary_for_model(
    metric_summary: dict, family_id: str
) -> dict | None:
    model_results = [
        row
        for row in metric_summary.get("model_results", [])
        if as_string(row.get("model_id")) == family_id
    ]
    if not model_results:
        return None

    filtered = {
        key: value for key, value in metric_summary.items() if key != "model_results"
    }
    filtered["model_results"] = model_results
    model_route_ids = {
        as_string(row.get("model_route_id"))
        for row in model_results
        if as_string(row.get("model_route_id"))
    }
    if "_signal_groups" in filtered:
        filtered["_signal_groups"] = [
            group
            for group in filtered.get("_signal_groups") or []
            if as_string(group.get("model_route_id")) in model_route_ids
        ]
    filtered["models_count"] = len(model_results)
    filtered["top_score"] = model_results[0].get("score") if model_results else None
    return filtered


def filter_eval_summary_for_model(summary: dict, family_id: str) -> dict | None:
    root_metrics = []
    for metric in summary.get("metrics", []):
        filtered_metric = filter_metric_summary_for_model(metric, family_id)
        if filtered_metric:
            root_metrics.append(filtered_metric)

    subtasks = []
    for subtask in summary.get("subtasks", []):
        subtask_metrics = []
        for metric in subtask.get("metrics", []):
            filtered_metric = filter_metric_summary_for_model(metric, family_id)
            if filtered_metric:
                subtask_metrics.append(filtered_metric)
        if subtask_metrics:
            subtasks.append(
                {
                    **subtask,
                    "metrics": subtask_metrics,
                    "metrics_count": len(subtask_metrics),
                    "metric_names": [
                        as_string(metric.get("metric_name"))
                        for metric in subtask_metrics
                        if as_string(metric.get("metric_name"))
                    ],
                }
            )

    if not root_metrics and not subtasks:
        return None

    filtered = {
        key: value
        for key, value in summary.items()
        if key
        not in {"metrics", "subtasks", "instance_data", "models_count", "top_score"}
    }
    filtered["metrics"] = root_metrics
    filtered["subtasks"] = subtasks
    filtered["subtasks_count"] = len(subtasks)
    filtered["metrics_count"] = len(root_metrics) + sum(
        len(subtask.get("metrics", [])) for subtask in subtasks
    )
    filtered["metric_names"] = sorted(
        {as_string(metric.get("metric_name")) for metric in root_metrics}
        | {
            as_string(metric.get("metric_name"))
            for subtask in subtasks
            for metric in subtask.get("metrics", [])
        }
        - {""}
    )
    primary_metrics = root_metrics or (
        subtasks[0].get("metrics", []) if subtasks else []
    )
    filtered["primary_metric_name"] = (
        as_string(primary_metrics[0].get("metric_name")) if primary_metrics else None
    )
    filtered["models_count"] = 1
    filtered["top_score"] = (
        primary_metrics[0].get("top_score")
        if len(primary_metrics) == 1 and not subtasks
        else None
    )

    filtered_metrics: list[dict] = list(root_metrics)
    for subtask in subtasks:
        filtered_metrics.extend(subtask.get("metrics", []))
    (
        row_repro_annotations,
        row_provenance_annotations,
        signal_groups,
        signal_groups_canonical,
    ) = collect_signal_rollup_inputs(filtered_metrics)
    filtered["reproducibility_summary"] = signals.summarize_reproducibility(
        row_repro_annotations
    )
    filtered["provenance_summary"] = signals.summarize_provenance(
        row_provenance_annotations, signal_groups
    )
    filtered["comparability_summary"] = signals.summarize_comparability(signal_groups)
    filtered["cross_party_summary_canonical"] = summarize_comparability_canonical(
        signal_groups_canonical
    )
    original_annotations = (summary.get("evalcards") or {}).get("annotations") or {}
    filtered_annotations = dict(original_annotations)
    filtered_annotations["benchmark_comparability"] = build_benchmark_comparability(
        signal_groups
    )
    filtered_annotations["benchmark_comparability_canonical"] = (
        build_benchmark_comparability_canonical(signal_groups_canonical)
    )
    filtered["evalcards"] = {"annotations": filtered_annotations}

    instance_urls: set[str] = set()
    models_with_instance = 0
    for metric in root_metrics:
        for row in metric.get("model_results", []):
            url = as_string(row.get("detailed_evaluation_results"))
            if url:
                instance_urls.add(url)
            if row.get("instance_level_data") is not None:
                models_with_instance += 1
    for subtask in subtasks:
        for metric in subtask.get("metrics", []):
            for row in metric.get("model_results", []):
                url = as_string(row.get("detailed_evaluation_results"))
                if url:
                    instance_urls.add(url)
                if row.get("instance_level_data") is not None:
                    models_with_instance += 1

    filtered["instance_data"] = {
        "available": bool(instance_urls),
        "url_count": len(instance_urls),
        "sample_urls": sorted(instance_urls)[:3],
        "models_with_loaded_instances": models_with_instance,
    }
    return filtered


def generate_readme(
    manifest: dict,
    eval_list: dict,
    benchmark_metadata: dict,
    hierarchy_path: Path | None = None,
) -> str:
    """Generate a README.md for the HF dataset with full manifest and data access docs."""
    generated_at = manifest.get("generated_at", "unknown")
    model_count = manifest.get("model_count", 0)
    eval_count = manifest.get("eval_count", 0)
    metric_eval_count = manifest.get("metric_eval_count", 0)
    source_config_count = manifest.get("source_config_count", 0)

    evals = eval_list.get("evals", [])
    total_models = eval_list.get("totalModels", model_count)

    # Build a compact runtime hierarchy tree for the generated dataset README.
    hierarchy_lines = []
    if hierarchy_path and hierarchy_path.exists():
        hierarchy_data = json.loads(hierarchy_path.read_text(encoding="utf-8"))
        families = hierarchy_data.get("families", [])

        def _eval_count_label(node: dict) -> str:
            count = node.get("evals_count")
            if not isinstance(count, int) or count <= 0:
                return ""
            return f" ({count} eval{'s' if count != 1 else ''})"

        for fam in families:
            family_label = fam.get("display_name") or fam.get("key") or "Unknown"
            hierarchy_lines.append(f"- **{family_label}**{_eval_count_label(fam)}")
            for leaf in fam.get("leaves", []):
                leaf_label = leaf.get("display_name") or leaf.get("key") or "Unknown"
                hierarchy_lines.append(f"  - {leaf_label}{_eval_count_label(leaf)}")

    hierarchy_tree = (
        "\n".join(hierarchy_lines) if hierarchy_lines else "_Hierarchy not available._"
    )

    # Collect benchmark card coverage
    card_keys = sorted(benchmark_metadata.keys()) if benchmark_metadata else []

    # Build per-eval quick reference table
    eval_table_rows = []
    for e in sorted(evals, key=lambda x: x.get("eval_summary_id", "")):
        eid = e["eval_summary_id"]
        name = e.get("display_name") or e.get("evaluation_name") or eid
        mcount = e.get("models_count", 0)
        metrics = ", ".join(e.get("metric_names", []))
        has_card = "yes" if e.get("benchmark_card") else "no"
        eval_table_rows.append(
            f"| `{eid}` | {name} | {mcount} | {metrics} | {has_card} |"
        )

    eval_table = "\n".join(eval_table_rows)

    readme = f"""\
---
license: mit
pretty_name: Eval Cards Backend
tags:
  - evaluation
  - benchmarks
  - model-evaluation
  - leaderboard
size_categories:
  - 1K<n<10K
---

# Eval Cards Backend Dataset

Pre-computed evaluation data powering the Eval Cards frontend.
Generated by the [eval-cards backend pipeline](https://github.com/evaleval/eval_cards_backend_pipeline).

> Last generated: **{generated_at}**

## Quick Stats

| Stat | Value |
|------|-------|
| Models | {total_models:,} |
| Evaluations (benchmarks) | {eval_count} |
| Metric-level evaluations | {metric_eval_count} |
| Source configs processed | {source_config_count} |
| Benchmark metadata cards | {len(card_keys)} |

---

## File Structure

```
.
├── README.md                        # This file
├── manifest.json                    # Pipeline metadata & generation timestamp
├── eval-hierarchy.json              # Full benchmark hierarchy with card status
├── model-cards.json                 # Array of all model summaries
├── eval-list.json                   # Array of all evaluation summaries
├── peer-ranks.json                  # Per-benchmark model rankings (averaged across metrics)
├── benchmark-metadata.json          # Benchmark cards (methodology, ethics, etc.)
├── developers.json                  # Developer index with model counts
├── instances/
│   └── {{model_route_id}}/{{evaluation_id}}.jsonl  # Pipeline-owned instance artifacts with hierarchy keys
├── records/
│   └── {{model_route_id}}/{{evaluation_id}}.json   # Pipeline-owned source record artifacts
├── models/
│   └── {{model_route_id}}.json      # Per-model detail  ({model_count:,} files)
├── evals/
│   └── {{eval_summary_id}}.json     # Per-eval detail with full model results ({eval_count} files)
└── developers/
    └── {{slug}}.json                # Per-developer model list
```

---

## How to Fetch Data

### Base URL

All files are accessible via the HuggingFace dataset file API:

```
https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/
```

### Access Patterns

**1. Bootstrap — load the manifest and eval list**

```
GET /manifest.json           → pipeline metadata, generation timestamp
GET /eval-list.json          → all evaluations with summary stats
GET /model-cards.json        → all models with summary stats
GET /eval-hierarchy.json     → benchmark taxonomy tree
```

**2. Drill into a specific evaluation**

```
GET /evals/{{eval_summary_id}}.json
```

The `eval_summary_id` comes from `eval-list.json → evals[].eval_summary_id`.

**3. Drill into a specific model**

```
GET /models/{{model_route_id}}.json
```

The `model_route_id` comes from `model-cards.json → [].model_route_id`.
Route IDs use double-underscore as separator: `anthropic__claude-opus-4-5`.

**4. Get benchmark metadata card**

```
GET /benchmark-metadata.json → full dictionary keyed by normalized benchmark name
```

Lookup key: use the `benchmark_leaf_key` from any eval summary.

**5. Get developer model list**

```
GET /developers/{{slug}}.json
```

The slug comes from `developers.json → [].developer` (lowercased, special chars replaced).

**6. Get peer rankings**

```
GET /peer-ranks.json → {{ eval_summary_id: {{ model_id: {{ position, total }} }} }}
```

Rankings are keyed by `eval_summary_id` (single benchmark level, not metric level).
Each metric within a benchmark is ranked independently (respecting `lower_is_better`),
then ranks are **averaged across all metrics** to produce a single position per model per benchmark.

**7. Access instance-level data**

Instance-level data (individual test examples, model responses, and per-instance scores)
is available for many benchmarks. To check and access it:

1. Check `eval-list.json → evals[].instance_data.available` to see if a benchmark has instance data
2. Load the eval detail: `GET /evals/{{eval_summary_id}}.json`
3. Each `model_results[]` entry has these fields:
    - `detailed_evaluation_results`: URL to the pipeline-owned JSONL file under `instances/...` when materialized, otherwise `null`
    - `source_record_url`: URL to the pipeline-owned source record JSON under `records/...`
    - `instance_level_data`: Pre-loaded sample instances metadata and examples
    - `instance_level_data.interaction_type`: `single_turn`, `multi_turn`, or `non_interactive`
    - `instance_level_data.instance_count`: Total number of instances
    - `instance_level_data.source_url`: Full URL to the pipeline-owned JSONL file when materialized, otherwise `null`
    - `instance_level_data.instance_examples`: Up to 5 sample instance rows

The instance JSONL files are written into this dataset under `instances/...`.
Each row is the original sample row augmented with a `hierarchy` object containing
the same keys used elsewhere in the pipeline, such as `category`, `eval_summary_id`,
`metric_summary_id`, `benchmark_family_key`, `benchmark_leaf_key`, `slice_key`,
and `metric_key`.

---

## Frontend Agent Instructions

These instructions are intended for the frontend agent or anyone refactoring the
frontend data layer. The goal is to consume backend-declared hierarchy directly
and stop reconstructing benchmark structure with frontend heuristics.

### Canonical Sources

Use these fields as the source of truth:

- `eval-list.json → evals[]`: canonical benchmark list, category assignment, display
  names, summary-score flags, and sibling summary links
- `evals/{{eval_summary_id}}.json`: canonical per-benchmark hierarchy for all models
- `models/{{model_route_id}}.json → hierarchy_by_category`: canonical per-model
  hierarchy grouped the same way as `eval-list/evals`
- `model_results[].detailed_evaluation_results`: canonical URL for instance artifacts
  owned by this dataset
- `instances/...jsonl → row.hierarchy`: canonical hierarchy keys for each instance row

Treat these fields as compatibility/fallback only:

- `models/{{model_route_id}}.json → evaluations_by_category`
- `models/{{model_route_id}}.json → evaluation_summaries_by_category`
- any frontend-only benchmark/category inference such as regexes or
  `inferCategoryFromBenchmark(...)`

### Required Frontend Changes

1. Replace category inference with backend categories.
    Read `category` directly from `eval-list`, `evals`, `models/.../hierarchy_by_category`,
    and `instances/...jsonl → row.hierarchy.category`. Do not re-derive categories from names.

2. Replace benchmark grouping heuristics with backend keys.
    Use `benchmark_family_key`, `benchmark_parent_key`, `benchmark_leaf_key`,
    `slice_key`, and `metric_key` as stable grouping identifiers. Use `display_name`
    for compact UI labels and `canonical_display_name` whenever a row, metric, or
    slice label needs full benchmark context.

3. Treat summary scores as rollups, not peer benchmarks.
    If `is_summary_score` is `true`, render the node as an overall/aggregate score for
    `summary_score_for`. If `summary_eval_ids` is present on a non-summary benchmark or
    composite, use those ids to surface the related overall score in the same section.
    Do not represent `overall` as a slice, subtask, tab, or child benchmark. It is a
    rollup metric layer attached to the parent suite/benchmark. In the current
    generated artifacts, many of these rollups are already flattened into the parent
    `metrics[]`, so there may be no separate `*_overall` eval to load.

4. Use `hierarchy_by_category` for model detail pages.
    This structure is already aligned to the backend hierarchy and includes
    `eval_summary_id`, benchmark keys, subtasks, metrics, summary-score annotations,
    and instance availability. The frontend should render this structure directly rather
    than rebuilding sections from raw `evaluations_by_category` records.

5. Use pipeline-owned instance artifacts.
    The `detailed_evaluation_results` URL now points to `instances/...` within this
    dataset. Those rows already include a `hierarchy` object, so benchmark detail pages
    can attach samples directly to the active benchmark/metric without remapping names.

6. Use pipeline-owned record artifacts for provenance.
    `source_record_url` now points to a backend-owned copy of the source evaluation
    record under `records/...`. The frontend should use that instead of any upstream
    dataset links.

### Suggested Frontend Refactor Plan

1. Data client layer:
    Make the data client treat `hierarchy_by_category` and `instances/...` as canonical.
    Remove transforms whose only job is to infer suite/benchmark/category structure.

2. Model detail page:
    Render `hierarchy_by_category` directly. Each rendered benchmark section should use:
    `eval_summary_id`, `display_name`, `is_summary_score`, `summary_eval_ids`,
    `metrics[]`, and `subtasks[]`. Only render `subtasks[]` as actual benchmark
    subdivisions; never synthesize an `Overall` subtask from summary-score data.
    Render root-level `metrics[]` before or alongside `subtasks[]`; for suites like
    `ACE` and `APEX Agents`, those root metrics are where the rolled-up overall values live.

3. Benchmark detail page:
    Use `evals/{{eval_summary_id}}.json` for model comparisons and use
    `model_results[].detailed_evaluation_results` for samples. Match sample rows via
    `row.hierarchy.eval_summary_id` and `row.hierarchy.metric_summary_id`. Do not
    assume there is a distinct `*_overall` eval file; for many benchmarks the overall
    rollup is represented by the parent eval's root `metrics[]`.

4. Summary-score UI:
    Render overall/aggregate scores in a visually distinct area. They should appear as
    rollups attached to their parent suite, not as sibling leaf benchmarks alongside
    real tasks like `Corporate Lawyer` or `DIY`. If the API exposes an eval like
    `*_overall`, treat it as the source of parent-level rollup metrics, not as a
    navigable child node in the hierarchy. If there is no such eval, use the parent
    benchmark's root `metrics[]` as the rollup source.

5. Cleanup pass:
    After the refactor is stable, remove frontend code paths that reconstruct hierarchy
    from names or raw records. Likely cleanup targets include modules such as
    `hf-data.ts`, `model-data.ts`, `eval-processing.ts`, and `benchmark-detail.tsx`.

### Rendering Rules

- Prefer backend-provided `display_name` over frontend formatting.
- Prefer `canonical_display_name` for row labels in tables, breadcrumbs, tooltips,
  chips, compare dialogs, and any surface where names like `Overall` or
  `Investment Banking` would be ambiguous without benchmark context.
- Use backend keys for equality/grouping and backend names for labels.
- If `summary_eval_ids` exists, render the linked summary evals near the relevant
  parent suite or benchmark.
- `subtasks[]` should map only to true `slice_key`-backed subdivisions. A rollup
  score such as `overall` belongs in the parent benchmark's metric area, not in
  `subtasks[]`.
- If `is_summary_score` is `true`, do not count the node as a standalone benchmark in
  breadcrumb logic, hierarchy trees, or benchmark totals.
- For samples, prefer `row.hierarchy.metric_summary_id` when available; fall back to
  `row.hierarchy.eval_summary_id` only when the instance artifact does not distinguish
  between multiple metrics.

---

## Key Schemas

### model-cards.json (array)

```jsonc
{{
  "model_family_id": "anthropic/claude-opus-4-5",     // HF-style model path
  "model_route_id": "anthropic__claude-opus-4-5",     // URL-safe slug (use for file lookups)
  "model_family_name": "claude-opus-4-5...",           // Display name
  "developer": "anthropic",
  "total_evaluations": 45,
  "benchmark_count": 7,
  "benchmark_family_count": 7,
  "categories_covered": ["agentic", "other"],
  "last_updated": "2026-04-07T08:15:57Z",
  "variants": [
    {{
      "variant_key": "default",
      "variant_label": "Default",
      "evaluation_count": 38,
      "raw_model_ids": ["anthropic/claude-opus-4-5"]
    }}
  ],
  "score_summary": {{ "min": 0.0, "max": 1.0, "avg": 0.45, "count": 38 }}
}}
```

### eval-list.json

```jsonc
{{
  "totalModels": {total_models},
  "evals": [
    {{
      "eval_summary_id": "hfopenllm_v2_bbh",          // Use for /evals/ file lookup
            "benchmark": "HF Open LLM v2",                  // Canonical benchmark display name
      "benchmark_family_key": "hfopenllm",             // Family grouping key
      "benchmark_family_name": "Hfopenllm",
      "benchmark_parent_key": "hfopenllm_v2",
      "benchmark_leaf_key": "bbh",                     // Leaf benchmark
      "benchmark_leaf_name": "BBH",
      "display_name": "BBH",
            "canonical_display_name": "BBH / Accuracy",     // Full contextual label for rows/metrics
      "is_summary_score": false,                        // true = this is a rollup score across all sub-benchmarks (e.g. "Overall"), not a standalone benchmark
      "summary_score_for": null,                        // if is_summary_score, the parent benchmark_parent_key this summarises
      "summary_score_for_name": null,                   // human-readable version of summary_score_for
      "summary_eval_ids": [],                           // eval_summary_ids of any summary-score siblings (e.g. ["hfopenllm_v2_overall"])
      "category": "general",                            // High-level: reasoning, agentic, safety, knowledge, etc.
      "tags": {{                                         // From benchmark metadata cards
        "domains": ["biology", "physics"],              // Subject domains
        "languages": ["English"],                       // Languages covered
        "tasks": ["Multiple-choice QA"]                 // Task types
      }},
      "models_count": 4492,
      "metrics_count": 1,
      "metric_names": ["Accuracy"],
      "primary_metric_name": "Accuracy",
      "benchmark_card": null,                           // non-null if metadata card exists
      "top_score": 0.8269,
      "instance_data": {{                                // Instance-level data availability
        "available": true,                              // Whether any model has instance URLs
        "url_count": 42,                                // Number of distinct instance data URLs
        "sample_urls": ["https://...samples.jsonl"],    // Up to 3 example URLs
        "models_with_loaded_instances": 0               // Models with pre-loaded instance data
      }},
      "metrics": [
        {{
          "metric_summary_id": "hfopenllm_v2_bbh_accuracy",
          "metric_name": "Accuracy",
          "lower_is_better": false,
          "models_count": 4574,
          "top_score": 0.8269
        }}
      ]
    }}
  ]
}}
```

### evals/{{eval_summary_id}}.json

```jsonc
{{
  "eval_summary_id": "ace_diy",
    "benchmark": "ACE",
  "benchmark_family_key": "ace",
    "benchmark_family_name": "ACE",
    "benchmark_parent_name": "ACE",
  "benchmark_leaf_key": "diy",
  "benchmark_leaf_name": "DIY",
    "canonical_display_name": "ACE / DIY / Score",
  "source_data": {{
    "dataset_name": "ace",
    "source_type": "hf_dataset",
    "hf_repo": "Mercor/ACE"
  }},
  "benchmark_card": null,
  "metrics": [
    {{
      "metric_summary_id": "ace_diy_score",
      "metric_name": "Score",
      "metric_key": "score",
      "lower_is_better": false,
      "model_results": [                                // Sorted by rank (best first)
        {{
          "model_id": "openai/gpt-5-1",
          "model_route_id": "openai__gpt-5-1",
          "model_name": "GPT 5.1",
          "developer": "openai",
          "score": 0.56,
                    "source_record_url": "https://.../records/openai__gpt-5-1/ace_openai_gpt_5_1_1773260200.json", // Pipeline-owned source evaluation record
                    "detailed_evaluation_results": "https://.../instances/openai__gpt-5-1/ace_openai_gpt_5_1_1773260200.jsonl", // Pipeline-owned instance artifact or null
          "instance_level_data": null                                // Pre-loaded instances (if available)
        }}
      ]
    }}
  ],
  "subtasks": []                                        // Nested benchmarks for composites
}}
```

### models/{{model_route_id}}.json

```jsonc
{{
  "model_info": {{
    "name": "claude-opus-4-5",
    "id": "anthropic/claude-opus-4-5",
    "developer": "anthropic",
    "family_id": "anthropic/claude-opus-4-5",
    "family_slug": "anthropic__claude-opus-4-5",
    "variant_key": "default"
  }},
  "model_family_id": "anthropic/claude-opus-4-5",
  "model_route_id": "anthropic__claude-opus-4-5",
  "evaluations_by_category": {{
    "agentic": [ /* evaluation objects */ ],
    "other": [ /* evaluation objects */ ]
  }},
    "hierarchy_by_category": {{                        // Canonical hierarchy-aligned eval groups for this model
        "agentic": [
            {{
                "eval_summary_id": "apex_agents_corporate_lawyer",
                "benchmark_family_key": "apex_agents",
                "benchmark_parent_key": "apex_agents",
                "benchmark_leaf_key": "corporate_lawyer",
                "display_name": "Corporate Lawyer",
                "canonical_display_name": "APEX Agents / Corporate Lawyer / Mean Score",
                "is_summary_score": false,
                "summary_eval_ids": ["apex_agents_overall"],
                "metrics": [{{ "metric_summary_id": "apex_agents_corporate_lawyer_mean_score", "model_results": [{{ "score": 0.71 }}] }}],
                "subtasks": []
            }}
        ]
    }},
    "evaluation_summaries_by_category": {{ /* compatibility alias of hierarchy_by_category */ }},
  "total_evaluations": 45,
  "categories_covered": ["agentic", "other"],
  "variants": [ /* variant details */ ]
}}
```

### eval-hierarchy.json

The benchmark taxonomy tree. Each node can be a **family** (top-level grouping),
**composite** (multi-benchmark suite), or **benchmark** (leaf with metrics/slices).

Nodes with `has_card: true` have matching benchmark metadata in `benchmark-metadata.json`.

```jsonc
{{
  "generated_at": "2026-04-26T12:00:00Z",
  "signal_version": "1.0",
  "schema_note": "Runtime-generated 2-level hierarchy...",
  "families": [
    {{
      "key": "helm_classic",
      "display_name": "Helm classic",
      "category": "general",
      "evals_count": 15,
      "eval_summary_ids": ["helm_classic_boolq", "..."],
      "reproducibility_summary": {{ "results_total": 123, "...": "..." }},
      "provenance_summary": {{ "total_results": 123, "...": "..." }},
      "comparability_summary": {{ "total_groups": 123, "...": "..." }},
      "leaves": [
        {{
          "key": "boolq",
          "display_name": "BoolQ",
          "category": "general",
          "evals_count": 1,
          "eval_summary_ids": ["helm_classic_boolq"],
          "reproducibility_summary": {{ "...": "..." }},
          "provenance_summary": {{ "...": "..." }},
          "comparability_summary": {{ "...": "..." }}
        }}
      ]
    }}
  ]
}}
```

**Flattening rules:** When a family contains only one child, the child is promoted
to the family level. This means families may have their content in different shapes:

| Shape | Fields present | Meaning |
|-------|---------------|---------|
| `composites` + `standalone_benchmarks` | Multi-member family | Iterate both arrays |
| `benchmarks` | Promoted single composite | Iterate `benchmarks` directly |
| `slices` + `metrics` | Promoted single benchmark | Leaf data at top level |

---

## Evaluation Manifest

`[x]` = benchmark metadata card available, `[ ]` = no card yet

{hierarchy_tree}

---

## Evaluation Index

| eval_summary_id | Name | Models | Metrics | Card |
|----------------|------|--------|---------|------|
{eval_table}

---

## Benchmark Metadata Cards

{len(card_keys)} benchmark cards are available in `benchmark-metadata.json`:

{chr(10).join(f"- `{k}`" for k in card_keys)}

Each card contains: `benchmark_details` (name, overview, domains), `methodology` (metrics, scoring),
`purpose_and_intended_users`, `data` (size, format, sources), `ethical_and_legal_considerations`.

---

## Data Sources

| Source | HF Repo | Purpose |
|--------|---------|---------|
| Benchmark cards | `{BENCHMARK_METADATA_DATASET_REPO}` | Auto-generated benchmark metadata |
| This dataset | `{DATASET_REPO}` | Pre-computed frontend data |

---

## Pipeline

Generated by `scripts/pipeline.py`. Run locally:

```bash
# Dry run (no upload)
python -m scripts.pipeline --dry-run

# Full run with upload
HF_TOKEN=hf_xxx python -m scripts.pipeline
```

Config version: `{manifest.get("config_version", 1)}`
"""
    return readme


def resolve_upload_target() -> str:
    """Pick the upload target, refusing accidental production writes from local shells.

    Local runs must set `CARD_BACKEND_OUTPUT_REPO` to a non-production
    dataset, or set `CARD_BACKEND_ALLOW_PRODUCTION=1` for an intentional
    manual prod push. CI runs (where `GITHUB_ACTIONS=true` is set
    automatically by the GitHub-hosted runner) deploy to production by
    default — owner PR review at merge time is the gate, not the env flag.
    """
    target = (os.environ.get("CARD_BACKEND_OUTPUT_REPO") or "").strip()
    allow_production = (
        os.environ.get("CARD_BACKEND_ALLOW_PRODUCTION") == "1"
        or os.environ.get("GITHUB_ACTIONS") == "true"
    )
    if not target:
        if not allow_production:
            raise RuntimeError(
                "CARD_BACKEND_OUTPUT_REPO is required for local uploads. "
                "Set it to a non-production dataset (e.g. `j-chim/temp_evalcard_backend`); "
                "intentional local prod uploads also need CARD_BACKEND_ALLOW_PRODUCTION=1."
            )
        return PRODUCTION_DATASET_REPO
    if target == PRODUCTION_DATASET_REPO and not allow_production:
        raise RuntimeError(
            f"Refusing to upload to production target {PRODUCTION_DATASET_REPO}. "
            "Set CARD_BACKEND_ALLOW_PRODUCTION=1 to override."
        )
    return target


def upload_output() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required unless --dry-run is used")

    upload_target = resolve_upload_target()
    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=upload_target, repo_type="dataset", private=False, exist_ok=True
        )
    except Exception as error:
        print(f"create_repo warning: {error}", file=sys.stderr)

    delete_stale_remote_files(api, token, OUTPUT_DIR, repo_id=upload_target)

    api.upload_large_folder(
        repo_id=upload_target,
        repo_type="dataset",
        folder_path=str(OUTPUT_DIR),
    )


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    load_instance_in_dry_run = os.environ.get("LOAD_INSTANCE_IN_DRY_RUN") == "1"
    config_batch_size = parse_positive_int(os.environ.get("CONFIG_BATCH_SIZE"), 4)
    config_limit = os.environ.get("CONFIG_LIMIT")
    explicit_configs = [
        c.strip()
        for c in as_string(
            os.environ.get("CONFIGS") or os.environ.get("CONFIG_NAMES")
        ).split(",")
        if c.strip()
    ]
    configured_local_dataset_dir = (
        as_string(os.environ.get("EEE_LOCAL_DATASET_DIR")).strip()
        or DEFAULT_LOCAL_DATASET_DIR
    )
    configured_local_metadata_dir = (
        as_string(os.environ.get("BENCHMARK_METADATA_LOCAL_DIR")).strip()
        or DEFAULT_LOCAL_BENCHMARK_METADATA_DIR
    )
    force_refresh_snapshot = os.environ.get("EEE_REFRESH_SNAPSHOT") == "1"
    force_refresh_metadata = os.environ.get("BENCHMARK_METADATA_REFRESH") == "1"
    allow_skipped_configs = os.environ.get("ALLOW_SKIPPED_CONFIGS") == "1"
    hf_token = os.environ.get("HF_TOKEN")

    local_dataset_dir = ensure_local_dataset_snapshot(
        configured_local_dataset_dir, hf_token, force_refresh_snapshot
    )
    local_metadata_dir = ensure_local_benchmark_metadata_snapshot(
        configured_local_metadata_dir, hf_token, force_refresh_metadata
    )
    if not local_metadata_dir:
        raise RuntimeError(
            "Failed to cache benchmark metadata from evaleval/auto-benchmarkcards"
        )

    # Pick up any per-run override of `CARD_BACKEND_OUTPUT_REPO` BEFORE
    # any URL-emitting code runs. Without this re-bind, the README and
    # the URL prefix guard in `validate_output_contract` would silently
    # use the import-time value.
    reload_dataset_target()

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    random.seed(42)
    load_metric_registry()

    clean_output_dir()
    print(
        f"[pipeline] {json.dumps({'event': 'metric_registry.loaded', 'registry_path': str(DEFAULT_METRIC_REGISTRY_PATH), 'entry_count': len(METRIC_REGISTRY_ENTRIES), 'alias_count': len(METRIC_REGISTRY_ALIAS_LOOKUP)})}"
    )
    cards, metadata_lookup, benchmark_metadata = load_benchmark_metadata(
        local_metadata_dir
    )
    print(
        f"[pipeline] {json.dumps({'event': 'metadata.loaded', 'benchmark_card_count': len(cards), 'metadata_key_count': len(metadata_lookup), 'metadata_cache_dir': local_metadata_dir, 'metadata_repo': BENCHMARK_METADATA_DATASET_REPO})}"
    )

    all_configs = explicit_configs or discover_configs(local_dataset_dir, hf_token)
    if config_limit:
        all_configs = all_configs[
            : max(
                1,
                min(
                    parse_positive_int(config_limit, len(all_configs)), len(all_configs)
                ),
            )
        ]

    skipped_configs: list[str] = []
    evaluations: list[dict] = []

    for i in range(0, len(all_configs), config_batch_size):
        batch = all_configs[i : i + config_batch_size]
        print(
            f"[pipeline] {json.dumps({'event': 'config.batch.start', 'batch_index': i // config_batch_size, 'batch_size': len(batch), 'configs': batch})}"
        )

        for config in batch:
            try:
                files = list_json_files_for_config(config, local_dataset_dir, hf_token)
                print(
                    f"[pipeline] {json.dumps({'event': 'config.discovery', 'config': config, 'data_json_files_found': len(files), 'discovery_pages': 1, 'discovery_error': None})}"
                )
                loaded_rows = 0
                failed_files: list[str] = []
                for dataset_path in files:
                    record = None
                    last_error = None
                    for attempt in range(1, FILE_READ_MAX_RETRIES + 1):
                        try:
                            record = read_dataset_json(
                                dataset_path, local_dataset_dir, hf_token
                            )
                            break
                        except Exception as error:
                            last_error = error
                            if attempt < FILE_READ_MAX_RETRIES:
                                time.sleep(FILE_READ_RETRY_DELAY_SEC * attempt)

                    if record is None:
                        failed_files.append(dataset_path)
                        print(
                            f"[pipeline] {json.dumps({'event': 'file.load.error', 'config': config, 'path': dataset_path, 'message': str(last_error) if last_error else 'unknown'})}"
                        )
                        continue

                    source_record_url = raw_url_for_dataset_path(dataset_path)
                    eval_results = (
                        record.get("evaluation_results")
                        if isinstance(record.get("evaluation_results"), list)
                        else []
                    )
                    first_result = eval_results[0] if eval_results else None
                    benchmark = (
                        as_string(record.get("evaluation_id")).split("/")[0]
                        if record.get("evaluation_id")
                        else None
                    )
                    passthrough = {
                        k: v for k, v in record.items() if k not in KNOWN_TOP_LEVEL_KEYS
                    }
                    detailed_meta = normalize_detailed_eval_meta(
                        record.get("detailed_evaluation_results")
                    )
                    eval_obj = {
                        "schema_version": as_string(record.get("schema_version")),
                        "evaluation_id": as_string(record.get("evaluation_id")),
                        "retrieved_timestamp": as_string(
                            record.get("retrieved_timestamp")
                        ),
                        "benchmark": benchmark,
                        "source_data": (first_result or {}).get("source_data"),
                        "source_metadata": record.get("source_metadata"),
                        "eval_library": record.get("eval_library"),
                        "model_info": record.get("model_info") or {},
                        "generation_config": (first_result or {}).get(
                            "generation_config"
                        ),
                        "source_record_url": source_record_url,
                        "detailed_evaluation_results_meta": detailed_meta,
                        "detailed_evaluation_results": resolve_detailed_results_url(
                            record, source_record_url
                        ),
                        "passthrough_top_level_fields": passthrough or None,
                        "evaluation_results": eval_results,
                        "benchmark_card": None,
                        "instance_level_data": None,
                        "_raw_record_payload": record,
                        # Raw EEE config name with original punctuation
                        # (e.g. ``tau-bench-2_airline``); the registry's
                        # scoped_aliases keys preserve hyphens, so the
                        # pipeline-normalized ``benchmark_family_key``
                        # cannot be used for resolution.
                        "_eee_source_config": config,
                    }
                    evaluations.append(eval_obj)
                    loaded_rows += 1

                if failed_files:
                    message = (
                        f"Failed to load {len(failed_files)} files for config {config}"
                    )
                    print(
                        f"[pipeline] {json.dumps({'event': 'config.load.partial', 'config': config, 'row_count': loaded_rows, 'failed_files': len(failed_files), 'sample_failed_paths': failed_files[:5]})}"
                    )
                    if not allow_skipped_configs:
                        raise RuntimeError(message)

                print(
                    f"[pipeline] {json.dumps({'event': 'config.load.ok', 'config': config, 'discovered_data_json_files': len(files), 'discovery_pages': 1, 'row_count': loaded_rows})}"
                )
            except Exception as error:
                print(
                    f"[pipeline] {json.dumps({'event': 'config.load.error', 'config': config, 'message': str(error)})}"
                )
                if allow_skipped_configs:
                    print(f"Skipping config {config}: {error}", file=sys.stderr)
                    skipped_configs.append(config)
                else:
                    raise

        print(
            f"[pipeline] {json.dumps({'event': 'config.batch.done', 'batch_index': i // config_batch_size, 'cumulative_evaluations': len(evaluations), 'cumulative_skipped': len(skipped_configs)})}"
        )

    if (not dry_run) or load_instance_in_dry_run:
        with_instance = 0
        missing_instance = 0
        for idx, evaluation in enumerate(evaluations, start=1):
            instance_data = maybe_load_instance_data(
                evaluation, local_dataset_dir, hf_token
            )
            if instance_data:
                evaluation["instance_level_data"] = instance_data
                with_instance += 1
            else:
                missing_instance += 1
            if idx % 100 == 0 or idx == len(evaluations):
                print(
                    f"[pipeline] {json.dumps({'event': 'instance.batch.progress', 'processed': idx, 'total': len(evaluations), 'with_instance_data': with_instance, 'missing_instance_data': missing_instance})}"
                )
        print(
            f"[pipeline] {json.dumps({'event': 'instance.load.summary', 'total': len(evaluations), 'with_instance_data': with_instance, 'missing_instance_data': missing_instance})}"
        )

    for evaluation in evaluations:
        raw_model_info = evaluation.get("model_info") or {}
        identity = canonical_model_identity(raw_model_info)
        display_identity = aggregated_display_identity(raw_model_info)
        model_info = dict(evaluation.get("model_info") or {})
        model_info.update(
            {
                "normalized_id": identity["normalized_id"],
                "family_id": display_identity["family_id"],
                "family_slug": display_identity["family_slug"],
                "family_name": display_identity["family_name"],
                "variant_key": display_identity["variant_key"],
                "variant_label": display_identity["variant_label"],
                "model_route_id": display_identity["model_route_id"],
            }
        )
        evaluation["model_info"] = model_info
        # Direct per-evaluation card lookup using the EEE benchmark name —
        # behavior unchanged for any benchmark whose name matches a card
        # in `metadata_lookup`.
        evaluation["benchmark_card"] = lookup_benchmark_card(
            metadata_lookup,
            evaluation.get("benchmark"),
            canonical_benchmark_family_key(evaluation.get("benchmark")),
        )

        # Fallback ONLY when the direct lookup misses: try the shared
        # `dataset_name` from `evaluation_results[*].source_data`. This
        # covers benchmarks whose ABC card name diverges from the EEE
        # benchmark name (e.g. EEE `fibble_arena` ↔ ABC
        # `fibble_arena_daily`) — without it, `benchmark_card` stays
        # None, `top_level_benchmark_owns_slices` returns False, and
        # `classify_evaluation_result` promotes each sub-variant to its
        # own leaf instead of treating it as a slice of the family.
        #
        # Two guards keep the fallback from misfiring:
        #   1. **Shared across all results.** Aggregator records (e.g.
        #      `llm-stats` wrapping 8-12+ scraped benchmarks per record)
        #      scatter their per-result `dataset_name`s; we only use one
        #      when every result agrees, leaving `shared_dataset_name`
        #      None for aggregators.
        #   2. **Compact-key relatedness.** Even a shared dataset_name
        #      must overlap the EEE benchmark by compact-key substring
        #      (`fibble_arena` ⊂ `fibble_arena_daily` ✓). A single-record
        #      aggregator (e.g. an llm-stats record covering only
        #      `ARC-AGI v2`) has `llmstats` vs `arcagiv2` — unrelated, so
        #      we don't pull in an unrelated child card and accidentally
        #      flip every result onto the `owns_slices` path.
        if evaluation["benchmark_card"] is None:
            unique_dataset_names = {
                (r.get("source_data") or {}).get("dataset_name")
                for r in (evaluation.get("evaluation_results") or [])
                if isinstance(r, dict) and isinstance(r.get("source_data"), dict)
            }
            unique_dataset_names.discard(None)
            unique_dataset_names.discard("")
            shared_dataset_name = (
                next(iter(unique_dataset_names))
                if len(unique_dataset_names) == 1
                else None
            )
            eee_compact = compact_benchmark_key(evaluation.get("benchmark"))
            ds_compact = (
                compact_benchmark_key(shared_dataset_name)
                if shared_dataset_name
                else ""
            )
            if (
                eee_compact
                and ds_compact
                and (eee_compact in ds_compact or ds_compact in eee_compact)
            ):
                evaluation["benchmark_card"] = lookup_benchmark_card(
                    metadata_lookup,
                    evaluation.get("benchmark"),
                    canonical_benchmark_family_key(evaluation.get("benchmark")),
                    shared_dataset_name,
                )
        enriched_results = []
        for result in evaluation.get("evaluation_results") or []:
            enriched = dict(result)
            normalized = classify_evaluation_result(
                evaluation, enriched, evaluation["benchmark_card"]
            )
            enriched["normalized_result"] = normalized
            enriched_results.append(enriched)
        evaluation["evaluation_results"] = enriched_results

        evaluation["eval_summary_ids"] = sorted(
            {
                get_eval_group_id(evaluation, result)
                for result in evaluation.get("evaluation_results") or []
                if extract_score(result) is not None
            }
        )

        raw_record_payload = evaluation.pop("_raw_record_payload", None)
        record_relative_path = build_record_artifact_relative_path(evaluation)
        record_artifact_url = dataset_resolve_url(record_relative_path)
        evaluation["source_record_url"] = record_artifact_url

        raw_instance_url = as_string(evaluation.get("detailed_evaluation_results"))
        if raw_instance_url:
            # Only publish pipeline-owned instance artifact URLs. If we cannot
            # materialize the upstream samples into output/instances, clear the
            # public field instead of leaking raw source-dataset URLs.
            evaluation["detailed_evaluation_results"] = None
            artifact_text = read_text_from_dataset_url(
                raw_instance_url, local_dataset_dir, hf_token
            )
            if artifact_text is not None:
                instance_relative_path = build_instance_artifact_relative_path(
                    evaluation
                )
                instance_output_path = OUTPUT_DIR / instance_relative_path
                instance_output_path.parent.mkdir(parents=True, exist_ok=True)
                transformed_artifact_text = transform_instance_artifact_text(
                    evaluation, artifact_text
                )
                instance_output_path.write_text(
                    transformed_artifact_text, encoding="utf-8"
                )
                pipeline_instance_url = dataset_resolve_url(instance_relative_path)
                evaluation["detailed_evaluation_results"] = pipeline_instance_url
                evaluation["instance_artifact"] = {
                    "path": instance_relative_path,
                    "url": pipeline_instance_url,
                    "eval_summary_ids": evaluation.get("eval_summary_ids", []),
                }
                if evaluation.get("instance_level_data"):
                    evaluation["instance_level_data"] = {
                        **(evaluation.get("instance_level_data") or {}),
                        "source_url": pipeline_instance_url,
                        "instance_examples": [
                            annotate_instance_row(evaluation, row)
                            if isinstance(row, dict)
                            else row
                            for row in (
                                (evaluation.get("instance_level_data") or {}).get(
                                    "instance_examples"
                                )
                                or []
                            )
                        ],
                    }
            elif evaluation.get("instance_level_data"):
                evaluation["instance_level_data"] = {
                    **(evaluation.get("instance_level_data") or {}),
                    "source_url": None,
                }

        if raw_record_payload is not None:
            record_output_payload = dict(raw_record_payload)
            record_output_payload["source_record_url"] = record_artifact_url
            record_output_payload["detailed_evaluation_results"] = evaluation.get(
                "detailed_evaluation_results"
            )
            if evaluation.get("detailed_evaluation_results_meta") is not None:
                record_output_payload["detailed_evaluation_results_meta"] = (
                    evaluation.get("detailed_evaluation_results_meta")
                )
            write_json(OUTPUT_DIR / record_relative_path, record_output_payload)

    benchmark_groups: dict[str, dict] = {}
    model_family_groups: dict[str, list[dict]] = defaultdict(list)

    for evaluation in evaluations:
        family_id = as_string((evaluation.get("model_info") or {}).get("family_id"))
        if family_id:
            model_family_groups[family_id].append(evaluation)

        for result in evaluation.get("evaluation_results") or []:
            score = extract_score(result)
            if score is None:
                continue
            normalized = result.get("normalized_result") or {}
            eval_group_id = get_eval_group_id(evaluation, result)
            metric_summary_id = get_metric_summary_id(evaluation, result)
            group = benchmark_groups.setdefault(
                eval_group_id,
                {
                    "eval_summary_id": eval_group_id,
                    "benchmark": normalized.get("benchmark_parent_name")
                    or evaluation.get("benchmark"),
                    "benchmark_family_key": normalized.get("benchmark_family_key"),
                    "benchmark_family_name": normalized.get("benchmark_family_name"),
                    "benchmark_parent_key": normalized.get("benchmark_parent_key"),
                    "benchmark_parent_name": normalized.get("benchmark_parent_name"),
                    "benchmark_leaf_key": normalized.get("benchmark_leaf_key"),
                    "benchmark_leaf_name": normalized.get("benchmark_leaf_name"),
                    "benchmark_component_key": normalized.get(
                        "benchmark_component_key"
                    ),
                    "benchmark_component_name": normalized.get(
                        "benchmark_component_name"
                    ),
                    "evaluation_name": normalized.get("benchmark_leaf_name")
                    or normalized.get("benchmark_family_name"),
                    "display_name": normalized.get("benchmark_leaf_name")
                    or normalized.get("benchmark_family_name"),
                    "canonical_display_name": normalized.get("benchmark_leaf_name")
                    or normalized.get("benchmark_parent_name")
                    or normalized.get("benchmark_family_name"),
                    "is_summary_score": bool(normalized.get("is_summary_score")),
                    "category": infer_category_from_benchmark(
                        as_string(evaluation.get("benchmark"))
                    ),
                    "source_data": result.get("source_data"),
                    "benchmark_card": None,
                    "tags": {"domains": [], "languages": [], "tasks": []},
                    "subtasks": {},
                    "_source_metadata_aggregate": {},
                    "_eee_source_config": evaluation.get("_eee_source_config"),
                },
            )

            evaluation_source_metadata = evaluation.get("source_metadata")
            if isinstance(evaluation_source_metadata, dict):
                aggregate = group["_source_metadata_aggregate"]
                for sm_key, sm_value in evaluation_source_metadata.items():
                    if sm_value is not None and sm_key not in aggregate:
                        aggregate[sm_key] = sm_value

            # Set benchmark card and tags on first encounter
            if group["benchmark_card"] is None:
                _card = lookup_benchmark_card_for_parent(
                    metadata_lookup,
                    normalized.get("benchmark_leaf_name"),
                    normalized.get("benchmark_leaf_key"),
                    aux_values=(
                        evaluation.get("benchmark"),
                        normalized.get("benchmark_family_key"),
                        (result.get("source_data") or {}).get("dataset_name"),
                    ),
                    parent_values=(
                        normalized.get("benchmark_parent_key"),
                        normalized.get("benchmark_parent_name"),
                        evaluation.get("benchmark"),
                        normalized.get("benchmark_family_key"),
                    ),
                )
                if _card:
                    group["benchmark_card"] = _card
                    group["tags"] = extract_benchmark_tags(_card)
                    # Re-derive category from card domains (more accurate than name regex)
                    group["category"] = infer_category_from_benchmark(
                        as_string(evaluation.get("benchmark")), _card
                    )

            subtask_key = as_string(normalized.get("slice_key") or "__root__")
            subtask = group["subtasks"].setdefault(
                subtask_key,
                {
                    "subtask_key": None
                    if subtask_key == "__root__"
                    else normalized.get("slice_key"),
                    "subtask_name": normalized.get("slice_name"),
                    "display_name": normalized.get("slice_name")
                    or normalized.get("benchmark_leaf_name")
                    or normalized.get("benchmark_family_name"),
                    "canonical_display_name": join_display_name_parts(
                        normalized.get("benchmark_leaf_name")
                        or normalized.get("benchmark_parent_name")
                        or normalized.get("benchmark_family_name"),
                        normalized.get("slice_name"),
                    ),
                    "metrics": {},
                },
            )

            metric_summary = subtask["metrics"].setdefault(
                metric_summary_id,
                {
                    "metric_summary_id": metric_summary_id,
                    "legacy_eval_summary_id": slugify(
                        f"{evaluation.get('benchmark') or ((result.get('source_data') or {}).get('dataset_name')) or 'unknown'}__{result.get('evaluation_name') or 'unknown'}"
                    ),
                    "evaluation_name": result.get("evaluation_name"),
                    "display_name": " / ".join(
                        [
                            part
                            for part in [
                                normalized.get("benchmark_leaf_name"),
                                normalized.get("slice_name"),
                                normalized.get("metric_name"),
                            ]
                            if part
                        ]
                    ),
                    "canonical_display_name": normalized.get("canonical_display_name"),
                    "benchmark_leaf_key": normalized.get("benchmark_leaf_key"),
                    "benchmark_leaf_name": normalized.get("benchmark_leaf_name"),
                    "slice_key": normalized.get("slice_key"),
                    "slice_name": normalized.get("slice_name"),
                    "lower_is_better": bool(
                        (result.get("metric_config") or {}).get("lower_is_better")
                    ),
                    "metric_name": normalized.get("metric_name"),
                    "metric_id": normalized.get("metric_id"),
                    "metric_key": normalized.get("metric_key"),
                    "metric_source": normalized.get("metric_source"),
                    "metric_config": result.get("metric_config"),
                    "model_results": [],
                },
            )
            generation_args = (result.get("generation_config") or {}).get(
                "generation_args"
            )
            if not isinstance(generation_args, dict):
                generation_args = None
            agentic_for_spec = signals.is_agentic(
                evaluation.get("benchmark"),
                evaluation.get("benchmark_card"),
                generation_args,
            )
            reproducibility_gap = signals.compute_reproducibility_gap(
                generation_args, agentic_for_spec
            )

            metric_summary["model_results"].append(
                {
                    "model_id": as_string(
                        (evaluation.get("model_info") or {}).get("family_id")
                    ),
                    "model_route_id": as_string(
                        (evaluation.get("model_info") or {}).get("model_route_id")
                    ),
                    "model_name": as_string(
                        (evaluation.get("model_info") or {}).get("family_name")
                        or (evaluation.get("model_info") or {}).get("name")
                    ),
                    "developer": as_string(
                        (evaluation.get("model_info") or {}).get("developer")
                    ),
                    "variant_key": as_string(
                        (evaluation.get("model_info") or {}).get("variant_key")
                    )
                    or "default",
                    "raw_model_id": as_string(
                        (evaluation.get("model_info") or {}).get("id")
                    ),
                    "score": score,
                    "evaluation_id": evaluation.get("evaluation_id"),
                    "retrieved_timestamp": evaluation.get("retrieved_timestamp"),
                    # Carry provenance straight onto each per-model row so
                    # downstream consumers don't have to re-join against the
                    # parent evaluation record. Without this, the hierarchy
                    # view loses the evaluator_relationship / source_organization
                    # context and all 1st/3rd-party badges collapse to "other".
                    "source_metadata": evaluation.get("source_metadata"),
                    "source_data": evaluation.get("source_data"),
                    "source_record_url": evaluation.get("source_record_url"),
                    "detailed_evaluation_results": evaluation.get(
                        "detailed_evaluation_results"
                    ),
                    "detailed_evaluation_results_meta": evaluation.get(
                        "detailed_evaluation_results_meta"
                    ),
                    "passthrough_top_level_fields": evaluation.get(
                        "passthrough_top_level_fields"
                    ),
                    "instance_level_data": evaluation.get("instance_level_data"),
                    "normalized_result": normalized,
                    "evalcards": {
                        "annotations": {
                            "reproducibility_gap": reproducibility_gap,
                        }
                    },
                    # Carried per row for variant_divergence comparison,
                    # cross_party representative-setup.
                    "_generation_args": generation_args,
                }
            )

    peer_ranks: dict[str, dict[str, dict[str, int]]] = {}
    eval_summaries: list[dict] = []

    for summary in benchmark_groups.values():
        root_metrics: list[dict] = []
        subtask_summaries: list[dict] = []
        model_ids_for_group: set[str] = set()
        unique_metric_names: set[str] = set()
        total_metric_count = 0

        for subtask in summary["subtasks"].values():
            metric_summaries: list[dict] = []
            for metric_summary in subtask["metrics"].values():
                lower = bool(metric_summary.get("lower_is_better"))
                model_results = sorted(
                    metric_summary["model_results"],
                    key=lambda r: (r["score"], r["model_id"]),
                )
                if not lower:
                    model_results.reverse()
                metric_summary["model_results"] = model_results
                metric_summary["models_count"] = len(model_results)
                metric_summary["top_score"] = (
                    model_results[0]["score"] if model_results else None
                )

                # Compute provenance + variant_divergence + cross_party_divergence per group,
                # and attach annotations to each row.
                # Group-level summaries are stashed on the metric_summary for downstream aggregation
                attach_comparability_signals(metric_summary)

                metric_summaries.append(metric_summary)
                total_metric_count += 1
                unique_metric_names.add(as_string(metric_summary.get("metric_name")))
                model_ids_for_group.update(row["model_id"] for row in model_results)

                ranks: dict[str, dict[str, int]] = {}
                position = 0
                previous_score = None
                for idx, row in enumerate(model_results, start=1):
                    if previous_score is None or row["score"] != previous_score:
                        position = idx
                        previous_score = row["score"]
                    rank_entry = {"position": position, "total": len(model_results)}
                    ranks[row["model_id"]] = rank_entry
                    raw_id = row.get("raw_model_id", "")
                    if raw_id and raw_id != row["model_id"]:
                        ranks[raw_id] = rank_entry
                metric_summary["_ranks"] = ranks
                metric_summary["_total"] = len(model_results)

            metric_summaries.sort(
                key=lambda metric: (
                    as_string(metric.get("metric_name")),
                    as_string(metric.get("metric_summary_id")),
                )
            )
            if subtask.get("subtask_key") is None:
                root_metrics = metric_summaries
            else:
                subtask_summaries.append(
                    {
                        "subtask_key": subtask.get("subtask_key"),
                        "subtask_name": subtask.get("subtask_name"),
                        "display_name": subtask.get("display_name"),
                        "metrics": metric_summaries,
                        "metrics_count": len(metric_summaries),
                        "metric_names": [
                            as_string(metric.get("metric_name"))
                            for metric in metric_summaries
                        ],
                    }
                )

        subtask_summaries.sort(
            key=lambda subtask: as_string(subtask.get("display_name"))
        )
        summary["metrics"] = root_metrics
        summary["subtasks"] = subtask_summaries
        summary["subtasks_count"] = len(subtask_summaries)
        summary["metrics_count"] = total_metric_count
        summary["models_count"] = len(model_ids_for_group)
        summary["metric_names"] = sorted(name for name in unique_metric_names if name)
        primary_metrics = root_metrics or (
            subtask_summaries[0]["metrics"] if subtask_summaries else []
        )
        primary_metric = pick_primary_metric(primary_metrics)
        summary["primary_metric_name"] = (
            as_string(primary_metric.get("metric_name")) if primary_metric else None
        )
        summary["top_score"] = (
            primary_metric.get("top_score")
            if primary_metric and len(primary_metrics) == 1 and not subtask_summaries
            else None
        )

        # Peer ranks at single-benchmark level: average rank across all metrics
        all_metric_summaries = list(root_metrics)
        for st in subtask_summaries:
            all_metric_summaries.extend(st.get("metrics", []))
        metrics_with_ranks = [m for m in all_metric_summaries if m.get("_ranks")]
        if metrics_with_ranks:
            # Collect per-model rank positions across all metrics
            model_rank_sums: dict[str, list[float]] = defaultdict(list)
            max_total = 0
            for m in metrics_with_ranks:
                total = m.get("_total", 0)
                if total > max_total:
                    max_total = total
                for model_id, rank_info in m["_ranks"].items():
                    model_rank_sums[model_id].append(rank_info["position"])
            # Average and re-rank
            avg_ranks = []
            for model_id, positions in model_rank_sums.items():
                avg_ranks.append((sum(positions) / len(positions), model_id))
            avg_ranks.sort()
            benchmark_ranks: dict[str, dict[str, int]] = {}
            position = 0
            previous_avg = None
            for idx, (avg, model_id) in enumerate(avg_ranks, start=1):
                if previous_avg is None or avg != previous_avg:
                    position = idx
                    previous_avg = avg
                benchmark_ranks[model_id] = {
                    "position": position,
                    "total": len(avg_ranks),
                }
            peer_ranks[summary["eval_summary_id"]] = benchmark_ranks

        # Summarise instance-level data availability across all model results
        instance_urls: set[str] = set()
        models_with_instance = 0
        models_without_instance = 0
        for ms in all_metric_summaries:
            for row in ms.get("model_results", []):
                url = as_string(row.get("detailed_evaluation_results"))
                has_instance = row.get("instance_level_data") is not None
                if url:
                    instance_urls.add(url)
                if has_instance:
                    models_with_instance += 1
                elif url:
                    models_without_instance += 1
        summary["instance_data"] = {
            "available": bool(instance_urls),
            "url_count": len(instance_urls),
            "sample_urls": sorted(instance_urls)[:3],
            "models_with_loaded_instances": models_with_instance,
        }

        eval_summaries.append(summary)

    # Second pass: link summary scores to their sibling eval groups so the frontend
    # knows which groups roll up into which summary and vice-versa.
    parent_to_summary_eval_ids: dict[str, list[str]] = defaultdict(list)
    for summary in eval_summaries:
        if summary.get("is_summary_score"):
            parent_key = as_string(summary.get("benchmark_parent_key"))
            if parent_key:
                parent_to_summary_eval_ids[parent_key].append(
                    summary["eval_summary_id"]
                )

    for summary in eval_summaries:
        if summary.get("is_summary_score"):
            summary["summary_score_for"] = as_string(
                summary.get("benchmark_parent_key")
            )
            summary["summary_score_for_name"] = as_string(
                summary.get("benchmark_parent_name")
            )
        else:
            parent_key = as_string(summary.get("benchmark_parent_key"))
            sibling_summary_ids = parent_to_summary_eval_ids.get(parent_key, [])
            if sibling_summary_ids:
                summary["summary_eval_ids"] = sibling_summary_ids

    # Resolve each eval_summary against the registry so cross-suite
    # duplicates (the four ``helm_*_mmlu`` summaries, the three MMLU-Pro
    # spellings, etc.) carry a shared ``canonical_benchmark_id``. The
    # registry's scoped_aliases keys preserve original EEE punctuation,
    # so ``_eee_source_config`` (raw config name) is the lookup scope —
    # the pipeline-normalized ``benchmark_family_key`` would miss
    # hyphenated suites like ``tau-bench-2_airline``.
    canonical_resolution_count = 0
    for summary in eval_summaries:
        raw_value = (
            as_string(summary.get("benchmark_leaf_name"))
            or as_string(summary.get("benchmark_leaf_key"))
            or None
        )
        resolution = registry.resolve_benchmark(
            raw_value, summary.get("_eee_source_config")
        )
        summary["canonical_benchmark_id"] = resolution["canonical_id"]
        summary["canonical_benchmark_resolution"] = resolution
        if resolution["canonical_id"]:
            canonical_resolution_count += 1
    print(
        f"[pipeline] {json.dumps({'event': 'registry.eval_summaries_resolved', 'total': len(eval_summaries), 'resolved': canonical_resolution_count})}"
    )

    # Cross-suite Signal 4: re-group metric_summaries by
    # (canonical_benchmark_id, metric_key) and re-run cross_party
    # divergence on the union of rows. Result is attached as a sidecar
    # annotation (``cross_party_divergence_canonical``) so consumers can
    # compare against the legacy per-suite scope before we deprecate it.
    attach_canonical_cross_party_signals(eval_summaries)

    # ------------------------------------------------------------------
    # HOTFIX (2026-04-30): the ABC cards for ``helm_capabilities`` and
    # ``helm-instruct`` both carry ``benchmark_details.name = "Holistic
    # Evaluation of Language Models (HELM)"`` (the umbrella project
    # name, not the suite-specific name). The pipeline propagates this
    # card name into the suite-aggregate eval_summary's display fields,
    # so two top-level eval_summaries end up with byte-identical
    # ``display_name`` and the frontend renders them as visible
    # duplicates.
    #
    # The proper fix is either (a) the deferred family-collapse work
    # that uses registry parent_benchmark_id to collapse helm_* siblings
    # into one HELM family node, or (b) fixing the ABC cards upstream.
    # This hotfix is a targeted post-processing rename: it touches ONLY
    # the two eval_summaries whose ``display_name`` matches the
    # umbrella string AND whose ``benchmark_family_key`` is
    # ``helm_capabilities`` / ``helm_instruct``. Sub-benchmark
    # eval_summaries under those families (helm_capabilities_mmlu_pro,
    # helm_instruct_vicuna, etc.) keep their existing display names
    # because the rename is gated on the umbrella string match.
    #
    # Earlier attempt: adding ``helm_capabilities`` /
    # ``helm_instruct`` to PREFERRED_BENCHMARK_DISPLAY_NAMES caused a
    # 60-field regression — that lookup is consulted from many call
    # sites including child-row leaf-name resolution. Hotfix lives
    # here, post-construction, where it can be applied surgically to
    # exactly the two affected summaries. Remove this block when the
    # family-collapse work lands.
    _HELM_UMBRELLA_DISPLAY = "Holistic Evaluation of Language Models (HELM)"
    _HELM_HOTFIX_DISPLAYS = {
        "helm_capabilities": "HELM Capabilities",
        "helm_instruct": "HELM Instruct",
    }
    for summary in eval_summaries:
        family_key = as_string(summary.get("benchmark_family_key"))
        replacement = _HELM_HOTFIX_DISPLAYS.get(family_key)
        if not replacement:
            continue
        if as_string(summary.get("display_name")) != _HELM_UMBRELLA_DISPLAY:
            continue
        summary["display_name"] = replacement
        # Keep the four name fields aligned so downstream artifacts
        # (eval-hierarchy family display, comparison-index entry,
        # eval-list summary header) all show the disambiguated label.
        # Only overwrite when the existing value is the umbrella
        # string — a sub-benchmark row that escaped the gate would
        # have its own distinct value here.
        for field in (
            "benchmark_family_name",
            "benchmark_parent_name",
            "benchmark_leaf_name",
            "canonical_display_name",
        ):
            if as_string(summary.get(field)) == _HELM_UMBRELLA_DISPLAY:
                summary[field] = replacement

    eval_summaries.sort(
        key=lambda s: (-s.get("models_count", 0), as_string(s.get("eval_summary_id")))
    )

    comparison_index = build_comparison_index(eval_summaries, started_at)

    # Strip temporary fields from metric summaries before serialization
    for summary in eval_summaries:
        for metric in summary.get("metrics", []):
            metric.pop("_ranks", None)
            metric.pop("_total", None)
        for subtask in summary.get("subtasks", []):
            for metric in subtask.get("metrics", []):
                metric.pop("_ranks", None)
                metric.pop("_total", None)

    # Attach interpretive-signal annotations to each eval summary:
    # - reporting_completeness at the benchmark level
    # - benchmark_comparability listing divergent groups
    # - reproducibility_summary, provenance_summary, comparability_summary
    #   aggregates over the eval's rows / groups.
    for summary in eval_summaries:
        joined_record = {
            "autobenchmarkcard": summary.get("benchmark_card") or {},
            "eee_eval": {
                "source_metadata": summary.pop("_source_metadata_aggregate", {}) or {}
            },
            "evalcards": {},
        }
        completeness = signals.compute_reporting_completeness(joined_record)

        all_metrics: list[dict] = list(summary.get("metrics", []))
        for subtask in summary.get("subtasks", []):
            all_metrics.extend(subtask.get("metrics", []))

        (
            row_repro_annotations,
            row_provenance_annotations,
            signal_groups,
            signal_groups_canonical,
        ) = collect_signal_rollup_inputs(all_metrics)

        repro_summary = signals.summarize_reproducibility(row_repro_annotations)
        provenance_summary = signals.summarize_provenance(
            row_provenance_annotations, signal_groups
        )
        comparability_summary = signals.summarize_comparability(signal_groups)
        cross_party_summary_canonical = summarize_comparability_canonical(
            signal_groups_canonical
        )

        summary["evalcards"] = {
            "annotations": {
                "reporting_completeness": completeness,
                "benchmark_comparability": build_benchmark_comparability(signal_groups),
                "benchmark_comparability_canonical": (
                    build_benchmark_comparability_canonical(signal_groups_canonical)
                ),
            }
        }
        summary["reproducibility_summary"] = repro_summary
        summary["provenance_summary"] = provenance_summary
        summary["comparability_summary"] = comparability_summary
        summary["cross_party_summary_canonical"] = cross_party_summary_canonical

    aggregated_model_family_groups: dict[str, list[dict]] = defaultdict(list)
    for family_evals in model_family_groups.values():
        for evaluation in family_evals:
            display_identity = aggregated_display_identity(
                evaluation.get("model_info") or {}
            )
            aggregated_model_family_groups[display_identity["family_id"]].append(
                evaluation
            )

    model_summaries: list[dict] = []
    model_cards: list[dict] = []

    for family_id, family_evals in aggregated_model_family_groups.items():
        family_evals_sorted = sorted(
            family_evals, key=lambda e: as_string(e.get("retrieved_timestamp"))
        )
        latest = family_evals_sorted[-1]
        model_info = latest.get("model_info") or {}
        display_identity = aggregated_display_identity(model_info)
        route_id = as_string(
            display_identity.get("model_route_id") or family_id.replace("/", "__")
        )
        family_name = as_string(
            display_identity.get("family_name")
            or model_info.get("family_name")
            or model_info.get("name")
            or family_id.split("/")[-1]
        )
        params_billions: float | None = None

        by_category: dict[str, list[dict]] = defaultdict(list)
        raw_model_ids = sorted(
            {
                as_string((e.get("model_info") or {}).get("id"))
                for e in family_evals
                if as_string((e.get("model_info") or {}).get("id"))
            }
        )
        variants_map: dict[str, dict] = {}
        score_values: list[float] = []
        last_updated = None

        # ---- FIX 2: collect per-benchmark scores for model card ----
        benchmark_names_set: set[str] = set()
        # key = eval_summary_id, value = best score entry for that metric
        best_per_metric: dict[str, dict] = {}

        for evaluation in family_evals:
            category = infer_category_from_benchmark(
                as_string(evaluation.get("benchmark"))
            )
            by_category[category].append(evaluation)
            iso = iso_from_epoch_string(evaluation.get("retrieved_timestamp"))
            last_updated = max_iso(last_updated, iso)

            if params_billions is None:
                params_billions = derive_model_params_billions(
                    evaluation.get("model_info") or {}
                )

            evaluation_display_identity = aggregated_display_identity(
                evaluation.get("model_info") or {}
            )
            model_variant_key = as_string(
                evaluation_display_identity.get("variant_key") or "default"
            )
            variant = variants_map.setdefault(
                model_variant_key,
                {
                    "variant_key": model_variant_key,
                    "variant_label": as_string(
                        evaluation_display_identity.get("variant_label") or "Default"
                    ),
                    "evaluation_count": 0,
                    "raw_model_ids": set(),
                    "last_updated": None,
                },
            )
            variant["evaluation_count"] += 1
            raw_id = as_string((evaluation.get("model_info") or {}).get("id"))
            if raw_id:
                variant["raw_model_ids"].add(raw_id)
            variant["last_updated"] = max_iso(variant["last_updated"], iso)

            for result in evaluation.get("evaluation_results") or []:
                normalized = result.get("normalized_result") or {}
                benchmark_display_name = as_string(
                    normalized.get("benchmark_parent_name")
                    or normalized.get("benchmark_family_name")
                )
                if benchmark_display_name:
                    benchmark_names_set.add(benchmark_display_name)
                score = extract_score(result)
                if score is not None:
                    score_values.append(score)

                    # Track best score per eval_summary_id for the model card
                    esid = get_eval_group_id(evaluation, result)
                    metric_config = result.get("metric_config") or {}
                    lower_is_better = bool(metric_config.get("lower_is_better"))
                    eval_name = as_string(result.get("evaluation_name"))

                    prev = best_per_metric.get(esid)
                    is_better = (
                        prev is None
                        or (lower_is_better and score < prev["score"])
                        or (not lower_is_better and score > prev["score"])
                    )
                    if is_better:
                        best_per_metric[esid] = {
                            "benchmark": benchmark_display_name
                            or as_string(evaluation.get("benchmark")),
                            "benchmarkKey": esid,
                            "canonical_display_name": as_string(
                                normalized.get("canonical_display_name")
                                or normalized.get("display_name")
                            ),
                            "evaluation_name": eval_name,
                            "score": score,
                            "metric": as_string(
                                metric_config.get("evaluation_description") or eval_name
                            ),
                            "unit": as_string(metric_config.get("unit")) or None,
                            "lower_is_better": lower_is_better,
                        }

        # Build top_benchmark_scores: deduplicate per benchmark (keep best metric),
        # sort by absolute score descending, cap at 15 entries
        top_benchmark_scores = sorted(
            best_per_metric.values(),
            key=lambda s: -abs(s["score"]),
        )[:15]
        # Strip None units to keep JSON compact
        for entry in top_benchmark_scores:
            if entry.get("unit") is None:
                del entry["unit"]

        summary_model_info = dict(model_info)
        summary_model_info.update(
            {
                "id": family_id,
                "name": family_name,
                "family_id": family_id,
                "family_name": family_name,
                "model_route_id": route_id,
                "variant_key": "default",
                "variant_label": "Default",
                "model_version": None,
            }
        )

        summary = {
            "model_info": summary_model_info,
            "model_family_id": family_id,
            "model_route_id": route_id,
            "model_family_name": family_name,
            "raw_model_ids": raw_model_ids,
            "evaluations_by_category": dict(by_category),
            "evaluation_summaries_by_category": {},
            "hierarchy_by_category": {},
            "total_evaluations": len(family_evals),
            "last_updated": last_updated,
            "categories_covered": [],
            "variants": [
                {
                    "variant_key": v["variant_key"],
                    "variant_label": v["variant_label"],
                    "evaluation_count": v["evaluation_count"],
                    "raw_model_ids": sorted(v["raw_model_ids"]),
                    "last_updated": v["last_updated"],
                }
                for v in variants_map.values()
            ],
        }
        filtered_eval_summaries = [
            filtered_summary
            for filtered_summary in (
                filter_eval_summary_for_model(eval_summary, family_id)
                for eval_summary in eval_summaries
            )
            if filtered_summary is not None
        ]
        summary_categories = sorted(
            {
                as_string(filtered_summary.get("category") or "other")
                for filtered_summary in filtered_eval_summaries
                if as_string(filtered_summary.get("category") or "other")
            }
        )
        summary["categories_covered"] = summary_categories
        summary["evaluation_summaries_by_category"] = {
            category: [
                filtered_summary
                for filtered_summary in filtered_eval_summaries
                if as_string(filtered_summary.get("category") or "other") == category
            ]
            for category in summary_categories
        }
        summary["hierarchy_by_category"] = summary["evaluation_summaries_by_category"]

        # Aggregate row-level reproducibility_gap + provenance annotations
        # across this model's filtered hierarchy, and per-group signals
        model_repro_annotations: list[dict] = []
        model_provenance_annotations: list[dict] = []
        model_signal_groups: list[dict] = []
        model_signal_groups_canonical: list[dict] = []
        for filtered_summary in filtered_eval_summaries:
            filtered_metrics: list[dict] = list(filtered_summary.get("metrics", []))
            for subtask in filtered_summary.get("subtasks", []):
                filtered_metrics.extend(subtask.get("metrics", []))
            for metric in filtered_metrics:
                for entry in metric.get("_signal_groups") or []:
                    if as_string(entry.get("model_route_id")) == route_id:
                        model_signal_groups.append(entry)
                for entry in metric.get("_signal_groups_canonical") or []:
                    if as_string(entry.get("model_route_id")) == route_id:
                        model_signal_groups_canonical.append(entry)
                for row in metric.get("model_results", []):
                    annotations = (row.get("evalcards") or {}).get("annotations") or {}
                    repro = annotations.get("reproducibility_gap")
                    if repro is not None:
                        model_repro_annotations.append(repro)
                    prov = annotations.get("provenance")
                    if prov is not None:
                        model_provenance_annotations.append(prov)
        model_repro_summary = signals.summarize_reproducibility(model_repro_annotations)
        model_provenance_summary = signals.summarize_provenance(
            model_provenance_annotations, model_signal_groups
        )
        model_comparability_summary = signals.summarize_comparability(
            model_signal_groups
        )
        model_cross_party_summary_canonical = summarize_comparability_canonical(
            model_signal_groups_canonical
        )
        summary["reproducibility_summary"] = model_repro_summary
        summary["provenance_summary"] = model_provenance_summary
        summary["comparability_summary"] = model_comparability_summary
        summary["cross_party_summary_canonical"] = model_cross_party_summary_canonical
        model_summaries.append(summary)

        if score_values:
            score_summary = {
                "count": len(score_values),
                "min": min(score_values),
                "max": max(score_values),
                "average": sum(score_values) / len(score_values),
            }
        else:
            score_summary = {"count": 0, "min": None, "max": None, "average": None}

        model_cards.append(
            {
                "model_family_id": family_id,
                "model_route_id": route_id,
                "model_family_name": family_name,
                "developer": as_string(model_info.get("developer")),
                "params_billions": params_billions,
                "total_evaluations": len(family_evals),
                "benchmark_count": len(
                    {
                        as_string(e.get("benchmark"))
                        for e in family_evals
                        if as_string(e.get("benchmark"))
                    }
                ),
                "benchmark_family_count": len(
                    {
                        as_string(
                            (
                                (result.get("normalized_result") or {}).get(
                                    "benchmark_family_key"
                                )
                            )
                        )
                        for evaluation in family_evals
                        for result in evaluation.get("evaluation_results") or []
                        if as_string(
                            (
                                (result.get("normalized_result") or {}).get(
                                    "benchmark_family_key"
                                )
                            )
                        )
                    }
                ),
                "categories_covered": summary_categories,
                "last_updated": last_updated,
                "variants": summary["variants"],
                "score_summary": score_summary,
                "reproducibility_summary": model_repro_summary,
                "provenance_summary": model_provenance_summary,
                "comparability_summary": model_comparability_summary,
                "cross_party_summary_canonical": model_cross_party_summary_canonical,
                # ---- FIX 2 continued: include benchmark names and per-benchmark
                # scores so the frontend compare dialog and domain pills work ----
                "benchmark_names": sorted(benchmark_names_set),
                "top_benchmark_scores": top_benchmark_scores,
            }
        )

    model_cards.sort(
        key=lambda m: (-m["total_evaluations"], as_string(m["model_route_id"]))
    )
    model_summaries.sort(key=lambda m: as_string(m.get("model_route_id")))

    lite_model_cards = build_lightweight_model_cards(model_cards)

    eval_list = {
        "evals": [
            {
                "eval_summary_id": s["eval_summary_id"],
                "canonical_benchmark_id": s.get("canonical_benchmark_id"),
                "benchmark": s["benchmark"],
                "benchmark_family_key": s.get("benchmark_family_key"),
                "benchmark_family_name": s.get("benchmark_family_name"),
                "benchmark_parent_key": s.get("benchmark_parent_key"),
                "benchmark_parent_name": s.get("benchmark_parent_name"),
                "benchmark_leaf_key": s.get("benchmark_leaf_key"),
                "benchmark_leaf_name": s.get("benchmark_leaf_name"),
                "benchmark_component_key": s.get("benchmark_component_key"),
                "benchmark_component_name": s.get("benchmark_component_name"),
                "evaluation_name": s["evaluation_name"],
                "display_name": s.get("display_name"),
                "canonical_display_name": s.get("canonical_display_name"),
                "is_summary_score": s.get("is_summary_score", False),
                "summary_score_for": s.get("summary_score_for"),
                "summary_score_for_name": s.get("summary_score_for_name"),
                "summary_eval_ids": s.get("summary_eval_ids", []),
                "category": s.get("category", "other"),
                "models_count": s["models_count"],
                "metrics_count": s.get("metrics_count"),
                "subtasks_count": s.get("subtasks_count"),
                "metric_names": s.get("metric_names"),
                "primary_metric_name": s.get("primary_metric_name"),
                "benchmark_card": s["benchmark_card"],
                "tags": s.get("tags", {"domains": [], "languages": [], "tasks": []}),
                "source_data": s["source_data"],
                "metrics": [
                    {
                        "metric_summary_id": metric["metric_summary_id"],
                        "metric_name": metric.get("metric_name"),
                        "metric_id": metric.get("metric_id"),
                        "metric_key": metric.get("metric_key"),
                        "metric_source": metric.get("metric_source"),
                        "canonical_display_name": metric.get("canonical_display_name"),
                        "lower_is_better": metric.get("lower_is_better"),
                        "models_count": metric.get("models_count"),
                        "top_score": metric.get("top_score"),
                    }
                    for metric in s.get("metrics", [])
                ],
                "subtasks": [
                    {
                        "subtask_key": subtask.get("subtask_key"),
                        "subtask_name": subtask.get("subtask_name"),
                        "display_name": subtask.get("display_name"),
                        "canonical_display_name": subtask.get("canonical_display_name"),
                        "metrics_count": subtask.get("metrics_count"),
                        "metric_names": subtask.get("metric_names"),
                        "metrics": [
                            {
                                "metric_summary_id": metric["metric_summary_id"],
                                "metric_name": metric.get("metric_name"),
                                "metric_id": metric.get("metric_id"),
                                "metric_key": metric.get("metric_key"),
                                "metric_source": metric.get("metric_source"),
                                "canonical_display_name": metric.get(
                                    "canonical_display_name"
                                ),
                                "lower_is_better": metric.get("lower_is_better"),
                                "models_count": metric.get("models_count"),
                                "top_score": metric.get("top_score"),
                            }
                            for metric in subtask.get("metrics", [])
                        ],
                    }
                    for subtask in s.get("subtasks", [])
                ],
                "top_score": s.get("top_score"),
                "instance_data": s.get(
                    "instance_data",
                    {
                        "available": False,
                        "url_count": 0,
                        "sample_urls": [],
                        "models_with_loaded_instances": 0,
                    },
                ),
                "evalcards": s.get("evalcards"),
                "reproducibility_summary": s.get("reproducibility_summary"),
                "provenance_summary": s.get("provenance_summary"),
                "comparability_summary": s.get("comparability_summary"),
                "cross_party_summary_canonical": s.get(
                    "cross_party_summary_canonical"
                ),
            }
            for s in eval_summaries
        ],
        "totalModels": len(model_cards),
    }

    lite_eval_list = build_lightweight_eval_list(eval_list)

    # ---- FIX 3: group developers by slug to merge case variants ----
    # e.g. "anthropic" and "Anthropic" both slugify to "anthropic"
    dev_group_by_slug: dict[str, list[dict]] = defaultdict(list)
    dev_name_by_slug: dict[str, str] = {}
    for card in model_cards:
        developer = as_string(card.get("developer") or "Unknown")
        slug = slugify_developer(developer)
        dev_group_by_slug[slug].append(card)
        # Keep the most common name variant (or the capitalized one)
        existing_name = dev_name_by_slug.get(slug)
        if existing_name is None or (
            developer[0:1].isupper() and not existing_name[0:1].isupper()
        ):
            dev_name_by_slug[slug] = developer

    developers = [
        {"developer": dev_name_by_slug[slug], "model_count": len(models)}
        for slug, models in dev_group_by_slug.items()
    ]
    developers.sort(key=lambda d: (-d["model_count"], as_string(d["developer"])))

    dev_summaries = []
    for slug, models in dev_group_by_slug.items():
        developer = dev_name_by_slug[slug]
        sorted_models = sorted(
            models, key=lambda m: as_string(m.get("model_family_name"))
        )
        dev_summaries.append(
            {"developer": developer, "slug": slug, "models": sorted_models}
        )

    # Corpus-level aggregates: walks every eval_summary's per-row
    # annotations + per-eval completeness + per-group signal entries to build
    # one summary artifact for paper / dashboard consumption. Must run before
    # the strip pass since it reads `_signal_groups` off the metric_summaries.
    repro_inputs: list[tuple] = []
    provenance_row_inputs: list[tuple] = []
    group_inputs: list[tuple] = []
    group_inputs_canonical: list[tuple] = []
    completeness_inputs: list[tuple] = []
    base_field_count = len(signals.BASE_REPRODUCIBILITY_FIELDS)

    for eval_summary in eval_summaries:
        category = as_string(eval_summary.get("category")) or None
        completeness = (
            (eval_summary.get("evalcards") or {}).get("annotations") or {}
        ).get("reporting_completeness")
        if isinstance(completeness, dict):
            completeness_inputs.append((completeness, category))

        all_metrics: list[dict] = list(eval_summary.get("metrics", []))
        for subtask in eval_summary.get("subtasks", []):
            all_metrics.extend(subtask.get("metrics", []))
        for metric in all_metrics:
            for group in metric.get("_signal_groups") or []:
                group_inputs.append((group, category))
            for group in metric.get("_signal_groups_canonical") or []:
                group_inputs_canonical.append((group, category))
            for row in metric.get("model_results", []):
                annotations = (row.get("evalcards") or {}).get("annotations") or {}
                repro = annotations.get("reproducibility_gap")
                if isinstance(repro, dict):
                    # Derive is_agentic from required_field_count: any extras
                    # beyond the base set come from the agentic schema.
                    is_agentic = (
                        repro.get("required_field_count") or 0
                    ) > base_field_count
                    repro_inputs.append(
                        (
                            {"annotation": repro, "is_agentic": is_agentic},
                            category,
                        )
                    )
                provenance = annotations.get("provenance")
                if isinstance(provenance, dict):
                    provenance_row_inputs.append((provenance, category))

    corpus_aggregates = {
        "generated_at": started_at,
        "signal_version": signals.SIGNAL_VERSION,
        "stratification_dimensions": ["category"],
        "reproducibility": signals.stratify(
            repro_inputs, signals.aggregate_reproducibility
        ),
        "completeness": signals.stratify(
            completeness_inputs, signals.aggregate_completeness
        ),
        "provenance": signals.stratify_provenance(provenance_row_inputs, group_inputs),
        "comparability": signals.stratify(
            group_inputs, signals.aggregate_comparability
        ),
        # Cross-suite comparability: groups bucketed by registry-resolved
        # canonical_benchmark_id rather than per-suite metric_summary_id.
        # Sidecar to ``comparability``; surfaces multi-suite duplicates
        # (e.g. helm_classic_mmlu vs helm_lite_mmlu) that the per-suite
        # block cannot detect by construction. Variant fields will be
        # null-rate (canonical groups carry only cross_party_divergence).
        "comparability_canonical": signals.stratify(
            group_inputs_canonical, signals.aggregate_comparability
        ),
    }

    manifest = {
        "generated_at": started_at,
        "model_count": len(model_cards),
        "eval_count": len(eval_summaries),
        "metric_eval_count": sum(
            len(summary.get("metrics", []))
            + sum(
                len(subtask.get("metrics", []))
                for subtask in summary.get("subtasks", [])
            )
            for summary in eval_summaries
        ),
        "config_version": CONFIG_VERSION,
        "skipped_config_count": len(skipped_configs),
        "skipped_configs": skipped_configs,
        "source_config_count": len(all_configs),
        "summary_artifacts": {
            "model_cards": "model-cards.json",
            "model_cards_lite": "model-cards-lite.json",
            "eval_list": "eval-list.json",
            "eval_list_lite": "eval-list-lite.json",
            "comparison_index": "comparison-index.json",
            "corpus_aggregates": "corpus-aggregates.json",
            "eval_hierarchy": "eval-hierarchy.json",
        },
    }

    write_json(OUTPUT_DIR / "corpus-aggregates.json", corpus_aggregates)
    write_json(OUTPUT_DIR / "model-cards.json", model_cards)
    write_json(OUTPUT_DIR / "model-cards-lite.json", lite_model_cards)
    write_json(OUTPUT_DIR / "eval-list.json", eval_list)
    write_json(OUTPUT_DIR / "eval-list-lite.json", lite_eval_list)
    write_json(OUTPUT_DIR / "peer-ranks.json", peer_ranks)
    write_json(OUTPUT_DIR / "comparison-index.json", comparison_index)
    write_json(OUTPUT_DIR / "benchmark-metadata.json", benchmark_metadata)
    write_json(OUTPUT_DIR / "developers.json", developers)

    # eval-hierarchy.json: regenerate at runtime from current eval_summaries
    # (replaces a previous shutil.copy2 of `reports/eval_hierarchy.json`,
    # which was a stale snapshot from a separate generator script and only
    # covered ~7/20 of the live family keys).
    #
    # Two-level structure: family → leaf. `family_key` and `leaf_key` come
    # straight off each eval_summary, so the output always reflects the
    # current corpus. Per-family rollups aggregate across all the family's
    # leaves; per-leaf rollups aggregate the eval_summary(ies) that
    # share that `(family, leaf)` pair (typically one).
    #
    # KNOWN LIMITATION (2026-04-26): family keys are NOT collapsed via
    # `build_eval_hierarchy_report.py`'s FAMILY_RULES. Consequence:
    # `helm_air_bench`, `helm_classic`, `helm_lite`, `helm_instruct`,
    # `helm_capabilities`, and `helm_mmlu` each appear as a top-level
    # family instead of as composites under one `helm` family. Same for
    # `apex_v1` + `apex_agents` etc.
    def _collect_eval_signal_data(
        eval_summary: dict,
    ) -> tuple[list, list, list, list]:
        repro_anns: list[dict] = []
        prov_anns: list[dict] = []
        sig_groups: list[dict] = []
        sig_groups_canonical: list[dict] = []
        metrics_pool = list(eval_summary.get("metrics") or [])
        for subtask in eval_summary.get("subtasks") or []:
            metrics_pool.extend(subtask.get("metrics") or [])
        for metric in metrics_pool:
            sig_groups.extend(metric.get("_signal_groups") or [])
            sig_groups_canonical.extend(metric.get("_signal_groups_canonical") or [])
            for row in metric.get("model_results") or []:
                annotations = (row.get("evalcards") or {}).get("annotations") or {}
                repro = annotations.get("reproducibility_gap")
                if isinstance(repro, dict):
                    repro_anns.append(repro)
                prov = annotations.get("provenance")
                if isinstance(prov, dict):
                    prov_anns.append(prov)
        return repro_anns, prov_anns, sig_groups, sig_groups_canonical

    def _node_rollups(evals_for_node: list[dict]) -> dict:
        repro_all: list[dict] = []
        prov_all: list[dict] = []
        groups_all: list[dict] = []
        groups_canonical_all: list[dict] = []
        for es in evals_for_node:
            r, p, g, gc = _collect_eval_signal_data(es)
            repro_all.extend(r)
            prov_all.extend(p)
            groups_all.extend(g)
            groups_canonical_all.extend(gc)
        return {
            "reproducibility_summary": signals.summarize_reproducibility(repro_all),
            "provenance_summary": signals.summarize_provenance(prov_all, groups_all),
            "comparability_summary": signals.summarize_comparability(groups_all),
            "cross_party_summary_canonical": summarize_comparability_canonical(
                groups_canonical_all
            ),
        }

    families_in_progress: dict[str, dict] = {}
    for eval_summary in eval_summaries:
        family_key = as_string(eval_summary.get("benchmark_family_key")) or "unknown"
        family = families_in_progress.setdefault(
            family_key,
            {
                "key": family_key,
                "display_name": (
                    as_string(eval_summary.get("benchmark_family_name")) or family_key
                ),
                "category": as_string(eval_summary.get("category")) or "other",
                "_evals": [],
                "_leaves": {},
            },
        )
        family["_evals"].append(eval_summary)
        leaf_key = as_string(eval_summary.get("benchmark_leaf_key")) or as_string(
            eval_summary.get("eval_summary_id")
        )
        family["_leaves"].setdefault(leaf_key, []).append(eval_summary)

    runtime_hierarchy: dict[str, Any] = {
        "generated_at": started_at,
        "signal_version": signals.SIGNAL_VERSION,
        "schema_note": (
            "Runtime-generated 2-level hierarchy (family → leaf). "
            "Replaces the previously-shipped static reports/eval_hierarchy.json snapshot. "
            "Family keys are uncollapsed runtime values: `helm_classic`, `helm_lite`, "
            "`helm_air_bench` etc. each appear as separate top-level families instead "
            "of as composites under one `helm` family. Family-collapse is deferred to "
            "the planned evalcard-registry integration sweep so all artifacts in this "
            "directory share a single canonical-identity scheme."
        ),
        "families": [],
    }
    for family_key in sorted(families_in_progress):
        family = families_in_progress[family_key]
        family_rollups = _node_rollups(family["_evals"])
        leaves: list[dict] = []
        for leaf_key in sorted(family["_leaves"]):
            leaf_evals = family["_leaves"][leaf_key]
            leaf_rollups = _node_rollups(leaf_evals)
            # Within a single (family, leaf) bucket the eval_summaries
            # almost always share one canonical_benchmark_id, but the
            # corpus has at least one case (helm_classic_mmlu vs
            # helm_lite_mmlu both at leaf "mmlu" under different family
            # keys) where they collapse to canonical "mmlu". We surface
            # the leaf-local set so consumers can detect that.
            leaf_canonical_ids = sorted(
                {
                    as_string(es.get("canonical_benchmark_id"))
                    for es in leaf_evals
                    if as_string(es.get("canonical_benchmark_id"))
                }
            )
            leaves.append(
                {
                    "key": leaf_key,
                    "canonical_benchmark_id": (
                        leaf_canonical_ids[0]
                        if len(leaf_canonical_ids) == 1
                        else None
                    ),
                    "canonical_benchmark_ids": leaf_canonical_ids,
                    "display_name": (
                        as_string(leaf_evals[0].get("benchmark_leaf_name")) or leaf_key
                    ),
                    "category": as_string(leaf_evals[0].get("category")) or "other",
                    "evals_count": len(leaf_evals),
                    "eval_summary_ids": sorted(
                        as_string(es.get("eval_summary_id"))
                        for es in leaf_evals
                        if as_string(es.get("eval_summary_id"))
                    ),
                    **leaf_rollups,
                }
            )
        family_canonical_ids = sorted(
            {
                as_string(es.get("canonical_benchmark_id"))
                for es in family["_evals"]
                if as_string(es.get("canonical_benchmark_id"))
            }
        )
        runtime_hierarchy["families"].append(
            {
                "key": family["key"],
                "display_name": family["display_name"],
                "category": family["category"],
                "canonical_benchmark_ids": family_canonical_ids,
                "evals_count": len(family["_evals"]),
                "eval_summary_ids": sorted(
                    as_string(es.get("eval_summary_id"))
                    for es in family["_evals"]
                    if as_string(es.get("eval_summary_id"))
                ),
                **family_rollups,
                "leaves": leaves,
            }
        )

    # Frontend (`general-eval-card/lib/backend-artifacts.ts:67-79`) declares
    # `EvalHierarchy.stats` as required and `app/evals/page.tsx` reads
    # `hierarchy.stats.family_count` without a guard. The canonical semantics
    # come from `scripts/build_eval_hierarchy_report.py:647-654`. The runtime
    # hierarchy here is a 2-level (family → leaf) view built from
    # `eval_summaries`; we approximate the report-script counters from the
    # current corpus rather than from the report's tree:
    #   - `family_count`         : top-level families emitted
    #   - `composite_count`      : families with >1 leaf (multi-leaf suites)
    #   - `standalone_benchmark_count`: families with exactly one leaf
    #   - `single_benchmark_count`: total leaves under composite (multi-leaf)
    #     families (i.e. the per-benchmark members of those suites)
    #   - `slice_count`          : distinct `slice_key` values across all
    #     metrics in the corpus (root metrics + subtask metrics)
    #   - `metric_count`         : distinct `metric_key` values across the
    #     same metric set
    #   - `metric_rows_scanned`  : total `model_results` rows across all
    #     metrics (each row is one model×metric submission)
    runtime_family_count = len(runtime_hierarchy["families"])
    runtime_composite_count = sum(
        1 for fam in runtime_hierarchy["families"] if len(fam.get("leaves") or []) > 1
    )
    runtime_standalone_benchmark_count = sum(
        1 for fam in runtime_hierarchy["families"] if len(fam.get("leaves") or []) == 1
    )
    runtime_single_benchmark_count = sum(
        len(fam.get("leaves") or [])
        for fam in runtime_hierarchy["families"]
        if len(fam.get("leaves") or []) > 1
    )
    runtime_slice_keys: set[str] = set()
    runtime_metric_keys: set[str] = set()
    runtime_metric_rows_scanned = 0
    for eval_summary in eval_summaries:
        metrics_pool = list(eval_summary.get("metrics") or [])
        for subtask in eval_summary.get("subtasks") or []:
            metrics_pool.extend(subtask.get("metrics") or [])
        for metric in metrics_pool:
            slice_key_value = as_string(metric.get("slice_key"))
            if slice_key_value:
                runtime_slice_keys.add(slice_key_value)
            metric_key_value = as_string(metric.get("metric_key"))
            if metric_key_value:
                runtime_metric_keys.add(metric_key_value)
            runtime_metric_rows_scanned += len(metric.get("model_results") or [])

    runtime_hierarchy["stats"] = {
        "family_count": runtime_family_count,
        "composite_count": runtime_composite_count,
        "standalone_benchmark_count": runtime_standalone_benchmark_count,
        "single_benchmark_count": runtime_single_benchmark_count,
        "slice_count": len(runtime_slice_keys),
        "metric_count": len(runtime_metric_keys),
        "metric_rows_scanned": runtime_metric_rows_scanned,
    }

    write_json(OUTPUT_DIR / "eval-hierarchy.json", runtime_hierarchy)

    hierarchy_path = OUTPUT_DIR / "eval-hierarchy.json"
    readme_text = generate_readme(
        manifest, eval_list, benchmark_metadata, hierarchy_path
    )
    (OUTPUT_DIR / "README.md").write_text(readme_text, encoding="utf-8")

    # Strip Signals 3+4 internals carried for the rollup passes. After this
    # point the per-summary JSONs serialize cleanly: rows lose
    # `_generation_args` (input plumbing for variant_divergence comparison)
    # and metric_summaries lose `_signal_groups` (group-level summaries used
    # by the per-eval / per-model rollups).
    def _strip_signals_internals(summary_obj: dict) -> None:
        summary_obj.pop("_eee_source_config", None)
        all_metrics: list[dict] = list(summary_obj.get("metrics", []))
        for subtask in summary_obj.get("subtasks", []):
            all_metrics.extend(subtask.get("metrics", []))
        for metric in all_metrics:
            metric.pop("_signal_groups", None)
            metric.pop("_signal_groups_canonical", None)
            for row in metric.get("model_results", []):
                row.pop("_generation_args", None)

    for summary in eval_summaries:
        _strip_signals_internals(summary)
    for summary in model_summaries:
        for category_summaries in summary.get("hierarchy_by_category", {}).values():
            for filtered_summary in category_summaries:
                _strip_signals_internals(filtered_summary)

    for summary in model_summaries:
        write_json(OUTPUT_DIR / "models" / f"{summary['model_route_id']}.json", summary)
    for summary in eval_summaries:
        write_json(OUTPUT_DIR / "evals" / f"{summary['eval_summary_id']}.json", summary)
    for summary in dev_summaries:
        write_json(
            OUTPUT_DIR / "developers" / f"{summary['slug']}.json",
            {"developer": summary["developer"], "models": summary["models"]},
        )

    validate_output_contract(OUTPUT_DIR)

    # Parity-layer parquet artifacts under output/duckdb/v1/. Adds a
    # frontend-ready read surface alongside the existing JSON files;
    # run on every pipeline invocation so consumers can rely on the
    # path being present. Cleaning transforms (license, dev names,
    # params, variants, benchmark display names) are applied here per
    # PLAN_20260428.md. Emitted AFTER `validate_output_contract` because
    # the validator scans every output file as UTF-8 text.
    from scripts import parity_outputs

    parity_outputs.write_parity_artifacts(
        model_cards=model_cards,
        lite_model_cards=lite_model_cards,
        eval_list=eval_list,
        lite_eval_list=lite_eval_list,
        eval_summaries=eval_summaries,
        model_summaries=model_summaries,
        dev_summaries=dev_summaries,
        benchmark_metadata=benchmark_metadata,
        output_dir=OUTPUT_DIR,
    )

    manifest["artifact_sizes"] = collect_artifact_sizes(OUTPUT_DIR)
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    print(
        f"[pipeline] {json.dumps({'event': 'pipeline.summary', 'dry_run': dry_run, 'evaluations_loaded': len(evaluations), 'model_count': len(model_cards), 'eval_count': len(eval_summaries), 'skipped_config_count': len(skipped_configs)})}"
    )

    print(
        json.dumps(
            {
                "dry_run": dry_run,
                "model_count": len(model_cards),
                "eval_count": len(eval_summaries),
                "skipped_configs": skipped_configs,
                "output_dir": str(OUTPUT_DIR.resolve()),
            },
            indent=2,
        )
    )

    if not dry_run:
        upload_output()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

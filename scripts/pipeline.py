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


DATASET_REPO = "evaleval/card_backend"
EEE_DATASET_REPO = "evaleval/EEE_datastore"
BENCHMARK_METADATA_DATASET_REPO = "evaleval/auto-benchmarkcards"
EEE_DATASET_RAW_BASE = f"https://huggingface.co/datasets/{EEE_DATASET_REPO}/raw/main"
DATASET_RESOLVE_BASE = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"
CONFIG_VERSION = 1
OUTPUT_DIR = Path("output")
DEFAULT_LOCAL_DATASET_DIR = ".cache/eee_datastore"
DEFAULT_LOCAL_BENCHMARK_METADATA_DIR = ".cache/auto_benchmarkcards"
DEFAULT_METRIC_REGISTRY_PATH = Path("registry/metric_looking_strings.json")
FILE_READ_MAX_RETRIES = 5
FILE_READ_RETRY_DELAY_SEC = 1.5
VERSION_SUFFIX_REGEX = re.compile(r"^(.*?)-((?:19|20)\d{6})(?:-(.+))?$")
BENCHMARK_FAMILY_REGEXES = [
    re.compile(r"^(.*?)(\d+)(_arena)$"),
    re.compile(r"^(.*?)[_-]v(\d+)$"),
]
PASS_AT_REGEX = re.compile(r"pass\s*@?\s*(\d+)", flags=re.IGNORECASE)
PASS_AT_EXACT_REGEX = re.compile(r"^\s*pass\s*@?\s*(\d+)\s*$", flags=re.IGNORECASE)
EVAL_DESCRIPTION_METRIC_REGEX = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 @%+./_-]*?)\s+on\s+(.+?)\s*$")
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
}
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
    return re.sub(r"[^a-z0-9]+", "_", as_string(value).lower()).strip("_")


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


def humanize_slug(value: Any) -> str:
    parts = [p for p in re.split(r"[-_/]+", as_string(value)) if p]
    out = []
    for part in parts:
        if len(part) <= 3 and any(c.isdigit() for c in part):
            out.append(part.upper())
        else:
            out.append(part[:1].upper() + part[1:])
    return " ".join(out)


def parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def iso_from_epoch_string(value: Any) -> str | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def max_iso(left: str | None, right: str | None) -> str | None:
    if not left:
        return right
    if not right:
        return left
    return left if left > right else right


def load_benchmark_metadata(metadata_cache_dir: str) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    return load_benchmark_metadata_from_dir(Path(metadata_cache_dir))


def canonical_benchmark_family_key(value: Any) -> str:
    key = normalize_benchmark_key(value)
    if not key:
        return ""
    for regex in BENCHMARK_FAMILY_REGEXES:
        match = regex.match(key)
        if not match:
            continue
        candidate = normalize_benchmark_key("".join(part for index, part in enumerate(match.groups(), start=1) if index != 2))
        if candidate:
            return candidate
    return key


def compact_benchmark_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", as_string(value).lower())


def load_benchmark_metadata_from_dir(root_dir: Path) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
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
                if not isinstance(raw_card, dict) or not isinstance(raw_card.get("benchmark_details"), dict):
                    continue
                card_id = normalize_benchmark_key(raw_key)
                if card_id:
                    flat_map[card_id] = raw_card

    cards_dir = root_dir / "cards"
    if cards_dir.exists():
        for file_path in sorted(cards_dir.glob("*.json")):
            parsed = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict) and isinstance(parsed.get("benchmark_card"), dict):
                card = parsed["benchmark_card"]
                base_name = file_path.stem.replace("benchmark_card_", "")
            elif isinstance(parsed, dict) and isinstance(parsed.get("benchmark_details"), dict):
                card = parsed
                base_name = file_path.stem
            else:
                continue
            card_id = normalize_benchmark_key(base_name)
            if card_id and card_id not in flat_map:
                flat_map[card_id] = card

    for card_id, card in sorted(flat_map.items()):
        keys = candidate_benchmark_keys(card_id, card.get("benchmark_details", {}).get("name"))
        cards.append({"file_name": f"{card_id}.json", "base_name": card_id, "card": card, "keys": keys})
        for key in keys:
            lookup[key] = card

    return cards, lookup, flat_map


def has_cached_benchmark_metadata(cards_dir: Path, flat_metadata_path: Path) -> bool:
    return flat_metadata_path.exists() or (cards_dir.exists() and any(cards_dir.glob("*.json")))


def ensure_local_benchmark_metadata_snapshot(local_metadata_dir: str, hf_token: str | None, force_refresh: bool) -> str | None:
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
        stripped = normalize_benchmark_key(re.sub(r"^benchmark_card_", "", text, flags=re.IGNORECASE))
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


def lookup_benchmark_card(metadata_lookup: dict[str, dict], *values: Any) -> dict | None:
    for key in candidate_benchmark_keys(*values):
        if key in metadata_lookup:
            return metadata_lookup[key]
    return None


def iter_matching_benchmark_cards(metadata_lookup: dict[str, dict], *values: Any) -> list[dict]:
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


def lookup_benchmark_card_for_parent(metadata_lookup: dict[str, dict], *values: Any, parent_values: tuple[Any, ...] = ()) -> dict | None:
    candidates = iter_matching_benchmark_cards(metadata_lookup, *values)
    if not candidates:
        return None

    parent_keys = set(candidate_benchmark_keys(*parent_values))
    if not parent_keys:
        return candidates[0]

    compatible: list[dict] = []
    unconstrained: list[dict] = []
    for card in candidates:
        details = card.get("benchmark_details") if isinstance(card, dict) else {}
        appears_in = as_string_list((details or {}).get("appears_in"))
        if not appears_in:
            unconstrained.append(card)
            continue
        appears_in_keys = set(candidate_benchmark_keys(*appears_in))
        if parent_keys & appears_in_keys:
            compatible.append(card)

    if compatible:
        return compatible[0]
    if unconstrained:
        return unconstrained[0]
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
    candidates = [as_string(value).strip() for value in values if as_string(value).strip()]

    for candidate in candidates:
        preferred = PREFERRED_BENCHMARK_DISPLAY_NAMES.get(normalize_benchmark_key(candidate))
        if preferred:
            return preferred

    for candidate in candidates:
        if any(char.isupper() for char in candidate) or " " in candidate:
            return candidate

    fallback_text = as_string(fallback).strip()
    if fallback_text:
        preferred = PREFERRED_BENCHMARK_DISPLAY_NAMES.get(normalize_benchmark_key(fallback_text))
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
    METRIC_SUFFIX_ALIAS_CANDIDATES = sorted(candidate_set, key=lambda value: (-len(value.split("_")), -len(value), value))


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
        display = as_string(METRIC_REGISTRY_ENTRIES[metric_key].get("display_name")).strip()
        if display:
            return display
    if metric_key in BUILTIN_METRIC_DISPLAY_MAP:
        return BUILTIN_METRIC_DISPLAY_MAP[metric_key]
    if raw_label and canonicalize_metric_key(raw_label) == metric_key and normalize_benchmark_key(raw_label) == metric_key:
        return as_string(raw_label).strip()
    return humanize_metric_key(metric_key)


def infer_metric_from_value(metric_name: Any = None, metric_id: Any = None) -> dict | None:
    explicit_id = as_string(metric_id).strip()
    explicit_name = as_string(metric_name).strip()

    if explicit_id:
        metric_key = canonicalize_metric_key(explicit_id) or slugify(explicit_id)
        display = preferred_metric_display(metric_key, explicit_name or explicit_id.split(".")[-1])
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
    details = ((result.get("score_details") or {}).get("details") or {}) if isinstance(result, dict) else {}
    if not isinstance(details, dict):
        return None
    tab = as_string(details.get("tab")).strip()
    if not tab:
        return None
    return infer_metric_from_value(metric_name=tab)


def infer_metric_from_benchmark_card(card: dict | None) -> dict | None:
    metrics = (((card or {}).get("methodology") or {}).get("metrics") or []) if isinstance(card, dict) else []
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


def metric_namespace_component(metric_id: str, benchmark_family_key: str) -> tuple[str | None, str | None]:
    parts = [part for part in re.split(r"[./]+", as_string(metric_id)) if part]
    if len(parts) < 3:
        return None, None
    if normalize_benchmark_key(parts[0]) != normalize_benchmark_key(benchmark_family_key):
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


def split_metric_from_evaluation_name(raw_name: Any, benchmark_keys: list[str]) -> dict | None:
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
    if benchmark_family_name and normalize_benchmark_key(benchmark_family_name) == benchmark_key:
        return benchmark_family_name
    return canonical_benchmark_display_name(benchmark, fallback=benchmark or benchmark_family_name)


def infer_subset_slice_from_name(name: Any, benchmark: Any) -> tuple[str | None, str | None]:
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

    details = benchmark_card.get("benchmark_details") if isinstance(benchmark_card, dict) else {}
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
    return normalized in COMMON_LANGUAGE_SUBSET_KEYS or compact in COMMON_LANGUAGE_SUBSET_KEYS


def top_level_benchmark_owns_slices(benchmark: Any, benchmark_card: dict | None) -> bool:
    benchmark_key = normalize_benchmark_key(benchmark)
    if benchmark_card:
        return True
    if benchmark_key in {normalize_benchmark_key(key) for key in BENCHMARK_DEFAULT_METRICS}:
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
    source_data = result.get("source_data") if isinstance(result.get("source_data"), dict) else {}
    dataset_name = as_string((source_data or {}).get("dataset_name"))
    raw_name = as_string(result.get("evaluation_name")).strip()
    raw_name_key = normalize_benchmark_key(raw_name)
    dataset_key = normalize_benchmark_key(dataset_name)
    top_level_key = normalize_benchmark_key(benchmark or dataset_name)
    top_level_name = infer_top_level_benchmark_name(benchmark or dataset_name, benchmark_family_name)

    subset_key, subset_name = infer_subset_slice_from_name(dataset_name or raw_name, benchmark or dataset_name)
    if subset_key and subset_name:
        return top_level_key, top_level_name, subset_key, subset_name

    if raw_name and raw_name_key and dataset_key and raw_name_key == dataset_key:
        canonical_raw_name = canonical_benchmark_display_name(
            raw_name,
            benchmark or dataset_name,
            benchmark_family_name,
            fallback=top_level_name,
        )
        if is_language_subset_name(raw_name, benchmark_card) and raw_name_key != top_level_key:
            return top_level_key, top_level_name, raw_name_key, raw_name
        return raw_name_key, canonical_raw_name, None, None

    if component_name:
        component_summary_key = normalize_benchmark_key(component_name)
        if component_summary_key in SUMMARY_SCORE_LEAF_KEYS:
            return top_level_key, top_level_name, None, None
        if top_level_benchmark_owns_slices(benchmark or dataset_name, benchmark_card) or is_language_subset_name(component_name, benchmark_card):
            if component_key == top_level_key or normalize_benchmark_key(component_name) == top_level_key:
                return top_level_key, top_level_name, None, None
            return top_level_key, top_level_name, component_key, component_name
        return component_key or normalize_benchmark_key(component_name), component_name, None, None

    return top_level_key, top_level_name, None, None


def classify_evaluation_result(evaluation: dict, result: dict, benchmark_card: dict | None) -> dict:
    benchmark = as_string(evaluation.get("benchmark"))
    source_data = result.get("source_data") if isinstance(result.get("source_data"), dict) else {}
    dataset_name = as_string((source_data or {}).get("dataset_name"))
    benchmark_family_key = canonical_benchmark_family_key(benchmark or dataset_name)
    benchmark_card_name = as_string(((benchmark_card or {}).get("benchmark_details") or {}).get("name"))
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
    benchmark_keys = [candidate for candidate in {normalize_benchmark_key(benchmark), normalize_benchmark_key(dataset_name), benchmark_family_key} if candidate]

    metric_config = result.get("metric_config") if isinstance(result.get("metric_config"), dict) else {}
    metric = None
    metric_source = "unknown"
    component_name = None
    component_key = None
    raw_name_consumed_as_metric = False

    explicit_metric = infer_metric_from_value(metric_name=metric_config.get("metric_name"), metric_id=metric_config.get("metric_id"))
    if explicit_metric:
        metric = explicit_metric
        metric_source = "metric_config"
        component_name, component_key = metric_namespace_component(metric["metric_id"], benchmark_family_key)
        split_metric = split_metric_from_evaluation_name(raw_name, benchmark_keys)
        if split_metric and split_metric["metric"]["metric_key"] == metric["metric_key"]:
            if not component_name and not component_key:
                component_name = split_metric["component_name"]
                component_key = split_metric["component_key"]
            raw_name_consumed_as_metric = (
                split_metric["component_name"] is None and split_metric["component_key"] is None
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
        metric = split_metric_from_evaluation_description(metric_config.get("evaluation_description"))
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
    if raw_name and not component_name and not raw_name_consumed_as_metric and raw_name_key and raw_name_key not in benchmark_keys and raw_name_key != metric["metric_key"]:
        component_name = raw_name
        component_key = raw_name_key

    if component_name and not component_key:
        component_key = normalize_benchmark_key(component_name)

    benchmark_leaf_key, benchmark_leaf_name, slice_key, slice_name = infer_benchmark_leaf_and_slice(
        evaluation,
        result,
        benchmark_family_key or normalize_benchmark_key(benchmark or dataset_name),
        benchmark_family_name,
        component_key,
        component_name,
        benchmark_card,
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
        display_name = benchmark_leaf_name or benchmark_parent_name or benchmark_family_name
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


def ensure_local_dataset_snapshot(local_dataset_dir: str, hf_token: str | None, force_refresh: bool) -> str:
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


def list_json_files_for_config(config: str, local_dataset_dir: str | None, hf_token: str | None) -> list[str]:
    if local_dataset_dir:
        root = Path(local_dataset_dir) / "data" / config
        return sorted(str(p.relative_to(local_dataset_dir)).replace(os.sep, "/") for p in root.rglob("*.json") if p.is_file() and not p.name.endswith(".jsonl"))

    fs = HfFileSystem(token=hf_token)
    pattern = f"datasets/{EEE_DATASET_REPO}/data/{config}/**/*.json"
    paths = [p for p in fs.glob(pattern) if not p.endswith(".jsonl")]
    prefix = f"datasets/{EEE_DATASET_REPO}/"
    return sorted(p[len(prefix):] for p in paths)


def read_dataset_json(dataset_path: str, local_dataset_dir: str | None, hf_token: str | None) -> dict:
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

        file_path_match = re.search(r"file_path'?:\s*'([^']+)'", value, flags=re.IGNORECASE) or re.search(r'"file_path"\s*:\s*"([^"]+)"', value)
        format_match = re.search(r"format'?:\s*'([^']+)'", value, flags=re.IGNORECASE) or re.search(r'"format"\s*:\s*"([^"]+)"', value)
        rows_match = re.search(r"total_rows'?:\s*(\d+)", value, flags=re.IGNORECASE) or re.search(r'"total_rows"\s*:\s*(\d+)', value)
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
        file_path = as_string(value.get("file_path") or value.get("path") or value.get("url"))
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


def maybe_load_instance_data(record: dict, local_dataset_dir: str | None, hf_token: str | None) -> dict | None:
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
                text = (Path(local_dataset_dir) / dataset_path).read_text(encoding="utf-8")
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


def read_text_from_dataset_url(url: str, local_dataset_dir: str | None, hf_token: str | None) -> str | None:
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
    route_id = as_string((evaluation.get("model_info") or {}).get("model_route_id") or "unknown")
    evaluation_key = slugify(evaluation.get("evaluation_id") or evaluation.get("source_record_url") or "instance")
    return f"instances/{route_id}/{evaluation_key}.jsonl"


def build_record_artifact_relative_path(evaluation: dict) -> str:
    route_id = as_string((evaluation.get("model_info") or {}).get("model_route_id") or "unknown")
    evaluation_key = slugify(evaluation.get("evaluation_id") or "record")
    return f"records/{route_id}/{evaluation_key}.json"


def build_evaluation_hierarchy_payload(evaluation: dict) -> dict:
    category = infer_category_from_benchmark(as_string(evaluation.get("benchmark")), evaluation.get("benchmark_card"))
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
            if as_string(result.get("evaluation_result_id")).strip() == evaluation_result_id:
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
        annotated["hierarchy"] = build_result_hierarchy_payload(evaluation, matched_result)
    else:
        annotated["hierarchy"] = build_evaluation_hierarchy_payload(evaluation)
        if len(annotated["hierarchy"].get("eval_summary_ids", [])) == 1:
            annotated["hierarchy"]["eval_summary_id"] = annotated["hierarchy"]["eval_summary_ids"][0]
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
        transformed_lines.append(json.dumps(annotate_instance_row(evaluation, row), ensure_ascii=False))
    return "\n".join(transformed_lines) + ("\n" if transformed_lines else "")


def normalize_model_info(model_info: dict) -> dict:
    raw_id = as_string(model_info.get("id") or model_info.get("name") or "unknown/unknown")
    fallback_developer = as_string(model_info.get("developer") or raw_id.split("/")[0] or "unknown")
    if "/" in raw_id:
        parts = raw_id.split("/")
    else:
        parts = [slugify_developer(fallback_developer), raw_id]
    raw_developer = parts[0] if len(parts) > 1 else slugify_developer(fallback_developer)
    raw_model_name = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
    match = VERSION_SUFFIX_REGEX.match(raw_model_name)
    base_slug = match.group(1) if match else raw_model_name
    version_date = match.group(2) if match else None
    qualifier = match.group(3) if match else None

    return {
        "raw_id": raw_id,
        "developer": as_string(model_info.get("developer") or humanize_slug(raw_developer)),
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
    variant_parts = [p for p in [normalized["version_date"], normalized["qualifier"]] if p]
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


def infer_category_from_benchmark(benchmark_name: str, benchmark_card: dict | None = None) -> str:
    """Derive a high-level category, preferring benchmark card domains over regex."""
    # Try card domains first
    if benchmark_card:
        domains = ((benchmark_card.get("benchmark_details") or {}).get("domains") or [])
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
    if re.search(r"(appworld|swe_bench|tau_bench|browsecomp|agent|livecodebench|terminal_bench)", key):
        return "agentic"
    if re.search(r"(reward_bench)", key):
        return "safety"
    if re.search(r"(math|gsm|gpqa|mmlu|medqa|legalbench|boolq|hellaswag|quac|cnn_dailymail|musr)", key):
        return "reasoning"
    if re.search(r"(ifeval)", key):
        return "instruction_following"
    if re.search(r"(hfopenllm|helm)", key):
        return "general"
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
RUN_KIND_HARNESS = "harness"   # different agent / scaffold (terminal_bench, swe_bench)
RUN_KIND_VARIANT = "variant"   # raw_model_id varies (reasoning budget, snapshot)
RUN_KIND_RERUN = "rerun"       # same setup, re-evaluated later
RUN_KIND_DEFAULT = "default"   # the only submission for this peer


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
            tail = raw[len(family):].lstrip("-_/").strip()
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
            iso = datetime.fromtimestamp(float(retrieved), tz=timezone.utc).date().isoformat()
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
    by_model: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))

    for summary in eval_summaries:
        eval_summary_id = as_string(summary.get("eval_summary_id"))
        if not eval_summary_id:
            continue

        # Use root metrics if the eval has any; otherwise fall back to the
        # first subtask's metrics. Mirrors the same fallback that
        # ``primary_metric_name`` uses in eval-list.json.
        root_metrics = summary.get("metrics") or []
        subtasks = summary.get("subtasks") or []
        candidate_metrics = root_metrics or (subtasks[0].get("metrics") if subtasks else None) or []
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
                                "raw_model_id": as_string(sub.get("raw_model_id")) or None,
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
            for idx, (route_id, headline, submissions) in enumerate(route_entries, start=1):
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
        path.stem
        for path in evals_dir.glob("*.json")
        if path.is_file()
    }
    if eval_summary_ids and published_eval_files != eval_summary_ids:
        missing_files = sorted(eval_summary_ids - published_eval_files)
        extra_files = sorted(published_eval_files - eval_summary_ids)
        if missing_files:
            errors.append(f"Missing eval files for eval-list entries: {missing_files[:10]}")
        if extra_files:
            errors.append(f"Extra eval files not present in eval-list: {extra_files[:10]}")

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
            errors.append(f"{eval_path.name} missing top-level hierarchy keys: {missing_keys}")

        for metric in parsed.get("metrics", []):
            for row in metric.get("model_results", []):
                record_url = as_string(row.get("source_record_url"))
                if record_url and not record_url.startswith(f"{DATASET_RESOLVE_BASE}/records/"):
                    errors.append(f"{eval_path.name} has non-pipeline source_record_url: {record_url}")
                detailed_url = as_string(row.get("detailed_evaluation_results"))
                if detailed_url and not detailed_url.startswith(f"{DATASET_RESOLVE_BASE}/instances/"):
                    errors.append(f"{eval_path.name} has non-pipeline detailed_evaluation_results URL: {detailed_url}")
                instance_data = row.get("instance_level_data") or {}
                source_url = as_string(instance_data.get("source_url"))
                if source_url and not source_url.startswith(f"{DATASET_RESOLVE_BASE}/instances/"):
                    errors.append(f"{eval_path.name} has non-pipeline instance_level_data.source_url: {source_url}")

        for subtask in parsed.get("subtasks", []):
            for metric in subtask.get("metrics", []):
                for row in metric.get("model_results", []):
                    record_url = as_string(row.get("source_record_url"))
                    if record_url and not record_url.startswith(f"{DATASET_RESOLVE_BASE}/records/"):
                        errors.append(f"{eval_path.name} has non-pipeline source_record_url: {record_url}")
                    detailed_url = as_string(row.get("detailed_evaluation_results"))
                    if detailed_url and not detailed_url.startswith(f"{DATASET_RESOLVE_BASE}/instances/"):
                        errors.append(f"{eval_path.name} has non-pipeline detailed_evaluation_results URL: {detailed_url}")
                    instance_data = row.get("instance_level_data") or {}
                    source_url = as_string(instance_data.get("source_url"))
                    if source_url and not source_url.startswith(f"{DATASET_RESOLVE_BASE}/instances/"):
                        errors.append(f"{eval_path.name} has non-pipeline instance_level_data.source_url: {source_url}")

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
        missing_eval_summary_ids = sorted(expected_eval_summary_ids - actual_eval_summary_ids)
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
        raise RuntimeError("Output contract validation failed:\n- " + "\n- ".join(errors[:50]))


def delete_stale_remote_files(api: HfApi, token: str, output_dir: Path = OUTPUT_DIR) -> None:
    local_files = set(iter_output_relative_files(output_dir))
    remote_files = set(api.list_repo_files(DATASET_REPO, repo_type="dataset", token=token))
    stale_files = sorted(remote_files - local_files)
    if not stale_files:
        return

    chunk_size = 200
    for index in range(0, len(stale_files), chunk_size):
        chunk = stale_files[index : index + chunk_size]
        api.delete_files(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=token,
            delete_patterns=chunk,
            commit_message=f"Remove stale pipeline artifacts ({index + 1}-{index + len(chunk)})",
        )


def filter_metric_summary_for_model(metric_summary: dict, family_id: str) -> dict | None:
    model_results = [
        row
        for row in metric_summary.get("model_results", [])
        if as_string(row.get("model_id")) == family_id
    ]
    if not model_results:
        return None

    filtered = {key: value for key, value in metric_summary.items() if key != "model_results"}
    filtered["model_results"] = model_results
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
                    "metric_names": [as_string(metric.get("metric_name")) for metric in subtask_metrics if as_string(metric.get("metric_name"))],
                }
            )

    if not root_metrics and not subtasks:
        return None

    filtered = {
        key: value
        for key, value in summary.items()
        if key not in {"metrics", "subtasks", "instance_data", "models_count", "top_score"}
    }
    filtered["metrics"] = root_metrics
    filtered["subtasks"] = subtasks
    filtered["subtasks_count"] = len(subtasks)
    filtered["metrics_count"] = len(root_metrics) + sum(len(subtask.get("metrics", [])) for subtask in subtasks)
    filtered["metric_names"] = sorted(
        {
            as_string(metric.get("metric_name"))
            for metric in root_metrics
        }
        | {
            as_string(metric.get("metric_name"))
            for subtask in subtasks
            for metric in subtask.get("metrics", [])
        }
        - {""}
    )
    primary_metrics = root_metrics or (subtasks[0].get("metrics", []) if subtasks else [])
    filtered["primary_metric_name"] = as_string(primary_metrics[0].get("metric_name")) if primary_metrics else None
    filtered["models_count"] = 1
    filtered["top_score"] = primary_metrics[0].get("top_score") if len(primary_metrics) == 1 and not subtasks else None

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


def generate_readme(manifest: dict, eval_list: dict, benchmark_metadata: dict, hierarchy_path: Path | None = None) -> str:
    """Generate a README.md for the HF dataset with full manifest and data access docs."""
    generated_at = manifest.get("generated_at", "unknown")
    model_count = manifest.get("model_count", 0)
    eval_count = manifest.get("eval_count", 0)
    metric_eval_count = manifest.get("metric_eval_count", 0)
    source_config_count = manifest.get("source_config_count", 0)

    evals = eval_list.get("evals", [])
    total_models = eval_list.get("totalModels", model_count)

    # Build the hierarchy tree from eval-list data
    hierarchy_lines = []
    if hierarchy_path and hierarchy_path.exists():
        hierarchy_data = json.loads(hierarchy_path.read_text(encoding="utf-8"))
        families = hierarchy_data.get("families", [])

        def _card_mark(node: dict) -> str:
            return "[x]" if node.get("has_card") else "[ ]"

        def _render_metrics(metrics: list, indent: str) -> None:
            for m in metrics:
                hierarchy_lines.append(f"{indent}- {m['display_name']}")

        def _render_slices(slices: list, indent: str) -> None:
            for s in slices:
                hierarchy_lines.append(f"{indent}- {s['display_name']}")
                _render_metrics(s.get("metrics", []), indent + "  ")

        def _render_benchmark(bm: dict, indent: str) -> None:
            hierarchy_lines.append(f"{indent}- {_card_mark(bm)} {bm['display_name']}")
            _render_slices(bm.get("slices", []), indent + "  ")
            _render_metrics(bm.get("metrics", []), indent + "  ")

        for fam in families:
            hierarchy_lines.append(f"- {_card_mark(fam)} **{fam['display_name']}**")
            # Flattened family: slices/metrics directly
            _render_slices(fam.get("slices", []), "  ")
            _render_metrics(fam.get("metrics", []), "  ")
            # Flattened family: benchmarks directly
            for bm in fam.get("benchmarks", []):
                _render_benchmark(bm, "  ")
            # Standalone benchmarks
            for bm in fam.get("standalone_benchmarks", []):
                _render_benchmark(bm, "  ")
            # Composites
            for comp in fam.get("composites", []):
                hierarchy_lines.append(f"  - {_card_mark(comp)} **{comp['display_name']}**")
                _render_slices(comp.get("slices", []), "    ")
                _render_metrics(comp.get("metrics", []), "    ")
                for bm in comp.get("benchmarks", []):
                    _render_benchmark(bm, "    ")

    hierarchy_tree = "\n".join(hierarchy_lines) if hierarchy_lines else "_Hierarchy not available — run `build_eval_hierarchy_report.py` first._"

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
        eval_table_rows.append(f"| `{eid}` | {name} | {mcount} | {metrics} | {has_card} |")

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
  "stats": {{ "family_count": 20, "composite_count": 20, ... }},
  "families": [
    {{
      "key": "helm",                                   // Normalized key
      "display_name": "HELM",                          // Human-readable name
      "has_card": true,                                // Any child has metadata
      "category": "general",                           // High-level category
      "tags": {{                                        // Merged from all children
        "domains": ["biology", "physics", ...],
        "languages": ["English"],
        "tasks": ["Multiple-choice QA", ...]
      }},
      "standalone_benchmarks": [],
      "composites": [
        {{
          "key": "helm_capabilities",
          "display_name": "Helm capabilities",
          "has_card": true,
          "category": "general",
          "tags": {{ "domains": [...], "languages": [...], "tasks": [...] }},
          "benchmarks": [                              // Multi-benchmark composite
            {{
              "key": "gpqa",
              "display_name": "GPQA",
              "has_card": true,
              "tags": {{ "domains": ["biology", "physics", "chemistry"], ... }},
              "slices": [],
              "metrics": [{{ "key": "cot_correct", "display_name": "COT correct" }}]
            }}
          ],
          "summary_eval_ids": ["helm_capabilities_overall"] // eval_summary_ids of summary/rollup score evals for this composite (empty if none)
        }}
      ]
    }},
    {{
      "key": "global_mmlu_lite",                       // Flattened single-benchmark family
      "display_name": "Global MMLU Lite",
      "has_card": false,
      "category": "reasoning",
      "tags": {{ "domains": [], "languages": [], "tasks": [] }},
      "slices": [                                      // Slices directly on family
        {{ "key": "arabic", "display_name": "Arabic", "metrics": [...] }}
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

{chr(10).join(f'- `{k}`' for k in card_keys)}

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
python scripts/pipeline.py --dry-run

# Full run with upload
HF_TOKEN=hf_xxx python scripts/pipeline.py
```

Config version: `{manifest.get("config_version", 1)}`
"""
    return readme


def upload_output() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required unless --dry-run is used")

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
    except Exception as error:
        print(f"create_repo warning: {error}", file=sys.stderr)

    delete_stale_remote_files(api, token, OUTPUT_DIR)

    api.upload_large_folder(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=str(OUTPUT_DIR),
    )


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    load_instance_in_dry_run = os.environ.get("LOAD_INSTANCE_IN_DRY_RUN") == "1"
    config_batch_size = parse_positive_int(os.environ.get("CONFIG_BATCH_SIZE"), 4)
    config_limit = os.environ.get("CONFIG_LIMIT")
    explicit_configs = [c.strip() for c in as_string(os.environ.get("CONFIGS") or os.environ.get("CONFIG_NAMES")).split(",") if c.strip()]
    configured_local_dataset_dir = as_string(os.environ.get("EEE_LOCAL_DATASET_DIR")).strip() or DEFAULT_LOCAL_DATASET_DIR
    configured_local_metadata_dir = as_string(os.environ.get("BENCHMARK_METADATA_LOCAL_DIR")).strip() or DEFAULT_LOCAL_BENCHMARK_METADATA_DIR
    force_refresh_snapshot = os.environ.get("EEE_REFRESH_SNAPSHOT") == "1"
    force_refresh_metadata = os.environ.get("BENCHMARK_METADATA_REFRESH") == "1"
    allow_skipped_configs = os.environ.get("ALLOW_SKIPPED_CONFIGS") == "1"
    hf_token = os.environ.get("HF_TOKEN")

    local_dataset_dir = ensure_local_dataset_snapshot(configured_local_dataset_dir, hf_token, force_refresh_snapshot)
    local_metadata_dir = ensure_local_benchmark_metadata_snapshot(configured_local_metadata_dir, hf_token, force_refresh_metadata)
    if not local_metadata_dir:
        raise RuntimeError("Failed to cache benchmark metadata from evaleval/auto-benchmarkcards")

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    random.seed(42)
    load_metric_registry()

    clean_output_dir()
    print(
        f"[pipeline] {json.dumps({'event': 'metric_registry.loaded', 'registry_path': str(DEFAULT_METRIC_REGISTRY_PATH), 'entry_count': len(METRIC_REGISTRY_ENTRIES), 'alias_count': len(METRIC_REGISTRY_ALIAS_LOOKUP)})}"
    )
    cards, metadata_lookup, benchmark_metadata = load_benchmark_metadata(local_metadata_dir)
    print(
        f"[pipeline] {json.dumps({'event': 'metadata.loaded', 'benchmark_card_count': len(cards), 'metadata_key_count': len(metadata_lookup), 'metadata_cache_dir': local_metadata_dir, 'metadata_repo': BENCHMARK_METADATA_DATASET_REPO})}"
    )

    all_configs = explicit_configs or discover_configs(local_dataset_dir, hf_token)
    if config_limit:
        all_configs = all_configs[: max(1, min(parse_positive_int(config_limit, len(all_configs)), len(all_configs)))]

    skipped_configs: list[str] = []
    evaluations: list[dict] = []

    for i in range(0, len(all_configs), config_batch_size):
        batch = all_configs[i : i + config_batch_size]
        print(f"[pipeline] {json.dumps({'event': 'config.batch.start', 'batch_index': i // config_batch_size, 'batch_size': len(batch), 'configs': batch})}")

        for config in batch:
            try:
                files = list_json_files_for_config(config, local_dataset_dir, hf_token)
                print(f"[pipeline] {json.dumps({'event': 'config.discovery', 'config': config, 'data_json_files_found': len(files), 'discovery_pages': 1, 'discovery_error': None})}")
                loaded_rows = 0
                failed_files: list[str] = []
                for dataset_path in files:
                    record = None
                    last_error = None
                    for attempt in range(1, FILE_READ_MAX_RETRIES + 1):
                        try:
                            record = read_dataset_json(dataset_path, local_dataset_dir, hf_token)
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
                    eval_results = record.get("evaluation_results") if isinstance(record.get("evaluation_results"), list) else []
                    first_result = eval_results[0] if eval_results else None
                    benchmark = as_string(record.get("evaluation_id")).split("/")[0] if record.get("evaluation_id") else None
                    passthrough = {k: v for k, v in record.items() if k not in KNOWN_TOP_LEVEL_KEYS}
                    detailed_meta = normalize_detailed_eval_meta(record.get("detailed_evaluation_results"))
                    eval_obj = {
                        "schema_version": as_string(record.get("schema_version")),
                        "evaluation_id": as_string(record.get("evaluation_id")),
                        "retrieved_timestamp": as_string(record.get("retrieved_timestamp")),
                        "benchmark": benchmark,
                        "source_data": (first_result or {}).get("source_data"),
                        "source_metadata": record.get("source_metadata"),
                        "eval_library": record.get("eval_library"),
                        "model_info": record.get("model_info") or {},
                        "generation_config": (first_result or {}).get("generation_config"),
                        "source_record_url": source_record_url,
                        "detailed_evaluation_results_meta": detailed_meta,
                        "detailed_evaluation_results": resolve_detailed_results_url(record, source_record_url),
                        "passthrough_top_level_fields": passthrough or None,
                        "evaluation_results": eval_results,
                        "benchmark_card": None,
                        "instance_level_data": None,
                        "_raw_record_payload": record,
                    }
                    evaluations.append(eval_obj)
                    loaded_rows += 1

                if failed_files:
                    message = f"Failed to load {len(failed_files)} files for config {config}"
                    print(
                        f"[pipeline] {json.dumps({'event': 'config.load.partial', 'config': config, 'row_count': loaded_rows, 'failed_files': len(failed_files), 'sample_failed_paths': failed_files[:5]})}"
                    )
                    if not allow_skipped_configs:
                        raise RuntimeError(message)

                print(
                    f"[pipeline] {json.dumps({'event': 'config.load.ok', 'config': config, 'discovered_data_json_files': len(files), 'discovery_pages': 1, 'row_count': loaded_rows})}"
                )
            except Exception as error:
                print(f"[pipeline] {json.dumps({'event': 'config.load.error', 'config': config, 'message': str(error)})}")
                if allow_skipped_configs:
                    print(f"Skipping config {config}: {error}", file=sys.stderr)
                    skipped_configs.append(config)
                else:
                    raise

        print(f"[pipeline] {json.dumps({'event': 'config.batch.done', 'batch_index': i // config_batch_size, 'cumulative_evaluations': len(evaluations), 'cumulative_skipped': len(skipped_configs)})}")

    if (not dry_run) or load_instance_in_dry_run:
        with_instance = 0
        missing_instance = 0
        for idx, evaluation in enumerate(evaluations, start=1):
            instance_data = maybe_load_instance_data(evaluation, local_dataset_dir, hf_token)
            if instance_data:
                evaluation["instance_level_data"] = instance_data
                with_instance += 1
            else:
                missing_instance += 1
            if idx % 100 == 0 or idx == len(evaluations):
                print(f"[pipeline] {json.dumps({'event': 'instance.batch.progress', 'processed': idx, 'total': len(evaluations), 'with_instance_data': with_instance, 'missing_instance_data': missing_instance})}")
        print(f"[pipeline] {json.dumps({'event': 'instance.load.summary', 'total': len(evaluations), 'with_instance_data': with_instance, 'missing_instance_data': missing_instance})}")

    for evaluation in evaluations:
        identity = canonical_model_identity(evaluation.get("model_info") or {})
        model_info = dict(evaluation.get("model_info") or {})
        model_info.update(
            {
                "normalized_id": identity["normalized_id"],
                "family_id": identity["family_id"],
                "family_slug": identity["family_slug"],
                "family_name": identity["family_name"],
                "variant_key": identity["variant_key"],
                "variant_label": identity["variant_label"],
                "model_route_id": identity["model_route_id"],
            }
        )
        evaluation["model_info"] = model_info
        evaluation["benchmark_card"] = lookup_benchmark_card(
            metadata_lookup,
            evaluation.get("benchmark"),
            canonical_benchmark_family_key(evaluation.get("benchmark")),
        )
        enriched_results = []
        for result in evaluation.get("evaluation_results") or []:
            enriched = dict(result)
            normalized = classify_evaluation_result(evaluation, enriched, evaluation["benchmark_card"])
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
            artifact_text = read_text_from_dataset_url(raw_instance_url, local_dataset_dir, hf_token)
            if artifact_text is not None:
                instance_relative_path = build_instance_artifact_relative_path(evaluation)
                instance_output_path = OUTPUT_DIR / instance_relative_path
                instance_output_path.parent.mkdir(parents=True, exist_ok=True)
                transformed_artifact_text = transform_instance_artifact_text(evaluation, artifact_text)
                instance_output_path.write_text(transformed_artifact_text, encoding="utf-8")
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
                            annotate_instance_row(evaluation, row) if isinstance(row, dict) else row
                            for row in ((evaluation.get("instance_level_data") or {}).get("instance_examples") or [])
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
            record_output_payload["detailed_evaluation_results"] = evaluation.get("detailed_evaluation_results")
            if evaluation.get("detailed_evaluation_results_meta") is not None:
                record_output_payload["detailed_evaluation_results_meta"] = evaluation.get("detailed_evaluation_results_meta")
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
                    "benchmark": normalized.get("benchmark_parent_name") or evaluation.get("benchmark"),
                    "benchmark_family_key": normalized.get("benchmark_family_key"),
                    "benchmark_family_name": normalized.get("benchmark_family_name"),
                    "benchmark_parent_key": normalized.get("benchmark_parent_key"),
                    "benchmark_parent_name": normalized.get("benchmark_parent_name"),
                    "benchmark_leaf_key": normalized.get("benchmark_leaf_key"),
                    "benchmark_leaf_name": normalized.get("benchmark_leaf_name"),
                    "benchmark_component_key": normalized.get("benchmark_component_key"),
                    "benchmark_component_name": normalized.get("benchmark_component_name"),
                    "evaluation_name": normalized.get("benchmark_leaf_name") or normalized.get("benchmark_family_name"),
                    "display_name": normalized.get("benchmark_leaf_name") or normalized.get("benchmark_family_name"),
                    "canonical_display_name": normalized.get("benchmark_leaf_name") or normalized.get("benchmark_parent_name") or normalized.get("benchmark_family_name"),
                    "is_summary_score": bool(normalized.get("is_summary_score")),
                    "category": infer_category_from_benchmark(as_string(evaluation.get("benchmark"))),
                    "source_data": result.get("source_data"),
                    "benchmark_card": None,
                    "tags": {"domains": [], "languages": [], "tasks": []},
                    "subtasks": {},
                },
            )

            # Set benchmark card and tags on first encounter
            if group["benchmark_card"] is None:
                _card = lookup_benchmark_card_for_parent(
                    metadata_lookup,
                    normalized.get("benchmark_leaf_name"),
                    normalized.get("benchmark_leaf_key"),
                    evaluation.get("benchmark"),
                    normalized.get("benchmark_family_key"),
                    (result.get("source_data") or {}).get("dataset_name"),
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
                    "subtask_key": None if subtask_key == "__root__" else normalized.get("slice_key"),
                    "subtask_name": normalized.get("slice_name"),
                    "display_name": normalized.get("slice_name") or normalized.get("benchmark_leaf_name") or normalized.get("benchmark_family_name"),
                    "canonical_display_name": join_display_name_parts(
                        normalized.get("benchmark_leaf_name") or normalized.get("benchmark_parent_name") or normalized.get("benchmark_family_name"),
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
                    "lower_is_better": bool((result.get("metric_config") or {}).get("lower_is_better")),
                    "metric_name": normalized.get("metric_name"),
                    "metric_id": normalized.get("metric_id"),
                    "metric_key": normalized.get("metric_key"),
                    "metric_source": normalized.get("metric_source"),
                    "metric_config": result.get("metric_config"),
                    "model_results": [],
                },
            )
            metric_summary["model_results"].append(
                {
                    "model_id": as_string((evaluation.get("model_info") or {}).get("family_id")),
                    "model_route_id": as_string((evaluation.get("model_info") or {}).get("model_route_id")),
                    "model_name": as_string((evaluation.get("model_info") or {}).get("family_name") or (evaluation.get("model_info") or {}).get("name")),
                    "developer": as_string((evaluation.get("model_info") or {}).get("developer")),
                    "variant_key": as_string((evaluation.get("model_info") or {}).get("variant_key")) or "default",
                    "raw_model_id": as_string((evaluation.get("model_info") or {}).get("id")),
                    "score": score,
                    "evaluation_id": evaluation.get("evaluation_id"),
                    "retrieved_timestamp": evaluation.get("retrieved_timestamp"),
                    "source_record_url": evaluation.get("source_record_url"),
                    "detailed_evaluation_results": evaluation.get("detailed_evaluation_results"),
                    "detailed_evaluation_results_meta": evaluation.get("detailed_evaluation_results_meta"),
                    "passthrough_top_level_fields": evaluation.get("passthrough_top_level_fields"),
                    "instance_level_data": evaluation.get("instance_level_data"),
                    "normalized_result": normalized,
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
                model_results = sorted(metric_summary["model_results"], key=lambda r: (r["score"], r["model_id"]))
                if not lower:
                    model_results.reverse()
                metric_summary["model_results"] = model_results
                metric_summary["models_count"] = len(model_results)
                metric_summary["top_score"] = model_results[0]["score"] if model_results else None
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

            metric_summaries.sort(key=lambda metric: (as_string(metric.get("metric_name")), as_string(metric.get("metric_summary_id"))))
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
                        "metric_names": [as_string(metric.get("metric_name")) for metric in metric_summaries],
                    }
                )

        subtask_summaries.sort(key=lambda subtask: as_string(subtask.get("display_name")))
        summary["metrics"] = root_metrics
        summary["subtasks"] = subtask_summaries
        summary["subtasks_count"] = len(subtask_summaries)
        summary["metrics_count"] = total_metric_count
        summary["models_count"] = len(model_ids_for_group)
        summary["metric_names"] = sorted(name for name in unique_metric_names if name)
        primary_metrics = root_metrics or (subtask_summaries[0]["metrics"] if subtask_summaries else [])
        primary_metric = pick_primary_metric(primary_metrics)
        summary["primary_metric_name"] = as_string(primary_metric.get("metric_name")) if primary_metric else None
        summary["top_score"] = primary_metric.get("top_score") if primary_metric and len(primary_metrics) == 1 and not subtask_summaries else None

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
                benchmark_ranks[model_id] = {"position": position, "total": len(avg_ranks)}
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
                parent_to_summary_eval_ids[parent_key].append(summary["eval_summary_id"])

    for summary in eval_summaries:
        if summary.get("is_summary_score"):
            summary["summary_score_for"] = as_string(summary.get("benchmark_parent_key"))
            summary["summary_score_for_name"] = as_string(summary.get("benchmark_parent_name"))
        else:
            parent_key = as_string(summary.get("benchmark_parent_key"))
            sibling_summary_ids = parent_to_summary_eval_ids.get(parent_key, [])
            if sibling_summary_ids:
                summary["summary_eval_ids"] = sibling_summary_ids

    eval_summaries.sort(key=lambda s: (-s.get("models_count", 0), as_string(s.get("eval_summary_id"))))

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

    model_summaries: list[dict] = []
    model_cards: list[dict] = []

    for family_id, family_evals in model_family_groups.items():
        family_evals_sorted = sorted(family_evals, key=lambda e: as_string(e.get("retrieved_timestamp")))
        latest = family_evals_sorted[-1]
        model_info = latest.get("model_info") or {}
        route_id = as_string(model_info.get("model_route_id"))
        family_name = as_string(model_info.get("family_name") or model_info.get("name") or family_id.split("/")[-1])

        by_category: dict[str, list[dict]] = defaultdict(list)
        raw_model_ids = sorted({as_string((e.get("model_info") or {}).get("id")) for e in family_evals if as_string((e.get("model_info") or {}).get("id"))})
        variants_map: dict[str, dict] = {}
        score_values: list[float] = []
        last_updated = None

        # ---- FIX 2: collect per-benchmark scores for model card ----
        benchmark_names_set: set[str] = set()
        # key = eval_summary_id, value = best score entry for that metric
        best_per_metric: dict[str, dict] = {}

        for evaluation in family_evals:
            category = infer_category_from_benchmark(as_string(evaluation.get("benchmark")))
            by_category[category].append(evaluation)
            iso = iso_from_epoch_string(evaluation.get("retrieved_timestamp"))
            last_updated = max_iso(last_updated, iso)

            model_variant_key = as_string((evaluation.get("model_info") or {}).get("variant_key") or "default")
            variant = variants_map.setdefault(
                model_variant_key,
                {
                    "variant_key": model_variant_key,
                    "variant_label": as_string((evaluation.get("model_info") or {}).get("variant_label") or "Default"),
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
                            "benchmark": benchmark_display_name or as_string(evaluation.get("benchmark")),
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

        summary = {
            "model_info": model_info,
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
                "total_evaluations": len(family_evals),
                "benchmark_count": len({as_string(e.get("benchmark")) for e in family_evals if as_string(e.get("benchmark"))}),
                "benchmark_family_count": len(
                    {
                        as_string(((result.get("normalized_result") or {}).get("benchmark_family_key")))
                        for evaluation in family_evals
                        for result in evaluation.get("evaluation_results") or []
                        if as_string(((result.get("normalized_result") or {}).get("benchmark_family_key")))
                    }
                ),
                "categories_covered": summary_categories,
                "last_updated": last_updated,
                "variants": summary["variants"],
                "score_summary": score_summary,
                # ---- FIX 2 continued: include benchmark names and per-benchmark
                # scores so the frontend compare dialog and domain pills work ----
                "benchmark_names": sorted(benchmark_names_set),
                "top_benchmark_scores": top_benchmark_scores,
            }
        )

    model_cards.sort(key=lambda m: (-m["total_evaluations"], as_string(m["model_route_id"])))
    model_summaries.sort(key=lambda m: as_string(m.get("model_route_id")))

    eval_list = {
        "evals": [
            {
                "eval_summary_id": s["eval_summary_id"],
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
                                "canonical_display_name": metric.get("canonical_display_name"),
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
                "instance_data": s.get("instance_data", {"available": False, "url_count": 0, "sample_urls": [], "models_with_loaded_instances": 0}),
            }
            for s in eval_summaries
        ],
        "totalModels": len(model_cards),
    }

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
        if existing_name is None or (developer[0:1].isupper() and not existing_name[0:1].isupper()):
            dev_name_by_slug[slug] = developer

    developers = [
        {"developer": dev_name_by_slug[slug], "model_count": len(models)}
        for slug, models in dev_group_by_slug.items()
    ]
    developers.sort(key=lambda d: (-d["model_count"], as_string(d["developer"])))

    dev_summaries = []
    for slug, models in dev_group_by_slug.items():
        developer = dev_name_by_slug[slug]
        sorted_models = sorted(models, key=lambda m: as_string(m.get("model_family_name")))
        dev_summaries.append({"developer": developer, "slug": slug, "models": sorted_models})

    manifest = {
        "generated_at": started_at,
        "model_count": len(model_cards),
        "eval_count": len(eval_summaries),
        "metric_eval_count": sum(
            len(summary.get("metrics", [])) + sum(len(subtask.get("metrics", [])) for subtask in summary.get("subtasks", []))
            for summary in eval_summaries
        ),
        "config_version": CONFIG_VERSION,
        "skipped_config_count": len(skipped_configs),
        "skipped_configs": skipped_configs,
        "source_config_count": len(all_configs),
    }

    write_json(OUTPUT_DIR / "model-cards.json", model_cards)
    write_json(OUTPUT_DIR / "eval-list.json", eval_list)
    write_json(OUTPUT_DIR / "peer-ranks.json", peer_ranks)
    write_json(OUTPUT_DIR / "comparison-index.json", comparison_index)
    write_json(OUTPUT_DIR / "benchmark-metadata.json", benchmark_metadata)
    write_json(OUTPUT_DIR / "developers.json", developers)
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    # Copy eval hierarchy into output if available; generate README
    hierarchy_path = Path("reports/eval_hierarchy.json")
    if hierarchy_path.exists():
        shutil.copy2(hierarchy_path, OUTPUT_DIR / "eval-hierarchy.json")
    readme_text = generate_readme(manifest, eval_list, benchmark_metadata, hierarchy_path)
    (OUTPUT_DIR / "README.md").write_text(readme_text, encoding="utf-8")

    for summary in model_summaries:
        write_json(OUTPUT_DIR / "models" / f"{summary['model_route_id']}.json", summary)
    for summary in eval_summaries:
        write_json(OUTPUT_DIR / "evals" / f"{summary['eval_summary_id']}.json", summary)
    for summary in dev_summaries:
        write_json(OUTPUT_DIR / "developers" / f"{summary['slug']}.json", {"developer": summary["developer"], "models": summary["models"]})

    validate_output_contract(OUTPUT_DIR)

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

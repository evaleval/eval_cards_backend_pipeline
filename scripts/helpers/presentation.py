import re
from typing import Any

from scripts.helpers.benchmark_constants import PREFERRED_BENCHMARK_DISPLAY_NAMES
from scripts.helpers.benchmark_identity import normalize_benchmark_key
from scripts.helpers.slug_utils import as_string, humanize_slug


def humanize_token_key(value: Any) -> str:
    text = re.sub(r"[._/]+", " ", as_string(value))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return humanize_slug(text)


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
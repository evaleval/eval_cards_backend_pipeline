"""Backend parity helpers — Python ports of the TS transformations in
`general-eval-card/lib/`. These exist so the pipeline can emit
frontend-ready records under `output/duckdb/v1/*.parquet`.

Each helper here MIRRORS the current TS behavior verbatim, including
quirks. The migration target is parity, not correctness — corrections
land as a separate post-migration product decision.

Sources of truth (paste any spec deviation back into the spec first):
  general-eval-card/notes/transformations/01-12*.md
  general-eval-card/notes/transformations/reshape/03,05,06,14,16*.md
  general-eval-card/notes/transformations/reshape-design.md

Module layout follows the spec docs: one section per item.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any


# ---------------------------------------------------------------------------
# 01 — Identity canonicalization (lib/model-family.ts)
# ---------------------------------------------------------------------------

TOKEN_CASE_MAP: dict[str, str] = {
    "ai": "AI",
    "coder": "Coder",
    "command": "Command",
    "chat": "Chat",
    "claude": "Claude",
    "gemini": "Gemini",
    "gemma": "Gemma",
    "gpt": "GPT",
    "haiku": "Haiku",
    "instruct": "Instruct",
    "instant": "Instant",
    "llama": "Llama",
    "max": "Max",
    "mini": "Mini",
    "mistral": "Mistral",
    "opus": "Opus",
    "phi": "Phi",
    "plus": "Plus",
    "preview": "Preview",
    "pro": "Pro",
    "qwen": "Qwen",
    "reasoning": "Reasoning",
    "sonnet": "Sonnet",
    "thinking": "Thinking",
    "turbo": "Turbo",
    "yi": "Yi",
}

_V_VERSION_RE = re.compile(r"^v\d", re.IGNORECASE)
_DATE_8_RE = re.compile(r"^(.*?)-((?:19|20)\d{6})(?:-(.+))?$")
_DIGIT_DASH_DIGIT_RE = re.compile(r"(\d)-(?=\d(?:-|$))")
_PURE_NUMERIC_TOKEN_RE = re.compile(r"^\d+(\.\d+)?$")
_NUM_UNIT_TOKEN_RE = re.compile(r"^\d+(\.\d+)?[bkmt]$", re.IGNORECASE)


def _title_case_token(token: str) -> str:
    if not token:
        return token
    # Spec 01 / TS `titleCaseToken`: TOKEN_CASE_MAP lookup uses the raw token
    # (not lowercased). Normalized tokens are already lowercase upstream so
    # this matters only when called outside the pipeline's normalize flow.
    if token in TOKEN_CASE_MAP:
        return TOKEN_CASE_MAP[token]
    # Pure-numeric → return unchanged (`"4"` stays `"4"`).
    if _PURE_NUMERIC_TOKEN_RE.match(token):
        return token
    # Num+unit → uppercase the trailing unit (`"70b"` → `"70B"`,
    # `"1.5t"` → `"1.5T"`). Without this, every parameter-count token in a
    # model handle ends up lowercased in the displayed family name.
    if _NUM_UNIT_TOKEN_RE.match(token):
        return token[:-1] + token[-1].upper()
    if _V_VERSION_RE.match(token):
        return "v" + token[1:]
    return token[:1].upper() + token[1:]


_DATE_TOKEN_RE = re.compile(r"^(?:19|20)\d{6}$")


def _humanize_handle(handle: str) -> str:
    """Mirror `lib/model-family.ts:125-137 humanizeHandle`. Splits on `-`
    only (not `_`) and formats 8-digit date tokens via `_format_version_date`
    BEFORE title-casing."""
    if not handle:
        return ""
    parts = handle.split("-")
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if _DATE_TOKEN_RE.match(part):
            out.append(_format_version_date(part))
        else:
            out.append(_title_case_token(part))
    return " ".join(out)


_HANDLE_SEP_RE = re.compile(r"[_\s/]+")


def _normalize_handle(raw_handle: str) -> str:
    """Mirror `lib/model-family.ts:82-90 normalizeHandle`. Replaces
    underscore, whitespace AND slash with `-` (the slash matters when
    a name like `"my_model/"` is treated as a raw handle — without it,
    the trailing `/` survives into the family slug)."""
    s = raw_handle.strip().lower()
    s = _HANDLE_SEP_RE.sub("-", s)
    # Digit-dash-digit collapse runs BEFORE the dash run-collapse, per TS.
    s = _DIGIT_DASH_DIGIT_RE.sub(lambda m: m.group(1) + ".", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    return s


def _format_version_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _split_version_parts(normalized_handle: str) -> dict:
    match = _DATE_8_RE.match(normalized_handle)
    if not match:
        return {
            "familySlug": normalized_handle,
            "versionDate": None,
            "versionQualifier": None,
            "variantKey": "base",
            "variantLabel": "Current",
        }
    family_slug = match.group(1)
    yyyymmdd = match.group(2)
    qualifier_raw = match.group(3)
    iso_date = _format_version_date(yyyymmdd)
    qualifier_humanized = _humanize_handle(qualifier_raw) if qualifier_raw else None
    variant_key = yyyymmdd if not qualifier_raw else f"{yyyymmdd}-{qualifier_raw}"
    variant_label = (
        iso_date if not qualifier_humanized else f"{iso_date} · {qualifier_humanized}"
    )
    return {
        "familySlug": family_slug,
        "versionDate": iso_date,
        "versionQualifier": qualifier_humanized,
        "variantKey": variant_key,
        "variantLabel": variant_label,
    }


def _strip_namespace(value: str, namespace: str) -> str:
    """Mirror `lib/model-family.ts:46-60 stripNamespace`."""
    trimmed = value.strip()
    if not trimmed:
        return trimmed
    prefix = f"{namespace.lower()}/"
    if trimmed.lower().startswith(prefix):
        return trimmed[len(prefix) :]
    return trimmed


def _get_namespace(model_info: dict) -> str:
    """Mirror `lib/model-family.ts:62-69 getNamespace`."""
    raw_id = model_info.get("id") or ""
    parts = raw_id.split("/")
    if len(parts) > 1 and parts[0]:
        return parts[0].strip().lower()
    developer = model_info.get("developer")
    if developer is None:
        developer = "unknown"
    return re.sub(r"\s+", "-", str(developer).strip().lower())


def _get_raw_handle(model_info: dict, namespace: str) -> str:
    """Mirror `lib/model-family.ts:71-80 getRawHandle`. Picks the LAST
    slash-separated segment of the id, NOT everything after the first
    slash."""
    raw_id = model_info.get("id") or ""
    parts = raw_id.split("/")
    id_handle = parts[-1].strip() if parts else ""
    if id_handle:
        return id_handle
    raw_name = model_info.get("name") or ""
    stripped = _strip_namespace(raw_name, namespace)
    return stripped or raw_name.strip()


def get_canonical_model_identity(model_info: dict) -> dict:
    """Port of `lib/model-family.ts:165 getCanonicalModelIdentity`."""
    info = model_info or {}
    namespace = _get_namespace(info)
    raw_handle = _get_raw_handle(info, namespace)
    normalized_handle = _normalize_handle(raw_handle)
    parts = _split_version_parts(normalized_handle)
    family_slug = parts["familySlug"]
    family_name = _humanize_handle(family_slug)
    # TS always concatenates `${namespace}/${familySlug}` — even if one
    # side is empty, the slash is preserved.
    family_id = f"{namespace}/{family_slug}"
    variant_key = parts["variantKey"]
    variant_label = parts["variantLabel"]
    if variant_key == "base":
        variant_display_name = family_name
    else:
        variant_display_name = f"{family_name} ({variant_label})"

    return {
        "namespace": namespace,
        "rawHandle": raw_handle,
        "normalizedHandle": normalized_handle,
        "familySlug": family_slug,
        "familyId": family_id,
        "familyName": family_name,
        "variantKey": variant_key,
        "variantLabel": variant_label,
        "variantDisplayName": variant_display_name,
        "versionDate": parts["versionDate"],
        "versionQualifier": parts["versionQualifier"],
    }


def get_model_family_route_id(family_id: str) -> str:
    return (family_id or "").replace("/", "__")


# ---------------------------------------------------------------------------
# 02 — Setup-alias merging (lib/eval-processing.ts)
# ---------------------------------------------------------------------------

_SETUP_ALIAS_SEP_RE = re.compile(r"[-_\s]+")


def normalize_setup_alias_qualifier(qualifier: str | None) -> str:
    if not qualifier:
        return ""
    return _SETUP_ALIAS_SEP_RE.sub("-", qualifier.lower()).strip("-")


def is_setup_alias_qualifier(qualifier: str | None) -> bool:
    if not qualifier:
        return False
    normalized = normalize_setup_alias_qualifier(qualifier)
    if not normalized:
        return False
    if normalized in {"prompt", "fc", "function-calling"}:
        return True
    return normalized.startswith("thinking")


def normalize_variant(
    variant_key: str,
    family_id: str,
    variant_label: str | None = None,
) -> tuple[str, str]:
    """Return ``(variant_key, variant_label)`` after setup-alias collapse.

    Spec 02: ``base`` renames to ``default``/``Default``; an explicit
    ``default`` passthrough preserves the original label (caller can
    override with ``variant_label`` arg).
    """
    if variant_key == "base":
        return ("default", "Default")
    if variant_key == "default":
        return ("default", variant_label or "Default")

    synthetic_handle = f"{family_id}-{variant_key}".strip("/")
    identity = get_canonical_model_identity({"id": synthetic_handle})
    version_date = identity.get("versionDate")
    qualifier = identity.get("versionQualifier")
    if version_date and is_setup_alias_qualifier(qualifier):
        return (version_date, version_date)
    return (identity.get("variantKey") or variant_key, identity.get("variantLabel") or variant_key)


def merge_variants(variants: list[dict]) -> list[dict]:
    """Group already-normalized variants by ``variant_key`` and roll up.

    Preserves insertion order of the first occurrence per key. Timestamp
    comparison goes through `normalize_eval_timestamp` (Variant A) so
    unix-seconds strings and ISO strings sort correctly against each other
    — `>` on raw strings is lexicographic and wrong for mixed formats.
    """
    grouped: dict[str, dict] = {}
    for variant in variants:
        key = variant.get("variant_key") or "default"
        slot = grouped.setdefault(
            key,
            {
                "variant_key": key,
                "variant_label": variant.get("variant_label") or "Default",
                "evaluation_count": 0,
                "raw_model_ids": [],
                "_seen_ids": set(),
                "last_updated": None,
                "_last_ms": float("-inf"),
            },
        )
        slot["evaluation_count"] += int(variant.get("evaluation_count") or 0)
        # TS `Array.from(new Set([...]))` keeps empty strings; we mirror.
        for rid in variant.get("raw_model_ids") or []:
            if rid not in slot["_seen_ids"]:
                slot["_seen_ids"].add(rid)
                slot["raw_model_ids"].append(rid)
        ts = variant.get("last_updated")
        if ts is not None:
            ms = normalize_eval_timestamp(ts)
            if math.isfinite(ms) and ms > slot["_last_ms"]:
                slot["_last_ms"] = ms
                slot["last_updated"] = ts
            elif slot["last_updated"] is None:
                slot["last_updated"] = ts

    # `Array.from(new Set([...]))` in TS doesn't sort — but the spec table
    # for spec 02 shows raw_model_ids sorted in the rollup output. Use
    # locale-compare semantics so case-equivalent IDs match TS ordering.
    # Position-by-position uppercase flag mirrors the UCA tertiary level
    # (lowercase before uppercase), so `"yi-6b"` sorts before `"Yi-6B"`
    # and `"Ita"` before `"ITA"`.
    def _key(value: Any) -> tuple:
        text = str(value)
        return (text.casefold(), tuple(c.isupper() for c in text), text)

    rolled: list[dict] = []
    for slot in grouped.values():
        rolled.append(
            {
                "variant_key": slot["variant_key"],
                "variant_label": slot["variant_label"],
                "evaluation_count": slot["evaluation_count"],
                "raw_model_ids": sorted(slot["raw_model_ids"], key=_key),
                "last_updated": slot["last_updated"],
            }
        )
    return rolled


# ---------------------------------------------------------------------------
# 03 — License normalization (components/eval-card.tsx)
# ---------------------------------------------------------------------------


def shorten_license(license_text: str | None) -> str:
    if not license_text or license_text == "Not specified":
        return ""
    lowered = license_text.lower()
    if "creative commons attribution 4" in lowered:
        return "CC BY 4.0"
    if "creative commons zero" in lowered:
        return "CC0"
    if "apache license 2" in lowered or "apache 2" in lowered:
        return "Apache 2.0"
    if "mit license" in lowered:
        return "MIT"
    if "cc-by-sa" in lowered:
        return "CC BY-SA"
    if len(license_text) > 24:
        return license_text[:22] + "…"
    return license_text


# ---------------------------------------------------------------------------
# 04 — Dataset URL synthesis (components/eval-card.tsx Site A)
# ---------------------------------------------------------------------------


def synthesize_dataset_url(source_data: dict | None) -> str | None:
    if not isinstance(source_data, dict):
        return None
    if "dataset_url" in source_data and source_data.get("dataset_url") is not None:
        return source_data["dataset_url"]
    url = source_data.get("url")
    if isinstance(url, list):
        if url:
            head = url[0]
            if head is not None:
                return head
    elif url is not None:
        return url
    hf_repo = source_data.get("hf_repo")
    if hf_repo:
        return f"https://huggingface.co/datasets/{hf_repo}"
    return None


# ---------------------------------------------------------------------------
# 05 — Slug candidate generation (lib/model-data.ts)
# ---------------------------------------------------------------------------

_PIPELINE_SLUG_KEEP_RE = re.compile(r"[^a-zA-Z0-9._-]")
_PIPELINE_SLUG_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def pipeline_slugify(text: str | None) -> str:
    if text is None:
        return "unknown"
    s = _PIPELINE_SLUG_CTRL_RE.sub("", str(text))
    s = _PIPELINE_SLUG_KEEP_RE.sub("_", s)
    s = s.strip("_")
    return s if s else "unknown"


def get_developer_route_id(name: str) -> str:
    return pipeline_slugify((name or "").strip().lower())


def _ordered_unique(values: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if value and value not in seen:
            seen[value] = None
    return list(seen.keys())


def get_model_detail_slug_candidates(model_id: str) -> list[str]:
    if not model_id:
        return []
    with_slash = model_id.replace("/", "__")
    with_dots = with_slash.replace(".", "-")
    candidates = [
        pipeline_slugify(with_slash),
        pipeline_slugify(with_slash.lower()),
        pipeline_slugify(with_dots),
        pipeline_slugify(with_dots.lower()),
        pipeline_slugify(model_id),
        pipeline_slugify(model_id.lower()),
    ]
    return _ordered_unique(candidates)


def get_developer_slug_candidates(input_str: str) -> list[str]:
    if not input_str:
        return []
    underscore_slug = pipeline_slugify(input_str)
    lowercase_underscore = pipeline_slugify(input_str.lower())
    lower = input_str.lower()
    hyphen_slug = re.sub(r"-+$", "", re.sub(r"^-+", "", re.sub(r"[^a-z0-9]+", "-", lower)))
    compact_slug = re.sub(r"[^a-z0-9]+", "", lower)
    raw_candidates = [
        underscore_slug,
        lowercase_underscore,
        underscore_slug.replace("_", "-") if underscore_slug else "",
        lowercase_underscore.replace("_", "-") if lowercase_underscore else "",
        hyphen_slug,
        compact_slug,
    ]
    return _ordered_unique(raw_candidates)


# ---------------------------------------------------------------------------
# 06 — Developer name canonicalization (lib/model-data.ts)
# ---------------------------------------------------------------------------

KNOWN_DEVELOPER_NAMES: dict[str, str] = {
    "openai": "OpenAI",
    "google": "Google",
    "anthropic": "Anthropic",
    "meta": "Meta",
    "microsoft": "Microsoft",
    "mistralai": "Mistral AI",
    "deepseek": "DeepSeek",
    "deepseek-ai": "DeepSeek",
    "cohere": "Cohere",
    "nvidia": "NVIDIA",
    "alibaba": "Alibaba",
    "amazon": "Amazon",
    "apple": "Apple",
    "ibm": "IBM",
    "xai": "xAI",
    "x-ai": "xAI",
}

_LOWER_LEADING_RE = re.compile(r"^[a-z]")


def normalize_developer_name(name: str | None) -> str:
    if name is None:
        return ""
    key = name.strip().lower()
    if key in KNOWN_DEVELOPER_NAMES:
        return KNOWN_DEVELOPER_NAMES[key]
    if name == name.lower() and _LOWER_LEADING_RE.match(name):
        return name[0].upper() + name[1:]
    return name


# ---------------------------------------------------------------------------
# 07 — Timestamp normalization (lib/hf-data.ts, lib/eval-processing.ts)
# ---------------------------------------------------------------------------

_PARSE_FLOAT_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def _js_number(value: str) -> float:
    """Mirror JS ``Number(value)`` — strict; whole string must parse."""
    if value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _js_parse_float(value: str) -> float:
    """Mirror JS ``Number.parseFloat`` — lenient leading-prefix parse."""
    if value is None:
        return float("nan")
    m = _PARSE_FLOAT_RE.match(str(value))
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return float("nan")


def _iso_to_ms(value: str) -> float:
    """Best-effort ISO-8601 → ms-of-epoch; returns NaN on failure."""
    from datetime import datetime, timezone

    if not value:
        return float("nan")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return float("nan")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() * 1000.0


def normalize_eval_timestamp(value: Any) -> float:
    """Variant A — Number(value), ×1000 if numeric & no dash, else Date.getTime()."""
    if value is None:
        return float("nan")
    text = str(value)
    numeric = _js_number(text)
    is_nan = math.isnan(numeric)
    if not is_nan and "-" not in text:
        return numeric * 1000.0
    return _iso_to_ms(text)


def to_comparable_timestamp(timestamp: Any) -> float:
    """Variant B — parseFloat first (no ×1000), else Date.getTime(); ``-inf`` on failure."""
    if not timestamp:
        return float("-inf")
    parsed = _js_parse_float(str(timestamp))
    if math.isfinite(parsed):
        return parsed
    iso = _iso_to_ms(str(timestamp))
    return iso if math.isfinite(iso) else float("-inf")


def to_iso_8601(value: Any) -> str | None:
    """Render a unix-seconds string OR an ISO string as canonical ISO-8601.

    Mirrors JS ``new Date(...).toISOString()`` exactly — always emits
    millisecond precision (e.g. ``2023-11-14T22:13:20.000Z``) so the
    parity output matches `createModelSummary`'s `last_updated`
    formatting in TS. Returns ``None`` for unparseable input.
    """
    from datetime import datetime, timezone

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    dt: datetime | None = None
    if "-" not in text:
        try:
            seconds = float(text)
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            return None
    else:
        try:
            candidate = text
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        except ValueError:
            return None
    if dt is None:
        return None
    base = dt.strftime("%Y-%m-%dT%H:%M:%S")
    # JS `Date.prototype.toISOString()` truncates microseconds to ms; round
    # would emit `.147Z` where TS emits `.146Z` for `microsecond=146500`.
    millis = dt.microsecond // 1000
    return f"{base}.{millis:03d}Z"


# ---------------------------------------------------------------------------
# 08 — Benchmark display names (lib/model-data.ts)
# ---------------------------------------------------------------------------

BENCHMARK_NAMES: dict[str, str] = {
    "hfopenllm_v2": "HF Open LLM v2",
    "helm_lite": "HELM Lite",
    "helm_capabilities": "HELM Capabilities",
    "helm_classic": "HELM Classic",
    "helm_instruct": "HELM Instruct",
    "helm_mmlu": "HELM MMLU",
    "reward_bench": "RewardBench",
    "reward_bench_2": "RewardBench 2",
    "bfcl": "BFCL",
    "global_mmlu_lite": "Global MMLU Lite",
    "swe_bench": "SWE-bench",
    "arc_agi": "ARC-AGI",
    "tau_bench_2": "TAU-Bench 2",
    "ace": "ACE",
    "apex_agents": "APEX Agents",
    "apex_v1": "APEX v1",
    "appworld": "AppWorld",
    "browsecompplus": "BrowseComp+",
    "livecodebenchpro": "LiveCodeBench Pro",
    "sciarena": "SciArena",
    "terminal_bench_2_0": "Terminal Bench 2.0",
    "la_leaderboard": "LA Leaderboard",
    "theory_of_mind": "Theory of Mind",
    "fibble_arena": "Fibble Arena",
    "fibble1_arena": "Fibble Arena v1",
    "fibble2_arena": "Fibble Arena v2",
    "fibble3_arena": "Fibble Arena v3",
    "fibble4_arena": "Fibble Arena v4",
    "fibble5_arena": "Fibble Arena v5",
    "wordle_arena": "Wordle Arena",
}

_BENCHMARK_KEY_NORMALIZE_RE = re.compile(r"[-.\s]+")


def normalize_benchmark_key_for_lookup(name: str | None) -> str:
    if not name:
        return ""
    s = name.lower()
    s = _BENCHMARK_KEY_NORMALIZE_RE.sub("_", s)
    return s.strip("_")


def humanize_token_first_char(name: str | None) -> str:
    if not name:
        return ""
    parts = re.split(r"[_-]+", name)
    return " ".join(p[:1].upper() + p[1:] for p in parts if p)


def get_benchmark_display_name(benchmark: str | None) -> str:
    if not benchmark:
        return ""
    key = normalize_benchmark_key_for_lookup(benchmark)
    if key in BENCHMARK_NAMES:
        return BENCHMARK_NAMES[key]
    return humanize_token_first_char(benchmark)


# ---------------------------------------------------------------------------
# 09 — Metric display name expansion (lib/eval-processing.ts)
# ---------------------------------------------------------------------------

GENERIC_EVALUATION_NAMES: set[str] = {
    "score",
    "accuracy",
    "mean win rate",
    "exact match",
    "f1",
    "pass@1",
}


def get_benchmark_name_for_metric(evaluation: dict | None, result: dict | None) -> str:
    """First non-falsy of the precedence chain in spec 09."""
    evaluation = evaluation or {}
    result = result or {}
    result_source = result.get("source_data")
    if isinstance(result_source, dict):
        ds = result_source.get("dataset_name")
        if ds:
            return str(ds)
    bench = evaluation.get("benchmark")
    if bench:
        return str(bench)
    eval_source = evaluation.get("source_data")
    if isinstance(eval_source, dict):
        ds = eval_source.get("dataset_name")
        if ds:
            return str(ds)
    eval_name = result.get("evaluation_name")
    if eval_name:
        return str(eval_name)
    return str(evaluation.get("evaluation_id") or "")


def get_evaluation_display_name(evaluation: dict | None, result: dict | None) -> str:
    """Rule 1 — generic-name expansion. Dead in production but spec'd."""
    benchmark_name = get_benchmark_name_for_metric(evaluation, result)
    metric_name = str((result or {}).get("evaluation_name") or "").strip()
    if metric_name == benchmark_name:
        return metric_name
    if metric_name.lower() in GENERIC_EVALUATION_NAMES:
        return f"{benchmark_name} - {metric_name}"
    return metric_name


def prefers_benchmark_name(entry: dict | None) -> bool:
    """Rule 2 — return True if the entry's display name is generic enough
    to swap with the benchmark display name."""
    entry = entry or {}
    benchmark_display_name = get_benchmark_display_name(
        entry.get("benchmark_parent_name") or entry.get("benchmark") or ""
    )
    raw_display = (
        entry.get("evaluation_name")
        or entry.get("display_name")
        or entry.get("benchmark_leaf_name")
        or entry.get("eval_summary_id")
    )
    if not raw_display:
        return False
    normalized = str(raw_display).strip().lower()
    return bool(benchmark_display_name) and (
        normalized.startswith("accuracy on ")
        or normalized.startswith("score on ")
        or "for scorer" in normalized
        or "model_graded" in normalized
    )


# ---------------------------------------------------------------------------
# 10 — Params parsing (lib/model-data.ts + components/eval-detail.tsx)
# ---------------------------------------------------------------------------

_PARAMS_TOKEN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(trillion|tn|t|billion|bn|b|million|mn|m|thousand|k)\b",
    re.IGNORECASE,
)
_PARAMS_NAME_TOKEN_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*([tmbk])\b", re.IGNORECASE
)
_PARAMS_NAME_BONLY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*[bB]\b")


def _params_unit_to_billions(amount: float, unit: str) -> float | None:
    unit = unit.lower()
    if unit in {"trillion", "tn", "t"}:
        return amount * 1000.0
    if unit in {"billion", "bn", "b"}:
        return amount
    if unit in {"million", "mn", "m"}:
        return amount / 1000.0
    if unit in {"thousand", "k"}:
        return amount / 1_000_000.0
    return None


def parse_params_billions(value: Any) -> float | None:
    """Variant A — polymorphic; rejects non-positive numerics."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or value <= 0:
            return None
        return float(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    compact = normalized.replace(",", "")
    token = _PARAMS_TOKEN_RE.search(compact)
    if token:
        try:
            amount = float(token.group(1))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(amount) or amount <= 0:
            return None
        return _params_unit_to_billions(amount, token.group(2))
    parsed = _js_parse_float(compact)
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def parse_params_billions_from_text(value: Any) -> float | None:
    """Variant B — strings only; allows zero/negative values."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    compact = normalized.replace(",", "")
    token = _PARAMS_TOKEN_RE.search(compact)
    if token:
        try:
            amount = float(token.group(1))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(amount):
            return None
        return _params_unit_to_billions(amount, token.group(2))
    parsed = _js_parse_float(compact)
    return parsed if math.isfinite(parsed) else None


def parse_params_billions_from_model_name_evaldetail(name: Any) -> float | None:
    """Variant C — last-token wins, units {t, b, m, k}; reproduces context-window quirk."""
    if not name:
        return None
    matches = list(_PARAMS_NAME_TOKEN_RE.finditer(str(name)))
    if not matches:
        return None
    last = matches[-1]
    try:
        amount = float(last.group(1))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount):
        return None
    return _params_unit_to_billions(amount, last.group(2))


def _nullish_first(*values: Any) -> Any:
    """JS ``??``-chain — return the first non-None / non-undefined value."""
    for value in values:
        if value is not None:
            return value
    return None


def get_params_billions_from_model_info(model_info: dict | None) -> float | None:
    """Variant D — orchestrator over additional_details / parameter_count / name.

    Mirrors TS ``components/eval-detail.tsx:157-184`` which:
      1. Picks ONE raw value via ``??`` over ``additional_details.{params_billions,
         parameter_count, num_parameters, params}``;
      2. Returns it as-is if number;
      3. Parses it via Variant B if string and finite — otherwise falls
         through to ``model_info.parameter_count`` (string), then to
         Variant C on ``model_info.name``.
    """
    if not isinstance(model_info, dict):
        return None
    details = (
        model_info.get("additional_details")
        if isinstance(model_info.get("additional_details"), dict)
        else None
    )
    raw = _nullish_first(
        (details or {}).get("params_billions"),
        (details or {}).get("parameter_count"),
        (details or {}).get("num_parameters"),
        (details or {}).get("params"),
    ) if details else None

    if isinstance(raw, bool):
        # bool is neither number nor string in TS — falls through.
        pass
    elif isinstance(raw, (int, float)):
        return float(raw)
    elif isinstance(raw, str):
        parsed = parse_params_billions_from_text(raw)
        if parsed is not None and math.isfinite(parsed):
            return parsed

    pc = model_info.get("parameter_count")
    if isinstance(pc, str):
        parsed = parse_params_billions_from_text(pc)
        if parsed is not None and math.isfinite(parsed):
            return parsed
    return parse_params_billions_from_model_name_evaldetail(model_info.get("name"))


def parse_params_billions_from_model_name_compare(name: Any) -> float | None:
    """Variant E — last-token wins, units {b, m} only, strict whole-numeric."""
    if not name:
        return None
    matches = list(re.finditer(r"\b(\d+(?:\.\d+)?)\s*([bm])\b", str(name), re.IGNORECASE))
    if not matches:
        return None
    last = matches[-1]
    try:
        amount = float(last.group(1))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount):
        return None
    return amount if last.group(2).lower() == "b" else amount / 1000.0


def parse_params_billions_from_name_id_concat(name_id: str | None) -> float | None:
    """Variant F — first-token wins, ``b|B`` only."""
    if not name_id:
        return None
    m = _PARAMS_NAME_BONLY_RE.search(str(name_id))
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 11 — Benchmark-card attachment (lib/benchmark-metadata.ts)
# ---------------------------------------------------------------------------

_BENCHMARK_PREFIX_RE = re.compile(r"^[a-z0-9_]+ ?/", re.IGNORECASE)
_BENCHMARK_DASH_UNDER_RE = re.compile(r"[_-]+")
_BENCHMARK_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")


def normalize_benchmark_key_attach(name: str | None) -> str:
    if not name:
        return ""
    s = _BENCHMARK_PREFIX_RE.sub("", name)
    s = s.lower()
    s = _BENCHMARK_DASH_UNDER_RE.sub(" ", s)
    s = _BENCHMARK_WS_RE.sub(" ", s)
    return s.strip()


def candidate_benchmark_keys_attach(name: str | None) -> list[str]:
    """TS `candidateBenchmarkKeys`: returns ``[""]`` for an empty base
    (spec Group B). Empty base must remain in the list so the eventual
    ``map.get("")`` lookup runs."""
    base = normalize_benchmark_key_attach(name)
    candidates = [
        base,
        base.replace("-", " "),
        base.replace(" ", "-"),
        _NON_ALNUM_RE.sub("", base),
    ]
    seen: dict[str, None] = {}
    for value in candidates:
        if value not in seen:
            seen[value] = None
    return list(seen.keys())


def build_benchmark_card_map(cards: dict[str, dict] | list[dict]) -> dict[str, dict]:
    """First-write-wins ``Map<key, card>`` from a card collection."""
    iterable = cards.values() if isinstance(cards, dict) else cards
    card_map: dict[str, dict] = {}
    for card in iterable:
        if not isinstance(card, dict):
            continue
        details = card.get("benchmark_details") or {}
        name = details.get("name")
        if not name:
            continue
        for key in candidate_benchmark_keys_attach(name):
            if key not in card_map:
                card_map[key] = card
    return card_map


def get_benchmark_card_by_name(name: str | None, card_map: dict[str, dict]) -> dict | None:
    for key in candidate_benchmark_keys_attach(name):
        if key in card_map:
            return card_map[key]
    return None


def attach_benchmark_card_to_summary(summary: dict, card_map: dict[str, dict]) -> dict:
    """TS contract (`lib/model-data.ts:851`): only adds ``benchmark_card``;
    does NOT spread the card across the summary."""
    if summary.get("benchmark_card"):
        return summary
    candidates = [
        summary.get("evaluation_name"),
        summary.get("composite_benchmark_name"),
        summary.get("composite_benchmark_key"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        card = get_benchmark_card_by_name(candidate, card_map)
        if card is not None:
            return {**summary, "benchmark_card": card}
    return summary


def attach_benchmark_card_to_list_item(item: dict, card_map: dict[str, dict]) -> dict:
    """List path differs from summary in candidate ORDER only — preserved
    asymmetry per spec 11."""
    if item.get("benchmark_card"):
        return item
    candidates = [
        item.get("evaluation_name"),
        item.get("composite_benchmark_key"),
        item.get("composite_benchmark_name"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        card = get_benchmark_card_by_name(candidate, card_map)
        if card is not None:
            return {**item, "benchmark_card": card}
    return item


# ---------------------------------------------------------------------------
# 12 — Instance-level data normalization (lib/hf-data.ts)
# ---------------------------------------------------------------------------


def _coerce_str(value: Any) -> str:
    """Mirror JS `String(value)` / `JSON.stringify(value)` for non-strings.

    JS `JSON.stringify` emits compact output (no spaces between
    separators); Python's default `json.dumps` adds spaces (`{"a": 1}`
    vs `{"a":1}`). Match TS by passing explicit separators.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _exact_match_to_bool(value: Any) -> bool | None:
    if value == 1:
        return True
    if value == 0:
        return False
    return None


def _parse_one_instance(raw: Any, index: int) -> dict | None:
    if not isinstance(raw, dict):
        return None
    inp = raw.get("input")
    input_value: str = ""
    if isinstance(inp, str):
        input_value = inp
    elif isinstance(inp, dict) and inp.get("raw") is not None:
        input_value = _coerce_str(inp.get("raw"))
    elif isinstance(raw.get("prompt"), str):
        input_value = raw["prompt"]
    elif isinstance(raw.get("question"), str):
        input_value = raw["question"]
    else:
        doc = raw.get("doc")
        if isinstance(doc, dict) and isinstance(doc.get("question"), str):
            input_value = doc["question"]
        elif isinstance(doc, dict):
            try:
                input_value = json.dumps(doc, ensure_ascii=False)[:500]
            except (TypeError, ValueError):
                input_value = ""

    ground_truth: str | None = None
    inp_dict = inp if isinstance(inp, dict) else None
    if inp_dict is not None and "reference" in inp_dict:
        ref = inp_dict.get("reference")
        if isinstance(ref, list):
            ground_truth = ", ".join(_coerce_str(x) for x in ref)
        elif ref is not None:
            ground_truth = _coerce_str(ref)
    if ground_truth is None:
        for field in ("ground_truth", "target", "gold"):
            if raw.get(field) is not None:
                ground_truth = _coerce_str(raw[field])
                break
    if ground_truth is None:
        doc = raw.get("doc")
        if isinstance(doc, dict) and doc.get("answer") is not None:
            ground_truth = _coerce_str(doc["answer"])

    # `if/elif` chain mirrors the TS `else if` cascade: each branch commits
    # the response (possibly to `""`) and skips the rest. Earlier impl had
    # a `if not response:` fallthrough that broke that short-circuit.
    response = ""
    if raw.get("output") is not None:
        out_val = raw["output"]
        response = out_val if isinstance(out_val, str) else _coerce_str(out_val)
    elif raw.get("response") is not None:
        response = _coerce_str(raw["response"])
    elif raw.get("model_output") is not None:
        response = _coerce_str(raw["model_output"])
    elif isinstance(raw.get("answer_attribution"), list) and raw["answer_attribution"]:
        tail = raw["answer_attribution"][-1]
        if isinstance(tail, dict) and tail.get("extracted_value") is not None:
            response = _coerce_str(tail["extracted_value"])
    elif isinstance(raw.get("messages"), list):
        for msg in reversed(raw["messages"]):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    response = content
                elif content is not None:
                    response = _coerce_str(content)
                break
    elif (
        isinstance(raw.get("filtered_resps"), list)
        and raw["filtered_resps"]
        and isinstance(raw["filtered_resps"][0], list)
        and raw["filtered_resps"][0]
    ):
        response = _coerce_str(raw["filtered_resps"][0][0])
    elif (
        isinstance(raw.get("resps"), list)
        and raw["resps"]
        and isinstance(raw["resps"][0], list)
        and raw["resps"][0]
    ):
        response = _coerce_str(raw["resps"][0][0])

    # TS `??` chain on is_correct accepts any non-null value, not just bool.
    is_correct: Any = None
    evaluation = raw.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("is_correct") is not None:
        is_correct = evaluation["is_correct"]
    elif raw.get("is_correct") is not None:
        is_correct = raw["is_correct"]
    else:
        metrics = raw.get("metrics")
        if isinstance(metrics, dict) and "exact_match" in metrics:
            is_correct = _exact_match_to_bool(metrics.get("exact_match"))

    # TS uses Object.assign in this order: evaluation → performance →
    # metadata → metrics. Later sources OVERWRITE earlier ones on key
    # collision (so `raw.metrics.score` wins over `raw.evaluation.score`).
    metadata: dict = {}
    for source in ("evaluation", "performance", "metadata", "metrics"):
        chunk = raw.get(source)
        if isinstance(chunk, dict):
            metadata.update(chunk)
    metadata_value: dict | None = metadata if metadata else None

    # TS `??` (null-coalesce) semantics — empty string sample_id wins over
    # doc_id / id / index. Earlier impl used `or` (truthy-coalesce) which
    # silently rewrote `""` to the next field.
    if raw.get("sample_id") is not None:
        sample_id = raw.get("sample_id")
    elif raw.get("doc_id") is not None:
        sample_id = raw.get("doc_id")
    elif raw.get("id") is not None:
        sample_id = raw.get("id")
    else:
        sample_id = index

    choices = raw.get("choices")
    if choices is None and isinstance(raw.get("doc"), dict):
        choices = raw["doc"].get("choices")

    return {
        "sample_id": _coerce_str(sample_id),
        "input": input_value,
        "ground_truth": ground_truth,
        "response": response,
        "is_correct": is_correct,
        "metadata": metadata_value,
        "choices": choices,
    }


def parse_instance_level_data(data: Any) -> list[dict]:
    if not data:
        return []
    if isinstance(data, dict):
        examples = data.get("instance_examples")
        if isinstance(examples, list):
            iterable = examples
        else:
            return []
    elif isinstance(data, list):
        iterable = data
    else:
        return []
    out: list[dict] = []
    for index, raw in enumerate(iterable):
        parsed = _parse_one_instance(raw, index)
        if parsed is not None:
            out.append(parsed)
    return out


# ---------------------------------------------------------------------------
# Category inference (lib/benchmark-schema.ts:182-206)
# ---------------------------------------------------------------------------


_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Safety",
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
        "Agentic",
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
        "Reasoning",
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
        "Knowledge",
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


def infer_category_from_benchmark(benchmark_name: str | None) -> str:
    """Port of `inferCategoryFromBenchmark` (lib/benchmark-schema.ts:182).

    Used as a fallback when the pipeline emits a backend `category` of
    `"other"` (~84% of evals per the migration audit). Returns a frontend
    `CategoryType` literal: Safety / Agentic / Reasoning / Knowledge / General.
    """
    if not benchmark_name:
        return "General"
    name = str(benchmark_name).lower()
    for category, tokens in _CATEGORY_PATTERNS:
        if any(token in name for token in tokens):
            return category
    return "General"


# `lib/hf-data.ts:1425` — pure lookup-table mapping. Note that `other` maps
# to `General` per current TS contract (NOT a regex fallback). The regex
# `inferCategoryFromBenchmark` only fires on surfaces where TS calls it
# directly (eval-detail summaries via `hfEvalDetailToSummary`,
# variant-rebucket via `createModelSummary`).
_PIPELINE_CATEGORY_MAP: dict[str, str] = {
    "agentic": "Agentic",
    "reasoning": "Reasoning",
    "general": "General",
    "safety": "Safety",
    "knowledge": "Knowledge",
    "other": "General",
    "coding": "General",
    "instruction_following": "General",
    "language_understanding": "General",
}


def map_hf_category_single(value: str | None) -> str:
    """`mapHFCategories([value])[0]` — pure lookup, default `General`."""
    if not value:
        return "General"
    return _PIPELINE_CATEGORY_MAP.get(str(value).lower(), "General")


def map_hf_categories(values: list[str | None] | None) -> list[str]:
    """Port of `lib/hf-data.ts:1437 mapHFCategories`. Dedup-preserving order;
    if no inputs map, returns `["General"]`."""
    out: list[str] = []
    for value in values or []:
        if not value:
            continue
        mapped = _PIPELINE_CATEGORY_MAP.get(str(value).lower(), "General")
        if mapped not in out:
            out.append(mapped)
    return out if out else ["General"]


# ---------------------------------------------------------------------------
# Reshape — score normalization (used by aggregate / score-summary stats)
# ---------------------------------------------------------------------------


def normalize_summary_score(metric_config: dict | None, score: Any) -> float:
    """Port of `lib/model-data.ts:77 normalizeSummaryScore`.

    Returns ``score`` (NOT zero) when range is zero or non-positive — TS
    behavior despite some spec docs that incorrectly say it returns 0.
    """
    config = metric_config or {}
    try:
        min_score = float(config.get("min_score") if config.get("min_score") is not None else 0)
    except (TypeError, ValueError):
        min_score = 0.0
    try:
        max_score = float(config.get("max_score") if config.get("max_score") is not None else 1)
    except (TypeError, ValueError):
        max_score = 1.0
    rng = max_score - min_score
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return 0.0
    if rng > 0:
        return (score_f - min_score) / rng
    return score_f


# ---------------------------------------------------------------------------
# Reshape 14 — score summary stats
# ---------------------------------------------------------------------------


def compute_score_summary_stats(
    rows: list[dict], metric_config: dict | None, lower_is_better: bool, benchmark_display_name: str
) -> dict:
    """Replicate ``hfEvalDetailToSummary`` finalisation per spec reshape/14.

    ``rows`` is the post-#2-dedup ``metric.model_results`` list of the primary
    metric. Each row is expected to carry ``score``, ``model_name`` (or
    ``model_id``), ``source_metadata``, ``retrieved_timestamp``,
    ``generation_config``.
    """
    if not rows:
        return {
            "models_count": 0,
            "avg_score": None,
            "avg_score_norm": None,
            "best_model": None,
            "worst_model": None,
            "evaluator_names": [],
            "source_types": [],
            "latest_source_name": benchmark_display_name or None,
            "third_party_ratio": 0.0,
            "missing_generation_config_count": 0,
        }

    scores = []
    third_party = 0
    missing_cfg = 0
    source_types_seen: set[str] = set()
    latest_ts = float("-inf")
    latest_source_name: str | None = None

    for row in rows:
        score = row.get("score")
        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass
        sm = row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {}
        if (sm or {}).get("evaluator_relationship") == "third_party":
            third_party += 1
        st = (sm or {}).get("source_type")
        if st:
            source_types_seen.add(str(st))
        gen_cfg = row.get("generation_config") or row.get("_generation_config")
        if not gen_cfg:
            missing_cfg += 1
        ts = to_comparable_timestamp(row.get("retrieved_timestamp"))
        if ts >= latest_ts:
            latest_ts = ts
            latest_source_name = (sm or {}).get("source_name") or latest_source_name

    avg_score = sum(scores) / len(scores) if scores else None
    if avg_score is None:
        avg_norm = None
    else:
        config = metric_config or {}
        try:
            min_score = float(config.get("min_score", 0))
        except (TypeError, ValueError):
            min_score = 0.0
        try:
            max_score = float(config.get("max_score", 1))
        except (TypeError, ValueError):
            max_score = 1.0
        rng = max_score - min_score
        avg_norm = (avg_score - min_score) / rng if rng > 0 else 0.0

    def _row_label(r: dict) -> str:
        return str(r.get("model_name") or r.get("model_id") or "")

    sortable = [r for r in rows if r.get("score") is not None]
    if sortable:
        sortable_sorted = sorted(
            sortable, key=lambda r: float(r.get("score") or 0.0), reverse=not lower_is_better
        )
        best_row = sortable_sorted[0]
        worst_row = sortable_sorted[-1]
        best_model = {"model_name": _row_label(best_row), "score": float(best_row["score"])}
        worst_model = {"model_name": _row_label(worst_row), "score": float(worst_row["score"])}
    else:
        best_model = None
        worst_model = None

    return {
        "models_count": len(rows),
        "avg_score": avg_score,
        "avg_score_norm": avg_norm,
        "best_model": best_model,
        "worst_model": worst_model,
        # `evaluator_names` is permanently `[]` on the active TS path.
        "evaluator_names": [],
        "source_types": sorted(source_types_seen),
        "latest_source_name": latest_source_name or (benchmark_display_name or None),
        "third_party_ratio": (third_party / len(rows)) if rows else 0.0,
        "missing_generation_config_count": missing_cfg,
    }


# ---------------------------------------------------------------------------
# Reshape 16 — per-category counts (Path A fake, Path B real)
# ---------------------------------------------------------------------------


def category_stats_fake_distribution(total_evaluations: int, categories: list[str]) -> dict[str, int]:
    """Path A — proportional split currently emitted on the model-card grid."""
    stats: dict[str, int] = {}
    if not categories:
        return stats
    per_cat = max(1, total_evaluations // len(categories)) if len(categories) > 0 else 0
    remaining = total_evaluations
    for i, cat in enumerate(categories):
        is_last = i == len(categories) - 1
        count = remaining if is_last else min(per_cat, remaining)
        stats[cat] = count
        remaining -= count
    return stats


def category_stats_distinct_count(
    evaluations_by_category: dict[str, list[dict]] | None,
    categories_covered: list[str] | None = None,
) -> dict[str, int]:
    """Path B — distinct-benchmark count from the model-detail summary."""
    stats: dict[str, int] = {}
    if not evaluations_by_category:
        return stats
    iter_categories = categories_covered or list(evaluations_by_category.keys())
    for category in iter_categories:
        evals = evaluations_by_category.get(category) or []
        names: set[str] = set()
        for evaluation in evals:
            for result in evaluation.get("evaluation_results") or []:
                names.add(get_benchmark_name_for_metric(evaluation, result))
        stats[category] = len(names)
    return stats

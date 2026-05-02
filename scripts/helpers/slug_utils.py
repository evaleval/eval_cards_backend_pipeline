import re
from typing import Any

from scripts.helpers.benchmark_identity import normalize_benchmark_key

_V_VERSION_TOKEN_RE = re.compile(r"^v\d", re.IGNORECASE)


def as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


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
        if _V_VERSION_TOKEN_RE.match(part):
            out.append("v" + part[1:])
            continue
        if len(part) <= 3 and any(c.isdigit() for c in part):
            out.append(part.upper())
        else:
            out.append(part[:1].upper() + part[1:])
    return " ".join(out)

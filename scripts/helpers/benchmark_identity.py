import re
from typing import Any

NON_ALNUM_REGEX = re.compile(r"[^a-z0-9]+")
AGENTIC_NAME_REGEX = re.compile(
    r"(appworld|swe_bench|tau_bench|browsecomp|agent|livecodebench|terminal_bench)"
)
BENCHMARK_FAMILY_REGEXES = (
    re.compile(r"^(.*?)(\d+)(_arena)$"),
    re.compile(r"^(.*?)[_-]v(\d+)$"),
)
STANDALONE_VERSIONED_BENCHMARK_KEYS = {
    "apex_v1",
}


def normalize_benchmark_key(value: Any) -> str:
    return NON_ALNUM_REGEX.sub("_", str(value or "").lower()).strip("_")


def is_agentic_benchmark_name(value: Any) -> bool:
    key = normalize_benchmark_key(value)
    if not key:
        return False
    return bool(AGENTIC_NAME_REGEX.search(key))


def canonical_benchmark_family_key(value: Any) -> str:
    key = normalize_benchmark_key(value)
    if not key:
        return ""
    if key in STANDALONE_VERSIONED_BENCHMARK_KEYS:
        return key
    for regex in BENCHMARK_FAMILY_REGEXES:
        match = regex.match(key)
        if not match:
            continue
        candidate = normalize_benchmark_key(
            "".join(
                part for index, part in enumerate(match.groups(), start=1) if index != 2
            )
        )
        if candidate:
            return candidate
    return key

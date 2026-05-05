"""Producer-owned benchmark categorisation.

Maps each benchmark to one of `{General, Reasoning, Agentic, Safety,
Knowledge}` via a layered rule set in `registry/category_mapping.json`.
The frontend's category filter reads the typed enum directly; pattern-
matching fallbacks on the consumer side are dead code.

Match rule: case-insensitive substring against entries in the benchmark's
`domains[]`, `tasks[]`, and `registry_tags[]` arrays. First-match-wins
across the priority order `domains > tasks > tags`. Unmapped benchmarks
fall through to the default category (`General`).

Drift surfacing: every classification increments either the per-category
counter or the uncategorised counter; `log_category_summary` reports both
at end-of-run so operators see when the mapping needs refreshing.

Counter ownership note: the UDF is intended to be called once per
benchmark inside Stage J (e.g. while materialising `evals_view`). Callers
that invoke it per-fact-row will inflate the counters — wrap such calls
in a per-benchmark CTE before classifying.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


CATEGORY_MAPPING_PATH = (
    Path(__file__).resolve().parent / "registry" / "category_mapping.json"
)


def _load_mapping() -> dict:
    return json.loads(CATEGORY_MAPPING_PATH.read_text(encoding="utf-8"))


_MAPPING: dict = _load_mapping()
_DEFAULT_CATEGORY: str = _MAPPING.get("default_category", "General")
_CATEGORIES: tuple[str, ...] = tuple(_MAPPING.get("categories", []))
_PRIORITY_ORDER: tuple[str, ...] = tuple(
    _MAPPING.get("priority_order", ("domains", "tasks", "tags"))
)


_categorised_counter: Counter[str] = Counter()
_uncategorised_counter: int = 0


def reset_category_counter() -> None:
    global _uncategorised_counter
    _uncategorised_counter = 0
    _categorised_counter.clear()


def get_category_counts() -> tuple[Counter[str], int]:
    return Counter(_categorised_counter), _uncategorised_counter


def categories() -> tuple[str, ...]:
    return _CATEGORIES


def default_category() -> str:
    return _DEFAULT_CATEGORY


def _matches(haystack: Iterable[str] | None, patterns: Sequence[str]) -> bool:
    if not haystack:
        return False
    lower_patterns = [p.lower() for p in patterns]
    for item in haystack:
        if not isinstance(item, str):
            continue
        item_lower = item.lower()
        for p in lower_patterns:
            if p in item_lower:
                return True
    return False


def classify_benchmark(
    domains: Sequence[str] | None,
    tasks: Sequence[str] | None,
    registry_tags: Sequence[str] | None,
) -> str:
    """Classify a benchmark by its card domains, tasks, and registry tags.

    Walks signals in priority order; within each signal, walks rules in
    declaration order; returns the first matching category. Returns the
    default category when nothing matches.
    """
    global _uncategorised_counter

    signal_values: dict[str, Sequence[str] | None] = {
        "domains": domains,
        "tasks": tasks,
        "tags": registry_tags,
    }
    rules_root = _MAPPING.get("rules", {})
    for signal in _PRIORITY_ORDER:
        rules = rules_root.get(signal, [])
        values = signal_values.get(signal)
        for rule in rules:
            patterns = rule.get("patterns", [])
            if _matches(values, patterns):
                category = rule["category"]
                _categorised_counter[category] += 1
                return category
    _uncategorised_counter += 1
    _categorised_counter[_DEFAULT_CATEGORY] += 1
    return _DEFAULT_CATEGORY


def log_category_summary(log) -> None:
    total = sum(_categorised_counter.values())
    if total == 0:
        return
    log.info("=== category mapping summary ===")
    for cat in _CATEGORIES:
        n = _categorised_counter.get(cat, 0)
        log.info("  %s: %d", cat, n)
    if _uncategorised_counter:
        rate = _uncategorised_counter / total
        log.warning(
            "  uncategorised (fell through to %s): %d / %d (%.1f%%) — "
            "consider extending registry/category_mapping.json",
            _DEFAULT_CATEGORY,
            _uncategorised_counter,
            total,
            rate * 100,
        )

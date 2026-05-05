"""Producer-owned benchmark categorisation.

Curator-supplied direct lookup (`categorized.json` from the registry seed
dir / `temp_registry_override/`) takes precedence: maps benchmark display
names → ordered list of categories. First entry is the primary; the full
list is the union surfaced as `categories`.

Falls back to a layered rule set in `registry/category_mapping.json` —
case-insensitive substring against `domains[] / tasks[] / registry_tags[]`,
first-match-wins across `domains > tasks > tags`, default `General`.

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
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


CATEGORY_MAPPING_PATH = (
    Path(__file__).resolve().parent / "registry" / "category_mapping.json"
)

# `categorized.json` ships in temp_registry_override/ (or the registry seed
# dir when the curator has folded it in). Maps display names like "MMLU-Pro"
# / "AIME 2025" / "Humanity's Last Exam" → ordered list of fine-grained
# categories. The first entry is the primary; the full list is the union
# the producer attaches alongside the scalar `category`.
def _categorized_candidate_paths() -> list[Path]:
    paths: list[Path] = []
    seed = os.environ.get("EVALCARD_REGISTRY_SEED_DIR")
    if seed:
        paths.append(Path(seed) / "categorized.json")
    # Repo-local override (this file is at src/eval_card_backend/categorisation.py)
    repo_root = Path(__file__).resolve().parents[2]
    paths.append(repo_root / "temp_registry_override" / "categorized.json")
    return paths


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"[^a-z0-9-/.]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _load_categorized() -> dict[str, list[str]]:
    """Index categorized.json by lowercased display name. Curators key the
    file by display name (e.g. "AIME 2025", "APEX Agents", "AI2 Reasoning
    Challenge (ARC)") — never by slug or canonical id."""
    out: dict[str, list[str]] = {}
    for path in _categorized_candidate_paths():
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        for k, v in raw.items():
            if not isinstance(k, str):
                continue
            cats = [str(c) for c in v] if isinstance(v, list) else []
            if not cats:
                continue
            out[k.strip().lower()] = cats
        return out  # first hit wins
    return out


_CATEGORIZED: dict[str, list[str]] = _load_categorized()


# Map fine-grained curator categories (lowercase, ~18 values in
# categorized.json) onto the pipeline's typed CategoryType enum
# (5 broad values, frontend-facing). The scalar `category` column has
# to stay inside the typed enum — downstream tests + view definitions
# enforce it. The curated fine-grained labels survive untouched in
# `classify_benchmark_categories` for when we wire a categories[] array
# through to hierarchy.json.
_FINE_TO_BROAD: dict[str, str] = {
    "agentic":                       "Agentic",
    "software_engineering":          "Agentic",
    "robustness":                    "Safety",
    "safety":                        "Safety",
    "hallucination":                 "Safety",
    "knowledge":                     "Knowledge",
    "natural_sciences":              "Knowledge",
    "humanities_and_social_sciences":"Knowledge",
    "law":                           "Knowledge",
    "finance":                       "Knowledge",
    "mathematics":                   "Reasoning",
    "logical_reasoning":             "Reasoning",
    "applied_reasoning":             "Reasoning",
    "commonsense_reasoning":         "Reasoning",
    "linguistic_core":               "Reasoning",
    "general":                       "General",
    "multimodal":                    "General",
    "other":                         "General",
}


def _to_broad_category(fine: str) -> str:
    """Translate a curator fine-grained label to the typed CategoryType
    enum. Unknown labels default to 'General' so the pipeline never
    emits a category outside the enum."""
    return _FINE_TO_BROAD.get((fine or "").strip().lower(), _DEFAULT_CATEGORY)


def lookup_categorized(display_name: str | None) -> list[str]:
    """Look up curated categories by benchmark display name (case-insensitive
    exact match). Returns [] when not present — caller falls back to the
    rule-based matcher."""
    if not _CATEGORIZED or not display_name:
        return []
    return list(_CATEGORIZED.get(str(display_name).strip().lower(), []))


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
    benchmark_id: str | None = None,
    display_name: str | None = None,
) -> str:
    """Classify a benchmark.

    Lookup order:
      1. `categorized.json` (curator-supplied direct mapping). Returns the
         FIRST entry of the list — the primary/most-load-bearing category.
      2. Layered rule set in `registry/category_mapping.json`
         (substring match against domains > tasks > tags).
      3. `_DEFAULT_CATEGORY` ("General") on no match.
    """
    global _uncategorised_counter

    direct = lookup_categorized(display_name)
    if direct:
        # Curator file ships fine-grained labels; project to the typed enum.
        category = _to_broad_category(direct[0])
        _categorised_counter[category] += 1
        return category

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


def classify_benchmark_categories(
    domains: Sequence[str] | None,
    tasks: Sequence[str] | None,
    registry_tags: Sequence[str] | None,
    benchmark_id: str | None = None,
    display_name: str | None = None,
) -> list[str]:
    """Same lookup logic as `classify_benchmark` but returns the FULL
    category list (the union per spec). Single-entry list when only the
    rule-based / default category applies — keeps the JSON shape uniform."""
    direct = lookup_categorized(display_name)
    if direct:
        return direct
    return [classify_benchmark(domains, tasks, registry_tags,
                               benchmark_id, display_name)]


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

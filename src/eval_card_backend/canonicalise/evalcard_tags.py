"""Benchmark tag resolution — replaces the 5-bucket categorisation system.

Maps each benchmark to one or more of 17 evaluation tags using a curated
lookup table (registry/evalcard_tags.json, ported from the frontend's
categories.json). Multi-pass fuzzy matching with parent inheritance and
regex fallback. Tags overlap: a benchmark can be [mathematics, reasoning].

The 17-tag vocabulary:
  general, knowledge, safety, agentic,
  mathematics, logical_reasoning, commonsense_reasoning, applied_reasoning,
  software_engineering, linguistic_core,
  multimodal, natural_sciences, humanities_and_social_sciences,
  law, finance, hallucination, robustness
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TAGS_PATH = Path(__file__).resolve().parent.parent / "registry" / "evalcard_tags.json"

VALID_TAGS: frozenset[str] = frozenset({
    "general", "knowledge", "safety", "agentic",
    "mathematics", "logical_reasoning", "commonsense_reasoning", "applied_reasoning",
    "software_engineering", "linguistic_core",
    "multimodal", "natural_sciences", "humanities_and_social_sciences",
    "law", "finance", "hallucination", "robustness",
})

# Regex fallback patterns. Maps keyword patterns to tags. Replaces the old
# 5-bucket inferCategoryFromBenchmark with the 17-tag vocabulary.
_FALLBACK_RULES: tuple[tuple[re.Pattern[str], list[str]], ...] = (
    (re.compile(r"\b(?:safety|harmful|toxic|truthful|unsafe|civilcomments|jailbreak|red[\-_]?team|adversarial)\b", re.I),
     ["safety"]),
    (re.compile(r"\b(?:agent|swe[\-_]?bench|terminal[\-_]?bench|tau[\-_]?bench|appworld|browsecomp)\b", re.I),
     ["agentic"]),
    (re.compile(r"\b(?:math|gsm|aime|minerva|olympiad|arithmetic)\b", re.I),
     ["mathematics"]),
    (re.compile(r"\b(?:code|humaneval|livecodebench|mbpp|codecontests|apps|bigcodebench|swe)\b", re.I),
     ["software_engineering"]),
    (re.compile(r"\b(?:reasoning|bbh|musr|gpqa|arc[\-_]?c|logiqa|winogrande)\b", re.I),
     ["applied_reasoning"]),
    (re.compile(r"\b(?:mmlu|knowledge|trivia|medqa|legalbench|theory[\-_]?of[\-_]?mind)\b", re.I),
     ["knowledge"]),
    (re.compile(r"\b(?:multimodal|vision|vqa|mmmu|image|video|visual)\b", re.I),
     ["multimodal"]),
    (re.compile(r"\b(?:hallucin|faithful|factual)\b", re.I),
     ["hallucination"]),
    (re.compile(r"\b(?:robust|perturbation|noisy|corrupt)\b", re.I),
     ["robustness"]),
    (re.compile(r"\b(?:legal|law\b|jurisprudence)\b", re.I),
     ["law"]),
    (re.compile(r"\b(?:finance|financial|trading|accounting)\b", re.I),
     ["finance"]),
)

# Trailing parenthetical suffix pattern, e.g. "(accuracy)", "(MariusHobbhahn)"
_PAREN_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def _normalise_loose(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().strip())


def _normalise_tight(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Module-level lookup caches, built lazily.
_loose_index: dict[str, list[str]] | None = None
_tight_index: dict[str, list[str]] | None = None
_raw_tags: dict[str, list[str]] | None = None


def _ensure_loaded() -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    global _loose_index, _tight_index, _raw_tags
    if _loose_index is not None:
        return _loose_index, _tight_index, _raw_tags  # type: ignore[return-value]
    raw = json.loads(TAGS_PATH.read_text(encoding="utf-8"))
    _raw_tags = raw
    _loose_index = {}
    _tight_index = {}
    for name, tags in raw.items():
        loose_key = _normalise_loose(name)
        tight_key = _normalise_tight(name)
        _loose_index.setdefault(loose_key, tags)
        _tight_index.setdefault(tight_key, tags)
    return _loose_index, _tight_index, _raw_tags


def resolve_benchmark_tags(
    display_name: str,
    key: str,
    parent_tags: list[str] | None = None,
) -> list[str]:
    """Multi-pass fuzzy matching with parent inheritance.

    1. Loose match on display_name or key
    2. Tight match
    3. Strip trailing parenthetical suffixes, retry loose/tight
    4. Inherit from parent_tags if provided
    5. Fallback: rule-based regex
    6. Ultimate fallback: ["general"]
    """
    loose_idx, tight_idx, _ = _ensure_loaded()

    # Pass 1: loose match
    for candidate in (display_name, key):
        if not candidate:
            continue
        loose_key = _normalise_loose(candidate)
        if loose_key in loose_idx:
            return list(loose_idx[loose_key])

    # Pass 2: tight match
    for candidate in (display_name, key):
        if not candidate:
            continue
        tight_key = _normalise_tight(candidate)
        if tight_key in tight_idx:
            return list(tight_idx[tight_key])

    # Pass 3: strip trailing parenthetical suffixes, retry
    for candidate in (display_name, key):
        if not candidate:
            continue
        stripped = candidate
        while _PAREN_SUFFIX.search(stripped):
            stripped = _PAREN_SUFFIX.sub("", stripped).strip()
            if not stripped:
                break
            loose_key = _normalise_loose(stripped)
            if loose_key in loose_idx:
                return list(loose_idx[loose_key])
            tight_key = _normalise_tight(stripped)
            if tight_key in tight_idx:
                return list(tight_idx[tight_key])

    # Pass 4: parent inheritance
    if parent_tags:
        return list(parent_tags)

    # Pass 5: regex fallback
    for candidate in (display_name, key):
        if not candidate:
            continue
        for pattern, tags in _FALLBACK_RULES:
            if pattern.search(candidate):
                return list(tags)

    # Ultimate fallback
    return ["general"]


def resolve_benchmark_tags_json(display_name: str | None, key: str | None) -> str:
    """UDF-friendly wrapper. Returns JSON-encoded tag array."""
    tags = resolve_benchmark_tags(display_name or "", key or "")
    return json.dumps(tags)


def decorate_hierarchy_tags(families: list[dict]) -> None:
    """Walk hierarchy tree, attach derivedTags to every node.

    Bottom-up: benchmarks get own tags, composites get union of
    benchmarks, families get union of composites/benchmarks.
    """
    for fam in families:
        family_tags: set[str] = set()

        for layout in ("standalone_benchmarks", "benchmarks"):
            for bench in fam.get(layout) or []:
                _decorate_benchmark(bench)
                family_tags.update(bench.get("derivedTags") or [])

        for comp in fam.get("composites") or []:
            comp_tags: set[str] = set()
            for bench in comp.get("benchmarks") or []:
                _decorate_benchmark(bench)
                comp_tags.update(bench.get("derivedTags") or [])
            comp["derivedTags"] = sorted(comp_tags)
            family_tags.update(comp_tags)

        fam["derivedTags"] = sorted(family_tags)


def _decorate_benchmark(bench: dict) -> None:
    """Attach derivedTags to a benchmark and its slices."""
    bench_tags = resolve_benchmark_tags(
        bench.get("display_name") or "",
        bench.get("key") or "",
    )
    bench["derivedTags"] = bench_tags

    for sl in bench.get("slices") or []:
        slice_tags = resolve_benchmark_tags(
            sl.get("display_name") or "",
            sl.get("key") or "",
            parent_tags=bench_tags,
        )
        sl["derivedTags"] = slice_tags

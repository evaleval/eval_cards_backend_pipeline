"""Unit tests for benchmark categorisation."""
from __future__ import annotations

import pytest

from eval_card_backend import categorisation


@pytest.fixture(autouse=True)
def _reset_counters():
    categorisation.reset_category_counter()
    yield
    categorisation.reset_category_counter()


# Canonical 18-value category set sourced from
# temp_registry_override/categorized.json (curator-supplied).
_EXPECTED_CATEGORIES = {
    "agentic", "applied_reasoning", "commonsense_reasoning", "finance",
    "general", "hallucination", "humanities_and_social_sciences",
    "knowledge", "law", "linguistic_core", "logical_reasoning",
    "mathematics", "multimodal", "natural_sciences", "other",
    "robustness", "safety", "software_engineering",
}


def test_default_category_is_general() -> None:
    assert categorisation.default_category() == "general"


def test_categories_match_typed_enum() -> None:
    assert set(categorisation.categories()) == _EXPECTED_CATEGORIES


def test_classify_by_domain_safety() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=["Safety"], tasks=None, registry_tags=None
        )
        == "safety"
    )


def test_classify_by_domain_mathematics() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=["Mathematical Reasoning"], tasks=None, registry_tags=None
        )
        == "mathematics"
    )


def test_classify_case_insensitive_substring() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=["TOXICITY DETECTION"], tasks=None, registry_tags=None
        )
        == "safety"
    )


def test_priority_domains_beats_tasks() -> None:
    # Domains say safety, tasks say knowledge — domains win.
    assert (
        categorisation.classify_benchmark(
            domains=["Bias"],
            tasks=["question_answering"],
            registry_tags=None,
        )
        == "safety"
    )


def test_priority_tasks_beats_tags() -> None:
    # No domain match → tasks. Tags would map to mathematics, but tasks win.
    assert (
        categorisation.classify_benchmark(
            domains=["unrelated"],
            tasks=["agent"],
            registry_tags=["math"],
        )
        == "agentic"
    )


def test_classify_by_tag_when_no_domain_or_task() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=None, tasks=None, registry_tags=["MMLU"]
        )
        == "knowledge"
    )


def test_unmapped_falls_through_to_default() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=["something obscure"],
            tasks=["something else"],
            registry_tags=["unknown"],
        )
        == "general"
    )


def test_handles_none_inputs() -> None:
    assert categorisation.classify_benchmark(None, None, None) == "general"


def test_handles_empty_lists() -> None:
    assert categorisation.classify_benchmark([], [], []) == "general"


def test_uncategorised_counter_tracks_default_fallthroughs() -> None:
    categorisation.reset_category_counter()
    categorisation.classify_benchmark(["Safety"], None, None)              # safety
    categorisation.classify_benchmark(["unrelated"], None, None)            # general
    categorisation.classify_benchmark(["unrelated"], ["unrelated"], None)   # general
    counts, uncategorised = categorisation.get_category_counts()
    assert counts["safety"] == 1
    assert counts["general"] == 2
    assert uncategorised == 2


def test_categorized_json_lookup_wins_over_rules() -> None:
    """Curator-supplied display name lookup precedes the rule-based matcher."""
    # AIME → mathematics in categorized.json (curated). Rules would never
    # match because there's no domain/task/tag input.
    assert (
        categorisation.classify_benchmark(
            domains=None, tasks=None, registry_tags=None,
            benchmark_id="aime", display_name="AIME"
        )
        == "mathematics"
    )

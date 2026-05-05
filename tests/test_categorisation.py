"""Unit tests for benchmark categorisation."""
from __future__ import annotations

import pytest

from eval_card_backend import categorisation


@pytest.fixture(autouse=True)
def _reset_counters():
    categorisation.reset_category_counter()
    yield
    categorisation.reset_category_counter()


def test_default_category_is_general() -> None:
    assert categorisation.default_category() == "General"


def test_categories_match_typed_enum() -> None:
    assert set(categorisation.categories()) == {
        "General", "Reasoning", "Agentic", "Safety", "Knowledge"
    }


def test_classify_by_domain_safety() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=["Safety"], tasks=None, registry_tags=None
        )
        == "Safety"
    )


def test_classify_by_domain_reasoning() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=["Mathematical Reasoning"], tasks=None, registry_tags=None
        )
        == "Reasoning"
    )


def test_classify_case_insensitive_substring() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=["TOXICITY DETECTION"], tasks=None, registry_tags=None
        )
        == "Safety"
    )


def test_priority_domains_beats_tasks() -> None:
    # Domains say Safety, tasks say Knowledge — domains win.
    assert (
        categorisation.classify_benchmark(
            domains=["Bias"],
            tasks=["question_answering"],
            registry_tags=None,
        )
        == "Safety"
    )


def test_priority_tasks_beats_tags() -> None:
    # No domain match → tasks. Tags would map to Reasoning, but tasks wins.
    assert (
        categorisation.classify_benchmark(
            domains=["unrelated"],
            tasks=["agent"],
            registry_tags=["math"],
        )
        == "Agentic"
    )


def test_classify_by_tag_when_no_domain_or_task() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=None, tasks=None, registry_tags=["MMLU"]
        )
        == "Knowledge"
    )


def test_unmapped_falls_through_to_default() -> None:
    assert (
        categorisation.classify_benchmark(
            domains=["something obscure"],
            tasks=["something else"],
            registry_tags=["unknown"],
        )
        == "General"
    )


def test_handles_none_inputs() -> None:
    assert categorisation.classify_benchmark(None, None, None) == "General"


def test_handles_empty_lists() -> None:
    assert categorisation.classify_benchmark([], [], []) == "General"


def test_uncategorised_counter_tracks_default_fallthroughs() -> None:
    categorisation.reset_category_counter()
    categorisation.classify_benchmark(["Safety"], None, None)              # Safety
    categorisation.classify_benchmark(["unrelated"], None, None)            # General
    categorisation.classify_benchmark(["unrelated"], ["unrelated"], None)   # General
    counts, uncategorised = categorisation.get_category_counts()
    assert counts["Safety"] == 1
    assert counts["General"] == 2
    assert uncategorised == 2


def test_categorised_counter_does_not_double_count_default_match() -> None:
    # If a rule explicitly maps to General, the uncategorised counter must
    # NOT increment — only true fallthroughs are uncategorised.
    categorisation.reset_category_counter()
    categorisation.classify_benchmark(["Safety"], None, None)
    _, uncategorised = categorisation.get_category_counts()
    assert uncategorised == 0


def test_non_string_items_in_arrays_are_ignored() -> None:
    # Defensive: stray non-string entries shouldn't crash classification.
    assert (
        categorisation.classify_benchmark(
            domains=[None, 42, "Safety"], tasks=None, registry_tags=None
        )
        == "Safety"
    )

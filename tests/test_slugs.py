"""Unit tests for the URL slug helpers."""
from __future__ import annotations

from urllib.parse import unquote

from eval_card_backend.slugs import (
    is_summary_score,
    metric_summary_id,
    url_encode,
)


def test_url_encode_passes_through_simple_ids() -> None:
    assert url_encode("gpt-4o") == "gpt-4o"
    assert url_encode("mmlu") == "mmlu"


def test_url_encode_escapes_path_reserved_characters() -> None:
    # Slashes, colons, spaces are all unsafe for URL path segments
    # under safe='' — the slug must round-trip through unquote.
    encoded = url_encode("openai/gpt-4o")
    assert encoded == "openai%2Fgpt-4o"
    assert unquote(encoded) == "openai/gpt-4o"

    encoded_colon = url_encode("benchmark:metric")
    assert "%3A" in encoded_colon
    assert unquote(encoded_colon) == "benchmark:metric"


def test_url_encode_handles_unicode() -> None:
    encoded = url_encode("café")
    assert unquote(encoded) == "café"


def test_url_encode_none_passthrough() -> None:
    assert url_encode(None) is None


def test_metric_summary_id_uses_colon_separator() -> None:
    slug = metric_summary_id("mmlu", "accuracy")
    # `:` must be percent-encoded so the slug is path-safe;
    # unquote recovers the literal benchmark_id:metric_id form.
    assert "%3A" in slug
    assert unquote(slug) == "mmlu:accuracy"


def test_metric_summary_id_recovers_compound_ids() -> None:
    # When benchmark_id itself contains a slash, it stays escaped on the
    # round-trip, but the `:` separator is recoverable.
    slug = metric_summary_id("helm-lite/gsm8k", "exact_match")
    decoded = unquote(slug)
    assert decoded == "helm-lite/gsm8k:exact_match"


def test_metric_summary_id_none_when_either_arg_missing() -> None:
    assert metric_summary_id(None, "accuracy") is None
    assert metric_summary_id("mmlu", None) is None
    assert metric_summary_id(None, None) is None


def test_is_summary_score_keyword_match() -> None:
    # parent != benchmark + metric in the keyword set → True
    assert is_summary_score("overall", "helm-lite", "helm-lite-gsm8k") is True
    assert is_summary_score("aggregate", "suite", "suite-leaf") is True
    assert is_summary_score("total", "parent", "child") is True
    assert is_summary_score("all", "parent", "child") is True


def test_is_summary_score_case_insensitive() -> None:
    assert is_summary_score("OVERALL", "p", "c") is True
    assert is_summary_score("Aggregate", "p", "c") is True


def test_is_summary_score_excludes_non_keywords() -> None:
    assert is_summary_score("accuracy", "p", "c") is False
    assert is_summary_score("f1", "p", "c") is False


def test_is_summary_score_requires_real_parent() -> None:
    # Standalone benchmark with metric_id='overall' is NOT a rollup —
    # the parent must exist and be distinct from the benchmark itself.
    assert is_summary_score("overall", None, "helm-lite") is False
    assert is_summary_score("overall", "helm-lite", "helm-lite") is False


def test_is_summary_score_handles_nulls() -> None:
    assert is_summary_score(None, "p", "c") is False
    assert is_summary_score("overall", "p", None) is False

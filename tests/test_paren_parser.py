"""Unit tests for the paren-form subset recognizer + slice-key
disambiguation (``c++`` ≠ ``c``).

The paren parser lives inline in ``infer_benchmark_leaf_and_slice``;
these tests exercise it via direct calls so we don't have to round-trip
through the full pipeline. The sister test in
``test_pipeline_integration.py`` verifies the end-to-end behaviour for
SWE-PolyBench and Multi-SWE-bench fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline import (  # noqa: E402
    LANGUAGE_TOKEN_ALIASES,
    humanize_token_key,
    infer_benchmark_leaf_and_slice,
    normalize_subset_key,
)


def _evaluation(benchmark: str) -> dict:
    return {"benchmark": benchmark}


def _result(dataset_name: str, evaluation_name: str | None = None) -> dict:
    return {
        "source_data": {"dataset_name": dataset_name},
        "evaluation_name": evaluation_name or dataset_name,
    }


# ---------------------------------------------------------------------------
# normalize_subset_key — punctuation-aware language-token disambiguation
# ---------------------------------------------------------------------------


def test_normalize_subset_key_preserves_cpp_vs_c():
    assert normalize_subset_key("c") == "c"
    assert normalize_subset_key("c++") == "cpp"
    # Display layer keeps the original token so users see "C++" / "C".
    assert humanize_token_key("c++") == "C++"
    assert humanize_token_key("c") == "C"


def test_normalize_subset_key_preserves_csharp_vs_c():
    assert normalize_subset_key("c#") == "csharp"
    assert normalize_subset_key("f#") == "fsharp"


def test_normalize_subset_key_handles_uppercase_and_whitespace():
    assert normalize_subset_key("C++") == "cpp"
    assert normalize_subset_key("  C++  ") == "cpp"


def test_normalize_subset_key_falls_through_to_normal_normalize():
    # Tokens not in the alias map should match normalize_benchmark_key.
    assert normalize_subset_key("Java") == "java"
    assert normalize_subset_key("closed-book") == "closed_book"


def test_language_token_aliases_table_is_canonical_lowercase():
    """All map keys must be lowercase so the lookup at runtime hits."""
    for key in LANGUAGE_TOKEN_ALIASES:
        assert key == key.lower(), f"alias key {key!r} must be lowercase"


# ---------------------------------------------------------------------------
# infer_benchmark_leaf_and_slice — paren parser
# ---------------------------------------------------------------------------


def test_paren_parser_fires_for_swe_polybench_verified():
    """Prefix 'SWE-PolyBench Verified' extends the benchmark identity
    'swe-polybench' (startswith family + '_'), so the parser collapses
    the four language datasets into one leaf with subtasks."""
    leaf_key, leaf_name, slice_key, slice_name = infer_benchmark_leaf_and_slice(
        evaluation=_evaluation("swe-polybench"),
        result=_result("SWE-PolyBench Verified (Java)"),
        benchmark_family_key="swe_polybench",
        benchmark_family_name="SWE-PolyBench",
        component_key=None,
        component_name=None,
        benchmark_card=None,
    )
    assert leaf_key == "swe_polybench_verified"
    assert leaf_name == "SWE-PolyBench Verified"
    assert slice_key == "java"
    assert slice_name == "Java"


def test_paren_parser_fires_for_swe_polybench_unqualified():
    leaf_key, leaf_name, slice_key, slice_name = infer_benchmark_leaf_and_slice(
        evaluation=_evaluation("swe-polybench"),
        result=_result("SWE-PolyBench (Python)"),
        benchmark_family_key="swe_polybench",
        benchmark_family_name="SWE-PolyBench",
        component_key=None,
        component_name=None,
        benchmark_card=None,
    )
    assert leaf_key == "swe_polybench"
    assert leaf_name == "SWE-PolyBench"
    assert slice_key == "python"
    assert slice_name == "Python"


def test_paren_parser_disambiguates_c_vs_cpp():
    """Multi-SWE-bench has both ``(c)`` and ``(c++)`` slices. Without the
    normalize_subset_key fix they collide on key 'c' (since
    normalize_benchmark_key strips '+'). With the fix, c++ → cpp."""
    _, _, c_key, c_name = infer_benchmark_leaf_and_slice(
        evaluation=_evaluation("multi-swe-bench"),
        result=_result("Multi-SWE-bench (c)"),
        benchmark_family_key="multi_swe_bench",
        benchmark_family_name="Multi-SWE-bench",
        component_key=None,
        component_name=None,
        benchmark_card=None,
    )
    _, _, cpp_key, cpp_name = infer_benchmark_leaf_and_slice(
        evaluation=_evaluation("multi-swe-bench"),
        result=_result("Multi-SWE-bench (c++)"),
        benchmark_family_key="multi_swe_bench",
        benchmark_family_name="Multi-SWE-bench",
        component_key=None,
        component_name=None,
        benchmark_card=None,
    )
    assert c_key == "c"
    assert cpp_key == "cpp"
    assert c_key != cpp_key, "c and c++ must produce distinct slice keys"
    assert c_name == "C"
    assert cpp_name == "C++"


def test_paren_parser_does_not_fire_for_aggregator_records():
    """For ``llm-stats / MMLU (CoT)`` the prefix ``MMLU`` doesn't extend
    the benchmark identity ``llm-stats``, so the parser must NOT fire.
    Aggregator suites keep their existing per-row leaf behaviour."""
    leaf_key, _, slice_key, _ = infer_benchmark_leaf_and_slice(
        evaluation=_evaluation("llm-stats"),
        result=_result("MMLU (CoT)"),
        benchmark_family_key="llm_stats",
        benchmark_family_name="LLM Stats",
        component_key="mmlu_cot",
        component_name="MMLU (CoT)",
        benchmark_card=None,
    )
    assert slice_key is None, (
        f"paren parser should not fire for aggregator records "
        f"(benchmark='llm-stats'); got slice_key={slice_key!r}"
    )


def test_paren_parser_does_not_fire_for_helm_natural_questions():
    """``helm_classic / NaturalQuestions (closed-book)`` — same shape as
    the previous test, different aggregator. Prefix doesn't extend
    benchmark identity → no fire."""
    leaf_key, _, slice_key, _ = infer_benchmark_leaf_and_slice(
        evaluation=_evaluation("helm_classic"),
        result=_result("NaturalQuestions (closed-book)"),
        benchmark_family_key="helm_classic",
        benchmark_family_name="HELM Classic",
        component_key="naturalquestions_closed_book",
        component_name="NaturalQuestions (closed-book)",
        benchmark_card=None,
    )
    assert slice_key is None

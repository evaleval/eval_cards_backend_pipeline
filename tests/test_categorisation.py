"""Unit tests for benchmark tag resolution (replaces old categorisation)."""
from __future__ import annotations

from eval_card_backend.canonicalise import evalcard_tags


def test_valid_tags_count() -> None:
    assert len(evalcard_tags.VALID_TAGS) == 17


def test_resolve_mmlu() -> None:
    tags = evalcard_tags.resolve_benchmark_tags("MMLU", "mmlu")
    assert "knowledge" in tags


def test_resolve_swe_bench() -> None:
    tags = evalcard_tags.resolve_benchmark_tags("SWE-bench Verified", "swe-bench-verified")
    assert "software_engineering" in tags


def test_resolve_unknown_falls_back_to_general() -> None:
    tags = evalcard_tags.resolve_benchmark_tags("Unknown Benchmark XYZ", "unknown-xyz")
    assert tags == ["general"]


def test_resolve_safety_regex_fallback() -> None:
    tags = evalcard_tags.resolve_benchmark_tags("ToxicTest Safety", "toxic-test-safety")
    assert "safety" in tags


def test_parent_inheritance() -> None:
    tags = evalcard_tags.resolve_benchmark_tags(
        "UnknownSlice", "unknown-slice", parent_tags=["mathematics"]
    )
    assert tags == ["mathematics"]


def test_json_wrapper() -> None:
    result = evalcard_tags.resolve_benchmark_tags_json("MMLU", "mmlu")
    assert isinstance(result, str)
    import json
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert "knowledge" in parsed


def test_all_tags_in_vocabulary() -> None:
    """Spot-check that resolved tags are always from the 17-tag vocabulary."""
    test_cases = [
        ("MMLU", "mmlu"),
        ("GSM8K", "gsm8k"),
        ("HumanEval", "humaneval"),
        ("SWE-bench", "swe-bench"),
        ("GPQA", "gpqa"),
        ("HellaSwag", "hellaswag"),
    ]
    for display, key in test_cases:
        tags = evalcard_tags.resolve_benchmark_tags(display, key)
        for t in tags:
            assert t in evalcard_tags.VALID_TAGS, f"Tag '{t}' not in vocabulary for {display}"


def test_decorate_hierarchy_tags() -> None:
    families = [
        {
            "key": "test-family",
            "display_name": "Test Family",
            "benchmarks": [
                {
                    "key": "mmlu",
                    "display_name": "MMLU",
                    "slices": [
                        {"key": "mmlu-math", "display_name": "MMLU Math"},
                    ],
                },
            ],
        },
    ]
    evalcard_tags.decorate_hierarchy_tags(families)
    fam = families[0]
    assert "derivedTags" in fam
    assert len(fam["derivedTags"]) > 0
    bench = fam["benchmarks"][0]
    assert "derivedTags" in bench
    sl = bench["slices"][0]
    assert "derivedTags" in sl


def test_paren_suffix_stripping() -> None:
    """Stripping '(accuracy)' should still match GPQA Diamond."""
    tags = evalcard_tags.resolve_benchmark_tags(
        "GPQA Diamond (accuracy)", "gpqa-diamond-accuracy"
    )
    assert len(tags) > 0
    assert tags != ["general"], "Should resolve via paren-stripping, not fall to default"

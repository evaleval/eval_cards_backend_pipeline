"""Spec-faithfulness tests for scripts/signals.py.

Tests implementation of interpretive signals:
  - Reproducibility Gap test cases TC-R1..R6
  — Reporting Completeness test cases TC-C1..C4
  — Provenance test cases TC-P1..P4
  - Comparability test cases TC-V1..V2 + TC-CP1..CP3

Plus targeted edge-case coverage for "Handling of Missing,
Absent, and Optional Data" subsections and defensive behavior on
malformed inputs (non-dict generation_args, non-string task entries,
whitespace-differing org names, etc.).

The TC-R* tests pass `SPEC_BASE_REPRODUCIBILITY_FIELDS` explicitly so they
verify spec-literal behavior even though the pipeline runtime uses a smaller
active subset; top_p and prompt_template are currently disabled.

Run with:
    uv run --with pytest --with huggingface_hub --no-project pytest tests/
"""
from scripts import signals


# ---------------------------------------------------------------------------
# Signal 1: Reproducibility Gap
# ---------------------------------------------------------------------------


SPEC_BASE = signals.SPEC_BASE_REPRODUCIBILITY_FIELDS
SPEC_AGENTIC = signals.AGENTIC_REPRODUCIBILITY_FIELDS


def test_TC_R1_all_base_populated_non_agentic():
    gen_args = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "prompt_template": "Q: {q}\nA:",
    }
    result = signals.compute_reproducibility_gap(
        gen_args, is_agentic_benchmark=False, base_fields=SPEC_BASE
    )
    assert result["has_reproducibility_gap"] is False
    assert result["missing_fields"] == []
    assert result["required_field_count"] == 4
    assert result["populated_field_count"] == 4


def test_TC_R2_partial_base_populated_with_null():
    gen_args = {
        "temperature": 0.7,
        "top_p": 0.95,
        # max_tokens absent
        "prompt_template": None,
    }
    result = signals.compute_reproducibility_gap(
        gen_args, is_agentic_benchmark=False, base_fields=SPEC_BASE
    )
    assert result["has_reproducibility_gap"] is True
    assert result["missing_fields"] == ["max_tokens", "prompt_template"]
    assert result["required_field_count"] == 4
    assert result["populated_field_count"] == 2


def test_TC_R3_agentic_all_required_populated():
    gen_args = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "prompt_template": "",  # spec: empty string is populated
        "eval_plan": {"steps": []},
        "eval_limits": {"max_steps": 10},
    }
    result = signals.compute_reproducibility_gap(
        gen_args, is_agentic_benchmark=True, base_fields=SPEC_BASE
    )
    assert result["has_reproducibility_gap"] is False
    assert result["missing_fields"] == []
    assert result["required_field_count"] == 6
    assert result["populated_field_count"] == 6


def test_TC_R4_agentic_eval_plan_absent():
    gen_args = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "prompt_template": "Solve:",
        # eval_plan absent
        "eval_limits": {"max_steps": 10},
    }
    result = signals.compute_reproducibility_gap(
        gen_args, is_agentic_benchmark=True, base_fields=SPEC_BASE
    )
    assert result["has_reproducibility_gap"] is True
    assert result["missing_fields"] == ["eval_plan"]
    assert result["required_field_count"] == 6
    assert result["populated_field_count"] == 5


def test_TC_R5_generation_args_object_absent():
    result = signals.compute_reproducibility_gap(
        None, is_agentic_benchmark=False, base_fields=SPEC_BASE
    )
    assert result["has_reproducibility_gap"] is True
    assert set(result["missing_fields"]) == {
        "temperature",
        "top_p",
        "max_tokens",
        "prompt_template",
    }
    assert result["required_field_count"] == 4
    assert result["populated_field_count"] == 0


def test_TC_R5_agentic_generation_args_object_absent():
    result = signals.compute_reproducibility_gap(
        None, is_agentic_benchmark=True, base_fields=SPEC_BASE
    )
    assert result["has_reproducibility_gap"] is True
    assert set(result["missing_fields"]) == {
        "temperature",
        "top_p",
        "max_tokens",
        "prompt_template",
        "eval_plan",
        "eval_limits",
    }
    assert result["populated_field_count"] == 0


# ---------------------------------------------------------------------------
# Active runtime subset (top_p + prompt_template disabled as of 2026-04-26)
# ---------------------------------------------------------------------------


def test_active_subset_excludes_top_p_and_prompt_template():
    assert "top_p" not in signals.BASE_REPRODUCIBILITY_FIELDS
    assert "prompt_template" not in signals.BASE_REPRODUCIBILITY_FIELDS
    # Spec literal still records all four, so the disabled fields can be
    # restored with one edit.
    assert "top_p" in signals.SPEC_BASE_REPRODUCIBILITY_FIELDS
    assert "prompt_template" in signals.SPEC_BASE_REPRODUCIBILITY_FIELDS


def test_active_subset_temperature_and_max_tokens_populated_no_gap():
    # fibble_arena / wordle_arena pattern: temperature + max_tokens populated,
    # top_p + prompt_template absent. Under the active subset this should
    # NOT flag a gap.
    gen_args = {"temperature": 0.7, "max_tokens": 256}
    result = signals.compute_reproducibility_gap(gen_args, is_agentic_benchmark=False)
    assert result["has_reproducibility_gap"] is False
    assert result["missing_fields"] == []
    assert result["required_field_count"] == 2
    assert result["populated_field_count"] == 2


def test_active_subset_max_tokens_missing_flags_gap():
    gen_args = {"temperature": 0.7}
    result = signals.compute_reproducibility_gap(gen_args, is_agentic_benchmark=False)
    assert result["has_reproducibility_gap"] is True
    assert result["missing_fields"] == ["max_tokens"]
    assert result["required_field_count"] == 2
    assert result["populated_field_count"] == 1


def test_active_subset_agentic_still_requires_eval_plan_and_limits():
    gen_args = {"temperature": 0.0, "max_tokens": 1024}
    result = signals.compute_reproducibility_gap(gen_args, is_agentic_benchmark=True)
    assert result["has_reproducibility_gap"] is True
    assert set(result["missing_fields"]) == {"eval_plan", "eval_limits"}
    assert result["required_field_count"] == 4
    assert result["populated_field_count"] == 2


# TC-R6 (no EEE record) is structurally impossible in this pipeline (we
# iterate EEE records to build outputs). The pure function takes
# generation_args directly and never receives a "no record" signal; the
# caller simply doesn't invoke it for benchmarks with no runs. Documented
# in notes/interpretive-signals-planning.md "Edge case — spec §3.4
# 'triple has no EEE record'".


# ---------------------------------------------------------------------------
# is_agentic
# ---------------------------------------------------------------------------


def test_is_agentic_via_spec_tasks_literal():
    card = {
        "purpose_and_intended_users": {"tasks": ["agentic"]}
    }
    assert signals.is_agentic("foo", card, None) is True


def test_is_agentic_via_signal_a_agentic_eval_config():
    gen_args = {"agentic_eval_config": {"role": "agent"}}
    assert signals.is_agentic("foo", None, gen_args) is True


def test_is_agentic_via_signal_b_benchmark_name_regex():
    assert signals.is_agentic("swe-bench-verified-mini", None, None) is True
    assert signals.is_agentic("appworld_test_normal", None, None) is True


def test_is_agentic_returns_false_when_no_signal():
    assert signals.is_agentic("hfopenllm_v2", None, None) is False
    assert signals.is_agentic(
        "hfopenllm_v2", {"purpose_and_intended_users": {"tasks": ["mmlu"]}}, {}
    ) is False


def test_is_agentic_signal_a_null_value_does_not_trigger():
    # null per spec is treated as absent
    gen_args = {"agentic_eval_config": None}
    assert signals.is_agentic("hfopenllm", None, gen_args) is False


# ---------------------------------------------------------------------------
# Signal 1: edge cases / unexpected inputs
# ---------------------------------------------------------------------------


def test_compute_reproducibility_gap_empty_dict_matches_none():
    """{} and None for generation_args produce identical output."""
    via_none = signals.compute_reproducibility_gap(
        None, is_agentic_benchmark=False, base_fields=SPEC_BASE
    )
    via_empty = signals.compute_reproducibility_gap(
        {}, is_agentic_benchmark=False, base_fields=SPEC_BASE
    )
    assert via_none["has_reproducibility_gap"] == via_empty["has_reproducibility_gap"]
    assert via_none["missing_fields"] == via_empty["missing_fields"]
    assert via_none["populated_field_count"] == via_empty["populated_field_count"]


def test_compute_reproducibility_gap_missing_fields_preserve_required_order():
    """Stronger than TC-R5: missing_fields order matches base+agentic tuple order."""
    result = signals.compute_reproducibility_gap(
        {}, is_agentic_benchmark=True, base_fields=SPEC_BASE, agentic_fields=SPEC_AGENTIC
    )
    assert result["missing_fields"] == [
        "temperature",
        "top_p",
        "max_tokens",
        "prompt_template",
        "eval_plan",
        "eval_limits",
    ]


def test_compute_reproducibility_gap_non_dict_generation_args():
    """A non-dict generation_args (e.g., a stray string from upstream) flags
    everything missing, not crashes."""
    result = signals.compute_reproducibility_gap(
        "stringified",  # type: ignore[arg-type]
        is_agentic_benchmark=False,
        base_fields=SPEC_BASE,
    )
    assert result["has_reproducibility_gap"] is True
    assert result["populated_field_count"] == 0


def test_is_populated_non_dict_inputs_return_false():
    """Defensive: is_populated on non-dict obj returns False, never crashes."""
    assert signals.is_populated(None, "x") is False
    assert signals.is_populated([], "x") is False
    assert signals.is_populated("a string", "x") is False
    assert signals.is_populated(123, "x") is False


def test_is_populated_falsy_but_non_null_values():
    """is_populated treats 0, False, [], {} as populated (they're not None)."""
    assert signals.is_populated({"x": 0}, "x") is True
    assert signals.is_populated({"x": False}, "x") is True
    assert signals.is_populated({"x": []}, "x") is True
    assert signals.is_populated({"x": {}}, "x") is True


def test_is_agentic_via_spec_tasks_case_insensitive():
    """Spec §3.1 lists tokens as lowercase; impl normalizes case."""
    assert signals.is_agentic(
        "foo", {"purpose_and_intended_users": {"tasks": ["AGENTIC"]}}, None
    ) is True
    assert signals.is_agentic(
        "foo", {"purpose_and_intended_users": {"tasks": ["Tool_Use"]}}, None
    ) is True


def test_is_agentic_via_spec_tasks_strips_whitespace():
    assert signals.is_agentic(
        "foo", {"purpose_and_intended_users": {"tasks": ["  agentic  "]}}, None
    ) is True


def test_is_agentic_robust_to_non_string_task_entries():
    """Defensive: tasks list may contain non-strings without crashing."""
    card = {"purpose_and_intended_users": {"tasks": [None, 42, {"x": 1}, "agentic"]}}
    assert signals.is_agentic("foo", card, None) is True
    card_no_match = {"purpose_and_intended_users": {"tasks": [None, 42, {"x": 1}]}}
    assert signals.is_agentic("foo", card_no_match, None) is False


def test_is_agentic_robust_to_non_list_tasks_field():
    """If tasks isn't a list, the spec-tasks branch is skipped, no crash."""
    card = {"purpose_and_intended_users": {"tasks": "agentic"}}
    assert signals.is_agentic("foo", card, None) is False


def test_is_agentic_robust_to_missing_or_null_purpose_block():
    assert signals.is_agentic("foo", {}, None) is False
    assert signals.is_agentic("foo", {"purpose_and_intended_users": None}, None) is False


def test_is_agentic_signal_b_substring_match_in_benchmark_name():
    """Signal B uses regex search, so 'agent' matches inside the benchmark id."""
    assert signals.is_agentic("foo_agent_bar", None, None) is True


def test_is_agentic_signal_b_handles_none_benchmark_name():
    assert signals.is_agentic(None, None, None) is False


def test_summarize_reproducibility_filters_non_dict_entries():
    """Defensive: non-dict entries (None, garbage) are filtered before counting."""
    annotations = [
        {"has_reproducibility_gap": True, "required_field_count": 4, "populated_field_count": 0},
        None,
        "garbage",
        {"has_reproducibility_gap": False, "required_field_count": 4, "populated_field_count": 4},
    ]
    summary = signals.summarize_reproducibility(annotations)
    assert summary["results_total"] == 2
    assert summary["has_reproducibility_gap_count"] == 1


def test_summarize_reproducibility_required_zero_excluded_from_avg():
    """Rows with required_field_count == 0 don't divide-by-zero the avg."""
    annotations = [
        {"has_reproducibility_gap": False, "required_field_count": 0, "populated_field_count": 0},
        {"has_reproducibility_gap": False, "required_field_count": 4, "populated_field_count": 4},
    ]
    summary = signals.summarize_reproducibility(annotations)
    assert summary["populated_ratio_avg"] == 1.0


# ---------------------------------------------------------------------------
# Signal 2: Reporting Completeness
# ---------------------------------------------------------------------------


def _empty_record():
    return {"autobenchmarkcard": {}, "eee_eval": {"source_metadata": {}}, "evalcards": {}}


def test_TC_C1_all_full_populated_no_partial_no_reserved():
    """Spec §4.6 TC-C1: full fields populated, reserved fields present-but-absent.

    Score = full_count / (full_count + reserved_count).
    Per spec wording "no reserved fields populated", reserved fields exist
    in the set but score 0.
    """
    field_set = [
        {"path": "autobenchmarkcard.benchmark_details.name", "coverage": "full"},
        {"path": "autobenchmarkcard.benchmark_details.overview", "coverage": "full"},
        {"path": "eee_eval.source_metadata.source_type", "coverage": "full"},
        {"path": "evalcards.lifecycle_status", "coverage": "reserved"},
        {"path": "evalcards.preregistration_url", "coverage": "reserved"},
    ]
    record = {
        "autobenchmarkcard": {
            "benchmark_details": {"name": "Foo", "overview": "Bar"}
        },
        "eee_eval": {"source_metadata": {"source_type": "leaderboard"}},
        "evalcards": {},
    }
    result = signals.compute_reporting_completeness(record, field_set)
    assert result["completeness_score"] == 3 / 5  # full / (full + reserved)
    assert result["total_fields_evaluated"] == 5
    assert set(result["missing_required_fields"]) == {
        "evalcards.lifecycle_status",
        "evalcards.preregistration_url",
    }
    assert result["partial_fields"] == []


def test_TC_C2_all_full_populated_one_partial_two_of_four():
    """Spec §4.6 TC-C2: full populated, one partial 2/4, reserved absent.

    Score = (full_count + 0.5) / (full_count + 1 + reserved_count).
    """
    field_set = [
        {"path": "autobenchmarkcard.benchmark_details.name", "coverage": "full"},
        {"path": "eee_eval.source_metadata.source_type", "coverage": "full"},
        {
            "path": "autobenchmarkcard.data",
            "coverage": "partial",
            "subitem_paths": [
                "autobenchmarkcard.data.source",
                "autobenchmarkcard.data.size",
                "autobenchmarkcard.data.format",
                "autobenchmarkcard.data.annotation",
            ],
        },
        {"path": "evalcards.lifecycle_status", "coverage": "reserved"},
        {"path": "evalcards.preregistration_url", "coverage": "reserved"},
    ]
    record = {
        "autobenchmarkcard": {
            "benchmark_details": {"name": "Foo"},
            "data": {"source": "url", "size": 100},
        },
        "eee_eval": {"source_metadata": {"source_type": "leaderboard"}},
        "evalcards": {},
    }
    result = signals.compute_reporting_completeness(record, field_set)
    # full_count=2, partial contributes 0.5, reserved_count=2 (both score 0)
    assert result["completeness_score"] == (2 + 0.5) / (2 + 1 + 2)
    assert result["total_fields_evaluated"] == 5
    assert set(result["missing_required_fields"]) == {
        "evalcards.lifecycle_status",
        "evalcards.preregistration_url",
    }
    assert len(result["partial_fields"]) == 1
    pf = result["partial_fields"][0]
    assert pf["field_path"] == "autobenchmarkcard.data"
    assert pf["score"] == 0.5
    assert pf["populated_subitems"] == 2
    assert pf["total_subitems"] == 4


def test_partial_field_all_subitems_absent_excluded_from_partial_fields():
    """Spec §4.4: partial-coverage field with all sub-items absent →
    score 0, appears in missing_required_fields, NOT in partial_fields.
    """
    field_set = [
        {
            "path": "autobenchmarkcard.data",
            "coverage": "partial",
            "subitem_paths": [
                "autobenchmarkcard.data.source",
                "autobenchmarkcard.data.size",
            ],
        },
    ]
    record = _empty_record()
    result = signals.compute_reporting_completeness(record, field_set)
    assert result["completeness_score"] == 0.0
    assert result["missing_required_fields"] == ["autobenchmarkcard.data"]
    assert result["partial_fields"] == []


def test_TC_C3_empty_card_only_eee_source_metadata():
    """Realistic: empty ABC card, one EEE run populating source_metadata only.

    Almost all ABC fields score 0; the 3 source_metadata fields populate.
    """
    record = {
        "autobenchmarkcard": {},
        "eee_eval": {
            "source_metadata": {
                "source_type": "leaderboard",
                "source_organization_name": "OpenAI",
                "evaluator_relationship": "first_party",
            }
        },
        "evalcards": {},
    }
    result = signals.compute_reporting_completeness(record)

    total = result["total_fields_evaluated"]
    populated = sum(
        1 for fs in result["field_scores"] if fs["score"] > 0
    )
    assert populated == 3  # only the 3 EEE source_metadata fields
    assert result["completeness_score"] == 3 / total
    assert "autobenchmarkcard.benchmark_details.name" in result["missing_required_fields"]
    assert "evalcards.lifecycle_status" in result["missing_required_fields"]


# ---------------------------------------------------------------------------
# Signal 2: edge cases / unexpected inputs
# ---------------------------------------------------------------------------


def test_completeness_unknown_coverage_type_raises():
    """Defensive: an unrecognized coverage type surfaces as ValueError, not silent drift."""
    field_set = [{"path": "a.b", "coverage": "weird"}]
    try:
        signals.compute_reporting_completeness(_empty_record(), field_set)
    except ValueError as exc:
        assert "weird" in str(exc) or "coverage" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for unknown coverage type")


def test_completeness_partial_with_missing_subitem_paths_scores_zero():
    """Spec §4.2: partial-coverage field requires subitem_paths; missing
    key falls through to score 0 (treated as fully absent)."""
    field_set = [{"path": "a.b", "coverage": "partial"}]  # no subitem_paths
    record = {"a": {"b": "anything"}}
    result = signals.compute_reporting_completeness(record, field_set)
    assert result["field_scores"][0]["score"] == 0.0
    assert "a.b" in result["missing_required_fields"]
    assert result["partial_fields"] == []


def test_completeness_partial_with_empty_subitem_paths_scores_zero():
    field_set = [{"path": "a.b", "coverage": "partial", "subitem_paths": []}]
    result = signals.compute_reporting_completeness({"a": {"b": "x"}}, field_set)
    assert result["field_scores"][0]["score"] == 0.0


def test_completeness_empty_field_set():
    """Empty field set → no fields evaluated, score 0."""
    result = signals.compute_reporting_completeness(_empty_record(), [])
    assert result["completeness_score"] == 0.0
    assert result["total_fields_evaluated"] == 0
    assert result["missing_required_fields"] == []
    assert result["partial_fields"] == []
    assert result["field_scores"] == []


def test_completeness_path_resolution_traverses_non_dict_returns_absent():
    """If an intermediate segment is a list/scalar, the leaf is absent."""
    field_set = [{"path": "a.b.c", "coverage": "full"}]
    record = {"a": {"b": [1, 2, 3]}}  # b is a list, can't traverse to .c
    result = signals.compute_reporting_completeness(record, field_set)
    assert result["field_scores"][0]["score"] == 0.0
    assert "a.b.c" in result["missing_required_fields"]


def test_completeness_falsy_but_non_null_values_count_as_populated():
    """Spec §2.3: present and non-null = populated. 0, False, "", [] all qualify."""
    field_set = [
        {"path": "a.zero", "coverage": "full"},
        {"path": "a.empty_str", "coverage": "full"},
        {"path": "a.false_val", "coverage": "full"},
        {"path": "a.empty_list", "coverage": "full"},
    ]
    record = {"a": {"zero": 0, "empty_str": "", "false_val": False, "empty_list": []}}
    result = signals.compute_reporting_completeness(record, field_set)
    assert all(fs["score"] == 1.0 for fs in result["field_scores"])
    assert result["completeness_score"] == 1.0
    assert result["missing_required_fields"] == []


def test_completeness_explicit_null_leaf_treated_as_absent():
    """Spec §2.3: null is absent; same applies to completeness leaf values."""
    field_set = [{"path": "a.b", "coverage": "full"}]
    result = signals.compute_reporting_completeness({"a": {"b": None}}, field_set)
    assert result["field_scores"][0]["score"] == 0.0
    assert "a.b" in result["missing_required_fields"]


def test_completeness_does_not_mutate_input_field_set():
    """Pure function: the field_set passed in is returned unchanged."""
    field_set = [
        {"path": "a.b", "coverage": "partial", "subitem_paths": ["a.b.x"]},
        {"path": "c", "coverage": "full"},
    ]
    field_set_snapshot = [
        {"path": "a.b", "coverage": "partial", "subitem_paths": ["a.b.x"]},
        {"path": "c", "coverage": "full"},
    ]
    record = {"a": {"b": {"x": 1}}, "c": "v"}
    signals.compute_reporting_completeness(record, field_set)
    assert field_set == field_set_snapshot


def test_completeness_partial_one_third_fraction_precision():
    """1/3 is not representable exactly; assert with tolerance."""
    field_set = [{
        "path": "a.b",
        "coverage": "partial",
        "subitem_paths": ["a.b.x", "a.b.y", "a.b.z"],
    }]
    record = {"a": {"b": {"x": 1}}}
    result = signals.compute_reporting_completeness(record, field_set)
    assert abs(result["field_scores"][0]["score"] - 1 / 3) < 1e-9
    pf = result["partial_fields"][0]
    assert pf["populated_subitems"] == 1
    assert pf["total_subitems"] == 3


def test_completeness_partial_full_score_excluded_from_partial_fields():
    """Spec §4.2: partial_fields requires 0 < score < 1. Full score 1 → excluded."""
    field_set = [{
        "path": "a.b",
        "coverage": "partial",
        "subitem_paths": ["a.b.x", "a.b.y"],
    }]
    record = {"a": {"b": {"x": 1, "y": 2}}}
    result = signals.compute_reporting_completeness(record, field_set)
    assert result["field_scores"][0]["score"] == 1.0
    assert result["partial_fields"] == []
    assert "a.b" not in result["missing_required_fields"]


def test_completeness_root_segment_missing_returns_absent():
    """Path traversal where the very first segment doesn't exist."""
    field_set = [{"path": "missing_root.b", "coverage": "full"}]
    result = signals.compute_reporting_completeness({"present": {}}, field_set)
    assert result["field_scores"][0]["score"] == 0.0
    assert "missing_root.b" in result["missing_required_fields"]


def test_completeness_reserved_field_populated_scores_one():
    """Spec §4.4 reserved-populated case: score 1, NOT in missing_required_fields."""
    field_set = [{"path": "evalcards.lifecycle_status", "coverage": "reserved"}]
    record = {"autobenchmarkcard": {}, "eee_eval": {"source_metadata": {}},
              "evalcards": {"lifecycle_status": {"status": "active"}}}
    result = signals.compute_reporting_completeness(record, field_set)
    assert result["field_scores"][0]["score"] == 1.0
    assert "evalcards.lifecycle_status" not in result["missing_required_fields"]


def test_completeness_field_scores_preserve_field_set_order():
    """field_scores[i] corresponds to field_set[i]."""
    field_set = [
        {"path": "a.x", "coverage": "full"},
        {"path": "a.y", "coverage": "full"},
        {"path": "a.z", "coverage": "full"},
    ]
    record = {"a": {"y": 1}}
    result = signals.compute_reporting_completeness(record, field_set)
    assert [fs["field_path"] for fs in result["field_scores"]] == ["a.x", "a.y", "a.z"]


def test_TC_C4_both_reserved_fields_populated():
    """When both reserved fields are populated, score includes 2 extra.

    Compared to TC-C3 (empty card, only source_metadata populated), populating
    both reserved fields adds 2 to the numerator.
    """
    base_record = {
        "autobenchmarkcard": {},
        "eee_eval": {
            "source_metadata": {
                "source_type": "leaderboard",
                "source_organization_name": "OpenAI",
                "evaluator_relationship": "first_party",
            }
        },
        "evalcards": {},
    }
    base_result = signals.compute_reporting_completeness(base_record)

    record_with_reserved = {
        "autobenchmarkcard": {},
        "eee_eval": base_record["eee_eval"],
        "evalcards": {
            "lifecycle_status": {"status": "active"},
            "preregistration_url": "https://example.com/prereg",
        },
    }
    result = signals.compute_reporting_completeness(record_with_reserved)

    total = result["total_fields_evaluated"]
    base_populated = sum(1 for fs in base_result["field_scores"] if fs["score"] > 0)
    populated = sum(1 for fs in result["field_scores"] if fs["score"] > 0)
    assert populated == base_populated + 2
    assert result["completeness_score"] == (base_populated + 2) / total
    assert "evalcards.lifecycle_status" not in result["missing_required_fields"]
    assert "evalcards.preregistration_url" not in result["missing_required_fields"]


# ---------------------------------------------------------------------------
# is_populated — spec §2.3
# ---------------------------------------------------------------------------


def test_is_populated_present_non_null():
    assert signals.is_populated({"a": 1}, "a") is True


def test_is_populated_absent():
    assert signals.is_populated({}, "a") is False


def test_is_populated_explicit_null():
    assert signals.is_populated({"a": None}, "a") is False


def test_is_populated_empty_string_is_populated():
    # spec §3.4: empty prompt_template counts as present
    assert signals.is_populated({"prompt_template": ""}, "prompt_template") is True


# ---------------------------------------------------------------------------
# summarize_reproducibility — aggregate shape
# ---------------------------------------------------------------------------


def test_summarize_reproducibility_basic():
    annotations = [
        {"has_reproducibility_gap": True, "required_field_count": 4, "populated_field_count": 0},
        {"has_reproducibility_gap": True, "required_field_count": 4, "populated_field_count": 2},
        {"has_reproducibility_gap": False, "required_field_count": 4, "populated_field_count": 4},
    ]
    summary = signals.summarize_reproducibility(annotations)
    assert summary["results_total"] == 3
    assert summary["has_reproducibility_gap_count"] == 2
    assert summary["populated_ratio_avg"] == (0.0 + 0.5 + 1.0) / 3


def test_summarize_reproducibility_empty():
    summary = signals.summarize_reproducibility([])
    assert summary["results_total"] == 0
    assert summary["has_reproducibility_gap_count"] == 0
    assert summary["populated_ratio_avg"] is None


# ---------------------------------------------------------------------------
# Signal 3: Provenance
# ---------------------------------------------------------------------------


def _row(score=None, evaluation_id=None, org=None, relationship=None,
         generation_args=None, variant_key="default", model_route_id="m"):
    return {
        "score": score,
        "evaluation_id": evaluation_id,
        "source_metadata": {
            "evaluator_relationship": relationship,
            "source_organization_name": org,
        },
        "generation_args": generation_args,
        "variant_key": variant_key,
        "model_route_id": model_route_id,
    }


def test_TC_P1_single_first_party_no_other_reports():
    rows = [_row(score=0.8, evaluation_id="e1", org="OpenAI", relationship="first_party")]
    out = signals.compute_provenance(rows)
    assert len(out) == 1
    assert out[0]["source_type"] == "first_party"
    assert out[0]["is_multi_source"] is False
    assert out[0]["first_party_only"] is True
    assert out[0]["distinct_reporting_organizations"] == 1


def test_TC_P2_two_third_party_different_orgs():
    rows = [
        _row(score=0.8, evaluation_id="e1", org="Scale AI", relationship="third_party"),
        _row(score=0.81, evaluation_id="e2", org="Artificial Analysis", relationship="third_party"),
    ]
    out = signals.compute_provenance(rows)
    assert all(a["is_multi_source"] is True for a in out)
    assert all(a["first_party_only"] is False for a in out)
    assert all(a["distinct_reporting_organizations"] == 2 for a in out)


def test_TC_P3_one_first_party_two_third_party():
    rows = [
        _row(score=0.99, evaluation_id="e1", org="OpenAI", relationship="first_party"),
        _row(score=0.91, evaluation_id="e2", org="Scale AI", relationship="third_party"),
        _row(score=0.90, evaluation_id="e3", org="Artificial Analysis", relationship="third_party"),
    ]
    out = signals.compute_provenance(rows)
    assert all(a["is_multi_source"] is True for a in out)
    assert all(a["first_party_only"] is False for a in out)
    assert out[0]["source_type"] == "first_party"
    assert out[1]["source_type"] == "third_party"
    assert out[2]["source_type"] == "third_party"


def test_TC_P4_evaluator_relationship_absent():
    rows = [_row(score=0.8, evaluation_id="e1", org="OpenAI", relationship=None)]
    out = signals.compute_provenance(rows)
    assert out[0]["source_type"] == "unspecified"


def test_provenance_other_collapses_to_unspecified():
    rows = [_row(score=0.8, evaluation_id="e1", org="LLM Stats", relationship="other")]
    out = signals.compute_provenance(rows)
    assert out[0]["source_type"] == "unspecified"


def test_provenance_collaborative_passes_through_with_first_party_only_false():
    """Spec §5.4: `collaborative` is a distinct source_type. It is NOT
    collapsed to unspecified, and `first_party_only` is false even when the
    group has a single org (the row isn't first-party)."""
    rows = [_row(score=0.8, evaluation_id="e1", org="Acme", relationship="collaborative")]
    out = signals.compute_provenance(rows)
    assert out[0]["source_type"] == "collaborative"
    assert out[0]["first_party_only"] is False
    assert out[0]["distinct_reporting_organizations"] == 1


def test_provenance_collaborative_in_mixed_group_keeps_per_row_source_type():
    """A group with one collaborative row and one first_party row should
    produce per-row source_types that match each row's input."""
    rows = [
        _row(score=0.8, evaluation_id="e1", org="Acme", relationship="collaborative"),
        _row(score=0.85, evaluation_id="e2", org="Acme", relationship="first_party"),
    ]
    out = signals.compute_provenance(rows)
    assert out[0]["source_type"] == "collaborative"
    assert out[1]["source_type"] == "first_party"
    # Both rows see the same group-level distinct_orgs (1) and is_multi_source.
    assert out[0]["is_multi_source"] is False
    assert out[1]["is_multi_source"] is False
    # First-party row IS first_party_only (single org); collaborative row is not.
    assert out[0]["first_party_only"] is False
    assert out[1]["first_party_only"] is True


def test_provenance_first_party_only_false_when_all_orgs_null():
    # spec §5.2 pseudocode: first_party_only requires distinct_orgs == 1.
    # All-null orgs gives distinct_orgs == 0 → first_party_only is false even
    # when source_type is first_party.
    rows = [_row(score=0.8, evaluation_id="e1", org=None, relationship="first_party")]
    out = signals.compute_provenance(rows)
    assert out[0]["source_type"] == "first_party"
    assert out[0]["first_party_only"] is False
    assert out[0]["distinct_reporting_organizations"] == 0


def test_provenance_whitespace_normalization_collapses_orgs():
    # The corpus's only "false multi-org" is two NYU/Princeton consortium
    # strings differing only by a double-space vs single-space after a comma.
    rows = [
        _row(score=0.8, evaluation_id="e1", org="A,  B University", relationship="third_party"),
        _row(score=0.8, evaluation_id="e2", org="A, B University", relationship="third_party"),
    ]
    out = signals.compute_provenance(rows)
    assert all(a["is_multi_source"] is False for a in out)
    assert all(a["distinct_reporting_organizations"] == 1 for a in out)


def test_provenance_case_insensitive_org_normalization():
    rows = [
        _row(score=0.8, evaluation_id="e1", org="Hugging Face", relationship="third_party"),
        _row(score=0.81, evaluation_id="e2", org="hugging face", relationship="third_party"),
    ]
    out = signals.compute_provenance(rows)
    assert all(a["is_multi_source"] is False for a in out)


# ---------------------------------------------------------------------------
# normalize_org_name + canonicalize_setup_value helpers
# ---------------------------------------------------------------------------


def test_normalize_org_name_strips_and_lowercases():
    assert signals.normalize_org_name("  OpenAI  ") == "openai"
    assert signals.normalize_org_name("Scale AI") == "scale ai"


def test_normalize_org_name_collapses_internal_whitespace():
    assert signals.normalize_org_name("A,  B") == signals.normalize_org_name("A, B")


def test_normalize_org_name_returns_none_for_empty_or_non_string():
    assert signals.normalize_org_name(None) is None
    assert signals.normalize_org_name("") is None
    assert signals.normalize_org_name("   ") is None
    assert signals.normalize_org_name(123) is None


def test_canonicalize_setup_value_dict_order_independent():
    a = signals.canonicalize_setup_value({"a": 1, "b": 2})
    b = signals.canonicalize_setup_value({"b": 2, "a": 1})
    assert a == b


def test_canonicalize_setup_value_distinguishes_null_from_value():
    assert signals.canonicalize_setup_value(None) != signals.canonicalize_setup_value(2048)


# ---------------------------------------------------------------------------
# Signal 4 / compute_threshold
# ---------------------------------------------------------------------------


def test_compute_threshold_proportion_returns_005():
    threshold, basis = signals.compute_threshold({"metric_unit": "proportion"})
    assert threshold == 0.05
    assert basis == "proportion_or_continuous_normalized"


def test_compute_threshold_continuous_normalized_returns_005():
    threshold, basis = signals.compute_threshold({"metric_kind": "continuous_normalized"})
    assert threshold == 0.05
    assert basis == "proportion_or_continuous_normalized"


def test_compute_threshold_percent_returns_5():
    threshold, basis = signals.compute_threshold({"metric_unit": "percent"})
    assert threshold == 5.0
    assert basis == "percent"


def test_compute_threshold_range_5pct():
    threshold, basis = signals.compute_threshold({"min_score": 0, "max_score": 100})
    assert threshold == 5.0
    assert basis == "range_5pct"


def test_compute_threshold_fallback_default_when_no_info():
    threshold, basis = signals.compute_threshold({})
    assert threshold == 0.05
    assert basis == "fallback_default"


def test_compute_threshold_fallback_default_when_metric_config_none():
    threshold, basis = signals.compute_threshold(None)
    assert threshold == 0.05
    assert basis == "fallback_default"


# ---------------------------------------------------------------------------
# Signal 4A: Variant Divergence
# ---------------------------------------------------------------------------


def test_TC_V1_temperatures_low_divergence_below_threshold():
    """3 triples, temperatures [0.0, 0.3, 0.7], scores [0.80, 0.82, 0.83].
    Divergence 0.03 < threshold 0.05 → has_variant_divergence: false.
    """
    rows = [
        _row(score=0.80, evaluation_id="e1", generation_args={"temperature": 0.0}),
        _row(score=0.82, evaluation_id="e2", generation_args={"temperature": 0.3}),
        _row(score=0.83, evaluation_id="e3", generation_args={"temperature": 0.7}),
    ]
    out = signals.compute_variant_divergence(rows, {"metric_unit": "proportion"}, group_id="gid")
    assert out is not None
    assert out["has_variant_divergence"] is False
    # Use approximate comparison for floats
    assert abs(out["divergence_magnitude"] - 0.03) < 1e-9
    assert out["threshold_used"] == 0.05
    assert out["triple_count_in_group"] == 3
    assert any(f["field"] == "temperature" for f in out["differing_setup_fields"])


def test_TC_V2_max_tokens_high_divergence_above_threshold():
    """3 triples, max_tokens [2048, 4096, 8192], scores [0.65, 0.73, 0.77].
    Divergence 0.12 > threshold 0.05 → has_variant_divergence: true.
    """
    rows = [
        _row(score=0.65, evaluation_id="e1", generation_args={"max_tokens": 2048}),
        _row(score=0.73, evaluation_id="e2", generation_args={"max_tokens": 4096}),
        _row(score=0.77, evaluation_id="e3", generation_args={"max_tokens": 8192}),
    ]
    out = signals.compute_variant_divergence(rows, {"metric_unit": "proportion"}, group_id="gid")
    assert out is not None
    assert out["has_variant_divergence"] is True
    assert abs(out["divergence_magnitude"] - 0.12) < 1e-9
    assert out["scores_in_group"] == [0.65, 0.73, 0.77]
    assert out["triple_count_in_group"] == 3
    fields = {f["field"] for f in out["differing_setup_fields"]}
    assert "max_tokens" in fields


def test_variant_divergence_single_row_returns_none():
    rows = [_row(score=0.8, evaluation_id="e1", generation_args={"max_tokens": 2048})]
    out = signals.compute_variant_divergence(rows, {"metric_unit": "proportion"})
    assert out is None


def test_variant_divergence_identical_setups_returns_none():
    """Spec §6.1.4: identical setups → not a variant case."""
    rows = [
        _row(score=0.8, evaluation_id="e1", generation_args={"max_tokens": 2048}),
        _row(score=0.85, evaluation_id="e2", generation_args={"max_tokens": 2048}),
    ]
    out = signals.compute_variant_divergence(rows, {"metric_unit": "proportion"})
    assert out is None


def test_variant_divergence_null_vs_explicit_treated_as_distinct():
    """Spec §6.1.4: null is distinct from an explicit value."""
    rows = [
        _row(score=0.8, evaluation_id="e1", generation_args={"max_tokens": 2048}),
        _row(score=0.9, evaluation_id="e2", generation_args={"max_tokens": None}),
    ]
    out = signals.compute_variant_divergence(rows, {"metric_unit": "proportion"})
    assert out is not None
    assert any(f["field"] == "max_tokens" for f in out["differing_setup_fields"])


def test_variant_divergence_score_scale_anomaly_when_proportion_outside_unit():
    rows = [
        _row(score=80, evaluation_id="e1", generation_args={"max_tokens": 2048}),
        _row(score=85, evaluation_id="e2", generation_args={"max_tokens": 4096}),
    ]
    out = signals.compute_variant_divergence(rows, {"metric_unit": "proportion"})
    assert out is not None
    assert out["score_scale_anomaly"] is True


def test_variant_divergence_score_scale_anomaly_false_when_in_range():
    rows = [
        _row(score=0.8, evaluation_id="e1", generation_args={"max_tokens": 2048}),
        _row(score=0.9, evaluation_id="e2", generation_args={"max_tokens": 4096}),
    ]
    out = signals.compute_variant_divergence(rows, {"metric_unit": "proportion"})
    assert out is not None
    assert out["score_scale_anomaly"] is False


def test_variant_divergence_includes_group_variant_breakdown():
    rows = [
        _row(score=0.8, evaluation_id="e1", generation_args={"max_tokens": 2048}, variant_key="2025-08-07-high"),
        _row(score=0.9, evaluation_id="e2", generation_args={"max_tokens": 4096}, variant_key="2025-08-07-low"),
    ]
    out = signals.compute_variant_divergence(rows, {"metric_unit": "proportion"})
    assert out is not None
    breakdown = {entry["variant_key"]: entry["row_count"] for entry in out["group_variant_breakdown"]}
    assert breakdown == {"2025-08-07-high": 1, "2025-08-07-low": 1}


def test_variant_divergence_carries_group_id_and_signal_version():
    rows = [
        _row(score=0.8, evaluation_id="e1", generation_args={"max_tokens": 2048}),
        _row(score=0.9, evaluation_id="e2", generation_args={"max_tokens": 4096}),
    ]
    out = signals.compute_variant_divergence(rows, {"metric_unit": "proportion"}, group_id="model__metric")
    assert out["group_id"] == "model__metric"
    assert out["signal_version"] == signals.SIGNAL_VERSION


# ---------------------------------------------------------------------------
# Signal 4B: Cross-Party Divergence
# ---------------------------------------------------------------------------


def test_TC_CP1_two_orgs_identical_setups_no_divergence():
    """2 triples, different orgs, identical setups, scores [0.80, 0.81] → no divergence."""
    rows = [
        _row(score=0.80, evaluation_id="e1", org="OpenAI", relationship="first_party",
             generation_args={"temperature": 0.0}),
        _row(score=0.81, evaluation_id="e2", org="Scale AI", relationship="third_party",
             generation_args={"temperature": 0.0}),
    ]
    out = signals.compute_cross_party_divergence(rows, {"metric_unit": "proportion"})
    assert out is not None
    assert out["has_cross_party_divergence"] is False
    assert out["organization_count"] == 2
    assert abs(out["divergence_magnitude"] - 0.01) < 1e-9
    assert out["differing_setup_fields"] == []


def test_TC_CP2_three_orgs_diverge_above_threshold():
    """3 orgs, scores [0.994, 0.912, 0.905], divergence 0.089 > 0.05."""
    rows = [
        _row(score=0.994, evaluation_id="e1", org="OpenAI", relationship="first_party"),
        _row(score=0.912, evaluation_id="e2", org="Scale AI", relationship="third_party"),
        _row(score=0.905, evaluation_id="e3", org="Artificial Analysis", relationship="third_party"),
    ]
    out = signals.compute_cross_party_divergence(rows, {"metric_unit": "proportion"})
    assert out is not None
    assert out["has_cross_party_divergence"] is True
    assert abs(out["divergence_magnitude"] - 0.089) < 1e-9
    assert out["organization_count"] == 3
    assert set(out["scores_by_organization"].keys()) == {"OpenAI", "Scale AI", "Artificial Analysis"}


def test_TC_CP3_single_org_with_variants_returns_none():
    """One org with 3 variant reports, no other org → cross-party: null."""
    rows = [
        _row(score=0.65, evaluation_id="e1", org="OpenAI", relationship="first_party",
             generation_args={"max_tokens": 2048}),
        _row(score=0.73, evaluation_id="e2", org="OpenAI", relationship="first_party",
             generation_args={"max_tokens": 4096}),
        _row(score=0.77, evaluation_id="e3", org="OpenAI", relationship="first_party",
             generation_args={"max_tokens": 8192}),
    ]
    out = signals.compute_cross_party_divergence(rows, {"metric_unit": "proportion"})
    assert out is None


def test_cross_party_aggregates_within_org_via_median():
    """Spec §6.2.4: same org with multiple variant reports → consolidate via median."""
    rows = [
        _row(score=0.7, evaluation_id="e1", org="OpenAI", relationship="first_party"),
        _row(score=0.8, evaluation_id="e2", org="OpenAI", relationship="first_party"),
        _row(score=0.9, evaluation_id="e3", org="OpenAI", relationship="first_party"),
        _row(score=0.5, evaluation_id="e4", org="Scale AI", relationship="third_party"),
    ]
    out = signals.compute_cross_party_divergence(rows, {"metric_unit": "proportion"})
    assert out is not None
    # OpenAI's median of [0.7, 0.8, 0.9] is 0.8; Scale AI is 0.5; divergence 0.3.
    assert out["scores_by_organization"]["OpenAI"] == 0.8
    assert out["scores_by_organization"]["Scale AI"] == 0.5
    assert abs(out["divergence_magnitude"] - 0.3) < 1e-9


def test_cross_party_all_null_orgs_returns_none():
    rows = [
        _row(score=0.8, evaluation_id="e1", org=None, relationship="third_party"),
        _row(score=0.81, evaluation_id="e2", org=None, relationship="third_party"),
    ]
    out = signals.compute_cross_party_divergence(rows, {"metric_unit": "proportion"})
    assert out is None


def test_cross_party_excludes_null_orgs_from_set():
    """Mixed: one org named, one null → only one named org → returns None."""
    rows = [
        _row(score=0.8, evaluation_id="e1", org="OpenAI", relationship="first_party"),
        _row(score=0.81, evaluation_id="e2", org=None, relationship="third_party"),
    ]
    out = signals.compute_cross_party_divergence(rows, {"metric_unit": "proportion"})
    assert out is None


def test_cross_party_diverging_setups_listed_in_differing_setup_fields():
    rows = [
        _row(score=0.8, evaluation_id="e1", org="OpenAI", relationship="first_party",
             generation_args={"prompt_template": "custom"}),
        _row(score=0.7, evaluation_id="e2", org="Scale AI", relationship="third_party",
             generation_args={"prompt_template": "default"}),
    ]
    out = signals.compute_cross_party_divergence(rows, {"metric_unit": "proportion"})
    assert out is not None
    fields = {f["field"] for f in out["differing_setup_fields"]}
    assert "prompt_template" in fields


def test_cross_party_setups_absent_empty_differing_fields():
    """Spec §6.2.4: scores diverge but no setup fields differ → still flagged."""
    rows = [
        _row(score=0.6, evaluation_id="e1", org="OpenAI", relationship="first_party",
             generation_args=None),
        _row(score=0.9, evaluation_id="e2", org="Scale AI", relationship="third_party",
             generation_args=None),
    ]
    out = signals.compute_cross_party_divergence(rows, {"metric_unit": "proportion"})
    assert out is not None
    assert out["has_cross_party_divergence"] is True
    assert out["differing_setup_fields"] == []


# ---------------------------------------------------------------------------
# aggregated_setup - lower-median rule
# ---------------------------------------------------------------------------


def test_aggregated_setup_single_row_returns_its_setup():
    rows = [_row(score=0.8, evaluation_id="e1", generation_args={"max_tokens": 2048})]
    out = signals.aggregated_setup(rows)
    assert out == {"max_tokens": 2048}


def test_aggregated_setup_odd_count_returns_median_setup():
    """3 rows sorted by score: median is the middle row's setup."""
    rows = [
        _row(score=0.7, evaluation_id="e1", generation_args={"max_tokens": 1024}),
        _row(score=0.8, evaluation_id="e2", generation_args={"max_tokens": 2048}),
        _row(score=0.9, evaluation_id="e3", generation_args={"max_tokens": 4096}),
    ]
    out = signals.aggregated_setup(rows)
    assert out == {"max_tokens": 2048}


def test_aggregated_setup_even_count_returns_lower_median_setup():
    """4 rows: (n - 1) // 2 = 1 → second-lowest score's setup."""
    rows = [
        _row(score=0.6, evaluation_id="e1", generation_args={"max_tokens": 1024}),
        _row(score=0.7, evaluation_id="e2", generation_args={"max_tokens": 2048}),
        _row(score=0.8, evaluation_id="e3", generation_args={"max_tokens": 4096}),
        _row(score=0.9, evaluation_id="e4", generation_args={"max_tokens": 8192}),
    ]
    out = signals.aggregated_setup(rows)
    assert out == {"max_tokens": 2048}


def test_aggregated_setup_deterministic_under_input_reordering():
    rows = [
        _row(score=0.7, evaluation_id="e1", generation_args={"max_tokens": 1024}),
        _row(score=0.8, evaluation_id="e2", generation_args={"max_tokens": 2048}),
        _row(score=0.9, evaluation_id="e3", generation_args={"max_tokens": 4096}),
    ]
    forward = signals.aggregated_setup(rows)
    backward = signals.aggregated_setup(list(reversed(rows)))
    assert forward == backward


def test_aggregated_setup_tie_break_by_evaluation_id():
    """Two rows with identical scores → lexicographic tie-break on evaluation_id."""
    rows = [
        _row(score=0.8, evaluation_id="e_z", generation_args={"max_tokens": 9999}),
        _row(score=0.8, evaluation_id="e_a", generation_args={"max_tokens": 1111}),
    ]
    # n=2 → index 0 (lower median). Sort is by (score, evaluation_id);
    # "e_a" < "e_z" so the e_a row is at index 0.
    out = signals.aggregated_setup(rows)
    assert out == {"max_tokens": 1111}


# ---------------------------------------------------------------------------
# Rollup helpers
# ---------------------------------------------------------------------------


def test_summarize_provenance_counts_groups_separately_from_rows():
    per_row = [
        {"source_type": "first_party"},
        {"source_type": "first_party"},
        {"source_type": "third_party"},
        {"source_type": "unspecified"},
    ]
    per_group = [
        {"is_multi_source": True, "first_party_only_in_group": False},
        {"is_multi_source": False, "first_party_only_in_group": True},
        {"is_multi_source": False, "first_party_only_in_group": False},
    ]
    out = signals.summarize_provenance(per_row, per_group)
    assert out["total_results"] == 4
    assert out["total_groups"] == 3
    assert out["multi_source_groups"] == 1
    assert out["first_party_only_groups"] == 1
    assert out["source_type_distribution"] == {
        "first_party": 2,
        "third_party": 1,
        "collaborative": 0,
        "unspecified": 1,
    }


def test_summarize_provenance_distribution_always_has_four_buckets():
    out = signals.summarize_provenance([], [])
    assert set(out["source_type_distribution"].keys()) == {
        "first_party", "third_party", "collaborative", "unspecified"
    }
    assert all(v == 0 for v in out["source_type_distribution"].values())


def test_summarize_comparability_eligibility_vs_divergent():
    per_group = [
        {"variant_divergence": {"has_variant_divergence": True},
         "cross_party_divergence": None},
        {"variant_divergence": {"has_variant_divergence": False},
         "cross_party_divergence": {"has_cross_party_divergence": True}},
        {"variant_divergence": None,
         "cross_party_divergence": None},
    ]
    out = signals.summarize_comparability(per_group)
    assert out["total_groups"] == 3
    assert out["groups_with_variant_check"] == 2
    assert out["variant_divergent_count"] == 1
    assert out["groups_with_cross_party_check"] == 1
    assert out["cross_party_divergent_count"] == 1


def test_summarize_comparability_empty():
    out = signals.summarize_comparability([])
    assert out == {
        "total_groups": 0,
        "groups_with_variant_check": 0,
        "groups_with_cross_party_check": 0,
        "variant_divergent_count": 0,
        "cross_party_divergent_count": 0,
    }


# ---------------------------------------------------------------------------
# Corpus-level aggregates
# ---------------------------------------------------------------------------


def _repro_input(missing_fields, required, populated, is_agentic=False):
    return {
        "annotation": {
            "has_reproducibility_gap": len(missing_fields) > 0,
            "missing_fields": list(missing_fields),
            "required_field_count": required,
            "populated_field_count": populated,
        },
        "is_agentic": is_agentic,
    }


# --- aggregate_reproducibility  ---------------------------


def test_aggregate_reproducibility_empty_corpus():
    out = signals.aggregate_reproducibility([])
    assert out["total_triples"] == 0
    assert out["triples_with_reproducibility_gap"] == 0
    assert out["reproducibility_gap_rate"] is None  # division by zero avoided
    assert out["agentic_triples"] == 0
    # Per-field block still has all active-runtime keys with zero counts
    expected_fields = set(signals.BASE_REPRODUCIBILITY_FIELDS) | set(signals.AGENTIC_REPRODUCIBILITY_FIELDS)
    assert set(out["per_field_missingness"].keys()) == expected_fields


def test_aggregate_reproducibility_overall_gap_rate():
    rows = [
        _repro_input([], 2, 2),                              # complete
        _repro_input(["max_tokens"], 2, 1),                  # gap
        _repro_input(["temperature", "max_tokens"], 2, 0),   # gap
    ]
    out = signals.aggregate_reproducibility(rows)
    assert out["total_triples"] == 3
    assert out["triples_with_reproducibility_gap"] == 2
    assert abs(out["reproducibility_gap_rate"] - 2/3) < 1e-9


def test_aggregate_reproducibility_agentic_aware_denominator():
    """Spec §8.3.2: eval_plan/eval_limits use agentic-only denominator.

    With 4 total rows, only 1 agentic. eval_plan missing on the 1 agentic
    row → missing_rate = 1/1 (denominator is agentic_only=1), NOT 1/4.
    """
    rows = [
        _repro_input([], 2, 2, is_agentic=False),                            # base full
        _repro_input([], 2, 2, is_agentic=False),                            # base full
        _repro_input(["max_tokens"], 2, 1, is_agentic=False),                 # base gap
        _repro_input(["eval_plan"], 4, 3, is_agentic=True),                   # agentic, eval_plan absent
    ]
    out = signals.aggregate_reproducibility(rows)
    assert out["total_triples"] == 4
    assert out["agentic_triples"] == 1

    pf = out["per_field_missingness"]
    # max_tokens: base, denominator=4 (all triples), 1 missing
    assert pf["max_tokens"]["missing_count"] == 1
    assert pf["max_tokens"]["denominator"] == "all_triples"
    assert pf["max_tokens"]["denominator_count"] == 4
    assert abs(pf["max_tokens"]["missing_rate"] - 1/4) < 1e-9
    # eval_plan: agentic-only, denominator=1, 1 missing → rate=1.0
    assert pf["eval_plan"]["missing_count"] == 1
    assert pf["eval_plan"]["denominator"] == "agentic_only"
    assert pf["eval_plan"]["denominator_count"] == 1
    assert pf["eval_plan"]["missing_rate"] == 1.0
    # eval_limits: agentic-only, 0 missing → rate=0.0 (not None)
    assert pf["eval_limits"]["denominator"] == "agentic_only"
    assert pf["eval_limits"]["denominator_count"] == 1
    assert pf["eval_limits"]["missing_rate"] == 0.0


def test_aggregate_reproducibility_no_agentic_rows_agentic_field_rate_none():
    """When 0 agentic rows, agentic-field denominator is 0 → rate is None
    (not 0.0, not crash)."""
    rows = [
        _repro_input([], 2, 2, is_agentic=False),
        _repro_input(["temperature"], 2, 1, is_agentic=False),
    ]
    out = signals.aggregate_reproducibility(rows)
    pf = out["per_field_missingness"]
    assert pf["eval_plan"]["denominator_count"] == 0
    assert pf["eval_plan"]["missing_rate"] is None
    assert pf["eval_limits"]["missing_rate"] is None


def test_aggregate_reproducibility_unknown_missing_fields_ignored():
    """Active runtime tracks only BASE+AGENTIC fields; missing_fields entries
    outside that set don't break or get counted (forward-compat: spec base
    fields disabled in active subset don't show up)."""
    rows = [
        _repro_input(["temperature", "stale_field_name"], 2, 1, is_agentic=False),
    ]
    out = signals.aggregate_reproducibility(rows)
    pf = out["per_field_missingness"]
    expected = set(signals.BASE_REPRODUCIBILITY_FIELDS) | set(signals.AGENTIC_REPRODUCIBILITY_FIELDS)
    assert set(pf.keys()) == expected
    assert pf["temperature"]["missing_count"] == 1


# --- aggregate_completeness ---------------------------------------


def test_aggregate_completeness_empty_corpus():
    out = signals.aggregate_completeness([])
    assert out == {
        "total_benchmarks": 0,
        "completeness_score_mean": None,
        "completeness_score_median": None,
        "per_field_population": {},
    }


def test_aggregate_completeness_mean_and_median():
    inputs = [
        {"completeness_score": 0.0, "field_scores": []},
        {"completeness_score": 0.5, "field_scores": []},
        {"completeness_score": 1.0, "field_scores": []},
    ]
    out = signals.aggregate_completeness(inputs)
    assert out["total_benchmarks"] == 3
    assert abs(out["completeness_score_mean"] - 0.5) < 1e-9
    assert out["completeness_score_median"] == 0.5


def test_aggregate_completeness_per_field_aggregation():
    """For a field appearing in 3 benchmarks with scores [1.0, 0.5, 0.0]:
    mean=0.5, populated_rate=2/3 (any score>0), fully_populated_rate=1/3.
    """
    inputs = [
        {"completeness_score": 0.6, "field_scores": [{"field_path": "a.b", "score": 1.0, "coverage_type": "full"}]},
        {"completeness_score": 0.4, "field_scores": [{"field_path": "a.b", "score": 0.5, "coverage_type": "partial"}]},
        {"completeness_score": 0.2, "field_scores": [{"field_path": "a.b", "score": 0.0, "coverage_type": "full"}]},
    ]
    out = signals.aggregate_completeness(inputs)
    pop = out["per_field_population"]["a.b"]
    assert abs(pop["mean_score"] - 0.5) < 1e-9
    assert abs(pop["populated_rate"] - 2/3) < 1e-9
    assert abs(pop["fully_populated_rate"] - 1/3) < 1e-9
    assert pop["benchmark_count"] == 3


def test_aggregate_completeness_field_only_in_some_benchmarks():
    """Different benchmarks may report different field_scores — the
    aggregate per field uses only the benchmarks that include that field."""
    inputs = [
        {"completeness_score": 0.5, "field_scores": [
            {"field_path": "a.shared", "score": 1.0, "coverage_type": "full"},
            {"field_path": "a.only_b1", "score": 1.0, "coverage_type": "full"},
        ]},
        {"completeness_score": 0.5, "field_scores": [
            {"field_path": "a.shared", "score": 0.0, "coverage_type": "full"},
        ]},
    ]
    out = signals.aggregate_completeness(inputs)
    pop = out["per_field_population"]
    # Shared field: appears in both benchmarks, mean = 0.5
    assert pop["a.shared"]["benchmark_count"] == 2
    assert abs(pop["a.shared"]["mean_score"] - 0.5) < 1e-9
    # only_b1 appears in 1 benchmark; benchmark_count=1
    assert pop["a.only_b1"]["benchmark_count"] == 1


# --- aggregate_provenance ------------------------------------


def test_aggregate_provenance_empty_corpus():
    out = signals.aggregate_provenance([], [])
    assert out["total_triples"] == 0
    assert out["total_groups"] == 0
    assert out["multi_source_rate"] is None
    assert out["first_party_only_rate"] is None
    # All 4 source_type buckets always present
    assert set(out["source_type_distribution"].keys()) == {
        "first_party", "third_party", "collaborative", "unspecified"
    }
    assert all(v == 0 for v in out["source_type_distribution"].values())


def test_aggregate_provenance_distribution_and_rates():
    per_row = [
        {"source_type": "first_party"},
        {"source_type": "third_party"},
        {"source_type": "third_party"},
        {"source_type": "collaborative"},
    ]
    per_group = [
        {"is_multi_source": True, "first_party_only_in_group": False},
        {"is_multi_source": False, "first_party_only_in_group": True},
        {"is_multi_source": False, "first_party_only_in_group": False},
    ]
    out = signals.aggregate_provenance(per_row, per_group)
    assert out["total_triples"] == 4
    assert out["total_groups"] == 3
    assert out["multi_source_groups"] == 1
    assert abs(out["multi_source_rate"] - 1/3) < 1e-9
    assert out["first_party_only_groups"] == 1
    assert abs(out["first_party_only_rate"] - 1/3) < 1e-9
    assert out["source_type_distribution"] == {
        "first_party": 1, "third_party": 2, "collaborative": 1, "unspecified": 0
    }


def test_aggregate_provenance_unknown_bucket_collapses_to_unspecified():
    """If somehow a row has a source_type not in the 4-bucket set, it lands
    in `unspecified` (defensive — shouldn't happen with current pipeline)."""
    per_row = [{"source_type": "weird_value"}]
    out = signals.aggregate_provenance(per_row, [])
    assert out["source_type_distribution"]["unspecified"] == 1


# --- aggregate_comparability ---------------------------------


def test_aggregate_comparability_empty_corpus():
    out = signals.aggregate_comparability([])
    assert out["total_groups"] == 0
    assert out["variant_eligible_groups"] == 0
    assert out["variant_divergent_groups"] == 0
    assert out["variant_divergence_rate"] is None
    assert out["cross_party_eligible_groups"] == 0
    assert out["cross_party_divergence_rate"] is None


def test_aggregate_comparability_eligibility_aware_denominator():
    """variant_divergence_rate must use eligible-only denominator —
    counting against `total_groups` would conflate ineligible (None signal)
    with eligible-and-passed."""
    per_group = [
        {"variant_divergence": {"has_variant_divergence": True}, "cross_party_divergence": None},
        {"variant_divergence": {"has_variant_divergence": False}, "cross_party_divergence": None},
        {"variant_divergence": None, "cross_party_divergence": None},  # ineligible
        {"variant_divergence": None, "cross_party_divergence": None},  # ineligible
    ]
    out = signals.aggregate_comparability(per_group)
    assert out["total_groups"] == 4
    assert out["variant_eligible_groups"] == 2
    assert out["variant_divergent_groups"] == 1
    # 1/2 = 0.5, NOT 1/4
    assert out["variant_divergence_rate"] == 0.5


def test_aggregate_comparability_cross_party_zero_eligible_yields_none_rate():
    """Mirrors current real-corpus state: 0 cross-party-eligible groups →
    rate is None, not 0.0 (which would mean 'tested 0% divergent')."""
    per_group = [
        {"variant_divergence": {"has_variant_divergence": True}, "cross_party_divergence": None},
        {"variant_divergence": {"has_variant_divergence": False}, "cross_party_divergence": None},
    ]
    out = signals.aggregate_comparability(per_group)
    assert out["cross_party_eligible_groups"] == 0
    assert out["cross_party_divergence_rate"] is None


# --- stratify orchestrators -------------------------------------------------


def _identity_aggregator(items: list) -> dict:
    """Trivial aggregator for orchestrator tests: sum-of-values plus count."""
    return {"count": len(items), "sum": sum(items)}


def test_stratify_overall_plus_by_category():
    items = [(1, "agentic"), (2, "agentic"), (3, "reasoning"), (4, None)]
    out = signals.stratify(items, _identity_aggregator)
    # Overall includes EVERY item (including category=None ones)
    assert out["overall"] == {"count": 4, "sum": 10}
    # by_category excludes None-category items
    assert out["by_category"]["agentic"] == {"count": 2, "sum": 3}
    assert out["by_category"]["reasoning"] == {"count": 1, "sum": 3}
    assert "None" not in out["by_category"]
    # Sorted keys for determinism
    assert list(out["by_category"].keys()) == ["agentic", "reasoning"]


def test_stratify_empty_input():
    out = signals.stratify([], _identity_aggregator)
    assert out["overall"] == {"count": 0, "sum": 0}
    assert out["by_category"] == {}


def test_stratify_provenance_pairs_rows_and_groups_per_category():
    """Both rows and groups are stratified; categories union; missing-side
    is treated as empty for that category in that input."""
    rows = [
        ({"source_type": "first_party"}, "agentic"),
        ({"source_type": "third_party"}, "reasoning"),
        ({"source_type": "third_party"}, "agentic"),
    ]
    groups = [
        ({"is_multi_source": True, "first_party_only_in_group": False}, "agentic"),
        ({"is_multi_source": False, "first_party_only_in_group": False}, "reasoning"),
    ]
    out = signals.stratify_provenance(rows, groups)
    # Overall over all rows + groups
    assert out["overall"]["total_triples"] == 3
    assert out["overall"]["total_groups"] == 2
    # by_category
    agentic = out["by_category"]["agentic"]
    assert agentic["total_triples"] == 2  # 2 rows tagged agentic
    assert agentic["total_groups"] == 1   # 1 group tagged agentic
    assert agentic["multi_source_groups"] == 1
    reasoning = out["by_category"]["reasoning"]
    assert reasoning["total_triples"] == 1
    assert reasoning["total_groups"] == 1


def test_stratify_provenance_handles_category_only_in_one_input():
    """If a category appears in rows but not groups (or vice versa), the
    other input is treated as empty for that category."""
    rows = [({"source_type": "first_party"}, "appears_only_in_rows")]
    groups = [({"is_multi_source": False, "first_party_only_in_group": False}, "appears_only_in_groups")]
    out = signals.stratify_provenance(rows, groups)
    # Both categories present in by_category
    assert "appears_only_in_rows" in out["by_category"]
    assert "appears_only_in_groups" in out["by_category"]
    # rows-only category: total_triples=1, total_groups=0
    rows_only = out["by_category"]["appears_only_in_rows"]
    assert rows_only["total_triples"] == 1
    assert rows_only["total_groups"] == 0
    # groups-only category: total_triples=0, total_groups=1
    groups_only = out["by_category"]["appears_only_in_groups"]
    assert groups_only["total_triples"] == 0
    assert groups_only["total_groups"] == 1

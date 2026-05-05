"""Unit tests for the four interpretive signals.

Pure-logic only — no DuckDB / no I/O. Covers reproducibility,
completeness, provenance, and comparability rules, including
return-None semantics and edge cases.
"""
from __future__ import annotations

from eval_card_backend.signals.comparability import (
    aggregated_setup,
    compute_cross_party_divergence_py,
    compute_threshold,
    compute_variant_divergence_py,
    normalize_org_name,
)
from eval_card_backend.signals.completeness import compute_completeness_py
from eval_card_backend.signals.reproducibility import is_agentic_py
from eval_card_backend.signals.setup import (
    canonical_json,
    differing_setup_fields,
    fact_id_py,
    normalize_setup,
    variant_key_py,
)


# ---------- setup helpers ----------

def test_normalize_setup_collapses_float_repr():
    # `.8g` formatting absorbs cosmetic IEEE-float noise but preserves 8-sig-fig
    # differences. 0.7 and 0.70000000001 collapse; 0.7 and 0.7000001 don't.
    a = normalize_setup({"temperature": 0.7, "max_tokens": 100})
    b = normalize_setup({"temperature": 0.70000000001, "max_tokens": "100"})
    assert a == b
    assert a["temperature"] == 0.7
    assert a["max_tokens"] == 100


def test_normalize_setup_text_strip_only():
    out = normalize_setup({"prompt_template": " hello\r\nworld \r"})
    assert out["prompt_template"] == "hello\nworld"


def test_variant_key_stable_under_key_order():
    a = variant_key_py({"temperature": 0.7, "max_tokens": 100})
    b = variant_key_py({"max_tokens": 100, "temperature": 0.7})
    assert a == b


def test_variant_key_handles_none():
    assert variant_key_py(None) == variant_key_py({})


def test_fact_id_determinism():
    assert fact_id_py("eval_x", 0) == fact_id_py("eval_x", 0)
    assert fact_id_py("eval_x", 0) != fact_id_py("eval_x", 1)
    assert fact_id_py(None, 0) is None
    assert fact_id_py("", 0) is None
    # Either input missing → None. Earlier code defaulted result_idx=0
    # silently, conflating None and 0; that's a correctness hazard.
    assert fact_id_py("eval_x", None) is None
    assert len(fact_id_py("eval_x", 0)) == 16


def test_canonical_json_sorts_keys():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_json(None) is None


def test_differing_setup_fields_records_originals():
    out = differing_setup_fields([
        {"temperature": 0.0, "max_tokens": 100},
        {"temperature": 0.7, "max_tokens": 100},
    ])
    assert len(out) == 1
    assert out[0]["field"] == "temperature"
    # values is a JSON string for the STRUCT(field VARCHAR, "values" JSON)[]
    # struct shape. DuckDB requires this at Parquet write time.
    assert out[0]["values"] == "[0.0, 0.7]"


def test_differing_setup_fields_preserves_pre_normalisation_values():
    """Whitespace and float-repr noise survive in the recorded values list,
    even though they collapse during canonical-form dedup."""
    import json as _json

    # prompt_template: original whitespace / line endings preserved.
    out = differing_setup_fields([
        {"prompt_template": "  hello\r\n"},
        {"prompt_template": "world"},
    ])
    pt = next(d for d in out if d["field"] == "prompt_template")
    values = _json.loads(pt["values"])
    assert "  hello\r\n" in values   # not the normalised "hello"
    assert "world" in values

    # max_tokens: int / str distinct in raw, identical after normalisation
    # → only one canonical entry → not flagged as differing.
    out = differing_setup_fields([
        {"max_tokens": 100},
        {"max_tokens": "100"},
    ])
    mt = [d for d in out if d["field"] == "max_tokens"]
    assert mt == []


# ---------- reproducibility ----------

def test_repro_missing_fields_sql_non_agentic():
    """Verify the Stage E SQL list-construction produces the correct
    missing-field list for the non-agentic two-field rule (temperature +
    max_tokens). Same SQL pattern as in stage_e_per_row_signals."""
    import duckdb
    con = duckdb.connect()

    def missing(has_temp, has_max, has_plan, has_limits, agentic=False):
        return con.execute(
            f"""
            SELECT (CASE WHEN NOT {has_temp}      THEN ['temperature'] ELSE []::VARCHAR[] END)
                || (CASE WHEN NOT {has_max}       THEN ['max_tokens']  ELSE []::VARCHAR[] END)
                || (CASE WHEN {agentic} AND NOT {has_plan}   THEN ['eval_plan']   ELSE []::VARCHAR[] END)
                || (CASE WHEN {agentic} AND NOT {has_limits} THEN ['eval_limits'] ELSE []::VARCHAR[] END)
            """
        ).fetchone()[0]

    assert missing("TRUE",  "TRUE",  "FALSE", "FALSE") == []
    assert missing("FALSE", "TRUE",  "FALSE", "FALSE") == ["temperature"]
    assert missing("TRUE",  "FALSE", "FALSE", "FALSE") == ["max_tokens"]
    assert missing("FALSE", "FALSE", "FALSE", "FALSE") == ["temperature", "max_tokens"]
    # Agentic-required fields ignored when not agentic
    assert missing("TRUE",  "TRUE",  "FALSE", "FALSE", agentic=False) == []


def test_repro_missing_fields_sql_agentic():
    """Agentic adds eval_plan + eval_limits to the active required set."""
    import duckdb
    con = duckdb.connect()

    def missing(has_temp, has_max, has_plan, has_limits):
        return con.execute(
            f"""
            SELECT (CASE WHEN NOT {has_temp}      THEN ['temperature'] ELSE []::VARCHAR[] END)
                || (CASE WHEN NOT {has_max}       THEN ['max_tokens']  ELSE []::VARCHAR[] END)
                || (CASE WHEN TRUE AND NOT {has_plan}   THEN ['eval_plan']   ELSE []::VARCHAR[] END)
                || (CASE WHEN TRUE AND NOT {has_limits} THEN ['eval_limits'] ELSE []::VARCHAR[] END)
            """
        ).fetchone()[0]

    assert missing("TRUE", "TRUE", "TRUE", "TRUE") == []
    assert missing("TRUE", "TRUE", "FALSE", "FALSE") == ["eval_plan", "eval_limits"]
    assert missing("FALSE", "TRUE", "TRUE", "FALSE") == ["temperature", "eval_limits"]
    # Order matches required-fields ordering: base first, agentic after
    assert missing("FALSE", "FALSE", "FALSE", "FALSE") == [
        "temperature", "max_tokens", "eval_plan", "eval_limits"
    ]


def test_is_agentic_purpose_shape_counter_fires_on_non_dict():
    """Cards in the wild have been seen with `purpose_and_intended_users` as a
    list/string/scalar. Rule 1 has to skip them, but it must increment the
    shape counter so silent agentic mis-classification surfaces in the run
    summary instead of vanishing."""
    from eval_card_backend.signals.reproducibility import (
        _purpose_shape_counter,
        reset_purpose_shape_counter,
    )

    reset_purpose_shape_counter()
    is_agentic_py("mmlu", {"purpose_and_intended_users": ["a", "b"]}, None)
    is_agentic_py("mmlu", {"purpose_and_intended_users": "free text"}, None)
    is_agentic_py("mmlu", {"purpose_and_intended_users": 7}, None)
    # Properly-shaped purpose dicts must not increment the counter.
    is_agentic_py("mmlu", {"purpose_and_intended_users": {"tasks": []}}, None)
    is_agentic_py("mmlu", {"purpose_and_intended_users": None}, None)
    assert _purpose_shape_counter == {"list": 1, "str": 1, "int": 1}
    reset_purpose_shape_counter()


def test_is_agentic_three_rules():
    # Rule 1: card.tasks
    assert is_agentic_py(
        "mmlu",
        {"purpose_and_intended_users": {"tasks": ["agentic"]}},
        None,
    )
    # Rule 2: agentic_eval_config presence
    assert is_agentic_py(
        "mmlu", None, {"agentic_eval_config": {"k": 1}}
    )
    # Rule 3: name regex against canonical id
    assert is_agentic_py("swe-bench-verified", None, None)
    # No rule fires
    assert not is_agentic_py("mmlu", None, None)
    # Robust to JSON-string inputs (DuckDB UDF call shape)
    assert is_agentic_py(
        "mmlu", '{"purpose_and_intended_users": {"tasks": ["agentic"]}}', None
    )


# ---------- completeness ----------
#
# Completeness is per-fact-row. The UDF takes the benchmark card plus the
# row's three EEE source_metadata fields plus the two reserved evalcards
# fields.

def test_completeness_no_card_no_source_metadata():
    out = compute_completeness_py(None)
    assert out["total_fields_evaluated"] == 28
    assert out["populated_count"] == 0.0
    assert out["completeness_score"] == 0.0


def test_completeness_full_autobenchmarkcard_only():
    """Card filled, no EEE source_metadata, no reserved fields → 23/28."""
    card = {
        "benchmark_details": {"name": "Foo", "overview": "x", "data_type": "qa",
                               "domains": ["a"], "languages": ["en"],
                               "similar_benchmarks": ["s"], "resources": ["r"]},
        "purpose_and_intended_users": {"goal": "g", "audience": ["a"],
                                        "tasks": ["x"], "limitations": "l",
                                        "out_of_scope_uses": ["o"]},
        "methodology": {"methods": ["m"], "metrics": ["x"], "calculation": "c",
                         "interpretation": "i", "baseline_results": "b",
                         "validation": "v"},
        "data": {"source": "s", "size": "z", "format": "f", "annotation": "a"},
        "ethical_and_legal_considerations": {
            "privacy_and_anonymity": "p",
            "data_licensing": "d",
            "consent_procedures": "c",
            "compliance_with_regulations": "r",
        },
    }
    out = compute_completeness_py(card)
    # 22 full + 1 partial (full) + 0 EEE + 0 reserved = 23 / 28
    assert out["populated_count"] == 23
    assert abs(out["completeness_score"] - 23 / 28) < 1e-9


def test_completeness_full_card_plus_full_source_metadata():
    """All card + all 3 EEE source_metadata + 0 reserved = 26 / 28."""
    card = {
        "benchmark_details": {"name": "Foo", "overview": "x", "data_type": "qa",
                               "domains": ["a"], "languages": ["en"],
                               "similar_benchmarks": ["s"], "resources": ["r"]},
        "purpose_and_intended_users": {"goal": "g", "audience": ["a"],
                                        "tasks": ["x"], "limitations": "l",
                                        "out_of_scope_uses": ["o"]},
        "methodology": {"methods": ["m"], "metrics": ["x"], "calculation": "c",
                         "interpretation": "i", "baseline_results": "b",
                         "validation": "v"},
        "data": {"source": "s", "size": "z", "format": "f", "annotation": "a"},
        "ethical_and_legal_considerations": {
            "privacy_and_anonymity": "p",
            "data_licensing": "d",
            "consent_procedures": "c",
            "compliance_with_regulations": "r",
        },
    }
    out = compute_completeness_py(
        card,
        source_type="evaluation_run",
        source_organization_name="OpenAI",
        evaluator_relationship="first_party",
    )
    assert out["populated_count"] == 26
    assert abs(out["completeness_score"] - 26 / 28) < 1e-9
    assert "eee_eval.source_metadata.source_type" not in out["missing_required_fields"]
    assert "evalcards.lifecycle_status" in out["missing_required_fields"]


def test_completeness_no_card_with_source_metadata():
    """No card + all 3 EEE source_metadata = 3/28 (just the EEE fields)."""
    out = compute_completeness_py(
        None,
        source_type="evaluation_run",
        source_organization_name="OpenAI",
        evaluator_relationship="first_party",
    )
    assert out["populated_count"] == 3
    assert abs(out["completeness_score"] - 3 / 28) < 1e-9


def test_completeness_reserved_fields_count():
    """Reserved fields populated → score includes them."""
    out = compute_completeness_py(
        None,
        lifecycle_status="stable",
        preregistration_url="https://example.com/preregistered",
    )
    assert out["populated_count"] == 2
    assert "evalcards.lifecycle_status" not in out["missing_required_fields"]
    assert "evalcards.preregistration_url" not in out["missing_required_fields"]


def test_completeness_partial_data_section():
    card = {"data": {"source": "s", "size": "z"}}  # 2 of 4 sub-items
    out = compute_completeness_py(card)
    partials = [p for p in out["partial_fields"] if p["field_path"].endswith(".data")]
    assert len(partials) == 1
    assert partials[0]["score"] == 0.5
    assert partials[0]["populated_subitems"] == 2
    assert partials[0]["total_subitems"] == 4


def test_completeness_no_field_scores_in_output():
    """field_scores is no longer in the per-row return shape — recoverable but
    not carried per-row (would denormalise 28 entries onto every fact row)."""
    out = compute_completeness_py(None)
    assert "field_scores" not in out


# ---------- comparability ----------

def test_normalize_org_name():
    assert normalize_org_name("Foo  Inc") == "foo inc"
    assert normalize_org_name(None) is None
    assert normalize_org_name("  ") is None


def test_compute_threshold_four_rules():
    assert compute_threshold({"metric_unit": "proportion"}) == (
        0.05, "proportion"
    )
    assert compute_threshold({"metric_unit": "percent"}) == (5.0, "percent")
    t, b = compute_threshold({"min_score": 0, "max_score": 100})
    assert t == 5.0 and b == "range_5pct"
    assert compute_threshold({}) == (0.05, "fallback_default")
    assert compute_threshold(None) == (0.05, "fallback_default")


def test_aggregated_setup_lower_median():
    rows = [
        {"score": 0.5, "evaluation_id": "a", "generation_args": {"x": 1}},
        {"score": 0.7, "evaluation_id": "b", "generation_args": {"x": 2}},
        {"score": 0.9, "evaluation_id": "c", "generation_args": {"x": 3}},
    ]
    # n=3 → index 1 → x=2
    assert aggregated_setup(rows) == {"x": 2}
    # n=2 → index 0 (lower of two)
    assert aggregated_setup(rows[:2]) == {"x": 1}


def test_variant_divergence_returns_none_when_inapplicable():
    # < 2 rows
    assert compute_variant_divergence_py([], {}) is None
    # all identical setups
    rows = [
        {"fact_id": str(i), "evaluation_id": str(i), "score": 0.1 * i,
         "generation_args": {"temperature": 0.0},
         "evaluator_relationship": "first_party",
         "source_organization_name": "X"}
        for i in range(3)
    ]
    assert compute_variant_divergence_py(rows, {}) is None
    # < 2 scored rows
    rows[0]["score"] = None
    rows[1]["score"] = None
    rows[2]["generation_args"] = {"temperature": 0.7}
    assert compute_variant_divergence_py(rows, {}) is None


def test_variant_divergence_positive():
    rows = [
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {"temperature": 0.0, "max_tokens": 100},
         "evaluator_relationship": "first_party", "source_organization_name": "X"},
        {"fact_id": "2", "evaluation_id": "b", "score": 0.7,
         "generation_args": {"temperature": 0.7, "max_tokens": 100},
         "evaluator_relationship": "first_party", "source_organization_name": "X"},
    ]
    out = compute_variant_divergence_py(rows, {"metric_unit": "proportion"})
    assert out["has_variant_divergence"] is True
    assert out["threshold_basis"] == "proportion"
    fields = {f["field"] for f in out["differing_setup_fields"]}
    assert fields == {"temperature"}


def test_cross_party_divergence_returns_none_below_min_orgs():
    rows = [
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {}, "evaluator_relationship": "first_party",
         "source_organization_name": "Foo"},
    ]
    assert compute_cross_party_divergence_py(rows, {}) is None


def test_cross_party_divergence_normalised_orgs():
    rows = [
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {}, "evaluator_relationship": "first_party",
         "source_organization_name": "Foo Inc"},
        {"fact_id": "2", "evaluation_id": "b", "score": 0.7,
         "generation_args": {}, "evaluator_relationship": "third_party",
         "source_organization_name": "  foo  inc"},
    ]
    # Same normalised org → < 2 distinct → None
    assert compute_cross_party_divergence_py(rows, {}) is None


def test_cross_party_divergence_positive():
    rows = [
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {}, "evaluator_relationship": "first_party",
         "source_organization_name": "Foo"},
        {"fact_id": "2", "evaluation_id": "b", "score": 0.7,
         "generation_args": {}, "evaluator_relationship": "third_party",
         "source_organization_name": "Bar"},
    ]
    out = compute_cross_party_divergence_py(rows, {"metric_unit": "proportion"})
    assert out["has_cross_party_divergence"] is True
    assert out["organization_count"] == 2
    # Display-name keys preserved
    assert set(out["scores_by_organization"]) == {"Foo", "Bar"}

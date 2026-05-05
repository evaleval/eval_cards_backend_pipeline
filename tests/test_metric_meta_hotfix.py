"""Unit tests for the metric_meta_hotfix layered chain.

Each test exercises one layer of the chain (registry / EEE per-record /
heuristic / default) plus the synonym normalisation. The provenance counter
is reset between tests so per-test attribution is verifiable.
"""
from __future__ import annotations

import pytest

from eval_card_backend.metric_meta_hotfix import (
    derive_metric_meta,
    reset_provenance_counter,
    _provenance_counter,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_provenance_counter()
    yield
    reset_provenance_counter()


# ---------- registry layer wins ----------

def test_registry_metric_kind_wins_over_eee():
    out = derive_metric_meta(
        eee_metric_config={"metric_kind": "score", "metric_name": "Accuracy"},
        registry_metric_kind="accuracy",   # registry says authoritative
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name="Accuracy",
    )
    assert out["metric_kind"] == "accuracy"
    assert _provenance_counter[("metric_kind", "registry")] == 1


def test_registry_metric_unit_wins_over_eee():
    out = derive_metric_meta(
        eee_metric_config={"metric_unit": "percent"},
        registry_metric_kind=None,
        registry_metric_unit="proportion",
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_unit"] == "proportion"
    assert _provenance_counter[("metric_unit", "registry")] == 1


# ---------- EEE per-record fills when registry is silent ----------

def test_eee_metric_kind_used_when_registry_null():
    out = derive_metric_meta(
        eee_metric_config={"metric_kind": "accuracy"},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name="ARC score",
    )
    assert out["metric_kind"] == "accuracy"
    assert _provenance_counter[("metric_kind", "eee_record")] == 1


def test_eee_min_max_used_when_registry_null():
    out = derive_metric_meta(
        eee_metric_config={"min_score": 0.0, "max_score": 100.0},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["min_score"] == 0.0
    assert out["max_score"] == 100.0
    assert _provenance_counter[("min_score", "eee_record")] == 1
    assert _provenance_counter[("max_score", "eee_record")] == 1


# ---------- proportion-shape heuristic ----------

def test_heuristic_proportion_when_zero_to_one_continuous_eee():
    out = derive_metric_meta(
        eee_metric_config={"min_score": 0, "max_score": 1, "score_type": "continuous"},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_unit"] == "proportion"
    assert _provenance_counter[("metric_unit", "heuristic_proportion_shape")] == 1


def test_heuristic_proportion_when_score_type_only_in_registry():
    """Registry knows score_type but EEE doesn't — heuristic should still fire."""
    out = derive_metric_meta(
        eee_metric_config={},   # no score_type
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=0.0,
        registry_max_score=1.0,
        registry_lower_is_better=False,
        metric_name=None,
        registry_score_type="continuous",
    )
    assert out["metric_unit"] == "proportion"


def test_heuristic_does_not_fire_for_non_zero_one_range():
    """[0, 100] continuous should not be inferred as proportion."""
    out = derive_metric_meta(
        eee_metric_config={"min_score": 0, "max_score": 100, "score_type": "continuous"},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_unit"] is None
    assert _provenance_counter[("metric_unit", "default_null")] == 1


def test_heuristic_does_not_fire_for_binary_score_type():
    """Binary != proportion (per design discussion). Stay NULL."""
    out = derive_metric_meta(
        eee_metric_config={"min_score": 0, "max_score": 1, "score_type": "binary"},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_unit"] is None


# ---------- name regex heuristic for metric_kind ----------

def test_name_regex_pass_at_k():
    out = derive_metric_meta(
        eee_metric_config={}, registry_metric_kind=None, registry_metric_unit=None,
        registry_min_score=None, registry_max_score=None, registry_lower_is_better=None,
        metric_name="Pass@1",
    )
    assert out["metric_kind"] == "pass_rate"


def test_name_regex_em_acronym():
    out = derive_metric_meta(
        eee_metric_config={}, registry_metric_kind=None, registry_metric_unit=None,
        registry_min_score=None, registry_max_score=None, registry_lower_is_better=None,
        metric_name="EM",
    )
    assert out["metric_kind"] == "exact_match"


def test_name_regex_f1():
    out = derive_metric_meta(
        eee_metric_config={}, registry_metric_kind=None, registry_metric_unit=None,
        registry_min_score=None, registry_max_score=None, registry_lower_is_better=None,
        metric_name="F1",
    )
    assert out["metric_kind"] == "f1"


def test_name_regex_refusal_rate():
    out = derive_metric_meta(
        eee_metric_config={}, registry_metric_kind=None, registry_metric_unit=None,
        registry_min_score=None, registry_max_score=None, registry_lower_is_better=None,
        metric_name="Refusal Rate",
    )
    assert out["metric_kind"] == "refusal_rate"


def test_name_regex_accuracy_variants():
    for name in ["Accuracy", "Acc", "Overall accuracy", "BBQ accuracy", "IFEval Strict Acc"]:
        out = derive_metric_meta(
            eee_metric_config={}, registry_metric_kind=None, registry_metric_unit=None,
            registry_min_score=None, registry_max_score=None, registry_lower_is_better=None,
            metric_name=name,
        )
        assert out["metric_kind"] == "accuracy", f"name: {name!r}"


def test_name_regex_cost():
    out = derive_metric_meta(
        eee_metric_config={}, registry_metric_kind=None, registry_metric_unit=None,
        registry_min_score=None, registry_max_score=None, registry_lower_is_better=None,
        metric_name="Total cost",
    )
    assert out["metric_kind"] == "cost"


def test_name_regex_no_match_falls_through_to_score_default():
    out = derive_metric_meta(
        eee_metric_config={}, registry_metric_kind=None, registry_metric_unit=None,
        registry_min_score=None, registry_max_score=None, registry_lower_is_better=None,
        metric_name="LM Evaluated Safety",
    )
    assert out["metric_kind"] == "score"
    assert _provenance_counter[("metric_kind", "heuristic_default")] == 1


def test_metric_kind_default_when_name_is_null():
    out = derive_metric_meta(
        eee_metric_config={}, registry_metric_kind=None, registry_metric_unit=None,
        registry_min_score=None, registry_max_score=None, registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_kind"] == "score"


# ---------- synonym normalisation ----------

def test_metric_unit_percentage_normalises_to_percent():
    out = derive_metric_meta(
        eee_metric_config={"metric_unit": "percentage"},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_unit"] == "percent"


def test_metric_unit_unknown_passes_through():
    """Non-synonym values should pass through unchanged."""
    out = derive_metric_meta(
        eee_metric_config={"metric_unit": "tokens_per_second"},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_unit"] == "tokens_per_second"


# ---------- lower_is_better default ----------

def test_lower_is_better_defaults_to_false():
    out = derive_metric_meta(
        eee_metric_config={}, registry_metric_kind=None, registry_metric_unit=None,
        registry_min_score=None, registry_max_score=None, registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["lower_is_better"] is False
    assert _provenance_counter[("lower_is_better", "heuristic_default")] == 1


def test_lower_is_better_eee_used_when_registry_null():
    out = derive_metric_meta(
        eee_metric_config={"lower_is_better": True},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["lower_is_better"] is True


# ---------- inputs robustness ----------

def test_handles_non_dict_eee_metric_config():
    """UDF wrapper is supposed to coerce JSON, but the function should be
    robust to receiving None or a non-dict directly."""
    out = derive_metric_meta(
        eee_metric_config=None,
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    # All heuristic / default fallbacks should produce a coherent dict.
    assert out["metric_kind"] == "score"
    assert out["metric_unit"] is None
    assert out["lower_is_better"] is False


# ---------- per-row provenance columns ----------

def test_provenance_registry_for_kind_and_unit():
    out = derive_metric_meta(
        eee_metric_config={"metric_kind": "score", "metric_unit": "percent"},
        registry_metric_kind="accuracy",
        registry_metric_unit="proportion",
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_kind_provenance"] == "registry"
    assert out["metric_unit_provenance"] == "registry"


def test_provenance_eee_record_when_registry_null():
    out = derive_metric_meta(
        eee_metric_config={"metric_kind": "f1", "metric_unit": "proportion"},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_kind_provenance"] == "eee_record"
    assert out["metric_unit_provenance"] == "eee_record"


def test_provenance_heuristic_regex_for_kind():
    """Regex chain matched 'accuracy' in metric_name; provenance reflects it."""
    out = derive_metric_meta(
        eee_metric_config={},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name="ARC accuracy@1",
    )
    assert out["metric_kind"] == "accuracy"
    assert out["metric_kind_provenance"] == "heuristic_regex"


def test_provenance_heuristic_default_for_kind_catchall():
    """No regex match → 'score' default. Lets consumers filter the
    catchall slice with `metric_kind = 'score' AND
    metric_kind_provenance = 'heuristic_default'`."""
    out = derive_metric_meta(
        eee_metric_config={},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name="some unmatched name",
    )
    assert out["metric_kind"] == "score"
    assert out["metric_kind_provenance"] == "heuristic_default"


def test_provenance_proportion_shape_for_unit():
    out = derive_metric_meta(
        eee_metric_config={"min_score": 0, "max_score": 1, "score_type": "continuous"},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_unit"] == "proportion"
    assert out["metric_unit_provenance"] == "heuristic_proportion_shape"


def test_provenance_default_null_for_unit():
    out = derive_metric_meta(
        eee_metric_config={},
        registry_metric_kind=None,
        registry_metric_unit=None,
        registry_min_score=None,
        registry_max_score=None,
        registry_lower_is_better=None,
        metric_name=None,
    )
    assert out["metric_unit"] is None
    assert out["metric_unit_provenance"] == "default_null"

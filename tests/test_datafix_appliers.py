"""Applier behaviour for every datafix fix.

The doctrine every applier follows is what these tests pin down: fill a
missing value, never touch one that is present, never replace a container
that is not a dict, leave the caller's record alone, and count everything
that was seen but not filled.

Tables here are hand-built so the behaviour is tested independently of
what the vendored evidence happens to say today; the vendored tables'
own shape and counts are covered in `test_datafix_tables.py`.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import pyarrow as pa
import pytest

from eval_card_backend.datafix import registry
from eval_card_backend.datafix.arc_agi import ArcAgiLoggedRequestArgs
from eval_card_backend.datafix.core import (
    GENERATION_ARGS_POINTER,
    DatafixSummary,
    PatchEvent,
    apply_fixes,
    generation_arg,
    set_generation_arg,
)
from eval_card_backend.datafix.hfopenllm_v2 import HfOpenLlmHarnessConfigs
from eval_card_backend.datafix.inspect_plan_config import InspectPlanConfig
from eval_card_backend.datafix.vals_ai import ValsAiMaxTokens
from eval_card_backend.schemas.eee_arrow import (
    derive_pyarrow_schema,
    pad_record_for_cast,
)
from eval_card_backend.schemas.eee_types import EvaluationLog

# --- fixtures ---------------------------------------------------------------


def _record(config_slug: str | None, results: list[dict], *, model_id="test/model",
            evaluation_id="ev_1") -> dict:
    details = {"benchmark_slug": config_slug} if config_slug else {}
    return {
        "schema_version": "0.2.2",
        "evaluation_id": evaluation_id,
        "retrieved_timestamp": "2026-09-01T00:00:00Z",
        "source_metadata": {
            "source_type": "documentation",
            "source_organization_name": "test-org",
            "evaluator_relationship": "third_party",
            "additional_details": details,
        },
        "model_info": {"name": "model", "id": model_id},
        "eval_library": {"name": "test-lib", "version": "1.0"},
        "evaluation_results": results,
    }


def _result(name: str, *, result_id: str | None = None, generation_config=...) -> dict:
    out = {
        "evaluation_name": name,
        "source_data": {"dataset_name": "test", "source_type": "other"},
        "metric_config": {"lower_is_better": False},
        "score_details": {"score": 0.5},
    }
    if result_id is not None:
        out["evaluation_result_id"] = result_id
    if generation_config is not ...:
        out["generation_config"] = generation_config
    return out


VALS_DOCUMENTED = {"max_tokens": 30000, "temperature": 1.0}


def _vals_fix(patches, documented=("max_tokens",)) -> ValsAiMaxTokens:
    return ValsAiMaxTokens({
        "fix_id": "vals_ai_max_tokens",
        "source": {"dataset": "evaleval/EEE_datastore", "revision": "deadbeef"},
        "documented_statements": {
            "sage": {
                field: {
                    "value": VALS_DOCUMENTED[field],
                    "url": "https://example.invalid/sage",
                    "quote": "temperature 1, and produce at most 30K tokens",
                    "verified_on": "2026-09-21",
                    "page_sha256": "1" * 64,
                }
                for field in documented
            },
        },
        "patches": list(patches),
        "_table_sha256": "0" * 64,
    })


def _patch(result_id: str, *, evaluation_id="ev_1", field="max_tokens",
           tier="same_record", expected_old=None):
    return {
        "evaluation_id": evaluation_id,
        "evaluation_result_id": result_id,
        "field": field,
        "expected_old": expected_old,
        "new": VALS_DOCUMENTED[field],
        "tier": tier,
    }


GENERATIVE = {
    "output_type": "generate_until",
    "temperature": 0.0,
    "max_tokens": 1280,
    "num_fewshot": 0,
    "do_sample": False,
}
MATH_DEFAULT = {**GENERATIVE, "max_tokens": 1024, "num_fewshot": 4}


def _hfo_fix(*, verified=None, overrides=None, conflicts=None) -> HfOpenLlmHarnessConfigs:
    return HfOpenLlmHarnessConfigs({
        "fix_id": "hfopenllm_v2_harness_configs",
        "source": {"dataset": "open-llm-leaderboard/results", "revision": "cafe"},
        "defaults": {
            "IFEval": GENERATIVE,
            "MATH Level 5": MATH_DEFAULT,
            "BBH": {"output_type": "multiple_choice", "num_fewshot": 3},
        },
        "overrides": overrides or {},
        "conflicts": conflicts or {},
        "verified": verified or {
            "IFEval": ["Org/Model"],
            "MATH Level 5": ["Org/Model"],
            "BBH": ["Org/Model"],
        },
        "_table_sha256": "0" * 64,
    })


def _args(patched, idx):
    return patched["evaluation_results"][idx]["generation_config"]["generation_args"]


def _count(counters, fix_id, name):
    return counters[f"{fix_id}:{name}"]


def _stripped(counters) -> Counter:
    """Counters as `summarise` receives them, i.e. without the fix-id prefix
    the process summary uses to attribute them."""
    return Counter({key.split(":", 1)[1]: value for key, value in counters.items()})


# --- vals_ai_max_tokens -----------------------------------------------------


def test_fills_a_missing_cap_and_leaves_a_stated_one_alone():
    record = _record("sage", [
        _result("a", result_id="r0"),
        _result("b", result_id="r1", generation_config={
            "generation_args": {"max_tokens": 128000}
        }),
    ])
    fix = _vals_fix([_patch("r0"), _patch("r1")])
    patched, events, counters = fix.apply(record)

    assert [e.pointer for e in events] == [
        "/evaluation_results/0/generation_config/generation_args/max_tokens"
    ]
    assert _args(patched, 0)["max_tokens"] == 30000
    assert _args(patched, 1)["max_tokens"] == 128000
    assert _count(counters, "vals_ai_max_tokens", "stale") == 1


@pytest.mark.parametrize("generation_config", [
    ...,                                    # key absent
    None,                                   # explicit null
    {},                                     # no generation_args
    {"generation_args": None},              # null generation_args
    {"generation_args": {}},                # empty generation_args
    {"generation_args": {"temperature": 1.0}},   # other args present
])
def test_containers_are_created_as_needed(generation_config):
    record = _record("sage", [
        _result("a", result_id="r0", generation_config=generation_config)
    ])
    patched, events, _ = _vals_fix([_patch("r0")]).apply(record)
    assert len(events) == 1
    assert _args(patched, 0)["max_tokens"] == 30000


@pytest.mark.parametrize("generation_config", [
    "not-a-dict",
    {"generation_args": "not-a-dict"},
])
def test_a_non_dict_container_is_never_replaced(generation_config):
    record = _record("sage", [
        _result("a", result_id="r0", generation_config=generation_config)
    ])
    patched, events, counters = _vals_fix([_patch("r0")]).apply(record)

    assert patched is None and events == []
    assert _count(counters, "vals_ai_max_tokens", "skipped_malformed") == 1


def test_a_malformed_container_is_counted_once_however_many_fields_are_documented():
    """The containers are shared, so the result is skipped as a whole."""
    record = _record("sage", [
        _result("a", result_id="r0", generation_config={"generation_args": 7})
    ])
    fix = _vals_fix(
        [_patch("r0"), _patch("r0", field="temperature")],
        documented=("max_tokens", "temperature"),
    )
    patched, events, counters = fix.apply(record)

    assert patched is None and events == []
    assert _count(counters, "vals_ai_max_tokens", "skipped_malformed") == 1


def test_a_stale_patch_entry_is_a_no_op():
    """The value was missing when the table was built and is not any more."""
    record = _record("sage", [
        _result("a", result_id="r0", generation_config={
            "generation_args": {"max_tokens": 96000}
        })
    ])
    patched, events, counters = _vals_fix([_patch("r0")]).apply(record)

    assert patched is None and events == []
    assert _count(counters, "vals_ai_max_tokens", "stale") == 1


def test_a_gap_with_no_patch_entry_is_counted_not_filled():
    record = _record("sage", [_result("a", result_id="r0")])
    patched, events, counters = _vals_fix([]).apply(record)

    assert patched is None and events == []
    assert _count(counters, "vals_ai_max_tokens", "unpatched_gap") == 1


def test_a_benchmark_with_no_documented_statement_is_untouched():
    record = _record("unlisted", [_result("a", result_id="r0")])
    patched, events, counters = _vals_fix([_patch("r0")]).apply(record)

    assert (patched, events, list(counters)) == (None, [], [])


def test_each_documented_field_is_filled_as_the_type_it_carries():
    record = _record("sage", [_result("a", result_id="r0")])
    fix = _vals_fix(
        [_patch("r0"), _patch("r0", field="temperature")],
        documented=("max_tokens", "temperature"),
    )
    patched, events, counters = fix.apply(record)

    args = _args(patched, 0)
    assert [e.field for e in events] == ["max_tokens", "temperature"]
    assert args["max_tokens"] == 30000 and isinstance(args["max_tokens"], int)
    assert args["temperature"] == 1.0 and isinstance(args["temperature"], float)
    assert fix.summarise(_stripped(counters))["by_field"] == {
        "max_tokens": 1, "temperature": 1
    }


def test_a_temperature_already_stated_is_left_alone():
    record = _record("sage", [
        _result("a", result_id="r0", generation_config={
            "generation_args": {"temperature": 0.0}
        })
    ])
    fix = _vals_fix(
        [_patch("r0"), _patch("r0", field="temperature")],
        documented=("max_tokens", "temperature"),
    )
    patched, events, counters = fix.apply(record)

    assert _args(patched, 0) == {"temperature": 0.0, "max_tokens": 30000}
    assert [e.field for e in events] == ["max_tokens"]
    assert _count(counters, "vals_ai_max_tokens", "stale") == 1


def test_the_two_fields_are_corroborated_independently():
    """A cap the table corroborates is filled even where the temperature
    beside it stayed uncorroborated, and the gap is still counted."""
    record = _record("sage", [_result("a", result_id="r0")])
    fix = _vals_fix([_patch("r0")], documented=("max_tokens", "temperature"))
    patched, events, counters = fix.apply(record)

    assert [e.field for e in events] == ["max_tokens"]
    assert _args(patched, 0) == {"max_tokens": 30000}
    assert _count(counters, "vals_ai_max_tokens", "unpatched_gap") == 1


def test_unused_patches_are_reported():
    fix = _vals_fix([_patch("r0"), _patch("r-absent")])
    record = _record("sage", [_result("a", result_id="r0")])
    _, _, counters = fix.apply(record)

    assert fix.summarise(_stripped(counters))["unused"] == 1


def test_the_input_record_is_never_mutated_and_the_fix_is_idempotent():
    record = _record("sage", [_result("a", result_id="r0")])
    before = copy.deepcopy(record)
    fix = _vals_fix([_patch("r0")])

    patched, events, _ = fix.apply(record)
    assert record == before
    assert len(events) == 1

    again, events_again, _ = fix.apply(patched)
    assert again is None and events_again == []


# --- hfopenllm_v2_harness_configs -------------------------------------------


def test_generative_results_get_temperature_and_cap():
    record = _record(None, [
        _result("IFEval"),
        _result("MATH Level 5"),
    ], model_id="Org/Model")
    patched, events, _ = _hfo_fix().apply(record)

    assert _args(patched, 0) == {"temperature": 0.0, "max_tokens": 1280}
    assert _args(patched, 1) == {"temperature": 0.0, "max_tokens": 1024}
    assert len(events) == 4


def test_multiple_choice_results_are_untouched():
    record = _record(None, [_result("BBH"), _result("MMLU-PRO")], model_id="Org/Model")
    patched, events, counters = _hfo_fix().apply(record)

    assert (patched, events, list(counters)) == (None, [], [])


def test_model_matching_is_case_sensitive():
    record = _record(None, [_result("IFEval")], model_id="org/model")
    patched, events, counters = _hfo_fix(
        verified={"IFEval": ["Org/Model"], "MATH Level 5": [], "BBH": []}
    ).apply(record)

    assert patched is None and events == []
    assert _count(counters, "hfopenllm_v2_harness_configs", "unmatched") == 1


def test_an_unverified_model_is_counted_as_unmatched():
    record = _record(None, [_result("IFEval")], model_id="Org/Unknown")
    _, events, counters = _hfo_fix().apply(record)

    assert events == []
    assert _count(counters, "hfopenllm_v2_harness_configs", "unmatched") == 1


def test_a_model_whose_dumps_disagree_is_counted_as_a_conflict():
    record = _record(None, [_result("IFEval")], model_id="Org/Disputed")
    _, events, counters = _hfo_fix(
        verified={"IFEval": [], "MATH Level 5": [], "BBH": []},
        conflicts={"Org/Disputed": ["IFEval"]},
    ).apply(record)

    assert events == []
    assert _count(counters, "hfopenllm_v2_harness_configs", "conflict") == 1
    assert _count(counters, "hfopenllm_v2_harness_configs", "unmatched") == 0


def test_a_model_specific_override_wins_over_the_default():
    record = _record(None, [_result("MATH Level 5")], model_id="Org/Model")
    patched, _, _ = _hfo_fix(
        overrides={"Org/Model": {"MATH Level 5": {**MATH_DEFAULT, "max_tokens": 512}}}
    ).apply(record)

    assert _args(patched, 0)["max_tokens"] == 512


def test_a_stated_value_is_kept_and_only_the_missing_one_filled():
    record = _record(None, [
        _result("IFEval", generation_config={"generation_args": {"temperature": 0.7}})
    ], model_id="Org/Model")
    patched, events, _ = _hfo_fix().apply(record)

    assert _args(patched, 0) == {"temperature": 0.7, "max_tokens": 1280}
    assert len(events) == 1


def test_harness_fix_does_not_mutate_its_input_and_is_idempotent():
    record = _record(None, [_result("IFEval")], model_id="Org/Model")
    before = copy.deepcopy(record)
    fix = _hfo_fix()

    patched, events, _ = fix.apply(record)
    assert record == before
    assert len(events) == 2

    again, events_again, _ = fix.apply(patched)
    assert again is None and events_again == []


# --- inspect_plan_config ----------------------------------------------------

PLAN_CONFIG = {"max_connections": "5", "max_tokens": "4096", "temperature": "0.0"}


def _step(solver: str, params=None, passed=None) -> str:
    return json.dumps({
        "solver": solver,
        "params": params if params is not None else {},
        "params_passed": passed if passed is not None else {},
    })


def _plan(*steps: str, config=None) -> dict:
    return {
        "name": "plan",
        "steps": list(steps) or [_step("generate")],
        "config": PLAN_CONFIG if config is None else config,
    }


def _inspect_record(plan=None, *, version="inspect_ai:0.3.75", library="inspect_ai",
                    generation_args=None) -> dict:
    args = dict(generation_args or {})
    args["eval_plan"] = _plan() if plan is None else plan
    record = _record(None, [_result("task", generation_config={"generation_args": args})])
    record["eval_library"] = {"name": library, "version": version}
    return record


def _plan_fix() -> InspectPlanConfig:
    return InspectPlanConfig()


def _skipped(counters) -> dict:
    return _plan_fix().summarise(_stripped(counters))["skipped"]


def test_the_plan_config_fills_both_fields():
    patched, events, counters = _plan_fix().apply(_inspect_record())

    assert _args(patched, 0)["temperature"] == 0.0
    assert _args(patched, 0)["max_tokens"] == 4096
    assert [e.evidence for e in events] == ["eval_plan.config", "eval_plan.config"]
    assert _plan_fix().summarise(_stripped(counters))["by_field"] == {
        "max_tokens": 1, "temperature": 1
    }


def test_a_stated_value_is_kept_and_only_the_missing_one_is_filled():
    record = _inspect_record(generation_args={"temperature": 0.7})
    patched, events, _ = _plan_fix().apply(record)

    assert _args(patched, 0)["temperature"] == 0.7
    assert len(events) == 1


@pytest.mark.parametrize("library, version, reason", [
    ("inspect_ai", "inspect_ai:0.3.999", "version_not_checked"),
    ("inspect_ai", "inspect_ai:0.3.206.dev19+g6beb95d9,inference_scaling_paper:0.1.0",
     "library_names_another_package"),
    ("lm-evaluation-harness", "0.4.0", "library_not_inspect_ai"),
])
def test_only_the_checked_library_versions_are_accepted(library, version, reason):
    record = _inspect_record(library=library, version=version)
    patched, events, counters = _plan_fix().apply(record)

    assert patched is None and events == []
    assert _skipped(counters) == {reason: 1}


@pytest.mark.parametrize("plan, reason", [
    (_plan(_step("someone/custom_scaffold"), _step("generate")), "unknown_solver"),
    (_plan(_step("generate"), _step("prompt_template")), "terminal_solver_not_checked"),
    (_plan(_step("react", {"model": "openai/gpt-4"})), "terminal_solver_names_a_model"),
    (_plan(_step("react", {}, {"model": "openai/gpt-4"})), "terminal_solver_names_a_model"),
    ({"name": "plan", "steps": [], "config": PLAN_CONFIG}, "plan_without_steps"),
    ({"name": "plan", "steps": ["{not json"], "config": PLAN_CONFIG}, "step_unparsed"),
    ({"name": "plan", "steps": [{"solver": "generate"}], "config": PLAN_CONFIG},
     "step_unparsed"),
])
def test_a_plan_outside_the_rule_is_refused(plan, reason):
    patched, events, counters = _plan_fix().apply(_inspect_record(plan))

    assert patched is None and events == []
    assert _skipped(counters) == {reason: 1}


@pytest.mark.parametrize("key", [
    "temperature", "max_tokens", "max_completion_tokens", "max_output_tokens",
])
@pytest.mark.parametrize("shape", ["params", "params_passed", "nested"])
def test_a_step_that_sets_its_own_generation_args_is_refused(key, shape):
    """Such a step would generate under its own settings, so the plan config
    is no longer the whole story."""
    step = {
        "params": _step("prompt_template", {key: 1}),
        "params_passed": _step("prompt_template", {}, {key: 1}),
        "nested": _step("prompt_template", {}, {"model_args": {"config": {key: 1}}}),
    }[shape]
    plan = _plan(step, _step("generate"))

    patched, events, counters = _plan_fix().apply(_inspect_record(plan))

    assert patched is None and events == []
    assert _skipped(counters) == {"step_sets_generation_args": 1}


@pytest.mark.parametrize("config, filled, reason", [
    ({"temperature": "-0.5"}, {}, "malformed_temperature"),
    ({"temperature": "warm"}, {}, "malformed_temperature"),
    ({"temperature": "true"}, {}, "malformed_temperature"),
    ({"temperature": "NaN"}, {}, "malformed_temperature"),
    ({"temperature": "Infinity"}, {}, "malformed_temperature"),
    ({"max_tokens": "0"}, {}, "malformed_max_tokens"),
    ({"max_tokens": "4096.5"}, {}, "malformed_max_tokens"),
    ({"max_tokens": "\"4096\""}, {}, "malformed_max_tokens"),
    ({"max_tokens": "4096", "temperature": "x"}, {"max_tokens": 4096},
     "malformed_temperature"),
])
def test_a_value_that_is_not_a_finite_number_is_counted_not_filled(config, filled, reason):
    record = _inspect_record(_plan(config=config))
    patched, _, counters = _plan_fix().apply(record)

    written = {} if patched is None else {
        key: value for key, value in _args(patched, 0).items() if key != "eval_plan"
    }
    assert written == filled
    assert _skipped(counters) == {reason: 1}


def test_a_plan_without_a_config_is_left_alone():
    record = _inspect_record(_plan(config={"max_connections": "5"}))
    patched, events, counters = _plan_fix().apply(record)

    assert (patched, events, list(counters)) == (None, [], [])


def test_a_record_without_a_plan_is_left_alone():
    record = _record(None, [_result("task")])
    record["eval_library"] = {"name": "inspect_ai", "version": "inspect_ai:0.3.75"}

    assert _plan_fix().apply(record) == (None, [], Counter())


@pytest.mark.parametrize("generation_config", ["not-a-dict", {"generation_args": "x"}])
def test_a_non_dict_container_carries_no_plan_to_read(generation_config):
    record = _record(None, [_result("task", generation_config=generation_config)])
    record["eval_library"] = {"name": "inspect_ai", "version": "inspect_ai:0.3.75"}

    assert _plan_fix().apply(record) == (None, [], Counter())


def test_the_plan_fix_does_not_mutate_its_input_and_is_idempotent():
    record = _inspect_record()
    before = copy.deepcopy(record)
    fix = _plan_fix()

    patched, events, _ = fix.apply(record)
    assert record == before
    assert len(events) == 2

    again, events_again, _ = fix.apply(patched)
    assert again is None and events_again == []


# --- arc_agi_logged_request_args --------------------------------------------


def _arc_result(name: str, metric_id: str, *, generation_config=...) -> dict:
    out = _result(name, result_id=f"{name}::{metric_id}",
                  generation_config=generation_config)
    out["metric_config"] = {"metric_id": metric_id, "lower_is_better": False}
    return out


def _arc_patch(name="v1_Public_Eval", *, fields=None, expected_old=None) -> dict:
    return {
        "evaluation_id": "ev_1",
        "evaluation_result_id": f"{name}::score",
        "evaluation_name": name,
        "expected_old": expected_old,
        "fields": fields if fields is not None else {"max_tokens": 128000},
        "cap_key": "max_output_tokens",
        "dataset": "arcprize/arc_agi_v1_public_eval",
        "revision": "d" * 40,
        "directory": "a-model",
    }


def _arc_fix(patches) -> ArcAgiLoggedRequestArgs:
    return ArcAgiLoggedRequestArgs({
        "fix_id": "arc_agi_logged_request_args",
        "source": {"revision": "d" * 40 + "+" + "e" * 40},
        "patches": list(patches),
        "_table_sha256": "0" * 64,
    })


def test_the_logged_cap_and_temperature_are_filled():
    record = _record(None, [_arc_result("v1_Public_Eval", "score")])
    patch = _arc_patch(fields={"max_tokens": 128000, "temperature": 0.7})

    patched, events, counters = _arc_fix([patch]).apply(record)

    assert _args(patched, 0) == {"temperature": 0.7, "max_tokens": 128000}
    assert [e.evidence for e in events] == ["a-model", "a-model"]
    summary = _arc_fix([patch]).summarise(_stripped(counters))
    assert summary["by_cap_key"] == {"max_output_tokens": 1}
    assert summary["by_result_name"] == {"v1_Public_Eval": 1}


def test_the_paired_cost_result_is_never_touched():
    record = _record(None, [
        _arc_result("v1_Public_Eval", "score"),
        _arc_result("v1_Public_Eval", "cost_per_task"),
    ])
    patches = [_arc_patch(), {**_arc_patch(), "evaluation_result_id":
                              "v1_Public_Eval::cost_per_task"}]

    patched, events, _ = _arc_fix(patches).apply(record)

    assert len(events) == 1
    assert patched["evaluation_results"][1].get("generation_config") is None


def test_results_outside_the_public_sets_are_never_touched():
    record = _record(None, [_arc_result("v1_Semi_Private", "score")])
    patches = [{**_arc_patch(), "evaluation_result_id": "v1_Semi_Private::score"}]

    patched, events, counters = _arc_fix(patches).apply(record)

    assert (patched, events, list(counters)) == (None, [], [])


def test_a_public_score_with_no_patch_is_counted_as_a_gap():
    record = _record(None, [_arc_result("v2_Public_Eval", "score")])

    patched, events, counters = _arc_fix([]).apply(record)

    assert patched is None and events == []
    assert _count(counters, "arc_agi_logged_request_args", "unpatched_gap") == 1


def test_an_already_stated_cap_is_kept_and_counted_as_stale():
    record = _record(None, [_arc_result(
        "v1_Public_Eval", "score",
        generation_config={"generation_args": {"max_tokens": 64000}},
    )])

    patched, events, counters = _arc_fix([_arc_patch()]).apply(record)

    assert patched is None and events == []
    assert _count(counters, "arc_agi_logged_request_args", "stale") == 1


def test_an_arc_patch_with_an_expected_old_value_is_refused():
    record = _record(None, [_arc_result("v1_Public_Eval", "score")])
    patch = _arc_patch(expected_old=64000)

    patched, events, counters = _arc_fix([patch]).apply(record)

    assert patched is None and events == []
    assert _count(counters, "arc_agi_logged_request_args", "stale") == 1


@pytest.mark.parametrize("generation_config", ["not-a-dict", {"generation_args": "x"}])
def test_an_arc_non_dict_container_is_never_replaced(generation_config):
    record = _record(None, [_arc_result(
        "v1_Public_Eval", "score", generation_config=generation_config
    )])

    patched, events, counters = _arc_fix([_arc_patch()]).apply(record)

    assert patched is None and events == []
    assert _count(counters, "arc_agi_logged_request_args", "skipped_malformed") == 1


def test_unused_arc_patches_are_reported():
    fix = _arc_fix([_arc_patch(), {**_arc_patch(), "evaluation_id": "ev_absent"}])
    record = _record(None, [_arc_result("v1_Public_Eval", "score")])
    _, _, counters = fix.apply(record)

    assert fix.summarise(_stripped(counters))["unused"] == 1


def test_the_arc_fix_does_not_mutate_its_input_and_is_idempotent():
    record = _record(None, [_arc_result("v1_Public_Eval", "score")])
    before = copy.deepcopy(record)
    fix = _arc_fix([_arc_patch()])

    patched, events, _ = fix.apply(record)
    assert record == before
    assert len(events) == 1

    again, events_again, _ = fix.apply(patched)
    assert again is None and events_again == []


# --- two fixes over one record ----------------------------------------------
#
# Every fix reads the record as it arrived, so a value one of them fills is
# never what the next one sees as already stated. Two of them landing on the
# same field are then two proposals about the same gap.


class _Proposes:
    """A fix that proposes one value for one field of the first result."""

    def __init__(self, fix_id: str, field: str, value) -> None:
        self.fix_id = fix_id
        self.source_config = None
        self._field = field
        self._value = value

    source_config_revision = "0" * 40
    table_sha256 = "0" * 64

    def apply(self, record):
        counters: Counter = Counter()
        result = record["evaluation_results"][0]
        if generation_arg(result, self._field)[0] != "missing":
            return None, [], counters
        patched = copy.deepcopy(record)
        set_generation_arg(patched["evaluation_results"][0], self._field, self._value)
        event = PatchEvent(
            fix_id=self.fix_id,
            evaluation_id=record.get("evaluation_id"),
            result_index=0,
            evaluation_result_id=result.get("evaluation_result_id"),
            pointer=GENERATION_ARGS_POINTER.format(idx=0, field=self._field),
            old=None,
            new=self._value,
            evidence="test",
        )
        counters[f"{self.fix_id}:applied"] += 1
        return patched, [event], counters

    def summarise(self, counters):
        return {"applied": counters["applied"]}


@pytest.fixture
def two_fixes(monkeypatch):
    """Register two fixes and let the caller say what each proposes."""
    def register(*fixes):
        entries = tuple(
            registry.Entry(fix.fix_id, None, (lambda f=fix: f), lambda: "0" * 64,
                           "0" * 40)
            for fix in fixes
        )
        monkeypatch.setattr(registry, "_REGISTRY", entries)
        monkeypatch.setattr(registry, "_instances", dict(enumerate(fixes)))

    return register


def _one_result_record() -> dict:
    return _record(None, [_result("a", result_id="r0")])


def test_two_fixes_agreeing_on_a_field_fill_it_once(two_fixes):
    two_fixes(_Proposes("first", "max_tokens", 4096),
              _Proposes("second", "max_tokens", 4096))
    record = _one_result_record()

    patched, events, counters = apply_fixes(record, "any-config")

    assert _args(patched, 0) == {"max_tokens": 4096}
    assert [(e.fix_id, e.concurring_fix_ids) for e in events] == [
        ("first", ("second",))
    ]
    assert counters["first:cross_fix_conflict"] == 0
    assert counters["second:cross_fix_conflict"] == 0


def test_two_fixes_disagreeing_on_a_field_fill_nothing(two_fixes):
    two_fixes(_Proposes("first", "max_tokens", 4096),
              _Proposes("second", "max_tokens", 8192))
    record = _one_result_record()

    patched, events, counters = apply_fixes(record, "any-config")

    assert patched is None and events == []
    assert counters["first:cross_fix_conflict"] == 1
    assert counters["second:cross_fix_conflict"] == 1


def test_a_contradiction_is_reported_per_fix(two_fixes):
    two_fixes(_Proposes("first", "max_tokens", 4096),
              _Proposes("second", "max_tokens", 8192))
    account = DatafixSummary()

    apply_fixes(_one_result_record(), "any-config", summary=account)

    fixes = account.as_dict(enabled=True)["fixes"]
    assert [block["cross_fix_conflict"] for block in fixes.values()] == [1, 1]
    assert [block["results_patched"] for block in fixes.values()] == [0, 0]


def test_two_fixes_on_different_fields_both_land(two_fixes):
    two_fixes(_Proposes("first", "max_tokens", 4096),
              _Proposes("second", "temperature", 0.5))

    patched, events, _ = apply_fixes(_one_result_record(), "any-config")

    assert _args(patched, 0) == {"max_tokens": 4096, "temperature": 0.5}
    assert [e.fix_id for e in events] == ["first", "second"]


def test_a_later_fix_does_not_read_an_earlier_fix_s_fill(two_fixes):
    """Sequentially the second fix would find the field stated and hold off,
    and the contradiction would never be seen."""
    two_fixes(_Proposes("first", "max_tokens", 4096),
              _Proposes("second", "max_tokens", 8192))
    record = _one_result_record()
    before = copy.deepcopy(record)

    apply_fixes(record, "any-config")

    assert record == before


def test_the_reconciled_apply_does_not_mutate_its_input_and_is_idempotent(two_fixes):
    two_fixes(_Proposes("first", "max_tokens", 4096),
              _Proposes("second", "max_tokens", 4096))
    record = _one_result_record()
    before = copy.deepcopy(record)

    patched, events, _ = apply_fixes(record, "any-config")
    assert record == before
    assert len(events) == 1

    again, events_again, _ = apply_fixes(patched, "any-config")
    assert again is None and events_again == []


# --- what the patched records have to survive -------------------------------


def test_patched_records_still_satisfy_the_upstream_contract():
    vals_patched, _, _ = _vals_fix([_patch("r0")]).apply(
        _record("sage", [_result("a", result_id="r0")])
    )
    hfo_patched, _, _ = _hfo_fix().apply(
        _record(None, [_result("IFEval"), _result("MATH Level 5")], model_id="Org/Model")
    )
    plan_patched, _, _ = _plan_fix().apply(_inspect_record())
    arc_patched, _, _ = _arc_fix([_arc_patch()]).apply(
        _record(None, [_arc_result("v1_Public_Eval", "score")])
    )
    for patched in (vals_patched, hfo_patched, plan_patched, arc_patched):
        EvaluationLog.model_validate(patched)


def test_a_mixed_batch_of_patched_records_casts_to_the_arrow_schema():
    """Records whose `generation_config` was absent, null, empty or already
    populated all have to land in one Arrow batch together."""
    schema = derive_pyarrow_schema()
    shapes = [
        ...,
        None,
        {},
        {"generation_args": {}},
        {"generation_args": {"max_tokens": 128000}},
    ]
    fix = _vals_fix([_patch("r0", evaluation_id=f"ev_{i}") for i in range(len(shapes))])
    rows = []
    for i, shape in enumerate(shapes):
        record = _record(
            "sage",
            [_result("a", result_id="r0", generation_config=shape)],
            evaluation_id=f"ev_{i}",
        )
        patched, _, _ = fix.apply(record)
        rows.append(pad_record_for_cast(patched or record, schema))

    table = pa.Table.from_pylist(rows, schema=schema)
    assert table.num_rows == len(shapes)

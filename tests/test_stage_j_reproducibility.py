"""The setup-item rule as `stage_j_reproducibility` applies it, one
`eval_results_view` row at a time.

Each case builds the columns the rule reads, at the types the view carries
them, runs the stage, and reads back the gap flag, the `reproducibility_gap`
annotation (item states included) and the disclosed items' evidence.
"""
from __future__ import annotations

import json

import duckdb
import pytest

from eval_card_backend.canonicalise import stages
from eval_card_backend.signals.reproducibility import SETUP_ITEMS

ITEM_IDS = [item for item, _, _ in SETUP_ITEMS]

_COLUMNS = f"""
    row_no INTEGER,
    generation_config STRUCT(
        num_few_shot INTEGER,
        generation_args STRUCT(
            temperature DOUBLE, top_p DOUBLE, top_k DOUBLE,
            max_tokens INTEGER, reasoning BOOLEAN
        ),
        additional_details VARCHAR,
        prompt_template VARCHAR
    ),
    protocol_condition VARCHAR,
    eval_library STRUCT(name VARCHAR, version VARCHAR, fork VARCHAR),
    score_details STRUCT(
        score DOUBLE, standard_error DOUBLE, standard_deviation DOUBLE,
        sample_size INTEGER,
        confidence_interval STRUCT(lower DOUBLE, upper DOUBLE, confidence_level DOUBLE)
    ),
    split VARCHAR,
    scoring_mode VARCHAR,
    has_reproducibility_gap BOOLEAN,
    evalcards_annotations STRUCT(
        reproducibility_gap {stages._REPRODUCIBILITY_GAP_STRUCT},
        comparability_status VARCHAR
    )
"""

_DEFAULT_ANNOTATIONS = {"reproducibility_gap": None, "comparability_status": "single_variant"}
_STATES = {"d": "disclosed", "m": "missing", "n": "not_applicable", "u": "unknown"}


def _struct(template: dict, values: dict | None) -> dict | None:
    if values is None:
        return None
    unknown = set(values) - set(template)
    assert not unknown, f"not a column member: {unknown}"
    return {key: values.get(key) for key in template}


def _json_text(value):
    return value if value is None or isinstance(value, str) else json.dumps(value)


def _row(
    *,
    generation_args: dict | None = None,
    details=None,
    protocol=None,
    eval_library: dict | None = None,
    score: dict | None = None,
    split: str | None = None,
    scoring_mode: str | None = None,
    annotations: dict | None = _DEFAULT_ANNOTATIONS,
    generation_config_null: bool = False,
) -> list:
    args = _struct(
        dict.fromkeys(("temperature", "top_p", "top_k", "max_tokens", "reasoning")),
        generation_args,
    )
    generation_config = None if generation_config_null else {
        "num_few_shot": None,
        "generation_args": args,
        "additional_details": _json_text(details),
        "prompt_template": None,
    }
    score_details = _struct(
        dict.fromkeys(("score", "standard_error", "standard_deviation",
                       "sample_size", "confidence_interval")),
        score,
    )
    return [
        generation_config,
        _json_text(protocol),
        _struct(dict.fromkeys(("name", "version", "fork")), eval_library),
        score_details,
        split,
        scoring_mode,
        annotations,
    ]


class Reading:
    def __init__(self, evidence, gap, annotations):
        self.evidence = evidence
        self.gap = gap
        self.annotations = annotations
        self.repro = annotations["reproducibility_gap"]
        found = {entry["item"]: entry for entry in evidence}
        self.items = [
            {"item": item, "state": _STATES[code],
             "field": found.get(item, {}).get("field"),
             "value": found.get(item, {}).get("value")}
            for item, code in zip(ITEM_IDS, self.repro["item_states"], strict=True)
        ]
        self.by_item = {item["item"]: item for item in self.items}

    def state(self, item: str) -> str:
        return self.by_item[item]["state"]

    def field(self, item: str) -> str | None:
        return self.by_item[item]["field"]

    def value(self, item: str) -> str | None:
        return self.by_item[item]["value"]

    @property
    def disclosed(self) -> int:
        return self.repro["populated_count"]

    @property
    def applicable(self) -> int:
        return self.repro["required_count"]


def _connection(*rows: list) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE TABLE eval_results_view ({_COLUMNS})")
    for n, row in enumerate(rows):
        con.execute(
            "INSERT INTO eval_results_view VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            [n, *row],
        )
    return con


def _score_all(*rows: list) -> list[Reading]:
    con = _connection(*rows)
    stages.stage_j_reproducibility(con)
    return [
        Reading(evidence, gap, annotations)
        for evidence, gap, annotations in con.execute(
            "SELECT reproducibility_evidence, has_reproducibility_gap, evalcards_annotations "
            "FROM eval_results_view ORDER BY row_no"
        ).fetchall()
    ]


def _score(**kwargs) -> Reading:
    return _score_all(_row(**kwargs))[0]


# ---------------------------------------------------------------------------
# Disclosure under another field name, and what each item accepts
# ---------------------------------------------------------------------------


def test_a_study_disclosing_everything_in_its_protocol_is_credited():
    r = _score(protocol={
        "compaction": False,
        "feedback": "none",
        "reasoning_effort": "high",
        "reasoning_tokens": 32000,
        "scaffold": "S-adaptive",
        "token_limit": 10000000,
    })
    assert r.state("output_length") == "disclosed"
    assert r.state("harness") == "disclosed"
    assert r.state("reasoning_budget") == "disclosed"
    # Nothing says this run could not set its sampling controls.
    assert r.state("decoding") == "missing"
    # Agentic (a scaffold) but silent on how many times each cell ran.
    assert r.state("attempts") == "missing"
    assert r.state("context_handling") == "disclosed"
    assert r.state("test_set") == "missing"
    assert r.state("harness_version") == "missing"
    assert (r.disclosed, r.applicable) == (4, 8)
    assert r.gap is True


def test_decoding_is_asked_when_nothing_rules_sampling_out():
    r = _score(generation_args={"max_tokens": 2048})
    assert r.state("decoding") == "missing"
    assert r.state("output_length") == "disclosed"


def test_temperature_zero_is_a_disclosure():
    r = _score(generation_args={"temperature": 0.0})
    assert r.state("decoding") == "disclosed"
    assert r.field("decoding") == "generation_args.temperature"
    assert float(r.value("decoding")) == 0.0


def test_a_harness_property_does_not_stand_for_the_harness():
    r = _score(protocol={"compaction": True})
    assert r.state("context_handling") == "disclosed"
    assert r.state("harness") == "missing"
    assert r.state("attempts") == "missing"


def test_a_library_name_alone_is_not_a_harness_version():
    r = _score(eval_library={"name": "lm-evaluation-harness"})
    assert r.state("harness") == "disclosed"
    assert r.field("harness") == "row.eval_library"
    assert r.state("harness_version") == "missing"


def test_a_task_config_is_the_harness_version():
    r = _score(details={"lm_eval_task": "leaderboard_bbh_snarks"})
    assert r.state("harness_version") == "disclosed"


def test_the_library_struct_answers_both_harness_items():
    r = _score(eval_library={"name": "lm-evaluation-harness", "version": "0.4.2"})
    assert r.state("harness") == "disclosed"
    assert r.state("harness_version") == "disclosed"
    assert r.field("harness_version") == "row.eval_library_version"
    assert r.value("harness_version") == "\"0.4.2\""


def test_the_scored_item_count_is_the_test_set():
    r = _score(score={"sample_size": 86})
    assert r.state("test_set") == "disclosed"
    assert r.field("test_set") == "score.sample_size"
    assert r.value("test_set") == "86"


def test_uncertainty_statistics_are_not_setup():
    r = _score(score={
        "standard_error": 0.036,
        "confidence_interval": {"lower": 0.1, "upper": 0.2, "confidence_level": None},
    })
    assert r.disclosed == 0


def test_a_repeat_count_is_attempts():
    r = _score(protocol={"scaffold": "ReAct", "epochs": 5})
    assert r.state("attempts") == "disclosed"
    assert r.field("attempts") == "protocol.epochs"
    assert r.value("attempts") == "5"


def test_the_field_that_disclosed_an_item_is_reported():
    r = _score(protocol={"token_limit": 10_000_000})
    assert r.field("output_length") == "protocol.token_limit"
    assert r.value("output_length") == "10000000"


# ---------------------------------------------------------------------------
# Unknown applicability
# ---------------------------------------------------------------------------


def test_an_item_whose_applicability_is_unknown_leaves_the_denominator():
    r = _score()
    assert r.state("reasoning_budget") == "unknown"
    assert r.state("attempts") == "unknown"
    assert r.state("context_handling") == "unknown"
    # Nothing says the run could not sample or generate.
    assert r.state("decoding") == "missing"
    assert r.state("output_length") == "missing"
    assert (r.disclosed, r.applicable) == (0, 5)
    assert r.gap is True


def test_an_affirmative_value_settles_its_own_applicability():
    r = _score(generation_args={"reasoning": True})
    assert r.state("reasoning_budget") == "disclosed"


def test_reasoning_false_answers_the_item_without_marking_a_reasoning_run():
    r = _score(generation_args={"reasoning": False})
    assert r.state("reasoning_budget") == "disclosed"
    assert r.state("decoding") == "missing"


def test_reasoning_effort_none_discloses_without_making_the_item_applicable():
    r = _score(protocol={"reasoning_effort": "none"})
    # Disclosed, so it counts as applicable and disclosed all the same.
    assert r.state("reasoning_budget") == "disclosed"
    assert (r.disclosed, r.applicable) == (1, 6)
    assert r.state("decoding") == "missing"


def test_a_negative_marker_does_not_open_the_items_that_depend_on_it():
    r = _score(protocol={"compaction": False})
    assert r.state("context_handling") == "disclosed"
    assert r.value("context_handling") == "false"
    assert r.state("attempts") == "unknown"


def test_a_reasoning_run_is_still_asked_for_decoding():
    r = _score(protocol={"reasoning_effort": "high"})
    assert r.state("reasoning_budget") == "disclosed"
    assert r.state("decoding") == "missing"


# ---------------------------------------------------------------------------
# Scoring mode
# ---------------------------------------------------------------------------


def test_a_log_prob_row_has_no_decoding_or_output_length():
    r = _score(details={"output_type": "multiple_choice"})
    assert r.state("decoding") == "not_applicable"
    assert r.state("output_length") == "not_applicable"
    assert [i["item"] for i in r.items if i["state"] == "not_applicable"] == [
        "decoding", "output_length",
    ]
    assert "decoding" not in r.repro["missing_fields"]
    assert r.applicable == 3


@pytest.mark.parametrize("output_type", ["loglikelihood", "loglikelihood_rolling"])
def test_every_log_likelihood_output_type_is_log_prob(output_type):
    assert _score(details={"output_type": output_type}).state("decoding") == "not_applicable"


def test_generate_until_is_generative():
    r = _score(details={"output_type": "generate_until"})
    assert r.state("decoding") == "missing"
    assert r.state("output_length") == "missing"


def test_the_mode_is_never_read_off_a_benchmark_or_library_name():
    r = _score(eval_library={"name": "helm_mmlu"})
    assert r.state("decoding") == "missing"


def test_the_view_scoring_mode_beats_a_contradicting_output_type():
    log_prob, generative = _score_all(
        _row(scoring_mode="log_prob", details={"output_type": "generate_until"}),
        _row(scoring_mode="generative", details={"output_type": "multiple_choice"}),
    )
    assert log_prob.state("decoding") == "not_applicable"
    assert log_prob.state("output_length") == "not_applicable"
    assert generative.state("decoding") == "missing"
    assert generative.state("output_length") == "missing"


@pytest.mark.parametrize("scoring_mode", ["  LOG_PROB ", "LOG_PROB", "log_prob ", "Log_Prob"])
def test_only_the_exact_log_prob_string_is_a_classification(scoring_mode):
    r = _score(scoring_mode=scoring_mode, details={"output_type": "generate_until"})
    assert r.state("decoding") == "missing"
    assert r.state("output_length") == "missing"


@pytest.mark.parametrize("scoring_mode", [" GENERATIVE ", "Generative"])
def test_only_the_exact_generative_string_is_a_classification(scoring_mode):
    r = _score(scoring_mode=scoring_mode, details={"output_type": "multiple_choice"})
    assert r.state("decoding") == "not_applicable"


@pytest.mark.parametrize("scoring_mode", [None, "", "unknown", "perplexity"])
def test_an_unusable_scoring_mode_falls_back_to_the_output_type(scoring_mode):
    r = _score(scoring_mode=scoring_mode, details={"output_type": "loglikelihood"})
    assert r.state("decoding") == "not_applicable"


def test_the_protocol_output_type_is_read_too():
    r = _score(protocol={"output_type": "multiple_choice"})
    assert r.state("decoding") == "not_applicable"


def test_a_row_saying_nothing_about_its_mode_is_asked_for_both():
    r = _score(scoring_mode=None)
    assert r.state("decoding") == "missing"
    assert r.state("output_length") == "missing"


def test_a_log_prob_scoring_mode_alone_retires_decoding():
    assert _score(scoring_mode="log_prob").state("decoding") == "not_applicable"


# ---------------------------------------------------------------------------
# Where the evidence is read from
# ---------------------------------------------------------------------------


def test_decoding_protocol_limits_and_plan_are_read_where_the_view_carries_them():
    full, agent_only = _score_all(
        _row(
            generation_args={"temperature": 0.7, "max_tokens": 4096},
            details={
                "eval_limits": {"message_limit": 30},
                "eval_plan": {"name": "react", "steps": [1, 2]},
            },
            protocol={"scaffold": "S-adaptive"},
        ),
        _row(details={
            "eval_limits": {"message_limit": 30},
            "eval_plan": {"name": "react", "steps": [1, 2]},
        }),
    )
    assert full.field("decoding") == "generation_args.temperature"
    assert full.value("decoding") == "0.7"
    assert full.field("output_length") == "generation_args.max_tokens"
    assert full.field("harness") == "protocol.scaffold"

    assert agent_only.field("output_length") == "limits.message_limit"
    assert agent_only.field("harness") == "agent.eval_plan"
    # A message limit marks the run agentic, so attempts is asked.
    assert agent_only.state("attempts") == "missing"


def test_eval_limits_stored_as_a_json_string_are_read():
    r = _score(details={"eval_limits": json.dumps({"max_tokens": 512})})
    assert r.field("output_length") == "limits.max_tokens"
    assert r.value("output_length") == "512"


@pytest.mark.parametrize("details, protocol", [
    ("not json", "not json"),
    ("[1, 2]", "42"),
    ("{\"eval_limits\": \"{broken\"}", "{"),
])
def test_malformed_free_text_json_reads_as_absent(details, protocol):
    r = _score(details=details, protocol=protocol)
    assert (r.disclosed, r.applicable) == (0, 5)


def test_a_null_generation_config_reads_as_absent():
    r = _score(generation_config_null=True, protocol="not json")
    assert (r.disclosed, r.applicable) == (0, 5)


@pytest.mark.parametrize("placeholder", ["unknown", " N/A ", "", "Not Specified", "-"])
def test_a_placeholder_string_is_not_a_disclosure(placeholder):
    r = _score(
        eval_library={"name": "inspect_ai", "version": placeholder},
        details={"harness_version": placeholder},
        protocol={"split": placeholder},
    )
    assert r.state("harness") == "disclosed"
    assert r.state("harness_version") == "missing"
    assert r.state("test_set") == "missing"


def test_an_empty_object_or_list_is_not_a_disclosure():
    r = _score(details={"eval_plan": {}, "agentic_eval_config": []})
    assert r.state("harness") == "missing"


# ---------------------------------------------------------------------------
# What the stage writes
# ---------------------------------------------------------------------------


def test_every_item_has_a_state_and_only_disclosed_items_have_evidence():
    r = _score(
        generation_args={"temperature": 0.0},
        eval_library={"name": "inspect_ai"},
    )
    assert len(r.repro["item_states"]) == len(ITEM_IDS)
    assert ITEM_IDS == [
        "decoding", "reasoning_budget", "output_length", "harness",
        "harness_version", "context_handling", "test_set", "attempts",
    ]
    assert [e["item"] for e in r.evidence] == ["decoding", "harness"]
    for entry in r.evidence:
        assert entry["field"] is not None and entry["value"] is not None


def test_the_annotation_carries_item_ids():
    r = _score(
        generation_args={"temperature": 0.0},
        eval_library={"name": "inspect_ai"},
        details={"output_type": "generate_until"},
    )
    assert r.repro == {
        "has_reproducibility_gap": True,
        "missing_fields": ["output_length", "harness_version", "test_set"],
        "populated_count": 2,
        "required_count": 5,
        "item_states": "dumdmumu",
    }
    assert r.gap is True
    # The annotation's other members are left as they were.
    assert r.annotations["comparability_status"] == "single_variant"


def test_a_row_disclosing_every_applicable_item_has_no_gap():
    r = _score(
        scoring_mode="log_prob",
        eval_library={"name": "lm-evaluation-harness", "version": "0.4.2"},
        score={"sample_size": 1000},
    )
    assert r.gap is False
    assert r.repro["missing_fields"] == []
    assert (r.disclosed, r.applicable) == (3, 3)


def test_a_null_annotation_gains_a_reading_and_nothing_else():
    r = _score(annotations=None)
    assert r.annotations["comparability_status"] is None
    assert r.repro["required_count"] == 5
    assert r.repro["missing_fields"] == [
        "decoding", "output_length", "harness", "harness_version", "test_set",
    ]



@pytest.mark.parametrize("space", ["\u00a0", "\u3000", "\ufeff", "\u2003"])
def test_unicode_space_around_a_placeholder_still_reads_as_a_placeholder(space):
    # The paper's scorer trimmed with JavaScript's trim, which strips these.
    r = _score(
        details={"temperature": space},
        protocol={"scaffold": f"N/A{space}"},
    )
    assert r.state("decoding") == "missing"
    assert r.state("harness") == "missing"
    assert r.state("attempts") == "unknown"


def test_evidence_values_are_json_text():
    r = _score(
        generation_args={"temperature": 0.7},
        eval_library={"name": "inspect_ai", "version": "2.0"},
    )
    assert json.loads(r.value("decoding")) == 0.7
    assert json.loads(r.value("harness_version")) == "2.0"

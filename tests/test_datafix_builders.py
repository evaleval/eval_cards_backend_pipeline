"""Builders: table regeneration from evidence, on tiny synthetic trees.

The vendored tables are derived artifacts, so what matters is that the
derivation rules behave as documented and that a rebuild on unchanged
evidence is byte-identical. Golden checksums here pin the exact bytes a
build produces; they move only when the rules or the recorded evidence
text deliberately change.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from eval_card_backend.datafix import builders
from eval_card_backend.datafix.core import canonical_json

MATH_SUBTASKS = (
    "algebra", "counting_and_prob", "geometry", "intermediate_algebra",
    "num_theory", "prealgebra", "precalculus",
)


def _sha(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# --- vals_ai_max_tokens -----------------------------------------------------


def _vals_record(
    evaluation_id: str, slug: str, model_id: str, caps, temperatures=None
) -> dict:
    temperatures = temperatures if temperatures is not None else [None] * len(caps)
    results = []
    for idx, (cap, temperature) in enumerate(zip(caps, temperatures)):
        result = {
            "evaluation_result_id": f"{evaluation_id}:r{idx}",
            "evaluation_name": f"{slug}.task{idx}",
            "source_data": {"dataset_name": slug, "source_type": "other"},
            "metric_config": {"lower_is_better": False},
            "score_details": {"score": 1.0},
        }
        stated = {
            field: value
            for field, value in (("max_tokens", cap), ("temperature", temperature))
            if value is not None
        }
        if stated:
            result["generation_config"] = {"generation_args": stated}
        results.append(result)
    return {
        "schema_version": "0.2.2",
        "evaluation_id": evaluation_id,
        "retrieved_timestamp": "2026-09-01T00:00:00Z",
        "source_metadata": {
            "source_type": "documentation",
            "source_organization_name": "Vals.ai",
            "evaluator_relationship": "third_party",
            "additional_details": {"benchmark_slug": slug},
        },
        "model_info": {"name": model_id.split("/")[-1], "id": model_id},
        "eval_library": {"name": "Vals.ai", "version": "unknown"},
        "evaluation_results": results,
    }


def _write_vals_tree(root: Path, records) -> Path:
    for name, record in records:
        path = root / "data" / "vals-ai" / "dev" / "model" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    return root


def _vals_records() -> list:
    return [
        # The record itself states both documented values for its other result.
        ("a", _vals_record("ev_a", "sage", "org/a", [None, 30000], [None, 1.0])),
        # Nothing in this record states either, but the model's other record
        # on the same benchmark states both.
        ("b1", _vals_record("ev_b1", "sage", "org/b", [None], [None])),
        ("b2", _vals_record("ev_b2", "sage", "org/b", [30000], [1.0])),
        # The only corroboration available contradicts both documented values.
        ("c", _vals_record("ev_c", "sage", "org/c", [None, 128000], [None, 0.0])),
        # The cap is corroborated, the temperature beside it is not.
        ("e", _vals_record("ev_e", "sage", "org/e", [None, 30000], [None, 0.0])),
    ]


@pytest.fixture
def vals_tree(tmp_path) -> Path:
    return _write_vals_tree(tmp_path / "eee", _vals_records())


def test_each_missing_field_uses_its_benchmark_page(vals_tree):
    table, diagnostics = builders.build_vals_table(vals_tree, "deadbeef")
    assert {(p["evaluation_id"], p["field"], p["tier"]) for p in table["patches"]} == {
        (evaluation_id, field, "benchmark_page")
        for evaluation_id in ("ev_a", "ev_b1", "ev_c", "ev_e")
        for field in ("temperature", "max_tokens")
    }
    assert diagnostics["uncorroborated"] == []


def test_other_rows_do_not_override_a_page_default(vals_tree):
    table, _ = builders.build_vals_table(vals_tree, "deadbeef")
    fields = {p["field"] for p in table["patches"] if p["evaluation_id"] == "ev_e"}
    assert fields == {"max_tokens", "temperature"}


def test_each_documented_value_carries_the_sentence_it_was_read_from(vals_tree):
    table, _ = builders.build_vals_table(vals_tree, "deadbeef")

    sage = table["documented_statements"]["sage"]
    assert sage["max_tokens"]["value"] == 30000
    assert sage["temperature"]["value"] == 1.0
    assert sorted(table["documented_statements"]) == ["medcode", "medscribe", "mmmu", "sage"]
    for by_field in table["documented_statements"].values():
        for statement in by_field.values():
            assert statement["url"].startswith("https://")
            assert statement["quote"]
            assert statement["verified_on"] == "2026-10-21"
            assert len(statement["page_sha256"]) == 64


def test_a_duplicated_result_key_stops_the_build(tmp_path):
    root = _write_vals_tree(tmp_path / "eee", [
        ("a", _vals_record("ev_dup", "sage", "org/a", [None, 30000])),
        ("b", _vals_record("ev_dup", "sage", "org/a", [None, 30000])),
    ])

    with pytest.raises(RuntimeError, match="not unique"):
        builders.build_vals_table(root, "deadbeef")


def test_the_run_settings_table_is_byte_identical_across_rebuilds(vals_tree):
    first, _ = builders.build_vals_table(vals_tree, "deadbeef")
    second, _ = builders.build_vals_table(vals_tree, "deadbeef")

    assert canonical_json(first) == canonical_json(second)
    assert _sha(first) == (
        "cd65073687dba86a80db8aefbac0f269de0f71ae1d2cb6c261b1358b38ef18cd"
    )


# --- hfopenllm_v2_harness_configs -------------------------------------------


def _task(output_type="generate_until", *, temperature=0.0, max_gen_toks=1024,
          num_fewshot=4, do_sample=False) -> dict:
    if output_type != "generate_until":
        return {"output_type": output_type, "generation_kwargs": None,
                "num_fewshot": num_fewshot}
    return {
        "output_type": output_type,
        "generation_kwargs": {
            "do_sample": do_sample,
            "temperature": temperature,
            "max_gen_toks": max_gen_toks,
        },
        "num_fewshot": num_fewshot,
    }


def _dump(model_name: str, configs: dict, *, pretrained: str | None = None) -> dict:
    name = model_name if pretrained is None else pretrained
    return {
        "model_name": model_name,
        "config": {"model_args": f"pretrained={name},dtype=bfloat16"},
        "configs": configs,
    }


def _math_tasks(max_gen_toks=1024, subtasks=MATH_SUBTASKS) -> dict:
    return {
        f"leaderboard_math_{name}_hard": _task(max_gen_toks=max_gen_toks)
        for name in subtasks
    }


def _full_configs(math_cap=1024) -> dict:
    return {
        "leaderboard_ifeval": _task(max_gen_toks=1280, num_fewshot=0),
        **_math_tasks(math_cap),
        "leaderboard_bbh_one": _task("multiple_choice", num_fewshot=3),
        "leaderboard_bbh_two": _task("multiple_choice", num_fewshot=3),
    }


def _write_dumps(root: Path, dumps) -> Path:
    for directory, name, dump in dumps:
        path = root / directory / f"results_{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dump), encoding="utf-8")
    return root


@pytest.fixture
def results_tree(tmp_path) -> Path:
    return _write_dumps(tmp_path / "results", [
        # Identity comes from inside the dump, never from the directory.
        ("Misleading/Directory", "2024-01-01", _dump("real/model", _full_configs())),
        # A second, agreeing dump for the same model.
        ("real/model", "2024-06-01", _dump("real/model", _full_configs())),
        # Majority-config company, so the 512 model below reads as an override.
        ("other/model", "2024-06-01", _dump("other/model", _full_configs())),
        ("third/model", "2024-06-01", _dump("third/model", _full_configs())),
        # A shorter cap for one model only.
        ("small/model", "2024-06-01", _dump("small/model", _full_configs(512))),
        # Carries no IFEval or MATH tasks at all: ignored for both.
        ("bbh/only", "2024-06-01", _dump("bbh/only", {
            "leaderboard_bbh_one": _task("multiple_choice", num_fewshot=3),
        })),
        # Only some of the MATH subtasks, so it attests nothing for MATH.
        ("partial/model", "2024-06-01", _dump("partial/model", {
            "leaderboard_ifeval": _task(max_gen_toks=1280, num_fewshot=0),
            **_math_tasks(subtasks=MATH_SUBTASKS[:3]),
        })),
        # The dump's own name and the run's arguments disagree.
        ("liar/model", "2024-06-01",
         _dump("liar/model", _full_configs(), pretrained="someone/else")),
    ])


def test_identity_comes_from_inside_the_dump(results_tree):
    table, _ = builders.build_hfopenllm_table(results_tree, "cafe")

    assert "real/model" in table["verified"]["IFEval"]
    assert "Misleading/Directory" not in table["verified"]["IFEval"]


def test_agreeing_dumps_for_one_model_verify_it_once(results_tree):
    table, diagnostics = builders.build_hfopenllm_table(results_tree, "cafe")

    assert table["verified"]["IFEval"].count("real/model") == 1
    assert len(diagnostics["attesting_files"]["real/model"]["IFEval"]) == 2


def test_a_dump_without_a_benchmarks_tasks_is_ignored_for_it(results_tree):
    table, _ = builders.build_hfopenllm_table(results_tree, "cafe")

    assert "bbh/only" in table["verified"]["BBH"]
    assert "bbh/only" not in table["verified"]["IFEval"]
    assert "bbh/only" not in table["conflicts"]


def test_a_partial_benchmark_is_a_conflict_not_a_verification(results_tree):
    table, _ = builders.build_hfopenllm_table(results_tree, "cafe")

    assert table["conflicts"]["partial/model"] == ["MATH Level 5"]
    assert "partial/model" not in table["verified"]["MATH Level 5"]
    # The benchmark it did state completely is unaffected.
    assert "partial/model" in table["verified"]["IFEval"]


def test_a_dump_whose_identity_contradicts_its_arguments_attests_nothing(results_tree):
    table, diagnostics = builders.build_hfopenllm_table(results_tree, "cafe")

    assert "liar/model" not in table["verified"]["IFEval"]
    assert "someone/else" not in table["verified"]["IFEval"]
    assert [r["reason"] for r in diagnostics["files_rejected"]] == [
        "identity_contradicts_model_args"
    ]


@pytest.mark.parametrize("arg", ["peft", "delta"])
def test_a_dump_named_after_its_adapter_checkpoint_attests(tmp_path, arg):
    dump = _dump("adapter/model", _full_configs())
    dump["config"]["model_args"] = f"pretrained=base/model,{arg}=adapter/model,dtype=bfloat16"
    root = _write_dumps(tmp_path, [("adapter/model", "2024-01-01", dump)])

    table, diagnostics = builders.build_hfopenllm_table(root, "cafe")

    assert "adapter/model" in table["verified"]["IFEval"]
    assert "base/model" not in table["verified"]["IFEval"]
    assert diagnostics["files_rejected"] == []


@pytest.mark.parametrize("arg", ["peft", "delta"])
def test_an_adapter_run_named_after_the_base_model_attests_nothing(tmp_path, arg):
    """The run loaded other weights on top, so the base model's settings are
    not what it recorded."""
    dump = _dump("base/model", _full_configs())
    dump["config"]["model_args"] = f"pretrained=base/model,{arg}=adapter/model,dtype=bfloat16"
    root = _write_dumps(tmp_path, [("base/model", "2024-01-01", dump)])

    table, diagnostics = builders.build_hfopenllm_table(root, "cafe")

    assert table["verified"]["IFEval"] == []
    assert [r["reason"] for r in diagnostics["files_rejected"]] == [
        "identity_contradicts_model_args"
    ]


def test_a_dump_naming_both_an_adapter_and_a_delta_is_ambiguous(tmp_path):
    dump = _dump("adapter/model", _full_configs())
    dump["config"]["model_args"] = (
        "pretrained=base/model,peft=adapter/model,delta=other/model"
    )
    root = _write_dumps(tmp_path, [("adapter/model", "2024-01-01", dump)])

    table, diagnostics = builders.build_hfopenllm_table(root, "cafe")

    assert table["verified"]["IFEval"] == []
    assert [r["reason"] for r in diagnostics["files_rejected"]] == [
        "ambiguous_adapter_args"
    ]


def test_the_majority_config_is_the_default_and_the_rest_are_overrides(results_tree):
    table, _ = builders.build_hfopenllm_table(results_tree, "cafe")

    assert table["defaults"]["MATH Level 5"] == {
        "output_type": "generate_until",
        "temperature": 0.0,
        "max_tokens": 1024,
        "num_fewshot": 4,
        "do_sample": False,
    }
    assert table["overrides"] == {
        "small/model": {"MATH Level 5": {
            "output_type": "generate_until",
            "temperature": 0.0,
            "max_tokens": 512,
            "num_fewshot": 4,
            "do_sample": False,
        }}
    }


def test_loglikelihood_benchmarks_record_no_decoding_settings(results_tree):
    table, _ = builders.build_hfopenllm_table(results_tree, "cafe")

    assert table["defaults"]["BBH"] == {
        "output_type": "multiple_choice", "num_fewshot": 3
    }


def test_the_harness_table_is_byte_identical_across_rebuilds(results_tree):
    first, _ = builders.build_hfopenllm_table(results_tree, "cafe")
    second, _ = builders.build_hfopenllm_table(results_tree, "cafe")

    assert canonical_json(first) == canonical_json(second)
    assert _sha(first) == (
        "f69b935b88eeaba35fd7bf94e22409e3d5c6c93f0d10a1d08b08ce39b1136a25"
    )


# --- arc_agi_logged_request_args --------------------------------------------

V1_NAME, V2_NAME = "v1_Public_Eval", "v2_Public_Eval"
V1_DIR = builders.ARC_PUBLIC_RUNS[V1_NAME]["directory"]
V2_DIR = builders.ARC_PUBLIC_RUNS[V2_NAME]["directory"]
RUN_END = "2026-01-01T00:00:00+00:00"
RETRIEVED = "1798761600.0"  # 2026-12-31, after every run written here
COST = 0.25


def _attempt(task_id, test_id, index, correct, *, cap=("max_output_tokens", 128000),
             temperature=None, end=RUN_END, pair_index=...) -> dict:
    kwargs = dict([cap] if cap else [])
    if temperature is not None:
        kwargs["temperature"] = temperature
    metadata = {
        "task_id": task_id,
        "test_id": test_id,
        "pair_index": index if pair_index is ... else pair_index,
        "end_timestamp": end,
        "kwargs": kwargs,
    }
    return {"metadata": metadata, "correct": correct}


def _write_run(root: Path, directory: str, tasks: dict, *, attempt=_attempt,
               results=None, extra_files=None) -> Path:
    """One published run: the task files, plus the results.json they imply."""
    run = root / directory
    run.mkdir(parents=True, exist_ok=True)
    score, total_attempts, task_results = 0.0, 0, {}
    for task_id, pairs in tasks.items():
        payload = []
        for index, attempts in enumerate(pairs):
            payload.append({
                f"attempt_{n}": attempt(task_id, directory, index, correct)
                for n, correct in enumerate(attempts, start=1)
            })
            total_attempts += len(attempts)
        (run / f"{task_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        task_score = sum(any(a) for a in pairs) / len(pairs)
        score += task_score
        task_results[task_id] = {
            "score": task_score, "attempts": sum(len(a) for a in pairs)
        }
    published = {
        "score": score,
        "total_tasks": len(tasks),
        "total_attempts": total_attempts,
        "avg_cost_per_task": COST,
        "task_results": task_results,
    }
    published.update(results or {})
    (run / "results.json").write_text(json.dumps(published), encoding="utf-8")
    for name, content in (extra_files or {}).items():
        (run / name).write_text(content, encoding="utf-8")
    return run


TASKS = {"aaa11111": [[True, False], [True]], "bbb22222": [[False]]}
# Two tasks, one fully solved and one not: score 1.0 of 2 tasks = 0.5.
RUN_SCORE = 0.5


def _arc_record(evaluation_id: str, raw_model_id: str, *, result_name=V1_NAME,
                score=RUN_SCORE, cost=COST, retrieved=RETRIEVED) -> dict:
    def result(metric_id, value):
        return {
            "evaluation_result_id": f"{result_name}::{metric_id}",
            "evaluation_name": result_name,
            "source_data": {"dataset_name": "arc", "source_type": "other"},
            "metric_config": {"metric_id": metric_id, "lower_is_better": False},
            "score_details": {"score": value},
        }

    return {
        "schema_version": "0.3.0",
        "evaluation_id": evaluation_id,
        "retrieved_timestamp": retrieved,
        "source_metadata": {
            "source_type": "documentation",
            "source_organization_name": "ARC Prize Foundation",
            "evaluator_relationship": "third_party",
        },
        "model_info": {
            "name": raw_model_id,
            "id": f"org/{raw_model_id}",
            "additional_details": {"raw_model_id": raw_model_id},
        },
        "eval_library": {"name": "ARC Prize leaderboard", "version": "unknown"},
        "evaluation_results": [result("score", score), result("cost_per_task", cost)],
    }


def _write_arc_datastore(root: Path, records) -> Path:
    for name, record in records:
        path = root / "data" / "arc-agi" / "org" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    return root


@pytest.fixture
def arc_tree(tmp_path) -> tuple[Path, Path]:
    """A datastore holding one ARC record, and a run that produced it."""
    eee = _write_arc_datastore(tmp_path / "eee", [
        ("a", _arc_record("ev_a", "a-model")),
    ])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)
    return eee, runs


def _build_arc(arc_tree, histories=None):
    eee, runs = arc_tree
    return builders.build_arc_table(eee, "deadbeef", runs, histories or {})


def test_a_run_that_survives_every_gate_becomes_one_patch(arc_tree):
    table, diagnostics = _build_arc(arc_tree)

    assert table["patches"] == [{
        "evaluation_id": "ev_a",
        "evaluation_result_id": "v1_Public_Eval::score",
        "evaluation_name": V1_NAME,
        "expected_old": None,
        "fields": {"max_tokens": 128000},
        "cap_key": "max_output_tokens",
        "dataset": builders.ARC_PUBLIC_RUNS[V1_NAME]["dataset"],
        "revision": builders.ARC_PUBLIC_RUNS[V1_NAME]["revision"],
        "directory": "a-model",
        "metadata": {
            "source_data/additional_details/scored_tasks": "2",
            "score_details/uncertainty/num_samples": 2,
            "generation_config/additional_details/cap_key": "max_output_tokens",
            "generation_config/additional_details/request_log": (
                f"{builders.ARC_PUBLIC_RUNS[V1_NAME]['dataset']}@"
                f"{builders.ARC_PUBLIC_RUNS[V1_NAME]['revision']}/a-model"
            ),
        },
    }]
    assert diagnostics["rejected_runs"] == []
    assert diagnostics["rejected_results"] == []


def _rejection(diagnostics) -> str | None:
    reasons = [r["reason"] for r in diagnostics["rejected_runs"]]
    return reasons[0] if reasons else None


@pytest.mark.parametrize("field, value, reason", [
    ("test_id", "another-model", "test_id_mismatch"),
    ("task_id", "cccc3333", "task_id_mismatch"),
    ("pair_index", 7, "pair_index_mismatch"),
    ("pair_index", "0", "pair_index_mismatch"),
    ("pair_index", None, "pair_index_absent"),
    ("end_timestamp", None, "attempt_without_end_time"),
    ("kwargs", None, "attempt_without_kwargs"),
])
def test_an_attempt_that_does_not_name_its_own_run_is_refused(
    tmp_path, field, value, reason
):
    def attempt(task_id, test_id, index, correct):
        made = _attempt(task_id, test_id, index, correct)
        made["metadata"][field] = value
        return made

    eee = _write_arc_datastore(tmp_path / "eee", [("a", _arc_record("ev_a", "a-model"))])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS, attempt=attempt)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert table["patches"] == []
    assert _rejection(diagnostics) == reason


def test_a_task_file_results_json_does_not_list_is_refused(tmp_path, arc_tree):
    eee, runs = arc_tree
    (runs / V1_DIR / "a-model" / "cccc3333.json").write_text(
        json.dumps([{"attempt_1": _attempt("cccc3333", "a-model", 0, True)}]),
        encoding="utf-8",
    )

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert table["patches"] == []
    assert _rejection(diagnostics) == "task_set_mismatch"


@pytest.mark.parametrize("published, reason", [
    ({"score": 1.5}, "run_score_mismatch"),
    ({"total_tasks": 3}, "run_score_mismatch"),
    ({"total_attempts": 9}, "run_score_mismatch"),
    ({"task_results": {"aaa11111": {"score": 0.0}, "bbb22222": {"score": 0.0}}},
     "task_score_mismatch"),
])
def test_a_run_whose_score_does_not_recompute_is_refused(tmp_path, published, reason):
    eee = _write_arc_datastore(tmp_path / "eee", [("a", _arc_record("ev_a", "a-model"))])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS, results=published)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert table["patches"] == []
    assert _rejection(diagnostics) == reason


@pytest.mark.parametrize("caps, reason", [
    ([("max_tokens", 128000), None], "cap_absent"),
    ([("max_tokens", 128000), ("max_tokens", 64000)], "cap_conflict"),
    ([("max_tokens", 128000), ("max_output_tokens", 128000)], "cap_conflict"),
    ([("max_tokens", "128000"), ("max_tokens", "128000")], "cap_not_an_integer"),
])
def test_a_run_without_one_agreed_output_cap_is_refused(tmp_path, caps, reason):
    """A cap under another key, another value, or none at all, is not one
    setting the whole run was made under."""
    seen = iter(caps * 10)

    def attempt(task_id, test_id, index, correct):
        return _attempt(task_id, test_id, index, correct, cap=next(seen))

    eee = _write_arc_datastore(tmp_path / "eee", [("a", _arc_record("ev_a", "a-model"))])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS, attempt=attempt)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert table["patches"] == []
    assert _rejection(diagnostics) == reason


def test_a_cap_stated_under_two_keys_at_once_is_refused(tmp_path):
    def attempt(task_id, test_id, index, correct):
        made = _attempt(task_id, test_id, index, correct)
        made["metadata"]["kwargs"]["max_tokens"] = 128000
        return made

    eee = _write_arc_datastore(tmp_path / "eee", [("a", _arc_record("ev_a", "a-model"))])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS, attempt=attempt)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert table["patches"] == []
    assert _rejection(diagnostics) == "cap_ambiguous"


def test_a_temperature_every_request_states_is_taken(tmp_path):
    def attempt(task_id, test_id, index, correct):
        return _attempt(task_id, test_id, index, correct, temperature=0.7)

    eee = _write_arc_datastore(tmp_path / "eee", [("a", _arc_record("ev_a", "a-model"))])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS, attempt=attempt)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert table["patches"][0]["fields"] == {"max_tokens": 128000, "temperature": 0.7}
    assert diagnostics["temperature_not_taken"] == []


@pytest.mark.parametrize("temperatures, reason", [
    ([0.7, None], "temperature_partial"),
    ([0.7, 0.2], "temperature_conflict"),
])
def test_a_temperature_the_requests_disagree_on_is_dropped(
    tmp_path, temperatures, reason
):
    """The cap is unaffected: the run still stands, without a temperature."""
    seen = iter(temperatures * 10)

    def attempt(task_id, test_id, index, correct):
        return _attempt(task_id, test_id, index, correct, temperature=next(seen))

    eee = _write_arc_datastore(tmp_path / "eee", [("a", _arc_record("ev_a", "a-model"))])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS, attempt=attempt)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert table["patches"][0]["fields"] == {"max_tokens": 128000}
    assert [t["reason"] for t in diagnostics["temperature_not_taken"]] == [reason]


@pytest.mark.parametrize("overrides, reason", [
    ({"score": 0.6}, "score_mismatch"),
    ({"cost": 0.9}, "cost_mismatch"),
    ({"retrieved": "1700000000.0"}, "run_ended_after_retrieval"),
    ({"retrieved": "not-a-time"}, "retrieved_time_unreadable"),
])
def test_a_record_the_run_does_not_account_for_is_refused(tmp_path, overrides, reason):
    eee = _write_arc_datastore(tmp_path / "eee", [
        ("a", _arc_record("ev_a", "a-model", **overrides)),
    ])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert table["patches"] == []
    assert [r["reason"] for r in diagnostics["rejected_results"]] == [reason]


def test_a_published_number_is_matched_at_the_precision_it_carries(tmp_path):
    """0.5 is written to one decimal, so a run scoring 0.5 matches it and a
    run scoring 0.44 does not."""
    eee = _write_arc_datastore(tmp_path / "eee", [
        ("a", _arc_record("ev_a", "a-model", score=0.44)),
    ])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, _ = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert table["patches"] == []


def test_a_rewritten_run_is_refused(arc_tree):
    table, diagnostics = _build_arc(arc_tree, {V1_NAME: {"a-model/aaa11111.json": 2}})

    assert table["patches"] == []
    assert _rejection(diagnostics) == "run_rewritten"


def test_a_run_that_lost_a_file_is_refused(arc_tree):
    """The path is not in the directory any more, which is exactly what a
    second commit touching it did."""
    table, diagnostics = _build_arc(arc_tree, {V1_NAME: {"a-model/ddd44444.json": 2}})

    assert table["patches"] == []
    assert _rejection(diagnostics) == "run_rewritten"


def test_another_run_s_history_does_not_refuse_this_one(arc_tree):
    table, diagnostics = _build_arc(
        arc_tree, {V1_NAME: {"another-model/aaa11111.json": 2}}
    )

    assert len(table["patches"]) == 1
    assert diagnostics["rejected_runs"] == []


def test_every_record_of_the_same_model_that_passes_is_patched(tmp_path):
    eee = _write_arc_datastore(tmp_path / "eee", [
        ("first", _arc_record("ev_first", "a-model")),
        ("second", _arc_record("ev_second", "a-model")),
        # Read before the run finished, so it is a different run's number.
        ("early", _arc_record("ev_early", "a-model", retrieved="1700000000.0")),
    ])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert [p["evaluation_id"] for p in table["patches"]] == ["ev_first", "ev_second"]
    assert [r["reason"] for r in diagnostics["rejected_results"]] == [
        "run_ended_after_retrieval"
    ]


def test_each_public_set_is_filled_only_from_its_own_dataset(tmp_path):
    """One configuration can have run under different caps on the two sets,
    so a value never crosses from one to the other."""
    eee = _write_arc_datastore(tmp_path / "eee", [
        ("one", _arc_record("ev_one", "a-model")),
        ("two", _arc_record("ev_two", "a-model", result_name=V2_NAME)),
    ])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS)
    _write_run(
        runs / V2_DIR, "a-model", TASKS,
        attempt=lambda *args: _attempt(*args, cap=("max_tokens", 64000)),
    )

    table, _ = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert [(p["evaluation_name"], p["fields"]["max_tokens"]) for p in table["patches"]] == [
        (V1_NAME, 128000), (V2_NAME, 64000)
    ]


def test_a_run_with_no_record_to_patch_is_reported_not_read(tmp_path):
    eee = _write_arc_datastore(tmp_path / "eee", [("a", _arc_record("ev_a", "a-model"))])
    runs = tmp_path / "runs"
    _write_run(runs / V1_DIR, "a-model", TASKS)
    _write_run(runs / V1_DIR, "nobody-model", TASKS)
    (runs / V2_DIR).mkdir(parents=True, exist_ok=True)

    table, diagnostics = builders.build_arc_table(eee, "deadbeef", runs, {})

    assert len(table["patches"]) == 1
    assert [(r["directory"], r["reason"]) for r in diagnostics["rejected_runs"]] == [
        ("nobody-model", "no_record_for_this_model")
    ]


def test_the_arc_table_is_byte_identical_across_rebuilds(arc_tree):
    first, _ = _build_arc(arc_tree)
    second, _ = _build_arc(arc_tree)

    assert canonical_json(first) == canonical_json(second)
    assert _sha(first) == (
        "e0e1e2e3e7836ba45c4ce44305a002baccd9d1c9885586988ab57f2603c0602d"
    )


def test_the_run_history_is_read_in_one_pass(tmp_path):
    """A path a later commit touched again was rewritten after it was first
    published, which is what disqualifies the directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    run = ["git", "-C", str(repo)]
    subprocess.run(run + ["init", "-q"], check=True)
    subprocess.run(run + ["config", "user.email", "t@example.invalid"], check=True)
    subprocess.run(run + ["config", "user.name", "test"], check=True)
    for round_ in ("first", "second"):
        (repo / "once.json").write_text("1", encoding="utf-8")
        (repo / "twice.json").write_text(round_, encoding="utf-8")
        subprocess.run(run + ["add", "-A"], check=True)
        subprocess.run(run + ["commit", "-q", "-m", round_], check=True)

    history = builders.read_run_history(repo, "HEAD")

    assert history == {"once.json": 1, "twice.json": 2}


# --- rebuilding a vendored table --------------------------------------------
#
# A vendored table is what a bake actually fills, so a rebuild that differs
# from the shipped one has to be looked at before it lands.


def _isolated_data_dir(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "vendored"
    data_dir.mkdir()
    monkeypatch.setattr(builders, "DATA_DIR", data_dir)
    return data_dir


def _build_vals(tmp_path, tree: Path, *extra: str) -> int:
    from eval_card_backend import cli

    return cli.main([
        "datafix", "build-vals-table",
        "--eee-root", str(tree),
        "--eee-revision", "deadbeef",
        "--diagnostics", str(tmp_path / "diag.json"),
        *extra,
    ])


def _build_hfopenllm(tmp_path, tree: Path, *extra: str) -> int:
    from eval_card_backend import cli

    return cli.main([
        "datafix", "build-hfopenllm-table",
        "--results-dir", str(tree),
        "--revision", "c" * 40,
        "--diagnostics", str(tmp_path / "diag.json"),
        *extra,
    ])


def test_a_first_build_writes_the_table(tmp_path, monkeypatch, vals_tree):
    from eval_card_backend.datafix import vals_ai

    data_dir = _isolated_data_dir(monkeypatch, tmp_path)

    assert _build_vals(tmp_path, vals_tree) == 0
    assert (data_dir / vals_ai.TABLE_NAME).is_file()


def test_a_rebuild_on_unchanged_evidence_leaves_the_table_alone(
    tmp_path, monkeypatch, vals_tree
):
    from eval_card_backend.datafix import vals_ai

    data_dir = _isolated_data_dir(monkeypatch, tmp_path)
    _build_vals(tmp_path, vals_tree)
    path = data_dir / vals_ai.TABLE_NAME
    before = path.read_bytes()
    stamp = path.stat().st_mtime_ns

    assert _build_vals(tmp_path, vals_tree) == 0
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == stamp


def test_an_extra_patch_is_refused_and_reported(tmp_path, monkeypatch, capsys):
    data_dir = _isolated_data_dir(monkeypatch, tmp_path)
    from eval_card_backend.datafix import vals_ai

    tree = _write_vals_tree(tmp_path / "eee", _vals_records())
    _build_vals(tmp_path, tree)
    before = (data_dir / vals_ai.TABLE_NAME).read_bytes()
    capsys.readouterr()

    _write_vals_tree(tmp_path / "eee", [
        ("g", _vals_record("ev_g", "sage", "org/g", [None, 30000], [None, 1.0])),
    ])
    assert _build_vals(tmp_path, tree) == 1

    err = capsys.readouterr().err
    assert "refusing to change" in err
    assert "added max_tokens patches: 1" in err
    assert "added temperature patches: 1" in err
    assert "+ ev_g / ev_g:r0 max_tokens (tier benchmark_page)" in err
    assert "+ ev_g / ev_g:r0 temperature (tier benchmark_page)" in err
    assert (data_dir / vals_ai.TABLE_NAME).read_bytes() == before


def test_a_vanished_patch_is_refused_and_reported(tmp_path, monkeypatch, capsys):
    data_dir = _isolated_data_dir(monkeypatch, tmp_path)
    from eval_card_backend.datafix import vals_ai

    tree = _write_vals_tree(tmp_path / "eee", _vals_records())
    _build_vals(tmp_path, tree)
    before = (data_dir / vals_ai.TABLE_NAME).read_bytes()
    capsys.readouterr()

    # Removing the gap itself removes its patch from the frozen export.
    (tree / "data" / "vals-ai" / "dev" / "model" / "b1.json").unlink()
    assert _build_vals(tmp_path, tree) == 1

    err = capsys.readouterr().err
    assert "- ev_b1 / ev_b1:r0 max_tokens (tier benchmark_page)" in err
    assert (data_dir / vals_ai.TABLE_NAME).read_bytes() == before


def test_the_flag_accepts_a_changed_patch_table(tmp_path, monkeypatch, vals_tree):
    from eval_card_backend.datafix import vals_ai

    data_dir = _isolated_data_dir(monkeypatch, tmp_path)
    _build_vals(tmp_path, vals_tree)
    before = (data_dir / vals_ai.TABLE_NAME).read_bytes()

    _write_vals_tree(vals_tree, [
        ("g", _vals_record("ev_g", "sage", "org/g", [None, 30000], [None, 1.0])),
    ])
    assert _build_vals(tmp_path, vals_tree, "--accept-changes") == 0
    assert (data_dir / vals_ai.TABLE_NAME).read_bytes() != before


def test_a_changed_harness_table_is_refused_and_reported(
    tmp_path, monkeypatch, capsys, results_tree
):
    from eval_card_backend.datafix import hfopenllm_v2

    data_dir = _isolated_data_dir(monkeypatch, tmp_path)
    assert _build_hfopenllm(tmp_path, results_tree) == 0
    before = (data_dir / hfopenllm_v2.TABLE_NAME).read_bytes()
    capsys.readouterr()

    _write_dumps(results_tree, [
        ("fourth/model", "2024-06-01", _dump("fourth/model", _full_configs())),
    ])
    assert _build_hfopenllm(tmp_path, results_tree) == 1

    err = capsys.readouterr().err
    assert "verified IFEval: +1 / -0 model(s)" in err
    assert (data_dir / hfopenllm_v2.TABLE_NAME).read_bytes() == before

    assert _build_hfopenllm(tmp_path, results_tree, "--accept-changes") == 0
    assert (data_dir / hfopenllm_v2.TABLE_NAME).read_bytes() != before


# --- export -----------------------------------------------------------------


def _only_vals_fix(monkeypatch, table) -> None:
    from eval_card_backend.datafix import registry, vals_ai

    fix = vals_ai.ValsAiMaxTokens({**table, "_table_sha256": "0" * 64})
    entry = registry.Entry(
        vals_ai.FIX_ID, vals_ai.SOURCE_CONFIG, lambda: fix,
        lambda: "0" * 64, vals_ai.SOURCE_REVISION,
    )
    monkeypatch.setattr(registry, "_REGISTRY", (entry,))
    monkeypatch.setattr(registry, "_instances", {0: fix})


def test_the_export_refuses_a_populated_output_directory(tmp_path, monkeypatch):
    root = _write_vals_tree(tmp_path / "eee", [
        ("a", _vals_record("ev_a", "sage", "org/a", [None, 30000])),
    ])
    table, _ = builders.build_vals_table(root, "deadbeef")
    _only_vals_fix(monkeypatch, table)

    out = tmp_path / "out"
    out.mkdir()
    (out / "data").mkdir()

    with pytest.raises(FileExistsError, match="is not empty"):
        builders.export_patched_records(root, out)


def test_the_export_command_reports_a_populated_output_directory(
    tmp_path, monkeypatch, capsys
):
    from eval_card_backend import cli

    root = _write_vals_tree(tmp_path / "eee", [
        ("a", _vals_record("ev_a", "sage", "org/a", [None, 30000])),
    ])
    table, _ = builders.build_vals_table(root, "deadbeef")
    _only_vals_fix(monkeypatch, table)

    out = tmp_path / "out"
    out.mkdir()
    (out / "patch_manifest.jsonl").write_text("stale\n", encoding="utf-8")

    assert cli.main([
        "datafix", "export", "--eee-root", str(root), "--out", str(out)
    ]) == 2
    assert "is not empty" in capsys.readouterr().err
    assert (out / "patch_manifest.jsonl").read_text(encoding="utf-8") == "stale\n"


def test_the_export_writes_only_affected_records_with_a_stable_manifest(
    tmp_path, monkeypatch
):
    root = _write_vals_tree(tmp_path / "eee", [
        ("a", _vals_record("ev_a", "sage", "org/a", [None, 30000])),
        ("untouched", _vals_record("ev_u", "undocumented", "org/u", [None])),
    ])
    table, _ = builders.build_vals_table(root, "deadbeef")
    _only_vals_fix(monkeypatch, table)

    counts = builders.export_patched_records(root, tmp_path / "out")
    manifest = (tmp_path / "out" / "patch_manifest.jsonl").read_text(encoding="utf-8")

    assert counts == {
        "records_scanned": 2,
        "records_patched": 1,
        "results_patched": 2,
        "fields_patched": 8,
    }
    assert (tmp_path / "out" / "data/vals-ai/dev/model/a.json").exists()
    assert not (tmp_path / "out" / "data/vals-ai/dev/model/untouched.json").exists()
    manifest_rows = [json.loads(line) for line in manifest.splitlines()]
    assert {row["record_path"] for row in manifest_rows} == {"data/vals-ai/dev/model/a.json"}

    builders.export_patched_records(root, tmp_path / "out2")
    assert (tmp_path / "out2" / "patch_manifest.jsonl").read_text(
        encoding="utf-8"
    ) == manifest
    assert all(row["concurring_fix_ids"] == [] for row in manifest_rows)

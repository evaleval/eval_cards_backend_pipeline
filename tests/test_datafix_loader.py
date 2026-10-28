"""The overlay at the Stage A record boundary.

Records go through the typed loader exactly as they always did; the
overlay sits between the validate and cast gates. These tests drive the
loader over a hand-built datastore tree with the vendored tables in
place, so they cover the wiring and the run's own accounting rather than
the appliers (which `test_datafix_appliers.py` covers on their own).
"""
from __future__ import annotations

import json

import pytest

from eval_card_backend import datafix
from eval_card_backend.sources import eee as eee_src
from tests.eee_layout import write_eee_datastore

# A model the vendored harness-config table verifies, so the loader path
# exercises the real table.
VERIFIED_MODEL = "gpt2"


@pytest.fixture(autouse=True)
def _reset():
    eee_src.reset_drop_counter()
    yield
    eee_src.reset_drop_counter()


@pytest.fixture
def accounting() -> datafix.DatafixSummary:
    return datafix.DatafixSummary()


def _hfopenllm_record(model_id: str = VERIFIED_MODEL, eval_id: str = "ev_1") -> dict:
    return {
        "schema_version": "0.2.2",
        "evaluation_id": eval_id,
        "retrieved_timestamp": "2026-09-01T00:00:00Z",
        "source_metadata": {
            "source_type": "documentation",
            "source_organization_name": "Hugging Face",
            "evaluator_relationship": "third_party",
        },
        "model_info": {"name": model_id, "id": model_id, "additional_details": {"precision": "float16"}},
        "eval_library": {"name": "lm-evaluation-harness", "version": "0.4.0"},
        "evaluation_results": [
            {
                "evaluation_name": name,
                "source_data": {"dataset_name": name, "source_type": "hf_dataset"},
                "metric_config": {"lower_is_better": False},
                "score_details": {"score": 0.5},
            }
            for name in ("IFEval", "MATH Level 5", "BBH")
        ],
    }


def _generation_args(table, row: int, result: int):
    config = table.column("evaluation_results").to_pylist()[row][result]["generation_config"]
    return config["generation_args"] if config else None


def test_the_overlay_fills_gaps_when_enabled(tmp_path):
    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        ("hfopenllm_v2", "a.json", json.dumps(_hfopenllm_record())),
    ])

    table = eee_src.load_arrow_table(eee_root, ["hfopenllm_v2"], hf_token=None, datafix_enabled=True)

    assert table.num_rows == 1
    assert _generation_args(table, 0, 0)["max_tokens"] == 1280
    assert _generation_args(table, 0, 1)["max_tokens"] == 1024
    # The loglikelihood-scored benchmark is left exactly as it arrived.
    assert _generation_args(table, 0, 2) is None


def test_the_kill_switch_loads_raw_records(tmp_path, accounting):
    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        ("hfopenllm_v2", "a.json", json.dumps(_hfopenllm_record())),
    ])

    table = eee_src.load_arrow_table(
        eee_root, ["hfopenllm_v2"], hf_token=None, datafix_enabled=False,
        datafix_summary=accounting,
    )

    assert table.num_rows == 1
    assert _generation_args(table, 0, 0) is None
    summary = accounting.as_dict(enabled=False)
    assert summary["enabled"] is False
    assert all(
        fix["results_patched"] == 0 for fix in summary["fixes"].values()
    )


def test_the_summary_counts_what_was_filled_and_what_was_not(tmp_path, accounting):
    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        ("hfopenllm_v2", "a.json", json.dumps(_hfopenllm_record())),
        ("hfopenllm_v2", "b.json", json.dumps(
            _hfopenllm_record(model_id="nobody/unverified", eval_id="ev_2")
        )),
    ])

    eee_src.load_arrow_table(
        eee_root, ["hfopenllm_v2"], hf_token=None, datafix_summary=accounting, datafix_enabled=True
    )

    block = accounting.as_dict(enabled=True)["fixes"][
        "hfopenllm_v2_harness_configs"
    ]
    assert block["records_patched"] == 1
    assert block["results_patched"] == 3
    assert block["by_benchmark"] == {"IFEval": 1, "MATH Level 5": 1, "BBH": 1}
    assert block["unmatched"] == 3
    assert block["conflict"] == 0


def test_the_summary_is_reported_for_every_fix_even_with_nothing_to_do(
    tmp_path, accounting
):
    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        ("hfopenllm_v2", "a.json", json.dumps(_hfopenllm_record())),
    ])

    eee_src.load_arrow_table(
        eee_root, ["hfopenllm_v2"], hf_token=None, datafix_summary=accounting, datafix_enabled=True
    )

    summary = accounting.as_dict(enabled=True)
    assert set(summary["fixes"]) == {
        "vals_ai_max_tokens", "hfopenllm_v2_harness_configs",
        "inspect_plan_config", "arc_agi_logged_request_args", "published_source_metadata",
    }
    assert summary["fingerprint"] == datafix.fingerprint()
    assert summary["fixes"]["vals_ai_max_tokens"]["records_patched"] == 0
    assert summary["fixes"]["hfopenllm_v2_harness_configs"]["by_source_config"] == {
        "hfopenllm_v2": 3
    }


def test_a_malformed_record_is_still_dropped_by_the_existing_gates(tmp_path):
    """A fix must never rescue input the upstream contract rejects."""
    eee_root = tmp_path / "eee"
    broken = _hfopenllm_record(eval_id="ev_broken")
    del broken["model_info"]
    write_eee_datastore(eee_root, [
        ("hfopenllm_v2", "a.json", json.dumps(_hfopenllm_record())),
        ("hfopenllm_v2", "bad.json", json.dumps(broken)),
        ("hfopenllm_v2", "corrupt.json", "{not valid json"),
    ])

    table = eee_src.load_arrow_table(eee_root, ["hfopenllm_v2"], hf_token=None, datafix_enabled=True)

    assert table.num_rows == 1
    assert sum(eee_src._drop_counter.values()) == 2


def test_configs_a_patch_table_does_not_name_are_left_alone(tmp_path):
    eee_root = tmp_path / "eee"
    record = _hfopenllm_record()
    write_eee_datastore(eee_root, [("minicfg", "a.json", json.dumps(record))])

    table = eee_src.load_arrow_table(eee_root, ["minicfg"], hf_token=None, datafix_enabled=True)

    assert table.num_rows == 1
    assert _generation_args(table, 0, 0) is None


def _refuse_to_parse_tables(monkeypatch) -> None:
    """Make parsing any vendored table an error, wherever it is reached from."""
    from eval_card_backend.datafix import arc_agi, core, hfopenllm_v2, registry, vals_ai

    def refuse(name):
        raise AssertionError(f"parsed the patch table {name}")

    registry.reload()
    monkeypatch.setattr(core, "load_table", refuse)
    from eval_card_backend.canonicalise import resolution_hotfixes
    monkeypatch.setattr(resolution_hotfixes, "load_table", refuse)
    for module in (vals_ai, hfopenllm_v2, arc_agi):
        monkeypatch.setattr(module, "load_table", refuse)


def test_a_run_over_no_named_config_never_opens_a_patch_table(tmp_path, monkeypatch):
    """The registry answers which configs are covered, which overlay this is
    and what a fix that never ran did, all from its static entries. The fix
    that runs over every config carries no table, so a run touching no named
    config pays for no table read anywhere along the way: not to fingerprint
    the overlay, not to load, and not to report what it did."""
    from eval_card_backend.datafix import registry

    _refuse_to_parse_tables(monkeypatch)
    eee_root = tmp_path / "eee"
    write_eee_datastore(
        eee_root, [("minicfg", "a.json", json.dumps(_hfopenllm_record()))]
    )
    accounting = datafix.DatafixSummary()
    try:
        assert datafix.fingerprint()
        table = eee_src.load_arrow_table(
            eee_root, ["minicfg"], hf_token=None, datafix_summary=accounting, datafix_enabled=True
        )
        summary = accounting.as_dict(enabled=True)
    finally:
        registry.reload()

    assert table.num_rows == 1
    assert summary["fingerprint"] == datafix.fingerprint()
    assert set(summary["fixes"]) == {
        "vals_ai_max_tokens", "hfopenllm_v2_harness_configs",
        "inspect_plan_config", "arc_agi_logged_request_args", "published_source_metadata",
    }
    assert summary["fixes"]["vals_ai_max_tokens"]["table_sha256"]
    assert summary["fixes"]["vals_ai_max_tokens"]["source_revision"]


def test_the_kill_switch_never_opens_a_patch_table(tmp_path, monkeypatch):
    from eval_card_backend.datafix import registry

    _refuse_to_parse_tables(monkeypatch)
    eee_root = tmp_path / "eee"
    write_eee_datastore(
        eee_root, [("hfopenllm_v2", "a.json", json.dumps(_hfopenllm_record()))]
    )
    accounting = datafix.DatafixSummary()
    try:
        assert datafix.fingerprint()
        table = eee_src.load_arrow_table(
            eee_root, ["hfopenllm_v2"], hf_token=None, datafix_enabled=False,
            datafix_summary=accounting,
        )
        summary = accounting.as_dict(enabled=False)
    finally:
        registry.reload()

    assert table.num_rows == 1
    assert summary["enabled"] is False
    assert all(block["results_patched"] == 0 for block in summary["fixes"].values())


def test_two_runs_accounts_do_not_see_each_other(tmp_path):
    """The account belongs to a run, so a second run in the same process
    starts from nothing."""
    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        ("hfopenllm_v2", "a.json", json.dumps(_hfopenllm_record())),
    ])
    first, second = datafix.DatafixSummary(), datafix.DatafixSummary()

    eee_src.load_arrow_table(
        eee_root, ["hfopenllm_v2"], hf_token=None, datafix_summary=first, datafix_enabled=True
    )
    eee_src.load_arrow_table(
        eee_root, ["hfopenllm_v2"], hf_token=None, datafix_summary=second, datafix_enabled=True
    )

    for account in (first, second):
        block = account.as_dict(enabled=True)["fixes"]["hfopenllm_v2_harness_configs"]
        assert (block["records_patched"], block["results_patched"]) == (1, 3)


def test_the_in_record_rule_fills_a_config_no_table_names(tmp_path, accounting):
    """Its evidence is inside the record, so it runs wherever the record came
    from, and reports which config that was."""
    eee_root = tmp_path / "eee"
    record = _hfopenllm_record(eval_id="ev_plan")
    record["eval_library"] = {"name": "inspect_ai", "version": "inspect_ai:0.3.75"}
    record["evaluation_results"] = [{
        "evaluation_name": "task",
        "source_data": {"dataset_name": "task", "source_type": "hf_dataset"},
        "metric_config": {"lower_is_better": False},
        "score_details": {"score": 0.5},
        "generation_config": {"generation_args": {"eval_plan": {
            "name": "plan",
            "steps": [json.dumps({"solver": "generate", "params": {}})],
            "config": {"temperature": "0.0", "max_tokens": "4096"},
        }}},
    }]
    write_eee_datastore(eee_root, [("minicfg", "a.json", json.dumps(record))])

    table = eee_src.load_arrow_table(
        eee_root, ["minicfg"], hf_token=None, datafix_summary=accounting, datafix_enabled=True
    )

    args = _generation_args(table, 0, 0)
    assert (args["temperature"], args["max_tokens"]) == (0.0, 4096)
    block = accounting.as_dict(enabled=True)["fixes"]["inspect_plan_config"]
    assert block["by_source_config"] == {"minicfg": 1}
    assert block["by_field"] == {"max_tokens": 1, "temperature": 1}


def test_metadata_flags_default_off(monkeypatch):
    from eval_card_backend.config import Settings

    monkeypatch.delenv("EEE_DATAFIX", raising=False)
    monkeypatch.delenv("REPRO_EXEMPT_NON_GENERATIVE", raising=False)
    settings = Settings.from_env()
    assert settings.datafix_enabled is False
    assert settings.repro_exempt_declaration_sources == frozenset()


def test_default_loader_preserves_missing_source_fields(tmp_path):
    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        ("hfopenllm_v2", "a.json", json.dumps(_hfopenllm_record())),
    ])
    table = eee_src.load_arrow_table(eee_root, ["hfopenllm_v2"], hf_token=None)
    assert _generation_args(table, 0, 0) is None

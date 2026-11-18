"""The `max_attempts: 1` the shared schema used to write on its own.

Records written before 2026-09-10 state `max_attempts: 1` whether or not
the source said anything; records written since state it only when the
converter does. These tests pin which of those ones the overlay clears,
that a cleared record dedups against its re-emission in Stage E, and that
the rule agrees with the older HLE correction where the two overlap.
"""
from __future__ import annotations

import copy

import pytest

from eval_card_backend.datafix.core import DatafixSummary, apply_fixes
from tests.test_canonicalise_e2e import _assert_dedup_by_config_sums_to_row_counts
from tests.test_datafix_metadata import hle_record
from tests.test_stage_e_invariants import (
    _counts,
    _facts,
    _run,
    _snapshot_meta,
)
from tests.test_stage_e_invariants import _record as _pipeline_record

FIX_ID = "invented_max_attempts"
POINTER = "/evaluation_results/0/generation_config/generation_args/max_attempts"


def _record(
    *,
    max_attempts=1,
    library="test-lib",
    retrieved="2026-09-01T00:00:00Z",
    cron_run_date=None,
) -> dict:
    details = {} if cron_run_date is None else {"cron_run_date": cron_run_date}
    return {
        "schema_version": "0.2.2",
        "evaluation_id": "ev_1",
        "retrieved_timestamp": retrieved,
        "source_metadata": {
            "source_type": "documentation",
            "source_organization_name": "test-org",
            "evaluator_relationship": "third_party",
            "additional_details": details,
        },
        "model_info": {"name": "model", "id": "test/model"},
        "eval_library": {"name": library, "version": "1.0"},
        "evaluation_results": [{
            "evaluation_name": "task",
            "evaluation_result_id": "r0",
            "source_data": {"dataset_name": "test", "source_type": "other"},
            "metric_config": {"lower_is_better": False},
            "score_details": {"score": 0.5},
            "generation_config": {
                "generation_args": {"temperature": 0.0, "max_attempts": max_attempts},
            },
        }],
    }


def _args(record: dict) -> dict:
    return record["evaluation_results"][0]["generation_config"]["generation_args"]


def _apply(record: dict):
    return apply_fixes(record, "any-config")


def _assert_cleared(record: dict) -> None:
    before = copy.deepcopy(record)

    patched, events, counters = _apply(record)

    assert record == before
    assert _args(patched) == {"temperature": 0.0}
    assert [(e.fix_id, e.pointer, e.old, e.new) for e in events] == [
        (FIX_ID, POINTER, 1, None)
    ]
    assert counters[f"{FIX_ID}:cleared"] == 1


def _assert_kept(record: dict, reason: str | None) -> None:
    patched, events, counters = _apply(record)

    assert patched is None and events == []
    assert {k: v for k, v in counters.items() if k.startswith(FIX_ID)} == (
        {} if reason is None else {f"{FIX_ID}:{reason}": 1}
    )


# --- which ones are cleared -------------------------------------------------


def test_an_invented_one_on_an_old_record_is_cleared_and_its_old_value_recorded():
    _assert_cleared(_record())


def test_a_one_written_after_the_schema_stopped_inventing_it_is_kept():
    _assert_kept(_record(retrieved="2026-09-10T00:00:00Z"), "kept_written_after")


def test_a_one_the_inspect_converter_set_is_kept():
    _assert_kept(_record(library="inspect_ai"), "kept_deliberate_library")


@pytest.mark.parametrize("value", [3, 5, 0])
def test_any_other_value_is_kept(value):
    _assert_kept(_record(max_attempts=value), None)


def test_an_unstated_value_is_left_alone():
    record = _record()
    del _args(record)["max_attempts"]
    _assert_kept(record, None)


@pytest.mark.parametrize(
    "generation_config",
    [None, "not a dict", {"generation_args": None}, {"generation_args": [1]}],
    ids=["null-config", "config-not-a-dict", "null-args", "args-not-a-dict"],
)
def test_a_missing_or_malformed_generation_config_is_skipped(generation_config):
    record = _record()
    record["evaluation_results"][0]["generation_config"] = generation_config
    before = copy.deepcopy(record)

    patched, events, _ = _apply(record)

    assert patched is None and events == []
    assert record == before


def test_a_result_without_a_generation_config_is_skipped():
    record = _record()
    del record["evaluation_results"][0]["generation_config"]
    _assert_kept(record, None)


# --- when the record was written --------------------------------------------


def test_the_cron_run_date_dates_the_write_ahead_of_the_retrieval_time():
    _assert_cleared(
        _record(cron_run_date="2026-09-09", retrieved="2026-10-01T00:00:00Z")
    )
    _assert_kept(
        _record(cron_run_date="2026-09-10", retrieved="2026-08-01T00:00:00Z"),
        "kept_written_after",
    )


def test_an_epoch_seconds_retrieval_time_dates_the_write():
    # 1788998399 is 2026-09-09T23:59:59Z, the last second before the cutoff.
    _assert_cleared(_record(retrieved="1788998399.5"))
    _assert_kept(_record(retrieved="1788998400"), "kept_written_after")


@pytest.mark.parametrize(
    "retrieved,cron_run_date",
    [("not a time", None), ("20260901", None), ("2026-09-01T00:00:00Z", "last week")],
    ids=["unparseable-retrieval", "bare-number-date", "unparseable-cron-date"],
)
def test_a_record_with_no_readable_write_date_keeps_its_value(retrieved, cron_run_date):
    _assert_kept(_record(retrieved=retrieved, cron_run_date=cron_run_date), "undated")


def test_each_result_of_a_record_is_judged_on_its_own_value():
    record = _record()
    second = copy.deepcopy(record["evaluation_results"][0])
    second["evaluation_result_id"] = "r1"
    second["generation_config"]["generation_args"]["max_attempts"] = 3
    record["evaluation_results"].append(second)

    patched, events, _ = _apply(record)

    assert [e.result_index for e in events] == [0]
    assert "max_attempts" not in patched["evaluation_results"][0]["generation_config"]["generation_args"]
    assert patched["evaluation_results"][1] == second


# --- the HLE correction proposes the same clear -----------------------------


def test_the_hle_correction_and_this_rule_agree_on_an_old_hle_record():
    """Both read the record as it arrived and both propose clearing the 1,
    so the field is written once and neither counts a conflict."""
    record = hle_record(note="Sampled at temperature = 0.6, top_p = 0.8.")
    record["retrieved_timestamp"] = "2026-08-01T00:00:00Z"
    summary = DatafixSummary()

    patched, events, counters = apply_fixes(record, "hle", summary=summary)

    assert _args(patched) == {"temperature": 0.6, "top_p": 0.8}
    assert [
        (e.fix_id, e.concurring_fix_ids, e.old, e.new)
        for e in events if e.pointer == POINTER
    ] == [("published_source_metadata", (FIX_ID,), 1, None)]
    assert not [k for k in counters if k.endswith(":cross_fix_conflict")]
    fixes = summary.as_dict(enabled=True)["fixes"]
    assert fixes[FIX_ID]["cleared"] == 1
    assert fixes[FIX_ID]["cross_fix_conflict"] == 0
    assert fixes["published_source_metadata"]["cross_fix_conflict"] == 0


def test_an_hle_record_the_older_correction_does_not_recognise_is_still_cleared():
    record = hle_record()
    record["retrieved_timestamp"] = "2026-08-01T00:00:00Z"
    record["evaluation_results"][0]["metric_config"]["llm_scoring"]["input_prompt"] = (
        "Judge description."
    )

    patched, events, _ = apply_fixes(record, "hle")

    assert "max_attempts" not in _args(patched)
    assert [(e.fix_id, e.concurring_fix_ids) for e in events if e.pointer == POINTER] == [
        (FIX_ID, ())
    ]


# --- Stage E: a cleared record dedups against its re-emission ---------------


def _with_max_attempts(record: dict, value: int) -> dict:
    record = copy.deepcopy(record)
    record["evaluation_results"][0]["generation_config"]["generation_args"]["max_attempts"] = value
    return record


def test_a_reemission_without_the_invented_one_is_the_same_observation(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("EEE_DATAFIX", "1")

    out_dir = _run(tmp_path, monkeypatch, [
        ("alpha", "old.json", _with_max_attempts(
            _pipeline_record("ev_old", "2026-04-30T00:00:00Z", 0.8), 1)),
        ("alpha", "new.json", _pipeline_record("ev_new", "2026-05-02T00:00:00Z", 0.8)),
    ])

    assert [(c, e, s) for c, e, s, _ in _facts(out_dir)] == [("alpha", "ev_new", 0.8)]
    snap = _snapshot_meta(out_dir)
    _assert_dedup_by_config_sums_to_row_counts(snap)
    assert snap["dedup_by_config"] == {"alpha": _counts(2, 2, 1)}
    block = snap["datafix"]["fixes"][FIX_ID]
    assert (block["records_patched"], block["results_patched"]) == (1, 1)
    assert block["by_source_config"] == {"alpha": 1}
    assert block["cleared"] == 1


def test_without_the_overlay_the_two_copies_stay_apart(tmp_path, monkeypatch):
    out_dir = _run(tmp_path, monkeypatch, [
        ("alpha", "old.json", _with_max_attempts(
            _pipeline_record("ev_old", "2026-04-30T00:00:00Z", 0.8), 1)),
        ("alpha", "new.json", _pipeline_record("ev_new", "2026-05-02T00:00:00Z", 0.8)),
    ])

    assert [e for _, e, _, _ in _facts(out_dir)] == ["ev_new", "ev_old"]


def test_a_stated_multi_attempt_protocol_stays_a_separate_observation(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("EEE_DATAFIX", "1")

    out_dir = _run(tmp_path, monkeypatch, [
        ("alpha", "three.json", _with_max_attempts(
            _pipeline_record("ev_three", "2026-04-30T00:00:00Z", 0.8), 3)),
        ("alpha", "unstated.json", _pipeline_record("ev_unstated", "2026-05-02T00:00:00Z", 0.8)),
    ])

    assert [e for _, e, _, _ in _facts(out_dir)] == ["ev_three", "ev_unstated"]
    assert _snapshot_meta(out_dir)["datafix"]["fixes"][FIX_ID]["results_patched"] == 0

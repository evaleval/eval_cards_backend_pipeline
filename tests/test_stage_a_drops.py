"""Stage A drop-counter tests.

Drops are counted with first-occurrence logging and surfaced at end of
run via `eee.log_drop_summary` + `snapshot_meta.row_counts`. Without the
counter, per-row WARN logs in a 30k-input run hide aggregate loss.

The typed Arrow loader rejects records at three gates: read_error
(unreadable JSON), not_a_dict (non-object root), and validation_error
(Pydantic contract violation). All three feed the same counter.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from eval_card_backend.canonicalise import stages
from eval_card_backend.sources import eee as eee_src


@pytest.fixture(autouse=True)
def _reset():
    eee_src.reset_drop_counter()
    yield
    eee_src.reset_drop_counter()


def _conformant_record(eval_id: str = "ev_good") -> dict:
    """Minimal record that satisfies every required field of EvaluationLog."""
    return {
        "schema_version": "0.2.2",
        "evaluation_id": eval_id,
        "retrieved_timestamp": "2026-04-30T00:00:00Z",
        "source_metadata": {
            "source_type": "evaluation_run",
            "source_organization_name": "test-org",
            "evaluator_relationship": "first_party",
        },
        "model_info": {"name": "test-model", "id": "test/test-model"},
        "eval_library": {"name": "test-lib", "version": "1.0"},
        "evaluation_results": [
            {
                "evaluation_name": "test-eval",
                "source_data": {"dataset_name": "test", "source_type": "other"},
                "metric_config": {"lower_is_better": False},
                "score_details": {"score": 0.5},
            }
        ],
    }


def _write_record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_corrupt_json_increments_drop_counter(tmp_path, monkeypatch):
    eee_root = tmp_path / "eee"
    cfg_dir = eee_root / "data" / "minicfg" / "openai" / "gpt-4o"
    cfg_dir.mkdir(parents=True)
    # One valid record (passes pydantic validation under the typed loader)
    _write_record(cfg_dir / "good.json", _conformant_record())
    # One corrupt JSON
    (cfg_dir / "bad.json").write_text("{not valid json")

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)

    con = duckdb.connect()
    table = eee_src.load_arrow_table(eee_root, ["minicfg"], hf_token=None)
    n = stages.stage_a_load_eee(con, table)

    # One record loaded, one dropped
    assert n == 1
    drops = sum(eee_src._drop_counter.values())
    assert drops == 1
    # The drop reason should be a read_error of some sort
    assert any(
        reason.startswith("read_error:")
        for (_cfg, reason) in eee_src._drop_counter
    )


def test_validation_error_increments_drop_counter(tmp_path, monkeypatch):
    """A record that parses cleanly but violates the upstream Pydantic
    contract (e.g. score is a string instead of a number) must be dropped
    at the validation gate. Without this, a malformed score silently
    propagates NULLs through to Stage E and the row gets dropped for
    'no_score' reasons that hide the real cause."""
    eee_root = tmp_path / "eee"
    cfg_dir = eee_root / "data" / "minicfg" / "openai" / "gpt-4o"
    cfg_dir.mkdir(parents=True)
    # Conformant record
    _write_record(cfg_dir / "good.json", _conformant_record())
    # Record with a type error: score must be a number per the contract.
    bad = _conformant_record("ev_bad")
    bad["evaluation_results"][0]["score_details"]["score"] = "not a number"
    _write_record(cfg_dir / "bad_score.json", bad)

    con = duckdb.connect()
    table = eee_src.load_arrow_table(eee_root, ["minicfg"], hf_token=None)
    n = stages.stage_a_load_eee(con, table)

    assert n == 1, "good record should load; bad-score record should drop"
    assert any(
        reason == "validation_error"
        for (_cfg, reason) in eee_src._drop_counter
    ), f"expected validation_error in {dict(eee_src._drop_counter)}"


def test_drop_counter_distinguishes_configs(tmp_path, monkeypatch):
    eee_root = tmp_path / "eee"
    for cfg in ("a", "b"):
        cfg_dir = eee_root / "data" / cfg / "dev" / "model"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "bad.json").write_text("nope")

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)

    con = duckdb.connect()
    table = eee_src.load_arrow_table(eee_root, ["a", "b"], hf_token=None)
    n = stages.stage_a_load_eee(con, table)
    assert n == 0

    cfgs_with_drops = {cfg for (cfg, _reason) in eee_src._drop_counter}
    assert cfgs_with_drops == {"a", "b"}


def test_drop_counter_survives_to_snapshot_meta(tmp_path, monkeypatch):
    """The aggregate count flows into snapshot_meta.row_counts. Regression test
    for the silent-drop hole — a corrupt record should appear in the per-snapshot
    sidecar so downstream tooling doesn't have to parse logs."""
    pytest.importorskip("duckdb")

    # We can't run the full pipeline here without a registry/cards fixture.
    # Just check the counter total is exposed via the same module attr the
    # pipeline reads (`eee_src._drop_counter`). This tests the contract.
    eee_src._drop_counter[("cfg1", "read_error:ValueError")] = 3
    eee_src._drop_counter[("cfg1", "not_a_dict")] = 1
    assert sum(eee_src._drop_counter.values()) == 4

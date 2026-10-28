"""Stage cache: Stage A input fingerprint and the persisted datafix summary.

A `--from-stage` run does not re-read the raw records, so two things have
to come from the cache marker: proof that the cached records were loaded
under the same upstream revision, config selection and datafix overlay as
this run expects, and the account of what the overlay did to them.
"""
from __future__ import annotations

import json

import duckdb
import pytest

from eval_card_backend import datafix
from eval_card_backend.canonicalise.cache import STAGE_OUTPUTS, StageCache

SNAPSHOT = "2026-09-20T00:00:00Z"


def _inputs(**overrides) -> dict:
    base = {
        "eee_revision": "abc123",
        "configs": ["vals-ai"],
        "datafix_enabled": True,
        "datafix_fingerprint": datafix.fingerprint(),
    }
    base.update(overrides)
    return base


def _summary(**overrides) -> dict:
    base = {
        "enabled": True,
        "overlay_revision": datafix.OVERLAY_REVISION,
        "fingerprint": datafix.fingerprint(),
        "fixes": {"vals_ai_max_tokens": {"records_patched": 7, "results_patched": 9}},
    }
    base.update(overrides)
    return base


_DEFAULT_SUMMARY = object()


def _populated_cache(tmp_path, inputs, summary=_DEFAULT_SUMMARY) -> StageCache:
    """A cache holding Stage A's outputs, blessed the way a run blesses it."""
    con = duckdb.connect()
    for table in STAGE_OUTPUTS["A"]:
        con.execute(f"CREATE TABLE {table} AS SELECT 1 AS x")
    cache = StageCache(tmp_path, SNAPSHOT)
    cache.stage_a_inputs = inputs
    cache.datafix_summary = _summary() if summary is _DEFAULT_SUMMARY else summary
    cache.write_stage(con, "A")
    return cache


def _resuming_cache(tmp_path, inputs) -> StageCache:
    cache = StageCache(tmp_path, SNAPSHOT)
    cache.stage_a_inputs = inputs
    return cache


def test_matching_inputs_restore_normally(tmp_path):
    _populated_cache(tmp_path, _inputs())
    cache = _resuming_cache(tmp_path, _inputs())

    restored = cache.restore_through(duckdb.connect(), "A")

    assert set(restored) == set(STAGE_OUTPUTS["A"])


def test_a_flipped_kill_switch_refuses_to_resume(tmp_path):
    _populated_cache(tmp_path, _inputs())
    cache = _resuming_cache(tmp_path, _inputs(datafix_enabled=False))

    with pytest.raises(RuntimeError, match="datafix_enabled"):
        cache.restore_through(duckdb.connect(), "A")


def test_a_different_overlay_refuses_to_resume(tmp_path):
    _populated_cache(tmp_path, _inputs(datafix_fingerprint="0" * 64))
    cache = _resuming_cache(tmp_path, _inputs())

    with pytest.raises(RuntimeError, match="datafix_fingerprint"):
        cache.restore_through(duckdb.connect(), "A")


def test_a_different_upstream_revision_refuses_to_resume(tmp_path):
    _populated_cache(tmp_path, _inputs())
    cache = _resuming_cache(tmp_path, _inputs(eee_revision="def456"))

    with pytest.raises(RuntimeError, match="eee_revision"):
        cache.restore_through(duckdb.connect(), "A")


def test_a_run_that_restates_nothing_is_only_checked_on_what_it_knows(tmp_path):
    """A resume that re-selects no configs and restates no pin still knows
    which overlay it would use, and is checked on that alone. `None` is how
    it says it has nothing to state."""
    _populated_cache(tmp_path, _inputs())
    cache = _resuming_cache(
        tmp_path, _inputs(eee_revision=None, configs=None)
    )

    assert cache.restore_through(duckdb.connect(), "A")


def test_an_explicit_config_selection_must_match_the_cached_one(tmp_path):
    """A resume that names its own configs is asking for a selection the
    cached records may not hold."""
    _populated_cache(tmp_path, _inputs())
    cache = _resuming_cache(tmp_path, _inputs(configs=["hfopenllm_v2"]))

    with pytest.raises(RuntimeError, match="configs"):
        cache.restore_through(duckdb.connect(), "A")


def test_an_explicit_selection_that_resolves_to_nothing_is_a_mismatch(tmp_path):
    """An empty selection is a statement, not silence: the cached records
    cover a config, this run asked for none."""
    _populated_cache(tmp_path, _inputs())
    cache = _resuming_cache(tmp_path, _inputs(configs=[]))

    with pytest.raises(RuntimeError, match="configs"):
        cache.restore_through(duckdb.connect(), "A")


def test_a_cache_with_no_recorded_summary_refuses_to_resume(tmp_path):
    """Inputs recorded but no account of the overlay: nothing in this run can
    say what is in the cached records."""
    _populated_cache(tmp_path, _inputs(), summary=None)

    cache = _resuming_cache(tmp_path, _inputs())
    with pytest.raises(RuntimeError, match="no usable datafix summary"):
        cache.restore_through(duckdb.connect(), "A")


def test_a_cache_with_a_malformed_summary_refuses_to_resume(tmp_path):
    cache = _populated_cache(tmp_path, _inputs(), summary={"enabled": True})

    resuming = _resuming_cache(tmp_path, _inputs())
    with pytest.raises(RuntimeError, match="no usable datafix summary"):
        resuming.restore_through(duckdb.connect(), "A")
    assert json.loads((cache.dir / "_cache_schema.json").read_text())[
        "datafix_summary"
    ] == {"enabled": True}


def test_a_cache_with_no_recorded_inputs_refuses_to_resume(tmp_path):
    cache = _populated_cache(tmp_path, None)
    marker = json.loads((cache.dir / "_cache_schema.json").read_text())
    assert "stage_a_inputs" not in marker

    resuming = _resuming_cache(tmp_path, _inputs())
    with pytest.raises(RuntimeError, match="records no Stage A inputs"):
        resuming.restore_through(duckdb.connect(), "A")


def test_the_datafix_summary_survives_into_a_resumed_run(tmp_path):
    summary = _summary()
    _populated_cache(tmp_path, _inputs(), summary)

    cache = _resuming_cache(tmp_path, _inputs())
    restored = cache.cached_datafix_summary()
    accounting = datafix.DatafixSummary()
    accounting.restore(restored)

    assert restored == summary
    # The resumed run reports what is actually in eee_raw, not zeros.
    reported = accounting.as_dict(enabled=True)
    assert reported["fixes"]["vals_ai_max_tokens"]["results_patched"] == 9


def test_a_later_stage_rewrite_keeps_the_recorded_inputs(tmp_path):
    """Stage A ran once; every later stage's marker write has to carry its
    inputs and summary forward or the next resume loses them."""
    summary = {"enabled": True, "fixes": {}}
    cache = _populated_cache(tmp_path, _inputs(), summary)
    con = duckdb.connect()
    for table in STAGE_OUTPUTS["B"]:
        con.execute(f"CREATE TABLE {table} AS SELECT 1 AS x")
    cache.write_stage(con, "B")

    marker = json.loads((cache.dir / "_cache_schema.json").read_text())
    assert marker["stage_a_inputs"] == _inputs()
    assert marker["datafix_summary"] == summary

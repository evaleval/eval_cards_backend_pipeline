"""`scripts/derive_scoring_modes.py`: the part that decides agreement.

The derivation itself reads the leaderboard's dumps over the network. The
judgement it feeds is pure, and it is the part that can quietly say a stale
mapping is fine, so it is exercised here with synthetic derivations only.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "derive_scoring_modes.py"
    spec = importlib.util.spec_from_file_location("derive_scoring_modes", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["derive_scoring_modes"] = mod
    spec.loader.exec_module(mod)
    return mod


script = _load_script()

ON_FILE = {
    "bbh": {"mode": "log_prob"},
    "gpqa": {"mode": "log_prob"},
    "musr": {"mode": "log_prob"},
    "mmlu-pro": {"mode": "log_prob"},
    "ifeval": {"mode": "generative"},
    "math-level-5": {"mode": "generative"},
}


def _derivation(entries=None, dumps_read=25, unreadable=None):
    return script.Derivation(
        entries if entries is not None else {b: dict(e) for b, e in ON_FILE.items()},
        dumps_read,
        unreadable or {},
    )


def test_a_full_agreeing_sample_has_no_objections():
    assert script.objections(_derivation(), ON_FILE) == []


def test_a_run_that_downloaded_nothing_vouches_for_nothing():
    """The failure mode this exists for: derive nothing, compare nothing,
    report agreement. An empty sample is the weakest possible evidence, not
    the strongest."""
    found = script.objections(_derivation(entries={}, dumps_read=0), ON_FILE)
    assert found
    assert any("usable dump" in o for o in found)
    assert any("expected" in o for o in found)


def test_too_few_usable_dumps_is_an_objection():
    found = script.objections(_derivation(dumps_read=3), ON_FILE, min_dumps=10)
    assert found == ["only 3 usable dump(s), below the floor of 10"]


def test_a_missing_benchmark_is_an_objection():
    partial = {b: e for b, e in ON_FILE.items() if b != "musr"}
    found = script.objections(_derivation(entries=partial), ON_FILE)
    assert any("musr" in o for o in found)


def test_an_obsolete_entry_on_file_is_an_objection():
    """Nothing in the dumps produces it any more, so nothing justifies it."""
    stale = dict(ON_FILE, retired_benchmark={"mode": "generative"})
    found = script.objections(_derivation(), stale)
    assert any("retired_benchmark" in o for o in found)


def test_a_changed_mode_is_an_objection():
    flipped = dict(ON_FILE, ifeval={"mode": "log_prob"})
    found = script.objections(_derivation(), flipped)
    assert "ifeval: file says 'log_prob', dumps say 'generative'" in found


def test_an_unreadable_output_type_is_an_objection():
    """A recognised task reporting something new means the harness moved. It
    must not pass as agreement just because the modes still line up."""
    found = script.objections(
        _derivation(unreadable={"bbh": {"perplexity_v2"}}), ON_FILE
    )
    assert found == ["bbh: unreadable output_type(s) ['perplexity_v2']"]


def test_a_mangled_entry_on_file_does_not_read_as_a_mode():
    assert script.mode_on_file({"bbh": "log_prob"}, "bbh") is None
    assert script.mode_on_file({}, "bbh") is None


def test_read_mapping_survives_a_file_in_the_wrong_shape(tmp_path):
    path = tmp_path / "scoring_modes.yaml"
    path.write_text("sources: not-a-mapping\n")
    assert script.read_mapping(path) == {}


def test_the_shipped_mapping_holds_exactly_the_expected_benchmarks():
    on_file = script.read_mapping(script.MAPPING_PATH)
    assert set(on_file) == set(script.EXPECTED_BENCHMARKS)


@pytest.mark.parametrize("rotation", [0, 13, 37])
def test_sampling_does_not_depend_on_the_listing_order(rotation, monkeypatch):
    """`list_repo_files` promises no order, so the seeded sample has to be
    taken from a sorted listing or the reproducibility claim is empty."""
    files = [f"model-{i:03d}/results.json" for i in range(50)]
    rotated = files[rotation:] + files[:rotation]

    class _Api:
        def list_repo_files(self, repo, repo_type):
            return list(rotated)

    monkeypatch.setattr(script, "HfApi", _Api)
    assert script.sample_dumps(5, seed=7) == random.Random(7).sample(files, 5)

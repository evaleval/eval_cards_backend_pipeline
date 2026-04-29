"""Tests for ``scripts.verify_parity`` itself.

We can't trust the verifier's "OK: all surfaces match" output unless we
prove the verifier ACTUALLY catches divergences. These tests:

  1. Unit-test the comparison primitives (`_diff`, `_normalize_for_compare`,
     `_load_parquet_payloads`).
  2. End-to-end: run the verifier against a fixture pipeline output, then
     deliberately mutate one parquet row and confirm the verifier flags
     the divergence and exits non-zero.

If these tests pass, the verifier is sufficiently trustworthy to use as
the cross-repo parity gate.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import verify_parity


# ---------------------------------------------------------------------------
# Unit tests on the comparison primitives
# ---------------------------------------------------------------------------


def test_diff_returns_empty_for_identical_dicts():
    assert verify_parity._diff({"a": 1, "b": 2}, {"a": 1, "b": 2}) == []


def test_diff_detects_value_change():
    out = verify_parity._diff({"a": 1}, {"a": 2})
    assert len(out) == 1
    assert "~ a:" in out[0]


def test_diff_detects_missing_key_on_either_side():
    out_missing_left = verify_parity._diff({"a": 1}, {"a": 1, "b": 2})
    assert any(line.startswith("+ b") for line in out_missing_left)
    out_missing_right = verify_parity._diff({"a": 1, "b": 2}, {"a": 1})
    assert any(line.startswith("- b") for line in out_missing_right)


def test_diff_recurses_into_nested_dicts():
    out = verify_parity._diff(
        {"x": {"y": {"z": 1}}}, {"x": {"y": {"z": 2}}}
    )
    assert len(out) == 1
    assert "x.y.z" in out[0]


def test_diff_detects_list_length_mismatch():
    out = verify_parity._diff([1, 2, 3], [1, 2])
    assert len(out) == 1
    assert "length 3 vs 2" in out[0]


def test_diff_recurses_into_lists():
    out = verify_parity._diff([{"a": 1}, {"b": 2}], [{"a": 1}, {"b": 99}])
    assert any("[1].b" in line for line in out)


def test_normalize_for_compare_strips_dict_nones():
    """`None` keys in a dict are dropped so that 'TS missing vs parity None'
    doesn't fire as a difference. Documented quirk; tests confirm it."""
    out = verify_parity._normalize_for_compare({"a": 1, "b": None})
    assert out == {"a": 1.0}


def test_normalize_for_compare_does_not_strip_list_nones():
    """List elements stay positionally — `[1, None]` is not the same as `[1]`."""
    out = verify_parity._normalize_for_compare([1, None, 2])
    assert out == [1.0, None, 2.0]


def test_normalize_for_compare_coerces_ints_to_floats():
    """JSON has only floats; parquet returns Python ints. Coerce so
    `1 == 1.0` doesn't fire as a divergence."""
    assert verify_parity._normalize_for_compare(1) == 1.0
    assert verify_parity._normalize_for_compare(True) is True
    assert verify_parity._normalize_for_compare(False) is False


def test_normalize_for_compare_recurses_through_nested():
    out = verify_parity._normalize_for_compare(
        {"a": [{"b": 1, "c": None}, 2]}
    )
    assert out == {"a": [{"b": 1.0}, 2.0]}


# ---------------------------------------------------------------------------
# `_load_parquet_payloads` — keying logic
# ---------------------------------------------------------------------------


def test_load_parquet_payloads_keys_eval_surfaces_by_eval_summary_id(tmp_path):
    """`eval_list` and `eval_summaries` key on `eval_summary_id`; everything
    else keys on `model_route_id`."""
    pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("datasets")
    parquet_dir = tmp_path / "duckdb" / "v1"
    parquet_dir.mkdir(parents=True)

    from datasets import Dataset

    # Fabricate two minimal parquet files in the schema `_load_parquet_payloads`
    # expects (only the columns it actually reads need to be populated).
    Dataset.from_dict(
        {
            "model_route_id": [None],
            "eval_summary_id": ["EID-1"],
            "payload_json": ['{"evaluation_id":"EID-1"}'],
        }
    ).to_parquet(str(parquet_dir / "eval_summaries.parquet"))
    Dataset.from_dict(
        {
            "model_route_id": ["RT-1"],
            "eval_summary_id": [None],
            "payload_json": ['{"route_id":"RT-1"}'],
        }
    ).to_parquet(str(parquet_dir / "model_cards.parquet"))

    eval_loaded = verify_parity._load_parquet_payloads(parquet_dir, "eval_summaries")
    assert "EID-1" in eval_loaded

    card_loaded = verify_parity._load_parquet_payloads(parquet_dir, "model_cards")
    assert "RT-1" in card_loaded


def test_load_parquet_payloads_returns_empty_when_file_missing(tmp_path):
    """Missing parquet file returns empty dict, not error."""
    out = verify_parity._load_parquet_payloads(tmp_path, "doesnotexist")
    assert out == {}


# ---------------------------------------------------------------------------
# End-to-end: verifier exits 0 on match, non-zero on injected divergence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_pipeline_output(tmp_path_factory) -> Path:
    """Run the pipeline once against synthetic fixtures so the verifier
    has real parquet output to diff against."""
    import os

    from tests.conftest import _write_fixtures
    from scripts import pipeline as pipeline_module

    fixture_root = tmp_path_factory.mktemp("verify_parity_e2e")
    eee_root = fixture_root / "eee_dataset"
    abc_root = fixture_root / "abc_metadata"
    output_dir = fixture_root / "output"
    _write_fixtures(eee_root, abc_root)

    saved_output = pipeline_module.OUTPUT_DIR
    saved_argv = list(sys.argv)
    saved_env = {
        key: os.environ.get(key)
        for key in (
            "EEE_LOCAL_DATASET_DIR",
            "BENCHMARK_METADATA_LOCAL_DIR",
            "CONFIGS",
            "HF_TOKEN",
        )
    }
    try:
        pipeline_module.OUTPUT_DIR = output_dir
        os.environ["EEE_LOCAL_DATASET_DIR"] = str(eee_root)
        os.environ["BENCHMARK_METADATA_LOCAL_DIR"] = str(abc_root)
        os.environ["CONFIGS"] = ",".join(
            sorted(p.name for p in (eee_root / "data").iterdir() if p.is_dir())
        )
        os.environ.pop("HF_TOKEN", None)
        sys.argv = ["pipeline.py", "--dry-run"]
        rc = pipeline_module.main()
        assert rc == 0
    finally:
        pipeline_module.OUTPUT_DIR = saved_output
        sys.argv = saved_argv
        for key, val in saved_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
    return output_dir


def _run_verifier(pipeline_output: Path) -> tuple[int, str, str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.verify_parity",
        "--pipeline-output",
        str(pipeline_output),
        "--general-eval-card",
        "/Users/jchim/projects/evaleval/general-eval-card",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_verifier_exits_zero_on_pristine_pipeline_output(fixture_pipeline_output):
    """Sanity check — the unmodified fixture pipeline output passes parity."""
    if shutil.which("pnpm") is None:
        pytest.skip("pnpm not installed")
    rc, stdout, _ = _run_verifier(fixture_pipeline_output)
    assert rc == 0, f"verifier failed unexpectedly:\n{stdout}"
    assert "OK: all surfaces match" in stdout


def _mutate_one_parquet_payload(parquet_path: Path, surface: str) -> None:
    """Rewrite the parquet so one row's payload_json carries a deliberate
    divergence the verifier should catch.

    Uses pyarrow (transitive of `datasets`) — read the table, mutate the
    first row's payload, write it back. No SQL engine in the loop.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path))
    if table.num_rows == 0:
        raise RuntimeError(f"no rows in {parquet_path}")
    payloads = table.column("payload_json").to_pylist()
    payload = json.loads(payloads[0])
    # Corrupt one specific TS-shape field in row 0 — `evaluator_count`
    # is hardcoded to 0 by `hfModelCardToEvaluationCardData`, so any
    # other value will diff.
    payload["evaluator_count"] = 9999
    payloads[0] = json.dumps(payload, ensure_ascii=False)
    new_payload_col = pa.array(payloads, type=table.schema.field("payload_json").type)
    new_table = table.set_column(
        table.column_names.index("payload_json"),
        table.schema.field("payload_json"),
        new_payload_col,
    )
    pq.write_table(new_table, str(parquet_path))


def test_verifier_catches_injected_divergence(fixture_pipeline_output, tmp_path):
    """Negative test — corrupt one model_cards payload, confirm the
    verifier flags the field divergence and exits non-zero. Without
    this, "OK" results give us no signal that the verifier is wired to
    the right comparison primitives."""
    if shutil.which("pnpm") is None:
        pytest.skip("pnpm not installed")
    pytest.importorskip("pyarrow.parquet")

    # Make a writable copy of the pipeline output so we don't poison the
    # session fixture for other tests.
    mutated_root = tmp_path / "mutated_output"
    shutil.copytree(fixture_pipeline_output, mutated_root)
    parquet_path = mutated_root / "duckdb" / "v1" / "model_cards.parquet"
    _mutate_one_parquet_payload(parquet_path, "model_cards")

    rc, stdout, stderr = _run_verifier(mutated_root)
    assert rc != 0, f"verifier returned 0 on a known-divergent input:\n{stdout}\n{stderr}"
    # The exact divergence we injected — `evaluator_count: 9999` vs `0` —
    # must appear in the output so we know the verifier is actually
    # comparing this field, not silently accepting it.
    assert "evaluator_count" in stdout, f"divergence field not surfaced:\n{stdout}"
    assert "9999" in stdout, f"injected value not surfaced:\n{stdout}"

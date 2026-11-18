"""Two builds of the same corpus must produce byte-identical warehouses.

Unordered aggregations (`ANY_VALUE`, `ARRAY_AGG` without ORDER BY, `COPY`
of an unsorted relation, global-RNG dependence) make a rebuild differ
from the previous one without any input change, which defeats snapshot
diffing and the sync workflow's no-op detection. The existing tests cover
individual sites; this one bakes the whole fixture corpus a second time,
with the same snapshot id, and compares every parquet and JSON file of the
two warehouses by sha256.

The second build runs DuckDB on one thread while the first uses the
default worker count, so a tie that scan order happens to settle the same
way twice on one machine is given a different scan order here.

The comparison itself lives in `tests/warehouse_diff.py`.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.fixture_corpus import build_fixture_corpus
from tests.warehouse_diff import output_files, warehouse_differences


@pytest.fixture(scope="module")
def second_build(tmp_path_factory, _taxonomy_seed_stub, _collections_stub):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CANONICALISE_DUCKDB_THREADS", "1")
        return build_fixture_corpus(
            tmp_path_factory.mktemp("fixture_corpus_rebuild"),
            seed_dir=_taxonomy_seed_stub,
            collections_dir=_collections_stub,
        )


def test_duckdb_thread_count_follows_the_environment(monkeypatch):
    from eval_card_backend.canonicalise import pipeline

    monkeypatch.setenv("CANONICALISE_DUCKDB_THREADS", "1")
    con, db_dir = pipeline._connect_duckdb()
    try:
        assert con.execute("SELECT current_setting('threads')").fetchone()[0] == 1
    finally:
        pipeline._close_duckdb(con, db_dir)


def test_pipeline_connection_ignores_the_machine_time_zone():
    """The bake's DuckDB session is UTC on a machine in any zone, so SQL
    that reads the session zone gives the same answer everywhere. DuckDB
    reads `TZ` once per process, hence the child process."""
    probe = (
        "from eval_card_backend.canonicalise import pipeline\n"
        "con, db_dir = pipeline._connect_duckdb()\n"
        "print(con.execute(\"SELECT CAST(to_timestamp(0) AS TIMESTAMP)\").fetchone()[0])\n"
        "pipeline._close_duckdb(con, db_dir)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env={**os.environ, "TZ": "Asia/Kolkata"},
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "1970-01-01 00:00:00"


def test_rebuilding_the_corpus_is_byte_identical(fixture_corpus_build, second_build):
    first = output_files(fixture_corpus_build.out_dir)
    assert {"fact_results.parquet", "manifest.json", "snapshot_meta.json"} <= first.keys(), (
        f"the first build did not emit a full warehouse: {sorted(first)}"
    )
    problems = warehouse_differences(fixture_corpus_build.out_dir, second_build.out_dir)
    assert not problems, (
        f"{len(problems)} of {len(first)} warehouse files differ between two "
        f"builds of the same snapshot:\n  " + "\n  ".join(problems)
    )

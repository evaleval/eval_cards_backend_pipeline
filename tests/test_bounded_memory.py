"""Bounded-memory pipeline: per-config Stage A batches, file-backed DuckDB
with a memory limit, and release of tables no later stage reads."""
from __future__ import annotations

import json

import duckdb
import pyarrow as pa
import pytest

from eval_card_backend.canonicalise import pipeline, stages
from eval_card_backend.sources import eee as eee_src
from tests.eee_layout import write_eee_datastore
from tests.test_stage_a_drops import _conformant_record


@pytest.fixture(autouse=True)
def _reset():
    eee_src.reset_drop_counter()
    yield
    eee_src.reset_drop_counter()


def _corpus(tmp_path, monkeypatch):
    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        ("a", "a1.json", json.dumps(_conformant_record("ev_a1"))),
        ("a", "a2.json", json.dumps(_conformant_record("ev_a2"))),
        ("b", "b1.json", json.dumps(_conformant_record("ev_b1"))),
        ("c", "bad.json", "nope"),
    ])
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    return eee_root


def test_batched_load_matches_single_table_load(tmp_path, monkeypatch):
    eee_root = _corpus(tmp_path, monkeypatch)
    cfgs = ["a", "b", "c"]

    con_one = duckdb.connect()
    n_one = stages.stage_a_load_eee(
        con_one, eee_src.load_arrow_table(eee_root, cfgs, hf_token=None)
    )
    con_batched = duckdb.connect()
    batches = list(eee_src.iter_arrow_tables(eee_root, cfgs, hf_token=None))
    n_batched = stages.stage_a_load_eee(con_batched, iter(batches))

    assert [cfg for cfg, _ in batches] == ["a", "b"]  # c dropped entirely
    assert n_one == n_batched == 3
    q = "SELECT evaluation_id, source_config, _record_path FROM eee_raw ORDER BY ALL"
    assert con_one.execute(q).fetchall() == con_batched.execute(q).fetchall()
    assert con_one.execute("DESCRIBE eee_raw").fetchall() == \
        con_batched.execute("DESCRIBE eee_raw").fetchall()


def test_empty_batches_create_an_empty_typed_table(tmp_path, monkeypatch):
    eee_root = _corpus(tmp_path, monkeypatch)
    con = duckdb.connect()
    n = stages.stage_a_load_eee(
        con, eee_src.iter_arrow_tables(eee_root, ["c"], hf_token=None)
    )
    assert n == 0
    cols = {r[0] for r in con.execute("DESCRIBE eee_raw").fetchall()}
    assert {"evaluation_id", "source_config", "_record_path"} <= cols


def test_connect_duckdb_is_file_backed_and_honours_memory_limit(monkeypatch):
    monkeypatch.setenv("CANONICALISE_MEMORY_LIMIT", "512MB")
    con, db_dir = pipeline._connect_duckdb()
    try:
        assert (db_dir / "canonicalise.duckdb").exists()
        limit = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        assert limit.replace(" ", "").upper() in ("512.0MIB", "512MB", "512.0MB", "488.2MIB")
        tmp = con.execute("SELECT current_setting('temp_directory')").fetchone()[0]
        assert tmp.startswith(str(db_dir))
    finally:
        pipeline._close_duckdb(con, db_dir)
    assert not db_dir.exists()


def test_release_tables_drops_only_the_listed_ones():
    con = duckdb.connect()
    con.execute("CREATE TABLE keep AS SELECT 1 AS x")
    con.execute("CREATE TABLE gone AS SELECT 1 AS x")
    pipeline._release_tables(con, ("gone", "never_existed"))
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert names == {"keep"}


def test_release_schedule_covers_every_fact_grain_intermediate():
    released = {t for ts in pipeline._RELEASE_AFTER.values() for t in ts}
    assert released == {
        "eee_raw", "results_exploded", "results_resolved",
        "fact_results_staging", "fact_results_signaled",
        "fact_results_grouped", "fact_results_grouped_annotated",
    }
    # Every released table is a cached stage output, so --from-stage can
    # still restore it after the drop.
    from eval_card_backend.canonicalise.cache import STAGE_OUTPUTS
    cached = {t for ts in STAGE_OUTPUTS.values() for t in ts}
    assert released - cached == {
        "fact_results_grouped", "fact_results_grouped_annotated",
    }


def test_recover_stage_e_stats_without_tables_is_zero():
    con = duckdb.connect()
    s = pipeline._recover_stage_e_stats(con)
    assert (s.pre, s.post, s.n_dropped_dedup) == (0, 0, 0)
    assert pipeline._configs_from_eee_raw(con) == []


def test_single_table_input_still_accepted():
    con = duckdb.connect()
    _, schema = eee_src._arrow_table_schema()
    rec = {f.name: None for f in schema}
    rec.update({"source_config": "x", "_record_path": "data/x/1.json"})
    t = pa.Table.from_pylist([rec], schema=schema)
    assert stages.stage_a_load_eee(con, t) == 1


def test_record_batches_never_split_a_record():
    con = duckdb.connect()
    # Interleaved like the explode's lateral range(): r1,r2,r3,r1,r2,r1
    con.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES "
        "('r1', 1), ('r2', 2), ('r3', 3), ('r1', 4), ('r2', 5), ('r1', 6)"
        ") v(source_record_path, x)"
    )
    m, n = stages.record_batches(con, "t", "source_record_path", target_rows=3)
    rows = con.execute(f"SELECT k, b FROM {m} ORDER BY b, k").fetchall()
    # r1 has 3 rows and fills batch 0; r2 (2) and r3 (1) share batch 1.
    assert (n, rows) == (2, [("r1", 0), ("r2", 1), ("r3", 1)])
    m, n = stages.record_batches(con, "t", "source_record_path", target_rows=100)
    assert n == 1 and con.execute(f"SELECT COUNT(DISTINCT b) FROM {m}").fetchone()[0] == 1
    con.execute("CREATE TABLE empty AS SELECT * FROM t WHERE FALSE")
    assert stages.record_batches(con, "empty", "source_record_path")[1] == 0


def test_record_batches_refuse_null_keys():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES ('a', 1), (NULL, 2)) v(k, x)")
    with pytest.raises(RuntimeError, match="NULL"):
        stages.record_batches(con, "t", "k", target_rows=1)


def test_materialise_in_batches_matches_single_statement():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE src AS SELECT 'rec' || (i % 4)::VARCHAR AS source_record_path, "
        "i AS x FROM range(10) r(i)"
    )
    stages.materialise_in_batches(
        con, "out",
        lambda pred: f"SELECT source_record_path, x * 2 AS y FROM src WHERE {pred}",
        source_table="src", key_col="source_record_path", target_rows=4,
    )
    expect = con.execute(
        "SELECT source_record_path, x * 2 FROM src ORDER BY source_record_path, x"
    ).fetchall()
    assert con.execute("SELECT * FROM out ORDER BY source_record_path, y").fetchall() == expect
    assert con.execute("SELECT COUNT(*) FROM out").fetchone()[0] == 10
    assert not [r for r in con.execute("SHOW TABLES").fetchall() if r[0].startswith("_batch_map")]


def test_stage_a_chunks_a_large_config(tmp_path, monkeypatch):
    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        ("big", f"r{i}.json", json.dumps(_conformant_record(f"ev_{i}"))) for i in range(7)
    ])
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.setenv("CANONICALISE_BATCH_ROWS", "3")
    batches = list(eee_src.iter_arrow_tables(eee_root, ["big"], hf_token=None))
    assert [(c, t.num_rows) for c, t in batches] == [("big", 3), ("big", 3), ("big", 1)]
    con = duckdb.connect()
    assert stages.stage_a_load_eee(con, iter(batches)) == 7
    ids = [r[0] for r in con.execute("SELECT evaluation_id FROM eee_raw ORDER BY rowid").fetchall()]
    assert ids == sorted(ids) and len(set(ids)) == 7

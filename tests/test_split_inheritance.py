"""The dataset split: inherited by a total from its record's parts (Stage D),
then part of the Stage J cell grain.

A split is a property of the RUN. Sources state it on the per-task rows and
leave the benchmark total bare: Apertus stamps all 171 MMLU subject rows
`validation` and none of its three extraction totals. Putting `split` into
the cell grain without reading that evidence puts a source's own total and
the parts it was computed from in different cells, and the parts-only cell
wins the page with a number nobody published. So Stage D first gives a bare
total (`observation_role = 'whole'`) the one split its record's part rows
agree on, and only then does `split` join the grain, where two wholes on two
splits become two cells that never pool.

Everything here runs the real pipeline over the `fixtures_splits` EEE
fixture (six records, one model), so `split`, `split_source`, `is_part` and
`observation_role` come out of Stages C and D and the cell shape out of
Stage J.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    """Run the pipeline over `fixtures_splits` once; return the snapshot dir."""
    pytest.importorskip("duckdb")
    import os

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    tmp = tmp_path_factory.mktemp("splits")
    prior = {k: os.environ.get(k) for k in (
        "EEE_LOCAL_DATASET_DIR", "BENCHMARK_METADATA_LOCAL_DIR",
        "EEE_REFRESH_SNAPSHOT", "BENCHMARK_METADATA_REFRESH",
    )}
    os.environ["EEE_LOCAL_DATASET_DIR"] = str(FIXTURES / "eee")
    os.environ["BENCHMARK_METADATA_LOCAL_DIR"] = str(FIXTURES / "auto_benchmarkcards")
    os.environ.pop("EEE_REFRESH_SNAPSHOT", None)
    os.environ.pop("BENCHMARK_METADATA_REFRESH", None)
    try:
        out = pipeline.run(
            Settings.from_env(),
            configs=["fixtures_splits"],
            snapshot_id="2026-04-30T00:00:00Z",
            warehouse_dir=str(tmp / "warehouse"),
            registry_local_dir=str(FIXTURES / "entity_registry"),
            cache_root=str(tmp / "cache"),
        )
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert out is not None
    return Path(out)


@pytest.fixture(scope="module")
def con(snapshot):
    c = duckdb.connect()
    for table in ("fact_results", "eval_results_view"):
        c.execute(
            f"CREATE VIEW {table} AS SELECT * FROM "
            f"read_parquet('{snapshot}/{table}.parquet')"
        )
    return c


def _facts(con, benchmark):
    return con.execute(
        "SELECT evaluation_name_tail, observation_role, split, split_source "
        "FROM (SELECT split_part(benchmark_raw, ' ', -1) AS evaluation_name_tail, "
        "             observation_role, split, split_source, fact_id "
        "      FROM fact_results WHERE benchmark_key = ?) "
        "ORDER BY observation_role, evaluation_name_tail, fact_id",
        [benchmark],
    ).fetchall()


def _cells(con, benchmark):
    return con.execute(
        "SELECT split, score, value_level, fact_row_count, is_headline "
        "FROM eval_results_view WHERE benchmark_id = ? "
        "ORDER BY COALESCE(split, '')",
        [benchmark],
    ).fetchall()


# ---------------------------------------------------------------------------
# Stage D — inheritance
# ---------------------------------------------------------------------------


def test_rows_that_agree_give_the_bare_rows_their_split(con):
    """(a) Two category rows on `test`, a bare `.overall` and a bare third
    category: one record on one benchmark is one run, so the total AND the
    bare category ran on `test` too. Stated rows keep `stated`; the bare
    ones say `inherited`."""
    rows = con.execute(
        "SELECT observation_role, split, split_source, COUNT(*) "
        "FROM fact_results WHERE benchmark_key = 'omni-math' "
        "GROUP BY 1, 2, 3 ORDER BY 1, 3"
    ).fetchall()
    assert rows == [
        ("part", "test", "inherited", 1),
        ("part", "test", "stated", 2),
        ("whole", "test", "inherited", 1),
    ]


def test_parts_that_disagree_leave_the_total_unspecified(con):
    """(b) One part on `test`, one on `validation`: the record does not say
    which split its total ran on, so the pipeline does not guess."""
    rows = con.execute(
        "SELECT observation_role, split, split_source "
        "FROM fact_results WHERE benchmark_key = 'lexam' ORDER BY 1, 2"
    ).fetchall()
    assert rows == [
        ("part", "test", "stated"),
        ("part", "validation", "stated"),
        ("whole", None, None),
    ]


def test_parts_that_state_nothing_give_nothing(con):
    """(c) Parts with no split are no evidence: the total stays bare and
    nothing is labelled inherited."""
    rows = con.execute(
        "SELECT DISTINCT split, split_source "
        "FROM fact_results WHERE benchmark_key = 'cnn-dailymail'"
    ).fetchall()
    assert rows == [(None, None)]


def test_a_stated_split_on_the_total_is_never_overridden(con):
    """(e) The total says `test`, its one part says `validation`. The total's
    own statement stands, labelled `stated`; nothing is inherited."""
    rows = con.execute(
        "SELECT observation_role, split, split_source "
        "FROM fact_results WHERE benchmark_key = 'disinfo-bench' ORDER BY 1"
    ).fetchall()
    assert rows == [
        ("part", "validation", "stated"),
        ("whole", "test", "stated"),
    ]


def test_inheritance_is_scoped_to_the_record_and_benchmark(con):
    """omni-math's `test` parts must not reach lexam's or cnn-dailymail's
    totals, which live in other records: the join is on (record, benchmark),
    not on the source."""
    assert con.execute(
        "SELECT DISTINCT benchmark_key FROM fact_results "
        "WHERE split_source = 'inherited'"
    ).fetchall() == [("omni-math",)]


def test_rows_in_registry_slice_children_are_evidence_for_the_parent():
    """A fact that resolved to a slice child is a part of the parent, so a
    suite total beside its children's rows (TruthfulQA-multilingual's 31
    languages) reads their split. Keyed by the child's parent, never by the
    child itself; a row that is a part of its own benchmark still keys there.
    The fixture registry has no parent edges, so this pins the evidence SQL
    directly."""
    from eval_card_backend.canonicalise import stages

    c = duckdb.connect()
    c.execute(
        """
        CREATE TABLE flat AS SELECT * FROM (VALUES
            -- record r1: suite total (bare) + two child-benchmark rows on test
            ('r1', 'p1', 'suite',   NULL,    FALSE, 'whole', NULL),
            ('r1', 'p1', 'suite-a', 'suite', FALSE, 'whole', 'test'),
            ('r1', 'p1', 'suite-b', 'suite', FALSE, 'whole', 'test'),
            -- record r2: the children disagree
            ('r2', 'p2', 'suite',   NULL,    FALSE, 'whole', NULL),
            ('r2', 'p2', 'suite-a', 'suite', FALSE, 'whole', 'test'),
            ('r2', 'p2', 'suite-b', 'suite', FALSE, 'whole', 'validation'),
            -- record r3: a plain part row keys on its own benchmark
            ('r3', 'p3', 'bench',   NULL,    TRUE,  'part',  'train')
        ) t(evaluation_id, source_record_path, benchmark_key,
            parent_benchmark_id, is_part, observation_role, split)
        """
    )
    rows = c.execute(
        f"SELECT evaluation_id, benchmark_key, _n_part_splits, _n_parts, "
        f"_part_split FROM ({stages._part_splits_sql('flat')}) ORDER BY 1, 2"
    ).fetchall()
    # every row keys on its own benchmark; a child row keys on its parent as
    # well, as a part of it
    assert rows == [
        ("r1", "suite", 1, 2, "test"),
        ("r1", "suite-a", 1, 0, "test"),
        ("r1", "suite-b", 1, 0, "test"),
        ("r2", "suite", 2, 2, "validation"),
        ("r2", "suite-a", 1, 0, "test"),
        ("r2", "suite-b", 1, 0, "validation"),
        ("r3", "bench", 1, 1, "train"),
    ]


def test_disagreeing_record_is_named_in_the_log(snapshot, caplog):
    """One INFO line per record whose parts disagree, naming the record and
    the splits; a summary line for the inherited totals; a summary count for
    records whose parts state no split."""
    from eval_card_backend.canonicalise import stages

    c = duckdb.connect()
    c.execute(
        "CREATE TABLE fact_results_staging AS SELECT * FROM "
        f"read_parquet('{snapshot}/fact_results.parquet')"
    )
    with caplog.at_level(logging.INFO, logger="eval_card_backend.canonicalise.stages"):
        stages._log_split_inheritance(c)
    text = caplog.text
    assert "2 row(s) across 1 record(s) inherited the split" in text
    assert "record ev_split_disagree / lexam: rows disagree on the split" in text
    assert "['test', 'validation']" in text
    assert "1 record(s) have bare rows beside part rows that state no split" in text
    # the agreeing and the stated records are not reported as problems
    assert "ev_split_agree" not in text
    assert "ev_split_stated" not in text


# ---------------------------------------------------------------------------
# Stage J — split in the cell grain
# ---------------------------------------------------------------------------


def test_total_and_its_parts_share_one_cell_after_inheritance(con):
    """(a) The reason the inheritance runs first: with the total and the bare
    category both on `test` beside the stated `test` parts, the cell is one
    `whole` reading of 0.5 and the parts stay out of the value. Without it the
    total would sit alone in one NULL-split cell, the bare category in a
    second, and the stated parts in a third."""
    cells = _cells(con, "omni-math")
    assert len(cells) == 1
    split, score, level, n, headline = cells[0]
    assert split == "test"
    assert abs(score - 0.5) < 1e-9
    assert level == "whole"
    assert n == 1
    assert headline


def test_two_wholes_on_two_splits_are_two_cells(con):
    """(d) Two records score the same benchmark on `test` and `validation`.
    They are two measurements: two rows, neither pooled into the other, and
    exactly one of them heads the page."""
    cells = _cells(con, "wildbench")
    assert [(c[0], round(c[1], 2), c[2], c[3]) for c in cells] == [
        ("test", 0.71, "whole", 1),
        ("validation", 0.65, "whole", 1),
    ]
    assert sum(1 for c in cells if c[4]) == 1


def test_disagreeing_parts_split_into_their_own_cells(con):
    """(b) The unspecified total keeps its own cell and heads the page (the
    publisher's number); each part sits in the cell of the split it states."""
    cells = _cells(con, "lexam")
    assert [(c[0], round(c[1], 2), c[2]) for c in cells] == [
        (None, 0.7, "whole"),
        ("test", 0.6, "single"),
        ("validation", 0.8, "single"),
    ]
    assert [c[4] for c in cells] == [True, False, False]


def test_split_is_on_the_view_row(con):
    assert "split" in {
        r[1] for r in con.execute("PRAGMA table_info('eval_results_view')").fetchall()
    }

"""Whole-vs-part: what a row measured, and what Stage J does with it.

A cell's number must come from rows that measured the BENCHMARK, never from
rows that measured a part of it. MMLU's three answer-extraction totals and its
183 subject rows used to median together and publish 0.750 where the source
reported 0.704.

The classification comes from RESOLUTION, not from `slice_key`. `slice_key` is
a display axis minted whenever a benchmark sees more than one raw spelling, so
it lands on the total as readily as on the parts — all 186 Apertus MMLU rows
carry one, the three totals included. The structured benchmark resolver instead
reports the subset it found and whether that subset is a real part or just a
longer way of spelling the benchmark, and `is_part` follows that.

Resolution has a third answer, and `observation_role` keeps it: `unknown`, the
row whose name the structured path declines to read. Anything flat or spaced
goes there, and it is most of HELM. Such a row still resolves — through the
plain alias index, which states that two spellings name one benchmark and
states nothing about whether the row measured the whole of it. Those cells
pool exactly as they always have; they are labelled `pooled` (or `single`)
rather than `whole`, so the page stops claiming a verification that never
happened.

Everything here runs the real pipeline over the `fixtures_tasks` EEE fixture,
so `is_part` and `observation_role` come out of Stage C and the tests pin the
resolution path that produces them as well as the rule that consumes them.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    """Run the pipeline over `fixtures_tasks` once; return the snapshot dir."""
    pytest.importorskip("duckdb")
    import os

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    tmp = tmp_path_factory.mktemp("tasks")
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
            configs=["fixtures_tasks"],
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


def _cell(con, benchmark):
    return con.execute(
        "SELECT score, value_level, fact_row_count FROM eval_results_view "
        "WHERE benchmark_id = ?", [benchmark],
    ).fetchone()


# ---------------------------------------------------------------------------
# Stage C — the classification itself
# ---------------------------------------------------------------------------


def test_a_dotted_subset_is_a_part(con):
    """`omni-math.category.<x>` resolves to `omni-math` with a real subset, so
    each row measured one category, not the benchmark."""
    rows = con.execute(
        "SELECT DISTINCT benchmark_subset, is_part, observation_role "
        "FROM fact_results WHERE benchmark_key = 'omni-math' ORDER BY 1"
    ).fetchall()
    assert rows == [
        ("category alpha", True, "part"), ("category beta", True, "part"),
        ("category delta", True, "part"), ("category gamma", True, "part"),
    ]


def test_a_flat_exact_alias_is_a_whole_and_a_promoted_flat_name_is_not(con):
    """`lexam` carries no structure to read, so the structured path declines
    it; the plain alias index then answers with a BYTE-EXACT hit on `lexam`
    itself, and that is the registry stating this spelling IS the benchmark,
    so the row is a verified whole. `xstest`'s three spellings, by contrast,
    reached their canonical through slice promotion, not an exact alias, so
    nothing verified what they measured and they stay `unknown`."""
    assert con.execute(
        "SELECT DISTINCT benchmark_subset, is_part, observation_role, "
        "       benchmark_resolution_strategy "
        "FROM fact_results WHERE benchmark_key = 'lexam'"
    ).fetchall() == [(None, False, "whole", "exact")]
    assert con.execute(
        "SELECT DISTINCT is_part, observation_role "
        "FROM fact_results WHERE benchmark_key = 'xstest'"
    ).fetchall() == [(False, "unknown")]


# ---------------------------------------------------------------------------
# Stage J — what the cell does with it
# ---------------------------------------------------------------------------


def test_repeated_whole_observations_pool_by_median(con):
    """(a) The llm-stats / Artificial Analysis shape: the same benchmark
    reported in three separate records. Repeated measurements of one quantity,
    so the middle one represents them. The number is what it always was; the
    label says `whole`, because each flat name is the registry's own exact
    spelling of the benchmark."""
    score, level, n = _cell(con, "lexam")
    assert abs(score - 0.78) < 1e-9
    assert level == "whole"
    assert n == 3


def test_settings_arms_of_one_benchmark_pool(con):
    """(a, continued) `…cot.few_shot` and `…cot.zero_shot` pool into one
    reading — the case `vals-ai/gpqa-diamond` used to blank on. Neither arm
    reports a subset, so neither is held out; neither is a structured match
    either, so the pool is labelled as one."""
    score, level, n = _cell(con, "simplesafetytests")
    assert abs(score - 0.50) < 1e-9
    assert level == "pooled"
    assert n == 2


def test_parts_only_and_no_registry_set_pools_the_parts(con):
    """(d) Four named categories, no whole, and no registry task set: the parts
    pool as the pipeline always pooled them, labelled so the page says what the
    number is. No page that had a value loses it."""
    score, level, n = _cell(con, "omni-math")
    assert abs(score - 0.25) < 1e-9
    assert level == "pooled_parts"
    assert n == 4


def test_three_spellings_of_one_benchmark_are_none_of_them_parts(con):
    """Case and separators do not change what a row measured: no spelling of
    `xstest.refusal set` reports a subset, so none is held out of the value.
    None is a verified whole either, so the three pool under a pooling label."""
    assert con.execute(
        "SELECT DISTINCT is_part FROM fact_results WHERE benchmark_key = 'xstest'"
    ).fetchall() == [(False,)]
    score, level, n = _cell(con, "xstest")
    assert abs(score - 0.70) < 1e-9
    assert level == "pooled"
    assert n == 3


def test_a_total_beside_unresolvable_parts_is_never_called_a_whole(con):
    """(f) `harmbench.category.<x>` does not resolve at segment level, so the
    category rows report no subset and the pipeline cannot hold them out of
    the benchmark's own `.overall` total: the cell still pools all four rows
    and reads 0.25 rather than the submitted 0.99.

    What it must NOT do is call that a whole-benchmark reading. Every row here
    is `unknown` — nothing was verified — so the cell is labelled a pooling of
    four rows, and a reader can see that the number is not the total the source
    published. Getting to 0.99 is registry work (children for the categories,
    or an alias that makes the subset resolvable), not a producer heuristic;
    this test pins the honest label in the meantime.
    """
    assert con.execute(
        "SELECT DISTINCT observation_role FROM fact_results "
        "WHERE benchmark_key = 'harmbench'"
    ).fetchall() == [("unknown",)]
    score, level, n = _cell(con, "harmbench")
    assert level != "whole"
    assert level == "pooled"
    assert n == 4
    assert abs(score - 0.25) < 1e-9

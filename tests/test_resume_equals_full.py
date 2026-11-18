"""A build resumed from the stage cache must equal a full build.

`--from-stage X` restores the cached outputs of every earlier stage and
runs the rest. Whatever a stage needs from an earlier one therefore has to
be in a cached table: a row a later stage wrote into an earlier stage's
table, or a count kept in a module global, is there in a full build and
gone in a resumed one.

For every resume point this runs the corpus `--to-stage <previous>` and
then `--from-stage <point>` over the same cache and warehouse, and compares
each parquet and JSON file with the full build by sha256.

The second check is the rule behind the first, stated on the tables: after
each stage of a full build, every cached table still in the connection
holds exactly the rows its cache file holds. It fails on a write the
corpus happens not to carry through to a published file.

Corpus: `tests/fixtures/` plus two slice-child benchmarks whose parent has
no registry row, so Stage G has a parent benchmark to synthesise.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise.cache import STAGE_ORDER, STAGE_OUTPUTS, StageCache
from tests.fixture_corpus import (
    SNAPSHOT_ID,
    build_fixture_corpus,
    registry_with_slice_children,
)
from tests.warehouse_diff import warehouse_differences

SYNTHESISED_PARENT = "mmlu-subjects"

RESUME_POINTS = [
    (STAGE_ORDER[i - 1], STAGE_ORDER[i]) for i in range(1, len(STAGE_ORDER))
]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, _taxonomy_seed_stub, _collections_stub):
    return {
        "seed_dir": _taxonomy_seed_stub,
        "collections_dir": _collections_stub,
        "registry_dir": registry_with_slice_children(
            tmp_path_factory.mktemp("resume_registry"), parent_id=SYNTHESISED_PARENT
        ),
    }


@pytest.fixture(scope="module")
def full_build(tmp_path_factory, corpus):
    return build_fixture_corpus(tmp_path_factory.mktemp("resume_full"), **corpus)


def test_the_corpus_has_a_parent_only_stage_g_knows(full_build):
    rows = duckdb.execute(
        f"SELECT COUNT(*) FROM read_parquet('{full_build.out_dir / 'merged_evals_view.parquet'}') "
        f"WHERE benchmark_id = ?",
        [SYNTHESISED_PARENT],
    ).fetchone()[0]
    assert rows == 1


@pytest.mark.parametrize(
    ("stop_after", "resume_at"), RESUME_POINTS, ids=[r for _, r in RESUME_POINTS]
)
def test_resumed_build_matches_the_full_build(
    stop_after, resume_at, full_build, corpus, tmp_path
):
    build_fixture_corpus(tmp_path, to_stage=stop_after, **corpus)
    resumed = build_fixture_corpus(tmp_path, from_stage=resume_at, **corpus)

    problems = warehouse_differences(full_build.out_dir, resumed.out_dir)
    assert not problems, (
        f"a build stopped after stage {stop_after} and resumed at stage "
        f"{resume_at} differs from the full build (first = full, second = "
        f"resumed) in {len(problems)} file(s):\n  " + "\n  ".join(problems)
    )


def _rows_not_in_cache(con, table: str, cached: str) -> str | None:
    try:
        extra, missing = con.execute(
            f"SELECT "
            f"(SELECT COUNT(*) FROM (SELECT * FROM {table} EXCEPT ALL "
            f"SELECT * FROM read_parquet('{cached}'))), "
            f"(SELECT COUNT(*) FROM (SELECT * FROM read_parquet('{cached}') "
            f"EXCEPT ALL SELECT * FROM {table}))"
        ).fetchone()
    except duckdb.Error as exc:
        return f"no longer comparable with its cache file ({exc})"
    if extra or missing:
        return f"{extra} row(s) the cache file lacks, {missing} row(s) only the cache file has"
    return None


def test_no_stage_writes_into_a_table_an_earlier_stage_cached(
    corpus, tmp_path, monkeypatch
):
    cache_dir = StageCache(tmp_path / "cache", SNAPSHOT_ID).dir
    drifted: list[str] = []
    compared: set[str] = set()
    live_when_cached: dict[str, set[str]] = {}
    write_stage = StageCache.write_stage

    def write_then_compare(self, con, stage_letter):
        write_stage(self, con, stage_letter)
        live = {
            name for (name,) in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        live_when_cached[stage_letter] = live
        for stage in STAGE_ORDER[: STAGE_ORDER.index(stage_letter) + 1]:
            for table in STAGE_OUTPUTS[stage]:
                if table not in live:
                    continue
                compared.add(table)
                found = _rows_not_in_cache(con, table, cache_dir / f"{table}.parquet")
                if found:
                    drifted.append(
                        f"after stage {stage_letter}, stage {stage} table {table}: {found}"
                    )

    monkeypatch.setattr(StageCache, "write_stage", write_then_compare)
    build_fixture_corpus(tmp_path, **corpus)

    assert {"canonical_benchmarks", "family_membership", "fact_results"} <= compared
    assert not drifted, (
        "a stage wrote into a table after that table was cached, so a resumed "
        "build restores different rows:\n  " + "\n  ".join(drifted)
    )
    # Past Stage G the catalog is the only table a reader can find, so one
    # that asks for the Stage A table fails instead of missing the
    # synthesised parents.
    assert "benchmark_catalog" in live_when_cached["J"]
    assert not {"canonical_benchmarks", "family_membership"} & live_when_cached["I"]


@pytest.mark.parametrize("resume_at", ["I", "J"])
def test_resuming_past_stage_g_leaves_no_superseded_table_to_read(
    resume_at, corpus, tmp_path, monkeypatch
):
    """A full build drops the tables Stage G supersedes. A resume past G has
    to restore the same connection, or a reader of `canonical_benchmarks`
    works when iterating with `--from-stage J` and fails only on a full build."""
    from eval_card_backend.canonicalise import stages

    stop_after = STAGE_ORDER[STAGE_ORDER.index(resume_at) - 1]
    build_fixture_corpus(tmp_path, to_stage=stop_after, **corpus)

    live_at_stage_j: set[str] = set()
    real = stages.stage_j_eval_results_view

    def recording(con, *args, **kwargs):
        live_at_stage_j.update(
            name for (name,) in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        )
        return real(con, *args, **kwargs)

    monkeypatch.setattr(stages, "stage_j_eval_results_view", recording)
    build_fixture_corpus(tmp_path, from_stage=resume_at, **corpus)

    assert "benchmark_catalog" in live_at_stage_j
    assert not {"canonical_benchmarks", "family_membership"} & live_at_stage_j

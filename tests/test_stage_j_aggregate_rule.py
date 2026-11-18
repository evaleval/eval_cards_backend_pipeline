"""Stage D aggregation levels and the Stage J cell-value rule.

A page shows one number per (model, benchmark, metric, conditions). These
tests pin down which submitted rows that number is allowed to come from:
the source's own benchmark total where it published one, its group rollups
otherwise, the rows themselves when they all name ONE task (reruns and
repeated records of one quantity, pooled by median), and nothing at all when
they name several tasks with no submitted aggregate and no registry task set
saying what the benchmark consists of (`AGGREGATE_LESS_CELL_LEVEL`).

A suite whose registry slice children ARE its task set is the one case where
the pipeline does state a number the source did not: the MEAN of those
children, and only when every one of them is present.

They run the pipeline through Stage I against the hand-built fixtures, then
mutate `fact_results` before materialising Stage J, so each case is built from
real canonical rows rather than a synthetic table.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _run_through_stage_i(tmp_path, monkeypatch, config: str) -> Path:
    eee_root = FIXTURES / "eee"
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv(
        "BENCHMARK_METADATA_LOCAL_DIR", str(FIXTURES / "auto_benchmarkcards")
    )
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        configs=[config],
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(tmp_path / "warehouse"),
        registry_local_dir=str(FIXTURES / "entity_registry"),
        cache_root=str(tmp_path / "cache"),
    )
    assert out_dir is not None
    return out_dir


def _materialise_view(out_dir: Path, mutate=None):
    from eval_card_backend.canonicalise import stages
    from eval_card_backend.canonicalise.resolver_setup import register_udfs
    from eval_card_backend.sources import registry as registry_src
    from eval_entity_resolver import Resolver

    con = duckdb.connect()
    register_udfs(
        con, Resolver(registry_src.load_alias_store(FIXTURES / "entity_registry"))
    )
    for table in (
        "fact_results", "benchmarks", "composites", "families", "models",
        "canonical_metrics",
    ):
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_parquet('{out_dir}/{table}.parquet')"
        )
    # the registry's own benchmark tree: the parent rollup reads its
    # `parent_benchmark_id` edges for the expected task set
    con.execute(
        "CREATE TABLE benchmark_catalog AS "
        "SELECT id, CAST(parent_benchmark_id AS VARCHAR) AS parent_benchmark_id, "
        "       CAST(metadata AS VARCHAR) AS metadata, "
        "       CAST(NULL AS VARCHAR) AS preferred_metric_id "
        f"FROM read_parquet('{FIXTURES}/entity_registry/canonical_benchmarks.parquet')"
    )
    if mutate is not None:
        mutate(con)
    stages.stage_j_eval_results_view(con, "2026-04-30T00:00:00Z")
    return con


_seq = iter(range(1000))


def _clone_fact(con, *, level: str, score: float, slice_key: str | None,
                metric: str | None = None, benchmark: str | None = None,
                model: str | None = None, is_part: bool | None = None,
                observation_role: str | None = None,
                judge: str | None = None,
                protocol: str | None = None,
                split: str | None = None,
                scale_conversion: str | None = None,
                score_canonical: float | None = None,
                lower_is_better: bool | None = None):
    """Clone the fixture's scored mmlu fact under a new identity.

    Every clone carries its own `fact_id` — the warehouse's row identity, and
    a repeat would fan out every fact-grain join Stage J makes.

    `is_part` overrides whether the row measured a PART of the benchmark
    rather than the benchmark itself. Clones inherit the template's, so a test
    that means "these rows are subjects, not totals" has to say so — that
    distinction is the whole point of the rule under test. It carries
    `observation_role` with it, since a row stated to be a part or a whole is
    a row resolution READ; pass `observation_role='unknown'` for the other
    case, a name the structured path never parsed."""
    n = next(_seq)
    replaces = [
        f"'{level}' AS aggregate_level",
        f"{score} AS score",
        f"{score if score_canonical is None else score_canonical}"
        " AS score_canonical",
        f"'agg-{n}-fact' AS fact_id",
        f"'agg-{n}-eval' AS evaluation_id",
        (f"'{slice_key}' AS slice_key" if slice_key is not None
         else "CAST(NULL AS VARCHAR) AS slice_key"),
    ]
    if metric:
        replaces += [
            f"'{metric}' AS metric_key", f"'{metric}' AS metric_key_effective",
            f"'{metric}' AS metric_base_key", f"'{metric}' AS metric_id",
        ]
    if benchmark:
        replaces += [
            f"'{benchmark}' AS benchmark_key", f"'{benchmark}' AS benchmark_id",
        ]
    if model:
        replaces += [
            f"'{model}' AS model_aggregation_key", f"'{model}' AS model_key",
            f"'{model}' AS model_raw",
        ]
    if is_part is not None:
        replaces += [f"{str(is_part).upper()} AS is_part"]
        if observation_role is None:
            observation_role = "part" if is_part else "whole"
    if observation_role is not None:
        replaces += [f"'{observation_role}' AS observation_role"]
    if judge is not None:
        replaces += [f"'{judge}' AS judge_condition"]
    if protocol is not None:
        replaces += [f"'{protocol}' AS protocol_condition"]
    if split is not None:
        replaces += [f"'{split}' AS split"]
    if scale_conversion is not None:
        replaces += [f"'{scale_conversion}' AS scale_conversion"]
    if lower_is_better is not None:
        replaces += [f"{str(lower_is_better).upper()} AS lower_is_better"]
    con.execute(
        "INSERT INTO fact_results SELECT * REPLACE (" + ", ".join(replaces) + ") "
        "FROM _fact_template"
    )


def _wipe_mmlu(con):
    """Keep one real mmlu fact row as the clone template, then clear the
    benchmark so each test builds the cell it means to test."""
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _fact_template AS "
        "SELECT * FROM fact_results "
        "WHERE benchmark_key = 'mmlu' AND score IS NOT NULL LIMIT 1"
    )
    con.execute("DELETE FROM fact_results WHERE benchmark_key = 'mmlu'")


def _cell(con, benchmark="mmlu", metric="accuracy"):
    return con.execute(
        "SELECT score, value_level, fact_row_count, aggregate_components, "
        "       metric_source_label, score_details.standard_error "
        "FROM eval_results_view "
        "WHERE benchmark_id = ? AND metric_id = ?",
        [benchmark, metric],
    ).fetchall()


# ---------------------------------------------------------------------------
# Stage D — the level itself
# ---------------------------------------------------------------------------


def test_aggregate_level_read_from_evaluation_name_tail(tmp_path, monkeypatch):
    """The fixture's plain `MMLU` / `Anatomy` names carry no aggregate marker,
    so every fact is a leaf. The marker is the last dotted segment."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_slices")
    con = duckdb.connect()
    levels = con.execute(
        f"SELECT DISTINCT aggregate_level FROM read_parquet('{out}/fact_results.parquet')"
    ).fetchall()
    assert levels == [("leaf",)]


@pytest.mark.parametrize(
    "tail,expected",
    [
        ("overall", "root"),
        ("OVERALL", "root"),
        ("fr_overall", "subgroup"),
        ("small_overall", "subgroup"),
        ("fr_stem", "leaf"),
        ("biology", "leaf"),
    ],
)
def test_aggregate_level_expression(tail, expected):
    """The classification itself: exactly `overall` is the benchmark total,
    `<group>_overall` a group rollup, anything else one task."""
    con = duckdb.connect()
    got = con.execute(
        """
        SELECT CASE
            WHEN lower(trim(split_part(?, '.', -1))) = 'overall' THEN 'root'
            WHEN ends_with(lower(trim(split_part(?, '.', -1))), '_overall')
                THEN 'subgroup'
            ELSE 'leaf'
        END
        """,
        [f"bench.bench.{tail}", f"bench.bench.{tail}"],
    ).fetchone()[0]
    assert got == expected


# ---------------------------------------------------------------------------
# Stage J — which rows the value comes from
# ---------------------------------------------------------------------------


def test_root_rows_win_over_subgroups_and_leaves(tmp_path, monkeypatch):
    """The source's own benchmark total is the value, however many task rows
    sit beside it. Three roots are three settings of one measurement and pool
    by median; the 60 leaves do not enter it."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        for s in (0.70, 0.71, 0.72):
            _clone_fact(con, level="root", is_part=False, score=s, slice_key=None)
        for s in (0.40, 0.95):
            _clone_fact(con, level="subgroup", is_part=True, score=s, slice_key="stem overall")
        for i in range(60):
            _clone_fact(con, level="leaf", is_part=True, score=0.1 + i / 100,
                        slice_key=f"subject {i}")

    rows = _cell(_materialise_view(out, mutate=mutate))
    assert len(rows) == 1
    score, level, n_inputs, components, _label, se = rows[0]
    assert abs(score - 0.71) < 1e-9
    assert level == "whole"
    assert n_inputs == 3
    assert [round(c["score"], 2) for c in components] == [0.70, 0.71, 0.72]
    # a median over three runs has no single standard error
    assert se is None


def test_parts_pool_when_the_cell_has_no_whole(tmp_path, monkeypatch, caplog):
    """No row measured the benchmark itself and the registry does not say what
    it consists of, so the parts pool exactly as the pipeline always pooled
    them — labelled `pooled_parts`, listed, and named in the log. No page that
    had a number loses it."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        for s_ in (0.30, 0.50, 0.90):
            _clone_fact(con, level="leaf", is_part=True, score=s_,
                        slice_key=f"subject {s_}")

    with caplog.at_level("WARNING"):
        con = _materialise_view(out, mutate=mutate)
    score, level, n_inputs, components, _l, se = _cell(con)[0]
    assert abs(score - 0.50) < 1e-9
    assert level == "pooled_parts"
    assert n_inputs == 3
    assert len(components) == 3
    assert se is None
    assert any("parts-only cell" in r.getMessage() for r in caplog.records)


def test_parts_never_enter_a_whole_value(tmp_path, monkeypatch):
    """The defect the rule exists for. MMLU's three answer-extraction totals
    and its 183 subject rows used to median together and publish 0.750 where
    the source reported 0.704; the subjects must stay out of the number."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        for s_ in (0.70, 0.71, 0.72):            # the totals
            _clone_fact(con, level="root", is_part=False, score=s_,
                        slice_key="mmlu")
        for i in range(10):                      # the subjects
            _clone_fact(con, level="leaf", is_part=True, score=0.95,
                        slice_key=f"subject {i}")

    score, level, n_inputs, components, _l, _se = _cell(
        _materialise_view(out, mutate=mutate)
    )[0]
    assert abs(score - 0.71) < 1e-9
    assert level == "whole"
    assert n_inputs == 3
    assert [round(c["score"], 2) for c in components] == [0.70, 0.71, 0.72]


def test_several_wholes_are_repeated_measurements_and_pool(tmp_path, monkeypatch):
    """Reruns, repeated leaderboard records and settings arms all land here:
    each measured the benchmark, so they are repeated readings of one quantity
    and the middle one represents them."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        for s_ in (0.50, 0.78, 0.85):
            _clone_fact(con, level="leaf", is_part=False, score=s_,
                        slice_key=None)

    score, level, n_inputs, components, _l, _se = _cell(
        _materialise_view(out, mutate=mutate)
    )[0]
    assert abs(score - 0.78) < 1e-9
    assert level == "whole"
    assert n_inputs == 3
    assert len(components) == 3


def test_wholes_on_different_splits_never_pool(tmp_path, monkeypatch):
    """Two whole readings on two dataset splits are two measurements, not a
    rerun: `split` is in the cell grain, so each gets its own row and its own
    number, and one of them heads the page. A third whole on one of those
    splits pools with its own split only."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _clone_fact(con, level="root", is_part=False, score=0.50,
                    slice_key=None, split="test")
        _clone_fact(con, level="root", is_part=False, score=0.60,
                    slice_key=None, split="test")
        _clone_fact(con, level="root", is_part=False, score=0.90,
                    slice_key=None, split="validation")

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT split, score, value_level, fact_row_count, is_headline "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu' ORDER BY split"
    ).fetchall()
    assert [(r[0], round(r[1], 2), r[2], r[3]) for r in rows] == [
        ("test", 0.55, "whole", 2),
        ("validation", 0.90, "whole", 1),
    ]
    assert sum(1 for r in rows if r[4]) == 1


def test_one_whole_row_is_the_reading(tmp_path, monkeypatch):
    """One whole observation IS the cell's number, and keeps its own
    context."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _clone_fact(con, level="leaf", is_part=False, score=0.61,
                    slice_key="anatomy")

    score, level, n_inputs, components, _l, _se = _cell(
        _materialise_view(out, mutate=mutate)
    )[0]
    assert abs(score - 0.61) < 1e-9
    assert level == "whole"
    assert n_inputs == 1
    assert components is None


def test_a_display_slice_does_not_make_a_row_a_part(tmp_path, monkeypatch):
    """`slice_key` is display-only. It is minted whenever a benchmark sees
    more than one raw spelling, so it lands on the total as readily as on the
    parts — all 186 Apertus MMLU rows carry one, the three totals included.
    Classifying on it would leave MMLU, MATH, INCLUDE and ACPBench with no
    whole observation at all."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _clone_fact(con, level="root", is_part=False, score=0.70,
                    slice_key="mmlu")

    score, level, _n, _c, _l, _se = _cell(
        _materialise_view(out, mutate=mutate)
    )[0]
    assert abs(score - 0.70) < 1e-9
    assert level == "whole"


def test_strict_mode_shows_nothing_for_a_parts_only_cell(tmp_path, monkeypatch,
                                                        caplog):
    """The other half of the switch, pinned so flipping the constant stays a
    one-line change. With `AGGREGATE_LESS_CELL_LEVEL = "none"` a cell that
    holds only parts, with no whole and no complete registry task set, shows
    no number at all instead of pooling them."""
    pytest.importorskip("duckdb")
    from eval_card_backend.canonicalise import stages

    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    monkeypatch.setattr(stages, "AGGREGATE_LESS_CELL_LEVEL", "none")

    def mutate(con):
        _wipe_mmlu(con)
        for value in (0.40, 0.50, 0.90):
            _clone_fact(con, level="leaf", is_part=True, score=value,
                        slice_key=f"subject {value}")

    with caplog.at_level("WARNING"):
        con = _materialise_view(out, mutate=mutate)
    score, level, n_inputs, components, _l, _se = _cell(con)[0]
    assert score is None
    assert level == "none"
    assert n_inputs == 0
    assert components is None
    assert any("parts-only cell" in r.getMessage() for r in caplog.records)


def test_single_fact_cell_keeps_its_own_context(tmp_path, monkeypatch):
    """A one-fact cell IS that fact, so its uncertainty and record pointer
    stay on the row and there is no component list."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    rows = con_rows = _materialise_view(out).execute(
        "SELECT fact_row_count, aggregate_components, value_level "
        "FROM eval_results_view WHERE fact_row_count = 1"
    ).fetchall()
    assert rows, "fixture has no single-fact cells"
    assert all(c is None for _n, c, _lvl in con_rows)


# ---------------------------------------------------------------------------
# Stage J — slice parents materialised from their children
# ---------------------------------------------------------------------------


def _make_suite(con):
    """Turn the fixture's mmlu row into a slice of a new `mmlu-suite`
    parent that has no facts of its own."""
    con.execute(
        "INSERT INTO benchmarks SELECT * REPLACE ("
        "  'mmlu-suite' AS benchmark_id, 'MMLU Suite' AS display_name,"
        "  'MMLU Suite' AS benchmark_display_name, FALSE AS is_slice,"
        "  CAST(NULL AS VARCHAR) AS parent_benchmark_id"
        ") FROM benchmarks WHERE benchmark_id = 'mmlu'"
    )
    con.execute(
        "INSERT INTO benchmarks SELECT * REPLACE ("
        "  'mmlu-b' AS benchmark_id, 'MMLU B' AS display_name,"
        "  'MMLU B' AS benchmark_display_name, TRUE AS is_slice,"
        "  'mmlu-suite' AS parent_benchmark_id"
        ") FROM benchmarks WHERE benchmark_id = 'mmlu'"
    )
    con.execute(
        "UPDATE benchmarks SET parent_benchmark_id = 'mmlu-suite', "
        "is_slice = TRUE WHERE benchmark_id = 'mmlu'"
    )
    # and the registry edges behind them — the expected task set
    con.execute(
        "INSERT INTO benchmark_catalog (id, parent_benchmark_id, metadata) VALUES "
        "('mmlu-suite', NULL, NULL), ('mmlu-b', 'mmlu-suite', NULL)"
    )
    con.execute(
        "UPDATE benchmark_catalog SET parent_benchmark_id = 'mmlu-suite' "
        "WHERE id = 'mmlu'"
    )


def _add_suite_child(con, child: str, display: str, role: str | None = None):
    """One more registry child of `mmlu-suite`, in both the dim and the
    registry tree.

    `role` stamps `benchmark_catalog.metadata` — `"aggregate"` marks a
    child that is the source's own rollup OVER its siblings, which the parent
    rollup must not count as a part of itself."""
    role_sql = "NULL" if role is None else f"'{{\"role\": \"{role}\"}}'"
    con.execute(
        "INSERT INTO benchmarks SELECT * REPLACE ("
        f"  '{child}' AS benchmark_id, '{display}' AS display_name,"
        f"  '{display}' AS benchmark_display_name, TRUE AS is_slice,"
        "  'mmlu-suite' AS parent_benchmark_id"
        ") FROM benchmarks WHERE benchmark_id = 'mmlu-b'"
    )
    con.execute(
        f"INSERT INTO benchmark_catalog (id, parent_benchmark_id, metadata) VALUES "
        f"('{child}', 'mmlu-suite', {role_sql})"
    )


def test_fact_less_slice_parent_is_the_mean_of_its_children(tmp_path, monkeypatch):
    """A suite whose task variants are registry slice children has no facts of
    its own. Without this it disappears from the product — children and no
    suite.

    Its value is the MEAN of its children's cells, not their median: these are
    the parts of one benchmark being combined into a whole, and the average of
    the parts is what a suite score is. Each child counts once, they are
    listed, and no per-fact context survives."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _add_suite_child(con, "mmlu-c", "MMLU C")
        # 0.40 / 0.60 / 0.95: mean 0.65, median would be 0.60
        for bench, value in (
            (None, 0.40), ("mmlu-b", 0.60), ("mmlu-c", 0.95),
        ):
            _clone_fact(con, level="root", is_part=False, score=value, slice_key=None,
                        model="m1", benchmark=bench)

    con = _materialise_view(out, mutate=mutate)
    rows = con.execute(
        "SELECT score, value_level, value_aggregation, fact_row_count, "
        "       children_present, children_expected, aggregate_components, "
        "       metric_source_label, score_details.standard_error, is_headline "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu-suite'"
    ).fetchall()
    assert len(rows) == 1
    (score, level, aggregation, n_children, present, expected,
     components, label, se, is_headline) = rows[0]
    assert abs(score - 0.65) < 1e-9
    assert level == "derived"
    assert aggregation == "mean"
    assert n_children == 3
    assert (present, expected) == (3, 3)
    assert sorted(round(c["score"], 2) for c in components) == [0.40, 0.60, 0.95]
    assert label is None and se is None
    assert is_headline is True
    # the children keep their own rows
    assert con.execute(
        "SELECT count(*) FROM eval_results_view "
        "WHERE benchmark_id IN ('mmlu', 'mmlu-b', 'mmlu-c')"
    ).fetchone()[0] == 3


def test_suite_mean_is_sample_weighted_when_every_child_says_its_n(
    tmp_path, monkeypatch
):
    """A 50-item category should not count as much as a 2,000-item one. When
    every part carries its sample count the suite mean is weighted by it."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _clone_fact(con, level="root", is_part=False, score=0.40, slice_key=None, model="m1")
        _clone_fact(con, level="root", is_part=False, score=0.60, slice_key=None,
                    benchmark="mmlu-b", model="m1")
        con.execute("UPDATE fact_results SET n_samples = 100 "
                    "WHERE benchmark_key = 'mmlu'")
        con.execute("UPDATE fact_results SET n_samples = 300 "
                    "WHERE benchmark_key = 'mmlu-b'")

    row = _materialise_view(out, mutate=mutate).execute(
        "SELECT score, value_aggregation FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite'"
    ).fetchone()
    # (0.40*100 + 0.60*300) / 400 = 0.55, where the flat mean would be 0.50
    assert abs(row[0] - 0.55) < 1e-9
    assert row[1] == "weighted_mean"


def test_suite_mean_is_unweighted_when_one_child_has_no_n(tmp_path, monkeypatch):
    """One absent weight and a weighted average silently becomes a different
    quantity, so the rule is all-or-nothing."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _clone_fact(con, level="root", is_part=False, score=0.40, slice_key=None, model="m1")
        _clone_fact(con, level="root", is_part=False, score=0.60, slice_key=None,
                    benchmark="mmlu-b", model="m1")
        con.execute("UPDATE fact_results SET n_samples = 100 "
                    "WHERE benchmark_key = 'mmlu'")
        con.execute("UPDATE fact_results SET n_samples = NULL "
                    "WHERE benchmark_key = 'mmlu-b'")

    row = _materialise_view(out, mutate=mutate).execute(
        "SELECT score, value_aggregation FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite'"
    ).fetchone()
    assert abs(row[0] - 0.50) < 1e-9
    assert row[1] == "mean"


def test_suite_mean_on_a_lower_is_better_metric(tmp_path, monkeypatch):
    """Direction does not change the arithmetic — a suite refusal rate is
    still the average of its tasks — but it must survive onto the row and
    invert the normalised twin."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _clone_fact(con, level="root", is_part=False, score=0.10, slice_key=None, model="m1",
                    lower_is_better=True)
        _clone_fact(con, level="root", is_part=False, score=0.30, slice_key=None, model="m1",
                    benchmark="mmlu-b", lower_is_better=True)

    row = _materialise_view(out, mutate=mutate).execute(
        "SELECT score, value_aggregation, lower_is_better, score_normalized "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu-suite'"
    ).fetchone()
    score, aggregation, lower, normalized = row
    assert abs(score - 0.20) < 1e-9
    assert aggregation == "mean"
    assert lower is True
    assert abs(normalized - 0.80) < 1e-9


def test_an_aggregate_child_is_not_a_part_of_its_parent(tmp_path, monkeypatch):
    """A child the registry marks `metadata.role = "aggregate"` is the
    source's own rollup OVER its siblings, not a part alongside them.
    Averaging it with them counts those results twice.

    BFCL-v3 is the corpus case: `bfcl-v3-single-turn` is Swiss AI's aggregate
    over the 13 single-turn categories and is published next to them. It keeps
    its own row and its own page; it just does not enter the parent's expected
    set, its coverage count, or its mean."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _add_suite_child(con, "mmlu-roll", "MMLU Rollup", role="aggregate")
        _clone_fact(con, level="root", is_part=False, score=0.40, slice_key=None, model="m1")
        _clone_fact(con, level="root", is_part=False, score=0.60, slice_key=None, model="m1",
                    benchmark="mmlu-b")
        # the source's own rollup over the two, published next to them
        _clone_fact(con, level="root", is_part=False, score=0.50, slice_key=None, model="m1",
                    benchmark="mmlu-roll")

    con = _materialise_view(out, mutate=mutate)
    score, level, aggregation, n, present, expected = con.execute(
        "SELECT score, value_level, value_aggregation, fact_row_count, "
        "       children_present, children_expected FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite'"
    ).fetchone()
    # the mean of the two PARTS, over a denominator that excludes the rollup
    assert abs(score - 0.50) < 1e-9
    assert level == "derived" and aggregation == "mean"
    assert (n, present, expected) == (2, 2, 2)
    assert [round(c["score"], 2) for c in con.execute(
        "SELECT aggregate_components FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite'").fetchone()[0]] == [0.40, 0.60]
    # and the rollup keeps its own row
    assert con.execute(
        "SELECT score, value_level FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-roll'"
    ).fetchone() == (0.50, "whole")


def test_a_diagnostic_child_is_not_a_part_of_its_parent(tmp_path, monkeypatch):
    """A child the registry marks `metadata.role = "diagnostic"` measures a
    different quantity under the parent's name (BFCL's format-sensitivity
    spread is not an accuracy). It keeps its own row and page, but it is
    neither in the parent's expected task set (denominator) nor in the mean
    (numerator): the suite value is the mean of the real parts alone."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _add_suite_child(con, "mmlu-diag", "MMLU Diagnostic", role="diagnostic")
        _clone_fact(con, level="root", is_part=False, score=0.40, slice_key=None, model="m1")
        _clone_fact(con, level="root", is_part=False, score=0.60, slice_key=None, model="m1",
                    benchmark="mmlu-b")
        # a spread-like number that would drag the mean if it counted
        _clone_fact(con, level="root", is_part=False, score=0.05, slice_key=None, model="m1",
                    benchmark="mmlu-diag")

    con = _materialise_view(out, mutate=mutate)
    score, level, aggregation, n, present, expected = con.execute(
        "SELECT score, value_level, value_aggregation, fact_row_count, "
        "       children_present, children_expected FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite'"
    ).fetchone()
    assert abs(score - 0.50) < 1e-9
    assert level == "derived" and aggregation == "mean"
    assert (n, present, expected) == (2, 2, 2)
    assert con.execute(
        "SELECT score, value_level FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-diag'"
    ).fetchone() == (0.05, "whole")


def test_an_aggregate_child_cannot_stand_in_for_a_missing_part(tmp_path,
                                                               monkeypatch):
    """The coverage gate counts PARTS. A suite whose rollup arrived but whose
    second category did not is still incomplete, and the rollup's presence must
    not disguise that."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _add_suite_child(con, "mmlu-roll", "MMLU Rollup", role="aggregate")
        _clone_fact(con, level="root", is_part=False, score=0.40, slice_key=None, model="m1")
        _clone_fact(con, level="root", is_part=False, score=0.50, slice_key=None, model="m1",
                    benchmark="mmlu-roll")

    con = _materialise_view(out, mutate=mutate)
    assert con.execute(
        "SELECT count(*) FROM eval_results_view WHERE benchmark_id = 'mmlu-suite'"
    ).fetchone()[0] == 0


def test_a_parent_with_its_own_root_rows_needs_no_children(tmp_path, monkeypatch,
                                                           caplog):
    """MT-Bench's shape after the registry stopped modelling its `overall` as
    a child: the parent publishes its own rows and keeps two turn children.

    The parent's number is its own submitted total, there is no derived row,
    and the children are not counted into it — the double count the old
    aggregate-plus-parts rollup would have produced."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _clone_fact(con, level="root", is_part=False, score=0.40, slice_key=None, model="m1",
                    judge='{"judges":["openai/gpt-4"],"label":"turn 1"}')
        _clone_fact(con, level="root", is_part=False, score=0.60, slice_key=None, model="m1",
                    benchmark="mmlu-b",
                    judge='{"judges":["openai/gpt-4"],"label":"turn 2"}')
        # the parent's own submitted total, under its own label
        _clone_fact(con, level="root", is_part=False, score=0.99, slice_key=None, model="m1",
                    benchmark="mmlu-suite",
                    judge='{"judges":["openai/gpt-4"],"label":"overall"}')

    with caplog.at_level("WARNING"):
        con = _materialise_view(out, mutate=mutate)
    rows = con.execute(
        "SELECT score, value_level, fact_row_count FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite'"
    ).fetchall()
    assert rows == [(0.99, "whole", 1)]
    # exactly one row: the children are NOT averaged into a second, competing
    # number for the same cell
    assert len(rows) == 1
    # The parent does still appear in the partial-coverage log, because its own
    # row and its children carry different judge LABELS and the suppression is
    # per condition. That is a logging artefact of labels living inside
    # `judge_condition` (Sol N2): no wrong number is published, and the fix is
    # a canonical condition identity separate from the display label.
    assert any("partial coverage" in r.getMessage() for r in caplog.records)


def test_partial_task_coverage_emits_no_suite_row(tmp_path, monkeypatch, caplog):
    """A mean over two of the suite's three tasks is a different quantity from
    the suite, and nothing on the page would say so. No row, and a line naming
    the cell — a missing task is usually a data or seed question."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _add_suite_child(con, "mmlu-c", "MMLU C")
        # mmlu-c publishes nothing
        _clone_fact(con, level="root", is_part=False, score=0.40, slice_key=None, model="m1")
        _clone_fact(con, level="root", is_part=False, score=0.60, slice_key=None,
                    benchmark="mmlu-b", model="m1")

    with caplog.at_level("WARNING"):
        con = _materialise_view(out, mutate=mutate)
    assert con.execute(
        "SELECT count(*) FROM eval_results_view WHERE benchmark_id = 'mmlu-suite'"
    ).fetchone()[0] == 0
    assert any("partial coverage 2/3" in r.getMessage() for r in caplog.records)


def test_parent_with_its_own_rows_is_not_rolled_up(tmp_path, monkeypatch):
    """Guard against double counting: a parent that publishes the cell itself
    keeps its own number and its children stay slices."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _clone_fact(con, level="root", is_part=False, score=0.40, slice_key=None, model="m1")
        _clone_fact(con, level="root", is_part=False, score=0.60, slice_key=None,
                    benchmark="mmlu-b", model="m1")
        # the parent's own submitted total, nothing like the children's mean
        _clone_fact(con, level="root", is_part=False, score=0.99, slice_key=None,
                    benchmark="mmlu-suite", model="m1")

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT score, value_level, fact_row_count FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite'"
    ).fetchall()
    assert len(rows) == 1
    score, level, n = rows[0]
    assert abs(score - 0.99) < 1e-9
    assert level == "whole"
    assert n == 1


def test_parent_rollup_keeps_the_children_condition_grain(tmp_path, monkeypatch):
    """Children judged differently are different readings. Rolling them into
    one parent row under a blank judge publishes a number no judge produced,
    and hands it to the merged best-result and the comparison index as though
    it were comparable. The parent gets one row per condition instead."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        # both children scored under both judges: each judge sees the whole
        # suite, so each judge gets its own complete suite reading
        for bench, a, b in ((None, 0.40, 0.50), ("mmlu-b", 0.60, 0.90)):
            _clone_fact(con, level="root", is_part=False, score=a, slice_key=None, model="m1",
                        benchmark=bench, judge='{"judges":["judge-a"]}')
            _clone_fact(con, level="root", is_part=False, score=b, slice_key=None, model="m1",
                        benchmark=bench, judge='{"judges":["judge-b"]}')

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT judge_condition, score, fact_row_count, children_present, "
        "       children_expected, is_headline "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu-suite' "
        "ORDER BY judge_condition"
    ).fetchall()
    assert len(rows) == 2
    assert [r[0] for r in rows] == ['{"judges":["judge-a"]}',
                                    '{"judges":["judge-b"]}']
    # judge A means 0.40/0.60, judge B means 0.50/0.90 — never mixed
    assert [round(r[1], 2) for r in rows] == [0.50, 0.70]
    assert all(r[2] == 2 and (r[3], r[4]) == (2, 2) for r in rows)
    # one summary reading per (benchmark, metric, model), as everywhere else
    assert sum(1 for r in rows if r[5]) == 1


def test_parent_own_row_suppresses_only_its_own_condition(tmp_path, monkeypatch):
    """A parent row under one judge says nothing about the same cell under
    another. Suppressing the whole cell would delete the derived readings for
    every other condition the children published."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        for bench, a, b in ((None, 0.40, 0.50), ("mmlu-b", 0.60, 0.90)):
            _clone_fact(con, level="root", is_part=False, score=a, slice_key=None, model="m1",
                        benchmark=bench, judge='{"judges":["judge-a"]}')
            _clone_fact(con, level="root", is_part=False, score=b, slice_key=None, model="m1",
                        benchmark=bench, judge='{"judges":["judge-b"]}')
        # the parent publishes this cell itself, but only under judge A
        _clone_fact(con, level="root", is_part=False, score=0.99, slice_key=None, model="m1",
                    benchmark="mmlu-suite", judge='{"judges":["judge-a"]}')

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT judge_condition, score, value_level, is_headline "
        "FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite' ORDER BY judge_condition"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0] == ('{"judges":["judge-a"]}', 0.99, "whole", True)
    judge_b, score_b, level_b, headline_b = rows[1]
    assert judge_b == '{"judges":["judge-b"]}'
    assert abs(score_b - 0.70) < 1e-9
    assert level_b == "derived"
    # the parent's own row is the page's reading; the derived row under the
    # other judge stays readable but never a second headline for the cell
    assert headline_b is False


def test_parent_bare_total_suppresses_derived_rows_on_every_split(tmp_path,
                                                                  monkeypatch):
    """The parent's own total states no split; its children (in other records,
    so nothing was inherited) are all on `val`. An unstated split is no
    evidence of a different run, and a derived mean beside the source's own
    total would count the children twice: the own row is the only row."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        for bench, s_ in ((None, 0.40), ("mmlu-b", 0.60)):
            _clone_fact(con, level="root", is_part=False, score=s_,
                        slice_key=None, model="m1", benchmark=bench, split="val")
        _clone_fact(con, level="root", is_part=False, score=0.99,
                    slice_key=None, model="m1", benchmark="mmlu-suite")

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT split, score, value_level, is_headline FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite'"
    ).fetchall()
    assert rows == [(None, 0.99, "whole", True)]


def test_parent_total_on_another_split_keeps_the_derived_row_unheadlined(
        tmp_path, monkeypatch):
    """The parent's own total says `test`; its children ran on `val`. Two
    measurements: the derived `val` row is emitted next to the own `test` row,
    and the own row is the page's one headline."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        for bench, s_ in ((None, 0.40), ("mmlu-b", 0.60)):
            _clone_fact(con, level="root", is_part=False, score=s_,
                        slice_key=None, model="m1", benchmark=bench, split="val")
        _clone_fact(con, level="root", is_part=False, score=0.99,
                    slice_key=None, model="m1", benchmark="mmlu-suite",
                    split="test")

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT split, score, value_level, is_headline FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite' ORDER BY split"
    ).fetchall()
    assert [(r[0], round(r[1], 2), r[2], r[3]) for r in rows] == [
        ("test", 0.99, "whole", True),
        ("val", 0.50, "derived", False),
    ]


_LOW = '{"reasoning_effort":"low"}'
_XHIGH = '{"reasoning_effort":"xhigh"}'


def _derived_suite_rows(tmp_path, monkeypatch, children):
    """Build `mmlu-suite` from `(benchmark, protocol, score)` children and
    return its rows as `(protocol_condition, score, is_headline)`."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    lower = children and len(children[0]) == 4

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        for child in children:
            bench, protocol, s_ = child[:3]
            _clone_fact(con, level="root", is_part=False, score=s_,
                        slice_key=None, model="m1", benchmark=bench,
                        protocol=protocol,
                        lower_is_better=True if lower else None)

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT protocol_condition, score, value_level, is_headline "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu-suite' "
        "ORDER BY protocol_condition"
    ).fetchall()
    assert {r[2] for r in rows} == {"derived"}
    return [(r[0], round(r[1], 2), r[3]) for r in rows]


def test_equally_wide_derived_rows_headline_the_best_scoring_arm(
        tmp_path, monkeypatch):
    """Two protocol arms, each over both children. The better one heads the
    page, as it would for a cell the source reported itself; the name of the
    condition does not decide it."""
    rows = _derived_suite_rows(tmp_path, monkeypatch, [
        (None, _LOW, 0.40), ("mmlu-b", _LOW, 0.60),
        (None, _XHIGH, 0.80), ("mmlu-b", _XHIGH, 0.90),
    ])
    assert rows == [(_LOW, 0.50, False), (_XHIGH, 0.85, True)]


def test_equally_wide_derived_rows_on_a_lower_is_better_metric(
        tmp_path, monkeypatch):
    """Direction flipped: the cheaper arm heads the page."""
    rows = _derived_suite_rows(tmp_path, monkeypatch, [
        (None, _LOW, 0.40, True), ("mmlu-b", _LOW, 0.60, True),
        (None, _XHIGH, 0.80, True), ("mmlu-b", _XHIGH, 0.90, True),
    ])
    assert rows == [(_LOW, 0.50, True), (_XHIGH, 0.85, False)]


def test_a_better_scoring_arm_over_part_of_the_suite_is_not_a_candidate(
        tmp_path, monkeypatch):
    """Score only chooses among arms that cover the suite. An arm over one of
    the two children gets no derived row, so it cannot head the page however
    it scored."""
    rows = _derived_suite_rows(tmp_path, monkeypatch, [
        (None, _LOW, 0.40), ("mmlu-b", _LOW, 0.60),
        (None, _XHIGH, 0.95),
    ])
    assert rows == [(_LOW, 0.50, True)]


# ---------------------------------------------------------------------------
# A headline is one run: every metric of a model reads one protocol condition
# ---------------------------------------------------------------------------


def _headlines(tmp_path, monkeypatch, facts, *, benchmark_id="mmlu",
               suite=False):
    """Build cells from `(benchmark, metric, protocol, score, lower)` facts
    and return the headline rows as `{metric_id: (protocol, score)}`."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        if suite:
            _make_suite(con)
        for bench, metric, protocol, s_, lower in facts:
            _clone_fact(con, level="root", is_part=False, score=s_,
                        slice_key=None, model="m1", benchmark=bench,
                        metric=metric, protocol=protocol,
                        lower_is_better=lower)

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT metric_id, protocol_condition, score FROM eval_results_view "
        f"WHERE benchmark_id = '{benchmark_id}' AND is_headline "
        "ORDER BY metric_id"
    ).fetchall()
    assert len({r[0] for r in rows}) == len(rows), "one headline per metric"
    return {r[0]: (r[1], round(r[2], 2)) for r in rows}


def test_a_models_metrics_headline_one_run_condition(tmp_path, monkeypatch):
    """The high-effort run scores best and costs most. Its cost is the one
    shown beside its score: the cheapest cost belongs to another run, and a
    row pairing the two describes a run nobody made."""
    got = _headlines(tmp_path, monkeypatch, [
        (None, "accuracy", _LOW, 0.50, False),
        (None, "accuracy", _XHIGH, 0.90, False),
        (None, "cost-per-task", _LOW, 0.10, True),
        (None, "cost-per-task", _XHIGH, 0.80, True),
    ])
    assert got == {"accuracy": (_XHIGH, 0.90), "cost-per-task": (_XHIGH, 0.80)}


def test_a_metric_missing_from_the_run_condition_keeps_its_own_best_arm(
        tmp_path, monkeypatch):
    """The model never reported cost under its headline condition. The cost
    it did report still heads its cell rather than the cell going blank."""
    got = _headlines(tmp_path, monkeypatch, [
        (None, "accuracy", _LOW, 0.50, False),
        (None, "accuracy", _XHIGH, 0.90, False),
        (None, "cost-per-task", _LOW, 0.10, True),
    ])
    assert got == {"accuracy": (_XHIGH, 0.90), "cost-per-task": (_LOW, 0.10)}


def test_a_single_metric_cell_still_headlines_its_best_arm(tmp_path,
                                                          monkeypatch):
    got = _headlines(tmp_path, monkeypatch, [
        (None, "accuracy", _LOW, 0.50, False),
        (None, "accuracy", _XHIGH, 0.90, False),
    ])
    assert got == {"accuracy": (_XHIGH, 0.90)}


def test_derived_parent_metrics_headline_one_run_condition(tmp_path,
                                                           monkeypatch):
    """Same rule one level up: the suite's derived score and derived cost
    come from the same condition."""
    facts = []
    for bench in (None, "mmlu-b"):
        facts += [
            (bench, "accuracy", _LOW, 0.50, False),
            (bench, "accuracy", _XHIGH, 0.90, False),
            (bench, "cost-per-task", _LOW, 0.10, True),
            (bench, "cost-per-task", _XHIGH, 0.80, True),
        ]
    got = _headlines(tmp_path, monkeypatch, facts,
                     benchmark_id="mmlu-suite", suite=True)
    assert got == {"accuracy": (_XHIGH, 0.90), "cost-per-task": (_XHIGH, 0.80)}


_UNKNOWN = '{"feedback":"unknown"}'


def test_an_unread_run_condition_never_outranks_a_clean_arm(tmp_path,
                                                            monkeypatch):
    """The model's only arm on the page's first metric is one we could not
    read, so it heads that cell. On the next metric the model also has a
    clean arm, and the clean arm heads that cell: an unread arm is never
    promoted over a known-clean one, and never becomes the run condition
    while a clean arm exists."""
    got = _headlines(tmp_path, monkeypatch, [
        (None, "accuracy", _UNKNOWN, 0.50, False),
        (None, "cost-per-task", _UNKNOWN, 0.10, True),
        (None, "cost-per-task", _LOW, 0.80, True),
    ])
    assert got == {"accuracy": (_UNKNOWN, 0.50), "cost-per-task": (_LOW, 0.80)}


def test_a_metric_with_no_derived_row_does_not_choose_the_run_condition(
        tmp_path, monkeypatch):
    """Accuracy covers one of the suite's two children, so the suite gets no
    accuracy row and cost is the only metric on its page. The run condition
    is chosen on cost, the page's default metric, not on the accuracy arms
    nobody will see."""
    facts = [
        (None, "accuracy", _LOW, 0.90, False),
        (None, "accuracy", _XHIGH, 0.50, False),
    ]
    for bench in (None, "mmlu-b"):
        facts += [
            (bench, "cost-per-task", _LOW, 0.80, True),
            (bench, "cost-per-task", _XHIGH, 0.10, True),
        ]
    got = _headlines(tmp_path, monkeypatch, facts,
                     benchmark_id="mmlu-suite", suite=True)
    assert got == {"cost-per-task": (_XHIGH, 0.10)}


def test_a_derived_row_follows_the_run_condition_of_the_parents_own_rows(
        tmp_path, monkeypatch):
    """The suite reports accuracy itself, under the low-effort run, and its
    cost is derived from the children. The derived cost that heads the page
    is the low-effort one: the model has one run condition on the page, and
    its own row already states it."""
    facts = [("mmlu-suite", "accuracy", _LOW, 0.70, False)]
    for bench in (None, "mmlu-b"):
        facts += [
            (bench, "cost-per-task", _LOW, 0.80, True),
            (bench, "cost-per-task", _XHIGH, 0.10, True),
        ]
    got = _headlines(tmp_path, monkeypatch, facts,
                     benchmark_id="mmlu-suite", suite=True)
    assert got == {"accuracy": (_LOW, 0.70), "cost-per-task": (_LOW, 0.80)}


_ANSWER_FEEDBACK = '{"feedback":"answer_feedback"}'


@pytest.mark.parametrize("clean", [None, '{"feedback":"none"}'])
def test_a_derived_answer_feedback_arm_is_never_the_headline(
        tmp_path, monkeypatch, clean):
    """Both children carry a clean arm and a better-scoring answer-feedback
    arm. The suite's derived answer-feedback row stays readable and never
    heads the page, exactly as on a page the source reported itself."""
    facts = []
    for bench in (None, "mmlu-b"):
        facts += [
            (bench, "accuracy", clean, 0.50, False),
            (bench, "accuracy", _ANSWER_FEEDBACK, 0.90, False),
        ]
    got = _headlines(tmp_path, monkeypatch, facts,
                     benchmark_id="mmlu-suite", suite=True)
    assert got == {"accuracy": (clean, 0.50)}


def test_a_derived_unread_arm_sorts_after_a_clean_one(tmp_path, monkeypatch):
    """An arm whose feedback could not be read scores better on both
    children. The clean arm still heads the suite."""
    facts = []
    for bench in (None, "mmlu-b"):
        facts += [
            (bench, "accuracy", _LOW, 0.50, False),
            (bench, "accuracy", _UNKNOWN, 0.90, False),
        ]
    got = _headlines(tmp_path, monkeypatch, facts,
                     benchmark_id="mmlu-suite", suite=True)
    assert got == {"accuracy": (_LOW, 0.50)}


def test_the_run_condition_comes_from_the_default_metric_not_from_own_rows(
        tmp_path, monkeypatch):
    """The suite reports only cost itself; its accuracy is derived. Accuracy
    is the page's default metric, so the run condition is accuracy's best
    arm, and the suite's own cost row under that arm is the one shown."""
    facts = [
        ("mmlu-suite", "cost-per-task", _LOW, 0.80, True),
        ("mmlu-suite", "cost-per-task", _XHIGH, 0.10, True),
    ]
    for bench in (None, "mmlu-b"):
        facts += [
            (bench, "accuracy", _LOW, 0.90, False),
            (bench, "accuracy", _XHIGH, 0.50, False),
        ]
    got = _headlines(tmp_path, monkeypatch, facts,
                     benchmark_id="mmlu-suite", suite=True)
    assert got == {"accuracy": (_LOW, 0.90), "cost-per-task": (_LOW, 0.80)}


def test_the_run_condition_is_chosen_among_clean_arms_first(tmp_path,
                                                            monkeypatch):
    """The model's only arm on the page's first metric is one we could not
    read. The run condition is the best clean arm on the next metric that
    has one, so the remaining metrics all show that one run."""
    got = _headlines(tmp_path, monkeypatch, [
        (None, "accuracy", _UNKNOWN, 0.50, False),
        (None, "cost-per-task", _LOW, 0.10, True),
        (None, "cost-per-task", _XHIGH, 0.80, True),
        (None, "exact-match", _LOW, 0.50, False),
        (None, "exact-match", _XHIGH, 0.90, False),
    ])
    assert got == {"accuracy": (_UNKNOWN, 0.50),
                   "cost-per-task": (_LOW, 0.10),
                   "exact-match": (_LOW, 0.50)}


def test_an_unread_own_row_does_not_set_the_run_condition_of_derived_rows(
        tmp_path, monkeypatch):
    """The suite reports accuracy itself under an arm we could not read; its
    derived cost has that arm and a clean one. The clean one heads cost."""
    facts = [("mmlu-suite", "accuracy", _UNKNOWN, 0.70, False)]
    for bench in (None, "mmlu-b"):
        facts += [
            (bench, "cost-per-task", _UNKNOWN, 0.10, True),
            (bench, "cost-per-task", _LOW, 0.80, True),
        ]
    got = _headlines(tmp_path, monkeypatch, facts,
                     benchmark_id="mmlu-suite", suite=True)
    assert got == {"accuracy": (_UNKNOWN, 0.70), "cost-per-task": (_LOW, 0.80)}


def test_a_reported_row_heads_its_cell_whatever_the_run_condition(
        tmp_path, monkeypatch):
    """The suite reports accuracy for m1 itself, under xhigh. The page's
    default metric is the derived exact-match, where m1 ran low, and the
    children also give a derived low accuracy. The cell the suite reported
    keeps its reported row as headline: it stays ranked, and the fact-grain
    map still has a headline fact for it."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)

        def fact(bench, metric, protocol, s_, model):
            _clone_fact(con, level="root", is_part=False, score=s_,
                        slice_key=None, model=model, benchmark=bench,
                        metric=metric, protocol=protocol)

        fact("mmlu-suite", "accuracy", _XHIGH, 0.80, "m1")
        for model in ("m2", "m3"):
            fact("mmlu-suite", "accuracy", None, 0.30, model)
        for bench in (None, "mmlu-b"):
            fact(bench, "exact-match", _LOW, 0.60, "m1")
            for model in ("m2", "m3", "m4"):
                fact(bench, "exact-match", None, 0.60, model)
            fact(bench, "accuracy", _LOW, 0.20, "m1")

    con = _materialise_view(out, mutate=mutate)
    rows = con.execute(
        "SELECT protocol_condition, round(score, 2), value_level, is_headline, "
        "       position, total FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite' AND metric_id = 'accuracy' "
        "AND model_key = 'm1' ORDER BY 1"
    ).fetchall()
    assert rows == [
        (_LOW, 0.20, "derived", False, None, None),
        (_XHIGH, 0.80, "whole", True, 1, 3),
    ]
    assert con.execute(
        "SELECT count(*) FROM fact_results f JOIN fact_headline h USING (fact_id) "
        "WHERE h.is_headline AND f.benchmark_key = 'mmlu-suite' "
        "AND f.metric_key = 'accuracy' AND f.model_aggregation_key = 'm1'"
    ).fetchone()[0] == 1


_JUDGE_A = '{"judges":["judge-a"]}'
_JUDGE_B = '{"judges":["judge-b"]}'


def test_judge_coverage_counts_the_rows_the_sources_reported(tmp_path,
                                                             monkeypatch):
    """m1 is reported under two judges. Judge A also reads m4 in a reported
    row; judge B reads m2 and m3 only in rows derived from the children.
    Coverage is a count of reported readings, so judge A heads m1's cell."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)

        def fact(bench, judge, s_, model):
            _clone_fact(con, level="root", is_part=False, score=s_,
                        slice_key=None, model=model, benchmark=bench,
                        judge=judge)

        fact("mmlu-suite", _JUDGE_A, 0.50, "m1")
        fact("mmlu-suite", _JUDGE_B, 0.70, "m1")
        fact("mmlu-suite", _JUDGE_A, 0.60, "m4")
        for model in ("m2", "m3"):
            for bench in (None, "mmlu-b"):
                fact(bench, _JUDGE_B, 0.40, model)

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT model_key, judge_condition FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu-suite' AND is_headline ORDER BY 1"
    ).fetchall()
    assert rows == [("m1", _JUDGE_A), ("m2", _JUDGE_B), ("m3", _JUDGE_B),
                    ("m4", _JUDGE_A)]


def test_derived_rows_under_two_judges_headline_the_first_by_name(
        tmp_path, monkeypatch):
    """A derived row's scale label is its first child's, so it says nothing
    about the judge. Between two derived judge rows the first by name heads
    the cell, whichever children happened to publish on the registry scale."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        for bench in (None, "mmlu-b"):
            _clone_fact(con, level="root", is_part=False, score=40.0,
                        score_canonical=0.40, scale_conversion="div100",
                        slice_key=None, model="m1", benchmark=bench,
                        judge=_JUDGE_A)
            _clone_fact(con, level="root", is_part=False, score=0.60,
                        slice_key=None, model="m1", benchmark=bench,
                        judge=_JUDGE_B)

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT judge_condition, scale_conversion, is_headline "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu-suite' ORDER BY 1"
    ).fetchall()
    assert rows == [(_JUDGE_A, "div100", True), (_JUDGE_B, "none", False)]


def test_the_view_build_leaves_no_working_table_behind(tmp_path, monkeypatch):
    """Every working table of the view build is dropped once read."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        # a derived suite row, so the parent-row tables are exercised too
        _wipe_mmlu(con)
        _make_suite(con)
        for bench in (None, "mmlu-b"):
            _clone_fact(con, level="root", is_part=False, score=0.5,
                        slice_key=None, model="m1", benchmark=bench)

    con = _materialise_view(out, mutate=mutate)
    assert con.execute(
        "SELECT count(*) FROM eval_results_view WHERE value_level = 'derived'"
    ).fetchone()[0] == 1
    left = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name LIKE '\\_erv\\_%' ESCAPE '\\' ORDER BY 1"
    ).fetchall()]
    assert left == []


def test_a_derived_row_is_summed_in_value_order(tmp_path, monkeypatch):
    """Three children whose scores do not add to the same double in every
    order, in each of the six ways to hand them out. The suite's published,
    canonical and normalised values are summed smallest value first,
    whichever child holds which score and whatever order the rows reach the
    rollup in, so a rebuild publishes the same bytes."""
    import itertools

    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    children = (None, "mmlu-b", "mmlu-c")
    perms = list(itertools.permutations((0.1, 0.2, 0.3)))

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)
        _add_suite_child(con, "mmlu-c", "MMLU C")
        for i, scores in enumerate(perms):
            # facts inserted last child first
            for bench, s_ in list(zip(children, scores))[::-1]:
                _clone_fact(con, level="root", is_part=False, score=s_,
                            slice_key=None, model=f"m{i}", benchmark=bench)

    rows = _materialise_view(out, mutate=mutate).execute(
        "SELECT model_key, score, score_canonical, score_normalized "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu-suite' ORDER BY 1"
    ).fetchall()
    assert rows == [
        (f"m{i}",) + (sum(sorted((a, b, c))) / 3,) * 3
        for i, (a, b, c) in enumerate(perms)
    ]


def _mixed_suite_page(tmp_path, monkeypatch):
    """`mmlu-suite` with own and derived rows on one page. The suite reports
    `accuracy` itself for m1, m2 and m3, and `exact-match` itself for m1
    only; m2, m3 and m4 get a derived `exact-match`. Counting the suite's
    own facts alone puts `accuracy` first (3 models to 1); the page holds
    `exact-match` for four models and `accuracy` for three."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _make_suite(con)

        def fact(bench, metric, protocol, s_, model):
            _clone_fact(con, level="root", is_part=False, score=s_,
                        slice_key=None, model=model, benchmark=bench,
                        metric=metric, protocol=protocol)

        fact("mmlu-suite", "exact-match", _LOW, 0.90, "m1")
        fact("mmlu-suite", "exact-match", _XHIGH, 0.50, "m1")
        fact("mmlu-suite", "accuracy", _LOW, 0.20, "m1")
        fact("mmlu-suite", "accuracy", _XHIGH, 0.80, "m1")
        for model in ("m2", "m3"):
            fact("mmlu-suite", "accuracy", None, 0.30, model)
        for model in ("m2", "m3", "m4"):
            for bench in (None, "mmlu-b"):
                fact(bench, "exact-match", None, 0.60, model)

    return _materialise_view(out, mutate=mutate)


def test_own_cells_take_the_run_condition_of_the_pages_default_metric(
        tmp_path, monkeypatch):
    """The page's default metric counts the derived rows the page publishes,
    so it is `exact-match`, and m1's run condition is its best `exact-match`
    arm. m1's `accuracy` shows that same run, not its own best arm."""
    con = _mixed_suite_page(tmp_path, monkeypatch)
    rows = con.execute(
        "SELECT metric_id, protocol_condition, round(score, 2) "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu-suite' "
        "AND model_key = 'm1' AND is_headline ORDER BY 1"
    ).fetchall()
    assert rows == [("accuracy", _LOW, 0.20), ("exact-match", _LOW, 0.90)]


def test_the_run_condition_metric_is_the_pages_primary_metric(tmp_path,
                                                              monkeypatch):
    """The metric a run condition is chosen on and the metric `evals_view`
    shows by default are one metric on every page, including one that mixes
    rows the suite reported with rows derived from its children."""
    from eval_card_backend.canonicalise import stages

    # the metric order lives only while the headline is being picked, so it
    # is read at the moment the pick returns
    first = {}
    real_pick = stages._materialise_headline_pick

    def recording_pick(con):
        real_pick(con)
        first.update(con.execute(
            "SELECT composite_slug || '/' || benchmark_key, metric_key "
            "FROM _erv_page_metric_rank WHERE _metric_rank = 1"
        ).fetchall())

    monkeypatch.setattr(stages, "_materialise_headline_pick", recording_pick)
    con = _mixed_suite_page(tmp_path, monkeypatch)
    stages.stage_j_models_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")
    primary = dict(con.execute(
        "SELECT composite_slug || '/' || benchmark_id, primary_metric_id "
        "FROM evals_view WHERE primary_metric_id IS NOT NULL"
    ).fetchall())
    assert first == primary
    suite = [m for page, m in primary.items() if page.endswith("/mmlu-suite")]
    assert suite == ["exact-match"]


def test_every_page_cell_has_exactly_one_headline_row(tmp_path, monkeypatch):
    """Own and derived rows on one page still give each (metric, model) one
    headline row."""
    con = _mixed_suite_page(tmp_path, monkeypatch)
    counts = con.execute(
        "SELECT metric_id, model_key, count(*) FILTER (WHERE is_headline) "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu-suite' "
        "GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    assert counts == [
        ("accuracy", "m1", 1), ("accuracy", "m2", 1), ("accuracy", "m3", 1),
        ("exact-match", "m1", 1), ("exact-match", "m2", 1),
        ("exact-match", "m3", 1), ("exact-match", "m4", 1),
    ]


# ---------------------------------------------------------------------------
# Scale classes never pool (H2)
# ---------------------------------------------------------------------------


def test_published_scores_are_never_medianed_across_scale_classes(tmp_path,
                                                                  monkeypatch):
    """One source publishes a win rate as 1, another as 100. Their published
    median is 50.5 — a number on no scale at all. The cell publishes the
    canonical value instead and says the published scale is `mixed`."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _clone_fact(con, level="root", is_part=False, score=1.0, slice_key=None, model="m1",
                    scale_conversion="none", score_canonical=1.0)
        _clone_fact(con, level="root", is_part=False, score=100.0, slice_key=None, model="m1",
                    scale_conversion="div100", score_canonical=1.0)

    row = _materialise_view(out, mutate=mutate).execute(
        "SELECT score, score_published, score_canonical, scale_conversion, "
        "       score_normalized "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()
    score, published, canonical, conversion, normalized = row
    assert conversion == "mixed"
    assert abs(score - 1.0) < 1e-9
    assert abs(canonical - 1.0) < 1e-9
    # no single number was published for this cell
    assert published is None
    assert abs(normalized - 1.0) < 1e-9


def test_mixed_scale_pool_on_a_lower_is_better_metric(tmp_path, monkeypatch):
    """Same rule with the direction flipped: the canonical value is still the
    one shown, and the normalised twin still inverts."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        _clone_fact(con, level="root", is_part=False, score=0.2, slice_key=None, model="m1",
                    scale_conversion="none", score_canonical=0.2,
                    lower_is_better=True)
        _clone_fact(con, level="root", is_part=False, score=20.0, slice_key=None, model="m1",
                    scale_conversion="div100", score_canonical=0.2,
                    lower_is_better=True)

    row = _materialise_view(out, mutate=mutate).execute(
        "SELECT score, score_published, score_canonical, scale_conversion, "
        "       lower_is_better, score_normalized "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()
    score, published, canonical, conversion, lower, normalized = row
    assert conversion == "mixed"
    assert lower is True
    assert abs(score - 0.2) < 1e-9
    assert abs(canonical - 0.2) < 1e-9
    assert published is None
    assert abs(normalized - 0.8) < 1e-9


def test_one_scale_class_still_publishes_the_source_number(tmp_path, monkeypatch):
    """The guard is only about DISAGREEING classes: a pool on one scale keeps
    publishing the source's own number, as it always has."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        for value in (40.0, 60.0):
            _clone_fact(con, level="root", is_part=False, score=value, slice_key=None,
                        model="m1", scale_conversion="div100",
                        score_canonical=value / 100)

    row = _materialise_view(out, mutate=mutate).execute(
        "SELECT score, score_published, score_canonical, scale_conversion "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()
    score, published, canonical, conversion = row
    assert conversion == "div100"
    assert abs(score - 50.0) < 1e-9
    assert abs(published - 50.0) < 1e-9
    assert abs(canonical - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Labels on a pooled cell
# ---------------------------------------------------------------------------


def test_pooled_cell_label_is_null_when_the_raw_names_differ(tmp_path, monkeypatch):
    """A median over rows the source named differently has no one label."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        for s, lbl in ((0.40, "flexible-extract"), (0.60, "strict-match")):
            _clone_fact(con, level="root", is_part=False, score=s, slice_key=None)
            con.execute(
                "UPDATE fact_results SET metric_source_label = ? "
                "WHERE score = ? AND benchmark_key = 'mmlu'", [lbl, s],
            )

    _score, _lvl, _n, _c, label, _se = _cell(
        _materialise_view(out, mutate=mutate)
    )[0]
    assert label is None


def test_pooled_cell_keeps_a_label_the_rows_share(tmp_path, monkeypatch):
    """When every pooled row carries the same label, it still describes the
    number and stays on the row."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _wipe_mmlu(con)
        for s in (0.40, 0.60):
            _clone_fact(con, level="root", is_part=False, score=s, slice_key=None)
        con.execute(
            "UPDATE fact_results SET metric_source_label = 'acc' "
            "WHERE benchmark_key = 'mmlu'"
        )

    _score, _lvl, _n, _c, label, _se = _cell(
        _materialise_view(out, mutate=mutate)
    )[0]
    assert label == "acc"

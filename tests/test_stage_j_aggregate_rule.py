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
        "CREATE TABLE canonical_benchmarks AS "
        "SELECT id, CAST(parent_benchmark_id AS VARCHAR) AS parent_benchmark_id, "
        "       CAST(metadata AS VARCHAR) AS metadata "
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
        "INSERT INTO canonical_benchmarks VALUES "
        "('mmlu-suite', NULL, NULL), ('mmlu-b', 'mmlu-suite', NULL)"
    )
    con.execute(
        "UPDATE canonical_benchmarks SET parent_benchmark_id = 'mmlu-suite' "
        "WHERE id = 'mmlu'"
    )


def _add_suite_child(con, child: str, display: str, role: str | None = None):
    """One more registry child of `mmlu-suite`, in both the dim and the
    registry tree.

    `role` stamps `canonical_benchmarks.metadata` — `"aggregate"` marks a
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
        f"INSERT INTO canonical_benchmarks VALUES "
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

"""The non-generative exemption: declarations, the rule, and what reads it.

The base reproducibility fields describe how text was sampled. A result
that sampled none has nothing to report for them, and the only thing
that says so is a statement by the evaluation's own source. These tests
cover the four places that matters: how a run is told which sources to
honour, that the SQL and the Python reading of the rule agree, that a
declaration can only land on the result it was made about, and that a
row the rule asks nothing of reads as outside the rule rather than as
answering it completely.
"""
from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import duckdb
import pytest

from eval_card_backend import datafix
from eval_card_backend.datafix.annotations import ANNOTATION_SCHEMA, _LmEvalOutputType
from eval_card_backend.canonicalise import sidecars, stages
from eval_card_backend.canonicalise.cache import STAGE_OUTPUTS, StageCache
from eval_card_backend.signals.reproducibility import (
    AGENTIC_REPRODUCIBILITY_FIELDS,
    BASE_REPRODUCIBILITY_FIELDS,
    DECLARATION_SOURCES,
    NON_GENERATIVE_VALUES,
    is_non_generative_py,
    parse_declaration_sources,
    required_repro_fields,
)
from tests.eee_layout import write_eee_datastore

FIXTURES = Path(__file__).parent / "fixtures"

LM_EVAL = "lm_eval_output_type"
REWARDBENCH = "rewardbench_model_type"


# ---------------------------------------------------------------------------
# Which sources a run honours
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", ",", " , "])
def test_an_unset_flag_honours_nothing(raw):
    assert parse_declaration_sources(raw) == frozenset()


def test_the_flag_is_a_comma_list_of_sources():
    assert parse_declaration_sources(LM_EVAL) == {LM_EVAL}
    assert parse_declaration_sources(
        f" {LM_EVAL} , {REWARDBENCH} "
    ) == {LM_EVAL, REWARDBENCH}


def test_an_unrecognised_source_is_a_typo_not_a_no_op():
    with pytest.raises(ValueError, match="lm_eval_outputtype"):
        parse_declaration_sources(f"{LM_EVAL},lm_eval_outputtype")


def test_the_settings_read_the_flag_from_the_environment(monkeypatch):
    from eval_card_backend.config import Settings

    monkeypatch.setenv("REPRO_EXEMPT_NON_GENERATIVE", REWARDBENCH)

    assert Settings.from_env().repro_exempt_declaration_sources == {REWARDBENCH}


# ---------------------------------------------------------------------------
# The values each source counts as non-generative
# ---------------------------------------------------------------------------


def test_the_reward_bench_types_are_matched_as_exact_strings():
    assert NON_GENERATIVE_VALUES[REWARDBENCH] == {
        "Seq. Classifier", "Custom Classifier", "DPO",
    }


@pytest.mark.parametrize("model_type", [
    "Generative", "Generative RM", "dpo", "seq. classifier",
    "Seq Classifier", "Seq. Classifier ", "Classifier", "", None,
])
def test_only_the_listed_reward_bench_types_are_exempt(model_type):
    assert not is_non_generative_py({REWARDBENCH}, REWARDBENCH, model_type)


@pytest.mark.parametrize("model_type", sorted(NON_GENERATIVE_VALUES[REWARDBENCH]))
def test_the_listed_reward_bench_types_are_exempt(model_type):
    assert is_non_generative_py({REWARDBENCH}, REWARDBENCH, model_type)


def test_generate_until_is_generative():
    assert not is_non_generative_py({LM_EVAL}, LM_EVAL, "generate_until")
    assert is_non_generative_py({LM_EVAL}, LM_EVAL, "multiple_choice")


def test_a_source_the_run_does_not_honour_changes_nothing():
    assert not is_non_generative_py({LM_EVAL}, REWARDBENCH, "DPO")
    assert not is_non_generative_py(frozenset(), LM_EVAL, "multiple_choice")


def test_harness_declarations_match_precision_and_prefer_record_evidence():
    class Evidence:
        fix_id = "test-harness"
        source_config_revision = "harness-revision"
        table_sha256 = "table-hash"

        def config_for(self, model, benchmark, precision):
            assert (model, benchmark) == ("org/model", "task")
            if precision == "float16":
                return {"output_type": "multiple_choice"}
            return None

    declarer = _LmEvalOutputType(Evidence())
    result = {"evaluation_name": "task"}
    record = {"model_info": {"id": "org/model", "additional_details": {
        "precision": "float16"}}, "evaluation_results": [result]}
    assert list(declarer.declare(record)) == [(0, "multiple_choice")]
    assert declarer.provenance(result, "eee-revision")["source_revision"] == "harness-revision"
    record["model_info"]["additional_details"]["precision"] = "bfloat16"
    assert list(declarer.declare(record)) == []
    result["generation_config"] = {"additional_details": {"output_type": "generate_until"}}
    assert list(declarer.declare(record)) == [(0, "generate_until")]
    assert declarer.provenance(result, "eee-revision") == {
        "source_fix_id": None, "source_revision": "eee-revision", "table_sha256": None,
    }


# ---------------------------------------------------------------------------
# One rule, two readings: the Stage E SQL and the Python helper
# ---------------------------------------------------------------------------

_DECLARATIONS = [
    (None, None),
    (LM_EVAL, "multiple_choice"),
    (LM_EVAL, "loglikelihood"),
    (LM_EVAL, "loglikelihood_rolling"),
    (LM_EVAL, "generate_until"),
    (LM_EVAL, "something_else"),
    (REWARDBENCH, "Seq. Classifier"),
    (REWARDBENCH, "Custom Classifier"),
    (REWARDBENCH, "DPO"),
    (REWARDBENCH, "Generative"),
    (REWARDBENCH, "Generative RM"),
]

_HONOURED = [
    frozenset(),
    frozenset({LM_EVAL}),
    frozenset({REWARDBENCH}),
    frozenset(DECLARATION_SOURCES),
]

_ALL_FIELDS = BASE_REPRODUCIBILITY_FIELDS + AGENTIC_REPRODUCIBILITY_FIELDS


def _rule_cases():
    """Every declaration against every honoured set, agentic or not, with
    each required field present and absent."""
    for honoured in _HONOURED:
        for source, value in _DECLARATIONS:
            for is_agentic in (False, True):
                for present in itertools.product(
                    (False, True), repeat=len(_ALL_FIELDS)
                ):
                    yield honoured, source, value, is_agentic, dict(
                        zip(_ALL_FIELDS, present)
                    )


def _sql_rule(honoured, rows) -> list[dict]:
    """The rule as Stage E computes it, from the real SQL builders."""
    con = duckdb.connect()
    flags = ", ".join(f"has_{f} BOOLEAN" for f in _ALL_FIELDS)
    con.execute(
        f"CREATE TABLE fact_results_staging ("
        f"row_id INTEGER, declared_output_source VARCHAR, "
        f"declared_output_value VARCHAR, is_agentic BOOLEAN, {flags})"
    )
    con.executemany(
        f"INSERT INTO fact_results_staging VALUES "
        f"(?, ?, ?, ?, {', '.join('?' * len(_ALL_FIELDS))})",
        rows,
    )
    non_generative_sql = stages._build_non_generative_sql(honoured)
    return con.execute(
        f"""
        WITH base AS (
            SELECT *,
                ({non_generative_sql}) AS is_non_generative,
                NOT ({non_generative_sql}) AS repro_base_applicable
            FROM fact_results_staging
        ),
        scored AS (
            SELECT *,
                ({stages._REPRO_MISSING_FIELDS_SQL}) AS repro_missing_fields,
                ({stages._REPRO_REQUIRED_COUNT_SQL}) AS repro_required_count
            FROM base
        )
        SELECT row_id, is_non_generative, repro_base_applicable,
               repro_missing_fields, repro_required_count,
               repro_required_count - len(repro_missing_fields)
                   AS repro_populated_count,
               len(repro_missing_fields) > 0 AS has_reproducibility_gap,
               repro_required_count > 0      AS repro_applicable
        FROM scored ORDER BY row_id
        """
    ).fetchall()


def test_the_sql_and_the_python_reading_of_the_rule_agree():
    cases = list(_rule_cases())
    by_honoured: dict[frozenset, list] = {}
    for idx, (honoured, source, value, is_agentic, present) in enumerate(cases):
        by_honoured.setdefault(honoured, []).append(
            (idx, source, value, is_agentic, *[present[f] for f in _ALL_FIELDS])
        )

    for honoured, rows in by_honoured.items():
        for row, sql in zip(rows, _sql_rule(honoured, rows)):
            idx, source, value, is_agentic = row[0], row[1], row[2], row[3]
            present = dict(zip(_ALL_FIELDS, row[4:]))

            non_generative = is_non_generative_py(honoured, source, value)
            required = required_repro_fields(
                is_agentic, base_applicable=not non_generative
            )
            missing = [f for f in required if not present[f]]
            expected = (
                idx,
                non_generative,
                not non_generative,
                missing,
                len(required),
                len(required) - len(missing),
                bool(missing),
                len(required) > 0,
            )
            assert sql == expected, (honoured, source, value, is_agentic, present)


def test_a_non_generative_agentic_row_still_owes_the_agentic_fields():
    assert required_repro_fields(True, base_applicable=False) == (
        AGENTIC_REPRODUCIBILITY_FIELDS
    )
    assert required_repro_fields(False, base_applicable=False) == ()


@pytest.mark.parametrize("honoured, is_agentic, expected", [
    (frozenset(), False, 2),
    (frozenset(), True, 4),
    (frozenset(DECLARATION_SOURCES), False, 0),
    (frozenset(DECLARATION_SOURCES), True, 2),
])
def test_the_required_count_is_two_four_or_zero(honoured, is_agentic, expected):
    row = (0, REWARDBENCH, "DPO", is_agentic, *[True] * len(_ALL_FIELDS))
    assert _sql_rule(honoured, [row])[0][4] == expected


# ---------------------------------------------------------------------------
# A declaration may only land on the result it was made about
# ---------------------------------------------------------------------------


def _annotation(**overrides) -> dict:
    row = {
        "source_record_path": "data/reward-bench/dev/model/a.json",
        "result_idx": 0,
        "evaluation_id": "ev_1",
        "evaluation_result_id": "ev_1:r0",
        "evaluation_name": "chat",
        "model_id": "org/model",
        "declaration_source": REWARDBENCH,
        "declared_value": "DPO",
        "source_fix_id": None,
        "source_revision": "abc123",
        "table_sha256": None,
    }
    row.update(overrides)
    return row


def _annotated_connection(annotations, *, members=()) -> duckdb.DuckDBPyConnection:
    """A connection carrying what the Stage D checks read: the facts, the
    exploded rows they came from, and the collection manifest."""
    con = duckdb.connect()
    stages._ensure_result_annotations(con)
    for row in annotations:
        con.execute(
            "INSERT INTO eee_result_annotations VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [row[k] for k in (
                "source_record_path", "result_idx", "evaluation_id",
                "evaluation_result_id", "evaluation_name", "model_id",
                "declaration_source", "declared_value", "source_fix_id",
                "source_revision", "table_sha256",
            )],
        )
    con.execute(
        """
        CREATE TABLE results_exploded AS
        SELECT 'data/reward-bench/dev/model/a.json' AS source_record_path,
               0 AS result_idx, 'ev_1' AS evaluation_id,
               'ev_1:r0' AS evaluation_result_id_raw, 'chat' AS evaluation_name,
               {'id': 'org/model'} AS model_info
        """
    )
    # The staged fact, carrying whatever the join found for its own key.
    con.execute(
        """
        CREATE TABLE fact_results_staging AS
        SELECT r.*, a.declaration_source AS declared_output_source,
               a.declared_value          AS declared_output_value
        FROM results_exploded r
        LEFT JOIN (SELECT * FROM eee_result_annotations LIMIT 1) a
               ON a.source_record_path = r.source_record_path
              AND a.result_idx         = r.result_idx
        """
    )
    con.execute(
        "CREATE TABLE collection_member_ids (evaluation_id VARCHAR, "
        "collection_id VARCHAR)"
    )
    for evaluation_id in members:
        con.execute(
            "INSERT INTO collection_member_ids VALUES (?, 'some-collection')",
            [evaluation_id],
        )
    return con


def test_a_matching_declaration_passes_the_checks():
    stages._assert_result_annotations(_annotated_connection([_annotation()]))


def test_two_declarations_on_one_result_are_refused_at_load():
    import pyarrow as pa

    rows = [_annotation(), _annotation(declaration_source=LM_EVAL,
                                       declared_value="multiple_choice")]
    with pytest.raises(RuntimeError, match="more than one"):
        stages.stage_a_load_result_annotations(
            duckdb.connect(),
            pa.Table.from_pylist(rows, schema=ANNOTATION_SCHEMA),
        )


@pytest.mark.parametrize("field, wrong", [
    ("evaluation_id", "ev_other"),
    ("evaluation_result_id", "ev_1:r9"),
    ("evaluation_name", "reasoning"),
    ("model_id", "org/other-model"),
])
def test_a_declaration_that_disagrees_with_its_result_is_refused(field, wrong):
    con = _annotated_connection([_annotation(**{field: wrong})])

    with pytest.raises(RuntimeError, match="disagree"):
        stages._assert_result_annotations(con)


def test_a_declaration_addressing_a_result_that_is_not_there_is_refused():
    con = _annotated_connection([_annotation(source_record_path="data/x.json")])

    with pytest.raises(RuntimeError, match="not in the corpus"):
        stages._assert_result_annotations(con)


def test_a_declaration_on_a_rewritten_evaluation_is_refused():
    con = _annotated_connection([_annotation()], members=["ev_1"])

    with pytest.raises(RuntimeError, match="collection adapter"):
        stages._assert_result_annotations(con)


def test_a_key_that_is_not_unique_would_fan_the_fact_grain_out():
    con = _annotated_connection([_annotation(), _annotation()])

    with pytest.raises(RuntimeError, match="not unique"):
        stages._assert_result_annotations(con)


def test_a_declaration_that_did_not_land_on_its_own_row_is_refused():
    """The join columns and the emitted columns have to stay the same
    pair; a staged row carrying someone else's declaration is caught
    whatever went wrong upstream of it."""
    con = _annotated_connection([_annotation()])
    con.execute(
        "CREATE OR REPLACE TABLE fact_results_staging AS "
        "SELECT *, 'lm_eval_output_type' AS declared_output_source, "
        "'multiple_choice' AS declared_output_value FROM results_exploded"
    )

    with pytest.raises(RuntimeError, match="does not name"):
        stages._assert_result_annotations(con)


def test_a_flatten_that_changed_the_fact_grain_is_refused():
    """A duplicated dimension id fans a fact out through an ordinary
    join, which no annotation check can see."""
    con = _annotated_connection([_annotation()])

    with pytest.raises(RuntimeError, match="changed the fact grain"):
        stages._assert_flatten_kept_the_row_count(con, 2)


def test_a_flatten_that_kept_the_fact_grain_passes():
    stages._assert_flatten_kept_the_row_count(
        _annotated_connection([_annotation()]), 1
    )


def test_a_run_carrying_no_declarations_keeps_the_grain_it_had():
    stages._assert_flatten_kept_the_row_count(_annotated_connection([]), 99)


# ---------------------------------------------------------------------------
# The rule a cached stage output was computed under
# ---------------------------------------------------------------------------


def _cache_inputs(**overrides) -> dict:
    base = {
        "eee_revision": "abc123",
        "configs": ["reward-bench"],
        "datafix_enabled": True,
        "datafix_fingerprint": datafix.fingerprint(),
        "repro_exempt_declaration_sources": [],
    }
    base.update(overrides)
    return base


def _blessed_cache(tmp_path, inputs) -> None:
    con = duckdb.connect()
    for table in STAGE_OUTPUTS["A"]:
        con.execute(f"CREATE TABLE {table} AS SELECT 1 AS x")
    cache = StageCache(tmp_path, "2026-09-20T00:00:00Z")
    cache.stage_a_inputs = inputs
    cache.datafix_summary = {
        "enabled": True, "fixes": {}, "fingerprint": datafix.fingerprint(),
    }
    cache.write_stage(con, "A")


def test_a_cache_built_under_another_rule_refuses_to_resume(tmp_path):
    _blessed_cache(tmp_path, _cache_inputs())
    cache = StageCache(tmp_path, "2026-09-20T00:00:00Z")
    cache.stage_a_inputs = _cache_inputs(
        repro_exempt_declaration_sources=[LM_EVAL]
    )

    with pytest.raises(RuntimeError, match="repro_exempt_declaration_sources"):
        cache.restore_through(duckdb.connect(), "A")


def test_a_cache_built_under_the_same_rule_resumes(tmp_path):
    _blessed_cache(tmp_path, _cache_inputs(
        repro_exempt_declaration_sources=[LM_EVAL]
    ))
    cache = StageCache(tmp_path, "2026-09-20T00:00:00Z")
    cache.stage_a_inputs = _cache_inputs(
        repro_exempt_declaration_sources=[LM_EVAL]
    )

    assert set(cache.restore_through(duckdb.connect(), "A")) == set(
        STAGE_OUTPUTS["A"]
    )


# ---------------------------------------------------------------------------
# What a zero-requirement row reads as downstream
# ---------------------------------------------------------------------------


def _reward_bench_record(eval_id: str, model_type: str | None, *, names) -> dict:
    model_info: dict = {"name": eval_id, "id": f"org/{eval_id}"}
    if model_type is not None:
        model_info["additional_details"] = {"model_type": model_type}
    return {
        "schema_version": "0.2.2",
        "evaluation_id": eval_id,
        "retrieved_timestamp": "2026-09-01T00:00:00Z",
        "source_metadata": {
            "source_type": "evaluation_run",
            "source_organization_name": "Allen Institute for AI",
            "source_name": "RewardBench",
            "evaluator_relationship": "third_party",
        },
        "model_info": model_info,
        "eval_library": {"name": "reward-bench", "version": "0.1.0"},
        "evaluation_results": [
            {
                "evaluation_result_id": f"{eval_id}:{name}",
                "evaluation_name": name,
                "source_data": {"dataset_name": name, "source_type": "other"},
                "metric_config": {"metric_name": "accuracy",
                                  "lower_is_better": False},
                "score_details": {"score": 0.5},
            }
            for name in names
        ],
    }


def _run(tmp_path, monkeypatch, honoured: str | None, snapshot: str,
         registry_dir: Path | None = None, extra_records=()) -> Path:
    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        # Declared non-generative: the rule asks nothing of these.
        ("reward-bench", "a.json",
         json.dumps(_reward_bench_record("ev_a", "Seq. Classifier",
                                         names=["chat", "safety"]))),
        # Declared generative: the rule is unchanged for it, and it
        # states neither base field.
        ("reward-bench", "b.json",
         json.dumps(_reward_bench_record("ev_b", "Generative",
                                         names=["chat"]))),
        # Undeclared: the rule is unchanged for it too.
        ("reward-bench", "c.json",
         json.dumps(_reward_bench_record("ev_c", None, names=["chat"]))),
        *extra_records,
    ])

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR",
                       str(FIXTURES / "auto_benchmarkcards"))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)
    if honoured is None:
        monkeypatch.delenv("REPRO_EXEMPT_NON_GENERATIVE", raising=False)
    else:
        monkeypatch.setenv("REPRO_EXEMPT_NON_GENERATIVE", honoured)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    return pipeline.run(
        Settings.from_env(),
        configs=["reward-bench"],
        snapshot_id=snapshot,
        warehouse_dir=str(tmp_path / "warehouse"),
        registry_local_dir=str(registry_dir or FIXTURES / "entity_registry"),
        cache_root=str(tmp_path / f"cache-{snapshot}"),
    )


@pytest.fixture
def exempt_run(tmp_path, monkeypatch) -> Path:
    return _run(tmp_path, monkeypatch, REWARDBENCH, "2026-04-30T00:00:01Z")


@pytest.fixture
def plain_run(tmp_path, monkeypatch) -> Path:
    return _run(tmp_path, monkeypatch, None, "2026-04-30T00:00:00Z")


def _read(out_dir: Path, name: str):
    con = duckdb.connect()
    return con.execute(
        f"SELECT * FROM read_parquet('{out_dir}/{name}.parquet')"
    ).fetchdf()


def test_a_declared_non_generative_row_owes_nothing(exempt_run):
    facts = _read(exempt_run, "fact_results").set_index("evaluation_id")

    assert facts.loc["ev_a", "repro_required_count"].tolist() == [0, 0]
    assert not facts.loc["ev_a", "repro_applicable"].any()
    assert not facts.loc["ev_a", "has_reproducibility_gap"].any()
    # Declared generative and undeclared rows are untouched.
    assert facts.loc["ev_b", "repro_required_count"] == 2
    assert facts.loc["ev_c", "repro_required_count"] == 2
    assert facts.loc["ev_b", "has_reproducibility_gap"]


def test_without_the_flag_every_row_owes_the_base_fields(plain_run):
    facts = _read(plain_run, "fact_results")

    assert set(facts["repro_required_count"]) == {2}
    assert facts["repro_applicable"].all()
    assert not facts["is_non_generative"].any()
    assert facts["has_reproducibility_gap"].all()


def test_the_view_says_whether_the_rule_applies_to_a_cell(exempt_run):
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT model_key,
               evalcards_annotations.reproducibility_gap.populated_count,
               evalcards_annotations.reproducibility_gap.required_count,
               evalcards_annotations.reproducibility_gap.applicable
        FROM read_parquet('{exempt_run}/eval_results_view.parquet')
        ORDER BY model_key, benchmark_id
        """
    ).fetchall()
    by_model = {r[0]: r[1:] for r in rows}

    assert by_model["org/ev_a"] == (0, 0, False)
    assert by_model["org/ev_b"] == (0, 2, True)


def test_a_model_the_rule_asks_nothing_of_is_not_complete(exempt_run):
    models = _read(exempt_run, "models_view").set_index("model_key")

    assert models.loc["org/ev_a", "reproducibility_status"] == "not_applicable"
    assert models.loc["org/ev_b", "reproducibility_status"] == "missing"
    assert models.loc["org/ev_c", "reproducibility_status"] == "missing"


def test_without_the_flag_no_model_is_outside_the_rule(plain_run):
    models = _read(plain_run, "models_view")

    assert set(models["reproducibility_status"]) == {"missing"}


def test_the_headline_counts_the_cells_the_rule_skipped(exempt_run):
    headline = json.loads((exempt_run / "headline.json").read_text())
    block = headline["reproducibility"]["overall"]

    assert headline["signal_version"] == "1.1+non-generative-exempt"
    # Two cells for ev_a, one each for ev_b and ev_c.
    assert block["total_triples"] == 4
    assert block["not_applicable_triples"] == 2
    assert block["triples_with_reproducibility_gap"] == 2
    assert block["reproducibility_gap_rate"] == 0.5
    assert block["applicable_triples"] == 2
    assert block["applicable_reproducibility_gap_rate"] == 1.0
    for field in BASE_REPRODUCIBILITY_FIELDS:
        per_field = block["per_field_missingness"][field]
        assert per_field["denominator"] == "base_applicable_triples"
        assert per_field["denominator_count"] == 2
        assert per_field["missing_count"] == 2
        assert per_field["missing_rate"] == 1.0


def test_without_the_flag_the_headline_is_the_rule_as_it_stands(plain_run):
    headline = json.loads((plain_run / "headline.json").read_text())
    block = headline["reproducibility"]["overall"]

    assert headline["signal_version"] == "1.1"
    assert block["total_triples"] == 4
    assert block["not_applicable_triples"] == 0
    assert block["triples_with_reproducibility_gap"] == 4
    for field in BASE_REPRODUCIBILITY_FIELDS:
        per_field = block["per_field_missingness"][field]
        assert per_field["denominator"] == "all_triples"
        assert per_field["denominator_count"] == 4


def test_the_rollup_reads_a_subset_of_rows_with_the_same_arithmetic(exempt_run):
    """Reading the gap rate over only the cells the rule applies to is the
    published rollup with a predicate, not a second calculation."""
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW fact_results AS SELECT * FROM "
        f"read_parquet('{exempt_run}/fact_results.parquet')"
    )
    published = json.loads((exempt_run / "headline.json").read_text())
    block = published["reproducibility"]["overall"]

    assert sidecars.reproducibility_rollup(con, None, exemption_on=True) == block

    applicable = sidecars.reproducibility_rollup(
        con, None, fact_predicate="fr.repro_required_count > 0", exemption_on=True
    )
    assert applicable["total_triples"] == 2
    assert applicable["not_applicable_triples"] == 0
    assert applicable["reproducibility_gap_rate"] == 1.0


def test_the_declarations_are_recorded_in_the_snapshot_meta(exempt_run):
    meta = json.loads((exempt_run / "snapshot_meta.json").read_text())
    exemption = meta["repro_exemption"]

    assert exemption["declaration_sources"] == [REWARDBENCH]
    assert exemption["declared"][REWARDBENCH]["by_value"] == {
        "Seq. Classifier": 2, "Generative": 1,
    }


def test_disabled_exemption_does_not_collect_declarations(plain_run):
    meta = json.loads((plain_run / "snapshot_meta.json").read_text())

    assert meta["repro_exemption"]["declaration_sources"] == []
    assert meta["repro_exemption"]["declared"] == {}


# ---------------------------------------------------------------------------
# A cell that carries no reproducibility reading at all
# ---------------------------------------------------------------------------


def _registry_with_a_parent_benchmark(tmp_path) -> Path:
    """The fixture registry plus a parent whose two children are the
    record's own cells, so Stage J derives a suite row over them.

    A derived row is a mean of its children's values. It is not backed by
    a result, so it carries no reproducibility reading.
    """
    reg = tmp_path / "registry-parent"
    reg.mkdir()
    for path in (FIXTURES / "entity_registry").glob("*.parquet"):
        shutil.copy(path, reg / path.name)
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE cb AS SELECT * REPLACE (CAST(parent_benchmark_id AS VARCHAR) "
        f"AS parent_benchmark_id) FROM read_parquet('{reg / 'canonical_benchmarks.parquet'}')"
    )
    for bid, name, parent in (("rb", "RewardBench", None),
                              ("rb-chat", "RewardBench Chat", "rb"),
                              ("rb-safety", "RewardBench Safety", "rb")):
        con.execute(
            "INSERT INTO cb BY NAME SELECT ? AS id, ? AS display_name, "
            "? AS parent_benchmark_id, 'reviewed' AS review_status, '{}' AS metadata, "
            "'[]' AS tags", [bid, name, parent],
        )
    con.execute(f"COPY cb TO '{reg / 'canonical_benchmarks.parquet'}' (FORMAT PARQUET)")
    con.execute(
        f"CREATE TABLE al AS SELECT * FROM read_parquet('{reg / 'aliases.parquet'}')"
    )
    for i, (raw, bid) in enumerate((("chat", "rb-chat"), ("safety", "rb-safety"))):
        con.execute(
            "INSERT INTO al BY NAME SELECT ? AS id, ? AS raw_value, "
            "'benchmark' AS entity_type, ? AS canonical_id, 'active' AS status, "
            "'exact' AS strategy, 1.0 AS confidence", [f"rb-alias-{i}", raw, bid],
        )
    con.execute(f"COPY al TO '{reg / 'aliases.parquet'}' (FORMAT PARQUET)")
    con.close()
    return reg


@pytest.fixture
def suite_exempt_run(tmp_path, monkeypatch) -> Path:
    return _run(tmp_path, monkeypatch, REWARDBENCH, "2026-04-30T00:00:02Z",
                _registry_with_a_parent_benchmark(tmp_path))


@pytest.fixture
def suite_plain_run(tmp_path, monkeypatch) -> Path:
    return _run(tmp_path, monkeypatch, None, "2026-04-30T00:00:03Z",
                _registry_with_a_parent_benchmark(tmp_path))


def _suite_rows(out_dir: Path) -> list[tuple]:
    con = duckdb.connect()
    return con.execute(
        f"""
        SELECT value_level, is_headline, has_reproducibility_gap,
               evalcards_annotations.reproducibility_gap.applicable
        FROM read_parquet('{out_dir}/eval_results_view.parquet')
        WHERE model_key = 'org/ev_a' AND benchmark_id = 'rb'
        """
    ).fetchall()


def test_a_derived_suite_row_carries_no_reading(suite_exempt_run):
    assert _suite_rows(suite_exempt_run) == [("derived", True, None, None)]


def test_a_model_read_only_through_exempt_cells_is_outside_the_rule(suite_exempt_run):
    """ev_a's two reward-bench cells are declared non-generative, and its
    third cell is the derived suite row, which says nothing either way.
    Nothing the rule applies to is left, so the model is outside it."""
    models = _read(suite_exempt_run, "models_view").set_index("model_key")

    assert models.loc["org/ev_a", "reproducibility_status"] == "not_applicable"


def test_without_the_flag_the_derived_row_reads_as_it_always_has(suite_plain_run):
    """With nothing honoured the band is the gap rate over every headline
    cell: ev_a's two cells have a gap and the derived row has none."""
    models = _read(suite_plain_run, "models_view").set_index("model_key")

    assert _suite_rows(suite_plain_run) == [("derived", True, None, None)]
    assert models.loc["org/ev_a", "reproducibility_status"] == "partial"



def test_mixed_applicability_is_reported_with_an_applicable_representative(caplog):
    con = duckdb.connect()
    con.execute("CREATE TABLE _erv_tri AS SELECT true AS triple_repro_applicable, "
                "false AS triple_all_repro_applicable, true AS rep_repro_applicable")
    stages._log_mixed_repro_applicability(con)
    assert "1 cell(s) hold rows that disagree" in caplog.text
    con.close()


@pytest.mark.parametrize("exempt_first", [True, False])
def test_a_mixed_cell_keeps_the_applicable_observations_gap(
    tmp_path, monkeypatch, exempt_first
):
    records = []
    for name, model_type in (
        ("mixed_a", "DPO" if exempt_first else "Generative"),
        ("mixed_z", "Generative" if exempt_first else "DPO"),
    ):
        record = _reward_bench_record(name, model_type, names=["chat"])
        record["model_info"]["id"] = "org/mixed"
        # Different evaluation times retain both observations after dedup.
        record["evaluation_timestamp"] = (
            "2026-01-01T00:00:00Z" if name.endswith("a") else "2026-01-02T00:00:00Z"
        )
        records.append(("reward-bench", f"{name}.json", json.dumps(record)))
    out = _run(tmp_path, monkeypatch, REWARDBENCH, "2026-04-30T00:00:04Z",
               extra_records=records)
    con = duckdb.connect()
    row = con.execute(
        f"SELECT has_reproducibility_gap, evalcards_annotations.reproducibility_gap "
        f"FROM read_parquet('{out}/eval_results_view.parquet') "
        "WHERE model_key = 'org/mixed'"
    ).fetchone()
    assert row == (True, {
        "missing_fields": ["max_tokens", "temperature"],
        "populated_count": 0, "required_count": 2, "applicable": True,
    })
    assert con.execute(
        f"SELECT reproducibility_status FROM read_parquet('{out}/models_view.parquet') "
        "WHERE model_key = 'org/mixed'"
    ).fetchone() == ("missing",)
    con.close()


def test_an_all_exempt_population_has_no_applicable_gap_rate(exempt_run):
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW fact_results AS SELECT * FROM "
        f"read_parquet('{exempt_run}/fact_results.parquet')"
    )
    block = sidecars.reproducibility_rollup(
        con, None, fact_predicate="fr.repro_required_count = 0", exemption_on=True
    )
    assert block["applicable_triples"] == 0
    assert block["applicable_reproducibility_gap_rate"] is None
    con.close()

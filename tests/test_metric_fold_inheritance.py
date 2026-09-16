"""A benchmark-level naming fold applies to the benchmark's slice children.

`benchmark_metric_folds` rows with no `source_config` rename a metric on one
benchmark (`global-mmlu-lite`: `score -> accuracy`). Stage C keyed them on
the resolved benchmark id, and a slice child is not the parent's id, so once
Global-MMLU-Lite's 18 language rows resolved to `global-mmlu-lite-<lang>`
they kept `score` while the totals folded to `accuracy`, and the page forked
into two metrics. A child is the same benchmark's data: a row on a child with
no naming row of its own for that metric inherits the parent's. A child's own
fold wins; scoped scale-conversion rows are never inherited; one level only.

The first half runs `_apply_metric_folds` on hand-built tables; the second
runs the real pipeline over a temporary registry (fixture registry + a
parent with two children carrying the parent's fold) and a temporary EEE
tree, and checks the derived parent row lands on `accuracy`.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import duckdb
import pytest

from tests.eee_layout import write_eee_datastore


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Stage C, on hand-built tables
# ---------------------------------------------------------------------------


def _con_with(rows, folds, parents, roles=None):
    """`results_resolved` rows as (benchmark_id, metric_id); folds as
    (benchmark_id, from, to, source_config); parents as (id, parent); roles
    as {id: metadata.role}."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE results_resolved (benchmark_id VARCHAR, metric_id VARCHAR, "
        "metric_raw VARCHAR, metric_config STRUCT(metric_name VARCHAR, "
        "metric_id VARCHAR, additional_details MAP(VARCHAR, VARCHAR)))"
    )
    for b, m in rows:
        con.execute(
            "INSERT INTO results_resolved VALUES (?, ?, ?, "
            "{'metric_name': ?, 'metric_id': NULL, 'additional_details': NULL})",
            [b, m, m, m],
        )
    con.execute(
        "CREATE TABLE benchmark_metric_folds (benchmark_id VARCHAR, "
        "from_metric_id VARCHAR, to_metric_id VARCHAR, source_config VARCHAR, "
        "scale_factor DOUBLE, scale_offset DOUBLE)"
    )
    for b, f, t, s in folds:
        con.execute(
            "INSERT INTO benchmark_metric_folds VALUES (?, ?, ?, ?, NULL, NULL)",
            [b, f, t, s],
        )
    con.execute(
        "CREATE TABLE canonical_benchmarks (id VARCHAR, parent_benchmark_id VARCHAR, "
        "metadata VARCHAR)"
    )
    for i, p in parents:
        role = (roles or {}).get(i)
        con.execute(
            "INSERT INTO canonical_benchmarks VALUES (?, ?, ?)",
            [i, p, json.dumps({"role": role}) if role else "{}"],
        )
    return con


def _effective(con):
    return con.execute(
        "SELECT benchmark_id, metric_id, metric_id_effective "
        "FROM results_resolved ORDER BY 1, 2"
    ).fetchall()


def test_child_inherits_the_parent_naming_fold(caplog):
    from eval_card_backend.canonicalise import stages

    con = _con_with(
        rows=[("suite", "score"), ("suite-a", "score"), ("suite-b", "score"),
              ("suite-a", "f1")],
        folds=[("suite", "score", "accuracy", None)],
        parents=[("suite", None), ("suite-a", "suite"), ("suite-b", "suite")],
    )
    with caplog.at_level(logging.INFO, logger="eval_card_backend.canonicalise.stages"):
        stages._apply_metric_folds(con)
    assert _effective(con) == [
        ("suite", "score", "accuracy"),
        ("suite-a", "f1", "f1"),            # no fold for this name anywhere
        ("suite-a", "score", "accuracy"),   # inherited
        ("suite-b", "score", "accuracy"),   # inherited
    ]
    assert ("fold suite score→accuracy inherited by 2 row(s) on 2 slice "
            "child(ren)") in caplog.text


def test_a_childs_own_fold_wins():
    from eval_card_backend.canonicalise import stages

    con = _con_with(
        rows=[("suite-a", "score"), ("suite-b", "score")],
        folds=[("suite", "score", "accuracy", None),
               ("suite-a", "score", "pass-at-1", None)],
        parents=[("suite", None), ("suite-a", "suite"), ("suite-b", "suite")],
    )
    stages._apply_metric_folds(con)
    assert _effective(con) == [
        ("suite-a", "score", "pass-at-1"),
        ("suite-b", "score", "accuracy"),
    ]


def test_a_benchmark_without_a_parent_is_unchanged():
    from eval_card_backend.canonicalise import stages

    con = _con_with(
        rows=[("other", "score"), ("suite", "score")],
        folds=[("suite", "score", "accuracy", None)],
        parents=[("suite", None), ("other", None)],
    )
    stages._apply_metric_folds(con)
    assert _effective(con) == [
        ("other", "score", "score"),
        ("suite", "score", "accuracy"),
    ]


def test_scoped_parent_rows_are_not_inherited():
    """A source-scoped row is a scale conversion for one publisher, not a
    rename; it does not fold the parent's rows and does not reach the
    children either."""
    from eval_card_backend.canonicalise import stages

    con = _con_with(
        rows=[("suite", "score"), ("suite-a", "score")],
        folds=[("suite", "score", "accuracy", "some_source")],
        parents=[("suite", None), ("suite-a", "suite")],
    )
    stages._apply_metric_folds(con)
    assert _effective(con) == [
        ("suite", "score", "score"),
        ("suite-a", "score", "score"),
    ]


def test_a_diagnostic_child_does_not_inherit_the_fold():
    """A child the registry marks `metadata.role = "diagnostic"` measures a
    different quantity under the parent's name (BFCL's format-sensitivity
    standard deviation is not an accuracy), so the parent's rename of its
    catch-all `score` does not describe it. It keeps `score`; a sibling part
    still inherits."""
    from eval_card_backend.canonicalise import stages

    con = _con_with(
        rows=[("suite", "score"), ("suite-a", "score"), ("suite-diag", "score")],
        folds=[("suite", "score", "accuracy", None)],
        parents=[("suite", None), ("suite-a", "suite"), ("suite-diag", "suite")],
        roles={"suite-diag": "diagnostic"},
    )
    stages._apply_metric_folds(con)
    assert _effective(con) == [
        ("suite", "score", "accuracy"),
        ("suite-a", "score", "accuracy"),
        ("suite-diag", "score", "score"),
    ]


def test_an_aggregate_child_still_inherits_the_fold():
    """An `aggregate` child is the same quantity rolled up over its
    siblings, so the parent's naming fold describes it too."""
    from eval_card_backend.canonicalise import stages

    con = _con_with(
        rows=[("suite-roll", "score")],
        folds=[("suite", "score", "accuracy", None)],
        parents=[("suite", None), ("suite-roll", "suite")],
        roles={"suite-roll": "aggregate"},
    )
    stages._apply_metric_folds(con)
    assert _effective(con) == [("suite-roll", "score", "accuracy")]


def test_a_self_parented_row_is_not_a_child():
    from eval_card_backend.canonicalise import stages

    con = _con_with(
        rows=[("suite", "f1")],
        folds=[("suite", "f1", "accuracy", None)],
        parents=[("suite", "suite")],
    )
    stages._apply_metric_folds(con)
    assert _effective(con) == [("suite", "f1", "accuracy")]  # own fold, once


# ---------------------------------------------------------------------------
# The Global-MMLU-Lite shape, end to end
# ---------------------------------------------------------------------------


def _record(eid, results):
    return json.dumps({
        "evaluation_id": eid,
        "schema_version": "0.2.2",
        "retrieved_timestamp": "2026-04-30T00:00:00Z",
        "evaluation_timestamp": "2026-04-29T00:00:00Z",
        "model_info": {"developer": "vendor", "name": "FoldModel",
                       "id": "vendor/fold-model", "inference_platform": "test"},
        "source_metadata": {"source_name": "Folds Inc", "source_type": "evaluation_run",
                            "source_organization_name": "Folds Inc",
                            "evaluator_relationship": "first_party"},
        "eval_library": {"name": "minibench", "version": "1.0"},
        "evaluation_results": results,
    })


def _result(name, score):
    return {
        "evaluation_name": name,
        "source_data": {"dataset_name": "gml", "source_type": "other"},
        # a bare "Score" channel: resolves to the catch-all `score`, which the
        # registry rule renames to accuracy on the benchmark
        "metric_config": {"metric_id": "score", "metric_name": "Score",
                          "evaluation_description": "Score", "metric_kind": None,
                          "metric_unit": None, "score_type": "continuous",
                          "min_score": 0.0, "max_score": 1.0, "lower_is_better": False},
        "score_details": {"score": score},
    }


@pytest.fixture(scope="module")
def gml_snapshot(tmp_path_factory, _taxonomy_seed_stub, _collections_stub):
    """Fixture registry + a `gml` parent with two language children carrying
    the parent's `score -> accuracy` fold; one record with two language rows
    and no total."""
    pytest.importorskip("duckdb")
    import os

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    tmp = tmp_path_factory.mktemp("folds")
    reg = tmp / "registry"
    reg.mkdir()
    for f in FIXTURES.joinpath("entity_registry").glob("*.parquet"):
        shutil.copy(f, reg / f.name)
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE cb AS SELECT * REPLACE (CAST(parent_benchmark_id AS VARCHAR) "
        f"AS parent_benchmark_id) FROM read_parquet('{reg / 'canonical_benchmarks.parquet'}')"
    )
    for bid, name, parent in (("gml", "GML", None), ("gml-arabic", "GML Arabic", "gml"),
                              ("gml-english", "GML English", "gml")):
        con.execute(
            "INSERT INTO cb BY NAME SELECT ? AS id, ? AS display_name, "
            "? AS parent_benchmark_id, 'reviewed' AS review_status, '{}' AS metadata, "
            "'[]' AS tags", [bid, name, parent],
        )
    con.execute(f"COPY cb TO '{reg / 'canonical_benchmarks.parquet'}' (FORMAT PARQUET)")
    con.execute(f"CREATE TABLE al AS SELECT * FROM read_parquet('{reg / 'aliases.parquet'}')")
    for i, (raw, bid) in enumerate((("gml", "gml"), ("Arabic", "gml-arabic"),
                                    ("English", "gml-english"))):
        con.execute(
            "INSERT INTO al BY NAME SELECT ? AS id, ? AS raw_value, 'benchmark' AS entity_type, "
            "? AS canonical_id, 'fixtures_folds' AS source_config, 'confirmed' AS status, "
            "'seed' AS strategy, 1.0 AS confidence", [f"gml-{i}", raw, bid],
        )
    con.execute(f"COPY al TO '{reg / 'aliases.parquet'}' (FORMAT PARQUET)")
    con.execute(f"CREATE TABLE fo AS SELECT * FROM read_parquet('{reg / 'benchmark_metric_folds.parquet'}')")
    con.execute(
        "INSERT INTO fo BY NAME SELECT 'gml' AS benchmark_id, 'score' AS from_metric_id, "
        "'accuracy' AS to_metric_id, 'naming: test' AS note"
    )
    con.execute(f"COPY fo TO '{reg / 'benchmark_metric_folds.parquet'}' (FORMAT PARQUET)")
    con.close()

    eee = tmp / "eee"
    write_eee_datastore(eee, [
        ("fixtures_folds", "01-languages.json", _record("ev_gml_langs", [
            _result("Arabic", 0.40), _result("English", 0.60),
        ])),
    ])

    # module scope runs before the function-scoped autouse env fixture, so
    # the taxonomy / collections stubs are set here for the pipeline run
    prior = {k: os.environ.get(k) for k in (
        "EEE_LOCAL_DATASET_DIR", "BENCHMARK_METADATA_LOCAL_DIR",
        "EEE_REFRESH_SNAPSHOT", "BENCHMARK_METADATA_REFRESH",
        "EVALCARD_REGISTRY_SEED_DIR", "COLLECTIONS_CURATED_PATH",
        "COLLECTIONS_VENDOR_DIR", "EEE_REVISION", "ENTITY_REGISTRY_REVISION",
        "BENCHMARK_METADATA_REVISION",
    )}
    os.environ["EEE_LOCAL_DATASET_DIR"] = str(eee)
    os.environ["BENCHMARK_METADATA_LOCAL_DIR"] = str(FIXTURES / "auto_benchmarkcards")
    os.environ["EVALCARD_REGISTRY_SEED_DIR"] = str(_taxonomy_seed_stub)
    os.environ["COLLECTIONS_CURATED_PATH"] = str(_collections_stub / "collections_curated.yaml")
    os.environ["COLLECTIONS_VENDOR_DIR"] = str(_collections_stub / "vendor_collections")
    for k in ("EEE_REFRESH_SNAPSHOT", "BENCHMARK_METADATA_REFRESH", "EEE_REVISION",
              "ENTITY_REGISTRY_REVISION", "BENCHMARK_METADATA_REVISION"):
        os.environ.pop(k, None)
    try:
        out = pipeline.run(
            Settings.from_env(),
            configs=["fixtures_folds"],
            snapshot_id="2026-04-30T00:00:00Z",
            warehouse_dir=str(tmp / "warehouse"),
            registry_local_dir=str(reg),
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


def test_language_rows_fold_to_the_parents_metric(gml_snapshot):
    con = duckdb.connect()
    rows = con.execute(
        "SELECT benchmark_id, metric_id, metric_key FROM "
        f"read_parquet('{gml_snapshot}/fact_results.parquet') ORDER BY 1"
    ).fetchall()
    assert rows == [
        ("gml-arabic", "score", "accuracy"),
        ("gml-english", "score", "accuracy"),
    ]


def test_derived_parent_row_lands_on_the_folded_metric(gml_snapshot):
    """The suite has no total of its own; with both children present its
    derived mean is emitted — on `accuracy`, the metric the page's totals
    would carry, not on the children's raw `score`."""
    con = duckdb.connect()
    rows = con.execute(
        "SELECT benchmark_id, metric_id, value_level, children_present, "
        "       children_expected, round(score, 6) "
        f"FROM read_parquet('{gml_snapshot}/eval_results_view.parquet') "
        "WHERE benchmark_id = 'gml'"
    ).fetchall()
    assert rows == [("gml", "accuracy", "derived", 2, 2, 0.5)]
    assert con.execute(
        "SELECT count(*) FROM "
        f"read_parquet('{gml_snapshot}/eval_results_view.parquet') "
        "WHERE metric_id = 'score'"
    ).fetchone() == (0,)

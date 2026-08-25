"""Stage J — collection_context.json.

The sidecar pre-joins a curated collection's own published cells (its
no-feedback pick plus an assisted companion) with every other EEE measurement
of the same model on the same benchmark. Everything here is hand-built on a
bare DuckDB connection: the sidecar reads six tables and a curated YAML, which
is a far smaller surface than a pipeline run.

Fixture shape (one benchmark, `minibench`, official_task_count = 10):

  collection `test-study`   dated model ids, four protocol conditions on
                            model-a and four on model-b
  composite  `board`        a leaderboard, undated model ids, percent-scaled
                            scores over two harvests, scaffolds recorded
  composite  `aggregator`   a scaffold-less aggregator on a raw metric that
                            folds into the canonical one
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pytest

from eval_card_backend.canonicalise import sidecars


COLLECTION = "test-study"
BENCHMARK = "minibench"
CONFIG = "minibench"
METRIC = "accuracy"
MAINTAINER_ORG = "bench-maintainer"
AGGREGATOR_ORG = "aggregator-org"
EXCLUDED_ORG = sorted(sidecars._CONTEXT_EXCLUDED_ORGS)[0]
OFFICIAL_TASK_COUNT = 10
SNAPSHOT = {"snapshot_id": "2026-04-30T00:00:00Z"}

MODEL_A_RAW = "vendor/model-a-20260101"     # dated, as the collection ships it
MODEL_B_RAW = "vendor/model-b-20260101"
MODEL_A_KEY = "vendor/model-a"              # undated, as the board ships it
MODEL_B_KEY = "vendor/model-b"

CURRENT_HARVEST = "200000.5"   # 1970-01-03
STALE_HARVEST = "1000.5"       # 1970-01-01


def _cond(feedback: str, variant: str) -> str:
    return json.dumps(
        {"feedback": feedback, "variant": variant},
        sort_keys=True, separators=(",", ":"),
    )


A_FULL = _cond("none", "full")
A_NARROW = _cond("none", "narrow")
A_UNKNOWN = _cond("unknown", "zz-most-trajectories")
A_FEEDBACK = _cond("answer_feedback", "zz-most-trajectories")
B1 = _cond("none", "a")
B2 = _cond("none", "b")
B3 = _cond("none", "c")
B4 = _cond("none", "d")

# (model_raw, protocol_condition, n_tasks, n_correct, extra trajectories on
# task 0). Extra trajectories move the tie-break without moving the task mean.
CONDITIONS = [
    (MODEL_A_RAW, A_FULL, 10, 6, 0),
    (MODEL_A_RAW, A_NARROW, 5, 4, 0),
    (MODEL_A_RAW, A_UNKNOWN, 10, 9, 10),
    (MODEL_A_RAW, A_FEEDBACK, 10, 8, 10),
    (MODEL_B_RAW, B1, 10, 5, 0),
    (MODEL_B_RAW, B2, 10, 7, 0),
    (MODEL_B_RAW, B3, 10, 4, 2),
    (MODEL_B_RAW, B4, 10, 3, 2),
]

AGG_KEY = {MODEL_A_RAW: MODEL_A_KEY, MODEL_B_RAW: MODEL_B_KEY}

# (model_aggregation_key, scaffold, published percent, harvest)
EXTERNAL = [
    (MODEL_A_KEY, "Alpha", 55.0, CURRENT_HARVEST),
    (MODEL_A_KEY, "Beta", 70.0, CURRENT_HARVEST),
    (MODEL_A_KEY, "Gamma", 61.0, STALE_HARVEST),      # delisted since
    (MODEL_B_KEY, "Alpha", 30.0, CURRENT_HARVEST),
    (MODEL_B_KEY, "Delta", 44.0, CURRENT_HARVEST),
]

# Scaffold-less aggregator rows: (model_aggregation_key, percent, harvest).
# model-a's 55.0 restates the board's Alpha entry; model-b's is its own point.
AGGREGATED = [
    (MODEL_A_KEY, 55.0, CURRENT_HARVEST),
    (MODEL_B_KEY, 38.0, CURRENT_HARVEST),
]


def _trajectory_rows(conditions=CONDITIONS, partial_credit=()):
    """One trajectory per task, plus `extra` duplicates of task 0. Scores are
    0/1 unless the (model, condition) is in `partial_credit`, where correct
    tasks score 0.5. The latter pins that source-specific score semantics are
    not rejected merely because `is_correct` has a different meaning."""
    rows = []
    for model_raw, condition, n_tasks, n_correct, extra in conditions:
        hit = 0.5 if (model_raw, condition) in partial_credit else 1.0
        for i in range(n_tasks):
            rows.append((model_raw, condition, f"t{i}",
                         hit if i < n_correct else 0.0, i < n_correct))
        for j in range(extra):
            rows.append((model_raw, condition, "t0",
                         hit if n_correct else 0.0, bool(n_correct)))
    return rows


def _published_score(rows, model_raw, condition) -> float:
    by_task: dict[str, list[float]] = {}
    for m, c, task, score, _ in rows:
        if m == model_raw and c == condition:
            by_task.setdefault(task, []).append(score)
    means = [sum(v) / len(v) for v in by_task.values()]
    return sum(means) / len(means)


def _build_con(
    *,
    conditions=CONDITIONS,
    external=EXTERNAL,
    aggregated=AGGREGATED,
    partial_credit=(),
    score_overrides: dict[tuple[str, str], float] | None = None,
    registry_metadata: str | None = None,
    maintainer_org: str = MAINTAINER_ORG,
    aggregator_org: str = AGGREGATOR_ORG,
):
    con = duckdb.connect()
    traj = _trajectory_rows(conditions, partial_credit)

    con.execute(
        "CREATE TABLE collection_trajectories_raw ("
        "collection_id VARCHAR, benchmark_raw VARCHAR, model_raw VARCHAR, "
        "task_id VARCHAR, protocol_condition VARCHAR, trajectory_idx INTEGER, "
        "score DOUBLE, is_correct BOOLEAN)"
    )
    for idx, (model_raw, condition, task, score, correct) in enumerate(traj):
        con.execute(
            "INSERT INTO collection_trajectories_raw VALUES (?,?,?,?,?,?,?,?)",
            [COLLECTION, CONFIG, model_raw, task, condition, idx, score, correct],
        )

    con.execute(
        "CREATE TABLE fact_results ("
        "collection_id VARCHAR, composite_slug VARCHAR, source_config VARCHAR, "
        "benchmark_key VARCHAR, metric_key VARCHAR, metric_key_effective VARCHAR, "
        "metric_kind VARCHAR, "
        "model_aggregation_key VARCHAR, model_raw VARCHAR, score DOUBLE, "
        "score_se DOUBLE, retrieved_timestamp VARCHAR, "
        "evaluation_timestamp VARCHAR, agent_scaffold_raw VARCHAR, "
        "org_id VARCHAR, protocol_condition VARCHAR)"
    )
    overrides = score_overrides or {}
    for model_raw, condition, *_ in conditions:
        score = overrides.get(
            (model_raw, condition), _published_score(traj, model_raw, condition)
        )
        con.execute(
            "INSERT INTO fact_results VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [COLLECTION, COLLECTION, CONFIG, BENCHMARK, METRIC, METRIC,
             "accuracy", AGG_KEY[model_raw], model_raw, score, 0.04, "3000.0",
             "2026-03-01", None, "study-org", condition],
        )
    for model_key, scaffold, score, harvest in external:
        con.execute(
            "INSERT INTO fact_results VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [None, "board", "board", BENCHMARK, METRIC, METRIC, "accuracy",
             model_key, model_key, score, 2.5, harvest, "2026-02-01",
             scaffold, maintainer_org, None],
        )
    # An aggregator reporting a raw metric that folds into the canonical one,
    # with no scaffold provenance: matched on metric_key_effective.
    for model_key, score, harvest in aggregated:
        con.execute(
            "INSERT INTO fact_results VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [None, "aggregator", "aggregator", BENCHMARK, "raw-score", METRIC,
             "accuracy", model_key, model_key, score, None, harvest,
             None, None, aggregator_org, None],
        )

    # eval_results_view: one row per protocol point the sidecar has to find
    # its verbatim protocol_condition in, plus the board's group classification.
    con.execute(
        "CREATE TABLE eval_results_view ("
        "composite_slug VARCHAR, benchmark_id VARCHAR, metric_id VARCHAR, "
        "metric_id_effective VARCHAR, "
        "model_key VARCHAR, protocol_condition VARCHAR, "
        "scale_conversion VARCHAR)"
    )
    for model_raw, condition, *_ in conditions:
        con.execute(
            "INSERT INTO eval_results_view VALUES (?,?,?,?,?,?,?)",
            [COLLECTION, BENCHMARK, METRIC, METRIC, AGG_KEY[model_raw],
             condition, "none"],
        )
    for model_key in sorted({e[0] for e in external}):
        con.execute(
            "INSERT INTO eval_results_view VALUES (?,?,?,?,?,?,?)",
            ["board", BENCHMARK, METRIC, METRIC, model_key, None, "div100"],
        )
    # The aggregator keeps its raw id in the view and folds to the canonical
    # one, exactly as llm-stats does in the warehouse.
    for model_key in sorted({a[0] for a in aggregated}):
        con.execute(
            "INSERT INTO eval_results_view VALUES (?,?,?,?,?,?,?)",
            ["aggregator", BENCHMARK, "raw-score", METRIC, model_key, None,
             "div100"],
        )
    # A second metric on the board, carrying a DIFFERENT scale conversion.
    # Scale resolution must stay scoped to the metric under test.
    for model_key in sorted({e[0] for e in external}):
        con.execute(
            "INSERT INTO eval_results_view VALUES (?,?,?,?,?,?,?)",
            ["board", BENCHMARK, "calibration-error", "calibration-error",
             model_key, None, "none"],
        )

    if registry_metadata is None:
        registry_metadata = json.dumps({
            "official_task_count": OFFICIAL_TASK_COUNT,
            "maintainer_org_id": MAINTAINER_ORG,
        })
    con.execute(
        "CREATE TABLE benchmarks AS SELECT ? AS benchmark_id, "
        "? AS composite_slug, TRY_CAST(? AS JSON) AS registry_metadata",
        [BENCHMARK, COLLECTION, registry_metadata],
    )
    con.execute(
        "CREATE TABLE canonical_benchmarks AS "
        "SELECT ? AS id, ? AS preferred_metric_id", [BENCHMARK, METRIC]
    )
    con.execute(
        "CREATE TABLE canonical_metrics AS SELECT ? AS id, "
        "CAST(NULL AS VARCHAR) AS metric_kind, CAST(0 AS DOUBLE) AS min_score, "
        "CAST(1 AS DOUBLE) AS max_score, FALSE AS lower_is_better", [METRIC]
    )
    con.execute(
        "CREATE TABLE models_view AS "
        "SELECT ? AS model_key, 'Model A' AS model_name "
        "UNION ALL SELECT ?, 'Model B'", [MODEL_A_KEY, MODEL_B_KEY]
    )
    con.execute(
        "CREATE TABLE composites AS "
        "SELECT 'board' AS composite_slug, 'The Board' AS composite_display_name "
        "UNION ALL SELECT 'aggregator', 'The Aggregator' "
        "UNION ALL SELECT ?, 'Test Study'", [COLLECTION]
    )
    return con


def _write_curated(tmp_path: Path, monkeypatch, *, has_trajectories=True,
                   aggregate="task_mean_score") -> None:
    path = tmp_path / "curated.yaml"
    path.write_text(
        f"{COLLECTION}:\n"
        f"  display_name: Test Study\n"
        f"  curated: true\n"
        f"  has_trajectories: {'true' if has_trajectories else 'false'}\n"
        f"  score_semantics:\n"
        f"    {CONFIG}: {{metric: accuracy, aggregate: {aggregate}}}\n"
    )
    monkeypatch.setenv("COLLECTIONS_CURATED_PATH", str(path))


def _write(con, tmp_path: Path, name: str = "out"):
    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    return sidecars.write_collection_context(con, out, SNAPSHOT), out


def _entry(path: Path) -> dict:
    return json.loads(path.read_text())[COLLECTION][BENCHMARK]


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_context_sidecar_shape(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    path, _ = _write(_build_con(), tmp_path)
    assert path is not None
    entry = _entry(path)

    assert entry["harvested_at"] == SNAPSHOT["snapshot_id"]
    assert entry["official_task_count"] == OFFICIAL_TASK_COUNT
    # The source set is every composite reporting the benchmark, not just the
    # maintainer's; display names ride along so the caption needs no join.
    assert entry["context_sources"] == [
        {"id": "aggregator", "display_name": "The Aggregator"},
        {"id": "board", "display_name": "The Board"},
    ]
    assert entry["context_source_display"] == "The Aggregator, The Board"
    assert entry["models_without_context"] == []
    assert sorted(entry["models"]) == [MODEL_A_KEY, MODEL_B_KEY]

    model_a = entry["models"][MODEL_A_KEY]
    assert model_a["display_name"] == "Model A"
    assert model_a["n_tasks"] == 10
    assert model_a["score_se"] == 0.04
    # one attempt per task, except task 0 in the unknown-feedback arm which
    # isn't the chosen condition
    assert (model_a["attempts_min"], model_a["attempts_max"]) == (1, 1)
    assert not any(key.startswith("band") for key in model_a)
    # external values are converted per each source's Stage J classification
    # (div100) and ordered score-desc; Gamma is a delisted board entry that
    # survives as a dated measurement, and the aggregator's 0.55 restatement of
    # Alpha is dropped as a re-report.
    assert [(p["scaffold"], p["score"]) for p in model_a["external"]] == [
        ("Beta", 0.7), ("Gamma", 0.61), ("Alpha", 0.55),
    ]
    assert model_a["external"][0]["run_date"] == "2026-02-01"
    assert model_a["external"][0]["score_se"] == 0.025
    assert model_a["external"][0]["source"] == "The Board"
    assert model_a["external"][0]["retrieved_at"] == "1970-01-03"

    # model-b keeps its own aggregator point: no scaffold, sourced, no SE
    unlabelled = [
        p for p in entry["models"][MODEL_B_KEY]["external"] if p["scaffold"] is None
    ]
    assert unlabelled == [{
        "scaffold": None, "source": "The Aggregator", "score": 0.38,
        "score_se": None, "run_date": None, "retrieved_at": "1970-01-03",
    }]


def test_assisted_companion_and_its_absence_are_both_reported(
    tmp_path, monkeypatch
):
    _write_curated(tmp_path, monkeypatch)
    path, _ = _write(_build_con(), tmp_path)
    entry = _entry(path)
    # model-a has a full-coverage answer-feedback arm, picked the same way as
    # the no-feedback one
    assert entry["models"][MODEL_A_KEY]["assisted"] == {
        "score": 0.8, "score_se": 0.04, "n_tasks": 10,
        "protocol_condition": A_FEEDBACK,
    }
    # model-b ran no assisted arm at all: named, with a null task count
    assert entry["models"][MODEL_B_KEY]["assisted"] is None
    assert entry["models_without_assisted"] == [
        {"display_name": "Model B", "n_tasks": None}
    ]


def test_assisted_below_coverage_is_kept_with_its_own_task_count(
    tmp_path, monkeypatch
):
    # model-a's assisted arm covers 5 of 10 tasks. Coverage is provenance, not
    # an eligibility gate, and the payload carries the exact task count.
    _write_curated(tmp_path, monkeypatch)
    conditions = [c for c in CONDITIONS if c[1] != A_FEEDBACK]
    conditions.append((MODEL_A_RAW, A_FEEDBACK, 5, 4, 10))
    path, _ = _write(_build_con(conditions=conditions), tmp_path)
    entry = _entry(path)
    assert entry["models"][MODEL_A_KEY]["assisted"]["n_tasks"] == 5
    assert "Model A" not in [
        m["display_name"] for m in entry["models_without_assisted"]
    ]
    # the no-feedback mark is untouched
    assert entry["models"][MODEL_A_KEY]["score"] == 0.6
    assert entry["models"][MODEL_A_KEY]["n_tasks"] == 10


def test_assisted_that_fails_an_integrity_gate_is_named_not_dropped_silently(
    tmp_path, monkeypatch, caplog
):
    # G3 still applies to the companion: a published score that will not
    # reproduce from its own released attempts is not rendered, and the model
    # is named rather than silently losing its second mark.
    _write_curated(tmp_path, monkeypatch)
    con = _build_con(score_overrides={(MODEL_A_RAW, A_FEEDBACK): 0.61})
    with caplog.at_level(logging.INFO, logger=sidecars.log.name):
        path, _ = _write(con, tmp_path)
    entry = _entry(path)
    assert entry["models"][MODEL_A_KEY]["assisted"] is None
    assert {"display_name": "Model A", "n_tasks": 10} in entry["models_without_assisted"]
    assert "no assisted mark" in caplog.text


def test_condition_selection_ties_and_unknown_exclusion(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    path, _ = _write(_build_con(), tmp_path)
    entry = _entry(path)

    # model-a: the fullest no-feedback arm wins over the narrower one, and
    # neither the unknown-feedback nor the answer-feedback arm competes even
    # though both carry twice the trajectories.
    assert entry["models"][MODEL_A_KEY]["protocol_condition"] == A_FULL
    assert entry["models"][MODEL_A_KEY]["score"] == 0.6
    # model-b: all four arms cover 10 tasks, so the trajectory count decides
    # (B3/B4 at 12) and the protocol string breaks the remaining tie.
    model_b = entry["models"][MODEL_B_KEY]
    assert model_b["protocol_condition"] == B3
    assert model_b["score"] == 0.4
    # B3's two extra trajectories both sit on task 0
    assert (model_b["attempts_min"], model_b["attempts_max"]) == (1, 3)


def test_aggregation_key_bridges_dated_and_undated_ids(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    path, _ = _write(_build_con(), tmp_path)
    entry = _entry(path)
    # the collection ships dated model ids and the board undated ones; both
    # land in one entry keyed by model_aggregation_key
    assert MODEL_A_RAW not in entry["models"]
    assert len(entry["models"][MODEL_A_KEY]["external"]) == 3


def test_output_is_byte_stable_across_runs(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    first, _ = _write(_build_con(), tmp_path, "run1")
    second, _ = _write(_build_con(), tmp_path, "run2")
    assert first.read_bytes() == second.read_bytes()


# ---------------------------------------------------------------------------
# currency
# ---------------------------------------------------------------------------


def test_currency_keeps_delisted_entries_with_their_own_date(
    tmp_path, monkeypatch
):
    _write_curated(tmp_path, monkeypatch)
    path, _ = _write(_build_con(), tmp_path)
    points = _entry(path)["models"][MODEL_A_KEY]["external"]
    # Gamma only appears in the older harvest. These are dated measurements,
    # not a live listing, so it stays — stamped with when it was last seen.
    by_scaffold = {p["scaffold"]: p for p in points}
    assert set(by_scaffold) == {"Alpha", "Beta", "Gamma"}
    assert by_scaffold["Gamma"]["retrieved_at"] < by_scaffold["Alpha"]["retrieved_at"]


def test_currency_keeps_only_the_latest_row_per_scaffold(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    # the board republished Alpha at a higher score in the current harvest
    external = [e for e in EXTERNAL if (e[0], e[1]) != (MODEL_A_KEY, "Alpha")]
    external += [
        (MODEL_A_KEY, "Alpha", 41.0, STALE_HARVEST),
        (MODEL_A_KEY, "Alpha", 55.0, CURRENT_HARVEST),
    ]
    path, _ = _write(_build_con(external=external), tmp_path)
    alpha = [
        p for p in _entry(path)["models"][MODEL_A_KEY]["external"]
        if p["scaffold"] == "Alpha"
    ]
    assert [p["score"] for p in alpha] == [0.55]


def test_distinct_scores_in_one_harvest_fail_loudly(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    external = list(EXTERNAL) + [(MODEL_A_KEY, "Alpha", 58.0, CURRENT_HARVEST)]
    with pytest.raises(RuntimeError, match="refusing to tie-break"):
        _write(_build_con(external=external), tmp_path)


def test_external_value_outside_unit_range_fails(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    external = [(m, s, v * 10, h) for m, s, v, h in EXTERNAL]
    with pytest.raises(RuntimeError, match=r"outside \[0, 1\]"):
        _write(_build_con(external=external), tmp_path)


# ---------------------------------------------------------------------------
# Integrity gates
# ---------------------------------------------------------------------------


def test_thin_no_feedback_cell_is_kept_with_its_task_count(
    tmp_path, monkeypatch
):
    _write_curated(tmp_path, monkeypatch)
    # model-b's only no-feedback arm now covers 5 of 10 tasks
    conditions = [c for c in CONDITIONS if c[0] != MODEL_B_RAW]
    conditions.append((MODEL_B_RAW, B1, 5, 3, 0))
    path, _ = _write(_build_con(conditions=conditions), tmp_path)
    entry = _entry(path)
    assert sorted(entry["models"]) == [MODEL_A_KEY, MODEL_B_KEY]
    assert entry["models"][MODEL_B_KEY]["n_tasks"] == 5
    assert entry["official_task_count"] == 10


def test_g3_recompute_mismatch_drops_one_model_not_the_bake(
    tmp_path, monkeypatch, caplog
):
    _write_curated(tmp_path, monkeypatch)
    con = _build_con(score_overrides={(MODEL_A_RAW, A_FULL): 0.61})
    with caplog.at_level(logging.WARNING, logger=sidecars.log.name):
        path, _ = _write(con, tmp_path)
    entry = _entry(path)
    assert sorted(entry["models"]) == [MODEL_B_KEY]
    assert "G3 recompute" in caplog.text


def test_is_correct_gap_does_not_drop_a_reproducible_score(
    tmp_path, monkeypatch
):
    _write_curated(tmp_path, monkeypatch)
    # model-a's correct tasks score 0.5, so task_mean_score is 0.30 while the
    # is_correct mean is 0.60. The declared score still reproduces exactly.
    con = _build_con(partial_credit={(MODEL_A_RAW, A_FULL)})
    path, _ = _write(con, tmp_path)
    entry = _entry(path)
    assert sorted(entry["models"]) == [MODEL_A_KEY, MODEL_B_KEY]
    assert entry["models"][MODEL_A_KEY]["score"] == 0.3


def test_models_without_context_lists_unmatched_models(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    # a third model with trajectories and a published cell but no board entry
    model_c_raw, model_c_key = "vendor/model-c-20260101", "vendor/model-c"
    AGG_KEY[model_c_raw] = model_c_key
    try:
        conditions = list(CONDITIONS) + [(model_c_raw, B1, 10, 6, 0)]
        con = _build_con(conditions=conditions)
        con.execute(
            "INSERT INTO models_view VALUES (?, 'Model C')", [model_c_key]
        )
        path, _ = _write(con, tmp_path)
    finally:
        AGG_KEY.pop(model_c_raw)
    entry = _entry(path)
    assert entry["models_without_context"] == ["Model C"]
    assert model_c_key not in entry["models"]


# ---------------------------------------------------------------------------
# fail-closed
# ---------------------------------------------------------------------------


def test_no_sidecar_without_curation(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch, has_trajectories=False)
    path, out = _write(_build_con(), tmp_path)
    assert path is None
    assert not (out / "collection_context.json").exists()


def test_no_sidecar_when_score_rule_is_not_recomputable(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch, aggregate="record_summary_mean")
    path, _ = _write(_build_con(), tmp_path)
    assert path is None


def test_missing_registry_metadata_warns_and_emits_nothing(
    tmp_path, monkeypatch, caplog
):
    _write_curated(tmp_path, monkeypatch)
    con = _build_con(registry_metadata=json.dumps(
        {"maintainer_org_id": MAINTAINER_ORG}
    ))
    with caplog.at_level(logging.WARNING, logger=sidecars.log.name):
        path, _ = _write(con, tmp_path)
    assert path is None
    assert "missing official_task_count" in caplog.text


def test_non_maintainer_sources_still_count(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    # the board is no longer the maintainer's: identity doesn't gate a source,
    # only data quality does
    path, _ = _write(_build_con(maintainer_org="someone-else"), tmp_path)
    entry = _entry(path)
    assert [s["id"] for s in entry["context_sources"]] == ["aggregator", "board"]


def test_quality_excluded_org_is_held_out(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    con = _build_con(aggregator_org=EXCLUDED_ORG)
    path, _ = _write(con, tmp_path)
    entry = _entry(path)
    assert [s["id"] for s in entry["context_sources"]] == ["board"]
    assert all(
        p["scaffold"] is not None
        for m in entry["models"].values() for p in m["external"]
    )


def test_rereport_drop_covers_a_near_miss_inside_the_window(
    tmp_path, monkeypatch
):
    # An aggregator restating a board entry rarely restates it to the bit. The
    # window, not just exact equality, is what collapses the two.
    _write_curated(tmp_path, monkeypatch)
    # 0.5503 vs the board's labelled 0.55: 0.0003 apart, inside the window.
    # Literal, not derived from the constant — a test that moves with the
    # value it pins would survive any change to it.
    inside = 55.03
    aggregated = [(MODEL_A_KEY, inside, CURRENT_HARVEST),
                  (MODEL_B_KEY, 38.0, CURRENT_HARVEST)]
    path, _ = _write(_build_con(aggregated=aggregated), tmp_path)
    entry = _entry(path)
    unlabelled = [p for p in entry["models"][MODEL_A_KEY]["external"]
                  if p["scaffold"] is None]
    assert unlabelled == []


def test_rereport_drop_keeps_a_point_outside_the_window(tmp_path, monkeypatch):
    # The real near-miss is 0.002 away (opus-4.5's 0.593 vs Letta Code's
    # 0.591): outside the window, so it stays as its own measurement.
    _write_curated(tmp_path, monkeypatch)
    # 0.552 vs the board's labelled 0.55: 0.002 apart, the same gap as the
    # real opus-4.5 near-miss, which must stay its own measurement.
    outside = 55.2
    aggregated = [(MODEL_A_KEY, outside, CURRENT_HARVEST),
                  (MODEL_B_KEY, 38.0, CURRENT_HARVEST)]
    path, _ = _write(_build_con(aggregated=aggregated), tmp_path)
    entry = _entry(path)
    unlabelled = [p for p in entry["models"][MODEL_A_KEY]["external"]
                  if p["scaffold"] is None]
    assert len(unlabelled) == 1
    assert unlabelled[0]["score"] == round(outside / 100, 6)


def test_official_task_count_is_context_not_an_eligibility_threshold(
    tmp_path, monkeypatch
):
    # Widening the registry denominator must not suppress thinner study cells.
    _write_curated(tmp_path, monkeypatch)
    registry = json.dumps({"official_task_count": 20})
    con = _build_con(registry_metadata=registry)
    path, _ = _write(con, tmp_path)
    entry = _entry(path)
    assert entry["official_task_count"] == 20
    assert entry["models"][MODEL_A_KEY]["n_tasks"] == 10
    assert entry["models"][MODEL_B_KEY]["n_tasks"] == 10

    # Larger cells retain their own counts against the same denominator.
    conditions = [
        (MODEL_A_RAW, A_FULL, 19, 12, 0),
        (MODEL_A_RAW, A_FEEDBACK, 19, 15, 0),
        (MODEL_B_RAW, B1, 19, 10, 0),
    ]
    con = _build_con(conditions=conditions, registry_metadata=registry)
    path, _ = _write(con, tmp_path)
    entry = _entry(path)
    assert entry["official_task_count"] == 20
    assert entry["models"][MODEL_A_KEY]["n_tasks"] == 19
    assert entry["models"][MODEL_A_KEY]["assisted"]["n_tasks"] == 19


def test_unlabelled_duplicate_scores_fail_loudly(tmp_path, monkeypatch):
    # A scaffold-less source publishing two different scores for one model at
    # one harvest is held to the same standard as a labelled one: the strip
    # reads as a spread, so an unexplained second number must not silently
    # widen it.
    _write_curated(tmp_path, monkeypatch)
    aggregated = [
        (MODEL_A_KEY, 55.0, CURRENT_HARVEST),
        (MODEL_A_KEY, 62.0, CURRENT_HARVEST),
        (MODEL_B_KEY, 38.0, CURRENT_HARVEST),
    ]
    with pytest.raises(RuntimeError, match="refusing to tie-break"):
        _write(_build_con(aggregated=aggregated), tmp_path)


def test_quality_holdout_survives_a_composite_that_spans_orgs(
    tmp_path, monkeypatch
):
    # A composite can carry rows from more than one org (`reward-bench` is
    # allenai + writer in the warehouse). Selecting the source on a clean org
    # must not carry the held-out org's rows in on its ticket.
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    con.execute(
        "INSERT INTO fact_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [None, "aggregator", "aggregator", BENCHMARK, "raw-score", METRIC,
         "accuracy", MODEL_B_KEY, MODEL_B_KEY, 99.0, None, CURRENT_HARVEST,
         None, None, EXCLUDED_ORG, None],
    )
    path, _ = _write(con, tmp_path)
    entry = _entry(path)
    scores = [p["score"] for p in entry["models"][MODEL_B_KEY]["external"]]
    assert 0.99 not in scores
    assert scores == sorted(scores, reverse=True)


def test_scale_resolution_is_scoped_to_the_metric_under_test(
    tmp_path, monkeypatch
):
    # The board publishes a second metric on this benchmark under a different
    # scale conversion. Resolving scale across all of a composite's metrics
    # would read as inconsistent and drop the source (the real case is
    # scale-seal-hle/hle: accuracy div100, calibration-error none).
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    assert con.execute(
        "SELECT count(DISTINCT scale_conversion) FROM eval_results_view "
        "WHERE composite_slug = 'board'"
    ).fetchone()[0] > 1
    path, _ = _write(con, tmp_path)
    entry = _entry(path)
    assert "board" in [s["id"] for s in entry["context_sources"]]
    # div100 applied, not the other metric's 'none'
    assert 0.55 in [p["score"] for p in entry["models"][MODEL_A_KEY]["external"]]


def test_unparseable_harvest_stamp_sorts_oldest_instead_of_aborting(
    tmp_path, monkeypatch
):
    # ISO-8601 stamps exist in the warehouse; CAST would abort the whole bake.
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    con.execute(
        "UPDATE fact_results SET retrieved_timestamp = '2026-02-01T00:00:00Z' "
        "WHERE composite_slug = 'board' AND agent_scaffold_raw = 'Beta'"
    )
    path, _ = _write(con, tmp_path)
    entry = _entry(path)
    beta = [p for p in entry["models"][MODEL_A_KEY]["external"]
            if p["scaffold"] == "Beta"]
    # the unparseable row is the only one for its key, so it survives with no
    # retrieved_at rather than vanishing or aborting
    assert len(beta) == 1
    assert beta[0]["retrieved_at"] is None
    # a key that also has a parseable row is unaffected
    alpha = [p for p in entry["models"][MODEL_A_KEY]["external"]
             if p["scaffold"] == "Alpha"]
    assert alpha[0]["retrieved_at"] == "1970-01-03"


def test_unparseable_stamp_never_displaces_a_parseable_one(
    tmp_path, monkeypatch
):
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    # a second Gamma row: unparseable stamp, different score. The parseable
    # row must still win its key, and the pair must not read as ambiguous.
    con.execute(
        "INSERT INTO fact_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [None, "board", "board", BENCHMARK, METRIC, METRIC, "accuracy",
         MODEL_A_KEY, MODEL_A_KEY, 88.0, 2.5, "not-a-timestamp", "2026-02-01",
         "Gamma", MAINTAINER_ORG, None],
    )
    path, _ = _write(con, tmp_path)
    entry = _entry(path)
    gamma = [p for p in entry["models"][MODEL_A_KEY]["external"]
             if p["scaffold"] == "Gamma"]
    assert [p["score"] for p in gamma] == [0.61]


def test_assisted_protocol_condition_absent_from_the_view_fails(
    tmp_path, monkeypatch
):
    # the assisted string is emitted into the payload like the no-feedback one
    # and gets the same byte-identity guarantee
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    con.execute(
        "DELETE FROM eval_results_view WHERE protocol_condition = ?",
        [A_FEEDBACK],
    )
    with pytest.raises(RuntimeError, match="assisted protocol_condition"):
        _write(con, tmp_path)


def test_single_matched_model_fails_the_join_assertion(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    external = [e for e in EXTERNAL if e[0] == MODEL_A_KEY]
    aggregated = [a for a in AGGREGATED if a[0] == MODEL_A_KEY]
    with pytest.raises(RuntimeError, match="matched an external entry"):
        _write(_build_con(external=external, aggregated=aggregated), tmp_path)


def test_flagged_scale_conversion_excludes_the_source(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    con.execute(
        "UPDATE eval_results_view SET scale_conversion = 'flagged' "
        "WHERE composite_slug IN ('board', 'aggregator')"
    )
    path, _ = _write(con, tmp_path)
    assert path is None


def test_aggregator_scale_resolves_on_its_own_unfolded_metric_id(
    tmp_path, monkeypatch
):
    _write_curated(tmp_path, monkeypatch)
    # the aggregator's view rows keep metric_id='raw-score'; keying the scale
    # lookup on the canonical id would find nothing and drop the source
    path, _ = _write(_build_con(), tmp_path)
    entry = _entry(path)
    assert "aggregator" in [s["id"] for s in entry["context_sources"]]
    scores = [p["score"] for p in entry["models"][MODEL_B_KEY]["external"]
              if p["source"] == "The Aggregator"]
    assert scores == [0.38]


def test_protocol_condition_absent_from_the_view_fails(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    con.execute(
        "DELETE FROM eval_results_view WHERE protocol_condition = ?", [A_FULL]
    )
    with pytest.raises(RuntimeError, match="absent from eval_results_view"):
        _write(con, tmp_path)


def test_non_accuracy_metric_kind_emits_nothing(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    # the registry leaves metric_kind NULL, so the observed per-row kind is
    # what the gate reads
    con.execute("UPDATE fact_results SET metric_kind = 'elo'")
    path, _ = _write(con, tmp_path)
    assert path is None


def test_metric_without_unit_bounds_emits_nothing(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    con.execute("UPDATE canonical_metrics SET max_score = CAST(100 AS DOUBLE)")
    path, _ = _write(con, tmp_path)
    assert path is None

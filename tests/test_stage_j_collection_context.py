"""Stage J — collection_context.json.

The sidecar pre-joins a curated collection's own published cell (plus a
re-run band computed from its released trajectories) with the external
(scaffold, model) entries the benchmark's maintainer publishes. Everything
here is hand-built on a bare DuckDB connection: the sidecar reads six tables
and a curated YAML, which is a far smaller surface than a pipeline run.

Fixture shape (one benchmark, `minibench`, official_task_count = 10):

  collection `test-study`   dated model ids, four protocol conditions on
                            model-a and four on model-b
  composite  `board`        the maintainer org's leaderboard, undated model
                            ids, percent-scaled scores over two harvests
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
OFFICIAL_TASK_COUNT = 10
SNAPSHOT = {"snapshot_id": "2026-04-30T00:00:00Z"}

MODEL_A_RAW = "vendor/model-a-20260101"     # dated, as the collection ships it
MODEL_B_RAW = "vendor/model-b-20260101"
MODEL_A_KEY = "vendor/model-a"              # undated, as the board ships it
MODEL_B_KEY = "vendor/model-b"

CURRENT_HARVEST = "2000.5"
STALE_HARVEST = "1000.5"


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


def _trajectory_rows(conditions=CONDITIONS, partial_credit=()):
    """One trajectory per task, plus `extra` duplicates of task 0. Scores are
    0/1 unless the (model, condition) is in `partial_credit`, where correct
    tasks score 0.5 — task_mean_score then diverges from the is_correct mean
    by more than G3's binarisation tolerance."""
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
    partial_credit=(),
    score_overrides: dict[tuple[str, str], float] | None = None,
    registry_metadata: str | None = None,
    maintainer_org: str = MAINTAINER_ORG,
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
        "benchmark_key VARCHAR, metric_key VARCHAR, metric_kind VARCHAR, "
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
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [COLLECTION, COLLECTION, CONFIG, BENCHMARK, METRIC, "accuracy",
             AGG_KEY[model_raw], model_raw, score, None, "3000.0",
             "2026-03-01", None, "study-org", condition],
        )
    for model_key, scaffold, score, harvest in external:
        con.execute(
            "INSERT INTO fact_results VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [None, "board", "board", BENCHMARK, METRIC, "accuracy",
             model_key, model_key, score, 2.5, harvest, "2026-02-01",
             scaffold, maintainer_org, None],
        )

    # eval_results_view: one row per protocol point the sidecar has to find
    # its verbatim protocol_condition in, plus the board's group classification.
    con.execute(
        "CREATE TABLE eval_results_view ("
        "composite_slug VARCHAR, benchmark_id VARCHAR, metric_id VARCHAR, "
        "model_key VARCHAR, protocol_condition VARCHAR, "
        "scale_conversion VARCHAR)"
    )
    for model_raw, condition, *_ in conditions:
        con.execute(
            "INSERT INTO eval_results_view VALUES (?,?,?,?,?,?)",
            [COLLECTION, BENCHMARK, METRIC, AGG_KEY[model_raw], condition, "none"],
        )
    for model_key in sorted({e[0] for e in external}):
        con.execute(
            "INSERT INTO eval_results_view VALUES (?,?,?,?,?,?)",
            ["board", BENCHMARK, METRIC, model_key, None, "div100"],
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
    # G1 is derived from the maintainer org, not an allowlist; the display
    # name rides along so caption 1 needs no client-side join
    assert entry["context_sources"] == [
        {"id": "board", "display_name": "The Board"}
    ]
    assert entry["context_source_display"] == "The Board"
    assert entry["models_without_context"] == []
    assert sorted(entry["models"]) == [MODEL_A_KEY, MODEL_B_KEY]

    model_a = entry["models"][MODEL_A_KEY]
    assert model_a["display_name"] == "Model A"
    assert model_a["n_tasks"] == 10
    # one attempt per task, except task 0 in the unknown-feedback arm which
    # isn't the chosen condition
    assert (model_a["attempts_min"], model_a["attempts_max"]) == (1, 1)
    assert model_a["band_runs"] == sidecars.BAND_RUNS
    assert model_a["band_method"] == "jeffreys"
    assert model_a["band_seed"] == sidecars.BAND_SEED
    assert model_a["band_lo"] <= model_a["band_hi"]
    # external values are converted per the board's Stage J classification
    # (div100) and ordered score-desc then scaffold
    assert [(p["scaffold"], p["score"]) for p in model_a["external"]] == [
        ("Beta", 0.7), ("Alpha", 0.55),
    ]
    assert model_a["external"][0]["run_date"] == "2026-02-01"
    assert model_a["external"][0]["score_se"] == 0.025


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
    assert len(entry["models"][MODEL_A_KEY]["external"]) == 2


def test_band_is_deterministic_across_runs(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    first, _ = _write(_build_con(), tmp_path, "run1")
    second, _ = _write(_build_con(), tmp_path, "run2")
    assert first.read_bytes() == second.read_bytes()


# ---------------------------------------------------------------------------
# currency
# ---------------------------------------------------------------------------


def test_currency_drops_stale_tuple(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    path, _ = _write(_build_con(), tmp_path)
    scaffolds = {
        p["scaffold"] for p in _entry(path)["models"][MODEL_A_KEY]["external"]
    }
    # Gamma only appears in the older harvest — delisted, not current
    assert scaffolds == {"Alpha", "Beta"}


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
# G2 / G3
# ---------------------------------------------------------------------------


def test_g2_coverage_drops_thin_model(tmp_path, monkeypatch, caplog):
    _write_curated(tmp_path, monkeypatch)
    # model-b's only no-feedback arm now covers 5 of 10 tasks
    conditions = [c for c in CONDITIONS if c[0] != MODEL_B_RAW]
    conditions.append((MODEL_B_RAW, B1, 5, 3, 0))
    with caplog.at_level(logging.INFO, logger=sidecars.log.name):
        path, _ = _write(_build_con(conditions=conditions), tmp_path)
    entry = _entry(path)
    assert sorted(entry["models"]) == [MODEL_A_KEY]
    assert "G2 coverage 5 of 10" in caplog.text


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


def test_g3_binarisation_gap_drops_model(tmp_path, monkeypatch, caplog):
    _write_curated(tmp_path, monkeypatch)
    # model-a's correct tasks score 0.5, so task_mean_score is 0.30 while the
    # is_correct mean is 0.60 — well past the partial-credit tolerance.
    con = _build_con(partial_credit={(MODEL_A_RAW, A_FULL)})
    with caplog.at_level(logging.WARNING, logger=sidecars.log.name):
        path, _ = _write(con, tmp_path)
    assert sorted(_entry(path)["models"]) == [MODEL_B_KEY]
    assert "G3 binarised mean" in caplog.text


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


def test_no_sidecar_when_no_composite_matches_the_maintainer(
    tmp_path, monkeypatch
):
    _write_curated(tmp_path, monkeypatch)
    path, _ = _write(_build_con(maintainer_org="someone-else"), tmp_path)
    assert path is None


def test_single_matched_model_fails_the_join_assertion(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    external = [e for e in EXTERNAL if e[0] == MODEL_A_KEY]
    with pytest.raises(RuntimeError, match="matched an external entry"):
        _write(_build_con(external=external), tmp_path)


def test_flagged_scale_conversion_excludes_the_source(tmp_path, monkeypatch):
    _write_curated(tmp_path, monkeypatch)
    con = _build_con()
    con.execute(
        "UPDATE eval_results_view SET scale_conversion = 'flagged' "
        "WHERE composite_slug = 'board'"
    )
    path, _ = _write(con, tmp_path)
    assert path is None


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

"""One number per cell, on one scale, in every artifact that ships it.

`score_canonical` is the only quantity the warehouse does arithmetic or
ordering with; the published number is display and provenance. The rule is
easy to state and easy to break in one place at a time, and each break looks
local until you put the artifacts side by side: a model summarised at
min = max = avg 86.0 for a metric bounded at 1, a subtask whose top reads
100.0 under a unit the same row calls a proportion, a "best model" picked
because its source reports percents.

So this is one test over one cell, read through all four shipped artifacts at
once. Two models, one benchmark, one metric, two scales: the model that
publishes percents scores WORSE than the one publishing fractions, and only
the canonical scale can see it. Anything that reverts a site to the published
number fails here, whichever site it was.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"

# The cell. `pct` publishes on 0-100 and is the weaker model; `frac`
# publishes on 0-1 and is the stronger one. Ordering them by the published
# number puts `pct` on top and prints 86.0 next to a [0, 1] metric.
PCT_PUBLISHED, PCT_CANONICAL = 86.0, 0.86
FRAC_PUBLISHED, FRAC_CANONICAL = 0.90, 0.90
SLICE = "mixed scale slice"


def _run_through_stage_i(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(FIXTURES / "eee"))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR",
                       str(FIXTURES / "auto_benchmarkcards"))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        configs=["fixtures_clean"],
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(tmp_path / "warehouse"),
        registry_local_dir=str(FIXTURES / "entity_registry"),
        cache_root=str(tmp_path / "cache"),
    )
    assert out_dir is not None
    return out_dir


def _plant_mixed_scale_cell(con) -> tuple[str, str]:
    """Reduce the corpus to one cell: two models on two scales.

    The rows are cloned from a real scored fact so every column the
    downstream SQL touches is populated the way the pipeline populates it;
    only identity, scale and score are restated. The model keys are the
    fixture's own, because `models_view` joins its display fields off the
    `models` dimension and an invented key has no row there. Every other fact
    goes, so each model's summary is exactly this one reading and nothing
    else can supply the number under test.
    """
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _tpl AS SELECT * FROM fact_results "
        "WHERE benchmark_key = 'mmlu' AND score IS NOT NULL LIMIT 1"
    )
    assert con.execute("SELECT count(*) FROM _tpl").fetchone()[0] == 1
    models = [r[0] for r in con.execute(
        "SELECT DISTINCT f.model_aggregation_key FROM fact_results f "
        "JOIN models m ON m.model_key = f.model_aggregation_key "
        "ORDER BY 1 LIMIT 2"
    ).fetchall()]
    assert len(models) == 2, models
    con.execute("DELETE FROM fact_results")
    for i, (model, published, canonical, conversion) in enumerate((
        (models[0], PCT_PUBLISHED, PCT_CANONICAL, "div100"),
        (models[1], FRAC_PUBLISHED, FRAC_CANONICAL, "none"),
    )):
        con.execute(
            "INSERT INTO fact_results SELECT * REPLACE ("
            f"  '{model}' AS model_aggregation_key, '{model}' AS model_key,"
            f"  '{model}' AS model_raw,"
            f"  'mixed-{i}-fact' AS fact_id, 'mixed-{i}-eval' AS evaluation_id,"
            f"  {published} AS score, {canonical} AS score_canonical,"
            f"  '{conversion}' AS scale_conversion,"
            f"  '{SLICE}' AS slice_key, '{SLICE}' AS slice_name,"
            "   FALSE AS lower_is_better,"
            "   'accuracy' AS metric_key, 'accuracy' AS metric_key_effective,"
            "   'accuracy' AS metric_base_key, 'accuracy' AS metric_id"
            ") FROM _tpl"
        )
    return models[0], models[1]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Every shipped artifact, built once off one planted cell."""
    pytest.importorskip("duckdb")
    from _pytest.monkeypatch import MonkeyPatch

    from eval_card_backend.canonicalise import sidecars, stages
    from eval_card_backend.canonicalise.resolver_setup import register_udfs
    from eval_card_backend.sources import registry as registry_src
    from eval_entity_resolver import Resolver

    tmp = tmp_path_factory.mktemp("canonical-contract")
    mp = MonkeyPatch()
    try:
        out_dir = _run_through_stage_i(tmp, mp)
    finally:
        mp.undo()

    con = duckdb.connect()
    register_udfs(
        con, Resolver(registry_src.load_alias_store(FIXTURES / "entity_registry"))
    )
    for table in ("fact_results", "benchmarks", "composites", "families",
                  "models", "canonical_metrics"):
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_parquet('{out_dir}/{table}.parquet')"
        )
    pct_model, frac_model = _plant_mixed_scale_cell(con)

    stages.stage_j_eval_results_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_models_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")
    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    index_path = sidecars.write_comparison_index(con, out_dir, snap)
    return con, json.loads(index_path.read_text()), pct_model, frac_model


def _model_row(con, model):
    return con.execute(
        "SELECT score_summary, top_scores FROM models_view WHERE model_key = ?",
        [model],
    ).fetchone()


def test_every_artifact_reports_the_cell_on_the_canonical_scale(artifacts):
    con, comparison_index, pct_model, frac_model = artifacts

    # (1) eval_results_view — the cell itself. `score` still carries what the
    # source published, which is the point of keeping both columns.
    erv = dict(con.execute(
        "SELECT model_key, [score, score_canonical] FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu' AND metric_id = 'accuracy'"
    ).fetchall())
    assert erv == {
        pct_model:  [PCT_PUBLISHED, PCT_CANONICAL],
        frac_model: [FRAC_PUBLISHED, FRAC_CANONICAL],
    }

    # (2) models_view.score_summary — a model's rows span benchmarks and so
    # span scales by construction. Summarising the published number reported
    # `aristotle/aristotle` as min = max = avg 86.0 on a [0, 1] metric.
    summary, top_scores = _model_row(con, pct_model)
    assert (summary["min"], summary["max"], summary["average"]) == (
        PCT_CANONICAL, PCT_CANONICAL, PCT_CANONICAL,
    )
    assert summary["count"] == 1

    # (3) models_view.top_scores — the same number, ranked and printed.
    assert top_scores
    assert [t["score"] for t in top_scores] == [PCT_CANONICAL] * len(top_scores)

    # (4) evals_view subtasks — a MIN/MAX across models, so it needs the one
    # scale they share. On published scores this read 86.0 for a proportion.
    subtasks, = con.execute(
        "SELECT subtasks FROM evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()
    tops = [
        m["top_score"]
        for s in subtasks if s["subtask_key"] == SLICE
        for m in s["metrics"] if m["metric_key"] == "accuracy"
    ]
    assert tops == [FRAC_CANONICAL]

    # (5) comparison-index — the leaderboard the model page ranks off. Ranking
    # on the published number would put the percent-publishing model first;
    # here it is second, on the same canonical value every artifact above
    # reported, while `score` keeps the source's own figure as provenance.
    entry = comparison_index["evals"][
        next(k for k in comparison_index["evals"] if k.endswith("mmlu"))
    ]
    by_model = {
        unquote(s["model_route_id"]): (s["score"], s["score_canonical"], s["rank"])
        for metric in entry["metrics"] if metric["metric_id"] == "accuracy"
        for s in metric["scores"]
    }
    assert by_model == {
        pct_model:  (PCT_PUBLISHED, PCT_CANONICAL, 2),
        frac_model: (FRAC_PUBLISHED, FRAC_CANONICAL, 1),
    }

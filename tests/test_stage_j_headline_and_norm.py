"""Stage J — what heads a page, and the number the list card shows.

Two rules a reader sees directly:

  * the metric that heads a page measures the task. Cost, latency, response
    length, degeneration and invalid-rate describe the run and are marked
    `role: diagnostic` in the registry; every model publishes them, so the
    coverage tiebreak used to hand them the headline.
  * the list card's normalised figure is the same number the result rows
    carry, averaged — not a second normalisation of the average published
    score, which skipped the canonical-scale conversion and the
    lower-is-better inversion.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _run_through_stage_i(tmp_path, monkeypatch, config: str) -> Path:
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(FIXTURES / "eee"))
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


def _materialise_views(out_dir: Path, mutate=None):
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
    # the fixture parquet predates preferred_metric_id; pad it like _load_dim
    con.execute(
        "CREATE TABLE canonical_benchmarks AS "
        "SELECT id, display_name, "
        "       CAST(parent_benchmark_id AS VARCHAR) AS parent_benchmark_id, "
        "       CAST(NULL AS VARCHAR) AS preferred_metric_id, "
        "       CAST(metadata AS VARCHAR) AS metadata, "
        "       CAST(NULL AS BOOLEAN) AS preferred_metric_llm_judged "
        f"FROM read_parquet('{FIXTURES}/entity_registry/canonical_benchmarks.parquet')"
    )
    if mutate is not None:
        mutate(con)
    stages.stage_j_eval_results_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_models_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_merged_evals_view(con, "2026-04-30T00:00:00Z")
    return con


_seq = iter(range(1000))


def _reset_mmlu(con):
    """Keep one real mmlu fact row as the clone template, then clear the
    benchmark so each test publishes exactly the metrics it means to test."""
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _fact_template AS "
        "SELECT * FROM fact_results "
        "WHERE benchmark_key = 'mmlu' AND score IS NOT NULL LIMIT 1"
    )
    con.execute("DELETE FROM fact_results WHERE benchmark_key = 'mmlu'")


def _add_metric(con, metric: str, scores, *, role=None, lower_is_better=False):
    """Publish `metric` on the fixture's mmlu page for one model per score."""
    con.execute(
        "INSERT INTO canonical_metrics (id, display_name, min_score, max_score, "
        "lower_is_better, metadata) VALUES (?, ?, 0, 1, ?, ?)",
        [metric, metric.title(), lower_is_better,
         '{"role": "diagnostic"}' if role == "diagnostic" else None],
    )
    for score in scores:
        n = next(_seq)
        con.execute(
            "INSERT INTO fact_results SELECT * REPLACE ("
            f"  '{metric}' AS metric_key, '{metric}' AS metric_key_effective,"
            f"  '{metric}' AS metric_base_key, '{metric}' AS metric_id,"
            f"  'hl-{n}' AS fact_id, 'hl-{n}-eval' AS evaluation_id,"
            f"  'model-{n}' AS model_aggregation_key, 'model-{n}' AS model_key,"
            f"  'model-{n}' AS model_raw, CAST(NULL AS VARCHAR) AS model_id,"
            f"  {score} AS score, {score} AS score_canonical,"
            f"  {'TRUE' if lower_is_better else 'FALSE'} AS lower_is_better,"
            "   'root' AS aggregate_level"
            ") FROM _fact_template"
        )


def _headline(con, benchmark="mmlu"):
    return con.execute(
        "SELECT primary_metric_id FROM evals_view WHERE benchmark_id = ?",
        [benchmark],
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Headline metric
# ---------------------------------------------------------------------------


def test_diagnostic_metric_loses_the_headline_to_an_outcome_metric(
    tmp_path, monkeypatch
):
    """The diagnostic covers more models and sorts first alphabetically, and
    still must not head the page."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _reset_mmlu(con)
        _add_metric(con, "aaa-word-count", [100, 200, 300], role="diagnostic")
        _add_metric(con, "zzz-win-rate", [0.4, 0.5])

    assert _headline(_materialise_views(out, mutate=mutate)) == "zzz-win-rate"


def test_registry_preference_still_outranks_an_outcome_metric(tmp_path, monkeypatch):
    """The diagnostic step sits BELOW the registry preference, never above it."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _reset_mmlu(con)
        _add_metric(con, "aaa-word-count", [100, 200, 300], role="diagnostic")
        _add_metric(con, "zzz-win-rate", [0.4, 0.5])
        _add_metric(con, "mmm-elo", [1200, 1300])
        con.execute("UPDATE canonical_benchmarks SET preferred_metric_id = 'mmm-elo'"
                    " WHERE id = 'mmlu'")

    assert _headline(_materialise_views(out, mutate=mutate)) == "mmm-elo"


def test_an_all_diagnostic_page_still_gets_a_headline(tmp_path, monkeypatch):
    """The step only demotes. A page whose every metric is a diagnostic still
    shows one rather than nothing."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _reset_mmlu(con)
        _add_metric(con, "aaa-word-count", [100, 200, 300], role="diagnostic")
        _add_metric(con, "bbb-latency", [1.0, 2.0], role="diagnostic")

    assert _headline(_materialise_views(out, mutate=mutate)) == "aaa-word-count"


def test_merged_page_applies_the_same_demotion(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _reset_mmlu(con)
        _add_metric(con, "aaa-word-count", [100, 200, 300], role="diagnostic")
        _add_metric(con, "zzz-win-rate", [0.4, 0.5])

    con = _materialise_views(out, mutate=mutate)
    assert con.execute(
        "SELECT preferred_metric_id FROM merged_evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()[0] == "zzz-win-rate"


# ---------------------------------------------------------------------------
# avg_score_norm
# ---------------------------------------------------------------------------


def test_avg_score_norm_is_the_mean_of_the_rows_own_normalised_scores(
    tmp_path, monkeypatch
):
    """The list card and the result rows must not disagree. Under a
    lower-is-better metric the published average and the normalised average
    move in opposite directions, which is where the two used to diverge."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _reset_mmlu(con)
        _add_metric(con, "refusal", [0.2, 0.4], lower_is_better=True)

    con = _materialise_views(out, mutate=mutate)
    avg, norm = con.execute(
        "SELECT avg_score, avg_score_norm FROM evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()
    per_model = [
        r[0] for r in con.execute(
            "SELECT score_normalized FROM eval_results_view "
            "WHERE benchmark_id = 'mmlu' AND is_headline ORDER BY score_normalized"
        ).fetchall()
    ]
    assert per_model == [0.6, 0.8]
    assert abs(avg - 0.3) < 1e-9
    assert abs(norm - 0.7) < 1e-9


def test_page_average_is_on_the_canonical_scale(tmp_path, monkeypatch):
    """The page average, like every other comparison on this view, is computed
    on `score_canonical`.

    The rows publish 40.0 and 60.0 because that is what the source printed; the
    page number is 0.5, the mean of 0.40 and 0.60 on the metric's registry
    scale. Averaging the published column instead works only while every row
    on the page happens to share one scale — and a `mixed` cell publishes the
    canonical number outright, so one row on the wrong scale silently turns
    the page average into arithmetic over two units."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _reset_mmlu(con)
        _add_metric(con, "pct", [0.0, 0.0])
        con.execute(
            "UPDATE fact_results SET score = 40.0, score_canonical = 0.40, "
            "scale_conversion = 'div100' WHERE metric_key = 'pct' "
            "AND model_key = (SELECT MIN(model_key) FROM fact_results "
            "                 WHERE metric_key = 'pct')"
        )
        con.execute(
            "UPDATE fact_results SET score = 60.0, score_canonical = 0.60, "
            "scale_conversion = 'div100' WHERE metric_key = 'pct' "
            "AND model_key = (SELECT MAX(model_key) FROM fact_results "
            "                 WHERE metric_key = 'pct')"
        )

    con = _materialise_views(out, mutate=mutate)
    avg, norm = con.execute(
        "SELECT avg_score, avg_score_norm FROM evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()
    assert abs(avg - 0.5) < 1e-9         # mean of the canonical 0.40 / 0.60
    assert abs(norm - 0.5) < 1e-9        # and the normalised twin agrees
    # the rows still carry the source's own numbers
    assert con.execute(
        "SELECT list(score ORDER BY score) FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu' AND metric_id = 'pct'"
    ).fetchone()[0] == [40.0, 60.0]


# ---------------------------------------------------------------------------
# One headline rule, every artifact (M1)
# ---------------------------------------------------------------------------


def _hierarchy_benchmark_nodes(hierarchy: dict):
    """Yield (composite_slug_or_None, node) for every benchmark node in the
    tree, whatever layout the family took (composites[], benchmarks[] or
    standalone_benchmarks[])."""
    for family in hierarchy.get("families", []):
        for composite in family.get("composites", []) or []:
            for node in composite.get("benchmarks", []) or []:
                yield composite.get("key"), node
        for bucket in ("benchmarks", "standalone_benchmarks"):
            for node in family.get(bucket, []) or []:
                # a flat family layout has one composite; its key is the
                # family key in the single-composite case
                yield family.get("key"), node


def _assert_hierarchy_agrees_with_evals_view(con, out_dir):
    from eval_card_backend.canonicalise import sidecars
    import json

    sidecars.write_hierarchy(con, out_dir, {"snapshot_id": "2026-04-30T00:00:00Z"})
    hierarchy = json.loads((out_dir / "hierarchy.json").read_text())

    exact = {
        (slug, bid): metric
        for slug, bid, metric in con.execute(
            "SELECT composite_slug, benchmark_id, primary_metric_id FROM evals_view"
        ).fetchall()
    }
    by_benchmark: dict[str, set] = {}
    for (_slug, bid), metric in exact.items():
        by_benchmark.setdefault(bid, set()).add(metric)

    checked = 0
    for slug, node in _hierarchy_benchmark_nodes(hierarchy):
        key = node.get("key")
        hierarchy_metric = node.get("primary_metric_key")
        if hierarchy_metric is None or key not in by_benchmark:
            continue
        expected = exact.get((slug, key))
        if expected is None:
            # layout gave no reliable composite; the node must still name a
            # metric that heads this benchmark on some page
            assert hierarchy_metric in by_benchmark[key], (
                f"{key}: hierarchy says {hierarchy_metric}, evals_view says "
                f"{sorted(by_benchmark[key])}"
            )
        else:
            assert hierarchy_metric == expected, (
                f"{slug}/{key}: hierarchy says {hierarchy_metric}, "
                f"evals_view says {expected}"
            )
        checked += 1
    return checked


def test_hierarchy_primary_metric_equals_the_views_headline(tmp_path, monkeypatch):
    """The hierarchy annotates the same pages the view scores. A node
    pointing at `cost-per-task` while the page headlines `score` is one
    benchmark with two answers, and the frontend reads both."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    assert _assert_hierarchy_agrees_with_evals_view(con, tmp_path) > 0


def test_hierarchy_demotes_a_diagnostic_metric_like_the_view_does(
    tmp_path, monkeypatch
):
    """The concrete disagreement Sol found: the diagnostic covers more models,
    so the hierarchy's coverage-only rule headlined it while the view demoted
    it. Both now demote it."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _reset_mmlu(con)
        _add_metric(con, "cost-per-task", [1.0, 2.0, 3.0], role="diagnostic")
        _add_metric(con, "task-success", [0.5, 0.6])

    con = _materialise_views(out, mutate=mutate)
    assert _headline(con) == "task-success"
    _assert_hierarchy_agrees_with_evals_view(con, tmp_path)


def test_hierarchy_honours_a_registry_preference_on_the_base_id(
    tmp_path, monkeypatch
):
    """A qualified reading (`accuracy::strict`) IS the preferred metric.
    Comparing the qualified key rejected it and the hierarchy fell back to
    coverage, disagreeing with the view."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")

    def mutate(con):
        _reset_mmlu(con)
        _add_metric(con, "wide-metric", [0.1, 0.2, 0.3])
        _add_metric(con, "task-success", [0.7])
        # publish the preferred metric under a scoring-variant qualifier
        con.execute(
            "UPDATE fact_results SET metric_key = 'task-success::strict', "
            "metric_key_effective = 'task-success::strict', "
            "metric_qualifier = 'strict' "
            "WHERE benchmark_key = 'mmlu' AND metric_base_key = 'task-success'"
        )
        con.execute(
            "UPDATE canonical_benchmarks SET preferred_metric_id = "
            "'task-success' WHERE id = 'mmlu'"
        )

    con = _materialise_views(out, mutate=mutate)
    assert _headline(con) == "task-success::strict"
    _assert_hierarchy_agrees_with_evals_view(con, tmp_path)

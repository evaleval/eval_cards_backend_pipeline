"""Stage J merged_evals_view + fold/scale-conversion columns.

Covers the merged-benchmark-view spec's pipeline surface:
  - P2: `metric_id_effective` (registry naming folds) on eval_results_view
  - P3: `score_canonical` / `scale_conversion` (group-suspect,
    per-row-unambiguous conversion)
  - P5: one merged summary row per resolved canonical benchmark;
    slice-grain rows for benchmarks with no top-level data
  - P6: comparison-index merged entries carry the family_id key;
    benchmark_index carries preferred_metric_id

Uses the standard fixtures + a view-materialise helper that ALSO loads
the registry fixture dims (canonical_benchmarks) so the merged view has
a resolved-benchmark universe; fold/preferred rows are injected per-test
via the mutate hook.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _run_through_stage_i(tmp_path, monkeypatch, config: str) -> Path:
    eee_root = FIXTURES / "eee"
    cards_root = FIXTURES / "auto_benchmarkcards"
    reg_root = FIXTURES / "entity_registry"
    warehouse = tmp_path / "warehouse"

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        configs=[config],
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
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
    alias_store = registry_src.load_alias_store(FIXTURES / "entity_registry")
    register_udfs(con, Resolver(alias_store))
    for table in (
        "fact_results", "benchmarks", "composites", "families", "models",
        "canonical_metrics",
    ):
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_parquet('{out_dir}/{table}.parquet')"
        )
    # Registry dim the merged view keys its universe on. The committed
    # fixture parquet predates preferred_metric_id; pad it like _load_dim.
    con.execute(
        "CREATE TABLE canonical_benchmarks AS "
        "SELECT id, display_name, CAST(NULL AS VARCHAR) AS preferred_metric_id "
        f"FROM read_parquet('{FIXTURES}/entity_registry/canonical_benchmarks.parquet')"
    )
    if mutate is not None:
        mutate(con)
    stages.stage_j_eval_results_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_models_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_merged_evals_view(con, "2026-04-30T00:00:00Z")
    return con


@pytest.fixture(scope="module")
def clean_out_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("merged_clean")
    mp = pytest.MonkeyPatch()
    try:
        yield _run_through_stage_i(tmp, mp, "fixtures_clean")
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def slices_out_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("merged_slices")
    mp = pytest.MonkeyPatch()
    try:
        yield _run_through_stage_i(tmp, mp, "fixtures_slices")
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# P5 — merged row basics
# ---------------------------------------------------------------------------


def test_merged_row_benchmark_grain(clean_out_dir):
    con = _materialise_views(clean_out_dir)
    row = con.execute(
        "SELECT evaluation_id, grain, preferred_metric_id, "
        "       preferred_from_registry, sources_count, results_count, "
        "       models_count, best_result, aggregate_sources, metrics "
        "FROM merged_evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()
    assert row is not None
    (eval_id, grain, preferred, from_registry, sources, results,
     models, best, agg_sources, metrics) = row
    assert eval_id == "mmlu"          # single-segment, no %2F
    assert grain == "benchmark"
    assert preferred == "accuracy"    # Q1 fallback: most observations
    assert from_registry is False
    assert sources >= 1 and results >= 1 and models >= 1
    assert best["model_name"] is not None
    assert best["score_canonical"] == pytest.approx(best["score"])
    assert len(agg_sources) == sources
    assert agg_sources[0]["slice_only"] is False
    assert any(m["metric_id"] == "accuracy" for m in metrics)


def test_merged_ids_disjoint_from_per_source(clean_out_dir):
    con = _materialise_views(clean_out_dir)
    overlap = con.execute(
        "SELECT count(*) FROM merged_evals_view m "
        "JOIN evals_view e USING (evaluation_id)"
    ).fetchone()[0]
    assert overlap == 0
    # every per-source id carries the %2F separator; merged ids never do
    bad = con.execute(
        "SELECT count(*) FROM merged_evals_view "
        "WHERE evaluation_id LIKE '%\\%2F%'"
    ).fetchone()[0]
    assert bad == 0


def test_unresolved_benchmarks_get_no_merged_row(clean_out_dir):
    # fixtures_clean includes rows whose benchmark never resolves; the
    # merged universe is canonical-only by construction.
    con = _materialise_views(clean_out_dir)
    stray = con.execute(
        "SELECT count(*) FROM merged_evals_view m "
        "LEFT JOIN canonical_benchmarks cb ON cb.id = m.benchmark_id "
        "WHERE cb.id IS NULL"
    ).fetchone()[0]
    assert stray == 0


# ---------------------------------------------------------------------------
# P2 — metric folds
# ---------------------------------------------------------------------------


def _inject_fold(con):
    con.execute(
        "CREATE TABLE benchmark_metric_folds AS "
        "SELECT 'mmlu' AS benchmark_id, 'accuracy' AS from_metric_id, "
        "       'score' AS to_metric_id, CAST(NULL AS VARCHAR) AS note"
    )


def test_fold_produces_metric_id_effective(clean_out_dir):
    con = _materialise_views(clean_out_dir, mutate=_inject_fold)
    rows = con.execute(
        "SELECT DISTINCT metric_id, metric_id_effective "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu'"
    ).fetchall()
    assert ("accuracy", "score") in rows
    # raw metric_id is never overwritten
    assert all(m == "accuracy" for m, _ in rows if m == "accuracy")
    # merged default follows the folded (effective) id
    preferred = con.execute(
        "SELECT preferred_metric_id FROM merged_evals_view "
        "WHERE benchmark_id = 'mmlu'"
    ).fetchone()[0]
    assert preferred == "score"


def test_registry_preferred_metric_wins(clean_out_dir):
    def mutate(con):
        con.execute(
            "UPDATE canonical_benchmarks SET preferred_metric_id = 'accuracy' "
            "WHERE id = 'mmlu'"
        )
    con = _materialise_views(clean_out_dir, mutate=mutate)
    row = con.execute(
        "SELECT preferred_metric_id, preferred_from_registry "
        "FROM merged_evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()
    assert row == ("accuracy", True)


# ---------------------------------------------------------------------------
# P3 — canonical-scale conversion
# ---------------------------------------------------------------------------


def _clone_mmlu_row(con, model_key: str, score: float):
    con.execute(
        "INSERT INTO fact_results SELECT * REPLACE ("
        f"  '{model_key}' AS model_aggregation_key,"
        f"  '{model_key}' AS model_raw,"
        f"  '{model_key}' AS model_key,"
        f"  CAST(NULL AS VARCHAR) AS model_id,"
        f"  '{model_key}-fact' AS fact_id,"
        f"  {score} AS score"
        ") FROM fact_results WHERE benchmark_key = 'mmlu' AND score IS NOT NULL "
        "LIMIT 1"
    )


def test_scale_conversion_per_row_unambiguous(clean_out_dir):
    def mutate(con):
        # mixed-scale group under accuracy's [0,1] bounds: a percent-style
        # row, a genuine fraction, and an ambiguous 1–1.5-band value
        _clone_mmlu_row(con, "synthetic-percent", 91.0)
        _clone_mmlu_row(con, "synthetic-ambiguous", 1.2)
        _clone_mmlu_row(con, "synthetic-overrange", 2123.0)
    con = _materialise_views(clean_out_dir, mutate=mutate)
    rows = dict(con.execute(
        "SELECT model_key, (scale_conversion, score_canonical) "
        "FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu' AND metric_id = 'accuracy'"
    ).fetchall())
    conv, canon = rows["synthetic-percent"]
    assert conv == "div100" and canon == pytest.approx(0.91)
    conv, canon = rows["synthetic-ambiguous"]
    assert conv == "flagged" and canon is None
    conv, canon = rows["synthetic-overrange"]
    assert conv == "flagged" and canon is None
    # the genuine in-bounds row is untouched even inside the suspect group
    genuine = [v for k, v in rows.items() if not k.startswith("synthetic")]
    assert all(c == "none" and s is not None for c, s in genuine)


def test_no_bounds_metric_passes_through(clean_out_dir):
    def mutate(con):
        con.execute(
            "UPDATE canonical_metrics SET min_score = NULL, max_score = NULL "
            "WHERE id = 'accuracy'"
        )
    con = _materialise_views(clean_out_dir, mutate=mutate)
    rows = con.execute(
        "SELECT scale_conversion, score, score_canonical "
        "FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu' AND metric_id = 'accuracy' "
        "AND score IS NOT NULL"
    ).fetchall()
    assert rows
    assert all(c == "no_bounds" and canon == s for c, s, canon in rows)


def test_flagged_rows_excluded_from_best(clean_out_dir):
    def mutate(con):
        # would win on raw magnitude, but is unconvertible (2123/100 is
        # still out of bounds) → must not become best_result
        _clone_mmlu_row(con, "synthetic-overrange", 2123.0)
        # forces the group suspect so 2123 lands in the flagged branch
        _clone_mmlu_row(con, "synthetic-percent", 91.0)
    con = _materialise_views(clean_out_dir, mutate=mutate)
    best = con.execute(
        "SELECT best_result FROM merged_evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()[0]
    assert best["model_key"] != "synthetic-overrange"
    assert best["score_canonical"] <= 1.0


# ---------------------------------------------------------------------------
# P5 — slice-grain merged pages
# ---------------------------------------------------------------------------


def test_slice_only_benchmark_gets_slice_grain_row(slices_out_dir):
    con = _materialise_views(slices_out_dir)
    rows = con.execute(
        "SELECT benchmark_id, grain, slices FROM merged_evals_view "
        "WHERE grain = 'slice'"
    ).fetchall()
    top_level = {
        r[0] for r in con.execute(
            "SELECT DISTINCT benchmark_id FROM eval_results_view "
            "WHERE NOT is_slice AND score IS NOT NULL"
        ).fetchall()
    }
    for bid, grain, slices in rows:
        assert bid not in top_level
        assert slices and all(s["slice_id"] for s in slices)


# ---------------------------------------------------------------------------
# P6 — sidecar surface
# ---------------------------------------------------------------------------


def test_comparison_index_merged_entries(clean_out_dir, tmp_path):
    from eval_card_backend.canonicalise import sidecars

    con = _materialise_views(clean_out_dir)
    path = sidecars.write_comparison_index(
        con, tmp_path, {"snapshot_id": "2026-04-30T00:00:00Z"}
    )
    payload = json.loads(path.read_text())
    merged = payload["evals"].get("mmlu")
    assert merged is not None
    assert merged["is_merged"] is True
    assert "family_id" in merged          # old-frontend shape assertion
    assert merged["composite_slug"] is None
    assert merged["parent_benchmark_id"] is None
    [metric] = merged["metrics"]
    assert metric["metric_id"] == "accuracy"
    assert metric["scores"]
    assert all("source_composite_slug" in s for s in metric["scores"])
    # per-source entries are untouched and still carry family_id
    per_source = [e for k, e in payload["evals"].items() if "%2F" in k]
    assert per_source and all("family_id" in e for e in per_source)


def test_benchmark_index_preferred_metric(clean_out_dir, tmp_path):
    from eval_card_backend.canonicalise import sidecars

    con = _materialise_views(clean_out_dir)
    path = sidecars.write_benchmark_index(
        con, tmp_path, {"snapshot_id": "2026-04-30T00:00:00Z"}
    )
    payload = json.loads(path.read_text())
    entry = payload["benchmarks"]["mmlu"]
    assert entry["preferred_metric_id"] == "accuracy"
    assert all(
        "preferred_metric_conversion" in a for a in entry["appearances"]
    )


def test_hierarchy_nodes_carry_benchmark_id(clean_out_dir, tmp_path):
    from eval_card_backend.canonicalise import sidecars

    con = _materialise_views(clean_out_dir)
    path = sidecars.write_hierarchy(
        con, tmp_path, {"snapshot_id": "2026-04-30T00:00:00Z"}
    )
    payload = json.loads(path.read_text())

    seen = []
    def walk_benchmarks(node):
        for b in node.get("benchmarks") or []:
            seen.append(b)
        for c in node.get("composites") or []:
            walk_benchmarks(c)
    for fam in payload["families"]:
        walk_benchmarks(fam)
        for b in fam.get("standalone_benchmarks") or []:
            seen.append(b)
    assert seen
    with_id = [b for b in seen if b.get("benchmark_id")]
    assert with_id, "no hierarchy node carries benchmark_id"
    merged_ids = {
        r[0] for r in con.execute(
            "SELECT benchmark_id FROM merged_evals_view"
        ).fetchall()
    }
    assert all(b["benchmark_id"] in merged_ids for b in with_id)

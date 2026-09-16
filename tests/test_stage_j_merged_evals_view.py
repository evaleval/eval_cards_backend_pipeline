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

import itertools

import json
import shutil
from pathlib import Path

import duckdb
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Skeleton of a fixture EEE record; tests fill in the identity + score they
# need. Scale classification and the rename rule are applied at resolution
# time now, so a scale case is a corpus + registry pair, not a post-hoc patch
# on the view connection.
_RECORD = {
    "evaluation_id": None,
    "schema_version": "0.2.2",
    "retrieved_timestamp": "2026-04-30T00:00:00Z",
    "evaluation_timestamp": "2026-04-29T00:00:00Z",
    "model_info": {"developer": "synthetic", "name": None, "id": None},
    "source_metadata": {
        "source_name": "OpenAI",
        "source_type": "evaluation_run",
        "source_organization_name": "OpenAI",
        "evaluator_relationship": "first_party",
    },
    "eval_library": {"name": "minibench", "version": "1.0"},
    "evaluation_results": [{
        "evaluation_name": "mmlu",
        "source_data": {"dataset_name": "mmlu", "source_type": "other"},
        "metric_config": {
            "metric_id": "mmlu.acc",
            "metric_name": "Accuracy",
            "evaluation_description": "Accuracy on mmlu",
            "score_type": "continuous",
            "min_score": 0.0,
            "max_score": 1.0,
            "lower_is_better": False,
        },
        "score_details": {"score": None},
    }],
}


def _record(model: str, score: float) -> str:
    payload = json.loads(json.dumps(_RECORD))
    payload["evaluation_id"] = f"ev_{model}"
    payload["model_info"]["name"] = model
    payload["model_info"]["id"] = model
    payload["evaluation_results"][0]["score_details"]["score"] = score
    return json.dumps(payload)


def _eee_with(tmp_path, extra, config="fixtures_clean", name="eee"):
    """Copy of the fixture EEE tree plus synthetic mmlu rows. Each entry is
    `(model, score)` under `config`, or `(config, model, score)` to place it
    in another source."""
    from tests.eee_layout import write_eee_datastore

    root = tmp_path / name
    if not root.exists():
        shutil.copytree(FIXTURES / "eee", root)
    rows = [e if len(e) == 3 else (config, *e) for e in extra]
    write_eee_datastore(root, [
        (cfg, f"{model}.json", _record(model, score))
        for cfg, model, score in rows
    ])
    return root


def _registry_with(tmp_path, *, bounds=None, folds=None, name="registry"):
    """Copy of the fixture registry carrying the curated rows a test needs
    BEFORE the pipeline runs: `bounds` re-points a canonical metric's registry
    bounds, `folds` writes `benchmark_metric_folds` rows as
    (benchmark, from_metric, to_metric, factor, offset, source_config)."""
    root = tmp_path / name
    if not root.exists():
        shutil.copytree(FIXTURES / "entity_registry", root)
    con = duckdb.connect()
    if bounds:
        con.execute(
            "CREATE TABLE cm AS SELECT * FROM "
            f"read_parquet('{root}/canonical_metrics.parquet')"
        )
        for metric_id, (lo, hi) in bounds.items():
            con.execute(
                "UPDATE cm SET min_score = ?, max_score = ? WHERE id = ?",
                [lo, hi, metric_id],
            )
        con.execute(f"COPY cm TO '{root}/canonical_metrics.parquet' (FORMAT PARQUET)")
    if folds:
        con.execute(
            "CREATE TABLE bmf (benchmark_id VARCHAR, from_metric_id VARCHAR, "
            "to_metric_id VARCHAR, scale_factor DOUBLE, scale_offset DOUBLE, "
            "source_config VARCHAR, note VARCHAR)"
        )
        for row in folds:
            con.execute("INSERT INTO bmf VALUES (?,?,?,?,?,?,NULL)", list(row))
        con.execute(
            f"COPY bmf TO '{root}/benchmark_metric_folds.parquet' (FORMAT PARQUET)"
        )
    return root


def _run_through_stage_i(tmp_path, monkeypatch, config: str,
                         *, eee_root=None, registry_root=None,
                         configs=None) -> Path:
    eee_root = eee_root or FIXTURES / "eee"
    cards_root = FIXTURES / "auto_benchmarkcards"
    reg_root = registry_root or FIXTURES / "entity_registry"
    warehouse = tmp_path / "warehouse"

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        configs=configs or [config],
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
    # fixture parquet predates preferred_metric_id /
    # preferred_metric_llm_judged; pad it like _load_dim does.
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


# Function-scoped ON PURPOSE, matching every other stage_j test file: a
# module-scoped fixture with its own MonkeyPatch runs BEFORE conftest's
# per-test autouse env stripping, so in CI the ambient *_REVISION pins
# leak into pipeline.run — which wipes the shared fixture dirs and
# downloads the real datasets (hangs this module, empties every module
# after it). Per-test setup keeps the conftest ordering guarantees.
@pytest.fixture
def clean_out_dir(tmp_path, monkeypatch):
    return _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")


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


_NAMING_FOLD = ("mmlu", "accuracy", "score", None, None, None)


def test_fold_produces_metric_id_effective(tmp_path, monkeypatch):
    """A naming rule renames at resolution: the row's grouping/link identity
    IS the renamed metric everywhere, and only the fact-level `metric_id` and
    the source label still say what the source called it."""
    out = _run_through_stage_i(
        tmp_path, monkeypatch, "fixtures_clean",
        registry_root=_registry_with(tmp_path, folds=[_NAMING_FOLD]),
    )
    con = _materialise_views(out)
    rows = con.execute(
        "SELECT DISTINCT metric_id, metric_id_effective, metric_source_label "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu'"
    ).fetchall()
    assert rows == [("score", "score", "Accuracy")]
    # the pre-rename id survives on the fact row, where the rule keys on it
    assert con.execute(
        "SELECT DISTINCT metric_id, metric_key FROM fact_results "
        "WHERE benchmark_key = 'mmlu'"
    ).fetchall() == [("accuracy", "score")]
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


_clone_seq = itertools.count()


def _clone_mmlu_row(con, model_key: str, score: float, metric: str | None = None):
    """Clone a scored mmlu fact row under a new model (and optionally a new
    metric). Scale classification already ran in Stage D, so the clone carries
    the canonical score explicitly rather than inheriting the source row's.

    The clone is a `root` row — a submitted benchmark-level reading — because
    that is what these tests are about (coverage, scale, defaults). Several of
    them in one cell are settings of one measurement and pool by median;
    leaving them `leaf` would make the cell an aggregate-less pile with no
    value at all, which is a different test.

    fact_id is unique per clone: it is the warehouse's row identity, and a
    repeat would fan out every fact-grain join Stage J makes."""
    metric_replace = (
        f", '{metric}' AS metric_key, '{metric}' AS metric_key_effective,"
        f"  '{metric}' AS metric_base_key, '{metric}' AS metric_id"
        if metric else ""
    )
    con.execute(
        "INSERT INTO fact_results SELECT * REPLACE ("
        f"  '{model_key}' AS model_aggregation_key,"
        f"  '{model_key}' AS model_raw,"
        f"  '{model_key}' AS model_key,"
        f"  CAST(NULL AS VARCHAR) AS model_id,"
        f"  '{model_key}-{metric or 'mmlu'}-{next(_clone_seq)}-fact' AS fact_id,"
        f"  {score} AS score,"
        f"  {score} AS score_canonical,"
        f"  'root' AS aggregate_level"
        f"{metric_replace}"
        ") FROM fact_results WHERE benchmark_key = 'mmlu' AND score IS NOT NULL "
        "LIMIT 1"
    )


def test_default_metric_prefers_distinct_model_coverage(clean_out_dir):
    """Fallback default metric counts models before observations: five rows
    across two models say less about a page than three across three."""
    def mutate(con):
        con.execute("DELETE FROM fact_results WHERE benchmark_key = 'mmlu'"
                    " AND metric_key <> 'accuracy'")
        for i in range(3):
            _clone_mmlu_row(con, f"cov-a-{i}", 0.5 + i / 100, metric="metric-a")
        for i in range(5):
            _clone_mmlu_row(con, f"cov-b-{i % 2}", 0.6 + i / 100, metric="metric-b")
        con.execute(
            "INSERT INTO canonical_metrics (id, display_name, min_score, "
            "max_score, lower_is_better) VALUES "
            "('metric-a', 'Metric A', 0, 1, FALSE), "
            "('metric-b', 'Metric B', 0, 1, FALSE)"
        )
    con = _materialise_views(clean_out_dir, mutate=mutate)
    assert con.execute(
        "SELECT preferred_metric_id, preferred_from_registry "
        "FROM merged_evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone() == ("metric-a", False)

    def mutate_preferred(con):
        mutate(con)
        con.execute(
            "UPDATE canonical_benchmarks SET preferred_metric_id = 'metric-b' "
            "WHERE id = 'mmlu'"
        )
    con2 = _materialise_views(clean_out_dir, mutate=mutate_preferred)
    assert con2.execute(
        "SELECT preferred_metric_id, preferred_from_registry "
        "FROM merged_evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone() == ("metric-b", True)


def test_a_preferred_metric_with_no_headline_rows_does_not_win(clean_out_dir):
    """The registry preference only wins when the metric actually reads on
    the page. `metric-c` is published for four models but only as assisted
    runs, which are never headline, so the fallback picks `metric-a`. Both
    the merged page and the per-source page apply the one rule."""
    assisted = '{"feedback":"answer_feedback"}'

    def mutate(con):
        con.execute("DELETE FROM fact_results WHERE benchmark_key = 'mmlu'"
                    " AND metric_key <> 'accuracy'")
        for i in range(3):
            _clone_mmlu_row(con, f"cov-a-{i}", 0.5 + i / 100, metric="metric-a")
        for i in range(4):
            con.execute(
                "INSERT INTO fact_results SELECT * REPLACE ("
                f"  'cov-c-{i}' AS model_aggregation_key,"
                f"  'cov-c-{i}' AS model_raw,"
                f"  'cov-c-{i}' AS model_key,"
                "  CAST(NULL AS VARCHAR) AS model_id,"
                f"  'cov-c-{i}-fact' AS fact_id,"
                "  'metric-c' AS metric_key,"
                "  'metric-c' AS metric_key_effective,"
                "  'metric-c' AS metric_id,"
                f"  {0.7 + i / 100} AS score,"
                f"  {0.7 + i / 100} AS score_canonical,"
                f"  '{assisted}' AS protocol_condition"
                ") FROM fact_results WHERE benchmark_key = 'mmlu'"
                " AND score IS NOT NULL LIMIT 1"
            )
        con.execute(
            "INSERT INTO canonical_metrics (id, display_name, min_score, "
            "max_score, lower_is_better) VALUES "
            "('metric-a', 'Metric A', 0, 1, FALSE), "
            "('metric-c', 'Metric C', 0, 1, FALSE)"
        )
        con.execute(
            "UPDATE canonical_benchmarks SET preferred_metric_id = 'metric-c' "
            "WHERE id = 'mmlu'"
        )

    con = _materialise_views(clean_out_dir, mutate=mutate)
    assert con.execute(
        "SELECT preferred_metric_id, preferred_from_registry "
        "FROM merged_evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone() == ("metric-a", False)
    assert con.execute(
        "SELECT DISTINCT primary_metric_id FROM evals_view "
        "WHERE benchmark_id = 'mmlu'"
    ).fetchall() == [("metric-a",)]


# ---------------------------------------------------------------------------
# P3 — canonical-scale conversion (Stage D)
# ---------------------------------------------------------------------------


def test_scale_conversion_per_row_unambiguous(tmp_path, monkeypatch):
    # mixed-scale group under accuracy's [0,1] bounds: a percent-style row, a
    # genuine fraction, and an ambiguous 1-1.5-band value
    out = _run_through_stage_i(
        tmp_path, monkeypatch, "fixtures_clean",
        eee_root=_eee_with(tmp_path, [
            ("synthetic-percent", 91.0),
            ("synthetic-ambiguous", 1.2),
            ("synthetic-overrange", 2123.0),
        ]),
    )
    con = _materialise_views(out)
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


def test_curated_scale_factor(tmp_path, monkeypatch):
    """A source-scoped rule converts by affine fact — factor AND offset, no
    detection. The WildBench rescale (x-1)/9 is the live case: raw 1 and 10
    must land exactly on the metric's [0,1] endpoints."""
    factor, offset = 0.1111111111111111, -0.1111111111111111
    out = _run_through_stage_i(
        tmp_path, monkeypatch, "fixtures_clean",
        eee_root=_eee_with(tmp_path, [("rescale-lo", 1.0), ("rescale-hi", 10.0)]),
        registry_root=_registry_with(
            tmp_path,
            bounds={"score": (0.0, 1.0)},
            folds=[
                _NAMING_FOLD,
                ("mmlu", "accuracy", "score", factor, offset, "fixtures_clean"),
            ],
        ),
    )
    con = _materialise_views(out)
    # a curated conversion is a stated fact about the source's scale, so the
    # view's displayed `score` IS the canonical number and the published one
    # moves to `score_published`
    rows = dict(con.execute(
        "SELECT model_key, "
        "(score_published, score, score_canonical, scale_conversion) "
        "FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu' AND metric_id = 'score' "
        "AND score IS NOT NULL"
    ).fetchall())
    assert rows["rescale-lo"][2] == pytest.approx(0.0, abs=1e-9)
    assert rows["rescale-hi"][2] == pytest.approx(1.0, abs=1e-9)
    assert all(
        canon == pytest.approx(raw * factor + offset, abs=1e-9)
        and shown == pytest.approx(canon)
        for raw, shown, canon, conv in rows.values() if conv == "curated"
    )
    # a row the affine rule takes out of the metric's range is flagged, never
    # silently placed on the canonical scale
    assert rows["openai/gpt-4o"][2:] == (None, "flagged")


def test_conversion_is_scoped_to_the_source_that_declared_it(tmp_path, monkeypatch):
    """A second publisher of the same benchmark + metric gets the RENAME but
    not the conversion: its published scale is its own, and guessing one for
    it is how a 33-68 WildBench number becomes a plausible-looking 0.9."""
    eee_root = _eee_with(tmp_path, [
        ("fixtures_clean", "scoped-source", 0.9),
        ("fixtures_other", "other-source", 333.0),
    ])
    out = _run_through_stage_i(
        tmp_path, monkeypatch, "fixtures_clean",
        configs=["fixtures_clean", "fixtures_other"],
        eee_root=eee_root,
        registry_root=_registry_with(
            tmp_path,
            bounds={"score": (0.0, 1.0)},
            folds=[
                _NAMING_FOLD,
                ("mmlu", "accuracy", "score", 0.1, -0.04, "fixtures_clean"),
            ],
        ),
    )
    con = _materialise_views(out)
    rows = dict(con.execute(
        "SELECT model_key, (scale_conversion, score_canonical) "
        "FROM eval_results_view WHERE benchmark_id = 'mmlu' "
        "AND metric_id = 'score' "
        "AND model_key IN ('scoped-source', 'other-source')"
    ).fetchall())
    assert rows["scoped-source"][0] == "curated"
    assert rows["scoped-source"][1] == pytest.approx(0.9 * 0.1 - 0.04)
    assert rows["other-source"] == ("flagged", None)


def test_no_bounds_metric_passes_through(tmp_path, monkeypatch):
    out = _run_through_stage_i(
        tmp_path, monkeypatch, "fixtures_clean",
        registry_root=_registry_with(tmp_path, bounds={"accuracy": (None, None)}),
    )
    con = _materialise_views(out)
    rows = con.execute(
        "SELECT scale_conversion, score, score_canonical, score_normalized "
        "FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu' AND metric_id = 'accuracy' "
        "AND score IS NOT NULL"
    ).fetchall()
    assert rows
    assert all(c == "no_bounds" and canon == s for c, s, canon, _n in rows)
    # No registry bounds means no scale to normalise against, and there is no
    # default one: a fabricated [0, 1] reported every Arena Elo as a perfect
    # 1.0. The column says "not normalisable" instead.
    assert all(n is None for _c, _s, _canon, n in rows)


def test_flagged_rows_excluded_from_best(tmp_path, monkeypatch):
    # synthetic-overrange would win on raw magnitude but is unconvertible
    # (2123/100 is still out of bounds); synthetic-percent forces the group
    # suspect so it lands in the flagged branch
    out = _run_through_stage_i(
        tmp_path, monkeypatch, "fixtures_clean",
        eee_root=_eee_with(tmp_path, [
            ("synthetic-overrange", 2123.0),
            ("synthetic-percent", 91.0),
        ]),
    )
    con = _materialise_views(out)
    best = con.execute(
        "SELECT best_result FROM merged_evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()[0]
    assert best["model_key"] != "synthetic-overrange"
    assert best["score_canonical"] <= 1.0


def test_mul100_and_percent_ambiguity_band(tmp_path, monkeypatch):
    # under [0,100] bounds, an all-fraction group converts whole; a group
    # topping out in (1, 1.5] is ambiguous and flagged whole
    registry = _registry_with(tmp_path, bounds={"accuracy": (0.0, 100.0)})
    out = _run_through_stage_i(
        tmp_path, monkeypatch, "fixtures_clean", registry_root=registry,
    )
    con = _materialise_views(out)
    rows = con.execute(
        "SELECT score, score_canonical, scale_conversion "
        "FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu' AND metric_id = 'accuracy' "
        "AND score IS NOT NULL"
    ).fetchall()
    assert rows
    assert all(c == "mul100" and canon == pytest.approx(s * 100) for s, canon, c in rows)

    out2 = _run_through_stage_i(
        tmp_path, monkeypatch, "fixtures_clean",
        eee_root=_eee_with(tmp_path, [("synthetic-onepointtwo", 1.2)]),
        registry_root=registry,
    )
    con2 = _materialise_views(out2)
    rows2 = con2.execute(
        "SELECT scale_conversion FROM eval_results_view "
        "WHERE benchmark_id = 'mmlu' AND metric_id = 'accuracy' "
        "AND score IS NOT NULL"
    ).fetchall()
    assert rows2 and all(c == ("flagged",) for c in map(tuple, rows2))


# ---------------------------------------------------------------------------
# P5 — slice-grain merged pages
# ---------------------------------------------------------------------------


def _reparent_mmlu_under_shell(con):
    """Turn mmlu into a slice of a new fact-less shell parent, so the
    shell qualifies as a slice-grain merged page (no top-level data)."""
    con.execute(
        "INSERT INTO canonical_benchmarks (id, display_name, "
        "preferred_metric_id, preferred_metric_llm_judged) "
        "VALUES ('mmlu-suite', 'MMLU Suite', NULL, NULL)"
    )
    con.execute(
        "INSERT INTO benchmarks SELECT * REPLACE ("
        "  'mmlu-suite' AS benchmark_id,"
        "  'MMLU Suite' AS display_name,"
        "  'MMLU Suite' AS benchmark_display_name,"
        "  FALSE AS is_slice,"
        "  CAST(NULL AS VARCHAR) AS parent_benchmark_id"
        ") FROM benchmarks WHERE benchmark_id = 'mmlu'"
    )
    con.execute(
        "UPDATE benchmarks SET parent_benchmark_id = 'mmlu-suite', "
        "is_slice = TRUE WHERE benchmark_id = 'mmlu'"
    )
    # the registry edge behind it — the parent's expected task set, which the
    # rollup requires to be complete before it states a suite number
    con.execute(
        "UPDATE canonical_benchmarks SET parent_benchmark_id = 'mmlu-suite' "
        "WHERE id = 'mmlu'"
    )


def test_slice_only_benchmark_gets_benchmark_grain_row(clean_out_dir):
    """A parent whose only data is its slice children carries a value rolled
    up from those children, so the merged page is benchmark-grain with a
    headline number."""
    con = _materialise_views(clean_out_dir, mutate=_reparent_mmlu_under_shell)
    row = con.execute(
        "SELECT grain, slices, best_result, aggregate_sources "
        "FROM merged_evals_view WHERE benchmark_id = 'mmlu-suite'"
    ).fetchone()
    assert row is not None, "shell parent got no merged row"
    grain, slices, best, agg_sources = row
    # The shell now carries a benchmark-level value of its own, rolled up
    # from its children, so it is no longer a slice-only page: it has a
    # comparable per-model number and therefore a best_result.
    assert grain == "benchmark"
    assert best["model_name"] is not None
    # mmlu itself still gets no benchmark-grain row (it is a slice now)
    assert con.execute(
        "SELECT count(*) FROM merged_evals_view WHERE benchmark_id = 'mmlu'"
    ).fetchone()[0] == 0


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

"""End-to-end tests over the hand-built fixture corpus in `tests/fixtures/`.

Covers seven edge cases:
  01 — clean resolution (all 5 entities exact)
  02 — model no_match (community fine-tune)
  03 — agentic via benchmark card tasks tag
  04 — agentic via generation_args.agentic_eval_config
  05 — variant divergence (3 rows, differing setups)
  06 — cross-party divergence (2 orgs, casing/whitespace edge case)
  07 — no-score row (dropped in Stage E with counter increment)

Each test runs the full pipeline against the fixture corpus and asserts
the specific behaviour the fixture exercises.

Tests use the real `eval-entity-resolver` against the fixture
aliases.parquet — same code path as production. The 02-no-match-model
fixture deliberately omits `community/fine-tune-7b` from the alias store
to exercise the no_match path.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _run_pipeline_per_config(tmp_path, monkeypatch, config: str):
    """Run the canonicalise pipeline scoped to a single fixture config.
    Returns the warehouse output dir."""
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

    settings = Settings.from_env()
    return pipeline.run(
        settings,
        configs=[config],
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        cache_root=str(tmp_path / "cache"),
    )


def _facts(out_dir: Path):
    con = duckdb.connect()
    return con.execute(
        f"SELECT * FROM read_parquet('{out_dir}/fact_results.parquet') ORDER BY evaluation_id"
    ).fetchdf()


# ---------------------------------------------------------------------------
# Fixture 01 — clean resolution
# ---------------------------------------------------------------------------


def test_fixture_01_clean_resolution(tmp_path, monkeypatch):
    """01 + 07 land in the same config; this asserts on the cleanly-resolved
    row alone (07 is dropped in Stage E)."""
    out = _run_pipeline_per_config(tmp_path, monkeypatch, "fixtures_clean")
    df = _facts(out)
    # fixtures_clean carries 01, 02, 07. 07 is dropped (no score) → 2 rows.
    assert len(df) == 2

    row01 = df[df["evaluation_id"] == "ev_01"].iloc[0]
    assert row01["model_id"] == "openai/gpt-4o"
    assert row01["benchmark_id"] == "mmlu"
    assert row01["metric_id"] == "accuracy"
    assert row01["org_id"] == "openai"
    assert row01["model_resolution_strategy"] == "exact"
    assert row01["benchmark_resolution_strategy"] == "exact"
    assert row01["metric_resolution_strategy"] == "exact"
    # Card-present path
    assert row01["benchmark_card_id"] == "mmlu"
    # Non-agentic
    assert bool(row01["is_agentic"]) is False


# ---------------------------------------------------------------------------
# Fixture 02 — model no_match
# ---------------------------------------------------------------------------


def test_fixture_02_no_match_model_keeps_row(tmp_path, monkeypatch):
    """02 has a model with no alias entry. The row must be PRESERVED (raw kept,
    canonical NULL) — never dropped for unresolved identity. `model_key`
    falls back to `model_raw` so downstream stages can still address the
    row, and group signals pool by `model_aggregation_key` (which falls
    back to `model_raw` too) so unresolved models still participate in
    comparability groups when their raw strings match."""
    out = _run_pipeline_per_config(tmp_path, monkeypatch, "fixtures_clean")
    df = _facts(out)

    row02 = df[df["evaluation_id"] == "ev_02"].iloc[0]
    assert row02["model_raw"] == "community/fine-tune-7b"
    assert row02["model_id"] is None or (
        # pandas may surface NULL as float NaN — accept either
        row02["model_id"] != row02["model_id"]
    )
    # model_key is the addressable identifier — non-null for any row that
    # had a source-supplied name.
    assert row02["model_key"] == "community/fine-tune-7b"
    assert row02["model_resolution_strategy"] == "no_match"
    # Other entities still resolve
    assert row02["benchmark_id"] == "mmlu"
    assert row02["metric_id"] == "accuracy"
    # Group signals are computed via raw fallback: with all three
    # aggregation keys present (model_raw + benchmark_id + metric_id),
    # the row participates in a comparability group.
    assert row02["comparability_group_id"] is not None
    assert row02["comparability_group_id"] == row02["comparability_group_id"]


def test_fixture_02_unresolved_surfaces_in_models_parquet(tmp_path, monkeypatch):
    """The unresolved community/fine-tune-7b model must appear in
    `models.parquet` and `models_view.parquet` — keyed on `model_key`,
    with `display_name` falling back to the raw source name and
    `review_status='unresolved'` so consumers can flag it."""
    pytest.importorskip("duckdb")
    out = _run_pipeline_per_config(tmp_path, monkeypatch, "fixtures_clean")

    import duckdb
    con = duckdb.connect()
    models = con.execute(
        f"SELECT model_key, model_id, display_name, review_status "
        f"FROM read_parquet('{out}/models.parquet') "
        f"WHERE model_key = 'community/fine-tune-7b'"
    ).fetchone()
    assert models is not None, "unresolved model dropped from models.parquet"
    model_key, model_id, display_name, review_status = models
    assert model_key == "community/fine-tune-7b"
    assert model_id is None
    assert display_name == "community/fine-tune-7b"
    assert review_status == "unresolved"

    view = con.execute(
        f"SELECT model_key, model_id, route_id, model_name "
        f"FROM read_parquet('{out}/models_view.parquet') "
        f"WHERE model_key = 'community/fine-tune-7b'"
    ).fetchone()
    assert view is not None, "unresolved model dropped from models_view.parquet"
    v_model_key, v_model_id, v_route_id, v_model_name = view
    assert v_model_key == "community/fine-tune-7b"
    assert v_model_id is None
    assert v_route_id == "community%2Ffine-tune-7b"
    assert v_model_name == "community/fine-tune-7b"


# ---------------------------------------------------------------------------
# Fixture 03 — agentic via card tasks
# ---------------------------------------------------------------------------


def test_fixture_03_agentic_via_card(tmp_path, monkeypatch):
    """03's benchmark card has tasks=['agentic'] → is_agentic via Rule 1.
    Repro then requires temperature + max_tokens + eval_plan + eval_limits;
    fixture sets all four → has_reproducibility_gap=False."""
    out = _run_pipeline_per_config(tmp_path, monkeypatch, "fixtures_agentic")
    df = _facts(out)

    row03 = df[df["evaluation_id"] == "ev_03"].iloc[0]
    assert bool(row03["is_agentic"]) is True
    assert bool(row03["has_reproducibility_gap"]) is False
    assert row03["repro_required_count"] == 4


# ---------------------------------------------------------------------------
# Fixture 04 — agentic via generation_args.agentic_eval_config
# ---------------------------------------------------------------------------


def test_fixture_04_agentic_via_config_no_card(tmp_path, monkeypatch):
    """04's benchmark (appworld) has NO card → exercises card-missing path.
    is_agentic fires via Rule 2 (agentic_eval_config presence) AND Rule 3
    (regex on 'appworld'). card_present should be FALSE on benchmarks dim."""
    out = _run_pipeline_per_config(tmp_path, monkeypatch, "fixtures_agentic")
    df = _facts(out)

    row04 = df[df["evaluation_id"] == "ev_04"].iloc[0]
    assert bool(row04["is_agentic"]) is True
    # benchmark_card_id NULL because no card matched appworld
    assert row04["benchmark_card_id"] is None or (
        row04["benchmark_card_id"] != row04["benchmark_card_id"]
    )

    # benchmarks.parquet should record card_present=False for appworld
    con = duckdb.connect()
    bench = con.execute(
        f"SELECT benchmark_id, card_present FROM read_parquet('{out}/benchmarks.parquet') "
        f"WHERE benchmark_id = 'appworld'"
    ).fetchone()
    assert bench == ("appworld", False)


# ---------------------------------------------------------------------------
# Fixture 05 — variant divergence
# ---------------------------------------------------------------------------


def test_fixture_05_variant_divergence(tmp_path, monkeypatch):
    """05a/b/c — same (model, benchmark, metric), differing setups, scores
    span 0.50–0.85 > 0.05 threshold (proportion).
    All 3 rows should have has_variant_divergence=True."""
    out = _run_pipeline_per_config(tmp_path, monkeypatch, "fixtures_variant")
    df = _facts(out)
    assert len(df) == 3

    assert all(df["has_variant_divergence"] == True)  # noqa: E712
    assert all(df["variant_threshold_basis"] == "proportion")
    # Magnitude = max - min = 0.85 - 0.50 = 0.35
    assert all(abs(df["variant_divergence_magnitude"] - 0.35) < 1e-9)

    # Differing setup fields list must include both temperature and max_tokens
    fields = {f["field"] for f in df.iloc[0]["variant_differing_fields"]}
    assert fields == {"temperature", "max_tokens"}

    # Single org → cross-party not applicable
    assert all(df["has_cross_party_divergence"].isnull())


# ---------------------------------------------------------------------------
# Fixture 06 — cross-party divergence
# ---------------------------------------------------------------------------


def test_fixture_06_cross_party_divergence(tmp_path, monkeypatch):
    """06a uses 'OpenAI', 06b uses 'Scale AI ' (trailing whitespace). After
    normalize_org_name, two distinct orgs → cross-party fires; magnitude
    0.85 - 0.65 = 0.20 > 0.05 threshold."""
    out = _run_pipeline_per_config(tmp_path, monkeypatch, "fixtures_xparty")
    df = _facts(out)
    assert len(df) == 2

    assert all(df["has_cross_party_divergence"] == True)  # noqa: E712
    assert all(df["cross_party_org_count"] == 2)

    sbo = df.iloc[0]["scores_by_organization"]
    if isinstance(sbo, list):  # MAP type can surface as list of pairs
        sbo = dict(sbo)
    # Display casing preserved (no trailing whitespace; original casing kept)
    assert set(sbo.keys()) == {"OpenAI", "Scale AI"}

    # No variant divergence — setups identical
    assert all(df["has_variant_divergence"].isnull())


# ---------------------------------------------------------------------------
# Fixture 07 — no-score row dropped
# ---------------------------------------------------------------------------


def test_fixture_07_no_score_dropped_with_counter(tmp_path, monkeypatch):
    """07 omits score_details.score → fails Pydantic validation in the
    typed Arrow loader and is dropped at Stage A. The counter surfacing
    the drop is `dropped_eee_records_stage_a` (loader-side rejection),
    not `dropped_rows_no_score` (Stage E score-IS-NULL drop), so the
    operator can tell a malformed record from a record that ran but
    didn't produce a score."""
    out = _run_pipeline_per_config(tmp_path, monkeypatch, "fixtures_clean")
    snap = json.loads((out / "snapshot_meta.json").read_text())

    # fixtures_clean has 3 EEE records (01, 02, 07); 07 dropped at Stage A.
    assert snap["row_counts"]["dropped_eee_records_stage_a"] == 1
    assert snap["row_counts"]["fact_results_pre_drop"] == 2
    assert snap["row_counts"]["fact_results"] == 2

    # ev_07 must NOT appear in fact_results
    df = _facts(out)
    assert "ev_07" not in df["evaluation_id"].tolist()


# ---------------------------------------------------------------------------
# Cross-fixture invariants
# ---------------------------------------------------------------------------


def test_all_configs_run_without_error(tmp_path, monkeypatch):
    """Smoke test: run the pipeline over the full fixture corpus (all 4 configs
    at once). Verifies no cross-config interaction issues."""
    eee_root = FIXTURES / "eee"
    cards_root = FIXTURES / "auto_benchmarkcards"
    reg_root = FIXTURES / "entity_registry"
    warehouse = tmp_path / "warehouse"

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out = pipeline.run(
        Settings.from_env(),
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        cache_root=str(tmp_path / "cache"),
    )

    df = _facts(out)
    # 13 EEE records total (10 + 3 from fixtures_slices); 1 has no score
    # → 12 fact rows.
    assert len(df) == 12

    # Resolved-keys counts
    assert df["benchmark_id"].notna().all()    # all benchmarks resolve
    assert df["metric_id"].notna().all()       # all metrics resolve
    # Only the community fine-tune fails resolution → 1 row with NULL model_id.
    assert df["model_id"].isna().sum() == 1

    # slice_key / slice_name columns are plumbed through Stages C→D→E→F→I.
    # The cross-config snapshot has 3 distinct raws resolving to mmlu —
    # "mmlu", "Anatomy", "Astronomy" — so EVERY row with benchmark_id=mmlu
    # gets a slice_key (including the literal-"mmlu" rows from older
    # fixtures, which represent the overall MMLU rollup as its own
    # slice). Only the swebench-verified + appworld rows stay NULL since
    # those benchmarks have one raw each in the snapshot.
    assert "slice_key" in df.columns
    assert "slice_name" in df.columns
    mmlu_rows = df[df["benchmark_id"] == "mmlu"]
    assert mmlu_rows["slice_key"].notna().all()
    assert set(mmlu_rows["slice_key"]) == {"mmlu", "anatomy", "astronomy"}
    non_mmlu = df[df["benchmark_id"] != "mmlu"]
    assert non_mmlu["slice_key"].isna().all()

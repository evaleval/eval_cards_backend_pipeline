"""Unit + e2e tests for stage caching.

Unit-level: StageCache roundtrip + discovery helpers.
E2E: --to-stage D writes only A..D outputs; --from-stage E re-using a
populated cache produces the same fact_results as a no-flag run.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from eval_card_backend.canonicalise.cache import (
    CACHE_SCHEMA_VERSION,
    STAGE_ORDER,
    STAGE_OUTPUTS,
    StageCache,
    discover_latest_snapshot,
    normalize_snapshot_id,
    validate_letter,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Unit: StageCache roundtrip
# ---------------------------------------------------------------------------


def test_stage_cache_write_load_roundtrip(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE foo AS SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(x, y)")

    cache = StageCache(tmp_path, "2026-04-30T00:00:00Z")
    cache.write_table(con, "foo")
    assert cache.has_table("foo")
    assert (cache.dir / "foo.parquet").exists()

    # Drop and reload
    con.execute("DROP TABLE foo")
    cache.load_table(con, "foo")
    rows = con.execute("SELECT * FROM foo ORDER BY x").fetchall()
    assert rows == [(1, "a"), (2, "b")]


def test_stage_cache_disabled_is_noop(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE foo AS SELECT 1 AS x")

    cache = StageCache(tmp_path, "snap", enabled=False)
    cache.write_table(con, "foo")
    assert not (cache.dir / "foo.parquet").exists()
    assert not cache.has_table("foo")


def test_stage_cache_load_missing_raises(tmp_path):
    con = duckdb.connect()
    cache = StageCache(tmp_path, "snap")
    with pytest.raises(FileNotFoundError, match="cache miss"):
        cache.load_table(con, "doesnt_exist")


def test_stage_cache_cached_tables(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE a AS SELECT 1 AS x")
    con.execute("CREATE TABLE b AS SELECT 2 AS y")
    cache = StageCache(tmp_path, "snap")
    cache.write_table(con, "a")
    cache.write_table(con, "b")
    assert cache.cached_tables() == {"a", "b"}


def _write_stages(con, cache, *letters):
    """Materialise every STAGE_OUTPUTS table for the given stages and cache
    them the way the orchestrator does (one write_stage call per stage)."""
    for letter in letters:
        for table in STAGE_OUTPUTS[letter]:
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT 1 AS x")
        cache.write_stage(con, letter)
        for table in STAGE_OUTPUTS[letter]:
            con.execute(f"DROP TABLE {table}")


def test_stage_cache_restore_through(tmp_path):
    """`restore_through("B")` loads A and B's outputs but not C's."""
    con = duckdb.connect()
    cache = StageCache(tmp_path, "snap")
    _write_stages(con, cache, "A", "B", "C")

    restored = cache.restore_through(con, "B")  # through B inclusive
    assert "results_exploded" in restored
    assert "eee_raw" in restored
    assert "results_resolved" not in restored


def test_writing_a_stage_drops_later_stage_caches(tmp_path):
    """A partial A-D rebuild inside a populated snapshot dir must not leave
    the previous generation's E..J parquets behind for a later resume."""
    con = duckdb.connect()
    cache = StageCache(tmp_path, "snap")
    _write_stages(con, cache, "A", "B", "C", "D", "E", "F")
    assert cache.has_table("fact_results")  # F's output

    _write_stages(con, cache, "D")  # re-run that stops at D

    assert cache.has_table("fact_results_staging")
    assert not cache.has_table("fact_results_signaled")  # E
    assert not cache.has_table("fact_results")  # F
    with pytest.raises(RuntimeError, match="incomplete for stages A..F"):
        cache.restore_through(con, "F")


def test_restore_refuses_an_interrupted_write(tmp_path):
    """Marker present but a declared table missing = a run that died between
    two COPYs. Restoring the survivors would mix generations."""
    con = duckdb.connect()
    cache = StageCache(tmp_path, "snap")
    _write_stages(con, cache, "A", "B")
    (cache.dir / "results_exploded.parquet").unlink()

    cache.assert_schema_current()  # the marker itself is still current
    with pytest.raises(RuntimeError, match="B:results_exploded"):
        cache.restore_through(con, "B")


# ---------------------------------------------------------------------------
# Unit: cache schema fingerprint
# ---------------------------------------------------------------------------


def _seed_cache(tmp_path):
    con = duckdb.connect()
    cache = StageCache(tmp_path, "snap")
    _write_stages(con, cache, "A")
    return con, cache


def test_writing_a_cache_stamps_the_schema_version(tmp_path):
    _, cache = _seed_cache(tmp_path)
    marker = json.loads((cache.dir / "_cache_schema.json").read_text())
    assert marker == {"cache_schema_version": CACHE_SCHEMA_VERSION}
    cache.assert_schema_current()


def test_a_bare_table_write_does_not_bless_the_cache(tmp_path):
    """Only a completed write_stage stamps the marker, so a half-written dir
    reads as stale rather than current."""
    con = duckdb.connect()
    cache = StageCache(tmp_path, "snap")
    con.execute("CREATE TABLE eee_raw AS SELECT 1 AS x")
    cache.write_table(con, "eee_raw")
    assert not (cache.dir / "_cache_schema.json").exists()
    with pytest.raises(RuntimeError, match="cache_schema_version=None"):
        cache.assert_schema_current()


def test_restore_refuses_a_cache_from_a_different_schema(tmp_path):
    con, cache = _seed_cache(tmp_path)
    (cache.dir / "_cache_schema.json").write_text(
        json.dumps({"cache_schema_version": CACHE_SCHEMA_VERSION - 1})
    )
    with pytest.raises(RuntimeError, match="Re-run from Stage A"):
        cache.restore_through(con, "A")


def test_restore_refuses_a_version_12_cache(tmp_path):
    """Pins the transition, not just the mechanism. Stage J's cached
    `eval_results_view` gained `scoring_mode`, so a cache stamped 12 no
    longer describes the tables this pipeline restores. Every other test here
    derives its expectation from the constant, which means putting the
    constant back would leave them all green."""
    con, cache = _seed_cache(tmp_path)
    (cache.dir / "_cache_schema.json").write_text(
        json.dumps({"cache_schema_version": 12})
    )
    with pytest.raises(RuntimeError, match="Re-run from Stage A"):
        cache.restore_through(con, "A")


def test_restore_refuses_a_cache_with_no_fingerprint(tmp_path):
    """A cache written before the fingerprint existed carries an unknown
    column shape, so it is stale by definition rather than trusted."""
    con, cache = _seed_cache(tmp_path)
    (cache.dir / "_cache_schema.json").unlink()
    with pytest.raises(RuntimeError, match="cache_schema_version=None"):
        cache.restore_through(con, "A")


# ---------------------------------------------------------------------------
# Unit: helpers
# ---------------------------------------------------------------------------


def test_validate_letter_canonical_form():
    assert validate_letter("a", label="x") == "A"
    assert validate_letter("E", label="x") == "E"


def test_validate_letter_rejects_unknown():
    with pytest.raises(ValueError, match="not a recognised stage"):
        validate_letter("Z", label="from_stage")
    with pytest.raises(ValueError, match="not a recognised stage"):
        validate_letter("H", label="from_stage")  # H was removed


def test_stage_outputs_keys_match_stage_order():
    assert set(STAGE_OUTPUTS) == set(STAGE_ORDER)


def test_discover_latest_snapshot(tmp_path):
    assert discover_latest_snapshot(tmp_path) is None
    (tmp_path / "2026-04-30T00-00-00Z").mkdir()
    (tmp_path / "2026-05-03T12-30-45Z").mkdir()
    (tmp_path / "2026-05-01T08-00-00Z").mkdir()
    assert discover_latest_snapshot(tmp_path) == "2026-05-03T12:30:45Z"


def test_discover_latest_snapshot_missing_root(tmp_path):
    assert discover_latest_snapshot(tmp_path / "nope") is None


def test_normalize_snapshot_id_accepts_dir_form():
    # Directory form (filesystem-safe) → ISO form (DuckDB-friendly).
    assert normalize_snapshot_id("2026-05-03T00-00-00Z") == "2026-05-03T00:00:00Z"
    # Already-canonical input passes through.
    assert normalize_snapshot_id("2026-05-03T00:00:00Z") == "2026-05-03T00:00:00Z"
    # No 'T' separator → not a snapshot timestamp; leave untouched.
    assert normalize_snapshot_id("nightly") == "nightly"


# ---------------------------------------------------------------------------
# E2E: --to-stage / --from-stage parity
# ---------------------------------------------------------------------------


def _run_full(tmp_path, monkeypatch, *, snapshot_id, cache_root, **kwargs):
    """Full-corpus e2e run with the fixture sources."""
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

    return pipeline.run(
        Settings.from_env(),
        snapshot_id=snapshot_id,
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        cache_root=str(cache_root),
        **kwargs,
    )


def test_to_stage_D_writes_only_stages_a_through_d(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    out = _run_full(
        tmp_path, monkeypatch,
        snapshot_id="2026-04-30T00:00:00Z",
        cache_root=cache_root,
        to_stage="D",
    )
    # No warehouse emit
    assert out is None

    cached = StageCache(cache_root, "2026-04-30T00:00:00Z").cached_tables()
    # All of A's outputs are present
    for table in STAGE_OUTPUTS["A"]:
        assert table in cached, f"A output {table} missing from cache"
    assert "results_exploded" in cached    # B
    assert "results_resolved" in cached    # C
    assert "fact_results_staging" in cached  # D
    # Stages E/F/G/I should NOT have run
    assert "fact_results_signaled" not in cached
    assert "fact_results" not in cached
    assert "benchmarks" not in cached
    assert "models" not in cached


def test_from_stage_E_matches_full_run_output(tmp_path, monkeypatch):
    """Run --to-stage D, then --from-stage E using the same cache. The final
    fact_results should equal a fresh full run on identical fixtures.

    Sanity check that cache restore preserves enough state for downstream
    stages to produce byte-equivalent (or row-equivalent) output.
    """
    cache_root = tmp_path / "cache"
    snapshot = "2026-04-30T00:00:00Z"

    # Step 1: full run (baseline)
    baseline = _run_full(
        tmp_path, monkeypatch,
        snapshot_id=snapshot,
        cache_root=tmp_path / "baseline_cache",
    )
    assert baseline is not None

    # Step 2: --to-stage D (populates the cache)
    _run_full(
        tmp_path, monkeypatch,
        snapshot_id=snapshot,
        cache_root=cache_root,
        to_stage="D",
    )

    # Step 3: --from-stage E (resumes from cache, emits warehouse)
    resumed_out = _run_full(
        tmp_path, monkeypatch,
        snapshot_id=snapshot,
        cache_root=cache_root,
        from_stage="E",
    )
    assert resumed_out is not None

    # Compare fact_results parquets — covers identity, signals, and group-derived
    # comparability columns so a regression in cache restore would surface.
    con = duckdb.connect()
    base_df = con.execute(
        f"SELECT * FROM read_parquet('{baseline / 'fact_results.parquet'}') "
        f"ORDER BY evaluation_id, result_idx"
    ).fetchdf()
    resumed_df = con.execute(
        f"SELECT * FROM read_parquet('{resumed_out / 'fact_results.parquet'}') "
        f"ORDER BY evaluation_id, result_idx"
    ).fetchdf()
    assert len(base_df) == len(resumed_df)
    assert set(base_df.columns) == set(resumed_df.columns)

    columns_to_check = [
        # Identity
        "evaluation_id", "result_idx", "model_id", "benchmark_id", "metric_id",
        # Score + scale
        "score", "metric_unit", "score_scale_anomaly",
        # Per-row signals (Stage E)
        "completeness_score", "has_reproducibility_gap", "is_agentic",
        # Group signals (Stage F)
        "comparability_group_id", "is_multi_source",
        "has_variant_divergence", "has_cross_party_divergence",
    ]
    for col in columns_to_check:
        assert col in base_df.columns, f"unexpected: {col} not in baseline"
        # Cast to object so .fillna() works uniformly across boolean / string /
        # numeric columns (pandas BooleanArray rejects string sentinels).
        base = base_df[col].astype(object).where(base_df[col].notna(), "__NULL__")
        resumed = resumed_df[col].astype(object).where(resumed_df[col].notna(), "__NULL__")
        assert base.tolist() == resumed.tolist(), f"divergence in column {col}"


def test_from_stage_with_no_cache_raises(tmp_path, monkeypatch):
    cache_root = tmp_path / "empty_cache"
    cache_root.mkdir()
    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    with pytest.raises(FileNotFoundError, match="no snapshot found under"):
        pipeline.run(
            Settings.from_env(),
            cache_root=str(cache_root),
            from_stage="E",
        )


def test_invalid_stage_letter_raises():
    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    with pytest.raises(ValueError, match="not a recognised stage"):
        pipeline.run(Settings.from_env(), from_stage="Z")


def test_from_after_to_raises():
    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    with pytest.raises(ValueError, match="comes after"):
        pipeline.run(Settings.from_env(), from_stage="F", to_stage="C")


def test_dir_form_snapshot_id_runs_to_completion(tmp_path, monkeypatch):
    """Operator copies the dir-form snapshot id from `warehouse/` and passes it
    via --snapshot-id; the run normalises it to ISO so Stage F's TIMESTAMP
    literal accepts it.
    """
    out = _run_full(
        tmp_path, monkeypatch,
        snapshot_id="2026-04-30T00-00-00Z",  # dir-form (with hyphens)
        cache_root=tmp_path / "cache",
    )
    assert out is not None
    assert out.name == "2026-04-30T00-00-00Z"


def test_no_cache_skips_writes(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    _run_full(
        tmp_path, monkeypatch,
        snapshot_id="2026-04-30T00:00:00Z",
        cache_root=cache_root,
        no_cache=True,
    )
    # Cache root should not have been created (no writes)
    assert not (cache_root / "2026-04-30T00-00-00Z").exists() or \
        not any((cache_root / "2026-04-30T00-00-00Z").glob("*.parquet"))

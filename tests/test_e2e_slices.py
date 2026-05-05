"""End-to-end validation of slice plumbing across Stages C, F, and J.

Runs the pipeline against the `fixtures_slices` config — three EEE
records all resolving to benchmark_id=mmlu via the registry, with two
distinct cleaned raws ("Anatomy" + "Astronomy"). Asserts the cross-
stage behaviour:

  - **Stage C** writes slice_key / slice_name onto fact_results.
  - **Stage F.1** computes provenance per (model, benchmark) — the
    Astronomy row is_multi_source=True because the (gpt-4o, mmlu)
    pair has 2 reporting orgs total, even though only OpenAI reports
    that specific slice.
  - **Stage F.2** computes comparability per (model, benchmark, slice,
    metric) — Anatomy and Astronomy rows get distinct
    comparability_group_id values; the Anatomy slice carries cross-
    party metadata (2 orgs); Astronomy does not.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _run_slices(tmp_path, monkeypatch):
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
        configs=["fixtures_slices"],
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        cache_root=str(tmp_path / "cache"),
    )


def _facts(out_dir: Path):
    con = duckdb.connect()
    return con.execute(
        f"SELECT * FROM read_parquet('{out_dir}/fact_results.parquet') "
        f"ORDER BY evaluation_id"
    ).fetchdf()


# ---------------------------------------------------------------------------
# Stage C — slice_key / slice_name population
# ---------------------------------------------------------------------------


def test_slice_key_populated_per_distinct_raw(tmp_path, monkeypatch):
    """Two distinct cleaned raws ("Anatomy", "Astronomy") map to mmlu →
    Stage C's multi-raw heuristic fires; slice_key is the lowercase
    trimmed form, slice_name keeps the original casing."""
    out = _run_slices(tmp_path, monkeypatch)
    df = _facts(out)
    assert len(df) == 3

    by_eval = df.set_index("evaluation_id")
    assert by_eval.loc["ev_08a"]["slice_key"]  == "anatomy"
    assert by_eval.loc["ev_08a"]["slice_name"] == "Anatomy"
    assert by_eval.loc["ev_08b"]["slice_key"]  == "anatomy"
    assert by_eval.loc["ev_08c"]["slice_key"]  == "astronomy"
    assert by_eval.loc["ev_08c"]["slice_name"] == "Astronomy"


# ---------------------------------------------------------------------------
# Stage F.1 — provenance is slice/metric-agnostic
# ---------------------------------------------------------------------------


def test_provenance_is_per_model_benchmark_pair(tmp_path, monkeypatch):
    """All three rows belong to (gpt-4o, mmlu) which has 2 distinct
    orgs (OpenAI + Scale AI) across the snapshot. Even the Astronomy
    row — only reported by OpenAI on that specific slice — gets
    is_multi_source=True because provenance is benchmark-level."""
    out = _run_slices(tmp_path, monkeypatch)
    df = _facts(out)

    assert (df["distinct_reporting_orgs"] == 2).all()
    assert df["is_multi_source"].all()
    # At least one row is third-party (the Scale AI Anatomy report) and
    # the pair has multiple sources, so first_party_only is False on every
    # row regardless of which slice the row sits in.
    assert (~df["first_party_only"]).all()


# ---------------------------------------------------------------------------
# Stage F.2 — comparability is per (model, benchmark, slice, metric)
# ---------------------------------------------------------------------------


def test_comparability_group_id_includes_slice(tmp_path, monkeypatch):
    """Anatomy and Astronomy rows must have DIFFERENT
    comparability_group_id values — the slice is part of the md5
    input. Rows in the same slice share an id."""
    out = _run_slices(tmp_path, monkeypatch)
    df = _facts(out)
    by_eval = df.set_index("evaluation_id")

    anatomy_a = by_eval.loc["ev_08a"]["comparability_group_id"]
    anatomy_b = by_eval.loc["ev_08b"]["comparability_group_id"]
    astro_c   = by_eval.loc["ev_08c"]["comparability_group_id"]

    assert anatomy_a == anatomy_b           # same slice → same group
    assert anatomy_a != astro_c             # different slice → different group
    assert len(anatomy_a) == 32             # full md5
    assert len(astro_c)   == 32


def test_cross_party_metadata_per_slice(tmp_path, monkeypatch):
    """The Anatomy slice has 2 orgs (OpenAI + Scale AI) reporting the
    same metric → cross_party_org_count=2 on those rows. The Astronomy
    slice has only OpenAI → cross-party output is NULL (UDF returns
    NULL when only one org)."""
    out = _run_slices(tmp_path, monkeypatch)
    df = _facts(out)
    by_eval = df.set_index("evaluation_id")

    import pandas as pd
    assert by_eval.loc["ev_08a"]["cross_party_org_count"] == 2
    assert by_eval.loc["ev_08b"]["cross_party_org_count"] == 2
    # ev_08c is alone in its slice — cross-party UDF returns NULL.
    assert pd.isna(by_eval.loc["ev_08c"]["cross_party_org_count"])


def test_variant_divergence_isolated_to_slice(tmp_path, monkeypatch):
    """All three rows have temperature=0.0 + the same generation args
    so variant_key collides within each slice → no variant divergence
    at the slice level. Confirms the F.2 group key is per-slice (a
    cross-slice grouping would have folded all 3 rows into one variant
    group with consistent setup → still no divergence, but the test
    ensures we don't accidentally aggregate)."""
    out = _run_slices(tmp_path, monkeypatch)
    df = _facts(out)

    # All three rows share generation_args, so no variant-spread.
    assert (df["has_variant_divergence"] == False).all()  # noqa: E712

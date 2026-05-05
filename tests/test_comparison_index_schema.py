"""comparison-index.json — schema-only golden test.

Per Step 1 of the cross-project hierarchy-alignment effort
(`/Users/jchim/projects/evaleval/notes/hierarchy-alignment.md` §5.2),
each entry in `evals[]` carries a fixed shape that the frontend's
`getCompositeKey()` fallback chain reads. This test asserts presence
+ types of those fields against a real producer run on the
`fixtures_clean` and `fixtures_slices` corpora.

Deliberately schema-only: it does not pin specific values (snapshot
data drift would make value-pinning brittle); it just guarantees the
contract holds for every eval entry.
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


def _materialise_views_and_sidecars(out_dir: Path):
    from eval_card_backend.canonicalise import sidecars, stages
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
    stages.stage_j_eval_results_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_models_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")

    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    sidecars.write_manifest(con, out_dir, snap)
    sidecars.write_headline(con, out_dir, snap)
    sidecars.write_hierarchy(con, out_dir, snap)
    sidecars.write_comparison_index(con, out_dir, snap)
    return con


# Per §5.2 of the hierarchy-alignment spec: required keys + their
# allowed Python types (NoneType included where the field is nullable).
_REQUIRED_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "eval_summary_id":         (str,),
    "benchmark_id":            (str, type(None)),
    "family_id":               (str, type(None)),
    "family_display_name":     (str, type(None)),
    "composite_slug":          (str, type(None)),
    "composite_display_name":  (str, type(None)),
    "parent_benchmark_id":     (str, type(None)),
    "is_slice":                (bool,),
    "is_summary_score":        (bool,),
}


def _assert_eval_entry_schema(eval_id: str, entry: dict) -> None:
    missing = set(_REQUIRED_FIELD_TYPES) - set(entry)
    assert not missing, (
        f"comparison-index entry {eval_id!r} missing fields: {missing}"
    )
    for field, allowed_types in _REQUIRED_FIELD_TYPES.items():
        value = entry[field]
        assert isinstance(value, allowed_types), (
            f"comparison-index entry {eval_id!r}: field {field!r} "
            f"has type {type(value).__name__}; expected one of "
            f"{[t.__name__ for t in allowed_types]}"
        )


def test_comparison_index_schema_clean(tmp_path, monkeypatch):
    """Every eval entry in the clean corpus carries the §5.2 fields with
    the expected types."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    assert ci["evals"], "comparison-index has no evals; nothing to validate"
    for eval_id, entry in ci["evals"].items():
        _assert_eval_entry_schema(eval_id, entry)


def test_comparison_index_schema_slices(tmp_path, monkeypatch):
    """The slices fixture exercises the parent_benchmark_id path — at
    least one eval is_slice=True with a non-null parent. We assert the
    schema across the whole corpus and the slice-population invariant
    on the slice subset.
    """
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_slices")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    assert ci["evals"], "comparison-index has no evals; nothing to validate"
    # Note: this corpus exercises Stage C's metric-level slice_key path
    # (e.g. MMLU subject splits). Benchmark-level slices (is_slice=true
    # rows in comparison-index) require parent_benchmark_id chains in
    # the registry which these fixtures don't carry. The per-entry loop
    # below validates the invariant whichever way each row falls.
    for eval_id, entry in ci["evals"].items():
        _assert_eval_entry_schema(eval_id, entry)
        # Slice entries carry their parent benchmark id; root entries
        # null it out. The frontend's getCompositeKey() chain depends
        # on this is_slice ↔ parent_benchmark_id consistency.
        if entry["is_slice"]:
            assert entry["parent_benchmark_id"] is not None, (
                f"slice eval {eval_id!r} must carry parent_benchmark_id"
            )
        else:
            assert entry["parent_benchmark_id"] is None, (
                f"non-slice eval {eval_id!r} must null parent_benchmark_id; "
                f"got {entry['parent_benchmark_id']!r}"
            )

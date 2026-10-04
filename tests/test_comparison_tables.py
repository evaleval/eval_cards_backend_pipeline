"""comparison-index.json as parquet tables: the round trip is exact."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from eval_card_backend.canonicalise import comparison_tables as ct
from eval_card_backend.canonicalise.cache import normalize_snapshot_id
from eval_card_backend.canonicalise.stages import snapshot_id_to_sql
from tests.test_stage_j_sidecars import (
    _comparison_index_with_merged,
    _materialise_views_and_sidecars,
    _run_through_stage_i,
)

SNAPSHOT_ID = "2026-04-30T00:00:00Z"


def _cell(route, score, rank, total, **extra):
    return {
        "model_route_id": route,
        "model_family_id": route.split("%2F")[0] + "%2Ffam",
        "model_family_name": "Fam",
        "developer": "",
        "variant_key": "default",
        "score": score,
        "rank": rank,
        "total": total,
        "submission_count": 1,
        "submission_axis": "default",
        "temperature": None,
        "max_tokens": None,
        **extra,
    }


def _per_source(route, score, rank, total, *, split=None, **extra):
    return _cell(
        route, score, rank, total,
        score_canonical=score / 100, scale_conversion="div100", split=split,
        **extra,
    )


def _metric(msid, name, group, group_order, scores, *, lo=0.0, hi=1.0, unit=None):
    return {
        "metric_summary_id": msid,
        "metric_name": name,
        "metric_id": name.lower(),
        "metric_key": name.lower(),
        "group": group,
        "group_order": group_order,
        "lower_is_better": group == "cost",
        "unit": unit,
        "canonical_min_score": lo,
        "canonical_max_score": hi,
        "scores": scores,
    }


def _eval(eval_id, metrics, *, merged=False, **fields):
    entry = {
        "evaluation_id": eval_id,
        "composite_slug": None if merged else eval_id.split("%2F")[0],
        "composite_display_name": None if merged else "Source",
        "benchmark_id": eval_id.split("%2F")[-1],
        "family_id": None,
        "family_display_name": None,
        "parent_benchmark_id": None,
        "is_slice": False,
        "display_name": "Bench",
        "derived_tags": [] if merged else ["reasoning", "math"],
        "is_summary_score": False,
        "summary_score_for": None,
        "metrics": metrics,
        **fields,
    }
    if merged:
        entry["is_merged"] = True
    return entry


def _by_model(evals: dict) -> dict:
    out: dict = {}
    for eval_id, entry in evals.items():
        for metric in entry["metrics"]:
            for cell in metric["scores"]:
                keep = [k for k in cell if k not in {
                    "model_route_id", "model_family_id", "model_family_name",
                    "developer", "variant_key", "split", "source_composite_slug",
                }]
                out.setdefault(cell["model_route_id"], {}).setdefault(eval_id, {})[
                    metric["metric_summary_id"]
                ] = {k: cell[k] for k in keep}
    return out


def _index() -> dict:
    a, b, c = "org%2Fa", "org%2Fb", "org%2Fc"
    evals = {
        "src%2Fbench": _eval("src%2Fbench", [
            _metric("bench%3Aacc", "Acc", "capability", 0, [
                _per_source(b, 91.0, 1, 3, split="test", temperature=0.0, max_tokens=1024),
                _per_source(a, 80.5, 2, 3),
                _per_source(c, 80.5, 2, 3, submission_count=2, submission_axis="protocol"),
            ], lo=0.0, hi="Infinity", unit="percent"),
            _metric("bench%3Acost", "Cost", "cost", 3, [
                _per_source(a, 0.123456789012345, 1, 1),
            ], lo="-Infinity", hi=None),
            _metric("bench%3Anone", "Zero", "other", 6, [], lo=None, hi=None),
        ], parent_benchmark_id="parent", is_slice=True, family_id="fam",
           family_display_name="Family"),
        "bench": _eval("bench", [
            _metric("bench%3Aacc", "Acc", "capability", 0, [
                _cell(b, 0.91, 1, 2, source_composite_slug="src"),
                _cell(a, 0.0, 2, 2, source_composite_slug="other"),
            ]),
        ], merged=True),
    }
    return {
        "by_model": _by_model(evals),
        "comparison_index_version": 2,
        "config_version": 2,
        "evals": evals,
        "generated_at": SNAPSHOT_ID,
        "metric_group_order": ["capability", "robustness", "efficiency", "cost",
                               "latency", "rank", "other"],
    }


def _round_trip(index: dict, tmp_path) -> dict:
    ct.write_tables(index, tmp_path, SNAPSHOT_ID)
    return ct.unflatten(ct.read_tables(tmp_path))


def _counts(index: dict) -> tuple[int, int, int]:
    metrics = [m for e in index["evals"].values() for m in e["metrics"]]
    return len(index["evals"]), len(metrics), sum(len(m["scores"]) for m in metrics)


def _assert_exact(index: dict, tmp_path) -> None:
    rebuilt = _round_trip(index, tmp_path)
    assert rebuilt == index
    # Equality alone treats 1 == 1.0 and ignores key spelling; the
    # serialised form does not.
    assert json.dumps(rebuilt, indent=2, sort_keys=True) == json.dumps(
        index, indent=2, sort_keys=True
    )
    rows = tuple(
        pq.read_metadata(tmp_path / name).num_rows for name in ct.TABLE_FILES
    )
    assert rows == _counts(index)


def test_round_trip_purpose_built(tmp_path):
    index = _index()
    _assert_exact(index, tmp_path)
    rebuilt = _round_trip(index, tmp_path)
    merged_cells = rebuilt["evals"]["bench"]["metrics"][0]["scores"]
    assert all("split" not in c and "score_canonical" not in c for c in merged_cells)
    per_source = rebuilt["evals"]["src%2Fbench"]
    assert "is_merged" not in per_source
    assert per_source["metrics"][0]["scores"][1]["split"] is None
    assert "split" in per_source["metrics"][0]["scores"][1]
    assert per_source["metrics"][0]["canonical_max_score"] == "Infinity"
    assert per_source["metrics"][1]["canonical_min_score"] == "-Infinity"


def test_tables_store_real_infinities_and_sort(tmp_path):
    ct.write_tables(_index(), tmp_path, SNAPSHOT_ID)
    metrics = pq.read_table(tmp_path / "comparison_metrics.parquet").to_pylist()
    assert metrics[1]["canonical_max_score"] == float("inf")
    assert metrics[2]["canonical_min_score"] == float("-inf")
    scores = pq.read_table(tmp_path / "comparison_scores.parquet").to_pylist()
    assert [(s["evaluation_id"], s["metric_summary_id"], s["row_ord"]) for s in scores] == [
        ("bench", "bench%3Aacc", 0), ("bench", "bench%3Aacc", 1),
        ("src%2Fbench", "bench%3Aacc", 0), ("src%2Fbench", "bench%3Aacc", 1),
        ("src%2Fbench", "bench%3Aacc", 2), ("src%2Fbench", "bench%3Acost", 0),
    ]
    evals = pq.read_table(tmp_path / "comparison_evals.parquet").to_pylist()
    assert [e["is_merged"] for e in evals] == [True, False]


@pytest.mark.parametrize("mutate", [
    lambda ix: ix["evals"]["bench"]["metrics"][0]["scores"][0].update(split=None),
    lambda ix: ix["evals"]["src%2Fbench"]["metrics"][0]["scores"][0].pop("split"),
    lambda ix: ix["evals"]["src%2Fbench"]["metrics"][0]["scores"][0].update(
        source_composite_slug="src"),
    lambda ix: ix["evals"]["src%2Fbench"].update(is_merged=False),
    lambda ix: ix["evals"]["src%2Fbench"]["metrics"][0].pop("unit"),
    lambda ix: ix.pop("metric_group_order"),
], ids=["merged-with-split", "per-source-without-split", "per-source-with-slug",
        "is-merged-false", "metric-missing-key", "top-level-missing-key"])
def test_key_set_mismatch_raises(mutate):
    index = _index()
    mutate(index)
    with pytest.raises(ValueError):
        ct.flatten(index, SNAPSHOT_ID)


@pytest.mark.parametrize("mutate", [
    lambda ix: ix["evals"]["bench"]["metrics"][0]["scores"][0].update(score=1),
    lambda ix: ix["evals"]["bench"]["metrics"][0]["scores"][0].update(rank=1.0),
    lambda ix: ix["by_model"]["org%2Fa"]["bench"]["bench%3Aacc"].update(rank=9),
], ids=["int-in-double", "float-in-integer", "by-model-not-a-repivot"])
def test_values_that_would_not_read_back_raise(mutate):
    index = _index()
    mutate(index)
    with pytest.raises(ValueError):
        ct.flatten(index, SNAPSHOT_ID)


@pytest.mark.parametrize("config", ["fixtures_clean", "fixtures_slices", "fixtures_splits"])
def test_round_trip_on_sidecar_fixtures(tmp_path, monkeypatch, config):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, config)
    _materialise_views_and_sidecars(out)
    index = json.loads((out / "comparison-index.json").read_text())
    assert index["evals"]
    assert ct.unflatten(ct.read_tables(out)) == index
    _assert_exact(index, tmp_path)


def test_round_trip_with_merged_entries(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_splits")
    index = _comparison_index_with_merged(out)
    assert any(e.get("is_merged") for e in index["evals"].values())
    assert ct.unflatten(ct.read_tables(out)) == index
    _assert_exact(index, tmp_path)


def test_round_trip_with_no_evals(tmp_path):
    index = {
        "by_model": {},
        "comparison_index_version": 2,
        "config_version": 2,
        "evals": {},
        "generated_at": SNAPSHOT_ID,
        "metric_group_order": ["capability", "other"],
    }
    _assert_exact(index, tmp_path)


@pytest.mark.parametrize("snapshot_id", [
    "2026-04-30T00:00:00Z", "2026-04-30T00:00:00+01:00", "2026-04-30T23-30-00-05:00",
])
def test_snapshot_id_matches_duckdb_written_tables(snapshot_id):
    sid = normalize_snapshot_id(snapshot_id)
    expected = duckdb.sql(f"SELECT TIMESTAMP '{snapshot_id_to_sql(sid)}'").fetchone()[0]
    tables = ct.flatten(_index(), sid)
    for table in tables.values():
        assert set(table.column("snapshot_id").to_pylist()) == {expected}


def _backfill_module():
    path = Path(__file__).parents[1] / "scripts" / "backfill_comparison_tables.py"
    spec = importlib.util.spec_from_file_location("backfill_comparison_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backfill_registers_tables_and_changes_nothing_else(tmp_path):
    (tmp_path / "comparison-index.json").write_text(
        json.dumps(_index(), indent=2, sort_keys=True)
    )
    meta = {
        "snapshot_id": SNAPSHOT_ID,
        "generated_at": "2026-04-30T12:00:00Z",
        "configs": ["b", "a"],
        "tables": ["fact_results.parquet", "merged_evals_view.parquet"],
        "row_counts": {"fact_results": 2, "ratio": 0.5},
        "sidecars": ["manifest.json"],
    }
    meta_path = tmp_path / "snapshot_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    backfill = _backfill_module()

    assert backfill.main([str(tmp_path)]) == 0
    expected = dict(meta, tables=meta["tables"] + list(ct.TABLE_FILES))
    assert meta_path.read_text() == json.dumps(expected, indent=2)
    assert backfill.main([str(tmp_path)]) == 0
    assert meta_path.read_text() == json.dumps(expected, indent=2)


def test_backfill_without_snapshot_meta(tmp_path, capsys):
    (tmp_path / "comparison-index.json").write_text(
        json.dumps(_index(), indent=2, sort_keys=True)
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"generated_at": SNAPSHOT_ID}))
    assert _backfill_module().main([str(tmp_path)]) == 0
    assert "tables not registered" in capsys.readouterr().out
    assert not (tmp_path / "snapshot_meta.json").exists()
    assert all((tmp_path / t).exists() for t in ct.TABLE_FILES)

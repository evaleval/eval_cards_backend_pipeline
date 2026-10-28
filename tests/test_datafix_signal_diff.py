"""Guards for the snapshot signal-diff script: its helpers, its abort paths,
and the invariants, driven over hand-built snapshot directories.

The synthetic snapshots carry two fact rows, one from a source the overlay
targets and one from another source that shares a model with it, plus the view
and dimension tables built over them. That is enough to make each invariant
fire: the columns are the ones the script names, with the shapes it reads them
at, and nothing else.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval_card_backend.datafix.table_io import write_json_bytes


_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "datafix_signal_diff.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("datafix_signal_diff", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before exec so the dataclass decorators can resolve the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sd = _load_module()


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_json_reader_accepts_plain_and_compressed_files(tmp_path, suffix):
    path = tmp_path / f"metadata{suffix}"
    payload = {"datafix": {"enabled": True}}
    write_json_bytes(path, json.dumps(payload).encode())
    assert sd._load_json(path) == payload


def _write_meta(directory: Path, eee_sha: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "snapshot_meta.json").write_text(
        json.dumps(
            {
                "snapshot_id": "2026-01-01T00:00:00Z",
                "eee_revision": eee_sha,
                "registry_revision": "reg",
                "cards_revision": "cards",
                "upstream_pins": {
                    "eee_datastore": {"repo_id": "evaleval/EEE_datastore", "sha": eee_sha},
                    "entity_registry": {"repo_id": "evaleval/entity-registry-data", "sha": "reg"},
                    "benchmark_metadata": {"repo_id": "evaleval/auto-benchmarkcards", "sha": "cards"},
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_pin_mismatch_aborts(tmp_path, capsys):
    before = _write_meta(tmp_path / "before", "aaa")
    after = _write_meta(tmp_path / "after", "bbb")

    assert sd.main([str(before), str(after)]) == 2
    err = capsys.readouterr().err
    assert "pin mismatch on eee_datastore" in err
    assert "Refusing to diff" in err


def test_missing_snapshot_meta_aborts(tmp_path, capsys):
    before = _write_meta(tmp_path / "before", "aaa")
    after = tmp_path / "after"
    after.mkdir()

    assert sd.main([str(before), str(after)]) == 2
    assert "not a snapshot directory" in capsys.readouterr().err


def test_pins_reads_both_shapes():
    pins = sd._pins(
        {
            "eee_revision": "a",
            "registry_revision": "b",
            "cards_revision": "c",
            "upstream_pins": {"eee_datastore": {"sha": "a"}, "entity_registry": {"sha": "b"}},
        }
    )
    assert pins == {
        "eee_datastore": "a",
        "entity_registry": "b",
        "eee_revision": "a",
        "registry_revision": "b",
        "cards_revision": "c",
    }


def test_flatten_walks_dicts_and_lists():
    flat = sd._flatten({"a": {"b": 1}, "c": [{"d": 2}, 3]})
    assert flat == {"a.b": 1, "c[0].d": 2, "c[1]": 3}


@pytest.mark.parametrize(
    "before, after, expected",
    [(1, 3, "+2"), (3, 1, "-2"), (None, 4, "changed"), ("x", "x", "0"), (1.0, 1.5, "+0.5")],
)
def test_delta_formats(before, after, expected):
    assert sd._delta(before, after) == expected


def test_fmt_moved_is_singular_for_one_row():
    assert sd._fmt_moved([("temperature", 1), ("max_tokens", 4)]) == (
        "temperature (1 row), max_tokens (4 rows)"
    )


def test_md_table_escapes_pipes():
    table = sd._md_table(["a"], [["x|y"]])
    assert table[2] == "| x\\|y |"


def test_scope_predicate_covers_every_table_spec():
    for spec in sd.TABLES:
        predicate = sd._scope_predicate(spec, "b")
        assert predicate is None or spec.scope[1] in predicate


# ---------------------------------------------------------------------------
# Synthetic snapshots
# ---------------------------------------------------------------------------

_REPRO_SUMMARY = pa.struct([
    ("results_total", pa.int32()),
    ("has_reproducibility_gap_count", pa.int32()),
    ("populated_ratio_avg", pa.float64()),
])
_COMPARABILITY_SUMMARY = pa.struct([
    ("variant_divergent_count", pa.int32()),
    ("cross_party_divergent_count", pa.int32()),
    ("groups_with_variant_check", pa.int32()),
])
_SCORE_SUMMARY = pa.struct([
    ("min", pa.float64()), ("max", pa.float64()),
    ("average", pa.float64()), ("count", pa.int32()),
])
_GENERATION_CONFIG = pa.struct([
    ("generation_args", pa.struct([
        ("temperature", pa.float64()), ("top_p", pa.float64()),
        ("top_k", pa.int32()), ("max_tokens", pa.int32()),
        ("reasoning", pa.string()),
    ])),
    ("num_few_shot", pa.int32()),
    ("prompt_template", pa.string()),
    ("additional_details", pa.string()),
])
_ANNOTATIONS = pa.struct([
    ("reproducibility_gap", pa.string()),
    ("provenance", pa.string()),
    ("comparability", pa.string()),
])
_DIFFERING_FIELDS = pa.list_(pa.struct([("field", pa.string())]))

# Column types the script reads at something other than a plain string.
_TYPES: dict[str, pa.DataType] = {
    "reproducibility_summary": _REPRO_SUMMARY,
    "comparability_summary": _COMPARABILITY_SUMMARY,
    "score_summary": _SCORE_SUMMARY,
    "generation_config": _GENERATION_CONFIG,
    "evalcards_annotations": _ANNOTATIONS,
    "variant_differing_fields": _DIFFERING_FIELDS,
    "cross_party_differing_fields": _DIFFERING_FIELDS,
    "repro_missing_fields": pa.list_(pa.string()),
    "temperature": pa.float64(),
    "top_p": pa.float64(),
    "top_k": pa.int32(),
    "max_tokens": pa.int32(),
    "n_samples": pa.int32(),
    "score": pa.float64(),
    "variant_divergence_magnitude": pa.float64(),
    "variant_divergence_threshold": pa.float64(),
    "cross_party_divergence_magnitude": pa.float64(),
    "cross_party_divergence_threshold": pa.float64(),
    "cross_party_org_count": pa.int32(),
    "repro_required_count": pa.int32(),
    "repro_populated_count": pa.int32(),
    "variant_count": pa.int32(),
    "missing_generation_config_count": pa.int32(),
    "num_few_shot": pa.int32(),
}
_BOOLEANS = frozenset({
    "is_part", "is_headline", "is_slice", "is_multi_source", "first_party_only",
    "is_summary_score", "is_verified_evaluator", "has_first_party",
    "has_third_party", "children_present", "score_scale_anomaly",
    "has_temperature", "has_top_p", "has_top_k", "has_prompt_template",
    "has_max_tokens", "has_reproducibility_gap", "has_variant_divergence",
    "has_cross_party_divergence",
})

# Setup fields the overlay does not write. They are on the fact table, so
# the report has to see them stay put.
_UNWRITTEN_INPUTS = (
    "top_p", "top_k", "prompt_template", "reasoning",
    "has_top_p", "has_top_k", "has_prompt_template",
)

_FACT_COLUMNS = tuple(dict.fromkeys(
    ("snapshot_id", "fact_id", "comparability_group_id", "evaluator_relationship")
    + sd.FACT_IDENTITY_COLUMNS
    + sd.FACT_SCORE_COLUMNS
    + sd.FACT_REPRESENTATIVE_ORDER_COLUMNS
    + sd.MOVABLE_COLUMNS["fact_results"]
    + _UNWRITTEN_INPUTS
    + ("is_headline",)
))
_ERV_COLUMNS = tuple(dict.fromkeys(
    ("snapshot_id", "evaluation_id")
    + sd.ERV_KEY
    + sd.ERV_IDENTITY_COLUMNS
    + sd.ERV_SCORE_COLUMNS
    + sd.ERV_REPRESENTATIVE_COLUMNS
    + sd.MOVABLE_COLUMNS["eval_results_view"]
    + sd.MEMBERWISE_COLUMNS["eval_results_view"]
    + ("is_headline", "metric_pair_key", "completeness_score")
))
_EVALS_COLUMNS = tuple(dict.fromkeys(
    ("snapshot_id", "evaluation_id", "composite_slug", "benchmark_id",
     "avg_score", "avg_score_norm", "top_score", "best_model", "worst_model")
    + sd.MOVABLE_COLUMNS["evals_view"]
))
_MODELS_VIEW_COLUMNS = tuple(dict.fromkeys(
    ("snapshot_id", "model_key", "developer", "score_summary", "top_scores")
    + sd.MOVABLE_COLUMNS["models_view"]
))
_SIMPLE_TABLES: dict[str, tuple[str, ...]] = {
    "merged_evals_view": ("snapshot_id", "evaluation_id", "benchmark_id"),
    "models": ("snapshot_id", "model_key", "model_name"),
    "benchmarks": ("snapshot_id", "composite_slug", "benchmark_id"),
    "composites": ("snapshot_id", "composite_slug"),
    "canonical_metrics": ("snapshot_id", "id"),
    "families": ("snapshot_id", "family_id"),
    "collection_trajectories": ("snapshot_id", "collection_id"),
}


def _field(name: str) -> pa.Field:
    if name in _TYPES:
        return pa.field(name, _TYPES[name])
    if name in _BOOLEANS:
        return pa.field(name, pa.bool_())
    return pa.field(name, pa.string())


def _write_table(directory: Path, name: str, columns: tuple[str, ...], rows: list[dict]):
    schema = pa.schema([_field(c) for c in columns])
    pq.write_table(
        pa.Table.from_pylist([{c: row.get(c) for c in columns} for row in rows], schema),
        directory / f"{name}.parquet",
    )


def _fact(fact_id: str, source_config: str, model: str, **overrides) -> dict:
    row = {
        "snapshot_id": "snap", "fact_id": fact_id, "evaluation_id": f"ev_{fact_id}",
        "source_config": source_config, "composite_slug": "comp",
        "model_aggregation_key": model, "model_key": model, "model_id": model,
        "benchmark_id": "bench", "benchmark_key": "bench",
        "metric_id": "acc", "metric_key": "acc", "metric_base_key": "acc",
        "metric_key_effective": "acc", "comparability_group_id": "grp",
        "evaluator_relationship": "third_party", "score": 0.5,
        "is_part": False, "is_headline": True, "score_scale_anomaly": False,
        "repro_missing_fields": ["temperature", "max_tokens"],
        "repro_required_count": 2, "repro_populated_count": 0,
        "has_reproducibility_gap": True,
        "has_temperature": False, "has_max_tokens": False,
        "has_top_p": False, "has_top_k": False, "has_prompt_template": False,
        "variant_key": "unset", "comparability_status": "single_variant",
        "has_variant_divergence": False, "has_cross_party_divergence": False,
        "variant_differing_fields": [], "cross_party_differing_fields": [],
    }
    row.update(overrides)
    return row


_FILLED = {
    "temperature": 0.0, "max_tokens": 1280,
    "has_temperature": True, "has_max_tokens": True,
    "repro_missing_fields": [], "repro_populated_count": 2,
    "has_reproducibility_gap": False, "variant_key": "t0-mt1280",
}


def _write_snapshot(directory: Path, *, filled: bool, datafix) -> Path:
    """One snapshot: a targeted fact, a fact from another source sharing its
    model, and the view and dimension tables built over them."""
    directory.mkdir(parents=True, exist_ok=True)
    target = _fact("f1", "vals-ai", "m1", **(_FILLED if filled else {}))
    other = _fact("f2", "other-source", "m1")
    _write_table(directory, "fact_results", _FACT_COLUMNS, [target, other])

    generation_config = {
        "generation_args": {
            "temperature": 0.0 if filled else None,
            "top_p": None, "top_k": None,
            "max_tokens": 1280 if filled else None,
            "reasoning": None,
        },
        "num_few_shot": None, "prompt_template": None, "additional_details": None,
    }
    _write_table(directory, "eval_results_view", _ERV_COLUMNS, [{
        "snapshot_id": "snap", "evaluation_id": "ev_f1", "composite_slug": "comp",
        "model_key": "m1", "benchmark_id": "bench", "metric_id": "acc",
        "protocol_condition": None, "judge_condition": None, "split": None,
        "generation_config": generation_config,
        "has_reproducibility_gap": not filled,
        "comparability_status": "single_variant",
        "has_variant_divergence": False, "has_cross_party_divergence": False,
        "evalcards_annotations": {
            "reproducibility_gap": "stable",
            "provenance": "stable",
            "comparability": "filled" if filled else "gap",
        },
        "is_headline": True,
    }])
    _write_table(directory, "evals_view", _EVALS_COLUMNS, [{
        "snapshot_id": "snap", "evaluation_id": "ev_f1", "composite_slug": "comp",
        "benchmark_id": "bench",
        "missing_generation_config_count": 0 if filled else 1,
        "reproducibility_summary": {
            "results_total": 1,
            "has_reproducibility_gap_count": 0 if filled else 1,
            "populated_ratio_avg": 1.0 if filled else 0.0,
        },
        "comparability_summary": {
            "variant_divergent_count": 0, "cross_party_divergent_count": 0,
            "groups_with_variant_check": 1,
        },
    }])
    _write_table(directory, "models_view", _MODELS_VIEW_COLUMNS, [{
        "snapshot_id": "snap", "model_key": "m1", "developer": "org",
        "variant_count": 1,
        "missing_generation_config_count": 0 if filled else 1,
        "reproducibility_status": "documented" if filled else "gap",
        "reproducibility_summary": {
            "results_total": 2,
            "has_reproducibility_gap_count": 1 if filled else 2,
            "populated_ratio_avg": 0.5 if filled else 0.0,
        },
        "comparability_summary": {
            "variant_divergent_count": 0, "cross_party_divergent_count": 0,
            "groups_with_variant_check": 1,
        },
        "score_summary": {"min": 0.5, "max": 0.5, "average": 0.5, "count": 2},
        "top_scores": "[]",
    }, {
        "snapshot_id": "snap", "model_key": "m2", "developer": "other-org",
        "variant_count": 1, "missing_generation_config_count": 0,
        "reproducibility_status": "documented",
        "reproducibility_summary": {
            "results_total": 0, "has_reproducibility_gap_count": 0,
            "populated_ratio_avg": 0.0,
        },
        "comparability_summary": {
            "variant_divergent_count": 0, "cross_party_divergent_count": 0,
            "groups_with_variant_check": 0,
        },
        "score_summary": {"min": 0.0, "max": 0.0, "average": 0.0, "count": 0},
        "top_scores": "[]",
    }])
    for name, columns in _SIMPLE_TABLES.items():
        _write_table(directory, name, columns, [
            {c: {"snapshot_id": "snap"}.get(c, f"{name}-1") for c in columns}
        ])

    (directory / "headline.json").write_text(
        json.dumps({
            "generated_at": "2026-01-01T00:00:00Z",
            "reproducibility": {"overall": {"gap": 0 if filled else 1}, "by_tag": {}},
            "comparability": {"overall": {"groups": 1}},
            "coverage": {"models": 2},
        }),
        encoding="utf-8",
    )
    meta = {
        "snapshot_id": "2026-01-01T00:00:00Z",
        "eee_revision": "aaa", "registry_revision": "reg", "cards_revision": "cards",
        "upstream_pins": {"eee_datastore": {"sha": "aaa"}},
    }
    if datafix is not None:
        meta["datafix"] = datafix
    (directory / "snapshot_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


OVERLAY_ON = {
    "enabled": True,
    "overlay_revision": 1,
    "fingerprint": "f" * 64,
    "fixes": {"a_fix": {"records_patched": 1, "results_patched": 1,
                        "by_source_config": {"vals-ai": 1}}},
}

# The same claim from a snapshot whose block predates per-config counts.
OVERLAY_ON_UNATTRIBUTED = {
    **OVERLAY_ON,
    "fixes": {"a_fix": {"records_patched": 1, "results_patched": 1}},
}

# The same overlay one step along: the two fixes it already had, plus one.
OVERLAY_BEFORE = {
    "enabled": True,
    "overlay_revision": 1,
    "fingerprint": "a" * 64,
    "fixes": {"a_fix": {"records_patched": 1, "results_patched": 1,
                        "by_source_config": {"vals-ai": 1}}},
}
OVERLAY_AFTER = {
    "enabled": True,
    "overlay_revision": 2,
    "fingerprint": "b" * 64,
    "fixes": {
        "a_fix": {"records_patched": 1, "results_patched": 1,
                  "by_source_config": {"vals-ai": 1}},
        "b_fix": {"records_patched": 1, "results_patched": 1,
                  "by_source_config": {"vals-ai": 1}},
    },
}


@pytest.fixture
def pair(tmp_path) -> tuple[Path, Path]:
    before = _write_snapshot(tmp_path / "before", filled=False, datafix=None)
    after = _write_snapshot(tmp_path / "after", filled=True, datafix=OVERLAY_ON)
    return before, after


def _run(pair, tmp_path, *extra: str) -> tuple[int, str]:
    out = tmp_path / "report.md"
    code = sd.main([str(pair[0]), str(pair[1]), "--out", str(out), *extra])
    return code, out.read_text(encoding="utf-8")


def test_an_overlay_on_off_pair_passes_every_invariant(pair, tmp_path):
    code, report = _run(pair, tmp_path)

    assert "[FAIL]" not in report
    assert code == 0


def test_the_expected_fingerprint_is_asserted(pair, tmp_path):
    assert _run(pair, tmp_path, "--expect-fingerprint", "f" * 64)[0] == 0

    code, report = _run(pair, tmp_path, "--expect-fingerprint", "0" * 64)
    assert code == 1
    assert "[FAIL] after snapshot overlay is the expected one" in report


def test_an_after_run_with_the_overlay_off_fails(tmp_path):
    before = _write_snapshot(tmp_path / "before", filled=False, datafix=None)
    after = _write_snapshot(
        tmp_path / "after", filled=True, datafix={**OVERLAY_ON, "enabled": False}
    )

    code, report = _run((before, after), tmp_path)

    assert code == 1
    assert "[FAIL] after snapshot built with the overlay" in report


def test_an_after_run_that_filled_nothing_fails(tmp_path):
    before = _write_snapshot(tmp_path / "before", filled=False, datafix=None)
    after = _write_snapshot(tmp_path / "after", filled=True, datafix={
        **OVERLAY_ON,
        "fixes": {"a_fix": {"records_patched": 0, "results_patched": 0}},
    })

    code, report = _run((before, after), tmp_path)

    assert code == 1
    assert "[FAIL] after snapshot built with the overlay" in report


def test_a_row_swapped_under_a_new_key_fails(pair, tmp_path):
    """Row counts still match, so only the per-key match catches it."""
    before, after = pair
    rows = pq.read_table(after / "models_view.parquet").to_pylist()
    rows[1]["model_key"] = "m3"
    _write_table(after, "models_view", _MODELS_VIEW_COLUMNS, rows)

    code, report = _run(pair, tmp_path)

    assert code == 1
    assert "models_view 1/1 unmatched" in report


def test_a_missing_table_fails(pair, tmp_path):
    (pair[1] / "composites.parquet").unlink()

    code, report = _run(pair, tmp_path)

    assert code == 1
    assert "[FAIL] every compared table present on both sides" in report
    assert "composites (after)" in report


def test_a_duplicate_key_fails(pair, tmp_path):
    before, after = pair
    for directory in (before, after):
        rows = pq.read_table(directory / "models_view.parquet").to_pylist()
        _write_table(directory, "models_view", _MODELS_VIEW_COLUMNS, rows + rows[:1])

    code, report = _run(pair, tmp_path)

    assert code == 1
    assert "[FAIL] keys unique and non-null on both sides" in report
    assert "duplicate key" in report


def test_an_unrelated_column_moving_on_an_in_scope_row_fails(pair, tmp_path):
    """`m1` is in scope through the targeted fact, so the scoped invariants
    let its row move; only the allowlist says the column is not ours."""
    before, after = pair
    rows = pq.read_table(after / "models_view.parquet").to_pylist()
    rows[0]["developer"] = "someone-else"
    _write_table(after, "models_view", _MODELS_VIEW_COLUMNS, rows)

    code, report = _run(pair, tmp_path)

    assert code == 1
    assert "[FAIL] only signal columns move: models_view.developer (1 row)" in report
    assert "[PASS] models_view: zero movement outside the target sources" in report


# ---------------------------------------------------------------------------
# Comparing two overlay-enabled snapshots
# ---------------------------------------------------------------------------


@pytest.fixture
def overlay_pair(tmp_path) -> tuple[Path, Path]:
    before = _write_snapshot(tmp_path / "before", filled=False, datafix=OVERLAY_BEFORE)
    after = _write_snapshot(tmp_path / "after", filled=True, datafix=OVERLAY_AFTER)
    return before, after


def _run_overlay(pair, tmp_path, *extra: str) -> tuple[int, str]:
    return _run(pair, tmp_path, "--before-overlay-fingerprint", "a" * 64, *extra)


def test_an_overlay_to_overlay_pair_passes_every_invariant(overlay_pair, tmp_path):
    code, report = _run_overlay(overlay_pair, tmp_path)

    assert "[FAIL]" not in report
    assert code == 0
    assert "Fixes the after side adds: b_fix." in report


def test_a_before_side_that_is_not_the_named_overlay_fails(overlay_pair, tmp_path):
    code, report = _run(overlay_pair, tmp_path, "--before-overlay-fingerprint", "c" * 64)

    assert code == 1
    assert "[FAIL] before snapshot built with the known overlay" in report


def test_an_after_side_that_adds_no_fix_fails(tmp_path):
    before = _write_snapshot(tmp_path / "before", filled=False, datafix=OVERLAY_BEFORE)
    after = _write_snapshot(
        tmp_path / "after", filled=True,
        datafix={**OVERLAY_AFTER, "fixes": OVERLAY_BEFORE["fixes"]},
    )

    code, report = _run_overlay((before, after), tmp_path)

    assert code == 1
    assert "[FAIL] after snapshot built with a later overlay that added fixes" in report


def test_a_new_fix_that_filled_nothing_fails(tmp_path):
    before = _write_snapshot(tmp_path / "before", filled=False, datafix=OVERLAY_BEFORE)
    after = _write_snapshot(tmp_path / "after", filled=True, datafix={
        **OVERLAY_AFTER,
        "fixes": {
            **OVERLAY_BEFORE["fixes"],
            "b_fix": {"records_patched": 0, "results_patched": 0,
                      "by_source_config": {}},
        },
    })

    code, report = _run_overlay((before, after), tmp_path)

    assert code == 1
    assert "[FAIL] after snapshot built with a later overlay that added fixes" in report


def test_the_targets_come_from_what_the_new_fixes_report(overlay_pair, tmp_path):
    code, report = _run_overlay(overlay_pair, tmp_path)

    assert code == 0
    assert "Target source configs: vals-ai." in report


def test_a_new_fix_reporting_no_source_config_stops_the_run(tmp_path):
    before = _write_snapshot(tmp_path / "before", filled=False, datafix=OVERLAY_BEFORE)
    after = _write_snapshot(tmp_path / "after", filled=True, datafix={
        **OVERLAY_AFTER,
        "fixes": {
            **OVERLAY_BEFORE["fixes"],
            "b_fix": {"records_patched": 1, "results_patched": 1},
        },
    })

    code, report = _run_overlay((before, after), tmp_path)

    assert code == 1
    assert "[FAIL] target source configs known" in report


def test_a_claim_the_parquet_does_not_bear_out_fails(tmp_path):
    """The same snapshot twice, the second one's metadata saying a fix filled
    something. Nothing in the tables moved, so the claim is not true of them."""
    before = _write_snapshot(tmp_path / "before", filled=False, datafix=None)
    after = tmp_path / "after"
    shutil.copytree(before, after)
    (after / "snapshot_meta.json").write_text(
        json.dumps({
            **json.loads((before / "snapshot_meta.json").read_text()),
            "datafix": OVERLAY_ON,
        }),
        encoding="utf-8",
    )

    code, report = _run((before, after), tmp_path)

    assert code == 1
    assert "[FAIL] a_fix: the tables moved where it reports patching" in report
    assert "vals-ai: 0 fact row(s) filled, 1 claimed" in report


def test_a_claim_naming_no_source_config_is_held_to_the_whole_pair(tmp_path):
    before = _write_snapshot(tmp_path / "before", filled=False, datafix=None)
    after = tmp_path / "after"
    shutil.copytree(before, after)
    (after / "snapshot_meta.json").write_text(
        json.dumps({
            **json.loads((before / "snapshot_meta.json").read_text()),
            "datafix": OVERLAY_ON_UNATTRIBUTED,
        }),
        encoding="utf-8",
    )

    code, report = _run((before, after), tmp_path)

    assert code == 1
    assert "[FAIL] fixes naming no source config: the tables moved" in report


def test_a_pair_that_really_moved_satisfies_the_claim(pair, tmp_path):
    code, report = _run(pair, tmp_path)

    assert code == 0
    assert "[PASS] a_fix: the tables moved where it reports patching" in report
    assert "vals-ai: 1 fact row(s) filled, 1 claimed" in report


def test_a_claim_larger_than_the_movement_is_allowed(pair, tmp_path):
    """A fix counts raw results, the tables count fact rows, and fewer of
    those can move without anything being wrong."""
    before, after = pair
    meta = json.loads((after / "snapshot_meta.json").read_text())
    meta["datafix"] = {
        **OVERLAY_ON,
        "fixes": {"a_fix": {"records_patched": 1, "results_patched": 40,
                            "by_source_config": {"vals-ai": 40}}},
    }
    (after / "snapshot_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    code, report = _run(pair, tmp_path)

    assert code == 0
    assert "vals-ai: 1 fact row(s) filled, 40 claimed" in report


def test_more_movement_than_was_claimed_fails(pair, tmp_path):
    before, after = pair
    meta = json.loads((after / "snapshot_meta.json").read_text())
    meta["datafix"] = {
        **OVERLAY_ON,
        "fixes": {"a_fix": {"records_patched": 1, "results_patched": 1,
                            "by_source_config": {"vals-ai": 0}}},
    }
    (after / "snapshot_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    code, report = _run(pair, tmp_path)

    assert code == 1
    assert "[FAIL] a_fix: the tables moved where it reports patching" in report


# ---------------------------------------------------------------------------
# What may move, and on which rows
# ---------------------------------------------------------------------------


def test_an_unfilled_fact_in_a_target_source_moving_its_cap_fails(pair, tmp_path):
    """A published-run cap on a fact the overlay never filled: it was stated
    before and stated after, so nothing the overlay does explains the move."""
    before, after = pair
    for directory, cap in ((before, 512), (after, 2048)):
        rows = pq.read_table(directory / "fact_results.parquet").to_pylist()
        rows.append(_fact("f3", "arc-agi", "m1", max_tokens=cap, has_max_tokens=True))
        _write_table(directory, "fact_results", _FACT_COLUMNS, rows)

    code, report = _run(pair, tmp_path, "--target-source-config", "vals-ai",
                        "--target-source-config", "arc-agi")

    assert code == 1
    assert (
        "[FAIL] fact_results: direct generation inputs move only on facts the "
        "overlay filled: max_tokens (1 row)"
    ) in report


def test_a_setup_field_the_overlay_does_not_write_moving_fails(pair, tmp_path):
    """`top_p` moves on the very fact that was filled, and is still not a
    field the overlay writes."""
    before, after = pair
    rows = pq.read_table(after / "fact_results.parquet").to_pylist()
    rows[0]["top_p"] = 0.9
    rows[0]["has_top_p"] = True
    _write_table(after, "fact_results", _FACT_COLUMNS, rows)

    code, report = _run(pair, tmp_path)

    assert code == 1
    assert "[FAIL] only signal columns move" in report
    assert "fact_results.top_p (1 row)" in report


def test_a_rewritten_cap_fails(pair, tmp_path):
    before, after = pair
    for directory, cap in ((before, 512), (after, 2048)):
        rows = pq.read_table(directory / "fact_results.parquet").to_pylist()
        rows[1]["max_tokens"] = cap
        rows[1]["has_max_tokens"] = True
        _write_table(directory, "fact_results", _FACT_COLUMNS, rows)

    code, report = _run(pair, tmp_path)

    assert code == 1
    assert "[FAIL] stated generation inputs neither rewritten nor cleared" in report


def test_a_generation_payload_member_the_overlay_does_not_write_moving_fails(
    pair, tmp_path
):
    before, after = pair
    rows = pq.read_table(after / "eval_results_view.parquet").to_pylist()
    rows[0]["generation_config"]["generation_args"]["top_p"] = 0.9
    _write_table(after, "eval_results_view", _ERV_COLUMNS, rows)

    code, report = _run(pair, tmp_path)

    assert code == 1
    assert "[FAIL] eval_results_view.generation_config: only" in report


def test_a_view_row_no_fill_reached_moving_fails(pair, tmp_path):
    """`m2` is in the target source and in scope for it, but no fill and no
    comparability group of a fill reaches it, so nothing on its row can have
    been caused by the overlay."""
    before, after = pair
    for directory in (before, after):
        rows = pq.read_table(directory / "fact_results.parquet").to_pylist()
        rows.append(_fact("f3", "vals-ai", "m2", comparability_group_id="grp2"))
        _write_table(directory, "fact_results", _FACT_COLUMNS, rows)
    rows = pq.read_table(after / "models_view.parquet").to_pylist()
    rows[1]["variant_count"] = 7
    _write_table(after, "models_view", _MODELS_VIEW_COLUMNS, rows)

    code, report = _run(pair, tmp_path)

    assert code == 1
    assert (
        "[FAIL] models_view: no movement on rows no fill reached: "
        "variant_count (1 row)"
    ) in report
    assert "[PASS] models_view: zero movement outside the target sources" in report

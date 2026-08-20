"""Collections (notes/collections-spec.md): collection_id derivation,
vendored collection adapters, and the protocol-aware view policy.

The adapter tests build a miniature protocol-varied "study" out of the e2e
fixture shapes: three member records whose exploded rows are installment
fragments, a vendored extract (manifest + synthetic results built through
the real extractor writer, so they take the same explode path production
does), and one ordinary third-party record on the same (benchmark, model)
for ranking comparisons.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pytest

from tests.eee_layout import write_eee_datastore
from tests.test_canonicalise_e2e import (
    _write_cards_fixture,
    _write_minimal_seed_fixture,
    _write_registry_fixture,
)

from eval_card_backend.sources import collections as collections_src


def _load_extractor_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "collections" / "aisi_inference_scaling.py"
    )
    spec = importlib.util.spec_from_file_location("aisi_extractor", path)
    mod = importlib.util.module_from_spec(spec)
    # dataclass field resolution looks the module up in sys.modules
    import sys

    sys.modules["aisi_extractor"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# slug + raw-key derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "UK AI Security Institute",
        "How Inference Compute Shapes Frontier LLM Evaluation",
        "  spaced   out  ",
        "Ünïcode & Sÿmbols!!",
        "a" * 200,
        "---already-dashed---",
        "",
    ],
)
def test_slug_sql_python_parity(raw):
    con = duckdb.connect()
    sql_val = con.execute(
        f"SELECT {collections_src.slug_sql('?')}", [raw]
    ).fetchone()[0]
    assert sql_val == collections_src.slug(raw)


def test_collection_raw_key_guards():
    con = duckdb.connect()
    expr = collections_src.collection_raw_key_sql("org", "name", "harness", "cfg")
    rows = [
        # plain case
        ("UK AISI", "My Study", "inspect_ai", "hle",
         "uk-aisi/my-study"),
        # harness bleed: source_name == eval_library name → key on config
        ("Some Org", "inspect_ai", "inspect_ai", "hle",
         "some-org/hle"),
        # missing org
        (None, "My Study", "x", "hle", "unknown/my-study"),
        # missing source_name
        ("Some Org", None, "x", "hle", "some-org/unlabeled"),
        # both missing
        (None, None, "x", "hle", "unknown/unlabeled"),
        # empty-after-slug org counts as missing
        ("###", "My Study", "x", "hle", "unknown/my-study"),
    ]
    for org, name, harness, cfg, expected in rows:
        got = con.execute(
            f"SELECT {expr} FROM (SELECT ? AS org, ? AS name, ? AS harness, ? AS cfg)",
            [org, name, harness, cfg],
        ).fetchone()[0]
        assert got == expected, (org, name, harness, cfg)


def test_curated_assertion_strict_and_lax():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE collection_keys AS "
        "SELECT 'org-a/study-a' AS raw_key, 'org-a/study-a' AS collection_id, "
        "'Study A' AS display_source_name, 3 AS n_rows"
    )
    curated_ok = {"my-study": {"merge_raw_keys": ["org-a/study-a"]}}
    collections_src.assert_curated_keys_observed(con, curated_ok, strict=True)

    # transition entries (one side of an upstream rename unobserved at any
    # single revision) only warn
    curated_partial = {"my-study": {"merge_raw_keys": ["org-a/study-a", "gone/key"]}}
    collections_src.assert_curated_keys_observed(con, curated_partial, strict=True)

    # a fully-detached entry (NO observed key) hard-fails strict runs
    curated_bad = {"my-study": {"merge_raw_keys": ["gone/key", "also/gone"]}}
    with pytest.raises(RuntimeError, match="no observed raw key"):
        collections_src.assert_curated_keys_observed(con, curated_bad, strict=True)
    # non-strict (config-subset run) degrades to a warning
    collections_src.assert_curated_keys_observed(con, curated_bad, strict=False)


# ---------------------------------------------------------------------------
# adapter fixture machinery
# ---------------------------------------------------------------------------

MODEL_ID = "openai/gpt-4o"
STUDY_ORG_A = "Test AISI Institute"
STUDY_ORG_B = "Test AISI Initiative"   # spelling-split twin
STUDY_NAME = "Mini Study Paper"
STUDY_SLUG = "mini-study-paper"


def _study_record(evaluation_id, org, eval_name, score, retrieved="2026-04-01T00:00:00Z"):
    return {
        "evaluation_id": evaluation_id,
        "schema_version": "0.3.0",
        "retrieved_timestamp": retrieved,
        "model_info": {"developer": "openai", "name": MODEL_ID, "id": MODEL_ID},
        "source_metadata": {
            "source_name": STUDY_NAME,
            "source_type": "evaluation_run",
            "source_organization_name": org,
            "evaluator_relationship": "third_party",
        },
        "eval_library": {"name": "inspect_ai", "version": "1.0"},
        "evaluation_results": [
            {
                "evaluation_name": eval_name,
                "source_data": {"dataset_name": "minibench", "source_type": "other"},
                "metric_config": {
                    "metric_name": "accuracy",
                    "evaluation_description": "accuracy",
                    "lower_is_better": False,
                },
                "score_details": {"score": score},
                "generation_config": {
                    "generation_args": {"temperature": 0.0, "max_tokens": 4096}
                },
            }
        ],
    }


def _ordinary_record(score=0.6):
    return {
        "evaluation_id": "minibench/ordinary/1",
        "schema_version": "0.3.0",
        "retrieved_timestamp": "2026-04-02T00:00:00Z",
        "model_info": {"developer": "openai", "name": MODEL_ID, "id": MODEL_ID},
        "source_metadata": {
            "source_name": "Other Leaderboard",
            "source_type": "documentation",
            "source_organization_name": "Other Org",
            "evaluator_relationship": "third_party",
        },
        "eval_library": {"name": "minibench", "version": "1.0"},
        "evaluation_results": [
            {
                "evaluation_name": "minibench",
                "source_data": {"dataset_name": "minibench", "source_type": "other"},
                "metric_config": {
                    "metric_name": "accuracy",
                    "evaluation_description": "accuracy",
                    "lower_is_better": False,
                },
                "score_details": {"score": score},
                "generation_config": {
                    "generation_args": {"temperature": 0.0, "max_tokens": 1024}
                },
            }
        ],
    }


PROTOCOL_NONE = {
    "scaffold": "S-adaptive", "compaction": False, "feedback": "none",
    "token_limit": 1000000, "reasoning_tokens": 16000,
    "reasoning_effort": "high",
}
PROTOCOL_FEEDBACK = {**PROTOCOL_NONE, "feedback": "answer_feedback"}


def _canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _synthetic_result(eval_id, retrieved, org, score, se, n_tasks):
    rec = _study_record(eval_id, org, "minibench", score, retrieved=retrieved)
    rec["evaluation_results"][0]["score_details"] = {
        "score": score,
        "uncertainty": {
            "standard_error": {"value": se, "method": "clustered_task_se"},
            "num_samples": n_tasks,
        },
        "details": {"n_trajectories": "4"},
    }
    return rec


def _write_vendor_fixture(vendor_root: Path, *, manifest_overrides=None,
                          members=None) -> None:
    """Vendored extract for the mini study: manifest + results.parquet
    (built through the real extractor writer) + trajectories.parquet."""
    extractor = _load_extractor_module()
    out = vendor_root / "test_study"
    out.mkdir(parents=True, exist_ok=True)

    if members is None:
        members = [
            {"record_uuid": "rec_a", "evaluation_id": "minibench/model/a",
             "path": "data/minibench/dev/model/rec_a.json",
             "config": "minibench", "n_results": 1},
            {"record_uuid": "rec_b", "evaluation_id": "minibench/model/b",
             "path": "data/minibench/dev/model/rec_b.json",
             "config": "minibench", "n_results": 1},
            {"record_uuid": "rec_c", "evaluation_id": "minibench/model/c",
             "path": "data/minibench/dev/model/rec_c.json",
             "config": "minibench", "n_results": 1},
        ]
    manifest = {
        "collection_id": "test-study",
        "study_slug": STUDY_SLUG,
        "extractor": "tests",
        # A pin is mandatory; the fixture EEE tree has no listing
        # file, so the consumed revision is unknown → warn-only compare.
        "eee_revision": "fixture-rev",
        "expected_drop_count": sum(m["n_results"] for m in members),
        "members": members,
    }
    manifest.update(manifest_overrides or {})
    (out / "manifest.json").write_text(json.dumps(manifest))

    synthetic = [
        {
            "record": _synthetic_result(
                "minibench/model/a", "2026-04-01T00:00:00Z", STUDY_ORG_A,
                0.5, 0.05, 10,
            ),
            "config": "minibench",
            "path": "data/minibench/dev/model/rec_a.json",
            "protocols": [(_canon(PROTOCOL_NONE), 40)],
        },
        {
            "record": _synthetic_result(
                "minibench/model/c", "2026-04-01T00:00:00Z", STUDY_ORG_B,
                0.95, 0.02, 10,
            ),
            "config": "minibench",
            "path": "data/minibench/dev/model/rec_c.json",
            "protocols": [(_canon(PROTOCOL_FEEDBACK), 40)],
        },
    ]
    extractor.write_results_parquet(synthetic, out / "results.parquet")

    import pyarrow as pa
    import pyarrow.parquet as pq
    traj_rows = [
        {
            "collection_id": "test-study", "benchmark_raw": "minibench",
            "model_raw": MODEL_ID, "task_id": f"t{i}",
            "protocol_condition": _canon(PROTOCOL_NONE),
            "trajectory_idx": 1, "score": float(i % 2), "is_correct": bool(i % 2),
            "total_tokens": 1000 + i, "output_tokens": 500, "reasoning_tokens": 100,
            "num_turns": 3, "tool_calls": 0, "n_pieces": 1,
            "wall_time_s": 10.0, "working_time_s": 5.0, "stop_reason": "submit",
            "partial_start": False, "unstitchable": False,
            "token_source_cumulative": True, "source_record_uuids": ["rec_a"],
        }
        for i in range(2)
    ]
    pq.write_table(pa.Table.from_pylist(traj_rows), out / "trajectories.parquet")


def _write_curated_fixture(path: Path) -> None:
    path.write_text(
        "test-study:\n"
        "  display_name: Test Study\n"
        "  kind: paper_study\n"
        "  curated: true\n"
        "  merge_raw_keys:\n"
        "    - test-aisi-institute/mini-study-paper\n"
        "    - test-aisi-initiative/mini-study-paper\n"
        "  protocol_axes:\n"
        "    - {key: feedback, type: categorical, values: [none, answer_feedback, unknown]}\n"
    )


def _write_study_eee(eee_root: Path, extra_records=()) -> None:
    files = [
        ("minibench", "rec_a.json", json.dumps(_study_record(
            "minibench/model/a", STUDY_ORG_A,
            "accuracy on minibench/S-adaptive/+1ep for scorer x", 0.2))),
        ("minibench", "rec_b.json", json.dumps(_study_record(
            "minibench/model/b", STUDY_ORG_A,
            "accuracy on minibench/S-adaptive/+2ep/abc for scorer x", 1.0))),
        ("minibench", "rec_c.json", json.dumps(_study_record(
            "minibench/model/c", STUDY_ORG_B,
            "accuracy on minibench/S-adaptive/+1ep for scorer y", 0.9))),
        ("minibench", "ordinary.json", json.dumps(_ordinary_record())),
    ]
    files.extend(extra_records)
    write_eee_datastore(eee_root, files)


def _run_pipeline(tmp_path, monkeypatch, *, snapshot="2026-04-30T00:00:00Z"):
    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    seed_root = tmp_path / "seed"
    _write_registry_fixture(reg_root)
    _write_minimal_seed_fixture(seed_root)
    _write_cards_fixture(cards_root)

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)
    monkeypatch.setenv("COLLECTIONS_VENDOR_DIR", str(tmp_path / "vendor_collections"))
    monkeypatch.setenv("COLLECTIONS_CURATED_PATH", str(tmp_path / "curated.yaml"))

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    return pipeline.run(
        Settings.from_env(),
        snapshot_id=snapshot,
        warehouse_dir=str(tmp_path / "warehouse"),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
    )


@pytest.fixture()
def adapter_out(tmp_path, monkeypatch):
    _write_study_eee(tmp_path / "eee")
    _write_vendor_fixture(tmp_path / "vendor_collections")
    _write_curated_fixture(tmp_path / "curated.yaml")
    out = _run_pipeline(tmp_path, monkeypatch)
    assert out is not None
    return out


# ---------------------------------------------------------------------------
# adapter end-to-end
# ---------------------------------------------------------------------------


def test_fragments_dropped_synthetics_injected(adapter_out):
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT evaluation_id, score, protocol_condition, collection_id,
               slice_key, is_verified_evaluator
        FROM read_parquet('{adapter_out}/fact_results.parquet')
        ORDER BY score
        """
    ).fetchall()
    scores = sorted(r[1] for r in rows)
    # fragments (0.2, 1.0, 0.9) gone; synthetics (0.5, 0.95) + ordinary (0.6)
    assert scores == [0.5, 0.6, 0.95]
    by_eval = {r[0]: r for r in rows}
    for eid in ("minibench/model/a", "minibench/model/c"):
        r = by_eval[eid]
        assert r[2] is not None and json.loads(r[2])["feedback"] in (
            "none", "answer_feedback"
        )
        assert r[3] == "test-study"       # curated merge (both spellings)
        assert r[4] is None               # slice-key exemption
    ordinary = by_eval["minibench/ordinary/1"]
    assert ordinary[2] is None
    assert ordinary[3] == "other-org/other-leaderboard"


def test_view_layer_protocol_policy(adapter_out):
    con = duckdb.connect()
    erv = con.execute(
        f"""
        SELECT protocol_condition, score, position, total, collection_id
        FROM read_parquet('{adapter_out}/eval_results_view.parquet')
        ORDER BY score
        """
    ).fetchall()
    # one view row per protocol point: NULL (ordinary), none-arm, feedback-arm
    assert len(erv) == 3
    by_score = {round(r[1], 4): r for r in erv}
    # feedback row: shown but never ranked
    assert by_score[0.95][2] is None
    # ranked pool = ordinary (0.6) + none-arm protocol row (0.5)
    assert by_score[0.6][2] == 1 and by_score[0.6][3] == 2
    assert by_score[0.5][2] == 2 and by_score[0.5][3] == 2
    assert by_score[0.5][4] == "test-study"

    ev = con.execute(
        f"""
        SELECT top_score, best_model.score, avg_score
        FROM read_parquet('{adapter_out}/evals_view.parquet')
        WHERE benchmark_id = 'minibench'
        """
    ).fetchone()
    # best-style rollups exclude the answer-feedback row (0.95)
    assert ev[0] == 0.6 and ev[1] == 0.6
    assert abs(ev[2] - (0.5 + 0.6) / 2) < 1e-9

    merged = con.execute(
        f"""
        SELECT best_result.score
        FROM read_parquet('{adapter_out}/merged_evals_view.parquet')
        WHERE benchmark_id = 'minibench'
        """
    ).fetchone()
    assert merged[0] == 0.6

    mv = con.execute(
        f"""
        SELECT score_summary.max, score_summary.average
        FROM read_parquet('{adapter_out}/models_view.parquet')
        WHERE model_key = '{MODEL_ID}'
        """
    ).fetchone()
    assert mv[0] == 0.6
    assert abs(mv[1] - (0.5 + 0.6) / 2) < 1e-9


def test_comparison_index_protocol_collapse(adapter_out):
    payload = json.loads((adapter_out / "comparison-index.json").read_text())
    eval_id = "minibench%2Fminibench"
    entry = payload["evals"][eval_id]
    (metric,) = entry["metrics"]
    (cell,) = metric["scores"]
    # collapsed to the best non-feedback row (ordinary 0.6 beats 0.5);
    # 2 protocol-legal rows stand behind the cell
    assert cell["score"] == 0.6
    assert cell["submission_count"] == 2
    assert cell["submission_axis"] == "protocol"
    assert cell["total"] == 1  # one model on the leaderboard


def test_collections_sidecar_and_trajectories(adapter_out):
    payload = json.loads((adapter_out / "collections.json").read_text())
    assert payload["test-study"]["curated"] is True
    assert payload["test-study"]["kind"] == "paper_study"
    stub = payload["other-org/other-leaderboard"]
    assert stub == {
        "display_name": "Other Leaderboard", "kind": "unknown", "curated": False,
    }
    # raw merge keys never surface as stub ids
    assert "test-aisi-institute/mini-study-paper" not in payload

    con = duckdb.connect()
    traj = con.execute(
        f"""
        SELECT collection_id, benchmark_id, model_id, count(*)
        FROM read_parquet('{adapter_out}/collection_trajectories.parquet')
        GROUP BY 1, 2, 3
        """
    ).fetchall()
    assert traj == [("test-study", "minibench", MODEL_ID, 2)]


def test_manifest_lists_collections_sidecar(adapter_out):
    manifest = json.loads((adapter_out / "manifest.json").read_text())
    assert manifest["summary_artifacts"]["collections"] == "collections.json"
    snap = json.loads((adapter_out / "snapshot_meta.json").read_text())
    assert "collections.json" in snap["sidecars"]


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def test_leak_guard_fires_on_unmanifested_member(tmp_path, monkeypatch):
    extra = ("minibench", "rec_new.json", json.dumps(_study_record(
        "minibench/model/NEW", STUDY_ORG_A,
        "accuracy on minibench/S-adaptive/+9ep for scorer x", 0.1)))
    _write_study_eee(tmp_path / "eee", extra_records=[extra])
    _write_vendor_fixture(tmp_path / "vendor_collections")
    _write_curated_fixture(tmp_path / "curated.yaml")
    with pytest.raises(RuntimeError, match="leak guard"):
        _run_pipeline(tmp_path, monkeypatch)


def test_drop_count_mismatch_fails(tmp_path, monkeypatch):
    _write_study_eee(tmp_path / "eee")
    members = [
        {"record_uuid": "rec_a", "evaluation_id": "minibench/model/a",
         "path": "data/minibench/dev/model/rec_a.json",
         "config": "minibench", "n_results": 2},   # wrong: record has 1
        {"record_uuid": "rec_b", "evaluation_id": "minibench/model/b",
         "path": "data/minibench/dev/model/rec_b.json",
         "config": "minibench", "n_results": 1},
        {"record_uuid": "rec_c", "evaluation_id": "minibench/model/c",
         "path": "data/minibench/dev/model/rec_c.json",
         "config": "minibench", "n_results": 1},
    ]
    _write_vendor_fixture(tmp_path / "vendor_collections", members=members)
    _write_curated_fixture(tmp_path / "curated.yaml")
    with pytest.raises(RuntimeError, match="out of sync"):
        _run_pipeline(tmp_path, monkeypatch)


def test_revision_mismatch_fails(tmp_path, monkeypatch):
    eee_root = tmp_path / "eee"
    _write_study_eee(eee_root)
    paths = sorted(
        p.relative_to(eee_root).as_posix()
        for p in (eee_root / "data").rglob("*.json")
    )
    (eee_root / ".eee_file_listing.json").write_text(
        json.dumps({"revision": "consumed-rev", "paths": paths})
    )
    _write_vendor_fixture(
        tmp_path / "vendor_collections",
        manifest_overrides={"eee_revision": "extractor-rev"},
    )
    _write_curated_fixture(tmp_path / "curated.yaml")
    with pytest.raises(RuntimeError, match="Re-run the extractor"):
        _run_pipeline(tmp_path, monkeypatch)


def test_curated_detach_fails_full_run(tmp_path, monkeypatch):
    _write_study_eee(tmp_path / "eee")
    _write_vendor_fixture(tmp_path / "vendor_collections")
    # An extra curated entry with NO observed raw key at all = detached.
    # (A partially-observed merge list — transition keys — only warns.)
    (tmp_path / "curated.yaml").write_text(
        "test-study:\n"
        "  display_name: Test Study\n"
        "  merge_raw_keys:\n"
        "    - test-aisi-institute/mini-study-paper\n"
        "    - test-aisi-initiative/mini-study-paper\n"
        "detached-study:\n"
        "  display_name: Detached Study\n"
        "  merge_raw_keys:\n"
        "    - never-observed/key\n"
    )
    with pytest.raises(RuntimeError, match="no observed raw key"):
        _run_pipeline(tmp_path, monkeypatch)

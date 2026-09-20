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
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from tests.eee_layout import write_eee_datastore
from tests.test_canonicalise_e2e import (
    _write_cards_fixture,
    _write_minimal_seed_fixture,
    _write_registry_fixture,
)

from eval_card_backend.sources import collections as collections_src

JUDGE_GPT = "openai/gpt-4o-2024-05-13"


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


def test_aisi_terminalbench_uses_binary_ever_correct_outcome(tmp_path):
    extractor = _load_extractor_module()
    member = extractor.Member(
        path="data/terminalbench/openai/model/record.json",
        uuid="record",
        config="terminalbench",
        record={
            "model_info": {"id": "openai/model"},
            # The upstream summary describes the raw fractional row scores,
            # not the paper-defined binary trajectory outcomes emitted below.
            "evaluation_results": [{
                "evaluation_name": "accuracy",
                "score_details": {"score": 3 / 7},
            }],
        },
        evaluation_id="terminalbench/openai-model/1",
        n_results=1,
        feedback="answer_feedback",
        reasoning_effort="high",
        reasoning_tokens=16000,
        generation_config=None,
        source_data=None,
    )
    cases = [
        # Fractional final reward, but a fully-correct submission: success.
        ("fractional-success", 0.5, [1.0], None, True),
        # The same fractional reward with only failed submissions: failure.
        ("fractional-failure", 0.5, [0.0, 0.0], None, False),
        # Work after a retroactive repetition cut is outside the credited
        # trajectory, so the pre-cut submission history is authoritative.
        ("post-cut-final-success", 1.0, [0.0], "repetition_guard", False),
        # A fully-correct final evaluation is also sufficient when submission
        # instrumentation is absent.
        ("missing-history-final-success", 1.0, None, None, True),
        ("missing-history-failure", 0.0, None, "tool_calls", False),
        # The harness stop is independent positive evidence.
        ("successful-stop", 0.0, None, "completed_on_successful_submit", True),
        ("failure", 0.0, [], "repetition_guard", False),
        ("no-evidence", None, None, None, None),
    ]
    trajectories = []
    stats = Counter()
    for index, (task, score, submissions, stop, expected) in enumerate(cases):
        row = {
            "sample_id": task,
            "evaluation": {
                "score": score,
                "is_correct": score > 0 if score is not None else None,
            },
            "metadata": {
                "epoch": index,
                "stop_reason": stop,
            },
        }
        if submissions is not None:
            row["metadata"]["submissions"] = json.dumps(
                [{"score": value} for value in submissions]
            )
        trajectory = extractor.parse_sample_row(
            member,
            row,
            stats,
        )
        assert trajectory is not None
        assert trajectory.outcome() == (
            None if expected is None else float(expected)
        )
        trajectory.protocol = {
            "scaffold": "S-adaptive",
            "compaction": True,
            "feedback": "answer_feedback",
            "token_limit": 10_000_000,
            "reasoning_tokens": 16_000,
            "reasoning_effort": "high",
        }
        trajectory.included = True
        trajectories.append(trajectory)

    extractor.summarise_terminalbench_outcomes(trajectories, stats)
    assert {
        key: stats[key]
        for key in (
            "terminalbench_submission_history_present",
            "terminalbench_submission_history_missing",
            "terminalbench_final_score_fallback",
            "terminalbench_final_score_fallback_positive",
            "terminalbench_post_cut_score1_rows",
            "terminalbench_unexplained_score1_submission0",
            "terminalbench_success_stop_submission0_conflicts",
            "terminalbench_outcome_unknown",
        )
    } == {
        "terminalbench_submission_history_present": 4,
        "terminalbench_submission_history_missing": 4,
        "terminalbench_final_score_fallback": 2,
        "terminalbench_final_score_fallback_positive": 1,
        "terminalbench_post_cut_score1_rows": 1,
        "terminalbench_unexplained_score1_submission0": 0,
        "terminalbench_success_stop_submission0_conflicts": 0,
        "terminalbench_outcome_unknown": 1,
    }

    malformed_stats = Counter()
    malformed = extractor.parse_sample_row(
        member,
        {
            "sample_id": "malformed-history",
            "evaluation": {"score": 0.0, "is_correct": False},
            "metadata": {"submissions": "{not-json"},
        },
        malformed_stats,
    )
    assert malformed is not None
    assert malformed.outcome() == 0.0
    assert malformed_stats["rows_submissions_parse_error"] == 1
    assert malformed_stats["terminalbench_submissions_parse_error"] == 1

    incomplete_stats = Counter()
    incomplete = extractor.parse_sample_row(
        member,
        {
            "sample_id": "incomplete-history",
            "evaluation": {"score": 0.0, "is_correct": False},
            "metadata": {"submissions": json.dumps([{"score": "unknown"}])},
        },
        incomplete_stats,
    )
    assert incomplete is not None
    assert incomplete.outcome() == 0.0
    assert incomplete_stats["rows_submissions_incomplete"] == 1
    assert incomplete_stats["terminalbench_submissions_incomplete"] == 1

    unexplained_stats = Counter()
    unexplained = extractor.parse_sample_row(
        member,
        {
            "sample_id": "unexplained-final-success",
            "evaluation": {"score": 1.0, "is_correct": True},
            "metadata": {
                "submissions": json.dumps([{"score": 0.0}]),
                "traj_stopping_reason": "tool_calls",
            },
        },
        unexplained_stats,
    )
    assert unexplained is not None
    extractor.summarise_terminalbench_outcomes(
        [unexplained], unexplained_stats
    )
    assert unexplained_stats[
        "terminalbench_unexplained_score1_submission0"
    ] == 1

    cells, dropped = extractor.build_cells(trajectories, Counter())
    assert dropped == []
    assert len(cells) == 1
    assert cells[0].score == pytest.approx(3 / 7)
    assert cells[0].n_tasks == 7

    reconciliation = extractor.reconcile_record_aggregates(
        [member], trajectories
    )["terminalbench"]
    assert reconciliation["records"] == 1
    assert reconciliation["exact"] == 1

    out = tmp_path / "trajectories.parquet"
    extractor.write_trajectories_parquet(trajectories, out)
    emitted = duckdb.connect().execute(
        "SELECT task_id, score, is_correct FROM read_parquet(?) ORDER BY task_id",
        [str(out)],
    ).fetchall()
    assert emitted == [
        ("failure", 0.0, False),
        ("fractional-failure", 0.0, False),
        ("fractional-success", 1.0, True),
        ("missing-history-failure", 0.0, False),
        ("missing-history-final-success", 1.0, True),
        ("no-evidence", None, None),
        ("post-cut-final-success", 0.0, False),
        ("successful-stop", 1.0, True),
    ]


def test_aisi_terminalbench_filters_to_papers_86_task_population():
    extractor = _load_extractor_module()
    rows = [
        SimpleNamespace(config="terminalbench", sample_id=f"paper-task-{i}")
        for i in range(86)
    ]
    rows.extend([
        SimpleNamespace(
            config="terminalbench", sample_id="filter-js-from-html"
        ),
        SimpleNamespace(
            config="terminalbench", sample_id="mcmc-sampling-stan"
        ),
        # The exclusion is scoped to this study's TerminalBench data.
        SimpleNamespace(config="hle", sample_id="filter-js-from-html"),
    ])
    stats = Counter()

    filtered = extractor.filter_paper_terminalbench_tasks(rows, stats)

    assert len(filtered) == 87
    assert {row.sample_id for row in filtered if row.config == "terminalbench"} == {
        f"paper-task-{i}" for i in range(86)
    }
    assert any(row.config == "hle" for row in filtered)
    assert stats["terminalbench_paper_task_count"] == 86
    assert stats["terminalbench_paper_excluded_tasks_observed"] == 2
    assert stats["terminalbench_paper_excluded_trajectories"] == 2

    with pytest.raises(ValueError, match="paper exclusions changed"):
        extractor.filter_paper_terminalbench_tasks(rows[:-2], Counter())


def test_aisi_frontiermath_keeps_raw_score_for_cell_aggregation(tmp_path):
    extractor = _load_extractor_module()
    member = extractor.Member(
        path="data/frontiermath/openai/model/record.json",
        uuid="record",
        config="frontiermath",
        record={
            "model_info": {"id": "openai/model"},
            "evaluation_results": [{
                "evaluation_name": "accuracy",
                "score_details": {"score": 0.25},
            }],
        },
        evaluation_id="frontiermath/openai-model/1",
        n_results=1,
        feedback=None,
        reasoning_effort="high",
        reasoning_tokens=16000,
        generation_config=None,
        source_data=None,
    )
    trajectories = []
    for index, is_correct in enumerate((True, False)):
        trajectory = extractor.parse_sample_row(
            member,
            {
                "sample_id": f"task-{index}",
                "evaluation": {"score": 0.25, "is_correct": is_correct},
                "metadata": {"epoch": index},
            },
            Counter(),
        )
        assert trajectory is not None
        trajectory.protocol = {
            "scaffold": "S-adaptive",
            "compaction": False,
            "feedback": "none",
            "token_limit": 1_000_000,
            "reasoning_tokens": 16_000,
            "reasoning_effort": "high",
        }
        trajectory.included = True
        trajectories.append(trajectory)

    cells, dropped = extractor.build_cells(trajectories, Counter())
    assert dropped == []
    assert len(cells) == 1
    # FrontierMath rows carry a pre-averaged task solve rate in `score`.
    # Its headline must therefore stay 0.25, not become mean(is_correct)=0.5.
    assert cells[0].score == 0.25

    reconciliation = extractor.reconcile_record_aggregates(
        [member], trajectories
    )["frontiermath"]
    assert reconciliation["records"] == 1
    assert reconciliation["exact"] == 1

    out = tmp_path / "frontiermath-trajectories.parquet"
    extractor.write_trajectories_parquet(trajectories, out)
    emitted = duckdb.connect().execute(
        "SELECT score, is_correct FROM read_parquet(?) ORDER BY task_id",
        [str(out)],
    ).fetchall()
    assert emitted == [(1.0, True), (0.0, False)]


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


def _synthetic_result(eval_id, retrieved, org, score, se, n_tasks,
                      judges=(JUDGE_GPT,)):
    rec = _study_record(eval_id, org, "minibench", score, retrieved=retrieved)
    # The study grades with an LLM judge and says so in the typed
    # `llm_scoring` struct, so these rows carry BOTH condition axes through
    # ingestion: the protocol point the adapter assigns and the judge
    # condition Stage D reads off the record.
    rec["evaluation_results"][0]["metric_config"]["llm_scoring"] = {
        "input_prompt": "grade the answer",
        "judges": [
            {"model_info": {"id": j, "name": j.split("/")[-1]}} for j in judges
        ],
    }
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
    # ranked pool = the model's ONE headline row (issue #47): the arms
    # compete, the best non-feedback arm represents the model, and the
    # others stay visible and unranked beneath it
    assert by_score[0.6][2] == 1 and by_score[0.6][3] == 1
    assert by_score[0.5][2] is None
    assert by_score[0.5][4] == "test-study"

    ev = con.execute(
        f"""
        SELECT top_score, best_model.score, avg_score
        FROM read_parquet('{adapter_out}/evals_view.parquet')
        WHERE benchmark_id = 'minibench'
        """
    ).fetchone()
    # best-style rollups exclude the answer-feedback row (0.95); the
    # average is over headline rows, so the model counts once (0.6)
    assert ev[0] == 0.6 and ev[1] == 0.6
    assert ev[2] == 0.6

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
    assert mv[1] == 0.6


def test_comparison_index_protocol_collapse(adapter_out):
    payload = json.loads((adapter_out / "comparison-index.json").read_text())
    eval_id = "minibench%2Fminibench"
    entry = payload["evals"][eval_id]
    (metric,) = entry["metrics"]
    (cell,) = metric["scores"]
    # the cell is the model's headline row (ordinary 0.6 beats the 0.5
    # arm); 2 protocol-legal arms stand behind it
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


def test_protocol_and_judge_conditions_survive_stage_f_together(adapter_out):
    """Ingestion fixture for the two condition axes at once: the study's rows
    carry a protocol point from the adapter AND a judge condition from the
    record's typed `llm_scoring`. Stage F's group key and hash must take both
    — the hash is md5 over the six grain components in order."""
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT protocol_condition, judge_condition, comparability_group_id,
               md5(md5(model_aggregation_key)
                   || md5(benchmark_key)
                   || md5(COALESCE(slice_key, ''))
                   || md5(metric_key)
                   || md5(COALESCE(protocol_condition, ''))
                   || md5(COALESCE(judge_condition, ''))
                   -- split joined the comparability key: a run on `test` and
                   -- a run on `train` are two measurements
                   || md5(COALESCE(split, ''))) AS expected
        FROM read_parquet('{adapter_out}/fact_results.parquet')
        WHERE collection_id = 'test-study'
        ORDER BY protocol_condition
        """
    ).fetchall()
    assert len(rows) == 2
    for protocol, judge, group_id, expected in rows:
        assert protocol is not None
        assert json.loads(judge) == {
            "judges": [JUDGE_GPT], "label": "accuracy",
        }
        assert group_id == expected
    # the two arms are two measurements: same judge, different protocol point
    assert rows[0][1] == rows[1][1]
    assert rows[0][0] != rows[1][0]
    assert rows[0][2] != rows[1][2]


def test_trajectory_emit_is_byte_identical_across_runs(tmp_path):
    """The trajectory sort has to be total. Two emits of the same rows in a
    different physical order must produce the same bytes; the natural key
    (collection, benchmark, model, protocol, task, idx) is not unique, so the
    whole row is the final tie-break."""
    from eval_card_backend.canonicalise import stages

    traj = [
        # three rows sharing the natural key, distinguished only by payload
        {"collection_id": "c", "benchmark_raw": "b", "model_raw": "m",
         "task_id": "t0", "protocol_condition": None, "trajectory_idx": None,
         "score": float(i), "is_correct": bool(i % 2),
         "total_tokens": 100 + i, "output_tokens": 50, "reasoning_tokens": 10,
         "num_turns": 2, "tool_calls": 1, "n_pieces": 1,
         "wall_time_s": 1.0, "working_time_s": 0.5, "stop_reason": "submit",
         "partial_start": False, "unstitchable": False,
         "token_source_cumulative": True, "source_record_uuids": ["u"]}
        for i in range(3)
    ]

    def emit(order, out_dir):
        con = duckdb.connect()
        collections_src.create_collection_tables(con)
        for row in order:
            con.execute(
                "INSERT INTO collection_trajectories_raw VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                list(row.values()),
            )
        for table, cols in (
            ("fact_results",
             "'' AS collection_id, '' AS model_raw, '' AS source_config, "
             "'' AS model_id, '' AS benchmark_id, '' AS model_key, "
             "'' AS benchmark_key, '' AS evaluation_id, '' AS composite_slug, "
             "'' AS metric_key, '' AS slice_key, '' AS fact_id"),
            ("benchmarks", "'' AS composite_slug, '' AS benchmark_id"),
            ("composites", "'' AS composite_slug"),
            ("families", "'' AS family_id"),
            ("models", "'' AS model_key"),
            ("canonical_metrics", "'' AS id"),
        ):
            con.execute(f"CREATE TABLE {table} AS SELECT {cols} WHERE FALSE")
        stages.stage_i_emit_warehouse_parquets(
            con, out_dir, "2026-04-30T00:00:00Z"
        )
        return (out_dir / "collection_trajectories.parquet").read_bytes()

    first = emit(traj, tmp_path / "a")
    second = emit(list(reversed(traj)), tmp_path / "b")
    assert first == second


# ---------------------------------------------------------------------------
# Aggregate-only members (privately held benchmarks, no transcripts)
# ---------------------------------------------------------------------------


def _agg_only_result(series: str, threshold: int, score: float, se: float,
                     num_samples: int, *, benchmark: str) -> dict:
    """One published result in the shape the AISI cyber export produces."""
    return {
        "evaluation_name": f"{benchmark}.{series}.tokens_{threshold}",
        "source_data": {
            "dataset_name": benchmark,
            "source_type": "url",
            "url": ["https://arxiv.org/abs/2606.17930"],
        },
        "metric_config": {
            "metric_id": "accuracy",
            "lower_is_better": False,
            "score_type": "continuous",
            "metric_parameters": {"token_threshold": threshold},
            "min_score": 0.0,
            "max_score": 1.0,
        },
        "score_details": {
            "score": score,
            "uncertainty": {
                "standard_error": {"value": se, "method": "analytic"},
                "num_samples": num_samples,
            },
            "details": {
                "token_threshold": str(threshold),
                "token_budget_cap": "50000000",
                "trajectory_count": "535",
                "trajectories_per_task": "5",
                "evaluated_task_count": str(num_samples),
                "se_definition": "between-task standard error",
            },
        },
        "generation_config": {
            "generation_args": {"eval_limits": {"token_limit": 50000000}},
            "additional_details": {"harness": "ReAct-style agent (Inspect)"},
        },
    }


def _agg_only_member(extractor, benchmark: str, results: list[dict]):
    return extractor.Member(
        path=f"data/{benchmark}/anthropic/claude-opus-4-6/record.json",
        uuid="record",
        config=benchmark,
        record={
            "schema_version": "0.3.0",
            "evaluation_id": f"{benchmark}/anthropic_claude-opus-4-6/1.0",
            "retrieved_timestamp": "1.0",
            "source_metadata": {
                "source_name": "How Inference Compute Shapes Frontier LLM Evaluation",
                "source_type": "evaluation_run",
                "source_organization_name": "UK AI Security Institute",
                "evaluator_relationship": "third_party",
            },
            "eval_library": {"name": "inspect_ai", "version": "unknown"},
            "model_info": {
                "name": "anthropic/claude-opus-4-6",
                "id": "anthropic/claude-opus-4-6",
                "additional_details": {
                    "deployment_type": "externally_managed",
                    "model_availability": "closed_weights",
                },
            },
            "evaluation_results": results,
        },
        evaluation_id=f"{benchmark}/anthropic_claude-opus-4-6/1.0",
        n_results=len(results),
        feedback=None,
        reasoning_effort=None,
        reasoning_tokens=None,
        generation_config=results[0]["generation_config"],
        source_data=results[0]["source_data"],
    )


def test_aggregate_only_flag_tracks_declaration():
    extractor = _load_extractor_module()
    assert extractor.aggregate_only("aisi-cyber-ctfs")
    assert extractor.aggregate_only("aisi-the-last-ones")
    assert not extractor.aggregate_only("terminalbench")
    assert not extractor.aggregate_only("hle")


def test_token_threshold_read_from_each_carrier():
    extractor = _load_extractor_module()
    typed = {"metric_config": {"metric_parameters": {"token_threshold": 5_000_000}}}
    assert extractor._result_token_threshold(typed) == 5_000_000
    stringy = {"score_details": {"details": {"token_threshold": "1500000"}}}
    assert extractor._result_token_threshold(stringy) == 1_500_000
    from_name = {"evaluation_name": "aisi-cyber-ctfs.cumulative_success.tokens_500000"}
    assert extractor._result_token_threshold(from_name) == 500_000
    assert extractor._result_token_threshold({"evaluation_name": "x.y"}) is None


def test_aggregate_only_cells_restate_every_threshold():
    """Each published threshold becomes one cell on the token_limit axis —
    the published score is restated, never recomputed."""
    extractor = _load_extractor_module()
    points = [(500_000, 0.4, 0.043596), (1_500_000, 0.530841, 0.044161),
              (5_000_000, 0.657944, 0.042046), (15_000_000, 0.745794, 0.037715),
              (50_000_000, 0.811215, 0.033419)]
    member = _agg_only_member(extractor, "aisi-cyber-ctfs", [
        _agg_only_result("cumulative_success", t, s, se, 107,
                         benchmark="aisi-cyber-ctfs")
        for t, s, se in points
    ])
    stats = Counter()
    cells, dropped = extractor.build_aggregate_only_cells([member], stats)

    assert dropped == []
    assert len(cells) == len(points)
    assert [c.protocol["token_limit"] for c in cells] == [t for t, _, _ in points]
    assert [c.score for c in cells] == [s for _, s, _ in points]
    assert [c.score_se for c in cells] == [se for _, _, se in points]
    for c in cells:
        assert c.trajectories == []            # nothing was streamed
        assert c.aggregate_method == "published_aggregate"
        assert c.n_tasks == 107
        assert c.n_trajectories_declared == 535
        assert c.protocol["scaffold"] == "ReAct"
        assert c.protocol["compaction"] is True
        assert c.protocol["feedback"] == "none"
        assert c.published_details["source"] == "published_aggregate"
    assert stats["aggregate_only_cells"] == len(points)


def test_aggregate_only_selects_declared_series_and_keeps_companion():
    """The Last Ones reports two series per threshold; only the declared one
    becomes the cell score, and the companion rides along as provenance."""
    extractor = _load_extractor_module()
    bench = "aisi-the-last-ones"
    results = [
        _agg_only_result("full_completion", 1_000_000, 0.0, 0.0, 5, benchmark=bench),
        _agg_only_result("partial_progress", 1_000_000, 0.19375, 0.022361, 5,
                         benchmark=bench),
        _agg_only_result("full_completion", 100_000_000, 0.0, 0.0, 5, benchmark=bench),
        _agg_only_result("partial_progress", 100_000_000, 0.575, 0.081298, 5,
                         benchmark=bench),
    ]
    cells, dropped = extractor.build_aggregate_only_cells(
        [_agg_only_member(extractor, bench, results)], Counter()
    )
    assert dropped == []
    assert [c.score for c in cells] == [0.19375, 0.575]
    assert [c.protocol["token_limit"] for c in cells] == [1_000_000, 100_000_000]
    assert all(c.published_details["full_completion_score"] == "0.0" for c in cells)


def test_aggregate_only_unplaceable_result_is_itemised_not_dropped():
    extractor = _load_extractor_module()
    bad = _agg_only_result("cumulative_success", 500_000, 0.4, 0.01, 107,
                           benchmark="aisi-cyber-ctfs")
    bad["evaluation_name"] = "aisi-cyber-ctfs.cumulative_success.no_threshold"
    bad["metric_config"].pop("metric_parameters")
    bad["score_details"]["details"].pop("token_threshold")
    stats = Counter()
    cells, dropped = extractor.build_aggregate_only_cells(
        [_agg_only_member(extractor, "aisi-cyber-ctfs", [bad])], stats
    )
    assert cells == []
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "no_token_threshold"
    assert stats["aggregate_only_results_unplaceable"] == 1


def test_aggregate_only_synthetic_record_declares_its_provenance():
    extractor = _load_extractor_module()
    member = _agg_only_member(extractor, "aisi-cyber-ctfs", [
        _agg_only_result("cumulative_success", 50_000_000, 0.811215, 0.033419,
                         107, benchmark="aisi-cyber-ctfs")
    ])
    cells, _ = extractor.build_aggregate_only_cells([member], Counter())
    records = extractor.build_synthetic_records(cells, Counter())

    assert len(records) == 1
    result = records[0]["record"]["evaluation_results"][0]
    details = result["score_details"]["details"]
    assert details["aggregation"] == "published_aggregate"
    assert details["source"] == "published_aggregate"
    assert details["token_threshold"] == "50000000"
    # the declared trajectory count, not the (empty) streamed one
    assert details["n_trajectories"] == "535"
    se = result["score_details"]["uncertainty"]["standard_error"]
    assert se["method"] == "published"
    assert se["value"] == 0.033419
    assert records[0]["protocols"][0][1] == 535

"""Stage E dedup accounting on hand-built corpora: which rows each pass keeps,
how `dedup_by_config` attributes them, and the exact conservation checks that
stop a run whose counts do not add up."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from tests.eee_layout import write_eee_datastore
from tests.test_canonicalise_e2e import (
    _assert_dedup_by_config_sums_to_row_counts,
    _write_cards_fixture,
    _write_minimal_seed_fixture,
    _write_registry_fixture,
)

SNAPSHOT_ID = "2026-05-03T00:00:00Z"


def _record(
    evaluation_id: str,
    retrieved: str,
    score: float,
    *,
    metric_unit: str | None = None,
) -> dict:
    metric_config = {
        "metric_id": "minibench.acc", "metric_name": "accuracy",
        "evaluation_description": "Accuracy on minibench",
        "lower_is_better": False,
    }
    if metric_unit is not None:
        metric_config.update(min_score=0.0, max_score=1.0, metric_unit=metric_unit)
    return {
        "evaluation_id": evaluation_id,
        "schema_version": "0.2.2",
        "retrieved_timestamp": retrieved,
        "model_info": {
            "developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o",
            "inference_platform": "openai-api",
        },
        "source_metadata": {
            "source_name": "OpenAI", "source_type": "documentation",
            "source_organization_name": "OpenAI",
            "evaluator_relationship": "first_party",
        },
        "eval_library": {"name": "minibench", "version": "1.0"},
        "evaluation_results": [{
            "evaluation_name": "minibench",
            "source_data": {"dataset_name": "minibench", "source_type": "other"},
            "metric_config": metric_config,
            "score_details": {"score": score},
            "generation_config": {
                "generation_args": {"temperature": 0.0, "max_tokens": 1024}
            },
        }],
    }


def _run(tmp_path: Path, monkeypatch, records, **kwargs) -> Path:
    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    seed_root = tmp_path / "seed"
    if not reg_root.exists():
        _write_registry_fixture(reg_root)
        _write_minimal_seed_fixture(seed_root)
        _write_cards_fixture(cards_root)
        write_eee_datastore(eee_root, [
            (config, name, json.dumps(record)) for config, name, record in records
        ])
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    return pipeline.run(
        Settings.from_env(),
        snapshot_id=SNAPSHOT_ID,
        warehouse_dir=str(tmp_path / "warehouse"),
        registry_local_dir=str(reg_root),
        taxonomy_seed_dir=str(seed_root),
        cache_root=str(tmp_path / "cache"),
        **kwargs,
    )


def _facts(out_dir: Path) -> list[tuple]:
    path = out_dir / "fact_results.parquet"
    return duckdb.connect().execute(
        f"SELECT source_config, evaluation_id, score, fact_id "
        f"FROM read_parquet('{path}') ORDER BY source_config, evaluation_id, score"
    ).fetchall()


def _snapshot_meta(out_dir: Path) -> dict:
    return json.loads((out_dir / "snapshot_meta.json").read_text())


def _counts(staging, after_fact_id, after_content) -> dict[str, int]:
    return {
        "staging": staging, "after_fact_id_pass": after_fact_id,
        "after_content_pass": after_content, "residual_timestamp_only": 0,
        "residual_generation_args_only": 0, "residual_both": 0,
    }


def test_cross_config_fact_id_collision_keeps_one_survivor(tmp_path, monkeypatch):
    """One evaluation_id ingested under two source_configs is one fact_id.
    The fact-id pass is global, so exactly one row survives and it is counted
    under the config of the row that won; a per-config distinct count would
    report a survivor in each config."""
    out_dir = _run(tmp_path, monkeypatch, [
        ("alpha", "old.json", _record("ev_shared", "2026-04-30T00:00:00Z", 0.3)),
        ("beta", "new.json", _record("ev_shared", "2026-05-02T00:00:00Z", 0.9)),
    ])
    facts = _facts(out_dir)
    assert [(c, e, s) for c, e, s, _ in facts] == [("beta", "ev_shared", 0.9)]

    snap = _snapshot_meta(out_dir)
    _assert_dedup_by_config_sums_to_row_counts(snap)
    assert snap["dedup_by_config"] == {
        "alpha": _counts(1, 0, 0),
        "beta": _counts(1, 1, 1),
    }
    assert snap["row_counts"]["dropped_rows_dedup"] == 1
    assert snap["row_counts"]["fact_results"] == 1


def test_null_fact_ids_are_never_collapsed_by_the_fact_id_pass(tmp_path, monkeypatch):
    """A record with an empty evaluation_id has no fact_id. NULL is not an
    identity, so two such rows reach the content pass and, with different
    values, both survive."""
    out_dir = _run(tmp_path, monkeypatch, [
        ("minibench", "a.json", _record("", "2026-04-30T00:00:00Z", 0.3)),
        ("minibench", "b.json", _record("", "2026-05-02T00:00:00Z", 0.9)),
    ])
    facts = _facts(out_dir)
    assert [(s, f) for _, _, s, f in facts] == [(0.3, None), (0.9, None)]

    snap = _snapshot_meta(out_dir)
    _assert_dedup_by_config_sums_to_row_counts(snap)
    assert snap["dedup_by_config"] == {"minibench": _counts(2, 2, 2)}
    assert snap["row_counts"]["dropped_rows_dedup"] == 0


def test_config_with_every_row_dropped_keeps_its_staging_count(tmp_path, monkeypatch):
    """A config whose rows are all no-score or sentinel still appears with
    its staging count and zero survivors, and the totals still conserve.
    Stage A refuses an EEE record without a score, so the no-score row is
    blanked in staging, where other producers can leave one."""
    from eval_card_backend.canonicalise import stages

    stage_d = stages.stage_d_join_dims_and_flatten

    def stage_d_with_a_missing_score(con, **kwargs):
        stage_d(con, **kwargs)
        con.execute(
            "UPDATE fact_results_staging SET score = NULL "
            "WHERE evaluation_id = 'ev_none'"
        )

    monkeypatch.setattr(
        stages, "stage_d_join_dims_and_flatten", stage_d_with_a_missing_score
    )
    out_dir = _run(tmp_path, monkeypatch, [
        ("dead", "no-score.json", _record("ev_none", "2026-04-30T00:00:00Z", 0.5)),
        ("dead", "sentinel.json", _record(
            "ev_sent", "2026-04-30T00:00:00Z", -1.0, metric_unit="proportion",
        )),
        ("live", "ok.json", _record("ev_ok", "2026-04-30T00:00:00Z", 0.8)),
    ])
    assert [(c, e) for c, e, _, _ in _facts(out_dir)] == [("live", "ev_ok")]

    snap = _snapshot_meta(out_dir)
    _assert_dedup_by_config_sums_to_row_counts(snap)
    assert snap["dedup_by_config"] == {
        "dead": _counts(2, 0, 0),
        "live": _counts(1, 1, 1),
    }
    assert snap["row_counts"]["dropped_rows_no_score"] == 1
    assert snap["row_counts"]["dropped_rows_sentinel"] == 1


_CORRUPTIONS = {
    "a materialised row lost": (
        "DELETE FROM fact_results_signaled",
        r"fact_results_signaled has 0 rows but _e_decisions keeps 1 by the "
        r"content pass; staging 2 - no_score 0 - sentinel 0 - fact_id drops 1 "
        r"- content drops 0 = 1, but fact_results_signaled has 0 rows",
    ),
    "a boundary count above the one before it": (
        "UPDATE stage_e_dedup_stats SET after_content_pass = 2 "
        "WHERE source_config = 'beta'",
        r"boundaries are not nonnegative with staging >= after_fact_id_pass "
        r">= after_content_pass for source_config\(s\) 'beta'",
    ),
    "a staging count that does not sum to the staging table": (
        "UPDATE stage_e_dedup_stats SET staging = 5 WHERE source_config = 'alpha'",
        r"stage_e_dedup_stats staging sums to 6, but fact_results_staging has 2 rows",
    ),
}


@pytest.mark.parametrize("corruption", sorted(_CORRUPTIONS))
def test_stage_e_stops_when_its_counts_do_not_conserve(
    tmp_path, monkeypatch, corruption,
):
    from eval_card_backend.canonicalise import stages

    sql, message = _CORRUPTIONS[corruption]
    partition = stages._apply_composite_partitions

    def corrupting_partition(con, **kwargs):
        partition(con, **kwargs)
        con.execute(sql)

    monkeypatch.setattr(stages, "_apply_composite_partitions", corrupting_partition)
    with pytest.raises(stages.StageEInvariantError, match=message):
        _run(tmp_path, monkeypatch, [
            ("alpha", "old.json", _record("ev_shared", "2026-04-30T00:00:00Z", 0.3)),
            ("beta", "new.json", _record("ev_shared", "2026-05-02T00:00:00Z", 0.9)),
        ])


def test_resume_from_a_cache_whose_stats_disagree_with_its_rows_fails(
    tmp_path, monkeypatch,
):
    """A cached `stage_e_dedup_stats` that no longer matches the cached fact
    rows is caught when the snapshot is described, before snapshot_meta.json
    is written."""
    from eval_card_backend.canonicalise import stages
    from eval_card_backend.canonicalise.cache import _snapshot_dir_name

    records = [
        ("alpha", "a.json", _record("ev_a", "2026-04-30T00:00:00Z", 0.3)),
        ("beta", "b.json", _record("ev_b", "2026-05-02T00:00:00Z", 0.9)),
    ]
    out_dir = _run(tmp_path, monkeypatch, records)
    (out_dir / "snapshot_meta.json").unlink()

    stats = tmp_path / "cache" / _snapshot_dir_name(SNAPSHOT_ID) / "stage_e_dedup_stats.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE TABLE s AS SELECT * FROM read_parquet('{stats}')")
    con.execute("UPDATE s SET after_content_pass = 0 WHERE source_config = 'beta'")
    con.execute(f"COPY s TO '{stats}' (FORMAT PARQUET)")

    with pytest.raises(
        stages.StageEInvariantError,
        match="fact_results has 2 rows but stage_e_dedup_stats "
              "after_content_pass sums to 1",
    ):
        _run(tmp_path, monkeypatch, records, from_stage="F")
    assert not (out_dir / "snapshot_meta.json").exists()


def test_resume_from_a_current_cache_without_the_stats_table_is_refused(
    tmp_path, monkeypatch,
):
    from eval_card_backend.canonicalise.cache import _snapshot_dir_name

    records = [("alpha", "a.json", _record("ev_a", "2026-04-30T00:00:00Z", 0.3))]
    _run(tmp_path, monkeypatch, records)
    cache_dir = tmp_path / "cache" / _snapshot_dir_name(SNAPSHOT_ID)
    (cache_dir / "stage_e_dedup_stats.parquet").unlink()

    with pytest.raises(
        RuntimeError,
        match=r"missing E:stage_e_dedup_stats\. .*Re-run from Stage A",
    ):
        _run(tmp_path, monkeypatch, records, from_stage="F")

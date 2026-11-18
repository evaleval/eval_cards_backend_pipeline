"""A reasoning-effort tier or thinking-token budget a source welds onto the
model name becomes the row's protocol point, so runs under different
conditions stay separate results of one model."""
from __future__ import annotations

import duckdb

from tests.test_stage_e_invariants import _record, _run


def _with_model(record: dict, model_id: str) -> dict:
    record["model_info"] = {**record["model_info"], "id": model_id}
    return record


def _rows(out_dir, table: str, columns: str) -> list[tuple]:
    return duckdb.connect().execute(
        f"SELECT {columns} FROM read_parquet('{out_dir / table}') ORDER BY score"
    ).fetchall()


def test_effort_tier_in_the_model_name_is_the_rows_protocol_point(tmp_path, monkeypatch):
    out_dir = _run(tmp_path, monkeypatch, [
        ("minibench", "base.json", _record("ev_base", "2026-05-02T00:00:00Z", 0.5)),
        ("minibench", "low.json", _with_model(
            _record("ev_low", "2026-05-02T00:00:00Z", 0.3), "openai/gpt-4o-low")),
        ("minibench", "high.json", _with_model(
            _record("ev_high", "2026-05-02T00:00:00Z", 0.9), "openai/gpt-4o-high")),
    ])
    facts = _rows(out_dir, "fact_results.parquet",
                  "model_raw, model_id, score, protocol_condition")
    assert facts == [
        ("openai/gpt-4o-low", "openai/gpt-4o", 0.3, '{"reasoning_effort":"low"}'),
        ("openai/gpt-4o", "openai/gpt-4o", 0.5, None),
        ("openai/gpt-4o-high", "openai/gpt-4o", 0.9, '{"reasoning_effort":"high"}'),
    ]
    # One display row per tier: the three scores are not pooled into one.
    view = _rows(out_dir, "eval_results_view.parquet", "model_id, score, protocol_condition")
    assert view == [
        ("openai/gpt-4o", 0.3, '{"reasoning_effort":"low"}'),
        ("openai/gpt-4o", 0.5, None),
        ("openai/gpt-4o", 0.9, '{"reasoning_effort":"high"}'),
    ]


def test_thinking_budget_in_the_model_name_is_the_rows_protocol_point(tmp_path, monkeypatch):
    out_dir = _run(tmp_path, monkeypatch, [
        ("minibench", "budget.json", _with_model(
            _record("ev_budget", "2026-05-02T00:00:00Z", 0.4), "openai/gpt-4o-thinking-16k")),
        ("minibench", "both.json", _with_model(
            _record("ev_both", "2026-05-02T00:00:00Z", 0.6), "openai/gpt-4o-thinking-16k-high")),
    ])
    assert _rows(out_dir, "fact_results.parquet",
                 "model_id, score, protocol_condition") == [
        ("openai/gpt-4o", 0.4, '{"reasoning_tokens":16000}'),
        ("openai/gpt-4o", 0.6, '{"reasoning_effort":"high","reasoning_tokens":16000}'),
    ]

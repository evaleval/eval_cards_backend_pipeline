"""Shared pytest fixtures for the integration test suite.

`pipeline_output` runs the full pipeline once per test session against a
synthetic EEE + ABC fixture under a tmp output directory. Tests then assert
on the resulting JSON files to catch transformation issues.

Fixture scenarios (one config per scenario):

  bench_variant     — 1 model, 3 rows from one org, differing max_tokens →
                      variant_divergence fires; cross_party_divergence null.
  bench_multiorg    — 1 model, 2 orgs, identical setups, divergent scores →
                      cross_party_divergence fires; variant_divergence null
                      (identical setups).
  bench_orgcollapse — 1 model, 2 rows whose org names differ only in
                      whitespace → whitespace normalization collapses to one
                      org → cross_party_divergence null, is_multi_source
                      false. One row also carries
                      `evaluator_relationship: collaborative` to exercise
                      spec §5.4 (collaborative passes through; first_party_only
                      remains false).
  bench_agentic     — 1 model, 1 row with `agentic_eval_config` set but no
                      `eval_plan`/`eval_limits` → reproducibility_gap fires
                      with the agentic-extras in `missing_fields`.
  bench_versioned   — 2 records with versioned raw IDs
                      (openai/gpt-5-2024-01-01-high, ..-low) → both collapse
                      to family_id=openai/gpt-5 with distinct variant_keys.
                      Exercises model identity normalization.
  bench_multiresult — 1 record carrying 3 results in `evaluation_results`,
                      each with its own evaluation_name + generation_args.
                      Verifies per-result iteration (vs the deprecated
                      `evaluation_results[0]`-only read).
  bench_lower_better — 3 records, lower_is_better metric (latency-style).
                      Verifies sort/rank flip and top_score selection.
  bench_setupalias  — 2 records, raw_ids `openai/gpt-5-fc` and
                      `openai/gpt-5-prompt`. Each record sets
                      `model_info.additional_details.mode` to the matching
                      alias (`fc` / `prompt`); the pipeline's
                      `is_setup_alias_mode` triggers on that field, NOT on
                      the suffix in the raw id alone. Both rows collapse to
                      family `openai/gpt-5` with variant_key=default —
                      verifies setup-alias collapsing (commits 6e705ba,
                      9866766).
  bench-fuzzy       — kebab-case EEE benchmark name; ABC card filename is
                      snake-case (`bench_fuzzy.json`). Verifies the kebab→
                      snake normalization in `lookup_benchmark_card`'s
                      candidate-key generation.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _result(
    *,
    benchmark: str,
    score: float,
    generation_args: dict | None,
    evaluation_name: str = "accuracy",
    metric_config: dict | None = None,
) -> dict:
    """Build a single `evaluation_results[]` entry."""
    return {
        "evaluation_name": evaluation_name,
        "source_data": {"dataset_name": benchmark},
        "metric_config": metric_config
        or {
            "evaluation_description": "Accuracy",
            "lower_is_better": False,
            "metric_id": f"{benchmark}.accuracy",
            "metric_name": "accuracy",
            "metric_kind": "score",
            "metric_unit": "proportion",
            "min_score": 0,
            "max_score": 1,
        },
        "score_details": {"score": score},
        "generation_config": (
            {"generation_args": generation_args}
            if generation_args is not None
            else {}
        ),
    }


def _eee_record(
    *,
    benchmark: str,
    record_id: str,
    model_id: str,
    developer: str,
    evaluator_relationship: str,
    source_organization_name: str,
    score: float | None = None,
    generation_args: dict | None = None,
    evaluation_name: str = "accuracy",
    metric_config: dict | None = None,
    results: list[dict] | None = None,
) -> dict:
    """Build an EEE record dict in the shape produced by upstream EEE.

    Either pass `score` + `generation_args` for a single-result record (the
    common case), or pass `results=[...]` for a multi-result record. The
    pipeline iterates `evaluation_results`, so a multi-result record produces
    one row per result.
    """
    if results is None:
        assert score is not None, "Provide score (single-result) or results=[...] (multi-result)"
        results = [
            _result(
                benchmark=benchmark,
                score=score,
                generation_args=generation_args,
                evaluation_name=evaluation_name,
                metric_config=metric_config,
            )
        ]
    return {
        "schema_version": "0.2.2",
        "evaluation_id": f"{benchmark}/{record_id}",
        "retrieved_timestamp": "1700000000",
        "source_metadata": {
            "evaluator_relationship": evaluator_relationship,
            "source_organization_name": source_organization_name,
            "source_type": "leaderboard",
            "source_name": "fixture",
        },
        "model_info": {
            "id": model_id,
            "developer": developer,
            "name": model_id.split("/")[-1],
        },
        "evaluation_results": results,
        "eval_library": {"name": "fixture", "version": "1.0.0"},
    }


def _abc_card(*, name: str, agentic: bool = False) -> dict:
    """Build a minimal ABC card. `agentic=True` adds spec-tasks-literal token."""
    tasks: list[Any] = ["question_answering"]
    if agentic:
        tasks = ["agentic", "tool_use"]
    return {
        "benchmark_details": {
            "name": name,
            "overview": f"Fixture benchmark {name}.",
            "data_type": "text",
            "domains": ["fixture"],
            "languages": ["en"],
            "similar_benchmarks": [],
            "resources": [],
            "benchmark_type": "benchmark",
        },
        "purpose_and_intended_users": {
            "goal": "fixture",
            "audience": ["test"],
            "tasks": tasks,
            "limitations": "",
            "out_of_scope_uses": [],
        },
        "data": {
            "source": "fixture",
            "size": "tiny",
            "format": "json",
            "annotation": "manual",
        },
        "methodology": {"methods": ["fixture"], "metrics": ["accuracy"]},
    }


def _write_fixtures(eee_root: Path, abc_root: Path) -> None:
    """Materialize the EEE + ABC fixture trees to disk."""
    eee_data_root = eee_root / "data"
    abc_cards_root = abc_root / "cards"
    eee_data_root.mkdir(parents=True, exist_ok=True)
    abc_cards_root.mkdir(parents=True, exist_ok=True)

    records: list[tuple[str, str, dict]] = []

    # bench_variant — variant divergence (3 rows, same model, same org,
    # different max_tokens; 0.85 - 0.65 = 0.20 > 0.05 threshold)
    for record_id, max_tokens, score in [
        ("run-1", 2048, 0.65),
        ("run-2", 4096, 0.73),
        ("run-3", 8192, 0.85),
    ]:
        records.append((
            "bench_variant",
            f"openai_gpt-5/{record_id}",
            _eee_record(
                benchmark="bench_variant",
                record_id=record_id,
                model_id="openai/gpt-5",
                developer="openai",
                evaluator_relationship="first_party",
                source_organization_name="Acme Inc",
                score=score,
                generation_args={"temperature": 0.0, "max_tokens": max_tokens},
            ),
        ))

    # bench_multiorg — cross-party divergence (1 model, 2 orgs, identical
    # setups, 0.85 vs 0.55; divergence 0.30 > 0.05)
    for record_id, org, score in [
        ("run-4", "OpenAI", 0.85),
        ("run-5", "Scale AI", 0.55),
    ]:
        records.append((
            "bench_multiorg",
            f"openai_gpt-5/{record_id}",
            _eee_record(
                benchmark="bench_multiorg",
                record_id=record_id,
                model_id="openai/gpt-5",
                developer="openai",
                evaluator_relationship=("first_party" if org == "OpenAI" else "third_party"),
                source_organization_name=org,
                score=score,
                generation_args={"temperature": 0.0, "max_tokens": 2048},
            ),
        ))

    # bench_orgcollapse — whitespace normalization collapses two near-duplicate
    # org names to one, so cross-party divergence does NOT fire. run-7 uses
    # `collaborative` to exercise spec §5.4 (collaborative passes through as
    # source_type, with first_party_only=false).
    for record_id, org, relationship, score in [
        ("run-6", "Acme  Inc", "third_party", 0.70),  # double-space
        ("run-7", "Acme Inc", "collaborative", 0.50),
    ]:
        records.append((
            "bench_orgcollapse",
            f"openai_gpt-5/{record_id}",
            _eee_record(
                benchmark="bench_orgcollapse",
                record_id=record_id,
                model_id="openai/gpt-5",
                developer="openai",
                evaluator_relationship=relationship,
                source_organization_name=org,
                score=score,
                generation_args={"temperature": 0.0, "max_tokens": 2048},
            ),
        ))

    # bench_agentic — single agentic record without eval_plan/eval_limits.
    # Uses ABC tasks-literal to mark agentic; expect missing_fields to include
    # eval_plan + eval_limits.
    records.append((
        "bench_agentic",
        "anthropic_claude/run-8",
        _eee_record(
            benchmark="bench_agentic",
            record_id="run-8",
            model_id="anthropic/claude-opus-4-5",
            developer="anthropic",
            evaluator_relationship="first_party",
            source_organization_name="Anthropic",
            score=0.42,
            generation_args={
                "temperature": 0.0,
                "max_tokens": 4096,
                "agentic_eval_config": {"agent_framework": "smolagents"},
                # eval_plan + eval_limits intentionally absent
            },
        ),
    ))

    # bench_versioned — versioned raw IDs that should collapse to one family
    # but distinct variant_keys via VERSION_SUFFIX_REGEX parsing.
    for record_id, raw_id, score in [
        ("run-9", "openai/gpt-5-2024-01-01-high", 0.90),
        ("run-10", "openai/gpt-5-2024-01-01-low", 0.60),
    ]:
        records.append((
            "bench_versioned",
            f"openai_gpt-5/{record_id}",
            _eee_record(
                benchmark="bench_versioned",
                record_id=record_id,
                model_id=raw_id,
                developer="openai",
                evaluator_relationship="first_party",
                source_organization_name="OpenAI",
                score=score,
                generation_args={"temperature": 0.0, "max_tokens": 2048},
            ),
        ))

    # bench_multiresult — single record with 3 results, each with its own
    # evaluation_name and generation_args. Verifies per-result iteration:
    # the deprecated evaluation_results[0]-only read would silently lose
    # results 1 and 2.
    records.append((
        "bench_multiresult",
        "openai_gpt-5/run-11",
        _eee_record(
            benchmark="bench_multiresult",
            record_id="run-11",
            model_id="openai/gpt-5",
            developer="openai",
            evaluator_relationship="first_party",
            source_organization_name="OpenAI",
            results=[
                _result(
                    benchmark="bench_multiresult",
                    evaluation_name="metric_a",
                    score=0.40,
                    generation_args={"temperature": 0.0, "max_tokens": 1024},
                    metric_config={
                        "lower_is_better": False,
                        "metric_id": "bench_multiresult.metric_a",
                        "metric_name": "metric_a",
                        "metric_kind": "score",
                        "metric_unit": "proportion",
                        "min_score": 0,
                        "max_score": 1,
                    },
                ),
                _result(
                    benchmark="bench_multiresult",
                    evaluation_name="metric_b",
                    score=0.70,
                    generation_args={"temperature": 0.0, "max_tokens": 2048},
                    metric_config={
                        "lower_is_better": False,
                        "metric_id": "bench_multiresult.metric_b",
                        "metric_name": "metric_b",
                        "metric_kind": "score",
                        "metric_unit": "proportion",
                        "min_score": 0,
                        "max_score": 1,
                    },
                ),
                _result(
                    benchmark="bench_multiresult",
                    evaluation_name="metric_c",
                    score=0.55,
                    generation_args={"temperature": 0.0, "max_tokens": 4096},
                    metric_config={
                        "lower_is_better": False,
                        "metric_id": "bench_multiresult.metric_c",
                        "metric_name": "metric_c",
                        "metric_kind": "score",
                        "metric_unit": "proportion",
                        "min_score": 0,
                        "max_score": 1,
                    },
                ),
            ],
        ),
    ))

    # bench_lower_better — lower_is_better metric (e.g., latency). 3 records,
    # 3 distinct models, scores 0.50, 0.30, 0.10. Best = lowest = 0.10 →
    # top_score should be 0.10 and the row at index 0 (rank 1) should
    # carry score 0.10.
    lower_metric_config = {
        "evaluation_description": "Latency",
        "lower_is_better": True,
        "metric_id": "bench_lower_better.latency",
        "metric_name": "latency",
        "metric_kind": "score",
        "metric_unit": "proportion",
        "min_score": 0,
        "max_score": 1,
    }
    for record_id, raw_id, developer, score in [
        ("run-12", "meta/llama-3", "meta", 0.50),
        ("run-13", "google/gemini-3", "google", 0.30),
        ("run-14", "mistral/mistral-large", "mistral", 0.10),
    ]:
        records.append((
            "bench_lower_better",
            f"{developer}/{record_id}",
            _eee_record(
                benchmark="bench_lower_better",
                record_id=record_id,
                model_id=raw_id,
                developer=developer,
                evaluator_relationship="third_party",
                source_organization_name="Latency Lab",
                score=score,
                generation_args={"temperature": 0.0, "max_tokens": 2048},
                metric_config=lower_metric_config,
            ),
        ))

    # bench_setupalias — setup-alias collapse fires when `model_info.
    # additional_details.mode` is a known alias (`fc`, `prompt`, ...).
    # Without that signal the suffix is treated as part of the model name.
    # We patch additional_details onto the post-build record below.
    for record_id, raw_id, mode, score in [
        ("run-15", "openai/gpt-5-fc", "fc", 0.62),
        ("run-16", "openai/gpt-5-prompt", "prompt", 0.71),
    ]:
        record = _eee_record(
            benchmark="bench_setupalias",
            record_id=record_id,
            model_id=raw_id,
            developer="openai",
            evaluator_relationship="first_party",
            source_organization_name="OpenAI",
            score=score,
            generation_args={"temperature": 0.0, "max_tokens": 2048},
        )
        record["model_info"]["additional_details"] = {"mode": mode}
        records.append(("bench_setupalias", f"openai_gpt-5/{record_id}", record))

    # bench-fuzzy — kebab-case EEE benchmark name vs snake-case ABC card
    # filename. Both normalize to the same `bench_fuzzy` lookup key.
    records.append((
        "bench-fuzzy",
        "openai_gpt-5/run-17",
        _eee_record(
            benchmark="bench-fuzzy",
            record_id="run-17",
            model_id="openai/gpt-5",
            developer="openai",
            evaluator_relationship="first_party",
            source_organization_name="OpenAI",
            score=0.55,
            generation_args={"temperature": 0.0, "max_tokens": 2048},
        ),
    ))

    for config, relpath, payload in records:
        out_path = eee_data_root / config / relpath
        out_path = out_path.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ABC cards — names mirror the configs. bench_agentic gets the
    # tasks-literal agentic marker so `is_agentic` fires. bench_fuzzy is
    # named with snake_case while the EEE config uses kebab-case
    # (`bench-fuzzy`); the pipeline's normalization should bridge them.
    for name, agentic in [
        ("bench_variant", False),
        ("bench_multiorg", False),
        ("bench_orgcollapse", False),
        ("bench_agentic", True),
        ("bench_versioned", False),
        ("bench_multiresult", False),
        ("bench_lower_better", False),
        ("bench_setupalias", False),
        ("bench_fuzzy", False),  # EEE side: bench-fuzzy (kebab); ABC: snake
    ]:
        card_path = abc_cards_root / f"{name}.json"
        card_path.write_text(json.dumps(_abc_card(name=name, agentic=agentic), indent=2), encoding="utf-8")


@pytest.fixture(scope="session")
def pipeline_output(tmp_path_factory) -> Path:
    """Run `pipeline.main()` once against fixture data; yield the output dir.

    Session-scoped: each integration test reads from the same output, sharing
    the cost of one pipeline run across all assertions.
    """
    fixture_root = tmp_path_factory.mktemp("evalcards_fixtures")
    eee_root = fixture_root / "eee_dataset"
    abc_root = fixture_root / "abc_metadata"
    output_dir = fixture_root / "output"
    _write_fixtures(eee_root, abc_root)

    from scripts import pipeline as pipeline_module

    saved_output_dir = pipeline_module.OUTPUT_DIR
    saved_argv = list(sys.argv)
    saved_env = {
        key: os.environ.get(key)
        for key in (
            "EEE_LOCAL_DATASET_DIR",
            "BENCHMARK_METADATA_LOCAL_DIR",
            "CONFIGS",
            "CONFIG_NAMES",
            "CONFIG_LIMIT",
            "HF_TOKEN",
        )
    }
    try:
        pipeline_module.OUTPUT_DIR = output_dir
        os.environ["EEE_LOCAL_DATASET_DIR"] = str(eee_root)
        os.environ["BENCHMARK_METADATA_LOCAL_DIR"] = str(abc_root)
        # Pin the configs so discovery doesn't depend on dir iteration order.
        os.environ["CONFIGS"] = ",".join(
            sorted(p.name for p in (eee_root / "data").iterdir() if p.is_dir())
        )
        os.environ.pop("CONFIG_NAMES", None)
        os.environ.pop("CONFIG_LIMIT", None)
        # Avoid any chance of HF upload even if --dry-run logic regresses.
        os.environ.pop("HF_TOKEN", None)
        sys.argv = ["pipeline.py", "--dry-run"]

        rc = pipeline_module.main()
        assert rc == 0, f"pipeline.main() returned non-zero: {rc}"
    finally:
        pipeline_module.OUTPUT_DIR = saved_output_dir
        sys.argv = saved_argv
        for key, val in saved_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    return output_dir

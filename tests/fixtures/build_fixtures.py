"""Materialise the test fixture corpus.

Run via: `uv run python tests/fixtures/build_fixtures.py`

Each EEE record is a single JSON file under
`tests/fixtures/eee/data/<config>/<dev>/<model>/<id>.json` (matches the
upstream EEE_datastore on-disk layout). Each fixture deliberately exercises
one behaviour; together they cover the pipeline's edge cases.

Re-run this script if the fixture set changes — the JSON and parquet
outputs are committed alongside the code so tests don't depend on script
execution at test time.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
EEE = ROOT / "eee" / "data"
CARDS = ROOT / "auto_benchmarkcards" / "cards"
REG = ROOT / "entity_registry"


# ---------------------------------------------------------------------------
# 7 EEE records
# ---------------------------------------------------------------------------


def _eee_record(
    *,
    config: str,
    dev: str,
    model: str,
    eval_id: str,
    model_id: str,
    org: str,
    evaluation_name: str,
    metric_name: str,
    metric_id: str,
    score,                  # may be None for the no-score fixture
    generation_args: dict,
    evaluator_relationship: str = "first_party",
    metric_kind: str | None = None,
    metric_unit: str | None = None,
    score_type: str = "continuous",
    min_score: float = 0.0,
    max_score: float = 1.0,
) -> dict:
    record = {
        "evaluation_id": eval_id,
        "schema_version": "0.2.2",
        "retrieved_timestamp": "2026-04-30T00:00:00Z",
        # Top-level evaluation_timestamp is what frontends should
        # display as "Updated"/eval-date — distinct from
        # retrieved_timestamp (snapshot pipeline scrape time). Stage J
        # views materialise the former; the latter is preserved on
        # fact_results for diagnostics + dedup tie-breaks.
        "evaluation_timestamp": "2026-04-29T00:00:00Z",
        "model_info": {
            "developer": dev, "name": model, "id": model_id,
            "inference_platform": "test",
        },
        "source_metadata": {
            "source_name": org,
            "source_type": "evaluation_run",
            "source_organization_name": org,
            "evaluator_relationship": evaluator_relationship,
        },
        "eval_library": {"name": "minibench", "version": "1.0"},
        "evaluation_results": [
            {
                "evaluation_name": evaluation_name,
                "source_data": {
                    "dataset_name": evaluation_name,
                    "source_type": "other",
                },
                "metric_config": {
                    "metric_id": metric_id,
                    "metric_name": metric_name,
                    "evaluation_description": f"{metric_name} on {evaluation_name}",
                    "metric_kind": metric_kind,
                    "metric_unit": metric_unit,
                    "score_type": score_type,
                    "min_score": min_score,
                    "max_score": max_score,
                    "lower_is_better": False,
                },
                "score_details": {"score": score} if score is not None else {"score": None},
                "generation_config": {"generation_args": generation_args},
            }
        ],
    }
    return record


def write_eee_fixtures():
    """7 hand-built EEE records, one per edge case."""
    if EEE.exists():
        for f in EEE.rglob("*.json"):
            f.unlink()

    fixtures = [
        # 01: clean resolution. All 5 entities resolve via 'exact'; non-agentic.
        # File path: data/<config>/<dev>/<model>/<eval_id>.json
        (
            "fixtures_clean", "openai", "gpt-4o", "01-resolves-cleanly",
            _eee_record(
                config="fixtures_clean", dev="openai", model="GPT-4o",
                eval_id="ev_01", model_id="openai/gpt-4o", org="OpenAI",
                evaluation_name="mmlu", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.85,
                generation_args={"temperature": 0.0, "max_tokens": 1024,
                                 "prompt_template": "default"},
            ),
        ),
        # 02: model no_match — community fine-tune.
        (
            "fixtures_clean", "community", "fine-tune-7b", "02-no-match-model",
            _eee_record(
                config="fixtures_clean", dev="community", model="FineTune7B",
                eval_id="ev_02", model_id="community/fine-tune-7b", org="OpenAI",
                evaluation_name="mmlu", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.40,
                generation_args={"temperature": 0.0, "max_tokens": 1024},
            ),
        ),
        # 03: agentic via card (benchmark card has tasks=['agentic']).
        (
            "fixtures_agentic", "openai", "gpt-4o", "03-agentic-via-card",
            _eee_record(
                config="fixtures_agentic", dev="openai", model="GPT-4o",
                eval_id="ev_03", model_id="openai/gpt-4o", org="OpenAI",
                evaluation_name="swebench-verified", metric_name="Pass@1",
                metric_id="swebench.pass1", score=0.55,
                generation_args={
                    "temperature": 0.0, "max_tokens": 4096,
                    "eval_plan": {"name": "scaffold v1"},
                    "eval_limits": {"time_limit": 3600},
                },
            ),
        ),
        # 04: agentic via generation_args.agentic_eval_config presence.
        (
            "fixtures_agentic", "anthropic", "claude-sonnet", "04-agentic-via-config",
            _eee_record(
                config="fixtures_agentic", dev="anthropic", model="Claude Sonnet",
                eval_id="ev_04", model_id="anthropic/claude-sonnet", org="Anthropic",
                evaluation_name="appworld",  # not in agentic regex; relies on config rule
                metric_name="Score", metric_id="appworld.score", score=0.62,
                generation_args={
                    "temperature": 0.0, "max_tokens": 2048,
                    "agentic_eval_config": {
                        "additional_details": {"loop": "react", "max_steps": "30"},
                    },
                    "eval_plan": {"name": "default"},
                    "eval_limits": {"time_limit": 1800},
                },
            ),
        ),
        # 05: variant divergence — 3 rows same triple, different setups, scores diverge.
        # Realised as 3 separate records under the same model+benchmark+metric.
        (
            "fixtures_variant", "openai", "gpt-4o", "05a-variant-low-temp",
            _eee_record(
                config="fixtures_variant", dev="openai", model="GPT-4o",
                eval_id="ev_05a", model_id="openai/gpt-4o", org="OpenAI",
                evaluation_name="mmlu", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.85,
                generation_args={"temperature": 0.0, "max_tokens": 1024},
            ),
        ),
        (
            "fixtures_variant", "openai", "gpt-4o", "05b-variant-mid-temp",
            _eee_record(
                config="fixtures_variant", dev="openai", model="GPT-4o",
                eval_id="ev_05b", model_id="openai/gpt-4o", org="OpenAI",
                evaluation_name="mmlu", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.50,
                generation_args={"temperature": 0.7, "max_tokens": 1024},
            ),
        ),
        (
            "fixtures_variant", "openai", "gpt-4o", "05c-variant-high-tokens",
            _eee_record(
                config="fixtures_variant", dev="openai", model="GPT-4o",
                eval_id="ev_05c", model_id="openai/gpt-4o", org="OpenAI",
                evaluation_name="mmlu", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.78,
                generation_args={"temperature": 0.0, "max_tokens": 4096},
            ),
        ),
        # 06: cross-party divergence — same triple, 2 orgs (case/whitespace
        # variant exercises normalisation), divergent scores.
        (
            "fixtures_xparty", "openai", "gpt-4o", "06a-xparty-openai",
            _eee_record(
                config="fixtures_xparty", dev="openai", model="GPT-4o",
                eval_id="ev_06a", model_id="openai/gpt-4o",
                org="OpenAI",   # canonical casing
                evaluation_name="mmlu", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.85,
                generation_args={"temperature": 0.0, "max_tokens": 1024},
            ),
        ),
        (
            "fixtures_xparty", "openai", "gpt-4o", "06b-xparty-thirdparty",
            _eee_record(
                config="fixtures_xparty", dev="openai", model="GPT-4o",
                eval_id="ev_06b", model_id="openai/gpt-4o",
                org="Scale AI ",   # different org + trailing whitespace edge case
                evaluation_name="mmlu", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.65,
                evaluator_relationship="third_party",
                generation_args={"temperature": 0.0, "max_tokens": 1024},
            ),
        ),
        # 07: no-score row — should be DROPPED in Stage E and counted in
        # `dropped_rows_no_score`. Every other field is sensible.
        (
            "fixtures_clean", "openai", "gpt-4o", "07-no-score",
            _eee_record(
                config="fixtures_clean", dev="openai", model="GPT-4o",
                eval_id="ev_07", model_id="openai/gpt-4o", org="OpenAI",
                evaluation_name="mmlu", metric_name="Accuracy",
                metric_id="mmlu.acc", score=None,
                generation_args={"temperature": 0.0, "max_tokens": 1024},
            ),
        ),
        # 08a, 08b, 08c — slice fixture. Three eval_results all resolving
        # to benchmark_id=mmlu via the registry: "Anatomy" + "Astronomy"
        # (≥2 distinct cleaned raws → Stage C populates slice_key) plus a
        # second org reporting on Anatomy (cross-party at the slice
        # level). Exercises:
        #  - Stage C: slice_key/slice_name populate; case-normalized.
        #  - Stage F.1: is_multi_source=True on ALL rows because the
        #    (gpt-4o, mmlu) pair has 2 distinct orgs total — even on the
        #    Astronomy row where only OpenAI reported. Slice/metric-agnostic.
        #  - Stage F.2: comparability_group_id differs between Anatomy
        #    and Astronomy rows (slice in the md5 input).
        (
            "fixtures_slices", "openai", "gpt-4o", "08a-anatomy-openai",
            _eee_record(
                config="fixtures_slices", dev="openai", model="GPT-4o",
                eval_id="ev_08a", model_id="openai/gpt-4o", org="OpenAI",
                evaluation_name="Anatomy", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.85,
                generation_args={"temperature": 0.0, "max_tokens": 1024},
            ),
        ),
        (
            "fixtures_slices", "openai", "gpt-4o", "08b-anatomy-thirdparty",
            _eee_record(
                config="fixtures_slices", dev="openai", model="GPT-4o",
                eval_id="ev_08b", model_id="openai/gpt-4o", org="Scale AI",
                evaluation_name="Anatomy", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.62,
                evaluator_relationship="third_party",
                generation_args={"temperature": 0.0, "max_tokens": 1024},
            ),
        ),
        (
            "fixtures_slices", "openai", "gpt-4o", "08c-astronomy-openai",
            _eee_record(
                config="fixtures_slices", dev="openai", model="GPT-4o",
                eval_id="ev_08c", model_id="openai/gpt-4o", org="OpenAI",
                evaluation_name="Astronomy", metric_name="Accuracy",
                metric_id="mmlu.acc", score=0.78,
                generation_args={"temperature": 0.0, "max_tokens": 1024},
            ),
        ),
    ]

    for cfg, dev, model, fname, record in fixtures:
        out_dir = EEE / cfg / dev / model
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{fname}.json").write_text(json.dumps(record, indent=2))


# ---------------------------------------------------------------------------
# 2 AutoBenchmarkCards
# ---------------------------------------------------------------------------


def write_card_fixtures():
    """mmlu.json: standard non-agentic card. swebench-verified.json: agentic
    via tasks tag. Fixture 02's benchmark resolves to mmlu so it shares the
    card; fixture 04's benchmark (appworld) has no card — exercises the
    `card_present = false` path."""
    CARDS.mkdir(parents=True, exist_ok=True)

    mmlu = {
        "benchmark_card": {
            "benchmark_details": {
                "name": "MMLU",
                "overview": "Massive Multitask Language Understanding",
                "data_type": "qa",
                "domains": ["test"],
                "languages": ["en"],
                "similar_benchmarks": ["bbh"],
                "resources": ["paper"],
            },
            "purpose_and_intended_users": {
                "goal": "evaluate broad knowledge",
                "audience": ["devs"],
                "tasks": ["qa"],
                "limitations": "training-set leakage",
                "out_of_scope_uses": ["medical advice"],
            },
            "data": {
                "source": "scraped", "size": "14k",
                "format": "json", "annotation": "manual",
            },
            "methodology": {
                "methods": ["multiple-choice"], "metrics": ["accuracy"],
                "calculation": "n_correct / n_total",
                "interpretation": "higher is better",
                "baseline_results": "0.25 (random)",
                "validation": "manual review",
            },
            "ethical_and_legal_considerations": {
                "privacy_and_anonymity": "n/a",
                "data_licensing": "MIT",
                "consent_procedures": "n/a",
                "compliance_with_regulations": "n/a",
            },
        }
    }
    (CARDS / "mmlu.json").write_text(json.dumps(mmlu, indent=2))

    swebench = {
        "benchmark_card": {
            "benchmark_details": {
                "name": "SWE-bench Verified", "overview": "Coding tasks",
                "data_type": "code",
                "domains": ["software"], "languages": ["en"],
                "similar_benchmarks": [],
                "resources": [],
            },
            "purpose_and_intended_users": {
                "goal": "agentic SWE",
                "audience": ["devs"],
                "tasks": ["agentic"],   # → triggers is_agentic via card rule
                "limitations": "limited to GitHub issues",
                "out_of_scope_uses": [],
            },
            "data": {
                "source": "github", "size": "500",
                "format": "patch", "annotation": "auto",
            },
            "methodology": {
                "methods": ["execute-and-test"], "metrics": ["pass@1"],
                "calculation": "n_pass / n_total",
                "interpretation": "higher is better",
                "baseline_results": "0",
                "validation": "ci",
            },
            "ethical_and_legal_considerations": {
                "privacy_and_anonymity": "n/a",
                "data_licensing": "MIT",
                "consent_procedures": "n/a",
                "compliance_with_regulations": "n/a",
            },
        }
    }
    (CARDS / "swebench-verified.json").write_text(json.dumps(swebench, indent=2))

    # NB: no card for appworld → fixture 04 exercises card-missing path.


# ---------------------------------------------------------------------------
# Minimal registry parquets
# ---------------------------------------------------------------------------


def write_registry_fixtures():
    REG.mkdir(parents=True, exist_ok=True)

    # aliases — ~30 entries; deliberately MISSING entry for fixture 02's model
    # (community/fine-tune-7b) so resolver returns no_match for it.
    aliases = [
        # Models
        {"raw_value": "openai/gpt-4o", "entity_type": "model",
         "canonical_id": "openai/gpt-4o"},
        {"raw_value": "anthropic/claude-sonnet", "entity_type": "model",
         "canonical_id": "anthropic/claude-sonnet"},
        # Benchmarks
        {"raw_value": "mmlu", "entity_type": "benchmark", "canonical_id": "mmlu"},
        # Slice-bearing aliases — multiple raws collapse to mmlu so the
        # fixtures_slices config exercises Stage C's multi-raw heuristic.
        {"raw_value": "Anatomy",   "entity_type": "benchmark", "canonical_id": "mmlu"},
        {"raw_value": "Astronomy", "entity_type": "benchmark", "canonical_id": "mmlu"},
        {"raw_value": "swebench-verified", "entity_type": "benchmark",
         "canonical_id": "swebench-verified"},
        {"raw_value": "appworld", "entity_type": "benchmark",
         "canonical_id": "appworld"},
        # Metrics
        {"raw_value": "Accuracy", "entity_type": "metric", "canonical_id": "accuracy"},
        {"raw_value": "Pass@1", "entity_type": "metric", "canonical_id": "pass-at-1"},
        {"raw_value": "Score", "entity_type": "metric", "canonical_id": "score"},
        # Orgs
        {"raw_value": "OpenAI", "entity_type": "org", "canonical_id": "openai"},
        {"raw_value": "Anthropic", "entity_type": "org", "canonical_id": "anthropic"},
        {"raw_value": "Scale AI", "entity_type": "org", "canonical_id": "scale-ai"},
        # Harnesses
        {"raw_value": "minibench 1.0", "entity_type": "harness",
         "canonical_id": "minibench"},
    ]
    pd.DataFrame([
        {"id": str(i), "raw_value": a["raw_value"],
         "entity_type": a["entity_type"], "canonical_id": a["canonical_id"],
         "source_config": None, "source_field": None,
         "status": "active", "strategy": "exact", "confidence": 1.0,
         "notes": None, "created_at": "", "updated_at": ""}
        for i, a in enumerate(aliases, 1)
    ]).to_parquet(REG / "aliases.parquet")

    # canonical_orgs
    pd.DataFrame([
        {"id": "openai", "display_name": "OpenAI", "parent_org_id": None,
         "website": "https://openai.com", "hf_org": "openai", "kind": "company",
         "tags": "[]", "metadata": "{}", "review_status": "reviewed",
         "created_at": "", "updated_at": ""},
        {"id": "anthropic", "display_name": "Anthropic", "parent_org_id": None,
         "website": "https://anthropic.com", "hf_org": "anthropic", "kind": "company",
         "tags": "[]", "metadata": "{}", "review_status": "reviewed",
         "created_at": "", "updated_at": ""},
        {"id": "scale-ai", "display_name": "Scale AI", "parent_org_id": None,
         "website": "https://scale.com", "hf_org": "scale", "kind": "company",
         "tags": "[]", "metadata": "{}", "review_status": "reviewed",
         "created_at": "", "updated_at": ""},
    ]).to_parquet(REG / "canonical_orgs.parquet")

    # canonical_models — exercises release_date, params_billions,
    # open_weights, and the typed `parents` JSON list (variant edge for
    # Claude Sonnet → Claude family).
    pd.DataFrame([
        {"id": "openai/gpt-4o", "display_name": "GPT-4o", "developer": "OpenAI",
         "org_id": "openai", "family": "GPT-4", "architecture": "transformer",
         "params_billions": None, "parents": "[]",
         "root_model_id": None, "lineage_origin_org_id": "openai",
         "open_weights": False, "release_date": "2024-05",
         "tags": "[]", "metadata": "{}", "review_status": "reviewed",
         "created_at": "", "updated_at": ""},
        {"id": "anthropic/claude-sonnet", "display_name": "Claude Sonnet",
         "developer": "Anthropic", "org_id": "anthropic", "family": "Claude",
         "architecture": "transformer", "params_billions": None,
         "parents": '[{"id": "anthropic/claude", "relationship": "variant"}]',
         "root_model_id": None, "lineage_origin_org_id": "anthropic",
         "open_weights": False, "release_date": "2024-06-20",
         "tags": "[]", "metadata": "{}", "review_status": "reviewed",
         "created_at": "", "updated_at": ""},
    ]).to_parquet(REG / "canonical_models.parquet")

    # canonical_benchmarks
    pd.DataFrame([
        {"id": "mmlu", "display_name": "MMLU",
         "description": "Multitask language understanding", "dataset_repo": None,
         "parent_benchmark_id": None, "tags": "[]", "metadata": "{}",
         "review_status": "reviewed", "created_at": "", "updated_at": ""},
        {"id": "swebench-verified", "display_name": "SWE-bench Verified",
         "description": "Coding tasks", "dataset_repo": None,
         "parent_benchmark_id": None, "tags": "[]", "metadata": "{}",
         "review_status": "reviewed", "created_at": "", "updated_at": ""},
        {"id": "appworld", "display_name": "AppWorld",
         "description": "Agent benchmark", "dataset_repo": None,
         "parent_benchmark_id": None, "tags": "[]", "metadata": "{}",
         "review_status": "reviewed", "created_at": "", "updated_at": ""},
    ]).to_parquet(REG / "canonical_benchmarks.parquet")

    # canonical_metrics — covering proportion (with min/max), score (no unit),
    # pass-at-1 (proportion).
    pd.DataFrame([
        {"id": "accuracy", "display_name": "Accuracy",
         "score_type": "continuous", "lower_is_better": False,
         "min_score": 0.0, "max_score": 1.0, "metadata": "{}",
         "review_status": "reviewed", "created_at": "", "updated_at": ""},
        {"id": "pass-at-1", "display_name": "Pass@1",
         "score_type": "continuous", "lower_is_better": False,
         "min_score": 0.0, "max_score": 1.0, "metadata": "{}",
         "review_status": "reviewed", "created_at": "", "updated_at": ""},
        {"id": "score", "display_name": "Score",
         "score_type": "continuous", "lower_is_better": False,
         "min_score": 0.0, "max_score": 1.0, "metadata": "{}",
         "review_status": "reviewed", "created_at": "", "updated_at": ""},
    ]).to_parquet(REG / "canonical_metrics.parquet")

    # eval_harnesses
    pd.DataFrame([
        {"id": "minibench", "display_name": "MiniBench Harness", "version": "1.0",
         "fork_url": None, "metadata": "{}", "review_status": "reviewed",
         "created_at": "", "updated_at": ""},
    ]).to_parquet(REG / "eval_harnesses.parquet")


def main():
    write_eee_fixtures()
    write_card_fixtures()
    write_registry_fixtures()
    print(f"Wrote fixtures to {ROOT}")


if __name__ == "__main__":
    main()

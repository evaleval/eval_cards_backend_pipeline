"""Regression testbed for the registry resolver.

Maintains an explicit list of (raw_value, entity_type, expected_canonical_id)
expectations covering:
  - Current resolutions we want to KEEP working (regressions = bad)
  - Future resolutions we WANT to enable (currently failing = work-in-progress)

Run before every registry change (baseline) and after every registry change
(verify). Any resolution that goes from CORRECT → INCORRECT is a regression.
Any resolution that goes from FAILING → CORRECT is progress.

Usage:
  uv run --with duckdb --with pyyaml --with pandas --with pyarrow \\
    --with 'eval-entity-resolver @ git+https://github.com/evaleval/evalcard-registry.git#subdirectory=packages/eval-entity-resolver' \\
    --no-project python scripts/test_resolver_regressions.py

Set REGISTRY_LOCAL_PARQUET_DIR to point at the local fixtures (default:
.cache/local_registry_with_models/). For testing against the official HF
dataset, unset REGISTRY_LOCAL_PARQUET_DIR.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

# Default to the registry repo's fixtures (where seed CLI writes when
# run with --local). The previous default was the now-stale
# .cache/local_registry_with_models/ — using that gives misleading
# pass counts because it lacks aliases added in later sessions.
DEFAULT_REGISTRY_DIR = Path("/Users/jchim/projects/evaleval/evalcard-registry/fixtures")
LOCAL_DIR = Path(os.environ.get("REGISTRY_LOCAL_PARQUET_DIR") or str(DEFAULT_REGISTRY_DIR)).resolve()
os.environ["REGISTRY_LOCAL_PARQUET_DIR"] = str(LOCAL_DIR)


# ============================================================================
# Expectation table: (raw_value, entity_type, expected_canonical_id, note)
# expected_canonical_id can be None to assert "should NOT resolve to anything"
# ============================================================================

EXPECTATIONS: list[tuple[str, str, str | None, str]] = [
    # ---------------- Models — confirmed working today ----------------
    # Canonical → canonical (sanity check)
    ("meta/llama-3.1-8b", "model", "meta/llama-3.1-8b", "canonical id resolves to itself"),
    ("anthropic/claude-3.5-sonnet", "model", "anthropic/claude-3.5-sonnet", "canonical id resolves to itself"),

    # Developer-prefix variants
    ("deepseek-ai/deepseek-v3", "model", "deepseek/deepseek-v3", "deepseek-ai → deepseek"),
    ("deepseek/deepseek-v3", "model", "deepseek/deepseek-v3", "short-prefix is canonical"),
    ("deepseek-ai/deepseek-r1", "model", "deepseek/deepseek-r1", "deepseek-ai r1"),
    # Phase 1.3 (Session 5h): Mistral Large 2407 is now its own canonical
    # (Large 2, 123B), distinct from Mistral Large 1 (2402) and Large 2.1 (2411).
    ("mistralai/mistral-large-2407", "model", "mistralai/mistral-large-2407", "Mistral Large 2 (2407) snapshot is its own canonical"),
    ("mistral/mistral-large-2407", "model", "mistralai/mistral-large-2407", "short-prefix mistral → mistralai/mistral-large-2407"),
    ("ai21-labs/jamba-1.5-mini", "model", "ai21/jamba-1.5-mini", "ai21-labs → ai21"),
    ("ai21/jamba-1.5-mini", "model", "ai21/jamba-1.5-mini", "ai21 short-prefix"),
    ("unknown/granite-4-0-h-small", "model", "ibm-granite/granite-4.0-h-small", "unknown sentinel → ibm-granite"),
    ("ibm/granite-4.0-h-small", "model", "ibm-granite/granite-4.0-h-small", "ibm/ → ibm-granite/"),

    # Qwen / Alibaba / Aliyun cluster
    ("qwen/qwen2-72b-instruct", "model", "alibaba/qwen2-72b", "qwen → alibaba"),
    ("alibaba/qwen2-72b-instruct", "model", "alibaba/qwen2-72b", "alibaba self"),
    ("qwen/qwen3-235b-a22b-instruct-2507", "model", "alibaba/qwen3-235b-a22b", "Qwen3-235b 2507 snapshot"),
    ("aliyun/qwen3-next-80b-a3b-thinking", "model", "alibaba/qwen3-next-80b-a3b", "aliyun thinking variant"),

    # Moonshot / Kimi cluster
    ("moonshotai/kimi-k2-instruct", "model", "moonshotai/kimi-k2", "instruct → base"),
    ("moonshot-ai/kimi-k2-instruct", "model", "moonshotai/kimi-k2", "moonshot-ai → moonshotai"),
    ("kimi/kimi-k2-5", "model", "moonshotai/kimi-k2.5", "bare kimi → moonshotai"),

    # Anthropic Claude family
    ("anthropic/claude-3-5-sonnet", "model", "anthropic/claude-3.5-sonnet", "claude-3-5 dash → 3.5 dot"),
    ("anthropic/claude-35-sonnet", "model", "anthropic/claude-3.5-sonnet", "claude-35 token → 3.5"),
    ("anthropic/claude-3-haiku", "model", "anthropic/claude-3-haiku", "haiku canonical"),
    ("anthropic/claude-2-1", "model", "anthropic/claude-2.1", "claude-2-1 → 2.1"),

    # Google Gemma
    ("google/gemma3-4b", "model", "google/gemma-3-4b", "gemma3 (no dash) → gemma-3"),
    ("google/gemma-3-4b", "model", "google/gemma-3-4b", "gemma-3 dash"),

    # Aleph Alpha
    ("aleph-alpha/luminous-base-13b", "model", "aleph-alpha/luminous-base", "with size suffix"),
    ("alephalpha/luminous-base", "model", "aleph-alpha/luminous-base", "no-dash dev, no size"),

    # zai
    ("z-ai/glm-4-5", "model", "zai/glm-4.5", "z-ai → zai, dash → dot"),
    ("zai/glm-4-5", "model", "zai/glm-4.5", "zai dash variant"),

    # tii Falcon
    ("tii-uae/falcon3-7b-instruct", "model", "tiiuae/falcon3-7b", "tii-uae → tiiuae"),
    ("tii-uae/falcon3-1b-instruct", "model", "tiiuae/falcon3-1b", "1b new canonical"),

    # OpenChat
    ("openchat/openchat-35", "model", "openchat/openchat-3.5", "openchat-35 → 3.5"),
    ("openchat/openchat-3-5", "model", "openchat/openchat-3.5", "openchat-3-5 → 3.5"),

    # Microsoft Phi-4 — reasoning-plus is its own canonical (Session 5g split per
    # Microsoft model card; distinct fine-tune from base phi-4).
    ("microsoft/phi4-reasoning-plus", "model", "microsoft/phi-4-reasoning-plus", "phi4 → phi-4-reasoning-plus"),
    ("microsoft/phi-4-reasoning-plus", "model", "microsoft/phi-4-reasoning-plus", "phi-4-reasoning-plus is distinct canonical"),

    # ---------------- Cross-org models we want to ENABLE next ----------------
    # These should currently FAIL — we'll add them in Tier 3 round
    ("qwen/qwq-32b", "model", "alibaba/qwq-32b", "TODO: qwq-32b currently spans 4 orgs"),
    ("anthropic/claude-4-opus", "model", "anthropic/claude-opus-4", "TODO: claude-4-opus naming variant"),
    ("anthropic/claude-4-5-sonnet", "model", "anthropic/claude-sonnet-4.5", "TODO: 4-5 → 4.5 collapse"),
    ("xai/grok-4-fast-reasoning", "model", "xai/grok-4", "TODO: grok-4 fast variant"),
    ("xiaomi/mimo-v2-flash", "model", "xiaomi/mimo-v2-flash", "Flash and Pro split per size-matters policy 2026-05-01"),
    ("xiaomi/mimo-v2-pro", "model", "xiaomi/mimo-v2-pro", "Flash and Pro split per size-matters policy 2026-05-01"),
    # Phase 1.6 (Session 5h): Nemotron Nano 9B v2 is now its own canonical (vendor-branded).
    ("nvidia/nvidia-nemotron-nano-9b-v2", "model", "nvidia/nvidia-nemotron-nano-9b-v2", "Nemotron Nano 9B v2 is its own canonical (vendor branding)"),
    ("mistral/magistral-medium", "model", "mistralai/magistral-medium", "Magistral Medium is its own canonical (no separate magistral family yet)"),
    ("eleutherai/pythia-6-9b", "model", "eleutherai/pythia-6.9b", "TODO: eleutherai pythia"),
    ("eleutherai/pythia-12b", "model", "eleutherai/pythia-12b", "TODO: eleutherai pythia 12b"),
    ("deepseek/deepseek-v3-2-speciale", "model", "deepseek/deepseek-v3.2", "TODO: V3.2-Speciale variant"),

    # ---------------- Things that should NOT resolve (negative tests) ----------------
    # Mistral 7B is its own product, not a "Mistral" family member
    ("mistralai/mistral-7b", "model", "mistralai/mistral-7b", "mistral-7b is own canonical, do NOT collapse to mistral"),
    # Llama-2-7b vs 13b vs 70b are different sizes, registry keeps separate
    ("meta/llama-2-7b", "model", "meta/llama-2-7b", "llama-2-7b stays distinct from 13b/70b"),
    # Qwen vs alibaba/qwen2 — base 7b should stay
    ("alibaba/qwen2-7b", "model", "alibaba/qwen2-7b", "qwen2-7b stays distinct from -72b"),
    # llama-4 scout vs maverick — registry keeps separate (despite earlier my-spike collapse)
    # Currently I have both lumped under meta/llama-4 in my YAML; this should be revised
    # under "different products" rule per audit.
    ("meta/llama-4-scout-17b-16e-instruct", "model", "meta/llama-4-scout-17b-16e",
     "Scout/maverick stay distinct per size-matters policy 2026-05-01"),

    # ---------------- Benchmarks — sanity ----------------
    ("MMLU", "benchmark", "mmlu", "benchmark MMLU"),
    ("mmlu", "benchmark", "mmlu", "benchmark mmlu lower"),
    ("GPQA", "benchmark", "gpqa", "benchmark gpqa"),
    ("IFEval", "benchmark", "ifeval", "benchmark ifeval"),
    ("MMLU-Pro", "benchmark", "mmlu-pro", "benchmark mmlu-pro"),

    # AA benchmarks — should now resolve. The corpus uses dot-prefixed names
    # like "artificial_analysis.gpqa" as benchmark_leaf_name; per-benchmark
    # canonicals exist via aliases on existing entries (gpqa, mmlu-pro, etc.)
    # plus new canonicals (intelligence-index, etc.).
    ("artificial_analysis.mmlu_pro", "benchmark", "mmlu-pro",
     "AA-rebroadcast of MMLU-Pro"),
    ("artificial_analysis.gpqa", "benchmark", "gpqa",
     "AA-rebroadcast of GPQA"),
    ("artificial_analysis.scicode", "benchmark", "scicode",
     "AA-rebroadcast of SciCode"),
    ("artificial_analysis.livecodebench", "benchmark", "livecodebench",
     "AA-rebroadcast of LiveCodeBench"),
    ("artificial_analysis.tau2", "benchmark", "tau2-bench",
     "AA's tau2 → tau2-bench"),
    ("artificial_analysis.hle", "benchmark", "hle",
     "AA HLE — new canonical"),
    # Phase 1.5 (Session 5h): aime-25 renamed to aime-2025 for year-explicit canonicals.
    ("artificial_analysis.aime_25", "benchmark", "aime-2025",
     "AA AIME 2025 — year-explicit canonical (renamed from aime-25)"),
    ("artificial_analysis.artificial_analysis_intelligence_index", "benchmark",
     "artificial-analysis-intelligence-index",
     "AA composite — new canonical"),
    ("artificial_analysis.artificial_analysis_coding_index", "benchmark",
     "artificial-analysis-coding-index",
     "AA coding composite"),
    ("artificial_analysis.artificial_analysis_math_index", "benchmark",
     "artificial-analysis-math-index",
     "AA math composite"),
]


def main() -> None:
    from eval_entity_resolver import AliasStore, Resolver

    local_dir = os.environ.get("REGISTRY_LOCAL_PARQUET_DIR")
    if local_dir:
        store = AliasStore.from_parquet(local_dir, read_only=True)
        print(f"Loaded {len(store.to_dataframe())} alias rows from local: {local_dir}")
    else:
        store = AliasStore.from_hf("evaleval/entity-registry-data", read_only=True)
        print(f"Loaded {len(store.to_dataframe())} alias rows from HF dataset")
    resolver = Resolver(store)

    print(f"\nRunning {len(EXPECTATIONS)} expectation tests...\n")

    counts: Counter[str] = Counter()
    failures: list[tuple] = []
    successes: list[tuple] = []

    for raw, etype, expected, note in EXPECTATIONS:
        result = resolver.resolve(raw, etype, None)
        actual = result.canonical_id

        if actual == expected:
            counts["correct"] += 1
            successes.append((raw, etype, expected, actual, note))
        elif expected is None and actual is not None:
            counts["unexpected_resolution"] += 1
            failures.append(("UNEXPECTED_RESOLVE", raw, etype, expected, actual, note))
        elif expected is not None and actual is None:
            counts["fails_to_resolve"] += 1
            failures.append(("FAILS_TO_RESOLVE", raw, etype, expected, actual, note))
        else:
            counts["wrong_resolution"] += 1
            failures.append(("WRONG", raw, etype, expected, actual, note))

    total = sum(counts.values())
    print(f"Result: {counts['correct']}/{total} pass")
    print(f"  correct:               {counts['correct']}")
    print(f"  fails_to_resolve:      {counts['fails_to_resolve']}")
    print(f"  wrong_resolution:      {counts['wrong_resolution']}")
    print(f"  unexpected_resolution: {counts['unexpected_resolution']}")
    print()

    if failures:
        print(f"=== {len(failures)} failures (these are the gap to close) ===")
        for kind, raw, etype, expected, actual, note in failures:
            print(f"  [{kind}] {raw!r} ({etype})")
            print(f"      expected: {expected!r}")
            print(f"      actual:   {actual!r}")
            print(f"      note:     {note}")
            print()


if __name__ == "__main__":
    main()

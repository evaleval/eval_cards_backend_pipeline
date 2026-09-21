"""Re-derive `registry/scoring_modes.yaml` from the harness's own result dumps.

The HF Open LLM Leaderboard v2 does not put `output_type` in the EEE record —
it lives in the leaderboard's per-model dumps (HF dataset
`open-llm-leaderboard/results`, at `configs[<task>].output_type`). The harness
config is fixed per task, so the mapping is a property of the benchmark, not of
any one model, and a sample settles it.

This script samples dumps, groups their task configs onto the benchmark ids the
warehouse uses, and reports what it found. It FAILS rather than writing when
sampled dumps disagree about a benchmark: a mapping that is only usually true
is worse than none, because consumers cannot see the exception.

    uv run python scripts/derive_scoring_modes.py            # report
    uv run python scripts/derive_scoring_modes.py --check    # CI: diff vs file

Never edit the mapping by hand from a benchmark's name. helm_* runs multiple
choice by generating the answer letter and is generative; only the dumps know.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = REPO_ROOT / "src" / "eval_card_backend" / "registry" / "scoring_modes.yaml"

RESULTS_REPO = "open-llm-leaderboard/results"
COMPOSITE_SLUG = "hf-open-llm-v2"

# lm-eval task prefix -> the benchmark_id the warehouse uses. Kept explicit:
# the warehouse's ids are its own, and a fuzzy match here would silently
# mis-file a task the day the leaderboard adds one.
TASK_PREFIX_TO_BENCHMARK = {
    "leaderboard_bbh": "bbh",
    "leaderboard_gpqa": "gpqa",
    "leaderboard_musr": "musr",
    "leaderboard_mmlu_pro": "mmlu-pro",
    "leaderboard_ifeval": "ifeval",
    "leaderboard_math": "math-level-5",
}

LOG_PROB_OUTPUT_TYPES = {"multiple_choice", "loglikelihood", "loglikelihood_rolling"}
GENERATIVE_OUTPUT_TYPES = {"generate_until", "generate", "generation"}


def benchmark_for_task(task: str) -> str | None:
    # Longest prefix wins so `leaderboard_mmlu_pro` never lands under a
    # shorter `leaderboard_mmlu` if one is ever added.
    best: str | None = None
    for prefix, benchmark in TASK_PREFIX_TO_BENCHMARK.items():
        if task.startswith(prefix) and (best is None or len(prefix) > len(best[0])):  # type: ignore[index]
            best = (prefix, benchmark)  # type: ignore[assignment]
    return best[1] if best else None  # type: ignore[index]


def mode_for_output_type(output_type: str | None) -> str | None:
    if not output_type:
        return None
    value = output_type.strip().lower()
    if value in LOG_PROB_OUTPUT_TYPES:
        return "log_prob"
    if value in GENERATIVE_OUTPUT_TYPES:
        return "generative"
    return None


def sample_dumps(limit: int, seed: int) -> list[str]:
    api = HfApi()
    files = [f for f in api.list_repo_files(RESULTS_REPO, repo_type="dataset") if f.endswith(".json")]
    random.seed(seed)
    return random.sample(files, min(limit, len(files)))


def derive(limit: int, seed: int) -> tuple[dict[str, dict[str, object]], int]:
    observed: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    dumps_read = 0

    for name in sample_dumps(limit, seed):
        try:
            path = hf_hub_download(RESULTS_REPO, name, repo_type="dataset")
            payload = json.loads(Path(path).read_text())
        except Exception as exc:  # a single unreadable dump must not end the run
            print(f"  skipped {name}: {type(exc).__name__}", file=sys.stderr)
            continue
        configs = payload.get("configs") or {}
        if not configs:
            continue
        dumps_read += 1
        for task, config in configs.items():
            benchmark = benchmark_for_task(task)
            if benchmark is None:
                continue
            mode = mode_for_output_type(config.get("output_type"))
            if mode is None:
                continue
            observed[benchmark]["mode"].add(mode)
            observed[benchmark]["tasks"].add(task)
            few = config.get("num_fewshot")
            if few is not None:
                observed[benchmark]["num_fewshot"].add(few)

    disagreements = {b: sorted(v["mode"]) for b, v in observed.items() if len(v["mode"]) > 1}
    if disagreements:
        raise SystemExit(
            "Sampled dumps disagree about the scoring mode; refusing to write a "
            f"mapping that is only usually true: {disagreements}"
        )

    entries: dict[str, dict[str, object]] = {}
    for benchmark, values in sorted(observed.items()):
        tasks = sorted(values["tasks"])
        shared = tasks[0] if len(tasks) == 1 else _common_prefix(tasks) + "*"
        few = sorted(values["num_fewshot"])
        entries[benchmark] = {
            "mode": next(iter(values["mode"])),
            "harness_tasks": shared,
            "num_fewshot": few[0] if len(few) == 1 else few,
        }
    return entries, dumps_read


def _common_prefix(values: list[str]) -> str:
    first, last = values[0], values[-1]
    i = 0
    while i < min(len(first), len(last)) and first[i] == last[i]:
        i += 1
    return first[:i]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25, help="dumps to sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check", action="store_true", help="exit non-zero if the file is stale")
    args = parser.parse_args()

    entries, dumps_read = derive(args.limit, args.seed)
    print(f"read {dumps_read} dumps; derived {len(entries)} benchmarks")
    for benchmark, entry in entries.items():
        print(f"  {benchmark:14s} {entry['mode']:10s} {entry['harness_tasks']}")

    current = yaml.safe_load(MAPPING_PATH.read_text()) if MAPPING_PATH.exists() else {}
    on_file = ((current.get("sources") or {}).get(COMPOSITE_SLUG) or {}).get("benchmarks") or {}
    drift = {
        b: (on_file.get(b, {}).get("mode"), e["mode"])
        for b, e in entries.items()
        if on_file.get(b, {}).get("mode") != e["mode"]
    }
    if drift:
        print(f"\nDRIFT vs {MAPPING_PATH.name}: {drift}", file=sys.stderr)
        return 1 if args.check else 0
    print(f"\n{MAPPING_PATH.name} agrees with the dumps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

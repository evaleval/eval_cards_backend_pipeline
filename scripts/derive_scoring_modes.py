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

`--check` only ever says the file is vouched for when the sample was large
enough to vouch for it: every expected benchmark derived from its own floor of
separate dumps, no entry on file that the dumps did not produce, and no output
type the harness has renamed under us. Silence from a sample that read nothing
is not agreement.

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
from typing import NamedTuple

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

# What a complete derivation looks like. A sample that produced anything else
# has not seen the leaderboard, and its silence about the rest is not evidence.
EXPECTED_BENCHMARKS = frozenset(TASK_PREFIX_TO_BENCHMARK.values())

# The claim is that a task's configuration is constant across an independent
# sample, so the floor is per benchmark: this many separate dumps have to have
# reported each one. A whole-sample count would let nine dumps of unrelated
# tasks carry one dump that decided all six.
MIN_USABLE_DUMPS = 10


class Derivation(NamedTuple):
    """What one sampling run saw."""

    entries: dict[str, dict[str, object]]
    dumps_read: int
    #: benchmark -> output_type values the harness reports that we cannot read.
    unreadable_output_types: dict[str, set[str]]
    #: benchmark -> how many distinct dumps reported a mode for it.
    contributing_dumps: dict[str, int]


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
    # Sorted before sampling and drawn from a generator of our own: the API
    # makes no promise about listing order, and seeding the global RNG would
    # make the sample depend on whatever else has drawn from it.
    api = HfApi()
    files = sorted(
        f for f in api.list_repo_files(RESULTS_REPO, repo_type="dataset") if f.endswith(".json")
    )
    return random.Random(seed).sample(files, min(limit, len(files)))


def derive(limit: int, seed: int) -> Derivation:
    observed: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    unreadable: dict[str, set[str]] = defaultdict(set)
    contributors: dict[str, set[str]] = defaultdict(set)
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
                # A task we recognise, reporting an output type we do not.
                # That is the harness moving under the mapping, not noise.
                unreadable[benchmark].add(str(config.get("output_type")))
                continue
            contributors[benchmark].add(name)
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
    return Derivation(
        entries,
        dumps_read,
        dict(unreadable),
        {benchmark: len(names) for benchmark, names in contributors.items()},
    )


def _common_prefix(values: list[str]) -> str:
    first, last = values[0], values[-1]
    i = 0
    while i < min(len(first), len(last)) and first[i] == last[i]:
        i += 1
    return first[:i]


def mode_on_file(on_file: dict, benchmark: str) -> object:
    entry = on_file.get(benchmark)
    return entry.get("mode") if isinstance(entry, dict) else None


def objections(
    derivation: Derivation,
    on_file: dict,
    min_dumps: int = MIN_USABLE_DUMPS,
) -> list[str]:
    """Every reason this run does not vouch for the checked-in mapping.

    An empty list is the only thing that means agreement. The comparison is
    over the expected key set rather than over whatever the run happened to
    derive, because a run that downloaded nothing derives nothing and would
    otherwise agree with anything. The dump floor applies to each benchmark
    separately, because what is being claimed is that a task's configuration
    is constant across an independent sample.
    """
    found: list[str] = []
    if derivation.dumps_read < min_dumps:
        found.append(
            f"only {derivation.dumps_read} usable dump(s), below the floor of {min_dumps}"
        )

    thin = {
        benchmark: derivation.contributing_dumps.get(benchmark, 0)
        for benchmark in EXPECTED_BENCHMARKS
        if derivation.contributing_dumps.get(benchmark, 0) < min_dumps
    }
    if thin:
        found.append(
            "vouched for by too few separate dumps (floor "
            f"{min_dumps}): {dict(sorted(thin.items()))}"
        )

    derived = set(derivation.entries)
    if derived != set(EXPECTED_BENCHMARKS):
        found.append(
            f"derived {sorted(derived)}, expected {sorted(EXPECTED_BENCHMARKS)}"
        )
    if derived != set(on_file):
        found.append(
            f"file holds {sorted(on_file)}, the dumps produced {sorted(derived)}"
        )

    for benchmark in sorted(derived & set(on_file)):
        was, now = mode_on_file(on_file, benchmark), derivation.entries[benchmark]["mode"]
        if was != now:
            found.append(f"{benchmark}: file says {was!r}, dumps say {now!r}")

    for benchmark, values in sorted(derivation.unreadable_output_types.items()):
        found.append(f"{benchmark}: unreadable output_type(s) {sorted(values)}")

    return found


def read_mapping(path: Path) -> dict:
    current = yaml.safe_load(path.read_text()) if path.exists() else {}
    if not isinstance(current, dict):
        return {}
    sources = current.get("sources")
    source = sources.get(COMPOSITE_SLUG) if isinstance(sources, dict) else None
    benchmarks = source.get("benchmarks") if isinstance(source, dict) else None
    return benchmarks if isinstance(benchmarks, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25, help="dumps to sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--min-dumps", type=int, default=MIN_USABLE_DUMPS,
        help="usable dumps below which the sample cannot vouch for the file",
    )
    parser.add_argument("--check", action="store_true", help="exit non-zero if the file is stale")
    args = parser.parse_args()

    derivation = derive(args.limit, args.seed)
    print(f"read {derivation.dumps_read} dumps; derived {len(derivation.entries)} benchmarks")
    for benchmark, entry in derivation.entries.items():
        seen_in = derivation.contributing_dumps.get(benchmark, 0)
        print(
            f"  {benchmark:14s} {entry['mode']:10s} {entry['harness_tasks']:26s}"
            f" ({seen_in} dumps)"
        )

    found = objections(derivation, read_mapping(MAPPING_PATH), args.min_dumps)
    if found:
        print(f"\n{MAPPING_PATH.name} is NOT vouched for by this run:", file=sys.stderr)
        for objection in found:
            print(f"  {objection}", file=sys.stderr)
        return 1 if args.check else 0
    print(f"\n{MAPPING_PATH.name} agrees with the dumps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

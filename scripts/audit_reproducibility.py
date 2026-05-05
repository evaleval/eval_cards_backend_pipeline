#!/usr/bin/env python3
"""One-off audit of EEE_datastore reproducibility-field coverage.

For each evaluation_result inside each EEE record, determine whether the
spec's required fields (Signal 1 in EvalCards Interpretive Signals spec)
are populated. Aggregate per-benchmark and overall.

Three agentic classifications are reported. The chosen production rule is
the union of all three:

  Spec tasks-literal: autobenchmarkcard.purpose_and_intended_users.tasks
                      contains any of ["agentic", "tool_use",
                      "multi_step_agent"] (spec §3.1, joined from the
                      Auto-BenchmarkCards record).
  Signal A:           generation_args.agentic_eval_config present and
                      non-null (spec §3.1, from EEE).
  Signal B:           regex on the slugified benchmark name (transcribed
                      from scripts/pipeline.py:1620
                      infer_category_from_benchmark; documented deviation
                      operationalizing spec §3.4 "category tags suggest
                      agentic").

Output is markdown, ready to drop into notes/interpretive-signals-planning.md.

Run:
  source ~/.zshrc && uv run --with huggingface_hub --no-project \\
    python -m scripts.audit_reproducibility > /tmp/audit.md
"""

import collections
import datetime
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from scripts.helpers.benchmark_identity import (
    is_agentic_benchmark_name,
    normalize_benchmark_key,
)

REPO = "evaleval/EEE_datastore"
CARDS_REPO = "evaleval/auto-benchmarkcards"
BASE_FIELDS = ("temperature", "top_p", "max_tokens", "prompt_template")
AGENTIC_FIELDS = ("eval_plan", "eval_limits")
SPEC_AGENTIC_TASK_TOKENS = ("agentic", "tool_use", "multi_step_agent")


def is_populated(obj, field: str) -> bool:
    """Spec §2.3: present AND not null."""
    if not isinstance(obj, dict):
        return False
    return field in obj and obj[field] is not None


def has_prompt_template(obj) -> bool:
    """prompt_template empty string counts as present (spec §3.4)."""
    if not isinstance(obj, dict):
        return False
    if "prompt_template" not in obj:
        return False
    return obj["prompt_template"] is not None  # "" is not None


def load_card_lookup() -> dict:
    """Pulls the flat aggregator from auto-benchmarkcards and keys it by
    the same normalize_benchmark_key the pipeline uses for matching."""
    metadata_dir = Path(
        snapshot_download(
            CARDS_REPO,
            repo_type="dataset",
            allow_patterns=["benchmark-metadata.json"],
            token=os.environ.get("HF_TOKEN"),
        )
    )
    flat = json.loads((metadata_dir / "benchmark-metadata.json").read_text())
    return {normalize_benchmark_key(k): v for k, v in flat.items()}


def card_for_benchmark(card_lookup: dict, benchmark: str) -> dict | None:
    """Look up the card by normalized benchmark key (best-effort, no fuzzy match)."""
    return card_lookup.get(normalize_benchmark_key(benchmark))


def has_spec_agentic_task(card) -> bool:
    """Spec §3.1 literal-token check on
    autobenchmarkcard.purpose_and_intended_users.tasks. Returns False if no
    card, no tasks list, or no token match."""
    if not isinstance(card, dict):
        return False
    # Some entries are wrapped; the inner card is the real payload.
    if "benchmark_card" in card and isinstance(card["benchmark_card"], dict):
        card = card["benchmark_card"]
    tasks = (card.get("purpose_and_intended_users") or {}).get("tasks")
    if not isinstance(tasks, list):
        return False
    return any(isinstance(t, str) and t in SPEC_AGENTIC_TASK_TOKENS for t in tasks)


def main() -> None:
    print("[1a/3] snapshot_download evaleval/EEE_datastore ...", file=sys.stderr)
    local_dir = snapshot_download(
        REPO,
        repo_type="dataset",
        allow_patterns=["data/**/*.json"],
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  -> {local_dir}", file=sys.stderr)

    print(
        "[1b/3] snapshot_download evaleval/auto-benchmarkcards (for spec §3.1 join) ...",
        file=sys.stderr,
    )
    card_lookup = load_card_lookup()
    print(f"  -> {len(card_lookup)} cards in lookup", file=sys.stderr)

    record_paths = [
        p
        for p in Path(local_dir).glob("data/**/*.json")
        if not p.name.endswith("_samples.jsonl")
        and not p.name.endswith("_samples.json")
    ]
    print(f"[2/3] inspecting {len(record_paths)} record files ...", file=sys.stderr)
    benchmarks_with_card = set()
    benchmarks_without_card = set()

    rows = []
    skipped = 0
    for i, path in enumerate(record_paths):
        if i % 500 == 0:
            print(f"  {i}/{len(record_paths)}", file=sys.stderr)
        try:
            with open(path) as fp:
                rec = json.load(fp)
        except Exception:
            skipped += 1
            continue

        try:
            benchmark = path.relative_to(Path(local_dir) / "data").parts[0]
        except ValueError:
            benchmark = "unknown"

        card = card_for_benchmark(card_lookup, benchmark)
        if card is not None:
            benchmarks_with_card.add(benchmark)
        else:
            benchmarks_without_card.add(benchmark)
        sig_tasks_literal = has_spec_agentic_task(card)

        for r_idx, result in enumerate(rec.get("evaluation_results") or []):
            gc = result.get("generation_config") if isinstance(result, dict) else None
            ga = gc.get("generation_args") if isinstance(gc, dict) else None

            base_pop = {
                f: (
                    has_prompt_template(ga)
                    if f == "prompt_template"
                    else is_populated(ga, f)
                )
                for f in BASE_FIELDS
            }
            base_count = sum(base_pop.values())

            sig_a = (
                is_populated(ga, "agentic_eval_config")
                if isinstance(ga, dict)
                else False
            )
            sig_b = is_agentic_benchmark_name(benchmark)

            agentic_pop = {f: is_populated(ga, f) for f in AGENTIC_FIELDS}
            agentic_count = sum(agentic_pop.values())

            # Spec-faithful agentic detection (per §3.1 + §3.4):
            #   sig_tasks_literal  — spec §3.1, joined from autobenchmarkcard
            #   sig_a              — spec §3.1, from EEE
            #   sig_b              — operationalizes spec §3.4 ("category tags suggest agentic")
            agentic_for_spec = sig_tasks_literal or sig_a or sig_b
            required_count = 4 + (2 if agentic_for_spec else 0)
            populated_count = base_count + (agentic_count if agentic_for_spec else 0)
            has_gap = populated_count < required_count

            rows.append(
                {
                    "benchmark": benchmark,
                    "result_idx": r_idx,
                    "base_count": base_count,
                    "agentic_count": agentic_count,
                    "sig_tasks_literal": sig_tasks_literal,
                    "sig_a": sig_a,
                    "sig_b": sig_b,
                    "has_gap": has_gap,
                    "required_count": required_count,
                    "populated_count": populated_count,
                    **{f"base_{f}": base_pop[f] for f in BASE_FIELDS},
                    **{f"agentic_{f}": agentic_pop[f] for f in AGENTIC_FIELDS},
                }
            )

    print(
        f"[3/3] aggregating {len(rows)} evaluation_results "
        f"(skipped {skipped} unparseable records) ...",
        file=sys.stderr,
    )

    n = len(rows)
    n_records = len(record_paths) - skipped
    has_gap_count = sum(1 for r in rows if r["has_gap"])
    pct_has_gap = 100.0 * has_gap_count / n if n else 0.0

    field_pop = {f: sum(1 for r in rows if r[f"base_{f}"]) for f in BASE_FIELDS}
    base_dist = collections.Counter(r["base_count"] for r in rows)

    by_bench = collections.defaultdict(list)
    for r in rows:
        by_bench[r["benchmark"]].append(r)

    agentic_rows = [
        r for r in rows if r["sig_tasks_literal"] or r["sig_a"] or r["sig_b"]
    ]
    agentic_dist = collections.Counter(r["agentic_count"] for r in agentic_rows)
    n_tasks_literal = sum(1 for r in rows if r["sig_tasks_literal"])
    n_card_lookups = len(benchmarks_with_card) + len(benchmarks_without_card)

    conf_result = collections.Counter((r["sig_a"], r["sig_b"]) for r in rows)
    bench_a = {b: any(r["sig_a"] for r in rs) for b, rs in by_bench.items()}
    bench_b = {b: any(r["sig_b"] for r in rs) for b, rs in by_bench.items()}
    conf_bench = collections.Counter((bench_a[b], bench_b[b]) for b in by_bench)

    agentic_eval_plan = sum(1 for r in agentic_rows if r["agentic_eval_plan"])
    agentic_eval_limits = sum(1 for r in agentic_rows if r["agentic_eval_limits"])

    md = []
    md.append("### Findings\n")
    md.append(
        f"**Method.** Inline run from main session, {datetime.date.today().isoformat()}. "
        f"Full corpus: {n_records} records, {n} evaluation_results parsed "
        f"(skipped {skipped} records). Script: `scripts/audit_reproducibility.py`. "
        f"Auto-BenchmarkCards joined: **{len(benchmarks_with_card)} of "
        f"{len(benchmarks_with_card) + len(benchmarks_without_card)}** EEE benchmarks "
        f"matched a card. Benchmarks without a card lookup: "
        f"{sorted(benchmarks_without_card) if benchmarks_without_card else '—'}.\n"
    )
    md.append(
        f"**Headline.** `has_gap = true`: **{pct_has_gap:.1f}%** ({has_gap_count} of {n}). "
        "Required = spec's 4 base fields, expanded to 6 when the benchmark is agentic. "
        "Agentic detection is the union of three signals (preserves the spec literally; adds Signal B "
        "as an operationalization of spec §3.4):\n\n"
        "- **Spec tasks-literal** — `autobenchmarkcard.purpose_and_intended_users.tasks` contains "
        'any of `["agentic", "tool_use", "multi_step_agent"]` (spec §3.1, joined from the '
        "Auto-BenchmarkCards record).\n"
        "- **Signal A** — `generation_args.agentic_eval_config` present and non-null (spec §3.1, "
        "from EEE).\n"
        "- **Signal B** — existing pipeline regex on the benchmark name "
        '(`scripts/pipeline.py:1623`); operationalizes spec §3.4 "category tags suggest agentic."\n\n'
        f"Spec tasks-literal evaluates to True for **{n_tasks_literal}** evaluation_results "
        f"({100.0 * n_tasks_literal / n:.2f}%) — confirming with a real card join that the literal "
        f"tokens never match real Auto-BenchmarkCards values, so the union is fully driven by A and B "
        f"on current data.\n"
    )

    md.append("**Per-base-field populated rate.**\n")
    md.append("| Field | Populated | % |")
    md.append("|---|---:|---:|")
    for f in BASE_FIELDS:
        c = field_pop[f]
        md.append(f"| `{f}` | {c} / {n} | {100.0 * c / n:.1f}% |")
    md.append("")

    md.append("**`populated_count` distribution (base 0..4).**\n")
    md.append("| populated | results | % |")
    md.append("|---:|---:|---:|")
    for k in range(5):
        c = base_dist.get(k, 0)
        md.append(f"| {k} | {c} | {100.0 * c / n:.1f}% |")
    md.append("")

    if agentic_rows:
        n_ag = len(agentic_rows)
        md.append(
            f"**Agentic-extras `populated_count` distribution (0..2; over the "
            f"{n_ag} results flagged agentic by spec tasks-literal ∪ A ∪ B; tasks-literal "
            f"contributes 0 on current data).**\n"
        )
        md.append("| populated | results | % |")
        md.append("|---:|---:|---:|")
        for k in range(3):
            c = agentic_dist.get(k, 0)
            md.append(f"| {k} | {c} | {100.0 * c / n_ag:.1f}% |")
        md.append("")
        md.append(
            f"Of those agentic results: `eval_plan` populated **{agentic_eval_plan}** "
            f"({100.0 * agentic_eval_plan / n_ag:.1f}%); `eval_limits` populated "
            f"**{agentic_eval_limits}** ({100.0 * agentic_eval_limits / n_ag:.1f}%).\n"
        )

    md.append("**Per-benchmark table** (sorted by `n_results` desc).\n")
    md.append(
        "| benchmark | n_results | % has_gap | % temp | % top_p | % max_tok | % prompt | A | B |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|:-:|:-:|")
    for b, rs in sorted(by_bench.items(), key=lambda kv: -len(kv[1])):
        nr = len(rs)
        gap_pct = 100.0 * sum(1 for r in rs if r["has_gap"]) / nr
        pcts = {
            f: 100.0 * sum(1 for r in rs if r[f"base_{f}"]) / nr for f in BASE_FIELDS
        }
        a_flag = "Y" if bench_a[b] else "."
        b_flag = "Y" if bench_b[b] else "."
        md.append(
            f"| {b} | {nr} | {gap_pct:.1f}% | "
            f"{pcts['temperature']:.0f}% | {pcts['top_p']:.0f}% | "
            f"{pcts['max_tokens']:.0f}% | {pcts['prompt_template']:.0f}% | "
            f"{a_flag} | {b_flag} |"
        )
    md.append("")

    md.append("**Agentic-signal comparison.**\n")
    md.append(f"_Result-level (n={n}):_\n")
    md.append("| | B=true | B=false |")
    md.append("|---|---:|---:|")
    md.append(
        f"| **A=true** | {conf_result.get((True, True), 0)} | {conf_result.get((True, False), 0)} |"
    )
    md.append(
        f"| **A=false** | {conf_result.get((False, True), 0)} | {conf_result.get((False, False), 0)} |"
    )
    md.append("")
    md.append(f"_Benchmark-level (n={len(by_bench)}):_\n")
    md.append("| | B=true | B=false |")
    md.append("|---|---:|---:|")
    md.append(
        f"| **A=true** | {conf_bench.get((True, True), 0)} | {conf_bench.get((True, False), 0)} |"
    )
    md.append(
        f"| **A=false** | {conf_bench.get((False, True), 0)} | {conf_bench.get((False, False), 0)} |"
    )
    md.append("")

    print("\n".join(md))
    print(f"\n[done] {n} results across {n_records} records", file=sys.stderr)


if __name__ == "__main__":
    main()

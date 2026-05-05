import re
from typing import Final

BENCHMARK_DEFAULT_METRICS: Final[dict[str, tuple[str, str]]] = {
    "global_mmlu_lite": ("Accuracy", "accuracy"),
}

BUILTIN_METRIC_DISPLAY_MAP: Final[dict[str, str]] = {
    "accuracy": "Accuracy",
    "exact_match": "Exact Match",
    "win_rate": "Win Rate",
    "mean_win_rate": "Mean Win Rate",
    "average_attempts": "Average Attempts",
    "average_latency_ms": "Average Latency (ms)",
    "latency_mean": "Latency Mean",
    "latency_std": "Latency Standard Deviation",
    "latency_p95": "Latency 95th Percentile",
    "rank": "Rank",
    "overall_accuracy": "Overall Accuracy",
    "total_cost": "Total Cost",
    "cost_per_task": "Cost per Task",
    "cost_per_100_calls": "Cost per 100 Calls",
    "elo": "Elo Rating",
    "score": "Score",
    "arc_score": "ARC Score",
    "mean_score": "Mean Score",
    "format_sensitivity_stddev": "Format Sensitivity Standard Deviation",
    "format_sensitivity_max_delta": "Format Sensitivity Max Delta",
}

PASS_AT_REGEX = re.compile(r"pass\s*@?\s*(\d+)", flags=re.IGNORECASE)
PASS_AT_EXACT_REGEX = re.compile(r"^\s*pass\s*@?\s*(\d+)\s*$", flags=re.IGNORECASE)
EVAL_DESCRIPTION_METRIC_REGEX = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 @%+./_-]*?)\s+on\s+(.+?)\s*$"
)



# Keeping until refactor is complete:
# Mapping from what frontend expects to backend categories. 
# This is legacy logic from the TS `inferCategoryFromBenchmark` regex (`lib/benchmark-schema.ts`) 
# moved from frontend; itw as needed so that a benchmark with no card domains still gets the same 
# canonical bucket the frontend would have computed at request time.
FRONTEND_REGEX_CATEGORY_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "safety",
        (
            "safety",
            "harmful",
            "toxic",
            "truthful",
            "unsafe",
            "civilcomments",
            "civil_comments",
            "jailbreak",
            "red-team",
            "adversarial",
        ),
    ),
    (
        "agentic",
        (
            "agent",
            "swe-bench",
            "swe_bench",
            "terminal-bench",
            "tau-bench",
            "tau_bench",
            "appworld",
            "browsecomp",
        ),
    ),
    (
        "reasoning",
        (
            "reasoning",
            "bbh",
            "math",
            "gsm",
            "gpqa",
            "musr",
            "code",
            "humaneval",
            "livecodebench",
        ),
    ),
    (
        "knowledge",
        (
            "mmlu",
            "knowledge",
            "trivia",
            "medqa",
            "legalbench",
            "theory_of_mind",
        ),
    ),
)
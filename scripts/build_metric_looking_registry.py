import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DATA_ROOT = Path(".cache/eee_datastore/data")
OUTPUT_PATH = Path("registry/metric_looking_strings.json")

DESCRIPTION_PREFIX_REGEX = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 @%+./_-]*?)\s+on\s+.+$")
PASS_AT_REGEX = re.compile(r"pass\s*@?\s*(\d+)", re.IGNORECASE)
EVALUATION_NAME_SUFFIX_PATTERNS = [
    re.compile(r"^(?P<prefix>.+?)[\s_-]+(?P<metric>pass[\s_@-]*\d+)$", re.IGNORECASE),
    re.compile(
        r"^(?P<prefix>.+?)[\s_-]+(?P<metric>mean[\s_-]+win[\s_-]+rate|mean[\s_-]+score|win[\s_-]+rate|avg(?:erage)?[\s_-]+attempts|avg(?:erage)?[\s_-]+latency(?:[\s_-]+ms)?|latency[\s_-]+mean|latency[\s_-]+std(?:andard[\s_-]+deviation)?|latency[\s_-]+p95|total[\s_-]+cost|overall[\s_-]+accuracy|accuracy|exact[\s_-]+match|em|rank)$",
        re.IGNORECASE,
    ),
]
ALIAS_MAP = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "em": "exact_match",
    "exact_match": "exact_match",
    "f1": "f1",
    "bleu_4": "bleu_4",
    "rouge_2": "rouge_2",
    "win_rate": "win_rate",
    "mean_win_rate": "mean_win_rate",
    "avg_attempts": "average_attempts",
    "average_attempts": "average_attempts",
    "avg_latency_ms": "average_latency_ms",
    "average_latency_ms": "average_latency_ms",
    "latency_mean": "latency_mean",
    "latency_mean_s": "latency_mean",
    "latency_std": "latency_std",
    "latency_std_s": "latency_std",
    "latency_standard_deviation": "latency_std",
    "latency_p95": "latency_p95",
    "latency_p95_s": "latency_p95",
    "latency_95th_percentile": "latency_p95",
    "rank": "rank",
    "overall_rank": "rank",
    "overall_accuracy": "overall_accuracy",
    "total_cost": "total_cost",
    "total_cost_usd": "total_cost",
    "cost": "cost",
    "cost_per_task": "cost_per_task",
    "cost_per_100_calls_usd": "cost_per_100_calls",
    "elo": "elo",
    "score": "score",
    "arc_score": "arc_score",
    "mean_score": "mean_score",
}
DISPLAY_NAME_MAP = {
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


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def canonical_metric_key(value: str) -> str:
    raw = value.strip()
    pass_match = PASS_AT_REGEX.search(raw)
    if pass_match:
        return f"pass_at_{pass_match.group(1)}"
    full_normalized = normalize_key(raw)
    normalized = normalize_key(raw.split(".")[-1])
    if full_normalized.endswith("format_sensitivity_stddev"):
        return "format_sensitivity_stddev"
    if full_normalized.endswith("format_sensitivity_max_delta"):
        return "format_sensitivity_max_delta"
    return ALIAS_MAP.get(normalized, normalized)


def choose_display_name(normalized: str, examples: Counter) -> str:
    if normalized in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[normalized]
    pass_match = re.match(r"pass_at_(\d+)$", normalized)
    if pass_match:
        return f"Pass@{pass_match.group(1)}"

    best_value = normalized.replace("_", " ").title()
    best_score = float("-inf")
    for value, count in examples.items():
        score = float(count)
        if "." in value:
            score -= 100
        if "_" in value:
            score -= 25
        if value == value.lower():
            score -= 10
        if value[:1].isupper():
            score += 8
        if " " in value:
            score += 5
        if "@" in value:
            score += 4
        if re.fullmatch(r"[A-Z0-9@.+ -]+", value):
            score += 3
        if score > best_score:
            best_score = score
            best_value = value
    return best_value


def add_entry(
    registry: dict[str, dict],
    counters: dict[str, Counter],
    source: str,
    value: str,
    normalized: str | None = None,
) -> None:
    value = value.strip()
    if not value:
        return
    key = normalized or canonical_metric_key(value)
    if not key:
        return
    entry = registry.setdefault(
        key,
        {
            "normalized": key,
            "examples": Counter(),
            "sources": set(),
            "total_count": 0,
        },
    )
    entry["examples"][value] += 1
    entry["sources"].add(source)
    entry["total_count"] += 1
    counters[source][key] += 1


def main() -> int:
    registry: dict[str, dict] = {}
    source_counters: dict[str, Counter] = defaultdict(Counter)
    file_count = 0

    for path in DATA_ROOT.rglob("*.json"):
        if path.name.endswith(".jsonl"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        file_count += 1
        for result in payload.get("evaluation_results") or []:
            metric_config = result.get("metric_config") or {}
            if isinstance(metric_config, dict):
                metric_id = metric_config.get("metric_id")
                metric_name = metric_config.get("metric_name")
                description = metric_config.get("evaluation_description")
                canonical_from_metric_id = None

                if isinstance(metric_id, str) and metric_id.strip():
                    canonical_from_metric_id = canonical_metric_key(metric_id)
                    add_entry(registry, source_counters, "metric_id", metric_id, canonical_from_metric_id)
                if isinstance(metric_name, str) and metric_name.strip():
                    add_entry(registry, source_counters, "metric_name", metric_name, canonical_from_metric_id)
                if isinstance(description, str):
                    match = DESCRIPTION_PREFIX_REGEX.match(description)
                    if match:
                        add_entry(registry, source_counters, "evaluation_description_prefix", match.group(1), canonical_from_metric_id)

            details = (result.get("score_details") or {}).get("details") or {}
            if isinstance(details, dict):
                tab = details.get("tab")
                if isinstance(tab, str) and tab.strip():
                    add_entry(registry, source_counters, "score_details_tab", tab)

            evaluation_name = result.get("evaluation_name")
            if isinstance(evaluation_name, str):
                stripped = evaluation_name.strip()
                for pattern in EVALUATION_NAME_SUFFIX_PATTERNS:
                    match = pattern.match(stripped)
                    if match:
                        add_entry(registry, source_counters, "evaluation_name_suffix", match.group("metric"))
                        break

    serializable_entries = []
    alias_to_normalized: dict[str, str] = {}
    for entry in registry.values():
        examples = entry.pop("examples")
        sources = entry.pop("sources")
        for value, _count in examples.items():
            alias_to_normalized[value] = entry["normalized"]
            alias_to_normalized[normalize_key(value)] = entry["normalized"]
            alias_to_normalized[normalize_key(value.split(".")[-1])] = entry["normalized"]
        serializable_entries.append(
            {
                **entry,
                "display_name": choose_display_name(entry["normalized"], examples),
                "sources": sorted(sources),
                "examples": [{"value": value, "count": count} for value, count in examples.most_common(10)],
            }
        )

    serializable_entries.sort(key=lambda item: (-item["total_count"], item["normalized"]))
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_dataset": "evaleval/EEE_datastore",
        "local_data_root": str(DATA_ROOT),
        "record_file_count": file_count,
        "registry_entry_count": len(serializable_entries),
        "source_totals": {source: sum(counter.values()) for source, counter in sorted(source_counters.items())},
        "alias_to_normalized": dict(sorted((key, value) for key, value in alias_to_normalized.items() if key)),
        "entries": serializable_entries,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} with {len(serializable_entries)} entries from {file_count} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

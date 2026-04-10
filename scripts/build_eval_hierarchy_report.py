import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.pipeline as pipeline


OUTPUT_JSON = Path("reports/eval_hierarchy.json")
OUTPUT_MD = Path("reports/eval_hierarchy.md")

FAMILY_RULES = [
    (re.compile(r"^helm_"), ("helm", "HELM")),
    (re.compile(r"^apex(?:_|-)"), ("apex", "Apex")),
    (re.compile(r"^fibble(?:\d+)?_arena$"), ("fibble", "Fibble")),
    (re.compile(r"^wordle_arena$"), ("wordle", "Wordle")),
    (re.compile(r"^tau_bench_2_"), ("tau_bench_2", "Tau-Bench 2")),
    (re.compile(r"^appworld"), ("appworld", "AppWorld")),
    (re.compile(r"^hfopenllm"), ("hfopenllm", "HF Open LLM")),
    (re.compile(r"^livecodebench"), ("livecodebench", "LiveCodeBench")),
    (re.compile(r"^terminal_bench"), ("terminal_bench", "Terminal-Bench")),
]


def infer_eval_family(config_name: str, benchmark_family_key: str) -> tuple[str, str]:
    normalized_config = pipeline.normalize_benchmark_key(config_name)
    for regex, value in FAMILY_RULES:
        if regex.match(normalized_config):
            return value
    family_key = benchmark_family_key or normalized_config
    return family_key, pipeline.humanize_token_key(family_key)


def get_or_create(container: dict, key: str, value_factory):
    if key not in container:
        container[key] = value_factory()
    return container[key]


def _empty_tags() -> dict:
    return {"domains": [], "languages": [], "tasks": []}


def _merge_tags(parent_tags: dict, child_tags: dict) -> dict:
    """Merge child tags into parent, deduplicating and preserving order."""
    merged = {}
    for key in ("domains", "languages", "tasks"):
        seen = set(parent_tags.get(key, []))
        combined = list(parent_tags.get(key, []))
        for val in child_tags.get(key, []):
            if val not in seen:
                seen.add(val)
                combined.append(val)
        merged[key] = combined
    return merged


def markdown_tree(report: dict) -> str:
    lines: list[str] = []
    lines.append("# EEE Eval Hierarchy")
    lines.append("")
    lines.append("## QA Summary")
    qa = report["qa"]
    stats = report["stats"]
    lines.append(f"- Families: `{stats['family_count']}`")
    lines.append(f"- Composite benchmarks: `{stats['composite_count']}`")
    lines.append(f"- Standalone benchmarks: `{stats['standalone_benchmark_count']}`")
    lines.append(f"- Benchmarks: `{stats['single_benchmark_count']}`")
    lines.append(f"- Slices: `{stats['slice_count']}`")
    lines.append(f"- Unique metrics: `{stats['metric_count']}`")
    lines.append(f"- Metric rows scanned: `{stats['metric_rows_scanned']}`")
    lines.append(f"- Fallback metrics: `{qa['fallback_metric_count']}`")
    lines.append(f"- Benchmarks that still look metric-like: `{qa['metric_like_single_benchmark_count']}`")
    lines.append(f"- Benchmarks where name matches the only metric: `{qa['single_equals_only_metric_count']}`")
    lines.append("")

    if qa["fallback_metrics"]:
        lines.append("### Fallback Metrics")
        for item in qa["fallback_metrics"]:
            lines.append(f"- `{item['composite_benchmark']}` -> `{item['single_benchmark']}` -> `{item['metric_name']}`")
        lines.append("")

    if qa["metric_like_single_benchmarks"]:
        lines.append("### Metric-Like Single Benchmarks")
        for item in qa["metric_like_single_benchmarks"]:
            lines.append(
                f"- `{item['composite_benchmark']}` -> `{item['single_benchmark']}` looked metric-like and had metrics {', '.join(f'`{metric}`' for metric in item['metrics'])}"
            )
        lines.append("")

    if qa["single_equals_only_metric"]:
        lines.append("### Single Benchmark Equals Its Only Metric")
        for item in qa["single_equals_only_metric"]:
            lines.append(f"- `{item['composite_benchmark']}` -> `{item['single_benchmark']}`")
        lines.append("")

    lines.append("## Hierarchy")
    lines.append("")
    def _card_mark(node):
        return " [x]" if node.get("has_card") else " [ ]"

    for family in report["families"]:
        lines.append(f"-{_card_mark(family)} {family['display_name']}")
        # Flattened families: slices/metrics/benchmarks directly on the family
        for slice_info in family.get("slices", []):
            lines.append(f"  - {slice_info['display_name']}")
            for metric in slice_info["metrics"]:
                lines.append(f"    - {metric['display_name']}")
        for metric in family.get("metrics", []):
            lines.append(f"  - {metric['display_name']}")
        for benchmark in family.get("benchmarks", []):
            lines.append(f"  -{_card_mark(benchmark)} {benchmark['display_name']}")
            for slice_info in benchmark.get("slices", []):
                lines.append(f"    - {slice_info['display_name']}")
                for metric in slice_info["metrics"]:
                    lines.append(f"      - {metric['display_name']}")
            for metric in benchmark["metrics"]:
                lines.append(f"    - {metric['display_name']}")
        for benchmark in family.get("standalone_benchmarks", []):
            lines.append(f"  -{_card_mark(benchmark)} {benchmark['display_name']}")
            for slice_info in benchmark.get("slices", []):
                lines.append(f"    - {slice_info['display_name']}")
                for metric in slice_info["metrics"]:
                    lines.append(f"      - {metric['display_name']}")
            for metric in benchmark["metrics"]:
                lines.append(f"    - {metric['display_name']}")
        for composite in family["composites"]:
            lines.append(f"  -{_card_mark(composite)} {composite['display_name']}")
            # Flattened composites have slices/metrics directly
            for slice_info in composite.get("slices", []):
                lines.append(f"    - {slice_info['display_name']}")
                for metric in slice_info["metrics"]:
                    lines.append(f"      - {metric['display_name']}")
            for metric in composite.get("metrics", []):
                lines.append(f"    - {metric['display_name']}")
            for benchmark in composite.get("benchmarks", []):
                lines.append(f"    -{_card_mark(benchmark)} {benchmark['display_name']}")
                for slice_info in benchmark.get("slices", []):
                    lines.append(f"      - {slice_info['display_name']}")
                    for metric in slice_info["metrics"]:
                        lines.append(f"        - {metric['display_name']}")
                for metric in benchmark["metrics"]:
                    lines.append(f"      - {metric['display_name']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    pipeline.load_metric_registry()

    local_dataset_dir = pipeline.ensure_local_dataset_snapshot(pipeline.DEFAULT_LOCAL_DATASET_DIR, None, False)
    local_metadata_dir = pipeline.ensure_local_benchmark_metadata_snapshot(pipeline.DEFAULT_LOCAL_BENCHMARK_METADATA_DIR, None, False)
    if not local_metadata_dir:
        raise RuntimeError("Benchmark metadata cache is missing; run the pipeline once to populate it.")

    _cards, metadata_lookup, _benchmark_metadata = pipeline.load_benchmark_metadata(local_metadata_dir)

    family_tree: dict[str, dict] = {}
    fallback_metrics: list[dict] = []
    metric_like_single_benchmarks: list[dict] = []
    single_equals_only_metric: list[dict] = []
    metric_rows_scanned = 0

    configs = pipeline.discover_configs(local_dataset_dir, None)
    for config in configs:
        files = pipeline.list_json_files_for_config(config, local_dataset_dir, None)
        for dataset_path in files:
            record = pipeline.read_dataset_json(dataset_path, local_dataset_dir, None)
            evaluation_id = pipeline.as_string(record.get("evaluation_id"))
            benchmark = evaluation_id.split("/")[0] if evaluation_id else config
            eval_results = record.get("evaluation_results") if isinstance(record.get("evaluation_results"), list) else []
            first_result = eval_results[0] if eval_results else {}
            evaluation = {
                "benchmark": benchmark,
                "source_data": (first_result or {}).get("source_data"),
            }
            benchmark_card = pipeline.lookup_benchmark_card(
                metadata_lookup,
                benchmark,
                pipeline.canonical_benchmark_family_key(benchmark),
                ((first_result or {}).get("source_data") or {}).get("dataset_name"),
            )

            family_key, family_display_name = infer_eval_family(config, pipeline.canonical_benchmark_family_key(benchmark))
            family_node = get_or_create(
                family_tree,
                family_key,
                lambda: {
                    "key": family_key,
                    "display_name": family_display_name,
                    "composites": {},
                },
            )

            composite_key = benchmark
            composite_display_name = pipeline.humanize_token_key(benchmark)
            composite_node = get_or_create(
                family_node["composites"],
                composite_key,
                lambda: {
                    "key": composite_key,
                    "display_name": composite_display_name,
                    "benchmarks": {},
                    "has_card": False,
                    "category": pipeline.infer_category_from_benchmark(benchmark, benchmark_card),
                },
            )
            if benchmark_card:
                composite_node["has_card"] = True

            for result in eval_results:
                score = pipeline.extract_score(result)
                if score is None:
                    continue
                metric_rows_scanned += 1
                normalized = pipeline.classify_evaluation_result(evaluation, result, benchmark_card)
                benchmark_key = normalized.get("benchmark_leaf_key") or pipeline.normalize_benchmark_key(benchmark)
                benchmark_display_name = normalized.get("benchmark_leaf_name") or composite_display_name
                slice_key = normalized.get("slice_key")
                slice_display_name = normalized.get("slice_name")
                metric_key = normalized["metric_key"]
                metric_display_name = normalized["metric_name"]

                benchmark_node = get_or_create(
                    composite_node["benchmarks"],
                    benchmark_key,
                    lambda: {
                        "key": benchmark_key,
                        "display_name": benchmark_display_name,
                        "has_card": False,
                        "tags": {"domains": [], "languages": [], "tasks": []},
                        "metrics": {},
                        "slices": {},
                        "metric_sources": Counter(),
                        "is_summary_score": False,
                    },
                )
                # Mark as summary score if the normalized result says so.
                # We only upgrade (False→True), never downgrade, since all results
                # for this leaf key should agree.
                if normalized.get("is_summary_score"):
                    benchmark_node["is_summary_score"] = True
                # Check for a card matching this leaf benchmark
                if not benchmark_node["has_card"]:
                    leaf_card = pipeline.lookup_benchmark_card(
                        metadata_lookup, benchmark_key, benchmark_display_name,
                    )
                    if leaf_card:
                        benchmark_node["has_card"] = True
                        benchmark_node["tags"] = pipeline.extract_benchmark_tags(leaf_card)
                target_metrics = benchmark_node["metrics"]
                if slice_key:
                    slice_node = get_or_create(
                        benchmark_node["slices"],
                        slice_key,
                        lambda: {
                            "key": slice_key,
                            "display_name": slice_display_name,
                            "metrics": {},
                        },
                    )
                    target_metrics = slice_node["metrics"]
                metric_node = get_or_create(target_metrics, metric_key, lambda: {"key": metric_key, "display_name": metric_display_name, "sources": set()})
                metric_node["sources"].add(normalized.get("metric_source") or "unknown")
                benchmark_node["metric_sources"][normalized.get("metric_source") or "unknown"] += 1

                if normalized.get("metric_source") == "fallback":
                    fallback_metrics.append(
                        {
                            "composite_benchmark": composite_key,
                            "single_benchmark": benchmark_display_name,
                            "metric_name": metric_display_name,
                        }
                    )

    families = []
    composite_count = 0
    standalone_benchmark_count = 0
    single_benchmark_count = 0
    slice_count = 0
    metric_count = 0

    for family in sorted(family_tree.values(), key=lambda item: item["display_name"].lower()):
        composites = []
        standalone_benchmarks = []
        for composite in sorted(family["composites"].values(), key=lambda item: item["display_name"].lower()):
            benchmarks = []
            summary_benchmark_nodes = []
            for single in sorted(composite["benchmarks"].values(), key=lambda item: item["display_name"].lower()):
                single_benchmark_count += 1
                metrics = sorted(single["metrics"].values(), key=lambda item: item["display_name"].lower())
                slices = []
                for slice_info in sorted(single["slices"].values(), key=lambda item: item["display_name"].lower()):
                    slice_metrics = sorted(slice_info["metrics"].values(), key=lambda item: item["display_name"].lower())
                    slice_count += 1
                    metric_count += len(slice_metrics)
                    slices.append(
                        {
                            "key": slice_info["key"],
                            "display_name": slice_info["display_name"],
                            "metrics": [
                                {
                                    "key": metric["key"],
                                    "display_name": metric["display_name"],
                                    "sources": sorted(metric["sources"]),
                                }
                                for metric in slice_metrics
                            ],
                        }
                    )
                metric_count += len(metrics)
                metric_names = [metric["display_name"] for metric in metrics]
                metric_like_key = pipeline.strict_metric_alias_lookup(single["display_name"])
                if metric_like_key:
                    metric_like_single_benchmarks.append(
                        {
                            "composite_benchmark": composite["key"],
                            "single_benchmark": single["display_name"],
                            "metrics": metric_names,
                        }
                    )
                if len(metrics) == 1 and pipeline.normalize_benchmark_key(single["display_name"]) == metrics[0]["key"]:
                    single_equals_only_metric.append(
                        {
                            "composite_benchmark": composite["key"],
                            "single_benchmark": single["display_name"],
                        }
                    )

                bm_node = {
                    "key": single["key"],
                    "display_name": single["display_name"],
                    "has_card": single.get("has_card", False),
                    "tags": single.get("tags", _empty_tags()),
                    "slices": slices,
                    "metrics": [
                        {
                            "key": metric["key"],
                            "display_name": metric["display_name"],
                            "sources": sorted(metric["sources"]),
                        }
                        for metric in metrics
                    ],
                }
                if single.get("is_summary_score"):
                    summary_benchmark_nodes.append(bm_node)
                else:
                    benchmarks.append(bm_node)

            # If there are real sub-benchmarks alongside summary nodes, keep only
            # the real benchmarks in the list and surface the summary eval IDs.
            # If ALL nodes are summaries (e.g. sciarena where every metric is
            # "overall_*"), treat them as regular benchmarks so the hierarchy
            # is not left empty.
            if benchmarks and summary_benchmark_nodes:
                # Compute the eval_summary_id each summary benchmark maps to.
                comp_norm_key = pipeline.normalize_benchmark_key(composite["key"])
                summary_eval_ids = [
                    pipeline.slugify(f"{comp_norm_key}__{s['key']}")
                    if s["key"] != comp_norm_key
                    else pipeline.slugify(comp_norm_key)
                    for s in summary_benchmark_nodes
                ]
            else:
                # Either no summaries, or ONLY summaries — keep everything as normal benchmarks.
                benchmarks = benchmarks + summary_benchmark_nodes
                summary_eval_ids = []
            has_card = composite.get("has_card", False) or any(b.get("has_card") for b in benchmarks)
            comp_category = composite.get("category", "other")
            # Bubble tags up: merge all benchmark tags into composite-level tags
            composite_tags = _empty_tags()
            for bm in benchmarks:
                composite_tags = _merge_tags(composite_tags, bm.get("tags", _empty_tags()))
            if len(benchmarks) == 1:
                # Single benchmark inside a composite is redundant — promote
                # the benchmark's content into the composite (or standalone).
                bm = benchmarks[0]
                norm_comp = pipeline.normalize_benchmark_key(composite["key"])
                norm_bm = pipeline.normalize_benchmark_key(bm["key"])
                if norm_comp == norm_bm and bm["display_name"].lower() == composite["display_name"].lower():
                    # Names match — treat as standalone benchmark.
                    standalone_benchmark_count += 1
                    bm["has_card"] = has_card
                    bm["tags"] = composite_tags
                    bm["category"] = comp_category
                    bm["summary_eval_ids"] = summary_eval_ids
                    standalone_benchmarks.append(bm)
                else:
                    # Names differ (e.g. "Helm mmlu" vs "Mmlu") — keep
                    # composite name but absorb benchmark content.
                    composite_count += 1
                    composites.append(
                        {
                            "key": composite["key"],
                            "display_name": composite["display_name"],
                            "has_card": has_card,
                            "tags": composite_tags,
                            "category": comp_category,
                            "slices": bm.get("slices", []),
                            "metrics": bm.get("metrics", []),
                            "summary_eval_ids": summary_eval_ids,
                        }
                    )
                continue
            composite_count += 1
            composites.append(
                {
                    "key": composite["key"],
                    "display_name": composite["display_name"],
                    "has_card": has_card,
                    "tags": composite_tags,
                    "category": comp_category,
                    "benchmarks": benchmarks,
                    "summary_eval_ids": summary_eval_ids,
                }
            )
        # Flatten redundant nesting when a family wraps a single identical child.
        # Flatten redundant nesting: if a family has exactly one child
        # (standalone or composite) and nothing else, promote that child.
        # Bubble tags up to family level from all children
        family_tags = _empty_tags()
        for sb in standalone_benchmarks:
            family_tags = _merge_tags(family_tags, sb.get("tags", _empty_tags()))
        for comp in composites:
            family_tags = _merge_tags(family_tags, comp.get("tags", _empty_tags()))

        # Derive family category from children (most common, or first found)
        child_cats = [sb.get("category", "other") for sb in standalone_benchmarks] + [c.get("category", "other") for c in composites]
        family_category = max(set(child_cats), key=child_cats.count) if child_cats else "other"

        if len(standalone_benchmarks) == 1 and not composites:
            sb = standalone_benchmarks[0]
            families.append(
                {
                    "key": sb["key"],
                    "display_name": sb["display_name"],
                    "has_card": sb.get("has_card", False),
                    "tags": family_tags,
                    "category": sb.get("category", family_category),
                    "standalone_benchmarks": [],
                    "composites": [],
                    "slices": sb.get("slices", []),
                    "metrics": sb.get("metrics", []),
                    "summary_eval_ids": sb.get("summary_eval_ids", []),
                }
            )
        elif len(composites) == 1 and not standalone_benchmarks:
            comp = composites[0]
            promoted = {
                "key": comp["key"],
                "display_name": comp["display_name"],
                "has_card": comp.get("has_card", False),
                "tags": family_tags,
                "category": comp.get("category", family_category),
                "standalone_benchmarks": [],
                "composites": [],
                "summary_eval_ids": comp.get("summary_eval_ids", []),
            }
            if "benchmarks" in comp:
                promoted["benchmarks"] = comp["benchmarks"]
            else:
                promoted["slices"] = comp.get("slices", [])
                promoted["metrics"] = comp.get("metrics", [])
            families.append(promoted)
        else:
            any_card = (
                any(sb.get("has_card") for sb in standalone_benchmarks)
                or any(c.get("has_card") for c in composites)
            )
            families.append(
                {
                    "key": family["key"],
                    "display_name": family["display_name"],
                    "has_card": any_card,
                    "tags": family_tags,
                    "category": family_category,
                    "standalone_benchmarks": standalone_benchmarks,
                    "composites": composites,
                }
            )

    report = {
        "stats": {
            "family_count": len(families),
            "composite_count": composite_count,
            "standalone_benchmark_count": standalone_benchmark_count,
            "single_benchmark_count": single_benchmark_count,
            "slice_count": slice_count,
            "metric_count": metric_count,
            "metric_rows_scanned": metric_rows_scanned,
        },
        "qa": {
            "fallback_metric_count": len(fallback_metrics),
            "fallback_metrics": fallback_metrics[:50],
            "metric_like_single_benchmark_count": len(metric_like_single_benchmarks),
            "metric_like_single_benchmarks": metric_like_single_benchmarks[:100],
            "single_equals_only_metric_count": len(single_equals_only_metric),
            "single_equals_only_metric": single_equals_only_metric[:100],
        },
        "families": families,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(markdown_tree(report), encoding="utf-8")

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit real EEE records against the vendored schema + the derived pyarrow schema.

Walks the local EEE snapshot (`.cache/eee_datastore/data/<config>/...`) by
default, skipping `IGNORED_CONFIGS`. For each record runs three checks:

1.  Pydantic `EvaluationLog.model_validate(record)` — catches contract
    violations the upstream schema explicitly forbids (e.g. missing required
    fields, wrong enum values, conditional `if/then/else` constraints).

2.  pyarrow cast via `pa.RecordBatch.from_pylist([padded], schema=SCHEMA)` —
    catches anything pyarrow rejects on cast (type mismatches surfaced after
    pad, list/dict-where-scalar-expected, etc).

3.  Optional-field population — records which optional top-level fields and
    selected nested fields are populated, per config. Surfaces the
    "per-config-optional" question from the migration plan.

Outputs a summary to stdout. Pass `--json` to emit a machine-readable JSON
report (consumed by `notes/05-eee-schema-reference.md` refresh workflow).

Usage:
    uv run python scripts/audit_eee_records.py
    uv run python scripts/audit_eee_records.py --max-per-config 200  # quick smoke
    uv run python scripts/audit_eee_records.py --json > audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
from pydantic import ValidationError

# Ensure src/ is importable when run directly via `uv run python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_card_backend.config import IGNORED_CONFIGS  # noqa: E402
from eval_card_backend.schemas.eee_arrow import (  # noqa: E402
    derive_pyarrow_schema,
    pad_record_for_cast,
    schema_version,
)
from eval_card_backend.schemas.eee_types import EvaluationLog  # noqa: E402


# Optional top-level fields whose population we track per config to surface
# "universal vs per-config-optional" patterns called out in the migration plan.
TRACKED_TOP_LEVEL = [
    "schema_version",
    "evaluation_timestamp",
    "detailed_evaluation_results",
]

# Selected nested fields whose population matters for downstream signals
# (reproducibility, comparability) — not exhaustive.
TRACKED_NESTED = [
    "evaluation_results[].generation_config.generation_args.temperature",
    "evaluation_results[].generation_config.generation_args.max_tokens",
    "evaluation_results[].generation_config.generation_args.prompt_template",
    "evaluation_results[].generation_config.generation_args.eval_plan",
    "evaluation_results[].generation_config.generation_args.eval_limits",
    "evaluation_results[].generation_config.generation_args.agentic_eval_config",
    "evaluation_results[].score_details.uncertainty",
    "evaluation_results[].metric_config.metric_parameters",
    "evaluation_results[].metric_config.llm_scoring",
]


def _has_nested(record: dict, path: str) -> bool:
    """Walk a dotted path with `[]` segments. Returns True if at least one
    leaf along the path is non-null."""
    parts = path.split(".")
    nodes: list = [record]
    for part in parts:
        next_nodes: list = []
        for node in nodes:
            if node is None:
                continue
            if part.endswith("[]"):
                key = part[:-2]
                if isinstance(node, dict) and isinstance(node.get(key), list):
                    next_nodes.extend(node[key])
            else:
                if isinstance(node, dict) and node.get(part) is not None:
                    next_nodes.append(node[part])
        nodes = next_nodes
        if not nodes:
            return False
    return True


def _iter_local_records(eee_root: Path):
    """Yield (config, path, record_dict) for every JSON record under
    `<eee_root>/data/<config>/.../*.json`, skipping IGNORED_CONFIGS."""
    data_dir = eee_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"EEE snapshot not found at {data_dir}")
    for config_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        config = config_dir.name
        if config in IGNORED_CONFIGS:
            continue
        for path in sorted(config_dir.rglob("*.json")):
            if path.name.endswith(".jsonl"):
                continue
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                yield config, path, {"_read_error": f"{type(exc).__name__}: {exc}"}
                continue
            if not isinstance(rec, dict):
                yield config, path, {"_read_error": f"non-dict (type={type(rec).__name__})"}
                continue
            yield config, path, rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eee-root", default=".cache/eee_datastore",
        help="Path to local EEE snapshot. Default: .cache/eee_datastore",
    )
    parser.add_argument(
        "--max-per-config", type=int, default=None,
        help="Sample at most N records per config (default: no cap).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON to stdout.",
    )
    parser.add_argument(
        "--example-errors", type=int, default=3,
        help="How many example error messages to keep per (config, error_class). Default: 3.",
    )
    args = parser.parse_args()

    eee_root = Path(args.eee_root).resolve()
    pa_schema = derive_pyarrow_schema()
    contract_version = schema_version()

    # Per-config aggregates
    record_counts: Counter[str] = Counter()
    read_errors: dict[str, list[str]] = defaultdict(list)
    schema_versions: dict[str, Counter[str]] = defaultdict(Counter)
    validation_failures: Counter[str] = Counter()
    cast_failures: Counter[str] = Counter()
    validation_examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    cast_examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    top_level_population: dict[str, Counter[str]] = defaultdict(Counter)
    nested_population: dict[str, Counter[str]] = defaultdict(Counter)
    per_config_seen: Counter[str] = Counter()

    for config, path, rec in _iter_local_records(eee_root):
        if args.max_per_config is not None and per_config_seen[config] >= args.max_per_config:
            continue
        per_config_seen[config] += 1
        record_counts[config] += 1

        if "_read_error" in rec:
            read_errors[config].append(f"{path.name}: {rec['_read_error']}")
            continue

        sv = rec.get("schema_version") or "<missing>"
        schema_versions[config][sv] += 1

        for fld in TRACKED_TOP_LEVEL:
            if rec.get(fld) is not None:
                top_level_population[config][fld] += 1
        for path_str in TRACKED_NESTED:
            if _has_nested(rec, path_str):
                nested_population[config][path_str] += 1

        # Pydantic validation
        try:
            EvaluationLog.model_validate(rec)
        except ValidationError as exc:
            validation_failures[config] += 1
            err_class = type(exc).__name__
            key = (config, err_class)
            if len(validation_examples[key]) < args.example_errors:
                validation_examples[key].append(f"{path.name}: {exc.errors()[0]}")
        except Exception as exc:  # defensive: any pydantic-internal error
            validation_failures[config] += 1
            key = (config, type(exc).__name__)
            if len(validation_examples[key]) < args.example_errors:
                validation_examples[key].append(f"{path.name}: {exc}")

        # pyarrow cast — pad first, then build a single-record batch.
        try:
            padded = pad_record_for_cast(rec, pa_schema)
            pa.RecordBatch.from_pylist([padded], schema=pa_schema)
        except Exception as exc:
            cast_failures[config] += 1
            err_class = type(exc).__name__
            key = (config, err_class)
            if len(cast_examples[key]) < args.example_errors:
                tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
                cast_examples[key].append(f"{path.name}: {tb}")

    # ----- Output -----
    total = sum(record_counts.values())
    total_validation_failures = sum(validation_failures.values())
    total_cast_failures = sum(cast_failures.values())

    summary = {
        "eee_root": str(eee_root),
        "vendored_schema_version": contract_version,
        "ignored_configs": sorted(IGNORED_CONFIGS),
        "total_records": total,
        "total_validation_failures": total_validation_failures,
        "total_cast_failures": total_cast_failures,
        "configs": [],
    }
    for config in sorted(record_counts):
        n = record_counts[config]
        cfg_entry = {
            "config": config,
            "records": n,
            "read_errors": len(read_errors[config]),
            "schema_versions": dict(schema_versions[config]),
            "validation_failures": validation_failures[config],
            "cast_failures": cast_failures[config],
            "top_level_population": {
                fld: top_level_population[config][fld] for fld in TRACKED_TOP_LEVEL
                if top_level_population[config][fld] > 0
            },
            "nested_population": {
                path_str: nested_population[config][path_str] for path_str in TRACKED_NESTED
                if nested_population[config][path_str] > 0
            },
            "validation_examples": {
                err_class: validation_examples[(config, err_class)]
                for (cfg2, err_class) in validation_examples
                if cfg2 == config
            },
            "cast_examples": {
                err_class: cast_examples[(config, err_class)]
                for (cfg2, err_class) in cast_examples
                if cfg2 == config
            },
        }
        summary["configs"].append(cfg_entry)

    if args.json:
        json.dump(summary, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    # Human-readable summary
    print(f"EEE schema audit — vendored contract v{contract_version}")
    print(f"  source: {eee_root}")
    print(f"  records sampled: {total} across {len(record_counts)} configs "
          f"(IGNORED_CONFIGS={sorted(IGNORED_CONFIGS)})")
    print(f"  pydantic validation failures: {total_validation_failures} "
          f"({100*total_validation_failures/max(total,1):.1f}%)")
    print(f"  pyarrow cast failures:        {total_cast_failures} "
          f"({100*total_cast_failures/max(total,1):.1f}%)")
    print()

    print("Per-config (sorted by record count):")
    for config in sorted(record_counts, key=lambda c: -record_counts[c]):
        n = record_counts[config]
        vf = validation_failures[config]
        cf = cast_failures[config]
        sv = schema_versions[config]
        sv_str = ", ".join(f"{k}={v}" for k, v in sv.most_common())
        print(f"  {config}: n={n}, validation_fail={vf}, cast_fail={cf}, schema_versions=[{sv_str}]")

    if validation_examples:
        print("\nSample validation errors (first {} per (config, class)):".format(args.example_errors))
        for (config, err_class), examples in sorted(validation_examples.items()):
            for ex in examples:
                print(f"  [{config}/{err_class}] {ex}")

    if cast_examples:
        print("\nSample cast errors (first {} per (config, class)):".format(args.example_errors))
        for (config, err_class), examples in sorted(cast_examples.items()):
            for ex in examples:
                print(f"  [{config}/{err_class}] {ex}")

    print("\nTop-level optional field population:")
    for fld in TRACKED_TOP_LEVEL:
        present_configs = [c for c in record_counts if top_level_population[c][fld] > 0]
        absent_configs = [c for c in record_counts if top_level_population[c][fld] == 0]
        print(f"  {fld}: present in {len(present_configs)}/{len(record_counts)} configs")
        if absent_configs and len(absent_configs) <= 10:
            print(f"    absent: {sorted(absent_configs)}")

    print("\nNested optional field population (config count):")
    for path_str in TRACKED_NESTED:
        present_configs = [c for c in record_counts if nested_population[c][path_str] > 0]
        print(f"  {path_str}: present in {len(present_configs)}/{len(record_counts)} configs")

    return 0


if __name__ == "__main__":
    sys.exit(main())

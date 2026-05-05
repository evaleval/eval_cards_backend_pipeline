"""Audit pipeline output for identity/display consistency.

Reads ``eval-list.json`` (default: ``output/eval-list.json``) and reports:
  - ``benchmark_leaf_key`` values that map to multiple ``benchmark_leaf_name``s
  - ``metric_key`` values that map to multiple ``metric_name``s

Usage:
  uv run --no-project python -m scripts.audit_normalization [path/to/eval-list.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_eval_list(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_eval_list(data: dict) -> dict:
    leaf_key_to_names: dict[str, set[str]] = defaultdict(set)
    metric_key_to_names: dict[str, set[str]] = defaultdict(set)

    for summary in data.get("evals") or []:
        if not isinstance(summary, dict):
            continue
        lk = summary.get("benchmark_leaf_key")
        ln = summary.get("benchmark_leaf_name")
        if lk is not None and ln is not None:
            leaf_key_to_names[str(lk)].add(str(ln))

        for metric in summary.get("metrics") or []:
            if not isinstance(metric, dict):
                continue
            mk = metric.get("metric_key")
            mn = metric.get("metric_name")
            if mk is not None and mn is not None:
                metric_key_to_names[str(mk)].add(str(mn))

    leaf_ambiguous = {
        k: sorted(v) for k, v in leaf_key_to_names.items() if len(v) > 1
    }
    metric_ambiguous = {
        k: sorted(v) for k, v in metric_key_to_names.items() if len(v) > 1
    }

    return {
        "eval_count": len(data.get("evals") or []),
        "leaf_key_ambiguous_count": len(leaf_ambiguous),
        "leaf_key_ambiguous": leaf_ambiguous,
        "metric_key_ambiguous_count": len(metric_ambiguous),
        "metric_key_ambiguous": metric_ambiguous,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_list_path",
        nargs="?",
        type=Path,
        default=Path("output/eval-list.json"),
        help="Path to eval-list.json",
    )
    args = parser.parse_args()
    path: Path = args.eval_list_path
    if not path.exists():
        print(
            json.dumps(
                {
                    "event": "audit.skip",
                    "reason": "file_not_found",
                    "path": str(path),
                }
            ),
            file=sys.stderr,
        )
        return 0

    report = audit_eval_list(_load_eval_list(path))
    print(json.dumps({"event": "audit.normalization", **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

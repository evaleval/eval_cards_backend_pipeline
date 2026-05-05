"""Diff pipeline.py vs pipeline_refactor.py outputs against fixture data.

Migration-only dev tool. Delete once pipeline_refactor.py replaces pipeline.py.

Usage (from repo root):
    uv run --with pytest --with huggingface_hub --with datasets --with pandas \\
      --with pyarrow \\
      --with 'eval-entity-resolver @ git+https://github.com/evaleval/evalcard-registry.git#subdirectory=packages/eval-entity-resolver' \\
      --no-project python -m scripts.diff_pipelines [--config NAME] [--keep-output]

Each pipeline runs in its own subprocess against a synthetic fixture tree
(reused from tests/conftest.py); a fresh interpreter per side keeps registry /
module caches from cross-contaminating. Output trees are then walked and
compared file-by-file: JSON deep-equal with `generated_at` masked, parquet
row-set compared after sort, README byte-equal modulo timestamp lines.

Exits 0 on equivalence, 1 on any unexplained mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


VOLATILE_KEYS = {"generated_at"}
MANIFEST_IGNORE_KEYS = {"generated_at", "artifact_sizes"}


def mask(value: Any, ignore: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            k: ("<masked>" if k in ignore else mask(v, ignore))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [mask(item, ignore) for item in value]
    return value


RUNNER_TEMPLATE = """\
import sys, importlib
from pathlib import Path

sys.path.insert(0, {repo_root!r})

mod = importlib.import_module({module_name!r})
mod.OUTPUT_DIR = Path({output_dir!r})
sys.argv = [{module_name!r}, '--dry-run']
sys.exit(mod.main())
"""


def run_pipeline(
    module_name: str,
    output_dir: Path,
    eee_root: Path,
    abc_root: Path,
    configs: list[str],
) -> None:
    env = os.environ.copy()
    env["EEE_LOCAL_DATASET_DIR"] = str(eee_root)
    env["BENCHMARK_METADATA_LOCAL_DIR"] = str(abc_root)
    env["CONFIGS"] = ",".join(configs)
    env["REGISTRY_LOCAL_PARQUET_DIR"] = str(
        REPO_ROOT / "tests" / "fixtures" / "registry_aliases"
    )
    env["EMIT_LEGACY_JSON"] = "1"
    for k in (
        "CONFIG_NAMES",
        "CONFIG_LIMIT",
        "HF_TOKEN",
        "CARD_BACKEND_OUTPUT_REPO",
        "CARD_BACKEND_ALLOW_PRODUCTION",
        "GITHUB_ACTIONS",
        "REGISTRY_DISABLE",
    ):
        env.pop(k, None)

    code = RUNNER_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        module_name=module_name,
        output_dir=str(output_dir),
    )
    print(f"[diff] running {module_name} -> {output_dir}")
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        raise RuntimeError(f"{module_name} exited {result.returncode}")


def list_files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def first_diff(a: Any, b: Any, path: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        keys_a, keys_b = set(a), set(b)
        only_a, only_b = keys_a - keys_b, keys_b - keys_a
        if only_a:
            return f"{path}: keys only in baseline: {sorted(only_a)[:5]}"
        if only_b:
            return f"{path}: keys only in candidate: {sorted(only_b)[:5]}"
        for k in sorted(keys_a):
            d = first_diff(a[k], b[k], f"{path}.{k}" if path else k)
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: list length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != b:
        return f"{path}: {a!r} vs {b!r}"
    return None


def diff_json(rel: Path, baseline: Path, candidate: Path) -> str | None:
    a = json.loads(baseline.read_text())
    b = json.loads(candidate.read_text())
    ignore = MANIFEST_IGNORE_KEYS if rel.name == "manifest.json" else VOLATILE_KEYS
    return first_diff(mask(a, ignore), mask(b, ignore))


def diff_parquet(rel: Path, baseline: Path, candidate: Path) -> str | None:
    import pyarrow.parquet as pq

    ta = pq.read_table(str(baseline))
    tb = pq.read_table(str(candidate))
    if ta.schema.names != tb.schema.names:
        return (
            f"schema column order differs: "
            f"{ta.schema.names} vs {tb.schema.names}"
        )
    if ta.num_rows != tb.num_rows:
        return f"row count {ta.num_rows} vs {tb.num_rows}"

    def rowkey(row: dict) -> str:
        return json.dumps(row, default=str, sort_keys=True)

    a_rows = sorted(ta.to_pylist(), key=rowkey)
    b_rows = sorted(tb.to_pylist(), key=rowkey)
    for i, (ra, rb) in enumerate(zip(a_rows, b_rows)):
        if ra != rb:
            d = first_diff(ra, rb, f"row[{i}]")
            return d or f"row[{i}] differs (no concrete field found)"
    return None


def diff_text(rel: Path, baseline: Path, candidate: Path) -> str | None:
    a = baseline.read_text().splitlines()
    b = candidate.read_text().splitlines()
    if len(a) != len(b):
        return f"line count {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if "generated_at" in x and "generated_at" in y:
            continue
        if x != y:
            return f"line {i+1}: {x!r} vs {y!r}"
    return None


def diff_trees(baseline: Path, candidate: Path) -> int:
    a, b = list_files(baseline), list_files(candidate)
    only_a, only_b = sorted(a - b), sorted(b - a)
    common = sorted(a & b)

    failures: list[tuple[str, str]] = []
    for rel in common:
        ba, ca = baseline / rel, candidate / rel
        try:
            if rel.suffix == ".json":
                d = diff_json(rel, ba, ca)
            elif rel.suffix == ".parquet":
                d = diff_parquet(rel, ba, ca)
            else:
                d = diff_text(rel, ba, ca)
        except Exception as e:
            d = f"<error: {e}>"
        if d:
            failures.append((str(rel), d))

    print(
        f"\n[diff] common files: {len(common)}, "
        f"only baseline: {len(only_a)}, only candidate: {len(only_b)}"
    )
    for rel in only_a:
        print(f"  ONLY-BASELINE  {rel}")
    for rel in only_b:
        print(f"  ONLY-CANDIDATE {rel}")
    for rel, msg in failures:
        print(f"  DIFF  {rel}\n        {msg}")

    bad = bool(only_a or only_b or failures)
    print(
        f"\n[diff] {'MISMATCH' if bad else 'EQUIVALENT'} "
        f"({len(failures)} content diffs)"
    )
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Comma-separated subset of fixture configs (default: all). "
        "E.g. --config bench_variant for a tight inner loop.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Retain tmp output dirs after run for manual inspection.",
    )
    args = parser.parse_args()

    try:
        from tests.conftest import _write_fixtures
    except ImportError as e:
        print(
            f"[diff] failed to import tests.conftest._write_fixtures: {e}\n"
            "       Did you invoke via `uv run --with pytest ...`?",
            file=sys.stderr,
        )
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="diff_pipelines_"))
    eee_root = workdir / "eee"
    abc_root = workdir / "abc"
    _write_fixtures(eee_root, abc_root)

    if args.config:
        configs = [c.strip() for c in args.config.split(",") if c.strip()]
    else:
        configs = sorted(
            p.name for p in (eee_root / "data").iterdir() if p.is_dir()
        )

    out_baseline = workdir / "out_baseline"
    out_candidate = workdir / "out_candidate"
    try:
        run_pipeline(
            "scripts.pipeline", out_baseline, eee_root, abc_root, configs
        )
        run_pipeline(
            "scripts.pipeline_refactor",
            out_candidate,
            eee_root,
            abc_root,
            configs,
        )
        rc = diff_trees(out_baseline, out_candidate)
    finally:
        if args.keep_output:
            print(f"\n[diff] outputs retained at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    return rc


if __name__ == "__main__":
    sys.exit(main())

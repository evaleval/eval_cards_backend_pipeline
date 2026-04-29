"""Cross-repo parity verification.

Runs the canonical TS adapters in `general-eval-card/lib/` against the
pipeline's JSON output, then diffs the resulting `EvaluationCardData`,
`BenchmarkEvalListItem`, `BenchmarkEvalSummary`, and
`ModelEvaluationSummary` shapes against the parity payloads emitted in
`output/duckdb/v1/*.parquet`.

Exits non-zero when any unexplained differences are found. Intended to
be the acceptance gate per PLAN_20260428.md §"Cross-repo parity
verification".

Usage::

    uv run --with datasets --no-project python -m scripts.verify_parity \\
      --pipeline-output ./output \\
      --general-eval-card ../general-eval-card

(`pyarrow` is a transitive dep of `datasets`; no extra `--with` needed.)

Differences are reported per-surface with the row identifier and a
field-level diff. The parity emitter (v3) writes ``payload_json`` as
exactly the canonical TS-adapter shape — routing keys live in scalar
parquet columns only, never duplicated in the payload — so this
verifier diffs raw payloads with no per-surface "ignore list."
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


def _load_parquet_payloads(parquet_dir: Path, name: str) -> dict[str, dict]:
    """Read ``output/duckdb/v1/<name>.parquet`` keyed by the natural id.

    Reads via ``pyarrow.parquet`` (transitive dep of `datasets`, which the
    pipeline uses to write these files) — no SQL engine in the loop.
    """
    import pyarrow.parquet as pq

    path = parquet_dir / f"{name}.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(str(path), columns=["model_route_id", "eval_summary_id", "payload_json"])
    route_ids = table.column("model_route_id").to_pylist()
    eval_ids = table.column("eval_summary_id").to_pylist()
    payload_jsons = table.column("payload_json").to_pylist()

    by_id: dict[str, dict] = {}
    # Surfaces whose natural row id lives in the parquet `eval_summary_id`
    # scalar column. `aggregate_eval_summaries` / `matrix_eval_summaries`
    # store `aggregate__<suite>` / `matrix__<suite>` there. Lite variants
    # mirror their non-lite siblings (model_cards_lite → model_route_id,
    # eval_list_lite → eval_summary_id).
    eval_id_keyed = {
        "eval_list",
        "eval_list_lite",
        "eval_summaries",
        "aggregate_eval_summaries",
        "matrix_eval_summaries",
    }
    for route_id, eval_id, payload_json in zip(route_ids, eval_ids, payload_jsons):
        payload = json.loads(payload_json)
        # Choose the natural ID per surface — same key the TS dump uses.
        if name in eval_id_keyed:
            key = eval_id or payload.get("evaluation_id") or payload.get("eval_summary_id")
        else:
            key = route_id or payload.get("route_id") or payload.get("model_route_id")
        if key is None:
            continue
        by_id[str(key)] = payload
    return by_id


def _normalize_for_compare(value: Any) -> Any:
    """Normalize values so structural equality matches across JSON ↔ JSON.

    JSON only has 64-bit floats; Python may receive ints from
    ``pyarrow`` when reading parquet. We normalize numeric types to
    floats and treat ``None`` / missing fields equivalently.
    """
    if isinstance(value, list):
        return [_normalize_for_compare(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_for_compare(v) for k, v in value.items() if v is not None}
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


# Allow up to ~1-ULP drift in float64 aggregates. Python's `sum()` and JS's
# `reduce((s,v) => s+v, 0)` both walk arrays left-to-right, but distinct
# IEEE-754 implementations and runtime library versions accumulate the
# round-off bits slightly differently when re-running the same dataset
# across the two languages. Tolerating ~1e-12 relative error keeps the
# verifier focused on genuine divergences (sort order, schema shape)
# without flagging 0.4796363636363637 vs 0.47963636363636364.
_FLOAT_REL_TOL = 1e-12
_FLOAT_ABS_TOL = 1e-15


def _floats_close(a: float, b: float) -> bool:
    import math

    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=_FLOAT_REL_TOL, abs_tol=_FLOAT_ABS_TOL)


# Lists where item order is determined by a sort on `score` and the items
# carry a stable identity. CPython's `sum()` and V8's `Array.reduce` over
# the same array accumulate IEEE-754 rounding bits slightly differently
# (e.g. `[0.6897, 0.3347, 0.6699, 0.5083, 0.5419, 0.4323]` sums to
# `3.1768` exactly in CPython but `3.1768000000000005` in V8). When the
# resulting averages happen to TIE on one side and not the other, two
# adjacent rows swap and the diff cascades. The data is identical — only
# the leaderboard tie-break order differs. We treat such permutations as
# equal here so the verifier doesn't flag cosmetic float-physics noise.
def _row_identity(row: Any) -> Any:
    if not isinstance(row, dict):
        return None
    info = row.get("model_info") if isinstance(row.get("model_info"), dict) else None
    if info and info.get("id"):
        return ("model_id", info["id"])
    if row.get("evaluation_id"):
        return ("eval_id", row["evaluation_id"])
    if row.get("model_route_id"):
        return ("model_route_id", row["model_route_id"])
    return None


def _score_keyed_lists_equal(a: list, b: list) -> bool:
    """Treat two score-sorted lists as equal if they're permutations under
    the score-tie equivalence class. Both lists must have the same length,
    every item must be a dict with `score` and a stable identity. Sort by
    `(score, identity)` so ULP-tied float averages can't reorder the diff.
    Returns False if the structure isn't recognized as score-keyed."""
    if len(a) != len(b):
        return False
    if not a:
        return True
    if not all(isinstance(r, dict) and "score" in r and _row_identity(r) is not None for r in a):
        return False
    if not all(isinstance(r, dict) and "score" in r and _row_identity(r) is not None for r in b):
        return False

    def _key(row: dict) -> tuple:
        try:
            score = float(row.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        # Bucket scores at ~12 decimal places so values that differ only
        # in the last 1-3 ULP (the cross-language float-accumulation gap)
        # collapse to the same primary key and identity becomes the
        # tiebreaker on both sides. Without this, parity's
        # ``0.5294666666666666`` and TS's ``0.5294666666666668`` sort to
        # different positions even though the verifier treats them as
        # equal under `_floats_close`.
        return (round(score, 12), _row_identity(row))

    a_sorted = sorted(a, key=_key)
    b_sorted = sorted(b, key=_key)
    # Compare element-wise with `_skip_score_check=True` to prevent
    # infinite recursion: if the sub-items themselves contain a
    # `model_results`-shaped list, the inner diff still needs ULP-safe
    # behavior, but we don't re-attempt the outer permutation check.
    for av, bv in zip(a_sorted, b_sorted):
        if _diff(av, bv, _skip_score_check=True):
            return False
    return True


def _diff(a: Any, b: Any, path: str = "", _skip_score_check: bool = False) -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        diffs: list[str] = []
        for key in sorted(set(a.keys()) | set(b.keys())):
            sub = f"{path}.{key}" if path else key
            if key not in a:
                diffs.append(f"+ {sub} = {b[key]!r}")
            elif key not in b:
                diffs.append(f"- {sub} = {a[key]!r}")
            else:
                diffs.extend(_diff(a[key], b[key], sub))
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"~ {path} length {len(a)} vs {len(b)}"]
        if not _skip_score_check and _score_keyed_lists_equal(a, b):
            return []
        out: list[str] = []
        for i, (av, bv) in enumerate(zip(a, b)):
            out.extend(_diff(av, bv, f"{path}[{i}]"))
        return out
    if isinstance(a, float) and isinstance(b, float):
        if _floats_close(a, b):
            return []
    if a != b:
        return [f"~ {path}: parity={a!r} vs ts={b!r}"]
    return []


def _run_ts_dump(
    general_eval_card_root: Path,
    pipeline_output: Path,
    out_path: Path,
) -> None:
    """Invoke `pnpm tsx scripts/dump-adapter-outputs.mts` to generate the
    expected payloads from the TS adapters."""
    if shutil.which("pnpm") is None:
        raise RuntimeError("`pnpm` not on PATH; cross-repo verifier requires it.")

    # Preload the server-only stub so `import "server-only"` at the top
    # of every lib file doesn't crash.
    hook_path = general_eval_card_root / "scripts" / "server_only_hook.cjs"
    env = dict(os.environ)
    existing = env.get("NODE_OPTIONS", "")
    env["NODE_OPTIONS"] = (
        f'{existing} --require "{hook_path}"' if existing else f'--require "{hook_path}"'
    )
    cmd = [
        "pnpm",
        "tsx",
        "scripts/dump-adapter-outputs.mts",
        "--pipeline-output",
        str(pipeline_output),
        "--out",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(general_eval_card_root),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"TS dump failed (rc={proc.returncode}); see stderr above."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-output",
        required=True,
        type=Path,
        help="Path to the pipeline `output/` directory (must already exist).",
    )
    parser.add_argument(
        "--general-eval-card",
        required=True,
        type=Path,
        help="Path to the general-eval-card repo root.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows compared per surface (debugging aid).",
    )
    args = parser.parse_args()

    pipeline_output: Path = args.pipeline_output
    general_eval_card_root: Path = args.general_eval_card
    parquet_dir = pipeline_output / "duckdb" / "v1"
    if not parquet_dir.exists():
        print(f"error: {parquet_dir} not found", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="parity_verify_") as tmpdir:
        ts_out = Path(tmpdir) / "expected.json"
        print(f"[verify_parity] running TS adapter dump → {ts_out}", file=sys.stderr)
        _run_ts_dump(general_eval_card_root, pipeline_output, ts_out)
        ts_dump = json.loads(ts_out.read_text(encoding="utf-8"))

    surface_diffs: dict[str, list[str]] = {}
    for surface_block in ts_dump:
        surface = surface_block["surface"]
        ts_by_id = surface_block["by_id"]
        parity_by_id = _load_parquet_payloads(parquet_dir, surface)

        all_keys = set(ts_by_id.keys()) | set(parity_by_id.keys())
        sample = sorted(all_keys)
        if args.max_rows:
            sample = sample[: args.max_rows]

        diffs: list[str] = []
        for key in sample:
            ts_payload = ts_by_id.get(key)
            parity_payload = parity_by_id.get(key)
            if ts_payload is None:
                diffs.append(f"[{key}] missing on TS side (parity has it)")
                continue
            if parity_payload is None:
                diffs.append(f"[{key}] missing on parity side (TS has it)")
                continue
            normalized_parity = _normalize_for_compare(parity_payload)
            normalized_ts = _normalize_for_compare(ts_payload)
            row_diffs = _diff(normalized_parity, normalized_ts)
            if row_diffs:
                preview = "\n  ".join(row_diffs[:8])
                more = f"\n  ... +{len(row_diffs) - 8} more" if len(row_diffs) > 8 else ""
                diffs.append(f"[{key}] {len(row_diffs)} field diffs:\n  {preview}{more}")
        surface_diffs[surface] = diffs

    print()
    total = 0
    for surface, diffs in surface_diffs.items():
        if not diffs:
            print(f"✓ {surface}: parity payloads match TS adapter output")
            continue
        total += len(diffs)
        print(f"✗ {surface}: {len(diffs)} divergences")
        for entry in diffs[:20]:
            print("  " + entry.replace("\n", "\n  "))
        if len(diffs) > 20:
            print(f"  ... +{len(diffs) - 20} more")
    print()
    if total:
        print(f"FAIL: {total} unexplained divergences across {len(surface_diffs)} surfaces.")
        return 1
    print("OK: all surfaces match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

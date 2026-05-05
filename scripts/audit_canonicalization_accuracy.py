"""Sample EEE_datastore and measure pipeline-final canonicalization accuracy.

Two-phase workflow:

    1. ``sample``  — walks ``.cache/eee_datastore/`` (or whatever
       ``EEE_LOCAL_DATASET_DIR`` points at), extracts every unique
       ``(raw_value, source_config)`` triple/pair for each entity type
       {model, benchmark, metric}, runs each through the pipeline's
       canonicalization function (the *same* function whose output ships in the
       published artifacts), and writes one CSV per entity type with the
       prediction plus a blank ``gold_canonical_id`` column.

    2. ``analyze`` — reads the now-labeled CSVs and prints precision, recall,
       resolution rate, and per-strategy breakdowns with Wilson 95%% CIs.
       Computes both unweighted (per-unique-pair) and row-weighted variants.

Why pipeline-final, not raw registry response:

    The deployed registry (``eval-entity-resolver``) is one layer in a stack.
    The pipeline wraps it — the in-repo ``metric_looking_strings.json`` is
    consulted *before* the registry for metrics; ``canonical_model_identity``
    tries multiple candidate forms; benchmark resolutions are rolled up to the
    root canonical (so per-domain children like ``tau2-airline`` collapse into
    ``tau2-bench``). The published output reflects the composite. To measure
    what end users actually see, we call the composite functions:

      - model     → ``pipeline.canonical_model_identity(model_info)['canonical_model_id']``
      - metric    → ``pipeline.canonicalize_metric_key(raw_metric_id)``
      - benchmark → ``registry.resolve_benchmark + get_canonical_benchmark_root``

    The CSV also records the raw registry response in ``registry_canonical_id``
    so you can see where the wrapping changed the answer.

Manual-review labeling convention for ``gold_canonical_id``:

    - The correct canonical id (e.g. ``openai/gpt-4o``) — counts as TP if equal
      to predicted, FN if predicted=None, FP if predicted differs.
    - ``NONE`` — no correct entry exists in the registry. Counts as TN if
      predicted=None, FP if predicted is anything else.
    - ``SKIP`` (or ``IDK``) — ambiguous, malformed input, or you can't decide.
      Excluded from all metrics.

Usage::

    uv run --with huggingface_hub --with datasets --with pandas --with pyarrow \\
      --with 'eval-entity-resolver @ git+https://github.com/evaleval/evalcard-registry.git#subdirectory=packages/eval-entity-resolver' \\
      --no-project python -m scripts.audit_canonicalization_accuracy sample \\
      --out-dir reports/canonicalization_accuracy --n-model 200 --n-benchmark 200

    # ... fill gold_canonical_id in the three CSVs ...

    uv run --no-project python -m scripts.audit_canonicalization_accuracy analyze \\
      --in-dir reports/canonicalization_accuracy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_EEE_DIR = Path(os.environ.get("EEE_LOCAL_DATASET_DIR", ".cache/eee_datastore"))
DEFAULT_OUT_DIR = Path("reports/canonicalization_accuracy")
ENTITY_TYPES = ("model", "benchmark", "metric")
SAMPLE_SEED = 42

CSV_COLUMNS = [
    "raw_value",
    "source_config",
    "row_count",
    "predicted_canonical_id",      # pipeline-final (what ships)
    "registry_canonical_id",       # raw registry response (diagnostic)
    "strategy",
    "confidence",
    "extra_context",               # JSON-encoded; helpful during review
    "gold_canonical_id",
    "notes",
]


# ----- EEE corpus extraction --------------------------------------------------


@dataclass(frozen=True)
class Sample:
    raw_value: str
    source_config: str
    extra_context_json: str = ""

    def key(self) -> tuple[str, str]:
        return (self.raw_value, self.source_config)


def _iter_eee_records(eee_dir: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield ``(source_config, record_json)`` for every EEE evaluation file."""
    data_dir = eee_dir / "data"
    if not data_dir.is_dir():
        raise SystemExit(
            f"EEE data directory not found: {data_dir}. Set EEE_LOCAL_DATASET_DIR or "
            "snapshot the dataset first (see eval_cards_backend_pipeline README)."
        )
    for config_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        source_config = config_dir.name
        for json_path in config_dir.rglob("*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    yield source_config, json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue


def collect_samples(eee_dir: Path) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """Walk EEE_datastore once. For each entity type, build a dict keyed by
    (raw_value, source_config) → {row_count, sample_record}.

    ``sample_record`` carries enough downstream context to (a) feed the pipeline's
    canonicalization functions (the model branch needs the full ``model_info``
    dict, not just the id) and (b) help the human reviewer disambiguate
    (``evaluation_name`` next to ``dataset_name``, etc.).
    """
    by_type: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        t: defaultdict(lambda: {"count": 0, "model_info": None, "context": {}}) for t in ENTITY_TYPES
    }
    file_count = 0
    for source_config, record in _iter_eee_records(eee_dir):
        file_count += 1
        model_info = record.get("model_info") or {}
        model_id = (model_info.get("id") or "").strip()
        if model_id:
            entry = by_type["model"][(model_id, source_config)]
            entry["count"] += 1
            if entry["model_info"] is None:
                entry["model_info"] = model_info
                entry["context"] = {
                    "name": model_info.get("name"),
                    "developer": model_info.get("developer"),
                }

        for result in record.get("evaluation_results") or []:
            source_data = result.get("source_data") or {}
            metric_config = result.get("metric_config") or {}

            bench_name = (result.get("evaluation_name") or "").strip()
            if bench_name:
                entry = by_type["benchmark"][(bench_name, source_config)]
                entry["count"] += 1
                if not entry["context"]:
                    entry["context"] = {
                        "dataset_name": source_data.get("dataset_name"),
                        "hf_repo": source_data.get("hf_repo"),
                        "benchmark_field": record.get("benchmark"),
                    }

            metric_id = (metric_config.get("metric_id") or "").strip()
            if metric_id:
                entry = by_type["metric"][(metric_id, source_config)]
                entry["count"] += 1
                if not entry["context"]:
                    entry["context"] = {
                        "metric_name": metric_config.get("metric_name"),
                        "metric_kind": metric_config.get("metric_kind"),
                        "metric_unit": metric_config.get("metric_unit"),
                    }

    print(
        f"[audit] scanned {file_count} EEE records: "
        + ", ".join(f"{t}={len(by_type[t])} unique" for t in ENTITY_TYPES)
    )
    return by_type


# ----- Sampling --------------------------------------------------------------


def sample_keys(
    pool: dict[tuple[str, str], dict[str, Any]], n: int | None
) -> list[tuple[str, str]]:
    """Uniform sample over unique pairs. ``n=None`` (or n>=|pool|) returns all."""
    keys = sorted(pool.keys())
    if n is None or n >= len(keys):
        return keys
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(keys, n)


# ----- Predictions: pipeline-final canonicalization --------------------------


def _pipeline_final_model(model_info: dict | None, raw_id: str) -> str | None:
    """Return what ``canonical_model_identity`` would emit as ``canonical_model_id``."""
    from scripts import pipeline

    minfo = dict(model_info or {})
    minfo.setdefault("id", raw_id)
    identity = pipeline.canonical_model_identity(minfo)
    return identity.get("canonical_model_id")


def _pipeline_final_benchmark(raw_value: str, source_config: str) -> str | None:
    """Replicate the benchmark canonicalization at scripts/pipeline.py:5336–5361.

    Caveat: the production pipeline first runs ``infer_benchmark_leaf_and_slice``
    on the raw EEE name to derive ``benchmark_leaf_name``/``benchmark_leaf_key``,
    which is then fed to the resolver. That pre-pass mostly handles subset/slice
    splitting (``tau-bench-2/airline`` → leaf ``tau-bench-2``, slice ``airline``).
    For benchmarks without slash/paren subset suffixes, the resolver's normalized
    matcher (case- and separator-insensitive) makes the pre-pass invariant.
    Subset-suffixed strings are flagged in the ``notes`` column at sample time
    so the reviewer can decide whether the gold should be the leaf or the slice.
    """
    from scripts import registry

    resolution = registry.resolve_benchmark(raw_value, source_config)
    canonical_id = resolution.get("canonical_id")
    if canonical_id:
        root_id = registry.get_canonical_benchmark_root(canonical_id)
        if root_id and root_id != canonical_id:
            canonical_id = root_id
    return canonical_id


def _pipeline_final_metric(raw_value: str) -> str | None:
    """Return what ``canonicalize_metric_key`` would emit (empty string → None)."""
    from scripts import pipeline

    result = pipeline.canonicalize_metric_key(raw_value)
    return result or None


def predict(
    entity_type: str, raw_value: str, source_config: str, sample_record: dict[str, Any]
) -> dict[str, Any]:
    """Produce both the pipeline-final and the raw-registry predictions."""
    from scripts import registry

    if entity_type == "model":
        registry_resp = registry.resolve_model(raw_value)
        final = _pipeline_final_model(sample_record.get("model_info"), raw_value)
    elif entity_type == "benchmark":
        registry_resp = registry.resolve_benchmark(raw_value, source_config)
        final = _pipeline_final_benchmark(raw_value, source_config)
    elif entity_type == "metric":
        registry_resp = registry.resolve_metric(raw_value)
        final = _pipeline_final_metric(raw_value)
    else:
        raise ValueError(f"unknown entity type: {entity_type}")

    return {
        "predicted_canonical_id": final,
        "registry_canonical_id": registry_resp.get("canonical_id"),
        "strategy": registry_resp.get("strategy"),
        "confidence": registry_resp.get("confidence", 0.0),
    }


# ----- CSV I/O ---------------------------------------------------------------


def _flag_notes(entity_type: str, raw_value: str, sample_record: dict[str, Any]) -> str:
    """Inline hints to help the reviewer (subset suffixes, dataset_name disagreements)."""
    notes: list[str] = []
    if entity_type == "benchmark":
        if "/" in raw_value:
            notes.append("contains '/' — possible slash-form subset")
        if "(" in raw_value and ")" in raw_value:
            notes.append("contains parens — possible paren-form subset")
        ctx = sample_record.get("context") or {}
        ds = (ctx.get("dataset_name") or "").strip()
        if ds and ds != raw_value:
            notes.append(f"dataset_name='{ds}' differs from evaluation_name")
    return "; ".join(notes)


def write_sample_csv(
    path: Path,
    entity_type: str,
    keys: list[tuple[str, str]],
    pool: dict[tuple[str, str], dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for raw_value, source_config in keys:
            sample_record = pool[(raw_value, source_config)]
            try:
                predictions = predict(entity_type, raw_value, source_config, sample_record)
            except Exception as error:  # don't let one bad row kill the whole run
                predictions = {
                    "predicted_canonical_id": None,
                    "registry_canonical_id": None,
                    "strategy": f"error:{type(error).__name__}",
                    "confidence": 0.0,
                }
            writer.writerow(
                {
                    "raw_value": raw_value,
                    "source_config": source_config,
                    "row_count": sample_record.get("count", 0),
                    "predicted_canonical_id": predictions["predicted_canonical_id"] or "",
                    "registry_canonical_id": predictions["registry_canonical_id"] or "",
                    "strategy": predictions["strategy"] or "",
                    "confidence": f"{predictions['confidence']:.4f}",
                    "extra_context": json.dumps(
                        sample_record.get("context") or {}, ensure_ascii=False, sort_keys=True
                    ),
                    "gold_canonical_id": "",
                    "notes": _flag_notes(entity_type, raw_value, sample_record),
                }
            )
    print(f"[audit] wrote {len(keys)} rows → {path}")


def read_labeled_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ----- Metrics ---------------------------------------------------------------


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    skipped: int = 0
    unlabeled: int = 0

    @property
    def labeled(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def predicted_positive(self) -> int:
        return self.tp + self.fp

    @property
    def gold_positive(self) -> int:
        return self.tp + self.fn


def _classify(predicted: str, gold: str) -> str:
    """Return one of {tp, fp, fn, tn, skipped, unlabeled}."""
    gold = (gold or "").strip()
    predicted = (predicted or "").strip()
    if not gold:
        return "unlabeled"
    upper = gold.upper()
    if upper in {"SKIP", "IDK", "?"}:
        return "skipped"
    if upper == "NONE":
        return "tn" if not predicted else "fp"
    # gold is a real canonical id
    if not predicted:
        return "fn"
    return "tp" if predicted == gold else "fp"


def tally(rows: list[dict[str, str]], weight_by: str = "uniform") -> Counts:
    counts = Counts()
    for row in rows:
        bucket = _classify(
            row.get("predicted_canonical_id", ""), row.get("gold_canonical_id", "")
        )
        if weight_by == "row_count":
            try:
                w = int(row.get("row_count") or 0)
            except ValueError:
                w = 0
            w = max(w, 1)
        else:
            w = 1
        setattr(counts, bucket, getattr(counts, bucket) + w)
    return counts


def wilson_ci(numerator: int, denominator: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval. Returns (point_estimate, lo, hi). NaN-safe at n=0."""
    if denominator <= 0:
        return float("nan"), float("nan"), float("nan")
    p = numerator / denominator
    n = denominator
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / (1 + z2 / n)
    return p, max(0.0, center - margin), min(1.0, center + margin)


def format_metric(label: str, num: int, den: int) -> str:
    if den <= 0:
        return f"  {label:18s}  n/a (no denominator)"
    p, lo, hi = wilson_ci(num, den)
    return f"  {label:18s}  {p*100:6.2f}%  [{lo*100:5.2f}%, {hi*100:5.2f}%]   ({num}/{den})"


# ----- Strategy breakdown ----------------------------------------------------


def strategy_breakdown(rows: list[dict[str, str]]) -> dict[str, Counts]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("strategy") or "(missing)"].append(row)
    return {strategy: tally(rs, weight_by="uniform") for strategy, rs in grouped.items()}


def wrap_vs_registry_breakdown(rows: list[dict[str, str]]) -> str:
    """How often does the pipeline's wrapping change the answer vs the raw registry?"""
    same = different = pipe_only = registry_only = 0
    for row in rows:
        p = (row.get("predicted_canonical_id") or "").strip()
        r = (row.get("registry_canonical_id") or "").strip()
        if p == r:
            same += 1
        elif p and not r:
            pipe_only += 1
        elif r and not p:
            registry_only += 1
        else:
            different += 1
    total = same + different + pipe_only + registry_only
    if total == 0:
        return "  (no rows)"
    return (
        f"  pipeline == registry        : {same}/{total}\n"
        f"  pipeline != registry (both) : {different}/{total}\n"
        f"  pipeline only (registry =∅) : {pipe_only}/{total}\n"
        f"  registry only (pipeline =∅) : {registry_only}/{total}"
    )


# ----- Phase entrypoints -----------------------------------------------------


def run_sample(args: argparse.Namespace) -> None:
    eee_dir = Path(args.eee_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    sizes = {"model": args.n_model, "benchmark": args.n_benchmark, "metric": args.n_metric}

    pools = collect_samples(eee_dir)
    for entity_type in ENTITY_TYPES:
        n = sizes[entity_type]
        if n is not None and n < 0:
            n = None  # negative = "enumerate all"
        keys = sample_keys(pools[entity_type], n)
        path = out_dir / f"sample_{entity_type}.csv"
        write_sample_csv(path, entity_type, keys, pools[entity_type])

    print(
        f"\n[audit] sampling complete. Open the CSVs in {out_dir} and fill "
        "gold_canonical_id (NONE = no correct entry; SKIP = exclude). Then run analyze."
    )


def run_analyze(args: argparse.Namespace) -> None:
    in_dir = Path(args.in_dir).resolve()
    summary_lines: list[str] = []
    for entity_type in ENTITY_TYPES:
        path = in_dir / f"sample_{entity_type}.csv"
        if not path.exists():
            print(f"[audit] missing {path}, skipping")
            continue
        rows = read_labeled_csv(path)
        unweighted = tally(rows, weight_by="uniform")
        weighted = tally(rows, weight_by="row_count")

        print(f"\n=== {entity_type.upper()}  ({path.name}) ===")
        print(f"  rows in file        : {len(rows)}")
        print(f"  labeled             : {unweighted.labeled}")
        print(f"  skipped (SKIP/IDK)  : {unweighted.skipped}")
        print(f"  unlabeled (blank)   : {unweighted.unlabeled}")
        print(f"  TP={unweighted.tp}  FP={unweighted.fp}  FN={unweighted.fn}  TN={unweighted.tn}")

        print()
        print("  -- Per-unique-pair (sample-uniform) --")
        print(format_metric("precision", unweighted.tp, unweighted.predicted_positive))
        print(format_metric("recall", unweighted.tp, unweighted.gold_positive))
        print(format_metric("accuracy", unweighted.tp + unweighted.tn, unweighted.labeled))
        print(format_metric("resolution rate", unweighted.predicted_positive, unweighted.labeled))

        print()
        print("  -- Row-weighted (corpus-impact) --")
        print(format_metric("precision", weighted.tp, weighted.predicted_positive))
        print(format_metric("recall", weighted.tp, weighted.gold_positive))
        print(format_metric("accuracy", weighted.tp + weighted.tn, weighted.labeled))

        print()
        print("  -- Pipeline wrapping vs raw registry response --")
        print(wrap_vs_registry_breakdown(rows))

        print()
        print("  -- Per registry-strategy (unweighted) --")
        for strategy, counts in sorted(strategy_breakdown(rows).items()):
            if counts.labeled == 0:
                continue
            p_acc, lo, hi = wilson_ci(counts.tp + counts.tn, counts.labeled)
            print(
                f"    {strategy:18s}  acc {p_acc*100:6.2f}% "
                f"[{lo*100:5.2f}%, {hi*100:5.2f}%]   "
                f"TP={counts.tp} FP={counts.fp} FN={counts.fn} TN={counts.tn}"
            )

        p_unw, p_lo, p_hi = wilson_ci(unweighted.tp, unweighted.predicted_positive)
        r_unw, r_lo, r_hi = wilson_ci(unweighted.tp, unweighted.gold_positive)
        summary_lines.append(
            f"{entity_type:9s}  n={unweighted.labeled:4d}  "
            f"P={p_unw*100:5.2f}% [{p_lo*100:5.2f},{p_hi*100:5.2f}]  "
            f"R={r_unw*100:5.2f}% [{r_lo*100:5.2f},{r_hi*100:5.2f}]"
        )

    if summary_lines:
        print("\n=== Paper table (unweighted, Wilson 95% CI) ===")
        for line in summary_lines:
            print(line)


# ----- CLI -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample", help="Build CSVs of sampled pairs with predictions.")
    sp.add_argument("--eee-dir", default=str(DEFAULT_EEE_DIR))
    sp.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    sp.add_argument("--n-model", type=int, default=200, help="-1 to enumerate all unique pairs")
    sp.add_argument("--n-benchmark", type=int, default=200, help="-1 to enumerate all")
    sp.add_argument("--n-metric", type=int, default=-1, help="default: enumerate all (small vocab)")
    sp.set_defaults(func=run_sample)

    ap = sub.add_parser("analyze", help="Read labeled CSVs and print precision/recall.")
    ap.add_argument("--in-dir", default=str(DEFAULT_OUT_DIR))
    ap.set_defaults(func=run_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

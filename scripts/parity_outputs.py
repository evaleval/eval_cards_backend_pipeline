"""Parity-layer parquet emitter.

Materializes ``output/duckdb/v1/*.parquet`` from the in-memory pipeline
products. Each table carries scalar routing/sort columns plus a
frontend-ready JSON payload that already incorporates the cleaning
transforms in `scripts/parity.py` AND the post-TS-processing shapes in
`scripts/parity_adapters.py`.

Per PLAN_20260428.md line 55: "The payload must represent the object
after current TypeScript processing, not the raw backend object." We
satisfy that by routing every payload through the matching TS-adapter
port in `parity_adapters` (e.g. `hf_model_card_to_evaluation_card_data`).

This is additive: existing JSON artifacts remain untouched.

Writer: ``datasets.Dataset.from_dict(...).to_parquet(...)`` — the HF
``datasets`` library is the natural choice for a backend that publishes
to HF (already a sibling package of ``huggingface_hub``). It builds an
Arrow table directly from typed columns and writes parquet without a
SQL query engine in the loop. Earlier revisions used DuckDB as the
writer, which forced per-row prepared-statement bindings (or a CSV
detour); ``datasets`` does the equivalent in pure Arrow at C speed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Value

from scripts import parity, parity_adapters


PARITY_PARQUET_DIR = Path("duckdb/v1")
PARITY_SCHEMA_VERSION = 3


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_json(value: Any) -> str:
    """Serialize the payload as-is. We deliberately do NOT strip ``None``
    entries: the TS adapter intentionally emits ``null`` for fields like
    ``best_model`` / ``worst_model`` / ``top_score`` / ``params_billions``
    when no value is available, and ``JSON.stringify`` keeps those nulls.
    Stripping would lose that data signal. The parity verifier's
    ``_normalize_for_compare`` already treats ``None`` and "missing" as
    equivalent, so this doesn't introduce false-positive divergences."""
    return json.dumps(value, ensure_ascii=False)


def _first_present(*values: Any) -> Any:
    """JS `??`-chain — first non-None value. Preserves legitimate ``0``
    counts which `or`-coalescing would silently rewrite to a fallback."""
    for value in values:
        if value is not None:
            return value
    return None


def _scalar_columns(payload: dict, *, developer_route_id: str | None = None) -> dict:
    """Extract the parquet routing columns from a TS-shape payload.

    The TS adapter outputs use renamed keys (``id`` not ``model_family_id``,
    ``evaluation_id`` not ``eval_summary_id``). Use null-coalesce
    semantics so a legitimate ``models_count: 0`` is not silently
    overwritten by ``evaluations_count``.

    ``developer_route_id`` is passed in by the caller because it's
    derived from the *raw* developer string (lowercased + slugified) and
    the canonical TS shape doesn't carry it directly. We do NOT derive
    it from ``payload["developer"]``: the adapter's
    ``normalize_developer_name`` may rewrite the input (e.g.
    ``"deepseek-ai" → "DeepSeek"``) and slugifying the rewritten name
    would yield a different route id (``"deepseek"``) than the canonical
    ``"deepseek-ai"`` that downstream URLs expect.
    """
    return {
        "model_route_id": _first_present(
            payload.get("route_id"), payload.get("model_route_id")
        ),
        "model_family_id": _first_present(
            payload.get("id"),
            payload.get("model_family_id"),
            payload.get("model_id"),
        ),
        "eval_summary_id": _first_present(
            payload.get("evaluation_id"), payload.get("eval_summary_id")
        ),
        "developer_route_id": developer_route_id,
        "developer": payload.get("developer"),
        "category": payload.get("category"),
        "benchmark_family_key": _first_present(
            payload.get("benchmark_family_key"),
            payload.get("composite_benchmark_key"),
        ),
        "models_count": _first_present(
            payload.get("models_count"), payload.get("evaluations_count")
        ),
        "total_evaluations": _first_present(
            payload.get("total_evaluations"), payload.get("evaluations_count")
        ),
        "last_updated": _first_present(
            payload.get("last_updated"), payload.get("latest_timestamp")
        ),
    }


# ---------------------------------------------------------------------------
# Per-row payload builders — each delegates to a `parity_adapters` port
# ---------------------------------------------------------------------------


def build_model_card_payload(card: dict) -> dict:
    """Run setup-alias variant merging (matching the TS request-time
    `normalizeSingleModelCardEntry` in `lib/hf-data.ts:750`) and then the
    EvaluationCardData adapter, so the parquet payload matches what the
    user-facing API serves — not the raw 6-variant pipeline output.

    Routing keys (``developer_route_id``, ``model_route_id``) live in
    scalar parquet columns only, derived from the raw inputs by
    ``write_parity_artifacts``."""
    normalized = parity_adapters.normalize_single_model_card_entry(card)
    return parity_adapters.hf_model_card_to_evaluation_card_data(normalized)


def _is_example_eval_entry(entry: dict) -> bool:
    """Mirror the TS request-time filter in `getEvalListData` /
    `getEvalListLiteData` (lib/model-data.ts:1251, 1291): drop entries
    whose `source_data.hf_repo` starts with `example://` so demo rows
    don't surface in the user-facing list."""
    source_data = entry.get("source_data") or {}
    hf_repo = source_data.get("hf_repo")
    return isinstance(hf_repo, str) and hf_repo.startswith("example://")


def build_eval_list_payload(entry: dict, card_map: dict[str, dict]) -> dict:
    """Run the TS list-item adapter. The ``benchmark_card`` is the only
    enrichment we keep — it's part of the canonical TS shape (the JSON
    path attaches it at request time via ``getBenchmarkCard``)."""
    benchmark_card = entry.get("benchmark_card")
    if not benchmark_card:
        attached = parity.attach_benchmark_card_to_list_item(entry, card_map)
        benchmark_card = attached.get("benchmark_card")
    enriched = dict(entry)
    enriched["benchmark_card"] = benchmark_card
    return parity_adapters.hf_eval_entry_to_list_item(enriched)


def build_eval_summary_payload(eval_summary: dict, card_map: dict[str, dict]) -> dict:
    """Run the TS detail adapter with ``benchmark_card`` attached. No
    sidecars: ``dataset_url`` is already nested at ``source_data.dataset_url``
    where consumers read it; ``license_short`` had no consumer (the
    deleted ``components/eval-card.tsx``)."""
    benchmark_card = eval_summary.get("benchmark_card")
    if not benchmark_card:
        attached = parity.attach_benchmark_card_to_summary(eval_summary, card_map)
        benchmark_card = attached.get("benchmark_card")
    enriched = dict(eval_summary)
    enriched["benchmark_card"] = benchmark_card
    return parity_adapters.hf_eval_detail_to_summary(enriched)


def build_model_summary_payload(summary: dict) -> dict | None:
    """Run ``flatten_model_evaluations`` + ``create_model_family_summary``
    against the per-model JSON detail to produce the exact TS shape, or
    ``None`` for models with no evaluations.

    Matches the TS dump (``dump-adapter-outputs.mts``: ``if
    (evaluations.length === 0) return []``) — zero-evaluation models are
    absent from both surfaces, so the parquet row count tracks the
    renderable-model count, not the raw input count."""
    evaluations = parity_adapters.flatten_model_evaluations(summary)
    if not evaluations:
        return None
    return parity_adapters.create_model_family_summary(evaluations)


# ---------------------------------------------------------------------------
# Composite + matrix derivations — re-use the TS adapter ports for the
# input shape, then run the canonical aggregator over them.
# ---------------------------------------------------------------------------


def build_aggregate_eval_summaries(
    eval_summaries: list[dict],
    card_map: dict[str, dict],
) -> list[dict]:
    """Materialize ``aggregate__<suite_key>`` rows per spec reshape/05.

    Inputs are pipeline eval_summaries; we run each through
    ``hf_eval_detail_to_summary`` first to get ``BenchmarkEvalSummary``
    shape, then group by ``benchmark_family_key`` (matching
    ``eval-hierarchy.json family.key``) and feed each group to
    ``aggregate_benchmark_summaries``.
    """
    summary_by_id: dict[str, dict] = {}
    suite_groups: dict[str, list[str]] = {}
    for eval_summary in eval_summaries:
        suite_key = (
            eval_summary.get("benchmark_family_key")
            or eval_summary.get("benchmark_parent_key")
        )
        if not suite_key:
            continue
        eval_id = eval_summary.get("eval_summary_id")
        if not eval_id:
            continue

        # Attach card + run adapter once per eval_summary.
        if eval_id not in summary_by_id:
            attached = (
                parity.attach_benchmark_card_to_summary(eval_summary, card_map)
                if not eval_summary.get("benchmark_card")
                else eval_summary
            )
            summary_by_id[eval_id] = parity_adapters.hf_eval_detail_to_summary(attached)
        suite_groups.setdefault(str(suite_key), []).append(eval_id)

    rows: list[dict] = []
    for suite_key, eval_ids in suite_groups.items():
        # Spec quirk: skip suites with fewer than two distinct sub-evals.
        unique_ids = list(dict.fromkeys(eval_ids))
        if len(unique_ids) < 2:
            continue
        summaries = [summary_by_id[i] for i in unique_ids]
        aggregated = parity_adapters.aggregate_benchmark_summaries(summaries, suite_key)
        if aggregated is None:
            continue
        rows.append(aggregated)
    return rows


def build_matrix_eval_summaries(eval_summaries: list[dict]) -> list[dict]:
    """Materialize ``matrix__<suite_key>`` rows per spec reshape/06."""
    suite_groups: dict[str, list[dict]] = {}
    for eval_summary in eval_summaries:
        if eval_summary.get("is_summary_score"):
            continue
        suite_key = (
            eval_summary.get("benchmark_family_key")
            or eval_summary.get("benchmark_parent_key")
        )
        if not suite_key:
            continue
        suite_groups.setdefault(str(suite_key), []).append(eval_summary)

    rows: list[dict] = []
    for suite_key, details in suite_groups.items():
        result = parity_adapters.build_single_metric_suite_matrix_summary(details, suite_key)
        if result is None:
            continue
        rows.append(result)
    return rows


def _build_developer_summary_payload(summary: dict) -> dict:
    """Run `hf_developer_detail_to_summary` to produce the TS-shape
    payload that `getDeveloperSummaryById` returns: ``developer`` (canonical),
    ``route_id``, ``model_count``, ``benchmark_count``, ``evaluation_count``,
    ``popular_evals``, ``models[]`` (post-adapter EvaluationCardData)."""
    return parity_adapters.hf_developer_detail_to_summary(
        {"developer": summary.get("developer"), "models": summary.get("models") or []}
    )


def _build_developer_list_entry(summary: dict) -> dict:
    """Same as the summary payload but without the heavy `models[]` —
    matches `getDeveloperList()` shape."""
    return parity_adapters.hf_developer_detail_to_list_entry(
        {"developer": summary.get("developer"), "models": summary.get("models") or []}
    )


# ---------------------------------------------------------------------------
# Parquet emission — `datasets.Dataset.from_dict(...).to_parquet(...)`
# ---------------------------------------------------------------------------


# `datasets.Features` describe the on-disk types so a fresh consumer can
# `read_parquet` the file without re-inferring. Mapping mirrors what the
# old DuckDB schema declared.
_PARQUET_DTYPE_MAP: dict[str, str] = {
    "VARCHAR": "string",
    "BIGINT": "int64",
    "DOUBLE": "float64",
    "BOOLEAN": "bool",
}


def _write_parquet(
    output_path: Path,
    columns: list[tuple[str, str]],
    rows_iter: Any,
) -> None:
    """Write a parquet file via `datasets.Dataset`.

    `columns` is a list of `(name, sql_kind)` tuples; `rows_iter` yields
    dicts keyed by column name. Cells get coerced per the column dtype
    (`VARCHAR` → str, `BIGINT` → int, `DOUBLE` → float, `BOOLEAN` → bool).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    column_table: dict[str, list[Any]] = {name: [] for name, _ in columns}
    coercers = {
        "VARCHAR": _to_str,
        "BIGINT": _to_int,
        "DOUBLE": _to_float,
        "BOOLEAN": lambda v: bool(v) if v is not None else False,
    }
    for row in rows_iter:
        for name, kind in columns:
            column_table[name].append(coercers[kind](row.get(name)))

    features = Features(
        {name: Value(_PARQUET_DTYPE_MAP[kind]) for name, kind in columns}
    )
    Dataset.from_dict(column_table, features=features).to_parquet(str(output_path))


_PAYLOAD_TABLE_COLUMNS: list[tuple[str, str]] = [
    ("record_type", "VARCHAR"),
    ("model_route_id", "VARCHAR"),
    ("model_family_id", "VARCHAR"),
    ("eval_summary_id", "VARCHAR"),
    ("developer_route_id", "VARCHAR"),
    ("developer", "VARCHAR"),
    ("category", "VARCHAR"),
    ("benchmark_family_key", "VARCHAR"),
    ("models_count", "BIGINT"),
    ("total_evaluations", "BIGINT"),
    ("last_updated", "VARCHAR"),
    ("payload_json", "VARCHAR"),
]


def _emit_table(
    output_path: Path,
    rows: "list[tuple[dict, dict | None]]",
    *,
    record_type: str,
) -> None:
    """Write a single parquet table with scalar routing columns + payload.

    ``rows`` is a list of ``(payload, scalar_overrides)`` tuples.
    ``scalar_overrides`` (optional) supplies values that can't be derived
    from the canonical TS-shape payload — currently only
    ``developer_route_id`` (which slugifies the *raw* developer name; the
    payload's normalized one would slugify differently)."""

    def _row_iter() -> Any:
        for payload, overrides in rows:
            scalars = _scalar_columns(payload, **(overrides or {}))
            yield {
                "record_type": record_type,
                "model_route_id": scalars["model_route_id"],
                "model_family_id": scalars["model_family_id"],
                "eval_summary_id": scalars["eval_summary_id"],
                "developer_route_id": scalars["developer_route_id"],
                "developer": scalars["developer"],
                "category": scalars["category"],
                "benchmark_family_key": scalars["benchmark_family_key"],
                "models_count": scalars["models_count"],
                "total_evaluations": scalars["total_evaluations"],
                "last_updated": scalars["last_updated"],
                "payload_json": _payload_json(payload),
            }

    _write_parquet(output_path, _PAYLOAD_TABLE_COLUMNS, _row_iter())


def write_parity_artifacts(
    *,
    model_cards: list[dict],
    lite_model_cards: list[dict],
    eval_list: dict,
    lite_eval_list: dict,
    eval_summaries: list[dict],
    model_summaries: list[dict],
    dev_summaries: list[dict],
    benchmark_metadata: dict,
    output_dir: Path,
) -> None:
    """Materialize all parity parquet tables under ``output_dir / duckdb / v1 /``.

    Idempotent — table layout is fixed, files are rewritten on each run.
    """
    parity_dir = output_dir / PARITY_PARQUET_DIR

    cards = (benchmark_metadata or {}).get("cards") if isinstance(benchmark_metadata, dict) else None
    if not isinstance(cards, dict):
        cards = {}
    card_map = parity.build_benchmark_card_map(cards)

    def _dev_route_id(raw_developer: Any) -> dict:
        return {"developer_route_id": parity.get_developer_route_id(raw_developer or "")}

    model_card_rows = [
        (build_model_card_payload(card), _dev_route_id(card.get("developer")))
        for card in model_cards
    ]
    lite_card_rows = [
        (build_model_card_payload(card), _dev_route_id(card.get("developer")))
        for card in lite_model_cards
    ]

    # eval_list / eval_summary surfaces don't carry a developer at the
    # row level — the scalar developer_route_id column is null for them.
    eval_list_rows = [
        (build_eval_list_payload(entry, card_map), None)
        for entry in (eval_list.get("evals") or [])
        if not _is_example_eval_entry(entry)
    ]
    lite_eval_list_rows = [
        (build_eval_list_payload(entry, card_map), None)
        for entry in (lite_eval_list.get("evals") or [])
        if not _is_example_eval_entry(entry)
    ]
    eval_summary_rows = [
        (build_eval_summary_payload(summary, card_map), None)
        for summary in eval_summaries
    ]

    model_summary_rows: list[tuple[dict, dict | None]] = []
    for summary in model_summaries:
        payload = build_model_summary_payload(summary)
        if payload is None:
            continue
        model_summary_rows.append(
            (payload, _dev_route_id((summary.get("model_info") or {}).get("developer")))
        )

    aggregate_rows = [(row, None) for row in build_aggregate_eval_summaries(eval_summaries, card_map)]
    matrix_rows = [(row, None) for row in build_matrix_eval_summaries(eval_summaries)]

    developer_rows = [
        (_build_developer_list_entry(summary), _dev_route_id(summary.get("developer")))
        for summary in dev_summaries
    ]
    developer_summary_rows = [
        (_build_developer_summary_payload(summary), _dev_route_id(summary.get("developer")))
        for summary in dev_summaries
    ]

    _emit_table(parity_dir / "model_cards.parquet", model_card_rows, record_type="model_card")
    _emit_table(parity_dir / "model_cards_lite.parquet", lite_card_rows, record_type="model_card_lite")
    _emit_table(parity_dir / "eval_list.parquet", eval_list_rows, record_type="eval_list_entry")
    _emit_table(parity_dir / "eval_list_lite.parquet", lite_eval_list_rows, record_type="eval_list_lite_entry")
    _emit_table(parity_dir / "eval_summaries.parquet", eval_summary_rows, record_type="eval_summary")
    _emit_table(parity_dir / "aggregate_eval_summaries.parquet", aggregate_rows, record_type="aggregate_eval_summary")
    _emit_table(parity_dir / "matrix_eval_summaries.parquet", matrix_rows, record_type="matrix_eval_summary")
    _emit_table(parity_dir / "model_summaries.parquet", model_summary_rows, record_type="model_summary")
    _emit_table(parity_dir / "developers.parquet", developer_rows, record_type="developer")
    _emit_table(parity_dir / "developer_summaries.parquet", developer_summary_rows, record_type="developer_summary")

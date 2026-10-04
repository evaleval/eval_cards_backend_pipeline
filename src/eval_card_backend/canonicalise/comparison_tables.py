"""The comparison index as three parquet tables, and back.

The pipeline writes the index only as these tables. Snapshots produced
before that also carry it as `comparison-index.json`, with the same content.

`flatten` takes the index in its JSON form (non-finite floats already
spelled "Infinity" / "-Infinity") and returns one table per
level: evals, metrics (one row per (evaluation_id, metric_summary_id)) and
score cells. `unflatten` rebuilds the identical dict, `by_model` included,
which is not stored: it is a re-pivot of the score rows.

Key presence is decided by the kind of the parent eval, never by NULL. A
merged eval (the only kind that carries the `is_merged` key) has cells with
`source_composite_slug`; a per-source eval has cells with `score_canonical`,
`scale_conversion` and `split`. Within a kind every key is present and may
be null, so a NULL column reads back as a present null. `flatten` raises on
any dict whose key set or value types the tables cannot reproduce.

No pipeline imports, so the same functions run on any comparison-index.json.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

EVALS_TABLE = "comparison_evals"
METRICS_TABLE = "comparison_metrics"
SCORES_TABLE = "comparison_scores"
TABLE_FILES = tuple(f"{t}.parquet" for t in (EVALS_TABLE, METRICS_TABLE, SCORES_TABLE))

_STR = pa.string()
_STR_LIST = pa.list_(pa.string())

TOP_LEVEL_FIELDS = [
    ("comparison_index_version", pa.int32()),
    ("config_version", pa.int32()),
    ("generated_at", _STR),
    ("metric_group_order", _STR_LIST),
]
EVAL_FIELDS = [
    ("evaluation_id", _STR),
    ("composite_slug", _STR),
    ("composite_display_name", _STR),
    ("benchmark_id", _STR),
    ("family_id", _STR),
    ("family_display_name", _STR),
    ("parent_benchmark_id", _STR),
    ("is_slice", pa.bool_()),
    ("display_name", _STR),
    ("derived_tags", _STR_LIST),
    ("is_summary_score", pa.bool_()),
    ("summary_score_for", _STR),
]
METRIC_FIELDS = [
    ("metric_summary_id", _STR),
    ("metric_name", _STR),
    ("metric_id", _STR),
    ("metric_key", _STR),
    ("group", _STR),
    ("group_order", pa.int32()),
    ("lower_is_better", pa.bool_()),
    ("unit", _STR),
    ("canonical_min_score", pa.float64()),
    ("canonical_max_score", pa.float64()),
]
CELL_FIELDS = [
    ("model_route_id", _STR),
    ("model_family_id", _STR),
    ("model_family_name", _STR),
    ("developer", _STR),
    ("variant_key", _STR),
    ("score", pa.float64()),
    ("score_canonical", pa.float64()),
    ("scale_conversion", _STR),
    ("rank", pa.int32()),
    ("total", pa.int32()),
    ("split", _STR),
    ("submission_count", pa.int32()),
    ("submission_axis", _STR),
    ("temperature", pa.float64()),
    ("max_tokens", pa.int32()),
    ("source_composite_slug", _STR),
]

_PER_SOURCE_ONLY = {"score_canonical", "scale_conversion", "split"}
_MERGED_ONLY = {"source_composite_slug"}
_ALL_CELL_KEYS = {name for name, _ in CELL_FIELDS}
_CELL_KEYS = {
    False: _ALL_CELL_KEYS - _MERGED_ONLY,
    True: _ALL_CELL_KEYS - _PER_SOURCE_ONLY,
}
_BY_MODEL_KEYS = {
    False: ("score", "score_canonical", "scale_conversion", "rank", "total",
            "submission_count", "submission_axis", "temperature", "max_tokens"),
    True: ("score", "rank", "total", "submission_count", "submission_axis",
           "temperature", "max_tokens"),
}
_TOP_LEVEL_KEYS = {name for name, _ in TOP_LEVEL_FIELDS} | {"evals", "by_model"}
_EVAL_KEYS = {name for name, _ in EVAL_FIELDS} | {"metrics"}
_METRIC_KEYS = {name for name, _ in METRIC_FIELDS} | {"scores"}

_SNAPSHOT_FIELD = ("snapshot_id", pa.timestamp("us"))
_ORD = pa.int32()

EVALS_SCHEMA = pa.schema(
    [_SNAPSHOT_FIELD] + EVAL_FIELDS[:8] + [("is_merged", pa.bool_())]
    + EVAL_FIELDS[8:] + TOP_LEVEL_FIELDS
)
METRICS_SCHEMA = pa.schema(
    [_SNAPSHOT_FIELD, ("evaluation_id", _STR), METRIC_FIELDS[0], ("metric_ord", _ORD)]
    + METRIC_FIELDS[1:]
)
SCORES_SCHEMA = pa.schema(
    [_SNAPSHOT_FIELD, ("evaluation_id", _STR), ("metric_summary_id", _STR),
     ("row_ord", _ORD)] + CELL_FIELDS
)

_NON_FINITE = {"Infinity": math.inf, "-Infinity": -math.inf}

# The top-level fields also ride in the evals schema metadata, so an index
# with no evals still round-trips.
_TOP_LEVEL_METADATA_KEY = b"comparison_index_top_level"


def _snapshot_ts(snapshot_id: str) -> datetime:
    # Same value as the DuckDB-written tables' `TIMESTAMP '<id>'`, which
    # keeps the wall-clock time and drops any UTC offset.
    return datetime.fromisoformat(snapshot_id.removesuffix("Z")).replace(tzinfo=None)


def _check_keys(obj: dict, expected: set[str], where: str) -> None:
    keys = set(obj)
    if keys != expected:
        raise ValueError(
            f"{where}: key set does not match its kind "
            f"(missing {sorted(expected - keys)}, extra {sorted(keys - expected)})"
        )


def _value(value, typ: pa.DataType, where: str):
    """Validate one JSON value against its column type; returns the value to
    store. Only values that read back identically are accepted."""
    if value is None:
        return None
    if typ == _STR:
        ok = isinstance(value, str)
    elif typ == pa.bool_():
        ok = isinstance(value, bool)
    elif pa.types.is_integer(typ):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif typ == pa.float64():
        if isinstance(value, str) and value in _NON_FINITE:
            return _NON_FINITE[value]
        ok = isinstance(value, float) and math.isfinite(value)
    elif typ == _STR_LIST:
        ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
    else:
        ok = False
    if not ok:
        raise ValueError(f"{where}: {value!r} cannot be stored as {typ}")
    return value


def _json_value(value):
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def flatten(index: dict, snapshot_id: str) -> dict[str, pa.Table]:
    """Return {table name: table} for a comparison index in its JSON form."""
    _check_keys(index, _TOP_LEVEL_KEYS, "comparison index")
    top = {
        name: _value(index[name], typ, name) for name, typ in TOP_LEVEL_FIELDS
    }
    sid = _snapshot_ts(snapshot_id)
    ev_cols = {f.name: [] for f in EVALS_SCHEMA}
    me_cols = {f.name: [] for f in METRICS_SCHEMA}
    sc_cols = {f.name: [] for f in SCORES_SCHEMA}
    by_model: dict[str, dict[str, dict[str, dict]]] = {}

    for eval_id in sorted(index["evals"]):
        entry = index["evals"][eval_id]
        merged = "is_merged" in entry
        if merged and entry["is_merged"] is not True:
            raise ValueError(f"{eval_id}: is_merged present but not true")
        _check_keys(entry, _EVAL_KEYS | ({"is_merged"} if merged else set()), eval_id)
        if entry["evaluation_id"] != eval_id:
            raise ValueError(f"{eval_id}: entry carries evaluation_id {entry['evaluation_id']!r}")
        ev_cols["snapshot_id"].append(sid)
        ev_cols["is_merged"].append(merged)
        for name, typ in EVAL_FIELDS:
            ev_cols[name].append(_value(entry[name], typ, f"{eval_id}.{name}"))
        for name, _ in TOP_LEVEL_FIELDS:
            ev_cols[name].append(top[name])

        seen_msids = set()
        for metric_ord, metric in enumerate(entry["metrics"]):
            msid = metric.get("metric_summary_id")
            where = f"{eval_id}/{msid}"
            _check_keys(metric, _METRIC_KEYS, where)
            if msid in seen_msids:
                raise ValueError(f"{where}: metric_summary_id repeats within the eval")
            seen_msids.add(msid)
            me_cols["snapshot_id"].append(sid)
            me_cols["evaluation_id"].append(eval_id)
            me_cols["metric_ord"].append(metric_ord)
            for name, typ in METRIC_FIELDS:
                me_cols[name].append(_value(metric[name], typ, f"{where}.{name}"))

            cell_keys = _CELL_KEYS[merged]
            for row_ord, cell in enumerate(metric["scores"]):
                _check_keys(cell, cell_keys, f"{where} cell {row_ord}")
                sc_cols["snapshot_id"].append(sid)
                sc_cols["evaluation_id"].append(eval_id)
                sc_cols["metric_summary_id"].append(msid)
                sc_cols["row_ord"].append(row_ord)
                for name, typ in CELL_FIELDS:
                    sc_cols[name].append(
                        _value(cell[name], typ, f"{where} cell {row_ord}.{name}")
                        if name in cell_keys else None
                    )
                by_model.setdefault(cell["model_route_id"], {}).setdefault(
                    eval_id, {}
                )[msid] = {k: cell[k] for k in _BY_MODEL_KEYS[merged]}

    if by_model != index["by_model"]:
        raise ValueError("by_model is not the re-pivot of the score cells")

    evals_schema = EVALS_SCHEMA.with_metadata(
        {_TOP_LEVEL_METADATA_KEY: json.dumps(top, sort_keys=True)}
    )
    return {
        EVALS_TABLE: pa.table(ev_cols, schema=evals_schema),
        METRICS_TABLE: pa.table(me_cols, schema=METRICS_SCHEMA),
        SCORES_TABLE: pa.table(sc_cols, schema=SCORES_SCHEMA),
    }


def unflatten(tables: dict[str, pa.Table]) -> dict:
    """Rebuild the comparison index, in its JSON form, from the three tables."""
    evals_t = tables[EVALS_TABLE].sort_by("evaluation_id")
    metrics_t = tables[METRICS_TABLE].sort_by(
        [("evaluation_id", "ascending"), ("metric_ord", "ascending")]
    )
    scores_t = tables[SCORES_TABLE].sort_by(
        [("evaluation_id", "ascending"), ("metric_summary_id", "ascending"),
         ("row_ord", "ascending")]
    )

    def columns(table: pa.Table, names) -> list[list]:
        return [table.column(n).to_pylist() for n in names]

    top_names = [name for name, _ in TOP_LEVEL_FIELDS]
    if evals_t.num_rows:
        index: dict = {
            name: col[0] for name, col in zip(top_names, columns(evals_t, top_names))
        }
    else:
        metadata = tables[EVALS_TABLE].schema.metadata or {}
        if _TOP_LEVEL_METADATA_KEY not in metadata:
            raise ValueError(f"{EVALS_TABLE} has no rows and no top-level metadata")
        index = json.loads(metadata[_TOP_LEVEL_METADATA_KEY])

    evals: dict[str, dict] = {}
    names = ["is_merged"] + [name for name, _ in EVAL_FIELDS]
    for row in zip(*columns(evals_t, names)):
        entry = dict(zip(names, row))
        if entry.pop("is_merged"):
            entry["is_merged"] = True
        entry["metrics"] = []
        evals[entry["evaluation_id"]] = entry

    metric_by_key: dict[tuple[str, str], dict] = {}
    names = ["evaluation_id", "metric_ord"] + [name for name, _ in METRIC_FIELDS]
    for row in zip(*columns(metrics_t, names)):
        rec = dict(zip(names, row))
        eval_id = rec.pop("evaluation_id")
        metrics = evals[eval_id]["metrics"]
        if rec.pop("metric_ord") != len(metrics):
            raise ValueError(f"{eval_id}: metric_ord is not a dense 0-based sequence")
        metric = {k: _json_value(v) for k, v in rec.items()}
        metric["scores"] = []
        metrics.append(metric)
        metric_by_key[(eval_id, metric["metric_summary_id"])] = metric

    by_model: dict[str, dict[str, dict[str, dict]]] = {}
    names = ["evaluation_id", "metric_summary_id", "row_ord"] + [n for n, _ in CELL_FIELDS]
    for row in zip(*columns(scores_t, names)):
        rec = dict(zip(names, row))
        eval_id = rec["evaluation_id"]
        msid = rec["metric_summary_id"]
        merged = "is_merged" in evals[eval_id]
        scores = metric_by_key[(eval_id, msid)]["scores"]
        if rec["row_ord"] != len(scores):
            raise ValueError(f"{eval_id}/{msid}: row_ord is not a dense 0-based sequence")
        cell = {k: _json_value(rec[k]) for k in _CELL_KEYS[merged]}
        scores.append(cell)
        by_model.setdefault(cell["model_route_id"], {}).setdefault(
            eval_id, {}
        )[msid] = {k: cell[k] for k in _BY_MODEL_KEYS[merged]}

    index["evals"] = evals
    index["by_model"] = by_model
    return index


def write_tables(index: dict, out_dir: Path, snapshot_id: str) -> list[Path]:
    paths = []
    for name, table in flatten(index, snapshot_id).items():
        path = Path(out_dir) / f"{name}.parquet"
        pq.write_table(table, path, compression="zstd")
        paths.append(path)
    return paths


def read_tables(snapshot_dir: Path) -> dict[str, pa.Table]:
    return {
        name: pq.read_table(Path(snapshot_dir) / f"{name}.parquet")
        for name in (EVALS_TABLE, METRICS_TABLE, SCORES_TABLE)
    }


def read_index(snapshot_dir: Path) -> dict:
    """The comparison index of a snapshot, rebuilt from its three tables."""
    return unflatten(read_tables(snapshot_dir))

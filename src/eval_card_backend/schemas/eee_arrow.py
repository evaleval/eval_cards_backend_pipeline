"""Translate the vendored EEE JSON Schema into a pyarrow.Schema.

The translator handles the four pyarrow-hostile patterns in the upstream
schema:

1.  `oneOf` (e.g. `evaluation_results[].source_data`): pyarrow has no native
    union type that survives Parquet round-tripping cleanly. Collapsed to
    a JSON-encoded VARCHAR. The upstream contract still validates the shape
    on read via the vendored Pydantic models.

2.  `additionalProperties: {"type": "string"}` (e.g. `*.additional_details`):
    becomes `MAP<string, string>`.

3.  `additionalProperties` with a union value type (`metric_parameters` allows
    string|number|boolean|null): pyarrow MAP can't carry a union value, so
    the field is JSON-encoded as VARCHAR.

4.  Deeply-nested optional STRUCTs: every leaf is nullable, and every
    optional STRUCT is itself nullable. The cast caller pads missing keys
    with None before constructing the RecordBatch.

`$ref` is resolved against the schema's top-level `$defs`.

`derive_pyarrow_schema()` re-walks the vendored JSON Schema each call (cheap;
sub-millisecond). When upstream bumps and we re-vendor, the next call picks
up the change without a separate codegen step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

_DUMP = lambda v: json.dumps(v, ensure_ascii=False, default=str)  # noqa: E731

_REGISTRY = Path(__file__).resolve().parent.parent / "registry"
_SCHEMA_PATH = _REGISTRY / "eee.schema.json"

# Top-level fields whose value should be stored as a JSON-encoded string
# rather than a structured pyarrow type. Used for fields with `oneOf`,
# union value types, or shapes that pyarrow can't represent losslessly.
_FORCE_JSON_STRING_PATHS: frozenset[str] = frozenset(
    {
        # source_data has a discriminated oneOf; keep it as raw JSON (Stage D
        # can extract specific fields via json_extract_string when needed).
        "evaluation_results.items.source_data",
        # metric_parameters allows union value types (str|number|bool|null);
        # pyarrow MAP requires a uniform value type.
        "evaluation_results.items.metric_config.metric_parameters",
    }
)


def load_eee_json_schema(path: Path | str | None = None) -> dict[str, Any]:
    schema_path = Path(path) if path else _SCHEMA_PATH
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref: {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _normalise_type(node: dict[str, Any]) -> str | None:
    """Resolve `"type"` when it's a list like `["null", "number"]`."""
    t = node.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        return non_null[0] if non_null else None
    return t


def _is_str_str_map(node: dict[str, Any]) -> bool:
    ap = node.get("additionalProperties")
    if not isinstance(ap, dict):
        return False
    return ap.get("type") == "string"


def _translate(node: dict[str, Any], root: dict[str, Any], path: str) -> pa.DataType:
    if "$ref" in node:
        return _translate(_resolve_ref(node["$ref"], root), root, path)

    if path in _FORCE_JSON_STRING_PATHS:
        return pa.string()

    if "oneOf" in node or "anyOf" in node:
        return pa.string()

    t = _normalise_type(node)

    if t == "string":
        return pa.string()
    if t == "integer":
        return pa.int64()
    if t == "number":
        return pa.float64()
    if t == "boolean":
        return pa.bool_()

    if t == "array":
        items = node.get("items", {})
        return pa.list_(_translate(items, root, path + ".items"))

    if t == "object":
        if "properties" in node and node["properties"]:
            fields = []
            for name, sub in node["properties"].items():
                child_path = f"{path}.{name}" if path else name
                fields.append(
                    pa.field(name, _translate(sub, root, child_path), nullable=True)
                )
            return pa.struct(fields)
        if _is_str_str_map(node):
            return pa.map_(pa.string(), pa.string())
        # Object with `additionalProperties` of a non-string-uniform type
        # (or unspecified) — fall through to JSON string.
        return pa.string()

    # Untyped or unknown: store the raw JSON value as a string.
    return pa.string()


def derive_pyarrow_schema(json_schema: dict[str, Any] | None = None) -> pa.Schema:
    """Translate the EEE JSON Schema into a pyarrow.Schema.

    The returned schema covers every top-level field declared in the
    upstream contract. Required fields are emitted with `nullable=False`;
    optional fields with `nullable=True`. Nested STRUCT leaves are all
    nullable so absent keys cast to NULL rather than raising.
    """
    schema = json_schema or load_eee_json_schema()
    required = set(schema.get("required", []))
    fields = []
    for name, sub in schema.get("properties", {}).items():
        dtype = _translate(sub, schema, name)
        fields.append(pa.field(name, dtype, nullable=name not in required))
    return pa.schema(fields)


def schema_version(json_schema: dict[str, Any] | None = None) -> str | None:
    """Return the upstream `version` field from the JSON Schema."""
    schema = json_schema or load_eee_json_schema()
    return schema.get("version")


def pad_record_for_cast(record: dict[str, Any], schema: pa.Schema) -> dict[str, Any]:
    """Recursively fill in missing keys so a record can be cast to `schema`.

    pyarrow's `RecordBatch.from_pylist` expects every record to have every
    key declared in the schema (missing keys raise rather than fill NULL).
    This helper walks the schema and pads with `None` / empty containers
    where a record omits an optional field. Records keep their existing
    values; this only adds keys that aren't there.
    """
    return {field.name: _pad(record.get(field.name), field.type) for field in schema}


def _pad(value: Any, dtype: pa.DataType) -> Any:
    if value is None:
        return None
    if pa.types.is_struct(dtype):
        if not isinstance(value, dict):
            return None
        return {
            f.name: _pad(value.get(f.name), f.type)
            for f in [dtype.field(i) for i in range(dtype.num_fields)]
        }
    if pa.types.is_list(dtype):
        if not isinstance(value, list):
            return None
        item_type = dtype.value_type
        return [_pad(v, item_type) for v in value]
    if pa.types.is_map(dtype):
        if not isinstance(value, dict):
            return None
        # MAP<string, string> only — the only MAP shape we emit. Nested
        # dicts/lists in the value position get JSON-encoded so downstream
        # parsers see valid JSON instead of Python repr (`"{'k': 'v'}"`).
        def _coerce_map_value(v):
            if v is None:
                return None
            if isinstance(v, str):
                return v
            if isinstance(v, (dict, list)):
                return _DUMP(v)
            return str(v)
        return [(str(k), _coerce_map_value(v)) for k, v in value.items()]
    if pa.types.is_string(dtype) and isinstance(value, (dict, list)):
        # Fields collapsed to JSON-string by the translator (oneOf,
        # union value types) arrive as dicts/lists; encode them on the way in.
        return _DUMP(value)
    # Scalars are passed through; pyarrow handles type coercion / NULL on cast.
    return value

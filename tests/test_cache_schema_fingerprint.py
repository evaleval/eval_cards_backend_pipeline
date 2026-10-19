"""The stage cache's table shapes are pinned to `CACHE_SCHEMA_VERSION`.

`--from-stage` restores cached parquets written by an earlier run. When a
stage output gains, loses or re-types a column and the version is not
bumped, the restored tables have the old shape and a later stage fails
deep inside its SQL, or silently reads the old columns. This test
fingerprints every table in `STAGE_OUTPUTS` after a fixture-corpus run and
compares it with `tests/cache_schema_fingerprint.json`, which records the
shapes together with the version they were generated at.

Regenerate the fingerprint (after bumping the version) with:

    UPDATE_CACHE_FINGERPRINT=1 HF_HUB_OFFLINE=1 uv run pytest tests/test_cache_schema_fingerprint.py -q
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pytest

from eval_card_backend.canonicalise.cache import (
    CACHE_SCHEMA_VERSION,
    STAGE_ORDER,
    STAGE_OUTPUTS,
)

FINGERPRINT_PATH = Path(__file__).parent / "cache_schema_fingerprint.json"
REGENERATE_CMD = (
    "UPDATE_CACHE_FINGERPRINT=1 HF_HUB_OFFLINE=1 uv run pytest "
    "tests/test_cache_schema_fingerprint.py -q"
)
ALL_TABLES: tuple[str, ...] = tuple(
    table for stage in STAGE_ORDER for table in STAGE_OUTPUTS[stage]
)

Shape = list[list[str]]  # [[column_name, column_type], ...] in column order


def _parquet_shape(path: Path) -> Shape:
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{path}')"
        ).fetchall()
    finally:
        con.close()
    return [[name, col_type] for name, col_type, *_ in rows]


@pytest.fixture(scope="module")
def actual_shapes(fixture_corpus_build) -> dict[str, Shape | None]:
    """Shape of every STAGE_OUTPUTS table in the fixture run's cache;
    `None` when the run did not cache that table. With
    UPDATE_CACHE_FINGERPRINT=1 the fingerprint file is rewritten first."""
    shapes: dict[str, Shape | None] = {}
    for table in ALL_TABLES:
        path = fixture_corpus_build.cache_dir / f"{table}.parquet"
        shapes[table] = _parquet_shape(path) if path.exists() else None
    if os.environ.get("UPDATE_CACHE_FINGERPRINT") == "1":
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "tables": {
                table: shape for table, shape in sorted(shapes.items())
                if shape is not None
            },
        }
        FINGERPRINT_PATH.write_text(_render_fingerprint(payload))
    return shapes


def _render_fingerprint(payload: dict) -> str:
    """One `["column", "TYPE"]` pair per line, so a diff of the file reads
    as a column-level diff."""
    lines = [
        "{",
        f'  "cache_schema_version": {payload["cache_schema_version"]},',
        '  "tables": {',
    ]
    tables = list(payload["tables"].items())
    for i, (table, shape) in enumerate(tables):
        lines.append(f'    "{table}": [')
        for j, pair in enumerate(shape):
            comma = "," if j < len(shape) - 1 else ""
            lines.append(f"      {json.dumps(pair)}{comma}")
        lines.append("    ]" + ("," if i < len(tables) - 1 else ""))
    lines += ["  }", "}", ""]
    return "\n".join(lines)


@pytest.fixture(scope="module")
def fingerprint(actual_shapes) -> dict:
    """Depends on `actual_shapes` so a regeneration run writes the file
    before it is read."""
    assert FINGERPRINT_PATH.exists(), (
        f"{FINGERPRINT_PATH.name} is missing; generate it with `{REGENERATE_CMD}`"
    )
    return json.loads(FINGERPRINT_PATH.read_text())


def _version_hint(fingerprint: dict) -> str:
    recorded = fingerprint["cache_schema_version"]
    if recorded == CACHE_SCHEMA_VERSION:
        return (
            f"CACHE_SCHEMA_VERSION is still {CACHE_SCHEMA_VERSION}; bump it in "
            f"cache.py and regenerate the fingerprint with `{REGENERATE_CMD}`"
        )
    return (
        f"CACHE_SCHEMA_VERSION was bumped to {CACHE_SCHEMA_VERSION} but the "
        f"fingerprint still records {recorded}; regenerate it with "
        f"`{REGENERATE_CMD}`"
    )


def test_fingerprint_version_matches_code(fingerprint):
    recorded = fingerprint["cache_schema_version"]
    assert recorded == CACHE_SCHEMA_VERSION, (
        f"CACHE_SCHEMA_VERSION is {CACHE_SCHEMA_VERSION} in cache.py but "
        f"{FINGERPRINT_PATH.name} was generated at {recorded}; regenerate it "
        f"with `{REGENERATE_CMD}`"
    )


def test_fingerprint_covers_every_cached_table(fingerprint, actual_shapes):
    """Tables declared in STAGE_OUTPUTS and produced by the fixture run must
    all be fingerprinted, and nothing else may be."""
    produced = {t for t, shape in actual_shapes.items() if shape is not None}
    recorded = set(fingerprint["tables"])
    assert recorded == produced, (
        f"fingerprint table set differs from the cached tables: "
        f"not fingerprinted {sorted(produced - recorded)}, "
        f"no longer cached {sorted(recorded - produced)}; "
        f"{_version_hint(fingerprint)}"
    )


@pytest.mark.parametrize("table", ALL_TABLES)
def test_cached_shape_is_fingerprinted(table, fingerprint, actual_shapes):
    actual = actual_shapes[table]
    if actual is None:
        pytest.skip(
            f"{table} is declared in STAGE_OUTPUTS but the fixture corpus run "
            f"did not cache it"
        )
    expected = fingerprint["tables"].get(table)
    assert expected is not None, (
        f"{table} has no entry in {FINGERPRINT_PATH.name}; {_version_hint(fingerprint)}"
    )
    if actual == expected:
        return

    actual_types = dict(map(tuple, actual))
    expected_types = dict(map(tuple, expected))
    added = sorted(set(actual_types) - set(expected_types))
    removed = sorted(set(expected_types) - set(actual_types))
    retyped = sorted(
        f"{col}: {expected_types[col]} -> {actual_types[col]}"
        for col in set(actual_types) & set(expected_types)
        if actual_types[col] != expected_types[col]
    )
    changes = [
        f"added {added}" if added else "",
        f"removed {removed}" if removed else "",
        f"retyped {retyped}" if retyped else "",
    ]
    if not any(changes):
        changes = [
            "columns reordered: "
            f"{[c for c, _ in actual]} vs {[c for c, _ in expected]}"
        ]
    pytest.fail(
        f"the cached shape of {table} changed ("
        + "; ".join(c for c in changes if c)
        + f") and {_version_hint(fingerprint)}"
    )

"""Two builds of the same corpus must produce byte-identical warehouses.

Unordered aggregations (`ANY_VALUE`, `ARRAY_AGG` without ORDER BY, `COPY`
of an unsorted relation, global-RNG dependence) make a rebuild differ
from the previous one without any input change, which defeats snapshot
diffing and the sync workflow's no-op detection. The existing tests cover
individual sites; this one bakes the whole fixture corpus a second time,
with the same snapshot id, and compares every parquet and JSON file of the
two warehouses by sha256.

`snapshot_meta.json` is compared with its `generated_at` key removed: that
is the wall-clock stamp of the run itself, the one field that is allowed to
differ between two builds of one snapshot.

On a mismatch the failure lists, per file, the first differing rows (for
parquet, via DuckDB) or the first differing key path (for JSON).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from tests.fixture_corpus import build_fixture_corpus

VOLATILE_JSON_KEYS: dict[str, frozenset[str]] = {
    "snapshot_meta.json": frozenset({"generated_at"}),
}


@pytest.fixture(scope="module")
def second_build(tmp_path_factory, _taxonomy_seed_stub, _collections_stub):
    return build_fixture_corpus(
        tmp_path_factory.mktemp("fixture_corpus_rebuild"),
        seed_dir=_taxonomy_seed_stub,
        collections_dir=_collections_stub,
    )


def _output_files(out_dir: Path) -> dict[str, Path]:
    return {
        p.relative_to(out_dir).as_posix(): p
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and p.suffix in {".parquet", ".json"}
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip_volatile(payload, keys: frozenset[str]):
    if isinstance(payload, dict):
        return {k: _strip_volatile(v, keys) for k, v in payload.items() if k not in keys}
    if isinstance(payload, list):
        return [_strip_volatile(v, keys) for v in payload]
    return payload


def _first_json_difference(a, b, path: str = "$") -> str | None:
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return (
                f"{path}: keys only in first {sorted(a.keys() - b.keys())}, "
                f"only in second {sorted(b.keys() - a.keys())}"
            )
        for key in a:
            found = _first_json_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = _first_json_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    if a != b:
        return f"{path}: {a!r} vs {b!r}"
    return None


def _describe_json_difference(a: Path, b: Path, volatile: frozenset[str]) -> str | None:
    first = _strip_volatile(json.loads(a.read_text()), volatile)
    second = _strip_volatile(json.loads(b.read_text()), volatile)
    return _first_json_difference(first, second)


def _short(row) -> str:
    text = repr(row)
    return text if len(text) <= 240 else text[:240] + "..."


def _describe_parquet_difference(a: Path, b: Path) -> str:
    con = duckdb.connect()
    try:
        only_first = con.execute(
            f"SELECT * FROM read_parquet('{a}') EXCEPT ALL "
            f"SELECT * FROM read_parquet('{b}') ORDER BY ALL LIMIT 3"
        ).fetchall()
        only_second = con.execute(
            f"SELECT * FROM read_parquet('{b}') EXCEPT ALL "
            f"SELECT * FROM read_parquet('{a}') ORDER BY ALL LIMIT 3"
        ).fetchall()
        if only_first or only_second:
            return (
                f"rows only in first build: {[_short(r) for r in only_first]}; "
                f"rows only in second build: {[_short(r) for r in only_second]}"
            )
        rows_a = con.execute(f"SELECT * FROM read_parquet('{a}')").fetchall()
        rows_b = con.execute(f"SELECT * FROM read_parquet('{b}')").fetchall()
    finally:
        con.close()
    for i, (x, y) in enumerate(zip(rows_a, rows_b)):
        if x != y:
            return (
                f"same rows, different order: first difference at row {i}: "
                f"{_short(x)} vs {_short(y)}"
            )
    return "same rows in the same order; the files differ only in encoding"


def test_rebuilding_the_corpus_is_byte_identical(fixture_corpus_build, second_build):
    first = _output_files(fixture_corpus_build.out_dir)
    second = _output_files(second_build.out_dir)
    assert {"fact_results.parquet", "manifest.json", "snapshot_meta.json"} <= first.keys(), (
        f"the first build did not emit a full warehouse: {sorted(first)}"
    )
    assert first.keys() == second.keys(), (
        f"the two builds emitted different files: only in first "
        f"{sorted(first.keys() - second.keys())}, only in second "
        f"{sorted(second.keys() - first.keys())}"
    )

    problems: list[str] = []
    for name in first:
        a, b = first[name], second[name]
        volatile = VOLATILE_JSON_KEYS.get(name, frozenset())
        if volatile:
            detail = _describe_json_difference(a, b, volatile)
            if detail:
                problems.append(f"{name}: {detail}")
            continue
        if _sha256(a) == _sha256(b):
            continue
        if a.suffix == ".json":
            detail = _describe_json_difference(a, b, frozenset())
            problems.append(
                f"{name}: {detail or 'identical content, different bytes'}"
            )
        else:
            problems.append(f"{name}: {_describe_parquet_difference(a, b)}")

    assert not problems, (
        f"{len(problems)} of {len(first)} warehouse files differ between two "
        f"builds of the same snapshot:\n  " + "\n  ".join(problems)
    )

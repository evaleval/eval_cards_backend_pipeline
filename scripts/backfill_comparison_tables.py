"""Write the comparison tables into a snapshot that only has comparison-index.json.

For warehouse snapshots produced before the producer emitted the tables.
Flattens the snapshot's own comparison-index.json with the producer's
function, writes the three parquets next to it, reads them back and checks
that the rebuilt index serialises to the same bytes as the file. Exits
non-zero on any mismatch. Once the round trip holds, the three tables are
added to snapshot_meta.json's `tables`; nothing else in that file changes.

    uv run python scripts/backfill_comparison_tables.py <snapshot_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from eval_card_backend.canonicalise import comparison_tables


def _snapshot_id(snapshot_dir: Path) -> str:
    # manifest.json carries the snapshot_id as its generated_at.
    for name, key in (("snapshot_meta.json", "snapshot_id"), ("manifest.json", "generated_at")):
        path = snapshot_dir / name
        sid = json.loads(path.read_text()).get(key) if path.exists() else None
        if sid:
            return sid
    raise SystemExit(f"no snapshot_id in {snapshot_dir}/snapshot_meta.json or manifest.json")


def _register_tables(snapshot_dir: Path) -> None:
    path = snapshot_dir / "snapshot_meta.json"
    if not path.exists():
        print(f"no {path.name}; tables not registered")
        return
    meta = json.loads(path.read_text())
    meta["tables"] += [t for t in comparison_tables.TABLE_FILES if t not in meta["tables"]]
    # Written the way the pipeline writes it.
    path.write_text(json.dumps(meta, indent=2))
    print(f"{path.name}: tables registered")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot_dir", type=Path)
    args = parser.parse_args(argv)
    snapshot_dir: Path = args.snapshot_dir

    snapshot_id = _snapshot_id(snapshot_dir)
    text = (snapshot_dir / "comparison-index.json").read_text()
    index = json.loads(text)
    paths = comparison_tables.write_tables(index, snapshot_dir, snapshot_id)

    rebuilt = comparison_tables.unflatten(comparison_tables.read_tables(snapshot_dir))
    if rebuilt != index or json.dumps(rebuilt, indent=2, sort_keys=True) != text:
        print("round trip FAILED: tables do not rebuild comparison-index.json", file=sys.stderr)
        return 1

    n_metrics = sum(len(e["metrics"]) for e in index["evals"].values())
    n_cells = sum(
        len(m["scores"]) for e in index["evals"].values() for m in e["metrics"]
    )
    print(f"snapshot_id {snapshot_id}")
    print(f"index: {len(index['evals'])} evals, {n_metrics} metrics, {n_cells} cells")
    for path in paths:
        rows = pq.read_metadata(path).num_rows
        print(f"{path.name}: {rows} rows, {path.stat().st_size} bytes")
    print("round trip exact (dict equal and byte-identical JSON)")
    _register_tables(snapshot_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

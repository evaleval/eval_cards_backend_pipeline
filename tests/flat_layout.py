"""Shared writer for the upstream EEE flat layout in test fixtures.

Mirrors the on-disk shape `eee.ensure_snapshot` produces:
    <eee_root>/flat/latest_manifest.json
    <eee_root>/flat/manifests/sha256_fixture/entries.jsonl
    <eee_root>/flat/objects/<s1>/<s2>/<name>.json

Object payloads are taken as raw text (some drop-path tests write
intentionally invalid JSON); the entries row is always well-formed —
upstream indexes corrupt objects too, the read/validate gates do the
dropping. Shard dirs come from sha256 of the filename stem, so tests
exercise the sharded shape while filenames stay readable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ENTRIES_PATH = "flat/manifests/sha256_fixture/entries.jsonl"


def write_flat_datastore(
    eee_root: Path, files: list[tuple[str, str, str]]
) -> None:
    """Write `(config, filename, raw_text)` triples as a flat datastore.

    Repeat calls on the same root accumulate (entries are merged by
    `object_path`, last write wins), so tests can build the corpus
    incrementally. Filename stems must be unique across configs.
    """
    entries: dict[str, dict] = {}
    entries_file = eee_root / _ENTRIES_PATH
    if entries_file.exists():
        with entries_file.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    entries[row["object_path"]] = row

    for config, filename, raw_text in files:
        stem = filename[:-5] if filename.endswith(".json") else filename
        shard = hashlib.sha256(stem.encode()).hexdigest()
        object_path = f"flat/objects/{shard[0:2]}/{shard[2:4]}/{stem}.json"
        out = eee_root / object_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(raw_text)
        payload = out.read_bytes()
        entries[object_path] = {
            "benchmark": config,
            "eval_schema_version": "0.2.2",
            "legacy_path": f"data/{config}/dev/model/{stem}.json",
            "object_path": object_path,
            "object_uuid": stem,
            "record_type": "aggregate",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    entries_file.parent.mkdir(parents=True, exist_ok=True)
    entries_file.write_text(
        "".join(json.dumps(row) + "\n" for row in entries.values())
    )
    (eee_root / "flat" / "latest_manifest.json").write_text(json.dumps({
        "entries_path": _ENTRIES_PATH,
        "aggregate_file_count": len(entries),
        "benchmark_count": len({row["benchmark"] for row in entries.values()}),
        "created_at": "2026-06-01T00:00:00Z",
        "eval_schema_versions": ["0.2.2"],
        "manifest_core_sha256": "fixture",
        "total_file_count": len(entries),
    }, indent=2))

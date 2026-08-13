"""Shared writer for the upstream EEE `data/` tree in test fixtures.

Mirrors the on-disk shape `eee.ensure_snapshot` produces:
    <eee_root>/data/<config>/<developer>/<model>/<name>.json

Object payloads are taken as raw text (some drop-path tests write
intentionally invalid JSON) — the read/validate gates do the dropping.
No listing file is written: `eee._local_paths` falls back to walking
`data/` on disk, which is also the path hand-built caches exercise.
"""
from __future__ import annotations

from pathlib import Path


def write_eee_datastore(
    eee_root: Path, files: list[tuple[str, str, str]]
) -> None:
    """Write `(config, filename, raw_text)` triples as a data/ tree.

    Repeat calls on the same root accumulate (last write per path wins),
    so tests can build the corpus incrementally.
    """
    for config, filename, raw_text in files:
        out = eee_root / "data" / config / "dev" / "model" / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(raw_text)

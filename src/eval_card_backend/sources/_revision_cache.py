"""Revision-aware cache marker for upstream HF snapshots.

When a source is downloaded at a pinned `revision` (commit SHA / branch /
tag), we record it in a marker file. On the next run, a cache is only
reused if its marker matches the requested revision — otherwise a stale
local cache would silently defeat the pin. When no revision is requested
(latest HEAD), any existing cache is accepted as before.
"""
from __future__ import annotations

from pathlib import Path

_MARKER = ".pinned_revision"


def cache_revision_ok(target: Path, revision: str | None) -> bool:
    """True when the cache at `target` may be reused for `revision`.

    - revision is None (latest): always reuse an existing cache.
    - revision pinned: reuse only if the marker records the same revision.
    """
    if revision is None:
        return True
    marker = target / _MARKER
    if not marker.exists():
        return False
    return marker.read_text(encoding="utf-8").strip() == revision


def write_cache_revision(target: Path, revision: str | None) -> None:
    """Record the revision a cache was downloaded at. No-op when latest."""
    if revision is None:
        return
    (target / _MARKER).write_text(revision, encoding="utf-8")

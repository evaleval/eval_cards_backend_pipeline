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


# `huggingface_hub` writes one metadata file per downloaded file under the
# local dir; its first line is the commit hash the file came from. Unlike
# `_MARKER` it is written on *every* download, pinned or not, so it tracks
# what is actually on disk.
_HF_DOWNLOAD_META_DIR = Path(".cache") / "huggingface" / "download"
_MAX_METADATA_SCAN = 4096


def cached_revision(target: Path | None) -> str | None:
    """The revision the snapshot on disk was actually downloaded at, or None.

    Prefers `huggingface_hub`'s per-file download metadata over `_MARKER`:
    the marker only records *pinned* downloads (`write_cache_revision` is a
    no-op when unpinned), so it can outlive the content it names, whereas
    the metadata is rewritten by every download.

    None means "unknown" and must never be treated as "HEAD": a hand-built
    fixture tree carries no record at all, and a cache assembled from more
    than one commit has no single revision to report.
    """
    if target is None:
        return None
    meta_root = Path(target) / _HF_DOWNLOAD_META_DIR
    if meta_root.is_dir():
        seen: set[str] = set()
        for i, meta in enumerate(meta_root.rglob("*.metadata")):
            if i >= _MAX_METADATA_SCAN:
                # Too large to verify the whole cache agrees; the caller
                # gets "unknown" rather than a possibly-partial answer.
                return None
            try:
                with meta.open(encoding="utf-8") as fh:
                    commit = fh.readline().strip()
            except OSError:
                continue
            if commit:
                seen.add(commit)
            if len(seen) > 1:
                return None
        if len(seen) == 1:
            return seen.pop()

    marker = Path(target) / _MARKER
    if marker.exists():
        try:
            return marker.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None

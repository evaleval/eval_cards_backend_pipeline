"""Revision-aware cache marker for upstream HF snapshots.

When a source is downloaded at a pinned `revision` (commit SHA / branch /
tag), we record it in a marker file. On the next run, a cache is only
reused if its marker matches the requested revision — otherwise a stale
local cache would silently defeat the pin. When no revision is requested
(latest HEAD), any existing cache is accepted as before.

A directory that is the corpus itself rather than a download (a committed
test fixture, a `seed --local` registry) is hand-built and is never wiped or
fetched into. A committed, empty `.hand_built` sentinel says so outright and
wins over any marker or hub metadata a stray download left behind, and over
a refresh flag, which raises `HandBuiltRefreshRefused` instead of deleting
the directory. Without the sentinel, the absence of both records is read as
hand-built, and a refresh still replaces the directory.
"""
from __future__ import annotations

import logging
from pathlib import Path

_MARKER = ".pinned_revision"
HAND_BUILT_SENTINEL = ".hand_built"

# `huggingface_hub` writes one metadata file per downloaded file under the
# local dir; its first line is the commit hash the file came from. Unlike
# `_MARKER` it is written on *every* download, pinned or not, so it tracks
# what is actually on disk.
_HF_DOWNLOAD_META_DIR = Path(".cache") / "huggingface" / "download"
_MAX_METADATA_SCAN = 4096


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


class HandBuiltRefreshRefused(RuntimeError):
    """A refresh was asked of a directory carrying the `.hand_built` sentinel."""


def has_sentinel(target: Path) -> bool:
    """True when `target` carries the `.hand_built` sentinel."""
    return (Path(target) / HAND_BUILT_SENTINEL).is_file()


def refuse_hand_built_refresh(target: Path, label: str, force_refresh: bool) -> None:
    """Raise when `force_refresh` targets a directory with the sentinel."""
    if force_refresh and has_sentinel(target):
        raise HandBuiltRefreshRefused(
            f"{label}: {target} is a hand-built corpus (it carries "
            f"{HAND_BUILT_SENTINEL}) and is never deleted or downloaded into. "
            f"A refresh must target a cache directory; unset the refresh flag "
            f"or point the source's local dir elsewhere."
        )


def is_hand_built(target: Path) -> bool:
    """True when `target` is the corpus itself, not a download: it carries
    the sentinel, or nothing records a download into it (no marker and no
    `huggingface_hub` download metadata). Files there were put in place by
    hand (a test fixture, a `seed --local` registry), so a revision pin must
    not read them as a stale cache.
    """
    target = Path(target)
    return has_sentinel(target) or (
        not (target / _MARKER).exists()
        and not (target / _HF_DOWNLOAD_META_DIR).is_dir()
    )


def log_hand_built(
    log: logging.Logger, label: str, target: Path, revision: str | None
) -> None:
    """Log that `target` is kept as is. A pin the loader cannot apply is
    worth a WARNING naming the directory and the pin; without a pin the
    same path is routine and stays at INFO."""
    if revision:
        log.warning(
            "%s hand-built snapshot at %s — nothing to sync; "
            "the pinned revision %s is not applied to it",
            label, target, revision,
        )
    else:
        log.info("%s hand-built snapshot at %s — nothing to sync", label, target)


def hand_built_snapshot(
    target: Path,
    revision: str | None,
    label: str,
    has_data: bool,
    log: logging.Logger,
    force_refresh: bool = False,
) -> bool:
    """True when the caller must return `target` untouched: it carries the
    sentinel, or it holds the corpus (`has_data`) and `is_hand_built` says
    so. The tree IS the source; wiping or fetching into it would be wrong
    whatever the pin says. Call it before any deletion: `force_refresh`
    raises `HandBuiltRefreshRefused` for a sentinel directory and returns
    False for any other, which the caller then empties and re-downloads.
    Logs the decision via `log_hand_built`."""
    refuse_hand_built_refresh(target, label, force_refresh)
    if not has_sentinel(target) and (
        force_refresh or not (has_data and is_hand_built(target))
    ):
        return False
    log_hand_built(log, label, target, revision)
    return True


def write_cache_revision(target: Path, revision: str | None) -> None:
    """Record the revision a cache was downloaded at. No-op when latest."""
    if revision is None:
        return
    (target / _MARKER).write_text(revision, encoding="utf-8")


def clear_cache_revision(target: Path) -> None:
    """Drop the marker, e.g. after re-indexing an existing cache unpinned."""
    (target / _MARKER).unlink(missing_ok=True)


def hf_download_commit(target: Path, filename: str) -> str | None:
    """Commit hash `huggingface_hub` recorded for one downloaded file, or None."""
    meta = Path(target) / _HF_DOWNLOAD_META_DIR / f"{filename}.metadata"
    try:
        with meta.open(encoding="utf-8") as fh:
            return fh.readline().strip() or None
    except OSError:
        return None


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

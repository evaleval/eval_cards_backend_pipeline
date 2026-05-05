"""Snapshot and read access for `evaleval/entity-registry-data`.

Provides:
  - `ensure_snapshot(local_dir, hf_token, force_refresh)`: download the registry
    parquets to local cache.
  - `load_alias_store(root)`: build an `AliasStore` for the resolver from the
    local cache.
  - `open_dim_paths(root)`: map of canonical_* table -> path (for DuckDB
    `read_parquet`).

Layout on disk after snapshot (the registry's HF dataset shape):
    <local_dir>/aliases/part-0.parquet
    <local_dir>/canonical_orgs/part-0.parquet
    <local_dir>/canonical_models/part-0.parquet
    <local_dir>/canonical_benchmarks/part-0.parquet
    <local_dir>/canonical_families/part-0.parquet     (added 2026-05-05)
    <local_dir>/canonical_composites/part-0.parquet   (added 2026-05-05)
    <local_dir>/canonical_metrics/part-0.parquet
    <local_dir>/eval_harnesses/part-0.parquet
    <local_dir>/manifest.json                          (added 2026-05-05)

`manifest.json` carries `schema_version` (registry.<MAJOR>.<MINOR>);
this module asserts the major matches `EXPECTED_REGISTRY_SCHEMA_MAJOR`
at snapshot-load time so a registry breaking change that ships before
the producer is updated fails fast with a clear error rather than
producing garbage downstream.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)

from eval_card_backend.config import ENTITY_REGISTRY_DATASET_REPO

# Registry schema major. Bumped when the registry removes/renames a
# column the producer reads. Minor bumps (additive columns) don't
# require a producer change. Coordinated with
# `eval-card-registry/scripts/publish_registry_data.py:SCHEMA_VERSION`.
EXPECTED_REGISTRY_SCHEMA_MAJOR = 2


class RegistrySchemaMismatch(RuntimeError):
    """Raised when the registry snapshot's manifest.json declares a
    schema_version major that the producer wasn't built for. Crash
    early — running with a mismatched registry corrupts downstream
    canonicalisation in subtle ways."""


DIM_TABLES: tuple[str, ...] = (
    "canonical_orgs",
    "canonical_models",
    "canonical_benchmarks",
    "canonical_families",
    "canonical_composites",
    "canonical_metrics",
    "eval_harnesses",
)

ALIASES_TABLE = "aliases"
ALL_TABLES: tuple[str, ...] = DIM_TABLES + (ALIASES_TABLE,)


def _has_registry_data(target: Path) -> bool:
    return any((target / table).exists() for table in ALL_TABLES)


def ensure_snapshot(
    local_dir: str, hf_token: str | None, force_refresh: bool
) -> Path:
    target = Path(local_dir).resolve()
    if force_refresh and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    if _has_registry_data(target):
        return target

    try:
        snapshot_download(
            repo_id=ENTITY_REGISTRY_DATASET_REPO,
            repo_type="dataset",
            local_dir=str(target),
            token=hf_token,
        )
    except Exception as exc:
        log.warning(
            "registry.ensure_snapshot: HF download failed (%s: %s); "
            "falling back to local-only mode at %s",
            type(exc).__name__,
            exc,
            target,
        )
    assert_manifest_compatible(target)
    return target


def assert_manifest_compatible(root: Path) -> None:
    """Verify the snapshot's `manifest.json` declares a schema_version
    major matching `EXPECTED_REGISTRY_SCHEMA_MAJOR`. Idempotent.

    Behaviour:
      - manifest.json missing → log a warning and continue. The
        registry's publish script (`publish_registry_data.py`) is what
        writes manifest.json, and older snapshots predate it. We don't
        want to break old caches; the producer just doesn't get the
        schema-drift early-warning until the next publish.
      - manifest.json present but unparseable → raise.
      - schema_version present but malformed (not "registry.M.N") →
        raise.
      - major mismatch → raise `RegistrySchemaMismatch`.
      - everything else → log the version and continue.
    """
    path = root / "manifest.json"
    if not path.exists():
        log.warning(
            "registry.manifest: %s missing; can't validate schema_version. "
            "Older snapshot? Producer will continue but won't catch "
            "schema drift. Re-publish the registry to enable.",
            path,
        )
        return

    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RegistrySchemaMismatch(
            f"registry manifest at {path} is unreadable ({exc!r})"
        ) from exc

    sv = manifest.get("schema_version")
    if not isinstance(sv, str) or not sv.startswith("registry."):
        raise RegistrySchemaMismatch(
            f"registry manifest schema_version malformed: {sv!r} "
            f"(expected 'registry.<major>.<minor>')"
        )

    parts = sv.split(".")
    try:
        major = int(parts[1])
    except (IndexError, ValueError) as exc:
        raise RegistrySchemaMismatch(
            f"registry manifest schema_version unparseable: {sv!r}"
        ) from exc

    if major != EXPECTED_REGISTRY_SCHEMA_MAJOR:
        raise RegistrySchemaMismatch(
            f"registry schema major mismatch: snapshot has {sv!r} "
            f"(major={major}), producer expects "
            f"major={EXPECTED_REGISTRY_SCHEMA_MAJOR}. "
            f"Update the producer or pin to a compatible snapshot."
        )

    log.info("registry.manifest: schema_version=%s (major matches)", sv)


def _resolve_table_path(root: Path, table: str) -> Path | None:
    """Return either the parquet file (single-file layout
    `<table>.parquet`) or the containing directory (HF parts layout
    `<table>/part-*.parquet`). Callers that pass the result to DuckDB's
    `read_parquet` should run it through `read_parquet_arg` so dirs
    become `<dir>/*.parquet` globs.
    """
    direct = root / f"{table}.parquet"
    if direct.exists():
        return direct
    table_dir = root / table
    if table_dir.is_dir() and any(table_dir.glob("*.parquet")):
        return table_dir
    return None


def read_parquet_arg(path: Path) -> str:
    """Convert a `_resolve_table_path` result into a DuckDB-readable
    parquet argument — file path or directory glob.
    """
    return str(path / "*.parquet") if path.is_dir() else str(path)


def open_dim_paths(root: Path) -> dict[str, Path]:
    """Return {table: path} for every dim table that's present.

    Path is either a single parquet file or a directory of `part-*.parquet`
    files. DuckDB's `read_parquet` accepts either form (use a glob if it's a
    directory).
    """
    out: dict[str, Path] = {}
    for table in DIM_TABLES:
        path = _resolve_table_path(root, table)
        if path is not None:
            out[table] = path
    return out


def aliases_path(root: Path) -> Path | None:
    return _resolve_table_path(root, ALIASES_TABLE)


def load_alias_store(root: Path):
    """Build a read-only `AliasStore` from the local cache.

    AliasStore.from_parquet reads `<root>/aliases.parquet` directly. The HF
    layout is `<root>/aliases/part-0.parquet`, so we adapt by passing the
    parent of the part file (renamed once to the layout the resolver expects).
    """
    from eval_entity_resolver import AliasStore

    direct = root / "aliases.parquet"
    if direct.exists():
        return AliasStore.from_parquet(root, read_only=True)

    table_dir = root / "aliases"
    if table_dir.is_dir():
        parts = sorted(table_dir.glob("*.parquet"))
        if parts:
            # AliasStore.from_parquet expects `<dir>/aliases.parquet`. Materialise
            # a small symlink so we don't copy the data.
            link = root / "aliases.parquet"
            if not link.exists():
                try:
                    link.symlink_to(parts[0])
                except OSError:
                    # Symlinks may be unsupported (e.g. some Windows filesystems);
                    # fall back to a hardlink-or-copy.
                    import os
                    try:
                        os.link(parts[0], link)
                    except OSError:
                        shutil.copy2(parts[0], link)
            return AliasStore.from_parquet(root, read_only=True)

    log.warning(
        "registry.load_alias_store: no aliases.parquet at %s — "
        "returning empty alias store", root,
    )
    return AliasStore.from_parquet(root, read_only=True)

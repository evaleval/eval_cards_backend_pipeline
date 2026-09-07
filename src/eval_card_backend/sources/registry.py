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
from eval_card_backend.config import ENTITY_REGISTRY_DATASET_REPO
from eval_card_backend.sources._revision_cache import (
    cache_revision_ok as _cache_revision_ok,
    cached_revision,  # re-exported: the revision this cache actually holds
    write_cache_revision as _write_cache_revision,
)

log = logging.getLogger(__name__)

__all__ = [
    "ALIASES_TABLE",
    "ALL_TABLES",
    "DIM_TABLES",
    "EXPECTED_REGISTRY_SCHEMA_MAJOR",
    "RegistrySchemaMismatch",
    "aliases_path",
    "assert_manifest_compatible",
    "cached_revision",
    "ensure_snapshot",
    "load_alias_store",
    "load_canonical_store",
    "load_resolver",
    "open_dim_paths",
    "read_parquet_arg",
]

# Registry schema major. Bumped when the registry removes/renames a
# column the producer reads. Minor bumps (additive columns) don't
# require a producer change. Coordinated with
# `eval-card-registry/scripts/publish_registry_data.py:SCHEMA_VERSION`.
EXPECTED_REGISTRY_SCHEMA_MAJOR = 3


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
    # New dim table from the model-resolution-rework. Older snapshots that
    # predate it won't ship the parquet; `open_dim_paths` simply omits the
    # key and `_load_dim` falls back to an empty table with the schema.
    # NB: the registry publishes this as `canonical_inference_platforms`
    # (consistent with the other canonical_* dims) — the name must match the
    # published parquet or the dim silently loads empty in production.
    "canonical_inference_platforms",
    # Metric naming folds for the merged benchmark view (registry.3.2+).
    # Stage A warns loudly when this loads empty on a registry snapshot
    # that should carry it — see stage_a_load_registry.
    "benchmark_metric_folds",
)

ALIASES_TABLE = "aliases"
ALL_TABLES: tuple[str, ...] = DIM_TABLES + (ALIASES_TABLE,)

_CANONICAL_STORE_KWARGS: dict[str, str] = {
    "canonical_models": "models_df",
    "canonical_benchmarks": "benchmarks_df",
    "canonical_families": "families_df",
    "canonical_composites": "composites_df",
    "canonical_metrics": "metrics_df",
    "eval_harnesses": "harnesses_df",
    "canonical_orgs": "orgs_df",
}


def _has_registry_data(target: Path) -> bool:
    # Two layouts count as data: the published HF layout (`<table>/` part
    # dirs) and the registry's `seed --local` fixtures layout (flat
    # `<table>.parquet`). Recognizing only the former made ensure_snapshot
    # treat freshly seeded fixtures as an empty cache and clobber them with
    # the published snapshot — breaking the documented local validation loop
    # (ENTITY_REGISTRY_LOCAL_DIR=../eval-card-registry/fixtures).
    return any(
        (target / table).exists() or (target / f"{table}.parquet").exists()
        for table in ALL_TABLES
    )


def ensure_snapshot(
    local_dir: str,
    hf_token: str | None,
    force_refresh: bool,
    revision: str | None = None,
) -> Path:
    target = Path(local_dir).resolve()
    if force_refresh and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    if _has_registry_data(target) and _cache_revision_ok(target, revision):
        return target
    if _has_registry_data(target):
        # Revision-mismatched cache — clear and re-download at the pin.
        shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=ENTITY_REGISTRY_DATASET_REPO,
            repo_type="dataset",
            revision=revision,
            local_dir=str(target),
            token=hf_token,
        )
        _write_cache_revision(target, revision)
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


def load_canonical_store(root: Path):
    """Build a ``CanonicalStore`` from either supported snapshot layout.

    ``CanonicalStore.from_parquet`` consumes the registry's flat fixture
    layout, whereas production snapshots use ``<table>/part-*.parquet``.
    Resolve both shapes here so the pipeline always gets the enrichment data
    needed for leaf ids, lineage, release dates, and organization folding.
    """
    import pandas as pd
    from eval_entity_resolver import CanonicalStore

    kwargs = {}
    for table, kwarg in _CANONICAL_STORE_KWARGS.items():
        path = _resolve_table_path(root, table)
        if path is None:
            if table == "canonical_models":
                raise FileNotFoundError(
                    f"required registry table {table!r} is missing under {root}"
                )
            continue
        try:
            kwargs[kwarg] = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            if table == "canonical_models":
                raise RuntimeError(
                    f"required registry table {table!r} is unreadable at {path}: {exc}"
                ) from exc
            log.warning(
                "registry.load_canonical_store: failed to read %s (%s: %s); "
                "using an empty %s table",
                path,
                type(exc).__name__,
                exc,
                table,
            )
    if kwargs["models_df"].empty:
        raise RuntimeError(
            f"required registry table 'canonical_models' is empty under {root}"
        )
    return CanonicalStore(**kwargs)


def load_resolver(root: Path):
    """Build the metadata-enriching resolver used by canonicalisation.

    The producer intentionally does not inject the registry service's HF-id
    checker here.  That checker is allowed to override a disagreeing exact
    alias, which is useful for live registry attestation but unsafe for a
    warehouse build: an HF-true id absent from ``canonical_models`` would
    replace a curated id and then lose all metadata in Stage G.  A future
    producer integration must be miss-only and carry an output-delta gate.
    """
    from eval_entity_resolver import Resolver

    return Resolver(
        load_alias_store(root),
        canonical_store=load_canonical_store(root),
    )

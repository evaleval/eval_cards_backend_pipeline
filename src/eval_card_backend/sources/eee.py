"""Snapshot and read access for `evaleval/EEE_datastore`.

Reads the upstream `data/` tree directly (the flat/ view is no longer
maintained upstream — frozen 2026-06-13):
    <local_dir>/data/<benchmark>/<developer>/<model>/<uuid>.json
    <local_dir>/.eee_file_listing.json   (written by ensure_snapshot)

Aggregate records are exactly the `*.json` files under `data/`; their
`*_samples.jsonl` instance companions are never downloaded (~GBs). The
config (benchmark) of a record is the first path segment under `data/`.

`ensure_snapshot` enumerates the repo once (at the pinned revision, or
HEAD resolved to a single SHA so all files come from one consistent
commit), persists the listing to `.eee_file_listing.json`, then top-ups
whatever aggregate objects are missing on disk. Re-runs reuse the stored
listing without network — new upstream data flows in only via
`EEE_REFRESH_SNAPSHOT=1`, a revision pin change, or force_refresh, same
contract as before.

`load_arrow_table` is the typed loader for Stage A: walks records,
validates each via the vendored Pydantic models (the upstream contract
from `every_eval_ever`), pads + casts to the derived `pa.Schema`, and
returns one Arrow table. Records that fail at any of the three gates
(read, validate, cast) are counted in a module-level drop counter and
the first occurrence per (config, reason) is logged. The aggregate
surfaces at end of run via `log_drop_summary`.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
from huggingface_hub import HfApi, hf_hub_download

from eval_card_backend.config import EEE_DATASET_REPO, IGNORED_CONFIGS
from eval_card_backend.sources._revision_cache import (
    cache_revision_ok as _cache_revision_ok,
    write_cache_revision as _write_cache_revision,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drop counter — first-occurrence-per-key logging keeps a 30k-record run from
# flooding stderr while still surfacing every distinct failure mode.
# ---------------------------------------------------------------------------

_drop_counter: Counter[tuple[str, str]] = Counter()
_drop_first_seen: set[tuple[str, str]] = set()


def reset_drop_counter() -> None:
    _drop_counter.clear()
    _drop_first_seen.clear()


def log_drop_summary() -> None:
    if not _drop_counter:
        return
    log.warning("--- Stage A EEE record drops ---")
    by_config: dict[str, Counter[str]] = {}
    for (cfg, reason), count in _drop_counter.items():
        by_config.setdefault(cfg, Counter())[reason] += count
    for cfg in sorted(by_config):
        breakdown = ", ".join(
            f"{reason}={n}" for reason, n in by_config[cfg].most_common()
        )
        total = sum(by_config[cfg].values())
        log.warning("  config=%s: %d dropped (%s)", cfg, total, breakdown)


def _record_drop(config: str, reason: str, path: str, detail: str | None = None) -> None:
    key = (config, reason)
    _drop_counter[key] += 1
    if key not in _drop_first_seen:
        _drop_first_seen.add(key)
        suffix = f": {detail}" if detail else ""
        log.warning(
            "Stage A: %s on %s (first occurrence; subsequent counted)%s",
            reason, path, suffix,
        )


# ---------------------------------------------------------------------------
# File enumeration — the `data/` tree replaces the flat manifest as the
# record index. Aggregates are `data/**/*.json`; `*_samples.jsonl`
# companions fall out on the extension alone.
# ---------------------------------------------------------------------------

_DATA_PREFIX = "data/"
_LISTING_PATH = ".eee_file_listing.json"


def _is_aggregate_path(path: str) -> bool:
    return path.startswith(_DATA_PREFIX) and path.endswith(".json")


def _config_of(path: str) -> str:
    return path.split("/")[1]


# Parsed listing memo — keyed on (resolved path, mtime_ns, size) so the
# per-config callers below don't reread the listing ~100 times per run.
_listing_cache: dict[tuple[str, int, int], list[str]] = {}


def _read_listing_file(listing_path: Path) -> list[str]:
    st = listing_path.stat()
    key = (str(listing_path.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _listing_cache:
        payload = json.loads(listing_path.read_text(encoding="utf-8"))
        _listing_cache[key] = payload["paths"]
    return _listing_cache[key]


def _local_paths(local_dir: Path) -> list[str]:
    """Aggregate paths known to a local snapshot dir.

    The listing file written by `ensure_snapshot` is authoritative when
    present; otherwise (e.g. hand-built test fixtures) fall back to
    walking `data/` on disk.
    """
    listing = Path(local_dir) / _LISTING_PATH
    if listing.exists():
        return _read_listing_file(listing)
    data_root = Path(local_dir) / "data"
    if not data_root.is_dir():
        return []
    return [
        p.relative_to(local_dir).as_posix()
        for p in data_root.rglob("*.json")
    ]


# Remote listing memo — one `list_repo_files` call per (repo, revision) per
# process. Mirrors the old behaviour where the downloaded manifest froze the
# remote view for the run.
_remote_listing_cache: dict[tuple[str, str], list[str]] = {}


def _remote_paths(hf_token: str | None, revision: str | None = None) -> list[str]:
    key = (EEE_DATASET_REPO, revision or "HEAD")
    if key not in _remote_listing_cache:
        files = HfApi().list_repo_files(
            EEE_DATASET_REPO,
            repo_type="dataset",
            revision=revision,
            token=hf_token,
        )
        _remote_listing_cache[key] = [f for f in files if _is_aggregate_path(f)]
    return _remote_listing_cache[key]


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

    # A cache is "present" when ensure_snapshot's own listing file exists,
    # OR when a data/ tree is already on disk without one (hand-built
    # caches, e.g. test fixtures) — the latter must never trigger a
    # network fetch into the caller's directory.
    has_data = (target / _LISTING_PATH).exists() or any(
        (target / "data").rglob("*.json")
    )
    # When a revision is pinned, the cache is only valid if it was
    # downloaded at that same revision (tracked via a marker file).
    # Otherwise a stale cache would silently defeat the pin.
    if has_data and not _cache_revision_ok(target, revision):
        # Stale or revision-mismatched cache — clear and re-download.
        shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        has_data = False

    if not has_data:
        # Resolve HEAD to one SHA so the listing and every downloaded file
        # come from the same upstream commit even if a merge lands mid-run.
        resolved = revision
        if resolved is None:
            try:
                resolved = HfApi().dataset_info(
                    EEE_DATASET_REPO, token=hf_token
                ).sha
            except Exception as exc:
                raise RuntimeError(
                    f"EEE revision lookup failed for {EEE_DATASET_REPO}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        log.info(
            "listing EEE data/ tree of %s (revision=%s) …",
            EEE_DATASET_REPO, resolved,
        )
        try:
            files = HfApi().list_repo_files(
                EEE_DATASET_REPO,
                repo_type="dataset",
                revision=resolved,
                token=hf_token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"EEE file listing failed for {EEE_DATASET_REPO}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        paths = sorted(f for f in files if _is_aggregate_path(f))
        n_samples = sum(
            1 for f in files
            if f.startswith(_DATA_PREFIX) and f.endswith("_samples.jsonl")
        )
        (target / _LISTING_PATH).write_text(
            json.dumps({"revision": resolved, "paths": paths}),
            encoding="utf-8",
        )
        _write_cache_revision(target, revision)
        log.info(
            "EEE listing: %d aggregate objects, %d instance files skipped",
            len(paths), n_samples,
        )

    if (target / _LISTING_PATH).exists():
        listing = json.loads(
            (target / _LISTING_PATH).read_text(encoding="utf-8")
        )
    else:
        # Hand-built cache: the data/ tree on disk IS the corpus — nothing
        # to top up, and downloads (if any were missing) would be wrong to
        # attempt against upstream anyway.
        listing = {"revision": revision, "paths": _local_paths(target)}
    wanted = [
        p for p in listing["paths"] if _config_of(p) not in IGNORED_CONFIGS
    ]
    n_ignored = len(listing["paths"]) - len(wanted)

    def _fetch(filename: str) -> None:
        try:
            hf_hub_download(
                repo_id=EEE_DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                revision=listing["revision"],
                local_dir=str(target),
                token=hf_token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"EEE download failed for {filename}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # Idempotent top-up: a presence/revision-valid cache may still be partial
    # (interrupted download, or IGNORED_CONFIGS narrowed since the last run) —
    # fetch whatever required objects are missing on disk.
    missing = [p for p in wanted if not (target / p).exists()]
    log.info(
        "EEE listing: %d aggregate objects kept, %d ignored-config rows skipped",
        len(wanted), n_ignored,
    )
    if missing:
        log.info(
            "downloading %d / %d missing EEE objects to %s …",
            len(missing), len(wanted), target,
        )
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(_fetch, missing))
        log.info("EEE snapshot ready at %s", target)
    else:
        log.info("EEE snapshot already present at %s — skipping download", target)
    return target


def discover_configs(local_dir: Path | None, hf_token: str | None) -> list[str]:
    if local_dir is not None:
        paths = _local_paths(Path(local_dir))
    else:
        paths = _remote_paths(hf_token)
    return sorted({_config_of(p) for p in paths})


def list_json_files(
    config: str, local_dir: Path | None, hf_token: str | None
) -> list[str]:
    """Return repo-relative JSON paths (e.g. `data/<config>/<dev>/<model>/<uuid>.json`)."""
    if local_dir is not None:
        paths = _local_paths(Path(local_dir))
    else:
        paths = _remote_paths(hf_token)
    return sorted(p for p in paths if _config_of(p) == config)


def read_record(
    dataset_path: str, local_dir: Path | None, hf_token: str | None
) -> dict[str, Any]:
    if local_dir is not None:
        return json.loads((Path(local_dir) / dataset_path).read_text(encoding="utf-8"))

    cached = hf_hub_download(
        repo_id=EEE_DATASET_REPO,
        filename=dataset_path,
        repo_type="dataset",
        token=hf_token,
    )
    return json.loads(Path(cached).read_text(encoding="utf-8"))


def iter_records(
    config: str, local_dir: Path | None, hf_token: str | None
) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in list_json_files(config, local_dir, hf_token):
        yield path, read_record(path, local_dir, hf_token)


# ---------------------------------------------------------------------------
# Typed loader — replaces the old temp-JSONL + read_json_auto pattern.
# ---------------------------------------------------------------------------


def load_arrow_table(
    eee_root: Path | None,
    configs: Iterable[str],
    hf_token: str | None,
) -> pa.Table:
    """Read EEE records, validate via Pydantic, cast to a typed Arrow table.

    The schema is derived from the vendored JSON Schema; see
    `schemas/eee_arrow.py` for the translation rules. Two extra columns are
    appended for downstream stages: `source_config` (config name) and
    `_record_path` (relative path of the source JSON).

    Records that fail (read error, non-dict, pydantic validation, pa cast)
    are dropped; the per-(config, reason) counter is updated and the first
    occurrence per key is logged. Caller should `reset_drop_counter()`
    before invocation and `log_drop_summary()` after.
    """
    # Local imports keep `sources.eee` module-import cheap when callers don't
    # need the typed path (e.g. discover_configs only).
    from pydantic import ValidationError

    from eval_card_backend.schemas.eee_arrow import (
        derive_pyarrow_schema,
        pad_record_for_cast,
    )
    from eval_card_backend.schemas.eee_types import EvaluationLog

    base_schema = derive_pyarrow_schema()
    # Schema for what the table actually holds = upstream contract +
    # pipeline-injected provenance columns.
    table_schema = pa.schema(
        list(base_schema)
        + [
            pa.field("source_config", pa.string(), nullable=False),
            pa.field("_record_path", pa.string(), nullable=False),
        ]
    )

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        cfg_paths = list_json_files(cfg, eee_root, hf_token)
        log.info("Stage A: loading config %s (%d records) …", cfg, len(cfg_paths))
        cfg_kept_before = len(rows)
        for path in cfg_paths:
            try:
                rec = read_record(path, eee_root, hf_token)
            except Exception as exc:
                _record_drop(cfg, f"read_error:{type(exc).__name__}", path, str(exc))
                continue
            if not isinstance(rec, dict):
                _record_drop(cfg, "not_a_dict", path, f"type={type(rec).__name__}")
                continue
            try:
                EvaluationLog.model_validate(rec)
            except ValidationError as exc:
                # Surface the first error path, not the full multi-error blob —
                # keeps the log line bounded.
                first = exc.errors()[0] if exc.errors() else {}
                loc = ".".join(str(p) for p in first.get("loc", []))
                msg = first.get("msg", "")
                _record_drop(cfg, "validation_error", path, f"{loc}: {msg}")
                continue
            except Exception as exc:
                _record_drop(
                    cfg, f"validation_error:{type(exc).__name__}", path, str(exc)
                )
                continue

            padded = pad_record_for_cast(rec, base_schema)
            padded["source_config"] = cfg
            padded["_record_path"] = path
            rows.append(padded)
        log.info(
            "Stage A: %s done — kept %d / %d",
            cfg, len(rows) - cfg_kept_before, len(cfg_paths),
        )

    if not rows:
        # Empty table with the right schema so downstream con.register +
        # SELECT works without special-casing the zero-row case.
        return pa.Table.from_pylist([], schema=table_schema)

    try:
        return pa.Table.from_pylist(rows, schema=table_schema)
    except Exception as exc:
        # Should not happen — pad_record_for_cast already handled missing
        # keys, and pydantic already accepted the record. If it does, surface
        # a clear error rather than raising the cryptic Arrow message.
        raise RuntimeError(
            f"pyarrow cast failed on {len(rows)} validated records: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

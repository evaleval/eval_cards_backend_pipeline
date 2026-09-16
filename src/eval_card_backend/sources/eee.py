"""Snapshot and read access for `evaleval/EEE_datastore`.

Reads the upstream `data/` tree directly:
    <local_dir>/data/<benchmark>/<developer>/<model>/<uuid>.json
    <local_dir>/.eee_file_listing.json   (written by ensure_snapshot)

Aggregate records are exactly the `*.json` files under `data/`; their
`*_samples.jsonl` instance companions are never downloaded (~GBs). The
config (benchmark) of a record is the first path segment under `data/`.

Index: the flat view, rebuilt daily upstream from `data/`
(`flat/latest_manifest.json` -> `entries_path` = `flat/manifests/
sha256_*/entries.jsonl`, one row per object with `legacy_path`,
`object_path`, `sha256`, `size_bytes`, `benchmark`, `record_type`). At the
pinned sha the descriptor and the row file are fetched once, the row file
is checked against `entries_sha256` / `entries_size_bytes`, and the
`record_type == "aggregate"` rows give the aggregate list keyed by
`legacy_path` plus a sha256 per object. Objects are downloaded from
`legacy_path` (same bytes as `object_path`), so the on-disk layout stays
the `data/` tree above. The datastore's root `manifest.json` is a frozen
artifact (entries stop in June 2026) and is never read. When the flat
descriptor is absent at the sha the loader falls back to enumerating the
repo tree (`list_repo_files`) with no sha256 map.

The flat rebuild lags data merges by up to a day and its descriptor names
no source commit, so `ensure_snapshot` compares `created_at` with the
pinned commit's date and warns when the flat index predates the commit:
records merged in between are not in the snapshot. Pin a flat-rebuild
commit to avoid the gap.

`ensure_snapshot` builds the index once (at the pinned revision, or HEAD
resolved to a single sha so all files come from one consistent commit),
persists it to `.eee_file_listing.json` (`{"revision", "paths",
"sha256": {path: sha}}`), then tops up: an object on disk whose sha256
matches the index is kept; a missing or mismatching one is (re)downloaded
and verified. A revision pin change therefore re-fetches only changed
objects; nothing on disk is ever deleted. Re-runs reuse the stored listing
without network. New upstream data flows in only via `EEE_REFRESH_SNAPSHOT=1`,
a revision pin change, or force_refresh.

`load_arrow_table` is the typed loader for Stage A: walks records,
validates each via the vendored Pydantic models (the upstream contract
from `every_eval_ever`), pads + casts to the derived `pa.Schema`, and
returns one Arrow table. Records that fail at any of the three gates
(read, validate, cast) are counted in a module-level drop counter and
the first occurrence per (config, reason) is logged. The aggregate
surfaces at end of run via `log_drop_summary`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from eval_card_backend.config import EEE_DATASET_REPO, IGNORED_CONFIGS
from eval_card_backend.sources._revision_cache import (
    cache_revision_ok as _cache_revision_ok,
    clear_cache_revision as _clear_cache_revision,
    hf_download_commit as _hf_download_commit,
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
# File enumeration — aggregates are `data/**/*.json`; `*_samples.jsonl`
# companions fall out on the extension alone.
# ---------------------------------------------------------------------------

_DATA_PREFIX = "data/"
_LISTING_PATH = ".eee_file_listing.json"
_FLAT_MANIFEST = "flat/latest_manifest.json"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _is_aggregate_path(path: str) -> bool:
    return path.startswith(_DATA_PREFIX) and path.endswith(".json")


def _config_of(path: str) -> str:
    return path.split("/")[1]


# Parsed listing memo — keyed on (resolved path, mtime_ns, size) so the
# per-config callers below don't reread the listing ~100 times per run.
_listing_cache: dict[tuple[str, int, int], dict[str, Any]] = {}


def _read_listing_file(listing_path: Path) -> dict[str, Any]:
    """Parsed listing with `paths` and a (possibly empty) `sha256` map.

    Listings written before the sha map existed carry only `revision` and
    `paths`; they load with an empty map, which means "unverifiable".
    """
    st = listing_path.stat()
    key = (str(listing_path.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _listing_cache:
        payload = json.loads(listing_path.read_text(encoding="utf-8"))
        payload.setdefault("sha256", {})
        _listing_cache[key] = payload
    return _listing_cache[key]


def cached_revision(local_dir: Path | None) -> str | None:
    """The EEE revision the snapshot on disk was downloaded at, or None.

    `ensure_snapshot` resolves HEAD to a concrete sha before listing and
    records it in the listing file, so this is truthful for unpinned runs
    too. None for hand-built fixture trees, which have no listing file.
    """
    if local_dir is None:
        return None
    listing = Path(local_dir) / _LISTING_PATH
    if not listing.exists():
        return None
    try:
        return json.loads(listing.read_text(encoding="utf-8")).get("revision")
    except (ValueError, OSError):
        return None


def _local_paths(local_dir: Path) -> list[str]:
    """Aggregate paths known to a local snapshot dir.

    The listing file written by `ensure_snapshot` is authoritative when
    present; otherwise (e.g. hand-built test fixtures) fall back to
    walking `data/` on disk.
    """
    listing = Path(local_dir) / _LISTING_PATH
    if listing.exists():
        return _read_listing_file(listing)["paths"]
    data_root = Path(local_dir) / "data"
    if not data_root.is_dir():
        return []
    return [
        p.relative_to(local_dir).as_posix()
        for p in data_root.rglob("*.json")
    ]


# ---------------------------------------------------------------------------
# Remote index — one flat descriptor + row-file fetch per (repo, revision)
# per process, so the remote view is frozen for the run.
# ---------------------------------------------------------------------------

_index_cache: dict[tuple[str, str], tuple[list[str], dict[str, str]]] = {}
_info_cache: dict[tuple[str, str], Any] = {}


def _dataset_info(hf_token: str | None, revision: str | None) -> Any:
    key = (EEE_DATASET_REPO, revision or "HEAD")
    if key not in _info_cache:
        try:
            info = HfApi().dataset_info(
                EEE_DATASET_REPO, revision=revision, token=hf_token
            )
        except Exception as exc:
            raise RuntimeError(
                f"EEE revision lookup failed for {EEE_DATASET_REPO}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        _info_cache[key] = info
        # HEAD resolves to a sha; the same answer serves lookups at that sha.
        _info_cache.setdefault((EEE_DATASET_REPO, info.sha), info)
    return _info_cache[key]


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_index_file(
    filename: str, hf_token: str | None, revision: str | None
) -> Path | None:
    """Fetch an index file into the HF hub cache; None if absent at `revision`."""
    try:
        cached = hf_hub_download(
            repo_id=EEE_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=hf_token,
        )
    except EntryNotFoundError:
        return None
    except Exception as exc:
        raise RuntimeError(
            f"EEE download failed for {filename}: {type(exc).__name__}: {exc}"
        ) from exc
    return Path(cached)


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str):
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _flat_index(
    descriptor: dict[str, Any], hf_token: str | None, revision: str | None
) -> tuple[list[str], dict[str, str]]:
    """Aggregate `legacy_path`s + sha256 from the flat row file the descriptor names."""
    entries_path = descriptor.get("entries_path")
    if not isinstance(entries_path, str):
        raise RuntimeError(f"EEE {_FLAT_MANIFEST} has no entries_path (keys={sorted(descriptor)})")
    local = _download_index_file(entries_path, hf_token, revision)
    if local is None:
        raise RuntimeError(
            f"EEE flat row file {entries_path} named by {_FLAT_MANIFEST} is absent "
            f"at revision={revision or 'HEAD'}"
        )
    expected_size = descriptor.get("entries_size_bytes")
    actual_size = local.stat().st_size
    if isinstance(expected_size, int) and actual_size != expected_size:
        raise RuntimeError(
            f"EEE flat row file {entries_path} size mismatch: expected "
            f"{expected_size} bytes, got {actual_size}"
        )
    expected_sha = descriptor.get("entries_sha256")
    if isinstance(expected_sha, str):
        actual_sha = _sha256_of(local)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"EEE flat row file {entries_path} sha256 mismatch: expected "
                f"{expected_sha}, got {actual_sha}"
            )

    paths: list[str] = []
    sha_map: dict[str, str] = {}
    n_rows = n_other = 0
    with local.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n_rows += 1
            row = json.loads(line)
            legacy = row.get("legacy_path")
            if row.get("record_type") != "aggregate" or not (
                isinstance(legacy, str) and _is_aggregate_path(legacy)
            ):
                n_other += 1
                continue
            paths.append(legacy)
            if isinstance(row.get("sha256"), str):
                sha_map[legacy] = row["sha256"]
    paths.sort()
    log.info(
        "EEE flat index %s: %d rows, %d aggregate objects (%d with sha256), "
        "%d non-aggregate rows skipped",
        entries_path, n_rows, len(paths), len(sha_map), n_other,
    )
    return paths, sha_map


def _log_flat_lag(
    descriptor: dict[str, Any], hf_token: str | None, revision: str | None, n_paths: int
) -> None:
    created = _parse_iso(descriptor.get("created_at"))
    declared = descriptor.get("aggregate_file_count")
    committed: datetime | None = None
    try:
        committed = getattr(_dataset_info(hf_token, revision), "last_modified", None)
    except RuntimeError as exc:
        log.info("EEE commit date lookup skipped: %s", exc)
    if committed is not None and committed.tzinfo is None:
        committed = committed.replace(tzinfo=timezone.utc)
    log.info(
        "EEE flat view: created_at=%s aggregate_file_count=%s (derived %d) "
        "pinned commit=%s dated %s",
        descriptor.get("created_at"), declared, n_paths, revision or "HEAD",
        committed.isoformat() if committed else "unknown",
    )
    if created is None or committed is None:
        return
    if created < committed:
        hours = (committed - created).total_seconds() / 3600
        log.warning(
            "flat index lags the pinned commit by %.1f hours; records merged "
            "in between are not in this snapshot; pin a flat-rebuild commit "
            "to avoid this", hours,
        )


def _list_repo_tree_paths(hf_token: str | None, revision: str | None) -> list[str]:
    try:
        files = HfApi().list_repo_files(
            EEE_DATASET_REPO,
            repo_type="dataset",
            revision=revision,
            token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"EEE file listing failed for {EEE_DATASET_REPO}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    n_samples = sum(
        1 for f in files
        if f.startswith(_DATA_PREFIX) and f.endswith("_samples.jsonl")
    )
    paths = sorted(f for f in files if _is_aggregate_path(f))
    log.info(
        "EEE repo tree: %d aggregate objects, %d instance files skipped",
        len(paths), n_samples,
    )
    return paths


def _remote_index(
    hf_token: str | None, revision: str | None
) -> tuple[list[str], dict[str, str]]:
    """Aggregate paths + sha256 map at `revision` (None = HEAD of main).

    Flat descriptor + verified row file when present; otherwise the repo
    tree listing with no sha256 map.
    """
    key = (EEE_DATASET_REPO, revision or "HEAD")
    if key in _index_cache:
        return _index_cache[key]

    descriptor_path = _download_index_file(_FLAT_MANIFEST, hf_token, revision)
    if descriptor_path is None:
        log.warning(
            "EEE flat descriptor %s missing at revision=%s; falling back to "
            "repo tree listing (no sha256 verification possible)",
            _FLAT_MANIFEST, revision or "HEAD",
        )
        paths, sha_map = _list_repo_tree_paths(hf_token, revision), {}
    else:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        paths, sha_map = _flat_index(descriptor, hf_token, revision)
        _log_flat_lag(descriptor, hf_token, revision, len(paths))

    _index_cache[key] = (paths, sha_map)
    return _index_cache[key]


def _remote_paths(hf_token: str | None, revision: str | None = None) -> list[str]:
    return _remote_index(hf_token, revision)[0]


# ---------------------------------------------------------------------------
# Snapshot sync
# ---------------------------------------------------------------------------


def _write_listing(
    target: Path, hf_token: str | None, revision: str | None
) -> None:
    """(Re)build the index at `revision` (HEAD resolved to a sha) and persist it."""
    # Resolve HEAD to one SHA so the listing and every downloaded file
    # come from the same upstream commit even if a merge lands mid-run.
    resolved = revision if revision is not None else _dataset_info(hf_token, None).sha
    log.info(
        "indexing EEE data/ tree of %s (revision=%s) …", EEE_DATASET_REPO, resolved,
    )
    paths, sha_map = _remote_index(hf_token, resolved)
    (target / _LISTING_PATH).write_text(
        json.dumps({"revision": resolved, "paths": paths, "sha256": sha_map}),
        encoding="utf-8",
    )
    if revision is None:
        # A marker left by an earlier pinned run would otherwise vouch for a
        # cache whose listing now names a different revision.
        _clear_cache_revision(target)
    else:
        _write_cache_revision(target, revision)
    log.info(
        "EEE listing: %d aggregate objects (%d verifiable) at revision=%s",
        len(paths), len(sha_map), resolved,
    )


def ensure_snapshot(
    local_dir: str,
    hf_token: str | None,
    force_refresh: bool,
    revision: str | None = None,
) -> Path:
    target = Path(local_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    listing_path = target / _LISTING_PATH

    # A data/ tree on disk without our listing file is a hand-built cache
    # (e.g. test fixtures): the tree IS the corpus, and a network fetch into
    # the caller's directory would be wrong. force_refresh still re-indexes.
    if not listing_path.exists() and not force_refresh and any(
        (target / "data").rglob("*.json")
    ):
        log.info("EEE hand-built snapshot at %s — nothing to sync", target)
        return target

    # Re-index when forced, when nothing is indexed yet, or when the pin no
    # longer matches the marker. A stale cache is not deleted: the listing is
    # rewritten at the new revision and the top-up below replaces only the
    # objects whose content actually changed.
    if (
        force_refresh
        or not listing_path.exists()
        or not _cache_revision_ok(target, revision)
    ):
        _write_listing(target, hf_token, revision)

    listing = _read_listing_file(listing_path)
    rev = listing["revision"]
    sha_map: dict[str, str] = listing["sha256"]
    wanted = [
        p for p in listing["paths"] if _config_of(p) not in IGNORED_CONFIGS
    ]
    n_ignored = len(listing["paths"]) - len(wanted)
    log.info(
        "EEE listing: %d aggregate objects kept, %d ignored-config rows skipped",
        len(wanted), n_ignored,
    )

    def _fetch(filename: str) -> None:
        try:
            hf_hub_download(
                repo_id=EEE_DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                revision=rev,
                local_dir=str(target),
                token=hf_token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"EEE download failed for {filename}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _verify(filename: str) -> None:
        expected = sha_map.get(filename)
        if expected is None:
            return
        actual = _sha256_of(target / filename)
        if actual != expected:
            raise RuntimeError(
                f"EEE sha256 mismatch for {filename} after download at "
                f"revision={rev}: expected {expected}, got {actual}"
            )

    def _sync(filename: str) -> str:
        """kept | downloaded | replaced for one object."""
        local = target / filename
        if local.exists():
            expected = sha_map.get(filename)
            if expected is not None:
                if _sha256_of(local) == expected:
                    return "kept"
            else:
                # Unverifiable object: trust it unless the hub metadata says it
                # was fetched at a different commit than the listing names.
                fetched_at = _hf_download_commit(target, filename)
                if (
                    fetched_at is None
                    or not _SHA_RE.match(rev or "")
                    or fetched_at == rev
                ):
                    return "kept"
            _fetch(filename)
            _verify(filename)
            return "replaced"
        _fetch(filename)
        _verify(filename)
        return "downloaded"

    # Idempotent top-up: verify what is on disk, fetch what is missing or
    # changed. Also completes a partial cache (interrupted download, or
    # IGNORED_CONFIGS narrowed since the last run).
    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = Counter(pool.map(_sync, wanted))
    fetched = outcomes["downloaded"] + outcomes["replaced"]
    if fetched:
        log.info(
            "EEE snapshot ready at %s: %d kept, %d downloaded, %d replaced",
            target, outcomes["kept"], outcomes["downloaded"], outcomes["replaced"],
        )
    else:
        log.info(
            "EEE snapshot already present at %s (%d objects verified/kept) — "
            "skipping download", target, outcomes["kept"],
        )
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

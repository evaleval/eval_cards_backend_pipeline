"""Snapshot and read access for `evaleval/EEE_datastore`.

Layout on disk after snapshot (upstream flat view):
    <local_dir>/flat/latest_manifest.json
    <local_dir>/flat/manifests/sha256_<hash>/entries.jsonl
    <local_dir>/flat/objects/<uuid[0:2]>/<uuid[2:4]>/<uuid>.json

The manifest's entries.jsonl indexes every record; the download is
manifest-driven, so only `record_type == "aggregate"` objects outside
`IGNORED_CONFIGS` are fetched (`<uuid>_samples.jsonl` instance companions
are never downloaded — ~4 GB).

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
from huggingface_hub import hf_hub_download

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


_MANIFEST_PATH = "flat/latest_manifest.json"

# Parsed entries.jsonl memo — keyed on (resolved path, mtime_ns, size) so the
# per-config callers below don't reparse a ~16 MB JSONL ~85 times per run.
_entries_cache: dict[tuple[str, int, int], list[dict[str, Any]]] = {}


def _parse_entries(entries_path: Path) -> list[dict[str, Any]]:
    st = entries_path.stat()
    key = (str(entries_path.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _entries_cache:
        with entries_path.open(encoding="utf-8") as fh:
            _entries_cache[key] = [json.loads(line) for line in fh if line.strip()]
    return _entries_cache[key]


def _load_entries(local_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads(
        (Path(local_dir) / _MANIFEST_PATH).read_text(encoding="utf-8")
    )
    return _parse_entries(Path(local_dir) / manifest["entries_path"])


def _load_entries_remote(hf_token: str | None) -> list[dict[str, Any]]:
    """Manifest + entries via the default HF cache (no local snapshot)."""
    manifest_file = hf_hub_download(
        repo_id=EEE_DATASET_REPO,
        filename=_MANIFEST_PATH,
        repo_type="dataset",
        token=hf_token,
    )
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    entries_file = hf_hub_download(
        repo_id=EEE_DATASET_REPO,
        filename=manifest["entries_path"],
        repo_type="dataset",
        token=hf_token,
    )
    return _parse_entries(Path(entries_file))


def _wanted_entry(entry: dict[str, Any]) -> bool:
    return (
        entry.get("record_type") == "aggregate"
        and entry.get("benchmark") not in IGNORED_CONFIGS
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

    has_data = (target / _MANIFEST_PATH).exists()
    # When a revision is pinned, the cache is only valid if it was
    # downloaded at that same revision (tracked via a marker file).
    # Otherwise a stale cache would silently defeat the pin.
    if has_data and not _cache_revision_ok(target, revision):
        # Stale or revision-mismatched cache — clear and re-download.
        shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        has_data = False

    def _fetch(filename: str) -> None:
        try:
            hf_hub_download(
                repo_id=EEE_DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                local_dir=str(target),
                token=hf_token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"EEE download failed for {filename}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    if not has_data:
        log.info(
            "downloading EEE flat manifest to %s (revision=%s) …",
            target, revision or "latest",
        )
        _fetch(_MANIFEST_PATH)
        manifest = json.loads(
            (target / _MANIFEST_PATH).read_text(encoding="utf-8")
        )
        _fetch(manifest["entries_path"])
        _write_cache_revision(target, revision)

    entries = _load_entries(target)
    wanted = [e for e in entries if _wanted_entry(e)]
    n_ignored = sum(
        1 for e in entries
        if e.get("record_type") == "aggregate" and e.get("benchmark") in IGNORED_CONFIGS
    )
    n_instance = sum(1 for e in entries if e.get("instance_object_path"))
    # Idempotent top-up: a presence/revision-valid cache may still be partial
    # (interrupted download, or IGNORED_CONFIGS narrowed since the last run) —
    # fetch whatever required objects are missing on disk.
    missing = [
        e["object_path"] for e in wanted if not (target / e["object_path"]).exists()
    ]
    log.info(
        "EEE manifest: %d aggregate objects kept, %d ignored-config rows skipped, "
        "%d instance files skipped",
        len(wanted), n_ignored, n_instance,
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
        if not (Path(local_dir) / _MANIFEST_PATH).exists():
            return []
        entries = _load_entries(Path(local_dir))
    else:
        entries = _load_entries_remote(hf_token)
    return sorted({e["benchmark"] for e in entries})


def list_json_files(
    config: str, local_dir: Path | None, hf_token: str | None
) -> list[str]:
    """Return repo-relative JSON paths (e.g. `flat/objects/<s1>/<s2>/<uuid>.json`)."""
    if local_dir is not None:
        if not (Path(local_dir) / _MANIFEST_PATH).exists():
            return []
        entries = _load_entries(Path(local_dir))
    else:
        entries = _load_entries_remote(hf_token)
    return sorted(
        e["object_path"] for e in entries
        if e["benchmark"] == config and e.get("record_type") == "aggregate"
    )


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

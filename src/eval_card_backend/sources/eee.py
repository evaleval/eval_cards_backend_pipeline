"""Snapshot and read access for `evaleval/EEE_datastore`.

Layout on disk after snapshot:
    <local_dir>/data/<config>/<dev>/<model>/<uuid>.json

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
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download

from eval_card_backend.config import EEE_DATASET_REPO, IGNORED_CONFIGS

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


def ensure_snapshot(local_dir: str, hf_token: str | None, force_refresh: bool) -> Path:
    target = Path(local_dir).resolve()
    if force_refresh and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    data_dir = target / "data"
    if data_dir.exists() and any(data_dir.iterdir()):
        log.info("EEE snapshot already present at %s — skipping download", target)
        return target

    log.info("downloading EEE snapshot to %s …", target)
    snapshot_download(
        repo_id=EEE_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=["data/**"],
        ignore_patterns=[f"data/{cfg}/**" for cfg in IGNORED_CONFIGS],
        max_workers=16,
        token=hf_token,
    )
    log.info("EEE snapshot ready at %s", target)
    return target


def discover_configs(local_dir: Path | None, hf_token: str | None) -> list[str]:
    if local_dir is not None:
        data_root = Path(local_dir) / "data"
        if not data_root.exists():
            return []
        return sorted(p.name for p in data_root.iterdir() if p.is_dir())

    fs = HfFileSystem(token=hf_token)
    entries = fs.ls(f"datasets/{EEE_DATASET_REPO}/data", detail=True)
    return sorted({entry["name"].split("/")[-1] for entry in entries if entry.get("name")})


def list_json_files(
    config: str, local_dir: Path | None, hf_token: str | None
) -> list[str]:
    """Return repo-relative JSON paths (e.g. `data/<config>/.../record.json`)."""
    if local_dir is not None:
        root = Path(local_dir) / "data" / config
        if not root.exists():
            return []
        return sorted(
            str(p.relative_to(local_dir)).replace(os.sep, "/")
            for p in root.rglob("*.json")
            if p.is_file() and not p.name.endswith(".jsonl")
        )

    fs = HfFileSystem(token=hf_token)
    pattern = f"datasets/{EEE_DATASET_REPO}/data/{config}/**/*.json"
    prefix = f"datasets/{EEE_DATASET_REPO}/"
    return sorted(
        p[len(prefix):] for p in fs.glob(pattern) if not p.endswith(".jsonl")
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

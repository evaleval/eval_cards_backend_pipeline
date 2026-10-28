"""Shared machinery for the EEE datafix overlay.

A fix is two separable halves: a vendored patch table (data, reviewed,
keyed by portable record identity) and a pure applier over one EEE record
dict. Nothing here knows about DuckDB, the warehouse, file paths inside
the backend, or the stage cache, so a fix can be replayed anywhere the
raw records are (see `builders.export_patched_records`).

Doctrine, enforced by every applier:

- fill missing values; an existing value changes only when a source rule
  identifies it as obsolete adapter metadata and records its old value;
- never replace a container that is not a dict (a `generation_config` or
  `generation_args` holding something else is counted as
  `skipped_malformed` and skipped);
- write only source-attested metadata fields; narrowly identified obsolete
  adapter defaults may be corrected, with the old value in the event ledger;
  provenance lives in the evidence tables and `snapshot_meta.datafix`;
- ambiguity leaves the value missing and is counted.

Every fix reads the record as it arrived, so what one fills can never
read to another as a value that was already there. Two fixes proposing
the same field are reconciled by `apply_fixes`: agreement writes the
value once, disagreement writes nothing and is counted.

`fingerprint()` identifies the overlay as a whole: the ordered fix ids,
each vendored table's sha256, and `OVERLAY_REVISION`. It answers from the
static registry and the tables' uncompressed JSON bytes, without parsing a
table. The stage cache stores it so a resumed run cannot mix Stage A
output built with one overlay into a later stage running another.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Protocol

from eval_card_backend.datafix.table_io import read_json_bytes

DATA_DIR = Path(__file__).resolve().parent / "data"

# Bump on any change to applier behaviour that could alter what a run
# fills, independently of the vendored tables' contents.
OVERLAY_REVISION = 6


@dataclass(frozen=True)
class PatchEvent:
    """One field written into one record, and every fix that proposed it."""

    fix_id: str
    evaluation_id: str
    result_index: int
    evaluation_result_id: str | None
    pointer: str
    old: Any
    new: Any
    evidence: str
    # Other fixes that proposed the same value for the same field, in
    # registry order. The write is attributed to `fix_id`, the first of them.
    concurring_fix_ids: tuple[str, ...] = ()

    def as_row(self, record_path: str) -> dict[str, Any]:
        return {
            "record_path": record_path,
            "result_index": self.result_index,
            "pointer": self.pointer,
            "fix_id": self.fix_id,
            "concurring_fix_ids": list(self.concurring_fix_ids),
            "evaluation_id": self.evaluation_id,
            "evaluation_result_id": self.evaluation_result_id,
            "old": self.old,
            "new": self.new,
            "evidence": self.evidence,
        }

    @property
    def field(self) -> str:
        """The generation arg the pointer names."""
        return self.pointer.rsplit("/", 1)[-1]

    @property
    def sort_key(self) -> tuple[int, str, str]:
        return (self.result_index, self.pointer, self.fix_id)


class Fix(Protocol):
    """One correction: a vendored table plus a pure applier."""

    fix_id: str
    # The source config the fix applies to, or None when its evidence is
    # inside the record and the rule holds for every config.
    source_config: str | None

    @property
    def source_config_revision(self) -> str: ...

    @property
    def table_sha256(self) -> str: ...

    def apply(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[PatchEvent], Counter]:
        """Return a patched copy (None when nothing changed), the events,
        and this record's counters. `record` is never mutated."""
        ...

    def summarise(self, counters: Counter) -> dict[str, Any]:
        """Per-fix `snapshot_meta.datafix` block from accumulated counters."""
        ...


# ---------------------------------------------------------------------------
# Record helpers: shared by the appliers, all "fill only what is missing".
# ---------------------------------------------------------------------------

GENERATION_ARGS_POINTER = "/evaluation_results/{idx}/generation_config/generation_args/{field}"


def generation_arg(result: dict[str, Any], field: str) -> tuple[str, Any]:
    """Classify one generation arg: `present` / `missing` / `malformed`.

    `missing` covers an absent or null `generation_config`, an absent or
    null `generation_args`, and an absent or null field. `malformed` is a
    container that exists and is not a dict, which must never be replaced.
    """
    gen_config = result.get("generation_config")
    if gen_config is None:
        return "missing", None
    if not isinstance(gen_config, dict):
        return "malformed", None
    gen_args = gen_config.get("generation_args")
    if gen_args is None:
        return "missing", None
    if not isinstance(gen_args, dict):
        return "malformed", None
    value = gen_args.get(field)
    if value is None:
        return "missing", None
    return "present", value


def set_generation_arg(result: dict[str, Any], field: str, value: Any) -> None:
    """Write one generation arg, creating the containers it needs.

    The caller has already established via `generation_arg` that the field
    is missing and no container is malformed.
    """
    gen_config = result.get("generation_config")
    if not isinstance(gen_config, dict):
        gen_config = {}
        result["generation_config"] = gen_config
    gen_args = gen_config.get("generation_args")
    if not isinstance(gen_args, dict):
        gen_args = {}
        gen_config["generation_args"] = gen_args
    gen_args[field] = value


def pointer_value(record: dict[str, Any], pointer: str) -> tuple[str, Any]:
    """Classify a metadata path without replacing malformed containers."""
    node: Any = record
    for part in pointer.strip("/").split("/"):
        if node is None:
            return "missing", None
        if isinstance(node, list) and part.isdigit():
            index = int(part)
            if index >= len(node):
                return "malformed", None
            node = node[index]
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return "malformed", None
    return ("missing", None) if node is None else ("present", node)


def set_pointer(record: dict[str, Any], pointer: str, value: Any) -> None:
    """Set an already checked dictionary leaf (None explicitly clears it)."""
    parts = pointer.strip("/").split("/")
    node: Any = record
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        else:
            if node.get(part) is None:
                node[part] = {}
            node = node[part]
    if value is None:
        node.pop(parts[-1], None)
    else:
        node[parts[-1]] = value


# ---------------------------------------------------------------------------
# Applying every fix registered for a config
# ---------------------------------------------------------------------------


def apply_fixes(
    record: dict[str, Any],
    source_config: str,
    summary: "DatafixSummary | None" = None,
) -> tuple[dict[str, Any] | None, list[PatchEvent], Counter]:
    """Run every fix registered for `source_config` over one record.

    Returns `(patched_copy | None, events, counters)`. The input dict is
    left untouched, and every fix sees it exactly as it arrived: a value
    one fix fills is never what another fix reads as already present.

    Two fixes proposing the same field of the same result are therefore
    both proposals about the same gap. Agreeing on the value, they fill it
    once, attributed to whichever of them the registry lists first, with
    the rest named on the event. Disagreeing, neither is preferred: the
    field stays missing and each of them is counted `cross_fix_conflict`.

    When a `summary` is given, what happened here is recorded in it; the
    accumulator belongs to one run, so nothing is carried between runs.
    """
    from eval_card_backend.datafix.registry import fixes_for

    fixes = fixes_for(source_config)
    counters: Counter = Counter()
    # Keyed by pointer, which names one field of one result.
    proposed: dict[str, list[PatchEvent]] = {}
    for fix in fixes:
        _, fix_events, fix_counters = fix.apply(record)
        counters.update(fix_counters)
        for event in fix_events:
            proposed.setdefault(event.pointer, []).append(event)

    events: list[PatchEvent] = []
    patched: dict[str, Any] | None = None
    for group in proposed.values():
        first = group[0]
        if any(other.new != first.new for other in group[1:]):
            for event in group:
                counters[f"{event.fix_id}:cross_fix_conflict"] += 1
            continue
        if patched is None:
            patched = copy.deepcopy(record)
        set_pointer(patched, first.pointer, first.new)
        events.append(
            first if len(group) == 1
            else replace(
                first,
                concurring_fix_ids=tuple(other.fix_id for other in group[1:]),
            )
        )

    if summary is not None:
        summary.record(
            events, counters, source_config, fix_ids=[f.fix_id for f in fixes]
        )
    return patched, events, counters


# ---------------------------------------------------------------------------
# Run-local accounting: one accumulator per run, reported in
# snapshot_meta.datafix.
# ---------------------------------------------------------------------------


class DatafixSummary:
    """Accumulates what the overlay did across a whole Stage A pass.

    One of these belongs to one run. Nothing about it is process-level, so
    two runs in one process keep separate counts.

    A fix's own counters say what it proposed; `records_patched` and
    `results_patched` say what was written. The two differ for a proposal
    another fix contradicted, which `cross_fix_conflict` counts.
    """

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._by_config: dict[str, Counter] = {}
        self._ran: set[str] = set()
        self._restored: dict[str, Any] | None = None

    def record(
        self,
        events: Iterable[PatchEvent],
        counters: Counter,
        source_config: str,
        fix_ids: Iterable[str] = (),
    ) -> None:
        self._ran.update(fix_ids)
        by_fix: dict[str, set[tuple[str, int]]] = {}
        for event in events:
            by_fix.setdefault(event.fix_id, set()).add(
                (event.evaluation_id, event.result_index)
            )
        for fix_id, results in by_fix.items():
            bucket = self._counters.setdefault(fix_id, Counter())
            bucket["records_patched"] += 1
            bucket["results_patched"] += len(results)
            self._by_config.setdefault(fix_id, Counter())[source_config] += len(results)
        for key, value in counters.items():
            fix_id, _, name = key.partition(":")
            self._counters.setdefault(fix_id, Counter())[name] += value

    def restore(self, payload: dict[str, Any]) -> None:
        """Adopt a summary persisted by an earlier run's Stage A.

        A `--from-stage B` resume never re-reads the records, so the only
        truthful account of what is in `eee_raw` is the one the run that
        built it wrote down.
        """
        self._counters.clear()
        self._by_config.clear()
        self._ran.clear()
        self._restored = copy.deepcopy(payload)

    def as_dict(self, *, enabled: bool) -> dict[str, Any]:
        """The `snapshot_meta.datafix` block.

        A fix that never ran is reported from the static registry alone:
        its checksum, its evidence revision and zeroes. Asking what a run
        that filled nothing did therefore opens no patch table.
        """
        if self._restored is not None:
            return copy.deepcopy(self._restored)
        from eval_card_backend.datafix import registry

        fixes: dict[str, Any] = {}
        for entry in registry.entries():
            counters = self._counters.get(entry.fix_id)
            block = {
                "table_sha256": entry.evidence_sha256(),
                "source_revision": entry.source_revision,
                "records_patched": (counters or Counter())["records_patched"],
                "results_patched": (counters or Counter())["results_patched"],
                # Which configs a fix reached is not always what it is
                # registered for: one fix runs over every config.
                "by_source_config": dict(
                    sorted(self._by_config.get(entry.fix_id, Counter()).items())
                ),
                "cross_fix_conflict": (counters or Counter())["cross_fix_conflict"],
            }
            if counters is not None or entry.fix_id in self._ran:
                block.update(
                    registry.fix(entry.fix_id).summarise(counters or Counter())
                )
            fixes[entry.fix_id] = block
        return {
            "enabled": enabled,
            "overlay_revision": OVERLAY_REVISION,
            "fingerprint": fingerprint(),
            "fixes": fixes,
        }


# ---------------------------------------------------------------------------
# Canonical JSON: the vendored tables and the export manifest are reviewed
# artifacts and must be byte-identical across reruns on the same evidence.
# ---------------------------------------------------------------------------


def canonical_json(payload: Any, *, indent: int | None = 2) -> str:
    """Sorted keys, fixed separators, trailing newline."""
    separators = (",", ": ") if indent is not None else (",", ":")
    return json.dumps(
        payload,
        sort_keys=True,
        indent=indent,
        separators=separators,
        ensure_ascii=False,
    ) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a sibling temp file + rename, so a reader never sees half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vendored_sha256(name: str) -> str:
    """Hash a table's uncompressed JSON bytes without parsing it."""
    return hashlib.sha256(read_json_bytes(DATA_DIR / name)).hexdigest()


def load_table(name: str) -> dict[str, Any]:
    """Read a vendored patch table and its sha256 as published."""
    raw = read_json_bytes(DATA_DIR / name)
    payload = json.loads(raw)
    payload["_table_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def fingerprint() -> str:
    """Identity of the overlay: ordered fix ids, table shas, revision.

    Read off the static registry and the tables' uncompressed JSON bytes: identifying
    the overlay never builds an applier or parses a table.
    """
    from eval_card_backend.datafix.registry import entries

    digest = hashlib.sha256()
    digest.update(f"overlay_revision={OVERLAY_REVISION}\n".encode("utf-8"))
    for entry in entries():
        digest.update(f"{entry.fix_id}={entry.evidence_sha256()}\n".encode("utf-8"))
    return digest.hexdigest()

"""The registered fixes, in a fixed order.

Order is part of the overlay's identity (it feeds `core.fingerprint`) and
decides which of two fixes proposing the same field the write is
attributed to, so new fixes are appended rather than inserted.

The registry itself is static: each entry pairs a source config with the
factory that builds the fix for it, and carries the two things the overlay
has to be able to state about a fix without running it ; the checksum of
its evidence and the revision that evidence was read at. So `covers()`,
`fixes_for()`, `fingerprint()` and a summary for a fix that never ran all
answer without opening a table. A fix is instantiated (and its table read)
the first time it is actually needed, and cached for the rest of the
process. `reload()` drops the cache; the builders call it after rewriting
a table so a build and a check in the same process see the same bytes.

A fix registered under `ALL_CONFIGS` runs over every config, so it is only
registered that way when its evidence is inside the record and it has no
table: otherwise a run touching none of the named configs would start
paying for one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from eval_card_backend.datafix import (
    arc_agi,
    hfopenllm_v2,
    inspect_plan_config,
    vals_ai,
)
from eval_card_backend.datafix.core import Fix

from eval_card_backend.canonicalise import resolution_hotfixes as metadata

ALL_CONFIGS = None


@dataclass(frozen=True)
class Entry:
    """One registered fix, as the registry can describe it unbuilt."""

    fix_id: str
    # The source config the fix applies to, or ALL_CONFIGS.
    source_config: str | None
    build: Callable[[], Fix]
    # Checksum of the evidence the fix is derived from: its vendored
    # table's bytes, or, for a fix whose evidence is in the record, the
    # rule's own constants.
    evidence_sha256: Callable[[], str]
    # The revision that evidence was read at, repeated here so a run can
    # report it without parsing the table. `test_datafix_tables.py` checks
    # it against what each table records.
    source_revision: str


_REGISTRY: tuple[Entry, ...] = (
    Entry(
        vals_ai.FIX_ID, vals_ai.SOURCE_CONFIG, vals_ai.build,
        vals_ai.evidence_sha256, vals_ai.SOURCE_REVISION,
    ),
    Entry(
        hfopenllm_v2.FIX_ID, hfopenllm_v2.SOURCE_CONFIG, hfopenllm_v2.build,
        hfopenllm_v2.evidence_sha256, hfopenllm_v2.SOURCE_REVISION,
    ),
    Entry(
        inspect_plan_config.FIX_ID, inspect_plan_config.SOURCE_CONFIG,
        inspect_plan_config.build, inspect_plan_config.evidence_sha256,
        inspect_plan_config.SOURCE_REVISION,
    ),
    Entry(
        arc_agi.FIX_ID, arc_agi.SOURCE_CONFIG, arc_agi.build,
        arc_agi.evidence_sha256, arc_agi.SOURCE_REVISION,
    ),
    Entry(
        metadata._PUBLISHED_METADATA_FIX_ID, ALL_CONFIGS,
        metadata.build_published_metadata_datafix, metadata._published_metadata_sha,
        metadata._PUBLISHED_METADATA_REVISION,
    ),
)

_instances: dict[int, Fix] = {}


def _fix(index: int) -> Fix:
    fix = _instances.get(index)
    if fix is None:
        fix = _REGISTRY[index].build()
        _instances[index] = fix
    return fix


def entries() -> tuple[Entry, ...]:
    """Every registered fix as static description, building none of them."""
    return _REGISTRY


def fix(fix_id: str) -> Fix:
    for index, entry in enumerate(_REGISTRY):
        if entry.fix_id == fix_id:
            return _fix(index)
    raise KeyError(fix_id)


def all_fixes() -> tuple[Fix, ...]:
    return tuple(_fix(i) for i in range(len(_REGISTRY)))


def reload() -> None:
    _instances.clear()


def _applies(config: str | None, source_config: str) -> bool:
    return config is ALL_CONFIGS or config == source_config


def fixes_for(source_config: str) -> tuple[Fix, ...]:
    return tuple(
        _fix(i) for i, entry in enumerate(_REGISTRY)
        if _applies(entry.source_config, source_config)
    )


def covers(source_config: str) -> bool:
    """Whether any fix runs for this config. A config no fix covers skips
    the overlay entirely and pays nothing for it, tables included."""
    return any(_applies(e.source_config, source_config) for e in _REGISTRY)


def covers_every_config() -> bool:
    return any(e.source_config is ALL_CONFIGS for e in _REGISTRY)


def named_configs() -> frozenset[str]:
    """The configs the registry names. Not the full set of covered configs
    when a fix is registered for every config; `covers` answers that."""
    return frozenset(
        e.source_config for e in _REGISTRY if e.source_config is not ALL_CONFIGS
    )


def table_shas() -> dict[str, Any]:
    return {e.fix_id: e.evidence_sha256() for e in _REGISTRY}

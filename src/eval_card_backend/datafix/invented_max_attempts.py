"""Clear the `max_attempts: 1` the shared EEE schema used to write on its own.

Until the upstream commit named by `SOURCE_REVISION`, the schema gave
`generation_config.generation_args.max_attempts` a default of 1, so every
record serialised through it states a single attempt whether or not its
source said anything. Records written since carry the field only when the
converter sets it. The same result therefore reads as two observations,
one per side of that commit, and the reproducibility signal reads the 1 as
a stated protocol.

The evidence is inside the record: the value, the library that produced
it, and the day it was written. A 1 is cleared when all three hold:

- the value is exactly `INVENTED_VALUE`;
- `eval_library.name` is not in `DELIBERATE_LIBRARIES`, the converters
  that set the field from their source;
- the record was written before `WRITTEN_BEFORE`. A cron-written record
  states its run date in `source_metadata.additional_details.cron_run_date`
  and that wins, since its `retrieved_timestamp` can be the source's date
  rather than the write's. Any other record is dated by
  `retrieved_timestamp`.

A record whose write date cannot be read keeps its value and is counted
`undated`. Clearing removes the key, which the typed `eee_raw` struct
stores exactly as it stores a record that never had it.
"""
from __future__ import annotations

import copy
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from eval_card_backend.datafix.core import (
    GENERATION_ARGS_POINTER,
    PatchEvent,
    canonical_json,
    generation_arg,
    set_pointer,
    sha256_text,
)

FIX_ID = "invented_max_attempts"
SOURCE_CONFIG = None

# every_eval_ever "fix(schema): do not invent max attempts", merged
# 2026-09-09 19:31 UTC.
SOURCE_REVISION = "09496fc02c6aa64ed696a95109be08bbd2164020"

FIELD = "max_attempts"
INVENTED_VALUE = 1
DELIBERATE_LIBRARIES = ("inspect_ai",)
# The first whole UTC day after the upstream commit.
WRITTEN_BEFORE = datetime(2026, 9, 10, tzinfo=timezone.utc)

EVIDENCE = "invented_schema_default"


def evidence_sha256() -> str:
    """Checksum of the rule's constants, which stand in for a patch table."""
    return sha256_text(canonical_json({
        "field": FIELD,
        "invented_value": INVENTED_VALUE,
        "deliberate_libraries": list(DELIBERATE_LIBRARIES),
        "written_before": WRITTEN_BEFORE.isoformat(),
        "upstream_commit": SOURCE_REVISION,
    }))


def _cron_run_instant(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _retrieved_instant(value: Any) -> datetime | None:
    """Read a `retrieved_timestamp` the way Stage E's
    `retrieved_instant_sql` does: epoch seconds between 1e9 and 4e9, or an
    ISO-8601 string taken as UTC when it names no zone."""
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    if not 1e9 <= seconds <= 4e9:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _written_at(record: dict[str, Any]) -> datetime | None:
    source = record.get("source_metadata")
    details = source.get("additional_details") if isinstance(source, dict) else None
    cron_run_date = details.get("cron_run_date") if isinstance(details, dict) else None
    if cron_run_date is not None:
        return _cron_run_instant(cron_run_date)
    return _retrieved_instant(record.get("retrieved_timestamp"))


def _record_verdict(record: dict[str, Any]) -> str:
    """What the record says about every invented-looking 1 it carries."""
    library = record.get("eval_library")
    if isinstance(library, dict) and library.get("name") in DELIBERATE_LIBRARIES:
        return "kept_deliberate_library"
    written = _written_at(record)
    if written is None:
        return "undated"
    if written >= WRITTEN_BEFORE:
        return "kept_written_after"
    return "cleared"


class InventedMaxAttempts:
    fix_id = FIX_ID
    source_config = SOURCE_CONFIG
    source_config_revision = SOURCE_REVISION

    @property
    def table_sha256(self) -> str:
        return evidence_sha256()

    def apply(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[PatchEvent], Counter]:
        events: list[PatchEvent] = []
        counters: Counter = Counter()
        verdict: str | None = None
        results = record.get("evaluation_results")
        for idx, result in enumerate(results if isinstance(results, list) else []):
            if not isinstance(result, dict):
                continue
            status, value = generation_arg(result, FIELD)
            if status == "malformed":
                counters[f"{FIX_ID}:skipped_malformed"] += 1
                continue
            # `True == 1` in Python; a boolean is not the schema's default.
            if status != "present" or isinstance(value, bool) or value != INVENTED_VALUE:
                continue
            if verdict is None:
                verdict = _record_verdict(record)
            counters[f"{FIX_ID}:{verdict}"] += 1
            if verdict == "cleared":
                events.append(PatchEvent(
                    fix_id=FIX_ID,
                    evaluation_id=record.get("evaluation_id"),
                    result_index=idx,
                    evaluation_result_id=result.get("evaluation_result_id"),
                    pointer=GENERATION_ARGS_POINTER.format(idx=idx, field=FIELD),
                    old=value,
                    new=None,
                    evidence=EVIDENCE,
                ))
        if not events:
            return None, events, counters
        patched = copy.deepcopy(record)
        for event in events:
            set_pointer(patched, event.pointer, event.new)
        return patched, events, counters

    def summarise(self, counters: Counter) -> dict[str, Any]:
        return {
            name: counters[name]
            for name in (
                "cleared", "kept_deliberate_library", "kept_written_after",
                "undated", "skipped_malformed",
            )
        }


def build() -> InventedMaxAttempts:
    """Build the fix that clears the schema-invented `max_attempts: 1`.

    TODO(upstream): https://github.com/evaleval/every_eval_ever/pull/258
    stopped the schema defaulting `max_attempts` to 1, but the records
    written before it still state the 1 and nothing upstream rewrites them.
    Remove when: the pinned datastore holds no record written before
    2026-09-10 that states `max_attempts: 1` outside the libraries in
    `DELIBERATE_LIBRARIES`, so a run reports `cleared: 0`.
    """
    return InventedMaxAttempts()

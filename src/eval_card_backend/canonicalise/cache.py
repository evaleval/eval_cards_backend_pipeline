"""Stage-output cache: COPY TO Parquet + CREATE TABLE AS read_parquet.

Each stage's terminal output tables are written to
`<root>/<snapshot_dir_name>/<table>.parquet` after the stage runs. When a
later run uses `--from-stage X`, the orchestrator loads the cached parquets
back into the in-memory DuckDB connection so stages earlier than X don't
have to recompute.

Stage outputs are declared as a static mapping (`STAGE_OUTPUTS`). Stages
that produce intermediate tables (e.g. F's `fact_results_grouped_annotated`)
don't appear here — only the table the *next* stage reads from.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# Fingerprint of the table shapes this code writes into a stage cache. Bump
# whenever a stage's cached output gains, loses or re-types a column: a
# `--from-stage` run against an older cache would otherwise restore the old
# shape and fail deep inside a later stage's SQL with a binder error.
CACHE_SCHEMA_VERSION = 14

_SCHEMA_MARKER = "_cache_schema.json"

# Ordered stage letters. 'H' was removed when completeness moved per-row.
STAGE_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "I", "J")

# Per-stage output tables. The list defines what the cache writes after the
# stage runs and what gets restored when --from-stage skips it.
STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "A": (
        "eee_raw",
        "cards_raw",
        "canonical_orgs",
        "canonical_models",
        "canonical_benchmarks",
        "canonical_metrics",
        "benchmark_metric_folds",
        "canonical_composites",
        "canonical_families",
        "eval_harnesses",
        "aliases",
        "composite_config_map",
        "family_membership",
        "slice_promotions",
    ),
    "B": (
        "results_exploded",
        # Collection-adapter outputs (collections spec): created empty
        # when no vendored collection applies, so the cache shape is stable.
        "collection_member_ids",
        "collection_protocol_map",
        "collection_merge_map",
        "collection_study_slugs",
        "collection_trajectories_raw",
    ),
    "C": ("results_resolved",),
    "D": ("fact_results_staging", "collection_keys"),
    "E": ("fact_results_signaled",),
    "F": ("fact_results",),
    "G": ("benchmarks", "composites", "families", "models"),
    "I": (),  # Stage I writes the warehouse parquets; nothing in-memory to cache.
    "J": ("eval_results_view", "models_view", "evals_view", "merged_evals_view",
          # fact-grain headline map; the sidecars read it after Stage J
          "fact_headline"),
}


# ---------------------------------------------------------------------------
# Snapshot-id encode / decode triad. Format invariant: canonical ISO form is
# `YYYY-MM-DDTHH:MM:SSZ` (no microseconds, UTC, trailing Z). Filesystem-safe
# dir form replaces `:` with `-`. DuckDB only accepts the canonical ISO form
# as a TIMESTAMP literal, so any user-supplied dir-form id has to be normalised
# back before it reaches the SQL layer. Discovery walks the cache root and
# decodes dir-form names to ISO via the same translation. Keep all three
# helpers together: a format change in one demands matching changes in the
# others.
# ---------------------------------------------------------------------------


def _make_snapshot_id() -> str:
    """Mint a fresh ISO-form snapshot id, second precision, UTC."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _snapshot_dir_name(snapshot_id: str) -> str:
    """ISO → filesystem-safe directory name."""
    return snapshot_id.replace(":", "-")


def normalize_snapshot_id(snapshot_id: str) -> str:
    """Coerce a user-supplied snapshot id to canonical ISO form.

    Operators copy directory names (`2026-05-03T00-00-00Z`) from the
    cache/warehouse into `--snapshot-id`. Pass it through verbatim and
    DuckDB rejects the timestamp literal mid-pipeline. Restore `:` in the
    time portion so both dir-form and ISO-form work the same.
    """
    if "T" not in snapshot_id:
        return snapshot_id
    date_part, _, time_part = snapshot_id.partition("T")
    return f"{date_part}T{time_part.replace('-', ':', 2)}"


def validate_letter(letter: str, *, label: str) -> str:
    upper = letter.upper()
    if upper not in STAGE_ORDER:
        raise ValueError(
            f"{label}={letter!r} not a recognised stage; "
            f"valid: {', '.join(STAGE_ORDER)}"
        )
    return upper


class StageCache:
    """Read/write per-stage parquet caches for a given snapshot.

    `enabled=False` makes writes a no-op while leaving reads available.
    Used by the `--no-cache` flag for runs that don't want the side-effect.
    """

    def __init__(self, root: Path | str, snapshot_id: str, enabled: bool = True) -> None:
        self.root = Path(root)
        self.snapshot_id = snapshot_id
        self.enabled = enabled
        self._dir = self.root / _snapshot_dir_name(snapshot_id)

    @property
    def dir(self) -> Path:
        return self._dir

    def _path(self, table: str) -> Path:
        return self._dir / f"{table}.parquet"

    def has_table(self, table: str) -> bool:
        return self._path(table).exists()

    @property
    def _marker_path(self) -> Path:
        return self._dir / _SCHEMA_MARKER

    def _write_schema_version(self) -> None:
        self._marker_path.write_text(
            json.dumps({"cache_schema_version": CACHE_SCHEMA_VERSION})
        )

    def _clear_schema_version(self) -> None:
        self._marker_path.unlink(missing_ok=True)

    def _prune_after(self, stage_letter: str) -> list[str]:
        """Delete cached outputs of every stage AFTER `stage_letter`.

        A rebuild that stops at Stage D inside an existing snapshot dir would
        otherwise leave E..J parquets from the previous generation behind, and
        a later `--from-stage` run would restore that mixture.
        """
        dropped: list[str] = []
        for stage in STAGE_ORDER[STAGE_ORDER.index(stage_letter) + 1:]:
            for table in STAGE_OUTPUTS[stage]:
                path = self._path(table)
                if path.exists():
                    path.unlink()
                    dropped.append(table)
        return dropped

    def assert_schema_current(self) -> None:
        """Fail fast when the on-disk cache was written by a pipeline whose
        stage-output shapes differ from this one's.

        A cache dir with no marker predates the fingerprint entirely, so it is
        treated as stale rather than trusted.
        """
        if not self._dir.exists():
            return
        try:
            found = json.loads(self._marker_path.read_text())["cache_schema_version"]
        except (OSError, ValueError, KeyError):
            found = None
        if found == CACHE_SCHEMA_VERSION:
            return
        raise RuntimeError(
            f"stage cache at {self._dir} declares cache_schema_version="
            f"{found!r}, but this pipeline writes {CACHE_SCHEMA_VERSION}. "
            f"The cached stage outputs have a different column shape and "
            f"cannot be resumed from. Re-run from Stage A (drop --from-stage, "
            f"or pass --from-stage A) to rebuild the cache."
        )

    def write_table(self, con, table: str) -> None:
        """Write one table's parquet. Deliberately does NOT stamp the schema
        marker — only a completed `write_stage` may do that, so an interrupted
        write can never leave a blessed but incomplete cache dir."""
        if not self.enabled:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(table)
        # Single-statement COPY; ZSTD matches Stage I's warehouse output
        # so cache and warehouse files compress comparably.
        con.execute(
            f"COPY (SELECT * FROM {table}) TO '{path}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD)"
        )

    def load_table(self, con, table: str) -> None:
        path = self._path(table)
        if not path.exists():
            raise FileNotFoundError(
                f"cache miss: {table} not in {self._dir}; "
                f"re-run without --from-stage or pass --snapshot-id <id> "
                f"of a snapshot that has it."
            )
        # DROP first so a re-run against the same connection doesn't duplicate.
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{path}')"
        )

    def cached_tables(self) -> set[str]:
        if not self._dir.exists():
            return set()
        return {p.stem for p in self._dir.glob("*.parquet")}

    def write_stage(self, con, stage_letter: str) -> None:
        """Write this stage's outputs, then bless the dir.

        Order matters: later stages' stale parquets go first, the marker is
        stamped last. An interrupted run therefore leaves an unmarked (and so
        unusable) cache rather than one that looks current.
        """
        if not self.enabled:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        self._clear_schema_version()
        dropped = self._prune_after(stage_letter)
        if dropped:
            log.info(
                "stage cache: stage %s rewrite dropped %d stale later-stage "
                "table(s): %s", stage_letter, len(dropped), sorted(dropped),
            )
        for table in STAGE_OUTPUTS[stage_letter]:
            self.write_table(con, table)
        self._write_schema_version()

    def restore_through(self, con, last_stage: str) -> list[str]:
        """Restore every cached output table for stages A .. last_stage.

        Returns the list of restored tables for logging.

        Every table STAGE_OUTPUTS declares for A..last_stage must be on
        disk. A missing one means the cache was written by a pipeline with
        a different STAGE_OUTPUTS layout, or by a run that died mid-write;
        either way restoring the survivors would silently mix generations,
        so fail here with an actionable message instead of surfacing a
        `CatalogException: table not found` deep inside a later stage.
        """
        self.assert_schema_current()
        wanted: list[tuple[str, str]] = []
        for stage in STAGE_ORDER:
            wanted.extend((stage, table) for table in STAGE_OUTPUTS[stage])
            if stage == last_stage:
                break
        missing = [f"{stage}:{table}" for stage, table in wanted
                   if not self.has_table(table)]
        if missing:
            raise RuntimeError(
                f"stage cache at {self._dir} is incomplete for stages "
                f"A..{last_stage}: missing {', '.join(missing)}. The cache "
                f"was written by a different STAGE_OUTPUTS layout or by an "
                f"interrupted run. Re-run from Stage A (drop --from-stage, "
                f"or pass --from-stage A) to rebuild it."
            )
        restored: list[str] = []
        for _, table in wanted:
            self.load_table(con, table)
            restored.append(table)
        return restored


def discover_latest_snapshot(root: Path | str) -> str | None:
    """Return the most-recent snapshot id under `root`, or None if none exist.

    Snapshots are subdirectories whose names round-trip via
    `_snapshot_dir_name`. Lexical-max sort works because the directory
    names sort the same way as the timestamps they encode (ISO-8601 with
    hyphens).
    """
    root_path = Path(root)
    if not root_path.exists():
        return None
    candidates = sorted(
        (p.name for p in root_path.iterdir() if p.is_dir()),
        reverse=True,
    )
    if not candidates:
        return None
    return normalize_snapshot_id(candidates[0])

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

import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

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
        "eval_harnesses",
        "composite_config_map",
        "family_membership",
        "slice_promotions",
    ),
    "B": ("results_exploded",),
    "C": ("results_resolved",),
    "D": ("fact_results_staging",),
    "E": ("fact_results_signaled",),
    "F": ("fact_results",),
    "G": ("benchmarks", "composites", "families", "models"),
    "I": (),  # Stage I writes the warehouse parquets; nothing in-memory to cache.
    "J": ("eval_results_view", "models_view", "evals_view"),
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

    def write_table(self, con, table: str) -> None:
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
        for table in STAGE_OUTPUTS[stage_letter]:
            self.write_table(con, table)

    def restore_through(self, con, last_stage: str) -> list[str]:
        """Restore every cached output table for stages A .. last_stage.

        Returns the list of restored tables for logging.

        Tables that aren't on disk are silently skipped — this is
        intentional, since some uses (tests, partial-stage scenarios)
        deliberately cache only a subset. The downside is that schema
        drift from a STAGE_OUTPUTS change in a newer pipeline version
        produces a cryptic `CatalogException: table not found` later
        rather than a clear "delete the stale cache" message here. If
        you hit a CatalogException early in a `--from-stage` rerun, the
        likely fix is `rm -rf .cache/canonicalise/<snapshot>` so the
        cache is rebuilt against the current STAGE_OUTPUTS layout.
        """
        restored: list[str] = []
        for stage in STAGE_ORDER:
            for table in STAGE_OUTPUTS[stage]:
                if self.has_table(table):
                    self.load_table(con, table)
                    restored.append(table)
            if stage == last_stage:
                break
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

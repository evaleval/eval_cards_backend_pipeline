"""Collections: submission-channel tagging + vendored collection adapters.

Implements notes/collections-spec.md (Deliverable A + the canonicalise-time
half of Deliverable B):

- **collection_id derivation**: every fact row is keyed to the
  (`source_organization_name`, `source_name`) pair it arrived under, via a
  raw-field slug — no registry resolution, so the id is stable per datastore
  revision. Curated entries in `collections_curated.yaml` may declare
  `merge_raw_keys` that fold several raw-derived ids into one curated id.
- **Vendored collection adapters**: `vendor/collections/<name>/` holds
  the output of a manually-run extractor (see
  `scripts/collections/aisi_inference_scaling.py`): reassembled synthetic
  results, stitched trajectories, and a manifest enumerating the member
  records. At canonicalise time the adapter drops the members' exploded
  fragment rows and injects the synthetic results in their place —
  pre-Stage C, so they flow through resolution/folds/hotfixes like any
  upstream row. Inconsistent inputs hard-fail (no-silent-pollution).

The tables this module creates on the connection are Stage B outputs
(cached/restored with `results_exploded`):

- `collection_member_ids(evaluation_id, collection_id)` — the synthetic-row
  discriminator: post-drop, any surviving row whose evaluation_id is
  in this set IS synthetic. Consumed by the Stage C slice-key exemption and
  the leak guard.
- `collection_protocol_map(evaluation_id, result_idx, collection_id,
  protocol_condition, n_trajectories)` — Stage D joins `protocol_condition`
  onto staging by (evaluation_id, result_idx).
- `collection_merge_map(raw_key, collection_id)` — curated raw-key folds.
- `collection_study_slugs(collection_id, study_slug)` — feeds the post-
  Stage-C leak guard.
- `collection_trajectories_raw` — the vendored trajectories, unioned across
  collections; Stage I joins resolved ids and emits
  `collection_trajectories.parquet`.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURATED_PATH = REPO_ROOT / "collections_curated.yaml"
DEFAULT_VENDOR_DIR = REPO_ROOT / "vendor" / "collections"

# Reserved protocol_condition keys . `feedback` is the only one:
# the rollup exclusions filter on it by name, so a collection encoding
# assistance under any other key would silently bypass them.
RESERVED_PROTOCOL_KEYS = {"feedback": ("none", "answer_feedback", "unknown")}

_SLUG_MAX = 80


def slug(value: str | None) -> str:
    """Lowercase, non-alphanumerics collapsed to `-`, trimmed, truncated at
    80 chars. Keep in lockstep with `slug_sql` (parity is tested)."""
    if value is None:
        return ""
    collapsed = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return collapsed[:_SLUG_MAX]


def slug_sql(expr: str) -> str:
    """SQL twin of `slug()`. NULL input yields '' (COALESCE at the edge)."""
    return (
        f"substr(trim(both '-' from regexp_replace(lower(COALESCE({expr}, '')), "
        f"'[^a-z0-9]+', '-', 'g')), 1, {_SLUG_MAX})"
    )


def source_label_slug_sql(
    source_name_expr: str, harness_expr: str, config_expr: str
) -> str:
    """Slug of the guard-adjusted source label — the source-name half of
    the raw collection key: `slug(source_name)`, except harness bleed
    (source_name == eval_library name) keys on `slug(source_config)` and
    a missing label falls back to `unlabeled`. Curated composite
    `source:` scoping matches this exact value, keeping it in lockstep
    with collection keys.
    """
    name_slug = (
        f"CASE WHEN {source_name_expr} IS NOT NULL "
        f"AND {source_name_expr} = {harness_expr} "
        f"THEN NULLIF({slug_sql(config_expr)}, '') "
        f"ELSE NULLIF({slug_sql(source_name_expr)}, '') END"
    )
    return f"COALESCE({name_slug}, 'unlabeled')"


def collection_raw_key_sql(
    org_expr: str, source_name_expr: str, harness_expr: str, config_expr: str
) -> str:
    """Raw collection key `slug(org) || '/' || slug(source_name)`
    with two guards:

    - harness bleed (source_name == eval_library name): key on
      `slug(org) || '/' || slug(source_config)` instead;
    - missing parts fall back to `unknown` / `unlabeled`.
    """
    org_slug = f"NULLIF({slug_sql(org_expr)}, '')"
    name_slug = source_label_slug_sql(source_name_expr, harness_expr, config_expr)
    return f"(COALESCE({org_slug}, 'unknown') || '/' || {name_slug})"


# ---------------------------------------------------------------------------
# Curated registry (collections_curated.yaml)
# ---------------------------------------------------------------------------


def curated_path() -> Path:
    """In-repo curated overrides file. `COLLECTIONS_CURATED_PATH` overrides
    for tests/fixtures (mirrors the taxonomy-seed env override pattern)."""
    import os

    override = os.environ.get("COLLECTIONS_CURATED_PATH")
    return Path(override) if override else DEFAULT_CURATED_PATH


def vendor_collections_dir() -> Path:
    """Vendored collection extracts. `COLLECTIONS_VENDOR_DIR` overrides for
    tests/fixtures."""
    import os

    override = os.environ.get("COLLECTIONS_VENDOR_DIR")
    return Path(override) if override else DEFAULT_VENDOR_DIR


def load_curated(path: Path | None = None) -> dict[str, dict]:
    """Load curated collection entries. Missing file → empty registry."""
    p = path or curated_path()
    if not p.exists():
        return {}
    import yaml

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"collections curated file {p} must be a mapping")
    for cid, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"curated collection {cid!r} must be a mapping")
        entry.setdefault("curated", True)
    return data


def assert_curated_keys_observed(
    con, curated: dict[str, dict], *, strict: bool
) -> None:
    """Build-time assertion: every curated ENTRY must match ≥1
    observed raw key, else the build fails loudly — prevents a curated
    entry silently detaching when the upstream raw fields drift.

    Individual unobserved `merge_raw_keys` members only WARN: transition
    entries legitimately list keys from both sides of an upstream rename
    (e.g. the AISI Institute/Initiative spelling fix), and any single
    datastore revision can only observe one side. The warning names the
    stale keys so they get cleaned up once the transition completes.

    `strict=False` (config-subset debug runs, where the collection's
    configs may simply not be loaded) degrades the detach failure to a
    warning too.
    """
    observed = {
        r[0] for r in con.execute(
            "SELECT DISTINCT raw_key FROM collection_keys"
        ).fetchall()
    }
    detached: list[str] = []
    for cid, entry in curated.items():
        keys = entry.get("merge_raw_keys") or [cid]
        missing = [k for k in keys if k not in observed]
        if missing and len(missing) == len(keys):
            detached.append(f"{cid}: no observed raw key among {keys}")
        elif missing:
            log.warning(
                "collections: curated entry %s has unobserved merge_raw_keys "
                "%s (transition keys? remove once the upstream rename is "
                "fully consumed)", cid, missing,
            )
    if not detached:
        return
    msg = (
        "curated collection entries do not match any observed raw key "
        "(raw source fields drifted upstream, or the entry has a typo): "
        + "; ".join(detached)
    )
    if strict:
        raise RuntimeError(f"Stage D collections assertion failed: {msg}")
    log.warning("collections (non-strict config-subset run): %s", msg)


# ---------------------------------------------------------------------------
# Vendor adapters
# ---------------------------------------------------------------------------

_TRAJECTORIES_DDL = (
    "collection_id VARCHAR, benchmark_raw VARCHAR, model_raw VARCHAR, "
    "task_id VARCHAR, protocol_condition VARCHAR, trajectory_idx INTEGER, "
    "score DOUBLE, is_correct BOOLEAN, "
    "total_tokens BIGINT, output_tokens BIGINT, reasoning_tokens BIGINT, "
    "num_turns INTEGER, tool_calls INTEGER, n_pieces INTEGER, "
    "wall_time_s DOUBLE, working_time_s DOUBLE, stop_reason VARCHAR, "
    "partial_start BOOLEAN, unstitchable BOOLEAN, "
    "token_source_cumulative BOOLEAN, source_record_uuids VARCHAR[]"
)


def create_collection_tables(con) -> None:
    """Create the (initially empty) collection tables on the connection.
    Idempotent per-connection; called at the top of Stage B's collection
    step and defensively by later stages for pre-collections caches."""
    con.execute(
        "CREATE TABLE IF NOT EXISTS collection_member_ids ("
        "evaluation_id VARCHAR, collection_id VARCHAR)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS collection_protocol_map ("
        "evaluation_id VARCHAR, result_idx INTEGER, collection_id VARCHAR, "
        "protocol_condition VARCHAR, n_trajectories INTEGER)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS collection_merge_map ("
        "raw_key VARCHAR, collection_id VARCHAR)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS collection_study_slugs ("
        "collection_id VARCHAR, study_slug VARCHAR)"
    )
    con.execute(
        f"CREATE TABLE IF NOT EXISTS collection_trajectories_raw "
        f"({_TRAJECTORIES_DDL})"
    )


def _consumed_eee_revision(eee_root: Path | None, pinned: str | None) -> str | None:
    """The EEE revision this run actually consumes: the explicit pin when
    set, else the revision recorded in the snapshot listing file (None for
    hand-built fixture trees)."""
    if pinned:
        return pinned
    if eee_root is None:
        return None
    listing = Path(eee_root) / ".eee_file_listing.json"
    if not listing.exists():
        return None
    try:
        return json.loads(listing.read_text(encoding="utf-8")).get("revision")
    except (ValueError, OSError):
        return None


def apply_vendor_collections(
    con,
    *,
    eee_root: Path | None,
    eee_revision: str | None,
    vendor_dir: Path | None = None,
    curated: dict[str, dict] | None = None,
) -> None:
    """Stage B collection step: create the collection tables, load the
    curated merge map, and apply every vendored collection adapter found
    under `vendor/collections/<name>/manifest.json`.

    Per adapter (hard-fail on inconsistency):

    1. Register the manifest's member evaluation_ids + study slug (the leak
       guard checks the FULL set even when no member is loaded, so brand-new
       upstream records can't slip past on a config-subset run).
    2. For members present in `eee_raw`: assert the manifest's pinned EEE
       revision equals the consumed revision, DELETE their exploded rows
       (count must reconcile with the manifest's per-member result counts),
       and inject the vendored synthetic results whose base member record is
       present.
    """
    create_collection_tables(con)

    curated = curated if curated is not None else load_curated()
    merge_rows = [
        (raw_key, cid)
        for cid, entry in curated.items()
        for raw_key in (entry.get("merge_raw_keys") or [])
    ]
    if merge_rows:
        con.executemany(
            "INSERT INTO collection_merge_map VALUES (?, ?)", merge_rows
        )

    vdir = vendor_dir or vendor_collections_dir()
    if not vdir.is_dir():
        return
    for manifest_path in sorted(vdir.glob("*/manifest.json")):
        _apply_one_adapter(con, manifest_path, eee_root, eee_revision)


def _apply_one_adapter(
    con, manifest_path: Path, eee_root: Path | None, eee_revision: str | None
) -> None:
    adapter_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection_id = manifest["collection_id"]
    study_slug = manifest["study_slug"]
    members = manifest["members"]
    if not manifest.get("eee_revision"):
        # The extract's pin is asserted — a manifest without one can't be
        # tied to any datastore state and must not ship.
        raise RuntimeError(
            f"collection {collection_id}: manifest at {manifest_path} carries "
            f"no eee_revision — regenerate the vendored extract with the "
            f"extractor's --revision flag."
        )
    bad_members = [
        m for m in members
        if not isinstance(m.get("evaluation_id"), str) or not m["evaluation_id"]
    ]
    if bad_members:
        # A NULL/empty member id would make the leak guard's NOT IN
        # three-valued and pass vacuously — reject the manifest outright.
        raise RuntimeError(
            f"collection {collection_id}: {len(bad_members)} manifest "
            f"member(s) lack a non-empty evaluation_id — corrupt extract."
        )

    con.execute(
        "INSERT INTO collection_study_slugs VALUES (?, ?)",
        [collection_id, study_slug],
    )
    con.executemany(
        "INSERT INTO collection_member_ids VALUES (?, ?)",
        [(m["evaluation_id"], collection_id) for m in members],
    )

    member_ids = {m["evaluation_id"] for m in members}
    present = {
        r[0] for r in con.execute(
            "SELECT DISTINCT e.evaluation_id FROM eee_raw e "
            "JOIN collection_member_ids m USING (evaluation_id) "
            "WHERE m.collection_id = ?",
            [collection_id],
        ).fetchall()
    }
    if not present:
        log.info(
            "collection %s: no member records in this run's corpus — adapter inert",
            collection_id,
        )
        return

    consumed = _consumed_eee_revision(eee_root, eee_revision)
    pinned = manifest.get("eee_revision")
    if consumed is not None and pinned is not None and consumed != pinned:
        raise RuntimeError(
            f"collection {collection_id}: vendored extract was generated at EEE "
            f"revision {pinned} but this run consumes {consumed}. Re-run the "
            f"extractor ({manifest.get('extractor', 'scripts/collections/')}) "
            f"at the consumed revision and commit the regenerated "
            f"vendor/collections/** in the same change as the pin bump."
        )
    if consumed is None:
        log.warning(
            "collection %s: consumed EEE revision unknown (hand-built cache?); "
            "cannot verify the vendored extract matches. Manifest pin: %s",
            collection_id, pinned,
        )

    # --- Drop member fragment rows, keyed strictly by manifest ids.
    expected_present_drop = sum(
        int(m.get("n_results", 0)) for m in members if m["evaluation_id"] in present
    )
    n_before = con.execute("SELECT count(*) FROM results_exploded").fetchone()[0]
    con.execute(
        "DELETE FROM results_exploded WHERE evaluation_id IN "
        "(SELECT evaluation_id FROM collection_member_ids WHERE collection_id = ?)",
        [collection_id],
    )
    n_dropped = n_before - con.execute(
        "SELECT count(*) FROM results_exploded"
    ).fetchone()[0]
    if n_dropped != expected_present_drop:
        raise RuntimeError(
            f"collection {collection_id}: dropped {n_dropped} member fragment "
            f"row(s) but the manifest accounts for {expected_present_drop} "
            f"across the {len(present)} member record(s) present. The vendored "
            f"extract is out of sync with the datastore — re-run the extractor."
        )
    if present == member_ids:
        expected_total = int(manifest["expected_drop_count"])
        if n_dropped != expected_total:
            raise RuntimeError(
                f"collection {collection_id}: full membership present but "
                f"dropped {n_dropped} != manifest.expected_drop_count "
                f"{expected_total}. Re-run the extractor."
            )
    else:
        log.warning(
            "collection %s: partial membership (%d/%d member records present — "
            "config-subset run?); dropped %d fragment row(s)",
            collection_id, len(present), len(member_ids), n_dropped,
        )

    # --- Inject synthetic results (pre-Stage C; EEE-shaped rows).
    results_path = adapter_dir / "results.parquet"
    if not results_path.exists():
        raise RuntimeError(
            f"collection {collection_id}: manifest present but "
            f"{results_path} missing — incomplete vendored extract."
        )
    rp = results_path.as_posix().replace("'", "''")

    # Column-set assertion: INSERT … BY NAME silently NULL-fills columns
    # the source lacks, so a pipeline-side explode-schema change without an
    # extractor re-run would degrade synthetic rows silently. Require the
    # parquet's columns to be exactly the explode columns (minus the two
    # computed here) plus the protocol attachments.
    parquet_cols = {
        r[0] for r in con.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{rp}'))"
        ).fetchall()
    }
    exploded_cols = {
        r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'results_exploded'"
        ).fetchall()
    }
    expected_cols = (
        (exploded_cols - {"evaluation_result_id", "fact_id"})
        | {"protocol_condition", "n_trajectories"}
    )
    if parquet_cols != expected_cols:
        raise RuntimeError(
            f"collection {collection_id}: vendored results.parquet column set "
            f"drifted from the pipeline's explode schema "
            f"(missing: {sorted(expected_cols - parquet_cols)}, "
            f"unexpected: {sorted(parquet_cols - expected_cols)}) — "
            f"re-run the extractor against the current code."
        )

    present_list = sorted(present)
    placeholders = ", ".join("?" for _ in present_list)
    try:
        con.execute(
            f"""
            INSERT INTO results_exploded BY NAME
            SELECT
                * EXCLUDE (protocol_condition, n_trajectories),
                COALESCE(
                    evaluation_result_id_raw,
                    evaluation_id || '#' || result_idx::VARCHAR
                ) AS evaluation_result_id,
                fact_id_udf(evaluation_id, CAST(result_idx AS INTEGER)) AS fact_id
            FROM read_parquet('{rp}')
            WHERE evaluation_id IN ({placeholders})
            """,
            present_list,
        )
    except Exception as exc:
        raise RuntimeError(
            f"collection {collection_id}: injecting synthetic results from "
            f"{results_path} failed ({type(exc).__name__}: {exc}). Most likely "
            f"the vendored extract was generated against an older EEE schema — "
            f"re-run the extractor."
        ) from exc
    n_injected = con.execute(
        f"SELECT count(*) FROM read_parquet('{rp}') "
        f"WHERE evaluation_id IN ({placeholders})",
        present_list,
    ).fetchone()[0]

    con.execute(
        f"""
        INSERT INTO collection_protocol_map
        SELECT evaluation_id, CAST(result_idx AS INTEGER),
               ?, protocol_condition,
               CAST(n_trajectories AS INTEGER)
        FROM read_parquet('{rp}')
        WHERE evaluation_id IN ({placeholders})
        """,
        [collection_id] + present_list,
    )

    trajectories_path = adapter_dir / "trajectories.parquet"
    if trajectories_path.exists():
        tp = trajectories_path.as_posix().replace("'", "''")
        con.execute(
            f"INSERT INTO collection_trajectories_raw BY NAME "
            f"SELECT * FROM read_parquet('{tp}')"
        )

    log.info(
        "collection %s: dropped %d fragment row(s), injected %d synthetic "
        "result(s) across %d member record(s)",
        collection_id, n_dropped, n_injected, len(present),
    )


def assert_cache_has_collections(con) -> None:
    """Guard for `--from-stage` runs that restore a cache written BEFORE
    the collections step existed (or before a vendored extract was added).

    `restore_through` silently skips missing tables, so a pre-collections
    cache restores fragment-laden `results_exploded` while the collection
    tables come back as empty stand-ins — the adapter never ran, the leak
    guard passes vacuously, and the 561 fragment rows would silently
    republish. Hard-fail instead: if vendored collection manifests exist
    on disk but fewer collections are registered on the connection, the
    restored state predates them.
    """
    vdir = vendor_collections_dir()
    if not vdir.is_dir():
        return
    manifests = sorted(vdir.glob("*/manifest.json"))
    if not manifests:
        return
    create_collection_tables(con)
    n_registered = con.execute(
        "SELECT count(DISTINCT collection_id) FROM collection_study_slugs"
    ).fetchone()[0]
    if n_registered >= len(manifests):
        return
    raise RuntimeError(
        f"stale cache: {len(manifests)} vendored collection extract(s) exist "
        f"under {vdir} but the restored cache registers only {n_registered} — "
        f"the cached tables predate the collection adapter and would "
        f"republish the raw fragment rows. Re-run from Stage A/B (or wipe "
        f"the cache snapshot) so the adapter runs."
    )


def assert_no_member_leak(con) -> None:
    """Post-Stage-C leak + new-record guard: zero surviving rows may
    carry a registered study's source_name slug without being in that
    collection's manifest. A hit means a member escaped the manifest or new
    upstream records arrived — either way the extract must be regenerated.
    """
    create_collection_tables(con)
    n_slugs = con.execute(
        "SELECT count(*) FROM collection_study_slugs"
    ).fetchone()[0]
    if n_slugs == 0:
        return
    rows = con.execute(
        f"""
        SELECT s.collection_id, count(*) AS n
        FROM results_resolved rr
        JOIN collection_study_slugs s
          ON {slug_sql('rr.source_metadata.source_name')} = s.study_slug
        WHERE rr.evaluation_id NOT IN
              (SELECT evaluation_id FROM collection_member_ids)
        GROUP BY 1
        """
    ).fetchall()
    if rows:
        detail = ", ".join(f"{cid}: {n} row(s)" for cid, n in rows)
        raise RuntimeError(
            f"collection leak guard: rows matching a registered study's "
            f"source_name are not in its manifest ({detail}). Either a member "
            f"escaped the manifest or new upstream records arrived — re-run "
            f"the collection extractor and commit the regenerated vendor "
            f"files before consuming this EEE revision."
        )

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

Private collections: some extracts are not vendored in this repo but
published to the private dataset `EEE_PRIVATE_DATASET_REPO` under
`collections/<name>/` (same three files). With `EEE_INCLUDE_PRIVATE=1`,
`ensure_private_collections` downloads them and the adapter applies them
self-contained: their member records are on no public repo, so nothing is
dropped and every synthetic result and trajectory is injected. With the
toggle off nothing is downloaded or read, and `active_collections` holds
the curated entries marked `private_source` out of the guards that expect
them to be observed; `assert_no_private_rows` fails the run if rows of
such a collection turn up anyway.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
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


def _private_source(entry: dict) -> dict | None:
    marker = entry.get("private_source")
    if not marker:
        return None
    return marker if isinstance(marker, dict) else {}


def _manifest_paths(vdir: Path | None) -> list[Path]:
    if vdir is None or not vdir.is_dir():
        return []
    return sorted(vdir.glob("*/manifest.json"))


def ensure_private_collections(
    local_dir: str,
    hf_token: str | None,
    force_refresh: bool,
    revision: str | None = None,
) -> tuple[Path, str | None]:
    """Download `collections/**` of the private dataset into `local_dir`.

    Returns the local collections dir and the commit it holds. The
    revision is resolved to one sha (HEAD when unpinned) and recorded in a
    marker; a dir whose marker already names it is reused without
    downloading, so a pinned re-run needs no network. A changed revision
    replaces the whole collections dir, so no extract of an older commit
    survives. With no revision requested, a dir that holds extracts but no
    marker is hand-built and is used as is. Only called when
    `EEE_INCLUDE_PRIVATE=1`.
    """
    import shutil

    from huggingface_hub import HfApi, hf_hub_download

    from eval_card_backend.config import EEE_PRIVATE_DATASET_REPO
    from eval_card_backend.sources._revision_cache import (
        _MARKER,
        cache_revision_ok,
        write_cache_revision,
    )

    target = Path(local_dir).resolve()
    coll_dir = target / "collections"
    has_marker = (target / _MARKER).exists()
    if (
        revision is None and not has_marker and not force_refresh
        and _manifest_paths(coll_dir)
    ):
        log.info("private collections: hand-built dir at %s, nothing to sync", coll_dir)
        return coll_dir, revision
    if (
        revision is not None and not force_refresh
        and cache_revision_ok(target, revision) and _manifest_paths(coll_dir)
    ):
        return coll_dir, revision

    repo = EEE_PRIVATE_DATASET_REPO
    try:
        api = HfApi()
        resolved = revision or api.dataset_info(repo, token=hf_token).sha
        if (
            not force_refresh and cache_revision_ok(target, resolved)
            and _manifest_paths(coll_dir)
        ):
            return coll_dir, resolved
        files = sorted(
            f for f in api.list_repo_files(
                repo, repo_type="dataset", revision=resolved, token=hf_token,
            )
            if f.startswith("collections/")
        )
        if not files:
            raise RuntimeError(f"no collections/ files at revision {resolved}")
        shutil.rmtree(coll_dir, ignore_errors=True)
        (target / _MARKER).unlink(missing_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        for filename in files:
            hf_hub_download(
                repo_id=repo, filename=filename, repo_type="dataset",
                revision=resolved, local_dir=str(target), token=hf_token,
            )
    except Exception as exc:
        raise RuntimeError(
            f"private collection source {repo} could not be read "
            f"({type(exc).__name__}: {exc}). EEE_INCLUDE_PRIVATE=1 requires an "
            f"HF_TOKEN with read access to that dataset; unset "
            f"EEE_INCLUDE_PRIVATE to build without it."
        ) from exc
    write_cache_revision(target, resolved)
    log.info(
        "private collections: %d file(s) from %s at %s into %s",
        len(files), repo, resolved, coll_dir,
    )
    return coll_dir, resolved


@dataclass(frozen=True)
class ActiveCollections:
    """What the private-source toggle leaves in force for one run.

    `curated`: curated entries whose curated-key guard applies.
    `exempt_composites`: taxonomy composites whose scoped-member guard is
    skipped because their only producer is a private collection that is
    switched off this run.
    `held_out`: curated entries of those private collections, which the
    no-private-rows guard checks against.
    """

    curated: dict[str, dict]
    exempt_composites: frozenset[str]
    held_out: dict[str, dict]


def active_collections(
    include_private: bool, *, curated: dict[str, dict] | None = None,
) -> ActiveCollections:
    """The one filter every toggle-dependent step reads. With the private
    source on, everything is in force; off, curated entries marked
    `private_source` and the composites they declare are held out."""
    curated = curated if curated is not None else load_curated()
    if include_private:
        return ActiveCollections(dict(curated), frozenset(), {})
    held_out = {
        cid: entry for cid, entry in curated.items()
        if _private_source(entry) is not None
    }
    exempt = frozenset(
        slug
        for entry in held_out.values()
        for slug in (_private_source(entry).get("composites") or [])
    )
    return ActiveCollections(
        {cid: e for cid, e in curated.items() if cid not in held_out},
        exempt,
        held_out,
    )


def assert_no_private_rows(con, held_out: dict[str, dict], *, where: str) -> None:
    """Hard-fail when rows of a private collection are on the connection
    although the private source is off (a stage cache restored from an
    earlier run, or the records reappearing in the public datastore). A row
    belongs to a private collection when its raw collection key is one of
    the entry's `merge_raw_keys`, or its collection_id is the entry's id.

    Checks every row-carrying table present, so the same call works after
    Stage A, after a cache restore and before a warehouse write.
    """
    if not held_out:
        return
    keys = sorted({
        k for cid, entry in held_out.items()
        for k in (entry.get("merge_raw_keys") or [cid])
    })
    ids = sorted(held_out)
    raw_key = collection_raw_key_sql(
        "source_metadata.source_organization_name",
        "source_metadata.source_name",
        "eval_library.name",
        "source_config",
    )
    hits: list[str] = []
    for table in (
        "eee_raw", "results_exploded", "results_resolved",
        "fact_results_staging", "fact_results_signaled", "fact_results",
        "eval_results_view",
    ):
        cols = {
            r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ?", [table],
            ).fetchall()
        }
        preds, params = [], []
        if {"source_metadata", "eval_library", "source_config"} <= cols:
            preds.append(f"{raw_key} IN ({', '.join('?' for _ in keys)})")
            params += keys
        if "collection_id" in cols:
            preds.append(f"collection_id IN ({', '.join('?' for _ in ids)})")
            params += ids
        if not preds:
            continue
        n = con.execute(
            f"SELECT count(*) FROM {table} WHERE " + " OR ".join(preds), params
        ).fetchone()[0]
        if n:
            hits.append(f"{table}: {n} row(s)")
    if hits:
        raise RuntimeError(
            f"private-source guard ({where}): EEE_INCLUDE_PRIVATE is off but "
            f"rows of a private collection ({', '.join(ids)}) are present "
            f"({'; '.join(hits)}). A stage cache from a run with the private "
            f"source on is being reused, or the public datastore carries "
            f"these records again. Rebuild from Stage A, or curate the "
            f"collection as public."
        )


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
    from eval_card_backend.sources import eee

    return pinned or eee.cached_revision(eee_root)


def apply_vendor_collections(
    con,
    *,
    eee_root: Path | None,
    eee_revision: str | None,
    vendor_dir: Path | None = None,
    curated: dict[str, dict] | None = None,
    private_collections_dir: Path | None = None,
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

    Adapters under `private_collections_dir` (the private dataset's
    extracts, passed only when `EEE_INCLUDE_PRIVATE=1`) are self-contained:
    see `_apply_one_adapter`.
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
    for manifest_path in _manifest_paths(vdir):
        _apply_one_adapter(con, manifest_path, eee_root, eee_revision)
    for manifest_path in _manifest_paths(private_collections_dir):
        _apply_one_adapter(
            con, manifest_path, eee_root, eee_revision, self_contained=True
        )


def _apply_one_adapter(
    con,
    manifest_path: Path,
    eee_root: Path | None,
    eee_revision: str | None,
    *,
    self_contained: bool = False,
) -> None:
    """Apply one adapter. A `self_contained` adapter (from the private
    dataset) has no member records in any loaded corpus: nothing is
    dropped, every synthetic result and trajectory is injected, and its pin
    is the private dataset revision it was downloaded at, so the EEE
    revision check does not apply. A member record present in `eee_raw`
    is then a hard failure (the study would be counted twice)."""
    adapter_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection_id = manifest["collection_id"]
    study_slug = manifest["study_slug"]
    members = manifest["members"]
    if not self_contained and not manifest.get("eee_revision"):
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
    if self_contained:
        if present:
            raise RuntimeError(
                f"collection {collection_id}: served from the private dataset "
                f"but {len(present)} of its member record(s) are also in the "
                f"EEE corpus (e.g. {sorted(present)[:3]}). Remove them upstream "
                f"or ship the collection as an in-repo vendored extract."
            )
        _inject_synthetic(con, adapter_dir, collection_id, None)
        return
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

    _inject_synthetic(con, adapter_dir, collection_id, sorted(present), n_dropped)


def _inject_synthetic(
    con,
    adapter_dir: Path,
    collection_id: str,
    present_list: list[str] | None,
    n_dropped: int = 0,
) -> None:
    """Inject the adapter's synthetic results, protocol points and
    trajectories. `present_list` limits the results to those base member
    records; None injects all of them."""
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

    if present_list is None:
        where, params = "TRUE", []
    else:
        where = f"evaluation_id IN ({', '.join('?' for _ in present_list)})"
        params = present_list
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
            WHERE {where}
            """,
            params,
        )
    except Exception as exc:
        raise RuntimeError(
            f"collection {collection_id}: injecting synthetic results from "
            f"{results_path} failed ({type(exc).__name__}: {exc}). Most likely "
            f"the vendored extract was generated against an older EEE schema — "
            f"re-run the extractor."
        ) from exc
    n_injected = con.execute(
        f"SELECT count(*) FROM read_parquet('{rp}') WHERE {where}",
        params,
    ).fetchone()[0]

    con.execute(
        f"""
        INSERT INTO collection_protocol_map
        SELECT evaluation_id, CAST(result_idx AS INTEGER),
               ?, protocol_condition,
               CAST(n_trajectories AS INTEGER)
        FROM read_parquet('{rp}')
        WHERE {where}
        """,
        [collection_id] + params,
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
        "result(s)",
        collection_id, n_dropped, n_injected,
    )


def assert_cache_has_collections(
    con, private_collections_dir: Path | None = None
) -> None:
    """Guard for `--from-stage` runs that restore a cache written BEFORE
    the collections step existed (or before a vendored extract was added).

    `restore_through` silently skips missing tables, so a pre-collections
    cache restores fragment-laden `results_exploded` while the collection
    tables come back as empty stand-ins — the adapter never ran, the leak
    guard passes vacuously, and the 561 fragment rows would silently
    republish. Hard-fail instead: if vendored collection manifests exist
    on disk but fewer collections are registered on the connection, the
    restored state predates them. Private-dataset extracts count too when
    the run has them (`private_collections_dir`).
    """
    vdir = vendor_collections_dir()
    manifests = _manifest_paths(vdir) + _manifest_paths(private_collections_dir)
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

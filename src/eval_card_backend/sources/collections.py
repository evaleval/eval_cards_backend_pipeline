"""Collections: submission-channel tagging + vendored collection adapters.

This module derives stable collection ids from raw submission fields and
applies processed collection extracts during canonicalisation:

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

Private extracts: some processed extracts are not vendored in this repo but
published to the private dataset `EEE_PRIVATE_DATASET_REPO` under
`collections/<name>/` (same three files). With `EEE_INCLUDE_PRIVATE=1`,
`ensure_private_collections` downloads them at `EEE_PRIVATE_REVISION`. The
adapter drops any public member result rows and injects the complete processed
result and trajectory set; with `EEE_PRIVATE_ALLOW_ZERO_MEMBERS=1` it also
accepts a corpus with no public members. With the toggle off, nothing private is read
and public member records flow through as ordinary records. Curated collection
metadata applies in both modes.
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
    survives. A dir carrying the `.hand_built` sentinel, or, with no
    revision requested, one that holds extracts but no marker, is hand-built
    and is used as is; the commit returned for it is None. A refresh of a
    dir with the sentinel raises `HandBuiltRefreshRefused`. Only called when
    `EEE_INCLUDE_PRIVATE=1`.
    """
    import shutil

    from huggingface_hub import HfApi, hf_hub_download

    from eval_card_backend.config import EEE_PRIVATE_DATASET_REPO
    from eval_card_backend.sources._revision_cache import (
        _MARKER,
        cache_revision_ok,
        has_sentinel,
        log_hand_built,
        refuse_hand_built_refresh,
        write_cache_revision,
    )

    target = Path(local_dir).resolve()
    coll_dir = target / "collections"
    refuse_hand_built_refresh(target, "private collections", force_refresh)
    if has_sentinel(target):
        log_hand_built(log, "private collections", coll_dir, revision)
        return coll_dir, None
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
    """Collection metadata and composite guard exemptions for one run.

    `curated`: curated entries whose curated-key guard applies.
    `exempt_composites`: taxonomy composites whose scoped-member guard is
    skipped. The composite remains active.
    `optional_curated`: entries allowed to be unobserved while the private
    toggle is off.
    """

    curated: dict[str, dict]
    exempt_composites: frozenset[str]
    optional_curated: frozenset[str]


def active_collections(
    include_private: bool,
    *,
    allow_zero_private_members: bool = False,
    curated: dict[str, dict] | None = None,
) -> ActiveCollections:
    """Return collection metadata active in either private-toggle mode.

    `include_private` controls only whether processed extracts are loaded. It
    does not hide collections whose aggregate member records are public. Off,
    and only with `allow_zero_private_members`, a private-source entry may be
    unobserved and exempts its declared composites from the strict
    scoped-member guard.
    """
    curated = curated if curated is not None else load_curated()
    exempt = frozenset()
    optional = frozenset()
    if not include_private and allow_zero_private_members:
        exempt = frozenset(
            composite
            for entry in curated.values()
            if _private_source(entry) is not None
            for composite in (_private_source(entry).get("composites") or [])
        )
        optional = frozenset(
            cid for cid, entry in curated.items()
            if _private_source(entry) is not None
        )
    return ActiveCollections(dict(curated), exempt, optional)


def assert_curated_keys_observed(
    con,
    curated: dict[str, dict],
    *,
    strict: bool,
    optional: frozenset[str] = frozenset(),
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
    warning too. Entries in `optional` may be wholly unobserved; this supports
    the toggle-off transition before their aggregate records are public.
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
        if missing and len(missing) == len(keys) and cid in optional:
            log.warning(
                "collections: optional curated entry %s is unobserved on this "
                "public revision while its processed extract is off",
                cid,
            )
        elif missing and len(missing) == len(keys):
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
        "CREATE TABLE IF NOT EXISTS collection_adapter_accounting ("
        "collection_id VARCHAR, members_total BIGINT, members_present BIGINT, "
        "members_absent BIGINT, members_absent_nonzero BIGINT, "
        "expected_drop BIGINT, dropped BIGINT, injected BIGINT, "
        "private_revision VARCHAR)"
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
    private_revision: str | None = None,
    strict_membership: bool = True,
    allow_zero_private_members: bool = False,
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

    Adapters under `private_collections_dir` are processed private extracts,
    passed only when `EEE_INCLUDE_PRIVATE=1`. They may now have public member
    records. Their immutable source pin is `private_revision`, already checked
    by `ensure_private_collections`; `manifest.eee_revision` describes the raw
    extraction input and is not an adapter pin.
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
            con,
            manifest_path,
            eee_root,
            eee_revision,
            private_extract=True,
            private_revision=private_revision,
            strict_membership=strict_membership,
            allow_zero_private_members=allow_zero_private_members,
        )


def _apply_one_adapter(
    con,
    manifest_path: Path,
    eee_root: Path | None,
    eee_revision: str | None,
    *,
    private_extract: bool = False,
    private_revision: str | None = None,
    strict_membership: bool = True,
    allow_zero_private_members: bool = False,
) -> None:
    """Apply one adapter.

    A `private_extract` adapter is downloaded at `private_revision`, which is
    independent of the public EEE revision. Public member rows are dropped
    when present, while the complete private result and trajectory set is
    always injected. A corpus with no public members is accepted only with
    `EEE_PRIVATE_ALLOW_ZERO_MEMBERS=1`. Otherwise, a strict full run
    requires every absent member to account for zero result rows.
    """
    adapter_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection_id = manifest["collection_id"]
    study_slug = manifest["study_slug"]
    members = manifest["members"]
    if not private_extract and not manifest.get("eee_revision"):
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
    try:
        member_result_counts = {
            m["evaluation_id"]: int(m.get("n_results", 0)) for m in members
        }
        expected_total = int(manifest["expected_drop_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"collection {collection_id}: manifest result accounting is "
            f"missing or invalid."
        ) from exc
    if len(member_result_counts) != len(members):
        raise RuntimeError(
            f"collection {collection_id}: manifest member evaluation_ids "
            f"must be unique."
        )
    if any(n < 0 for n in member_result_counts.values()):
        raise RuntimeError(
            f"collection {collection_id}: manifest n_results values must be "
            f"non-negative."
        )
    accounted_total = sum(member_result_counts.values())
    if accounted_total != expected_total:
        raise RuntimeError(
            f"collection {collection_id}: manifest members account for "
            f"{accounted_total} result row(s), not expected_drop_count "
            f"{expected_total}."
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
    if private_extract:
        missing = member_ids - present
        missing_nonempty = sorted(
            eid for eid in missing if member_result_counts[eid] > 0
        )
        missing_result_rows = sum(member_result_counts[eid] for eid in missing)
        public_revision = _consumed_eee_revision(eee_root, eee_revision)
        if (
            not present
            and missing_nonempty
            and strict_membership
            and not allow_zero_private_members
        ):
            raise RuntimeError(
                f"collection {collection_id}: zero public members are present "
                f"at datastore pin {public_revision or 'unknown'}, but the "
                f"manifest has {len(missing_nonempty)} member(s) with "
                f"n_results > 0 and EEE_PRIVATE_ALLOW_ZERO_MEMBERS is off."
            )
        if not present and missing_nonempty and allow_zero_private_members:
            log.warning(
                "collection %s: no public members at datastore pin %s "
                "(EEE_PRIVATE_ALLOW_ZERO_MEMBERS=1); %d manifest member(s) "
                "with n_results > 0 are absent",
                collection_id,
                public_revision or "unknown",
                len(missing_nonempty),
            )
        if present and missing_result_rows and strict_membership:
            raise RuntimeError(
                f"collection {collection_id}: {len(present)} public member "
                f"record(s) are present, but {len(missing_nonempty)} absent "
                f"member(s) account for {missing_result_rows} expected result "
                f"row(s) (e.g. {missing_nonempty[:3]}). The public snapshot is "
                f"an incomplete study ingest."
            )
        if present and missing_result_rows:
            log.warning(
                "collection %s: non-strict subset has %d public members and "
                "%d expected result row(s) in absent members",
                collection_id, len(present), missing_result_rows,
            )

        expected_present_drop = _validate_present_member_result_counts(
            con, collection_id, present, member_result_counts
        )
        n_before = con.execute(
            "SELECT count(*) FROM results_exploded"
        ).fetchone()[0]
        con.execute(
            "DELETE FROM results_exploded WHERE evaluation_id IN "
            "(SELECT evaluation_id FROM collection_member_ids "
            "WHERE collection_id = ?)",
            [collection_id],
        )
        n_dropped = n_before - con.execute(
            "SELECT count(*) FROM results_exploded"
        ).fetchone()[0]
        if n_dropped != expected_present_drop:
            raise RuntimeError(
                f"collection {collection_id}: dropped {n_dropped} public "
                f"member result row(s), but the manifest accounts for "
                f"{expected_present_drop} across the {len(present)} present "
                f"member record(s). The private extract and public datastore "
                f"are out of sync."
            )
        log.info(
            "collection %s: private extract revision=%s; public members "
            "%d/%d present, %d/%d absent members account for zero rows; "
            "dropped %d/%d manifest result row(s)",
            collection_id,
            private_revision or "unknown",
            len(present),
            len(member_ids),
            sum(member_result_counts[eid] == 0 for eid in missing),
            len(missing),
            n_dropped,
            expected_total,
        )
        n_injected = _inject_synthetic(
            con, adapter_dir, collection_id, None, n_dropped
        )
        _record_adapter_accounting(
            con,
            collection_id=collection_id,
            members_total=len(member_ids),
            members_present=len(present),
            members_absent=len(missing),
            members_absent_nonzero=len(missing_nonempty),
            expected_drop=expected_total,
            dropped=n_dropped,
            injected=n_injected,
            private_revision=private_revision,
        )
        return
    if not present:
        log.info(
            "collection %s: no member records in this run's corpus — adapter inert",
            collection_id,
        )
        _record_adapter_accounting(
            con,
            collection_id=collection_id,
            members_total=len(member_ids),
            members_present=0,
            members_absent=len(member_ids),
            members_absent_nonzero=sum(
                count > 0 for count in member_result_counts.values()
            ),
            expected_drop=expected_total,
            dropped=0,
            injected=0,
            private_revision=None,
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
    expected_present_drop = _validate_present_member_result_counts(
        con, collection_id, present, member_result_counts
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

    n_injected = _inject_synthetic(
        con, adapter_dir, collection_id, sorted(present), n_dropped
    )
    _record_adapter_accounting(
        con,
        collection_id=collection_id,
        members_total=len(member_ids),
        members_present=len(present),
        members_absent=len(member_ids - present),
        members_absent_nonzero=sum(
            member_result_counts[eid] > 0 for eid in member_ids - present
        ),
        expected_drop=expected_total,
        dropped=n_dropped,
        injected=n_injected,
        private_revision=None,
    )


def _validate_present_member_result_counts(
    con,
    collection_id: str,
    present: set[str],
    member_result_counts: dict[str, int],
) -> int:
    """Validate every present member before any result rows are deleted."""
    if not present:
        return 0
    placeholders = ", ".join("?" for _ in present)
    observed = dict(
        con.execute(
            f"SELECT evaluation_id, count(*) FROM results_exploded "
            f"WHERE evaluation_id IN ({placeholders}) GROUP BY evaluation_id",
            sorted(present),
        ).fetchall()
    )
    mismatches = [
        (
            evaluation_id,
            member_result_counts[evaluation_id],
            observed.get(evaluation_id, 0),
        )
        for evaluation_id in sorted(present)
        if observed.get(evaluation_id, 0) != member_result_counts[evaluation_id]
    ]
    if mismatches:
        detail = ", ".join(
            f"{evaluation_id}: manifest={expected}, observed={actual}"
            for evaluation_id, expected, actual in mismatches
        )
        raise RuntimeError(
            f"collection {collection_id}: public member result count mismatch "
            f"before deletion ({detail}). The collection extract and public "
            f"datastore are out of sync."
        )
    return sum(member_result_counts[eid] for eid in present)


def _record_adapter_accounting(
    con,
    *,
    collection_id: str,
    members_total: int,
    members_present: int,
    members_absent: int,
    members_absent_nonzero: int,
    expected_drop: int,
    dropped: int,
    injected: int,
    private_revision: str | None,
) -> None:
    con.execute(
        "INSERT INTO collection_adapter_accounting VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            collection_id,
            members_total,
            members_present,
            members_absent,
            members_absent_nonzero,
            expected_drop,
            dropped,
            injected,
            private_revision,
        ],
    )


def _inject_synthetic(
    con,
    adapter_dir: Path,
    collection_id: str,
    present_list: list[str] | None,
    n_dropped: int = 0,
) -> int:
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
    n_before = con.execute("SELECT count(*) FROM results_exploded").fetchone()[0]
    n_expected = con.execute(
        f"SELECT count(*) FROM read_parquet('{rp}') WHERE {where}", params
    ).fetchone()[0]
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
        "SELECT count(*) FROM results_exploded"
    ).fetchone()[0] - n_before
    if n_injected != n_expected:
        raise RuntimeError(
            f"collection {collection_id}: injected {n_injected} synthetic "
            f"result row(s), expected {n_expected}."
        )

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
    return n_injected


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
    """Post-Stage-C guard for unknown public records of an adapted study.

    Check record grain so an unknown record with no evaluation results still
    fails. Known manifest members may be public and are the normal private
    adapter input.
    """
    create_collection_tables(con)
    n_slugs = con.execute(
        "SELECT count(*) FROM collection_study_slugs"
    ).fetchone()[0]
    if n_slugs == 0:
        return
    rows = con.execute(
        f"""
        SELECT s.collection_id, count(DISTINCT e.evaluation_id) AS n
        FROM eee_raw e
        JOIN collection_study_slugs s
          ON {slug_sql('e.source_metadata.source_name')} = s.study_slug
        WHERE NOT EXISTS (
            SELECT 1
            FROM collection_member_ids m
            WHERE m.collection_id = s.collection_id
              AND m.evaluation_id = e.evaluation_id
        )
        GROUP BY 1
        """
    ).fetchall()
    if rows:
        detail = ", ".join(f"{cid}: {n} row(s)" for cid, n in rows)
        raise RuntimeError(
            f"collection leak guard: public records matching a registered "
            f"study's source_name are not in its manifest ({detail}). Re-run "
            f"the collection extractor, publish the regenerated private "
            f"extract, and bump EEE_PRIVATE_REVISION before consuming this "
            f"EEE revision."
        )

"""Canonicalisation pipeline orchestrator.

End-to-end flow:
  1. Resolve upstream snapshots (EEE, registry, cards) — fetch if missing.
  2. Preflight: hard-fail on empty EEE / empty registry; warn on missing cards.
  3. Build a DuckDB connection, register the resolver + helper UDFs.
  4. Load all three sources into in-memory tables (Stages A–B).
  5. Resolve identity (Stage C), flatten + JOIN dims (D), per-row signals (E),
     group signals (F), dim materialisation (G), completeness (H).
  6. Emit parquets (Stage I) + snapshot_meta.json sidecar.
  7. Print resolver / coercion / row-count summary.

Stage caching: each stage's terminal output tables are written to
`<cache_root>/<snapshot>/<table>.parquet` after the stage runs (unless
`no_cache=True`). When `from_stage=X` is set, the orchestrator restores
cached outputs for prior stages and starts compute at X. When `to_stage=Y`
is set, the orchestrator stops after Y (no warehouse emit, no snapshot
meta sidecar — the cache dir is the result). See
`canonicalise/cache.py` for the per-stage table contract.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb

from eval_card_backend import categorisation
from eval_card_backend.canonicalise import sidecars, stages, udfs
from eval_card_backend.canonicalise.cache import (
    STAGE_ORDER,
    StageCache,
    _make_snapshot_id,
    _snapshot_dir_name,
    discover_latest_snapshot,
    normalize_snapshot_id,
    validate_letter,
)
from eval_card_backend.canonicalise.resolver_setup import register_udfs
from eval_card_backend.config import (
    BENCHMARK_METADATA_DATASET_REPO,
    EEE_DATASET_REPO,
    ENTITY_REGISTRY_DATASET_REPO,
    IGNORED_CONFIGS,
    Settings,
)
from eval_card_backend.metric_meta_hotfix import (
    log_metric_meta_summary,
    reset_provenance_counter,
)
from eval_card_backend.signals.reproducibility import (
    log_purpose_shape_summary,
    reset_purpose_shape_counter,
)
from eval_card_backend.signals.setup import (
    log_json_coerce_summary,
    reset_json_coerce_counter,
)
from eval_card_backend.sources import benchmark_cards, eee, registry as registry_src

log = logging.getLogger(__name__)


def _hf_dataset_snapshot(repo_id: str, hf_token: str | None) -> dict | None:
    """Return ``{sha, last_modified}`` for an HF dataset repo, or None on
    failure. Used to stamp every snapshot with the upstream revision pins
    so consumers can answer "is this run consuming stale registry data?"
    without re-querying HF after the fact.

    `last_modified` is ISO-8601 (HF's API serialises ``datetime`` → str
    when it goes through the JSON path; we canonicalise here).
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        info = api.dataset_info(repo_id, token=hf_token)
        last_modified = getattr(info, "last_modified", None)
        if last_modified is not None and not isinstance(last_modified, str):
            # `huggingface_hub` returns a tz-aware datetime; ISO-8601 it
            # so manifest.json round-trips cleanly through any consumer.
            last_modified = last_modified.isoformat()
        return {"sha": info.sha, "last_modified": last_modified}
    except Exception as exc:
        # Don't fail the whole snapshot for a missing revision sidecar entry,
        # but DO surface the failure — bad repo id, auth issue, and network
        # blip all surface as `null` in snapshot_meta.json otherwise. Type +
        # message is enough; the full traceback is noisy on benign 503s.
        log.warning("hf_dataset_snapshot lookup failed for %s: %s: %s",
                    repo_id, type(exc).__name__, exc)
        return None


def _hf_revision(repo_id: str, hf_token: str | None) -> str | None:
    """Backward-compatible scalar-SHA helper. Prefer `_hf_dataset_snapshot`
    for new call sites that also want `last_modified`."""
    info = _hf_dataset_snapshot(repo_id, hf_token)
    return info["sha"] if info else None


def preflight(
    eee_root: Path | None, registry_root: Path | None, cards_root: Path | None
) -> None:
    errors: list[str] = []

    if eee_root is None or not (eee_root / "data").exists() or not any(
        (eee_root / "data").rglob("*.json")
    ):
        errors.append(
            f"EEE source empty/missing at {eee_root}/data (HF download failed?). "
            f"Set EEE_REFRESH_SNAPSHOT=1 to force re-download."
        )

    aliases = (
        registry_src.aliases_path(registry_root) if registry_root else None
    )
    if aliases is None or not aliases.exists():
        errors.append(
            f"Registry alias store missing under {registry_root}. "
            f"Pull evaleval/entity-registry-data."
        )
    else:
        # File present but the alias parquet may be empty (cold-start
        # registry). Reading it via DuckDB avoids pulling a pandas dep here
        # for a one-shot row count.
        import duckdb as _duckdb
        try:
            alias_count = _duckdb.connect().execute(
                f"SELECT COUNT(*) FROM read_parquet('{aliases}')"
            ).fetchone()[0]
        except Exception as exc:
            errors.append(
                f"Registry alias store at {aliases} unreadable: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            if alias_count == 0:
                errors.append(
                    f"Registry alias store at {aliases} is empty. "
                    f"Cold-start registry? Seed before running canonicalisation."
                )

    if cards_root is None or not Path(cards_root).exists():
        log.warning(
            "AutoBenchmarkCards source missing at %s — proceeding without "
            "card content. card_present will be false on every benchmark.",
            cards_root,
        )

    if errors:
        for err in errors:
            log.error(err)
        raise SystemExit("Preflight failed; see errors above.")


def run(
    settings: Settings,
    *,
    configs: list[str] | None = None,
    config_limit: int | None = None,
    snapshot_id: str | None = None,
    warehouse_dir: str | None = None,
    registry_local_dir: str | None = None,
    skip_preflight: bool = False,
    cache_root: str | Path = ".cache/canonicalise",
    no_cache: bool = False,
    from_stage: str | None = None,
    to_stage: str | None = None,
    taxonomy_seed_dir: str | Path | None = None,
) -> Path | None:
    """Run the canonicalisation pipeline. Returns the snapshot output directory,
    or `None` when `to_stage` cuts the run off before Stage I (no warehouse
    emit; the cache dir is the result).

    `warehouse_dir` and `registry_local_dir` override the values from
    `settings` for callers (typically tests) that need an isolated path
    without mutating env vars.

    `from_stage` / `to_stage` are stage letters (A, B, C, D, E, F, G, I) that
    bracket the slice to execute. `from_stage` skips earlier stages and
    restores their cached outputs; `to_stage` stops after that stage. When
    `from_stage` is set without `snapshot_id`, the orchestrator picks the
    latest snapshot directory under `cache_root`.
    """
    if from_stage is not None:
        from_stage = validate_letter(from_stage, label="from_stage")
    if to_stage is not None:
        to_stage = validate_letter(to_stage, label="to_stage")
    if from_stage and to_stage and STAGE_ORDER.index(from_stage) > STAGE_ORDER.index(to_stage):
        raise ValueError(
            f"from_stage={from_stage!r} comes after to_stage={to_stage!r}"
        )

    if from_stage and not snapshot_id:
        snapshot_id = discover_latest_snapshot(cache_root)
        if not snapshot_id:
            raise FileNotFoundError(
                f"--from-stage={from_stage} but no snapshot found under "
                f"{cache_root}. Run without --from-stage first to populate the cache."
            )
        log.info("from_stage=%s → using latest cache snapshot %s", from_stage, snapshot_id)

    snapshot_id = normalize_snapshot_id(snapshot_id) if snapshot_id else _make_snapshot_id()
    log.info("snapshot_id = %s", snapshot_id)
    cache = StageCache(cache_root, snapshot_id, enabled=not no_cache)

    if warehouse_dir is None:
        warehouse_dir = settings.warehouse_dir
    if registry_local_dir is None:
        registry_local_dir = settings.registry_local_dir

    # 1. fetch snapshots — cheap no-ops when local data is already on disk;
    # always called so the resolver can load the alias store regardless of
    # which stages run.
    eee_root = eee.ensure_snapshot(
        settings.eee_local_dir, settings.hf_token, settings.refresh_eee
    )
    cards_root = benchmark_cards.ensure_snapshot(
        settings.benchmark_metadata_local_dir,
        settings.hf_token,
        settings.refresh_benchmark_metadata,
    )
    registry_root = registry_src.ensure_snapshot(
        registry_local_dir, settings.hf_token, force_refresh=settings.refresh_registry
    )

    runs_stage_a = (from_stage in (None, "A"))
    if not skip_preflight and runs_stage_a:
        preflight(eee_root, registry_root, cards_root)

    # Config list + cards are only needed when Stage A actually runs.
    chosen: list[str] = []
    cards: dict = {}
    if runs_stage_a:
        chosen = _select_configs(eee_root, settings, configs, config_limit)
        cards = benchmark_cards.load_cards(cards_root) if cards_root else {}

    # 2. DuckDB + UDFs
    con = duckdb.connect()
    udfs.reset_resolver_counters()
    reset_json_coerce_counter()
    reset_provenance_counter()
    reset_purpose_shape_counter()
    categorisation.reset_category_counter()
    eee.reset_drop_counter()

    alias_store = registry_src.load_alias_store(registry_root)
    from eval_entity_resolver import Resolver

    resolver = Resolver(alias_store)
    register_udfs(con, resolver)

    # Restore cached prior outputs when starting mid-pipeline. Each table
    # gets DROP-CREATE'd from its parquet so the connection has the same
    # shape it would have had after running the prior stages from scratch.
    if from_stage and from_stage != "A":
        prior = STAGE_ORDER[STAGE_ORDER.index(from_stage) - 1]
        restored = cache.restore_through(con, prior)
        log.info(
            "from_stage=%s — restored %d cached table(s) from %s: %s",
            from_stage, len(restored), cache.dir, restored,
        )

        # The composite_config_map cache is one of stage A's outputs. A 0-row
        # restoration means a prior run wrote an empty map (e.g. taxonomy
        # seed dir was missing) — and silently restoring it now would
        # propagate that corruption into every downstream stage's
        # composite_slug / composite_display_name fallbacks. Fail fast and
        # tell the operator to refresh stage A.
        if "composite_config_map" in restored:
            ccm_rows = con.execute(
                "SELECT COUNT(*) FROM composite_config_map"
            ).fetchone()[0]
            if ccm_rows == 0:
                raise RuntimeError(
                    f"composite_config_map restored from cache at {cache.dir} "
                    f"has 0 rows. A prior run wrote a corrupt cache (most likely "
                    f"because the taxonomy seed dir was missing at that time). "
                    f"Re-run with --from-stage A to rebuild the cache from "
                    f"composites.yaml / families.yaml, or wipe the snapshot "
                    f"directory under {cache.root}."
                )

    # ---- Stage execution ----
    n_eee: int | None = None
    n_cards: int | None = None
    n_exploded: int | None = None
    n_synth_collisions: int | None = None
    stage_e_stats: stages.StageEStats | None = None
    n_unit_inconsistent: int | None = None
    out_dir: Path | None = None

    from_idx = STAGE_ORDER.index(from_stage) if from_stage else 0
    to_idx = STAGE_ORDER.index(to_stage) if to_stage else len(STAGE_ORDER) - 1

    for stage_idx, letter in enumerate(STAGE_ORDER):
        if stage_idx < from_idx:
            continue
        if stage_idx > to_idx:
            break

        if letter == "A":
            log.info("Stage A: loading sources …")
            arrow_table = eee.load_arrow_table(eee_root, chosen, settings.hf_token)
            n_eee = stages.stage_a_load_eee(con, arrow_table)
            log.info("  loaded %d EEE records (validated)", n_eee)
            n_cards = stages.stage_a_load_cards(con, cards)
            log.info("  loaded %d cards", n_cards)
            dim_paths = registry_src.open_dim_paths(registry_root)
            stages.stage_a_load_registry(
                con, dim_paths, registry_root=registry_root,
                taxonomy_seed_dir=Path(taxonomy_seed_dir) if taxonomy_seed_dir else None,
            )
            log.info("  registry dims loaded: %s", sorted(dim_paths))
        elif letter == "B":
            n_exploded = stages.stage_b_explode_evaluation_results(con)
            n_synth_collisions = stages.stage_b_count_synth_id_collisions(con)
            log.info(
                "Stage B: exploded %d evaluation_results rows (synth-id collisions: %d)",
                n_exploded, n_synth_collisions,
            )
            if n_synth_collisions > 0:
                log.warning(
                    "Stage B: %d synthesised evaluation_result_id(s) collided with "
                    "real EEE-supplied ids; downstream fact_id is no longer 1:1.",
                    n_synth_collisions,
                )
        elif letter == "C":
            log.info("Stage C: resolving identities …")
            stages.stage_c_resolve_identities(con)
        elif letter == "D":
            log.info("Stage D: flattening + joining dims …")
            stages.stage_d_join_dims_and_flatten(con)
        elif letter == "E":
            stage_e_stats = stages.stage_e_per_row_signals(con)
            log.info(
                "Stage E: %d rows in, %d rows out "
                "(dropped %d no_score, %d sentinel, %d fact_id collision)",
                stage_e_stats.pre, stage_e_stats.post,
                stage_e_stats.n_dropped_no_score,
                stage_e_stats.n_dropped_sentinel,
                stage_e_stats.n_dropped_dedup,
            )
        elif letter == "F":
            log.info("Stage F: group signals …")
            n_unit_inconsistent = stages.stage_f_group_signals(con, snapshot_id)
        elif letter == "G":
            # Per-row completeness is computed in Stage E, so Stage G can
            # derive `card_missing_count` from fact_results inline.
            log.info("Stage G: dim materialisation …")
            stages.stage_g_materialise_dim_tables(con, snapshot_id)
        elif letter == "I":
            out_dir = Path(warehouse_dir) / _snapshot_dir_name(snapshot_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            log.info("Stage I: emitting parquets to %s", out_dir)
            stages.stage_i_emit_warehouse_parquets(con, out_dir, snapshot_id)
        elif letter == "J":
            log.info("Stage J: building view layer …")
            stages.stage_j_eval_results_view(con, snapshot_id)
            stages.stage_j_models_view(con, snapshot_id)
            stages.stage_j_evals_view(con, snapshot_id)
            # Re-anchor out_dir when --from-stage J skips Stage I; the
            # warehouse path is deterministic from snapshot_id so we can
            # always recompute it.
            if out_dir is None:
                out_dir = Path(warehouse_dir) / _snapshot_dir_name(snapshot_id)
                out_dir.mkdir(parents=True, exist_ok=True)
            log.info("Stage J: emitting view parquets to %s", out_dir)
            stages.stage_j_emit_view_parquets(con, out_dir, snapshot_id)

        cache.write_stage(con, letter)

    # When --to-stage cuts the run off before warehouse emit, the cache dir
    # is the result. Skip snapshot_meta and final summaries that depend on
    # post-Stage-F state we may not have.
    if out_dir is None:
        log.info(
            "to_stage=%s — skipping warehouse emit; cache dir is the result: %s",
            to_stage, cache.dir,
        )
        udfs.log_resolver_summary()
        log_json_coerce_summary()
        log_purpose_shape_summary()
        log_metric_meta_summary(log)
        categorisation.log_category_summary(log)
        eee.log_drop_summary()
        return None

    # Recover row counts and configs list from cached tables when they
    # weren't computed in this run (i.e. earlier stages were restored from
    # cache rather than rerun). The tables themselves are still in the
    # connection.
    n_eee = n_eee if n_eee is not None else _table_count(con, "eee_raw")
    n_cards = n_cards if n_cards is not None else _table_count(con, "cards_raw")
    n_exploded = (
        n_exploded if n_exploded is not None else _table_count(con, "results_exploded")
    )
    if n_synth_collisions is None and n_exploded is not None:
        try:
            n_synth_collisions = stages.stage_b_count_synth_id_collisions(con)
        except duckdb.CatalogException:
            log.warning(
                "results_exploded missing; synthesised_id_collisions = None"
            )
    # When Stage E was restored from cache rather than re-run, recover row
    # counts from the cached tables. Sentinel and dedup breakdowns are
    # re-derived: the sentinel predicate is pure SQL against
    # `fact_results_staging`, and dedup count = pre - no_score - sentinel
    # - post. This keeps `snapshot_meta.row_counts` accurate across
    # `--from-stage F+` reruns instead of zeroing out two real signals.
    if stage_e_stats is None:
        pre_count = _table_count(con, "fact_results_staging")
        post_count = _table_count(con, "fact_results_signaled")
        n_no_score = None
        n_sentinel = None
        n_dedup = None
        if pre_count is not None:
            try:
                n_no_score = con.execute(
                    "SELECT count(*) FROM fact_results_staging WHERE score IS NULL"
                ).fetchone()[0]
                n_sentinel = con.execute(
                    f"SELECT count(*) FROM fact_results_staging "
                    f"WHERE score IS NOT NULL "
                    f"AND ({stages._SENTINEL_DROP_PREDICATE})"
                ).fetchone()[0]
                if post_count is not None:
                    n_dedup = max(
                        0,
                        pre_count - n_no_score - n_sentinel - post_count,
                    )
            except duckdb.CatalogException:
                log.warning(
                    "fact_results_staging missing; Stage E breakdowns = None"
                )
        stage_e_stats = stages.StageEStats(
            pre=pre_count if pre_count is not None else 0,
            n_dropped_no_score=n_no_score if n_no_score is not None else 0,
            n_dropped_sentinel=n_sentinel if n_sentinel is not None else 0,
            n_dropped_dedup=n_dedup if n_dedup is not None else 0,
            post=post_count if post_count is not None else 0,
        )
    if not chosen:
        try:
            chosen = [
                r[0] for r in con.execute(
                    "SELECT DISTINCT source_config FROM eee_raw "
                    "WHERE source_config IS NOT NULL ORDER BY source_config"
                ).fetchall()
            ]
        except duckdb.CatalogException:
            log.warning("eee_raw missing; snapshot_meta.configs = []")
            chosen = []

    # 12. Snapshot meta sidecar
    j_in_slice = STAGE_ORDER.index("J") in range(from_idx, to_idx + 1)
    meta = _build_snapshot_meta(
        snapshot_id=snapshot_id,
        chosen=chosen,
        settings=settings,
        n_eee=n_eee,
        n_cards=n_cards,
        n_exploded=n_exploded,
        n_synth_collisions=n_synth_collisions,
        stage_e_stats=stage_e_stats,
        n_unit_inconsistent=n_unit_inconsistent,
        view_layer_emitted=j_in_slice,
    )
    (out_dir / "snapshot_meta.json").write_text(json.dumps(meta, indent=2))

    # View-layer JSON sidecars (manifest, headline, hierarchy). Only emitted
    # when Stage J was in the executed slice — the consumers all key off
    # the view tables, which only exist on the connection then.
    if j_in_slice:
        sidecars.write_manifest(con, out_dir, meta)
        sidecars.write_headline(con, out_dir, meta)
        sidecars.write_hierarchy(con, out_dir, meta)
        sidecars.write_comparison_index(con, out_dir, meta)
        sidecars.write_benchmark_index(con, out_dir, meta)
        log.info(
            "Stage J: wrote sidecars (manifest, headline, hierarchy, "
            "comparison-index, benchmark_index) to %s",
            out_dir,
        )

    # 13. Summaries
    udfs.log_resolver_summary()
    log_json_coerce_summary()
    log_metric_meta_summary(log)
    categorisation.log_category_summary(log)
    eee.log_drop_summary()
    _log_canonicalisation_summary(con, stage_e_stats.post)

    return out_dir


def _select_configs(
    eee_root: Path,
    settings: Settings,
    configs: list[str] | None,
    config_limit: int | None,
) -> list[str]:
    """Resolve the EEE configs we'll process: discover, filter by user
    selection, drop unconditionally-ignored configs, then truncate to
    `config_limit` if set. `IGNORED_CONFIGS` applies even when a user
    explicitly passes the config name via `--configs` — these are not
    user-overridable.
    """
    all_configs = eee.discover_configs(eee_root, settings.hf_token)
    if configs is not None:
        wanted = set(configs)
        chosen = [c for c in all_configs if c in wanted]
    else:
        chosen = list(all_configs)
    excluded = [c for c in chosen if c in IGNORED_CONFIGS]
    if excluded:
        log.warning(
            "ignoring %d configs (upstream_data_quality): %s",
            len(excluded), excluded,
        )
        chosen = [c for c in chosen if c not in IGNORED_CONFIGS]
    if config_limit is not None:
        chosen = chosen[:config_limit]
    log.info(
        "running over %d / %d configs: %s",
        len(chosen), len(all_configs), chosen[:10],
    )
    return chosen


def _build_snapshot_meta(
    *,
    snapshot_id: str,
    chosen: list[str],
    settings: Settings,
    n_eee: int | None,
    n_cards: int | None,
    n_exploded: int | None,
    n_synth_collisions: int | None,
    stage_e_stats: stages.StageEStats,
    n_unit_inconsistent: int | None,
    view_layer_emitted: bool,
) -> dict:
    """Assemble `snapshot_meta.json` payload. Pure data: doesn't touch
    DuckDB. The HF-revision lookups talk to HF's HTTP API and are best-
    effort (see `_hf_revision`).
    """
    tables: list[str] = [
        "fact_results.parquet",
        "benchmarks.parquet",
        "composites.parquet",
        "families.parquet",
        "models.parquet",
        "canonical_metrics.parquet",
    ]
    sidecars: list[str] = []
    if view_layer_emitted:
        tables += [
            "eval_results_view.parquet",
            "models_view.parquet",
            "evals_view.parquet",
        ]
        sidecars = [
            "manifest.json", "headline.json", "hierarchy.json",
            "benchmark_index.json",
        ]

    # Single HTTP call per upstream — captures both sha and last_modified
    # so manifest.json can surface "registry parquet was last refreshed
    # at <ts>; this snapshot's run consumed it" without a follow-up query.
    eee_info = _hf_dataset_snapshot(EEE_DATASET_REPO, settings.hf_token)
    registry_info = _hf_dataset_snapshot(
        ENTITY_REGISTRY_DATASET_REPO, settings.hf_token
    )
    cards_info = _hf_dataset_snapshot(
        BENCHMARK_METADATA_DATASET_REPO, settings.hf_token
    )

    return {
        "snapshot_id": snapshot_id,
        "generated_at": _make_snapshot_id(),
        "configs": chosen,
        "eee_revision": eee_info["sha"] if eee_info else None,
        "registry_revision": registry_info["sha"] if registry_info else None,
        "cards_revision": cards_info["sha"] if cards_info else None,
        # Structured upstream-pin records — same data as the *_revision
        # scalars plus last_modified. Consumed by `write_manifest` so the
        # warehouse manifest carries enough info to diagnose stale-input
        # runs without re-querying HF.
        "upstream_pins": {
            "eee_datastore": {"repo_id": EEE_DATASET_REPO, **(eee_info or {"sha": None, "last_modified": None})},
            "entity_registry": {"repo_id": ENTITY_REGISTRY_DATASET_REPO, **(registry_info or {"sha": None, "last_modified": None})},
            "benchmark_metadata": {"repo_id": BENCHMARK_METADATA_DATASET_REPO, **(cards_info or {"sha": None, "last_modified": None})},
        },
        "tables": tables,
        "sidecars": sidecars,
        "row_counts": {
            "eee_records": n_eee,
            "exploded_results": n_exploded,
            "fact_results_pre_drop": stage_e_stats.pre,
            "fact_results": stage_e_stats.post,
            "dropped_rows_no_score": stage_e_stats.n_dropped_no_score,
            "dropped_rows_sentinel": stage_e_stats.n_dropped_sentinel,
            "dropped_rows_dedup": stage_e_stats.n_dropped_dedup,
            "dropped_eee_records_stage_a": sum(eee._drop_counter.values()),
            "cards": n_cards,
            "comparability_groups_metric_unit_inconsistent": n_unit_inconsistent,
            "synthesised_id_collisions": n_synth_collisions,
        },
    }


def _table_count(con, table: str) -> int | None:
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except duckdb.CatalogException:
        log.warning("table %s not present in connection; row count = None", table)
        return None


def _log_canonicalisation_summary(con, fact_count: int) -> None:
    log.info("=== canonicalisation summary ===")
    log.info("  fact_results rows: %d", fact_count)
    if fact_count == 0:
        log.warning("  fact_results is EMPTY — investigate before downstream use")
        return
    row = con.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE model_id IS NULL)        AS unresolved_model,
            COUNT(*) FILTER (WHERE benchmark_id IS NULL)    AS unresolved_benchmark,
            COUNT(*) FILTER (WHERE metric_id IS NULL)       AS unresolved_metric,
            COUNT(*) FILTER (WHERE org_id IS NULL)          AS unresolved_org,
            COUNT(*) FILTER (WHERE harness_id IS NULL)      AS unresolved_harness,
            COUNT(*) FILTER (WHERE score_scale_anomaly)     AS score_scale_anomalies,
            COUNT(DISTINCT (model_id, benchmark_id, metric_id))
                FILTER (WHERE model_id IS NOT NULL
                        AND benchmark_id IS NOT NULL
                        AND metric_id IS NOT NULL)          AS resolved_groups,
            COUNT(DISTINCT comparability_group_id)
                FILTER (WHERE variant_divergence_threshold IS NOT NULL)
                                                            AS variant_eligible,
            COUNT(DISTINCT comparability_group_id)
                FILTER (WHERE cross_party_divergence_threshold IS NOT NULL)
                                                            AS cross_eligible,
            COUNT(DISTINCT comparability_group_id)
                FILTER (WHERE has_variant_divergence)       AS variant_pos,
            COUNT(DISTINCT comparability_group_id)
                FILTER (WHERE has_cross_party_divergence)   AS cross_pos,
            COUNT(*)            FILTER (WHERE completeness_score IS NOT NULL) AS bc_n,
            AVG(completeness_score) FILTER (WHERE completeness_score IS NOT NULL) AS bc_mean,
            MIN(completeness_score) FILTER (WHERE completeness_score IS NOT NULL) AS bc_min,
            MAX(completeness_score) FILTER (WHERE completeness_score IS NOT NULL) AS bc_max
        FROM fact_results
        """
    ).fetchone()
    (
        unresolved_model, unresolved_benchmark, unresolved_metric,
        unresolved_org, unresolved_harness, score_scale_anomalies,
        resolved_groups, variant_eligible, cross_eligible,
        variant_pos, cross_pos, bc_n, bc_mean, bc_min, bc_max,
    ) = row
    log.info(
        "  unresolved: model=%d benchmark=%d metric=%d org=%d harness=%d",
        unresolved_model, unresolved_benchmark, unresolved_metric,
        unresolved_org, unresolved_harness,
    )
    log.info("  score_scale_anomalies: %d", score_scale_anomalies)
    log.info("  resolved (m,b,metric) groups: %d", resolved_groups)
    log.info("  variant-divergence eligible groups: %d", variant_eligible)
    log.info("  cross-party-divergence eligible groups: %d", cross_eligible)
    log.info("  variant-divergence groups: %d", variant_pos)
    log.info("  cross-party-divergence groups: %d", cross_pos)
    if bc_n:
        log.info(
            "  completeness (per-row): n=%d mean=%.3f min=%.3f max=%.3f",
            bc_n, bc_mean or 0.0, bc_min or 0.0, bc_max or 0.0,
        )
    no_card = con.execute(
        "SELECT COUNT(*) FROM benchmarks WHERE card_present = FALSE"
    ).fetchone()[0]
    log.info("  benchmarks without AutoBenchmarkCard: %d", no_card)

    # Card resolution stats: how many cards loaded, how many resolved to a
    # canonical benchmark_id, how many remain orphan (resolver couldn't match).
    # Operator-visible signal that the cards corpus is or isn't tracking the
    # registry's canonical names.
    cards_total = con.execute("SELECT COUNT(*) FROM cards_raw").fetchone()[0]
    cards_resolved = con.execute(
        "SELECT COUNT(*) FROM cards_raw WHERE benchmark_id IS NOT NULL"
    ).fetchone()[0]
    cards_orphan = cards_total - cards_resolved
    log.info(
        "  cards: total=%d resolved=%d orphan=%d", cards_total, cards_resolved, cards_orphan
    )

    # Canonical-id collapse detection: one canonical_id absorbing many
    # distinct raw forms is usually a registry-side parent-collapse pathology
    # — e.g. MMLU subtasks (Abstract Algebra, Philosophy, ...) all aliased to
    # parent `mmlu`. Loss of granularity is silent in the warehouse; this
    # surfaces it for registry-alias backfill triage. Threshold is intentionally
    # coarse — small collapses (2-3 raws) are normal casing/whitespace noise.
    _log_canonical_collapse_summary(con)


_COLLAPSE_THRESHOLD = 5


def _log_canonical_collapse_summary(con) -> None:
    log.info("  canonical-id collapse (>%d distinct raws):", _COLLAPSE_THRESHOLD)
    any_found = False
    for entity, raw_col, id_col in [
        ("model", "model_raw", "model_id"),
        ("benchmark", "benchmark_raw", "benchmark_id"),
        ("metric", "metric_raw", "metric_id"),
        ("org", "org_raw", "org_id"),
        ("harness", "harness_raw", "harness_id"),
    ]:
        rows = con.execute(
            f"""
            SELECT {id_col} AS canonical_id, COUNT(DISTINCT {raw_col}) AS n_raws
            FROM fact_results
            WHERE {id_col} IS NOT NULL
            GROUP BY 1
            HAVING n_raws > {_COLLAPSE_THRESHOLD}
            ORDER BY n_raws DESC
            LIMIT 5
            """
        ).fetchall()
        if rows:
            any_found = True
            top = ", ".join(f"{cid}×{n}" for cid, n in rows)
            log.info("    %s: %s", entity, top)
    if not any_found:
        log.info("    (none)")


run_pipeline = run  # alias for callers

from __future__ import annotations

import argparse
import json
import logging
import sys

from eval_card_backend.config import Settings
from eval_card_backend.sources import benchmark_cards, eee


def _cmd_summary(args: argparse.Namespace) -> int:
    """Inspect what's in the local cache without triggering downloads.

    The bare `eval-card-backend` invocation is meant for "what's already
    here?" — surfacing missing caches as `(not cached)` rather than
    silently kicking off multi-GB HF fetches. The `canonicalise` subcommand
    is the explicit fetch path.
    """
    from pathlib import Path

    settings = Settings.from_env()

    eee_local = Path(settings.eee_local_dir).resolve()
    cards_local = Path(settings.benchmark_metadata_local_dir).resolve()

    eee_root: Path | None = (
        eee_local if (eee_local / "data").is_dir() else None
    )
    cards_root: Path | None = cards_local if cards_local.exists() else None

    cards = benchmark_cards.load_cards(cards_root) if cards_root else {}

    if eee_root is None:
        configs: list[str] = []
    else:
        configs = eee.discover_configs(eee_root, hf_token=None)
        if args.configs:
            wanted = {c.strip() for c in args.configs.split(",") if c.strip()}
            configs = [c for c in configs if c in wanted]
        if args.config_limit:
            configs = configs[: args.config_limit]

    summary = {
        "eee_root": str(eee_root) if eee_root else "(not cached)",
        "benchmark_cards_root": str(cards_root) if cards_root else "(not cached)",
        "benchmark_card_count": len(cards),
        "config_count": len(configs),
        "configs": [
            {
                "name": c,
                "json_files": len(eee.list_json_files(c, eee_root, hf_token=None)),
            }
            for c in configs
        ],
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_canonicalise(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Quiet down some noisy loggers from upstream deps unless caller asks otherwise.
    # httpx emits one INFO line per HTTP request; with ~18k record-file fetches
    # that buries the actual pipeline log under per-file noise. The HF tqdm
    # bar is similarly per-file in non-TTY (CI) sinks — disable it too.
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()

    from eval_card_backend.canonicalise import pipeline

    settings = Settings.from_env()

    configs = None
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    out_dir = pipeline.run(
        settings,
        configs=configs,
        config_limit=args.config_limit,
        snapshot_id=args.snapshot_id,
        warehouse_dir=args.warehouse,
        registry_local_dir=args.registry_local_dir,
        skip_preflight=args.skip_preflight,
        cache_root=args.cache_root,
        no_cache=args.no_cache,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )
    if out_dir is None:
        print(
            f"\nStopped after stage {args.to_stage}; cache dir is the result "
            f"(under {args.cache_root})."
        )
    else:
        print(f"\nWrote snapshot to: {out_dir}")
    return 0


def _datafix_diagnostics_path(args: argparse.Namespace, fix_id: str):
    from pathlib import Path

    from eval_card_backend.config import REPO_ROOT

    if args.diagnostics:
        return Path(args.diagnostics)
    return REPO_ROOT / ".cache" / "datafix" / f"{fix_id}_diagnostics.json"


def _publish_table(table_name: str, table, changes, accept_changes: bool) -> int:
    """Write a rebuilt table, unless it would change a vendored one.

    A vendored table is a reviewed artifact: the numbers in it are what a
    bake fills, so a rebuild that produces something else has to be looked
    at before it lands. Identical content is a no-op rather than a rewrite.
    """
    from eval_card_backend.datafix import builders

    old = builders.vendored_table(table_name)
    if old is not None and not builders.table_differs(old, table):
        print(f"unchanged: {builders.DATA_DIR / table_name}")
        return 0
    if old is not None and not accept_changes:
        print(
            f"refusing to change the vendored {table_name}; "
            f"the rebuild differs from what is shipped:",
            file=sys.stderr,
        )
        for line in changes(old, table) or ["  content differs"]:
            print(line, file=sys.stderr)
        print(
            "Re-run with --accept-changes to write it, after reading the "
            "differences above.",
            file=sys.stderr,
        )
        return 1
    path = builders.write_table(table_name, table)
    print(f"wrote {path}")
    return 0


def _cmd_datafix_build_vals(args: argparse.Namespace) -> int:
    from collections import Counter
    from pathlib import Path

    from eval_card_backend.datafix import builders, vals_ai
    from eval_card_backend.datafix.core import atomic_write_text, canonical_json

    table, diagnostics = builders.build_vals_table(
        Path(args.eee_root), args.eee_revision
    )
    diag_path = _datafix_diagnostics_path(args, vals_ai.FIX_ID)
    atomic_write_text(diag_path, canonical_json(diagnostics))

    status = _publish_table(
        vals_ai.TABLE_NAME, table, builders.vals_table_changes, args.accept_changes
    )
    if status != 0:
        return status

    print(f"  patches: {len(table['patches'])}")
    for field, count in sorted(Counter(p["field"] for p in table["patches"]).items()):
        print(f"  {field}: {count}")
    for tier, count in sorted(Counter(p["tier"] for p in table["patches"]).items()):
        print(f"  {tier}: {count}")
    print(f"  uncorroborated gaps: {len(diagnostics['uncorroborated'])}")
    print(f"diagnostics: {diag_path}")
    return 0


def _cmd_datafix_build_hfopenllm(args: argparse.Namespace) -> int:
    import re
    from pathlib import Path

    from eval_card_backend.datafix import builders, hfopenllm_v2
    from eval_card_backend.datafix.core import atomic_write_text, canonical_json

    results_dir = Path(args.results_dir)
    revision = args.revision or results_dir.name
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        print(
            "--revision is required: the results directory name "
            f"({results_dir.name!r}) is not a commit sha.",
            file=sys.stderr,
        )
        return 2

    table, diagnostics = builders.build_hfopenllm_table(results_dir, revision)
    diag_path = _datafix_diagnostics_path(args, hfopenllm_v2.FIX_ID)
    atomic_write_text(diag_path, canonical_json(diagnostics))

    status = _publish_table(
        hfopenllm_v2.TABLE_NAME, table, builders.hfopenllm_table_changes,
        args.accept_changes,
    )
    if status != 0:
        return status

    print(f"  files read: {diagnostics['files_read']}, "
          f"rejected: {len(diagnostics['files_rejected'])}")
    for benchmark, models in table["verified"].items():
        print(f"  {benchmark}: {len(models)} verified models, "
              f"default {json.dumps(table['defaults'].get(benchmark))}")
    print(f"  overrides: {len(table['overrides'])}, "
          f"conflicts: {len(table['conflicts'])}")
    print(f"diagnostics: {diag_path}")
    return 0


def _cmd_datafix_build_arc_agi(args: argparse.Namespace) -> int:
    from collections import Counter
    from pathlib import Path

    from eval_card_backend.datafix import arc_agi, builders
    from eval_card_backend.datafix.core import atomic_write_text, canonical_json

    histories = {
        name: builders.read_run_history(
            Path(args.history_root) / source["directory"], source["revision"]
        )
        for name, source in builders.ARC_PUBLIC_RUNS.items()
    }
    table, diagnostics = builders.build_arc_table(
        Path(args.eee_root), args.eee_revision, Path(args.runs_root), histories
    )
    diag_path = _datafix_diagnostics_path(args, arc_agi.FIX_ID)
    atomic_write_text(diag_path, canonical_json(diagnostics))

    status = _publish_table(
        arc_agi.TABLE_NAME, table, builders.arc_table_changes, args.accept_changes
    )
    if status != 0:
        return status

    print(f"  runs read: {diagnostics['runs_read']}, "
          f"patches: {len(table['patches'])}")
    by_name = Counter(p["evaluation_name"] for p in table["patches"])
    for name, count in sorted(by_name.items()):
        print(f"  {name}: {count}")
    print(f"  with a temperature: "
          f"{sum(1 for p in table['patches'] if 'temperature' in p['fields'])}")
    for reason, count in sorted(
        Counter(r["reason"] for r in diagnostics["rejected_runs"]).items()
    ):
        print(f"  runs rejected, {reason}: {count}")
    for reason, count in sorted(
        Counter(r["reason"] for r in diagnostics["rejected_results"]).items()
    ):
        print(f"  results rejected, {reason}: {count}")
    print(f"diagnostics: {diag_path}")
    return 0


def _cmd_datafix_export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from eval_card_backend.datafix import builders

    configs = None
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    try:
        counts = builders.export_patched_records(
            Path(args.eee_root), Path(args.out), configs
        )
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"exported to {args.out}")
    for key in ("records_scanned", "records_patched", "results_patched", "fields_patched"):
        print(f"  {key}: {counts.get(key, 0)}")
    return 0


def _add_datafix_parser(subs) -> None:
    datafix = subs.add_parser(
        "datafix",
        help="Build and export the EEE datafix overlay's patch tables.",
    )
    ops = datafix.add_subparsers(dest="datafix_command", required=True)

    vals = ops.add_parser(
        "build-vals-table",
        help="Rebuild the Vals.ai output-cap patch table from an EEE snapshot.",
    )
    vals.add_argument("--eee-root", required=True, help="Local EEE snapshot directory.")
    vals.add_argument(
        "--eee-revision", required=True,
        help="Dataset commit sha the snapshot was taken at; recorded in the table.",
    )
    vals.add_argument(
        "--diagnostics",
        help="Where to write the builder's triage output (default: "
             ".cache/datafix/<fix_id>_diagnostics.json). Not vendored.",
    )
    vals.add_argument(
        "--accept-changes", action="store_true",
        help="Write the rebuilt table even though it differs from the "
             "vendored one. Without it the build prints the differences and "
             "exits non-zero, leaving the shipped table alone.",
    )
    vals.set_defaults(func=_cmd_datafix_build_vals)

    hfo = ops.add_parser(
        "build-hfopenllm-table",
        help="Rebuild the Open LLM v2 harness-config table from the published dumps.",
    )
    hfo.add_argument(
        "--results-dir", required=True,
        help="Local snapshot of open-llm-leaderboard/results.",
    )
    hfo.add_argument(
        "--revision",
        help="Dataset commit sha; defaults to the snapshot directory's name.",
    )
    hfo.add_argument(
        "--diagnostics",
        help="Where to write the builder's triage output (default: "
             ".cache/datafix/<fix_id>_diagnostics.json). Not vendored.",
    )
    hfo.add_argument(
        "--accept-changes", action="store_true",
        help="Write the rebuilt table even though it differs from the "
             "vendored one. Without it the build prints the differences and "
             "exits non-zero, leaving the shipped table alone.",
    )
    hfo.set_defaults(func=_cmd_datafix_build_hfopenllm)

    arc = ops.add_parser(
        "build-arc-agi-table",
        help="Rebuild the ARC Prize table from the published run outputs.",
    )
    arc.add_argument("--eee-root", required=True, help="Local EEE snapshot directory.")
    arc.add_argument(
        "--eee-revision", required=True,
        help="Dataset commit sha the snapshot was taken at; recorded in the table.",
    )
    arc.add_argument(
        "--runs-root", required=True,
        help="Directory holding one bulk snapshot per public set, in "
             "subdirectories named as the pinned sources are.",
    )
    arc.add_argument(
        "--history-root", required=True,
        help="Directory holding a clone of each published dataset, in the "
             "same subdirectories; read once for the commits that touched "
             "each run, so a rewritten run can be refused.",
    )
    arc.add_argument(
        "--diagnostics",
        help="Where to write the builder's triage output (default: "
             ".cache/datafix/<fix_id>_diagnostics.json). Not vendored.",
    )
    arc.add_argument(
        "--accept-changes", action="store_true",
        help="Write the rebuilt table even though it differs from the "
             "vendored one. Without it the build prints the differences and "
             "exits non-zero, leaving the shipped table alone.",
    )
    arc.set_defaults(func=_cmd_datafix_build_arc_agi)

    export = ops.add_parser(
        "export",
        help="Write patched copies of the affected records plus a patch manifest.",
    )
    export.add_argument("--eee-root", required=True, help="Local EEE snapshot directory.")
    export.add_argument(
        "--out", required=True,
        help="Output directory; must be empty or not exist yet.",
    )
    export.add_argument(
        "--configs",
        help="Comma-separated configs to export; default: every config a fix applies to.",
    )
    export.set_defaults(func=_cmd_datafix_export)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-card-backend")
    subs = parser.add_subparsers(dest="command")

    # Default behaviour: summary (preserves existing CLI shape).
    parser.add_argument(
        "--configs",
        help="Comma-separated EEE config names to load. Default: all discovered.",
    )
    parser.add_argument(
        "--config-limit",
        type=int,
        help="Stop after N configs (smoke-test convenience).",
    )

    canon = subs.add_parser(
        "canonicalise", help="Run the canonicalisation pipeline end-to-end."
    )
    canon.add_argument(
        "--configs",
        help="Comma-separated EEE config names to load.",
    )
    canon.add_argument(
        "--config-limit", type=int,
        help="Stop after N configs (smoke-test convenience).",
    )
    canon.add_argument(
        "--snapshot-id",
        help="ISO timestamp; default: now (UTC).",
    )
    canon.add_argument(
        "--warehouse", default="warehouse",
        help="Output warehouse root directory.",
    )
    canon.add_argument(
        "--registry-local-dir", default=None,
        help="Local cache dir for the evaleval/entity-registry-data dataset "
        "(default: $ENTITY_REGISTRY_LOCAL_DIR, else .cache/entity_registry). "
        "A non-None argparse default here would silently override the env "
        "var — the pipeline only falls back to Settings when this is None.",
    )
    canon.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip preflight checks (use only for diagnostic runs).",
    )
    canon.add_argument(
        "--cache-root", default=".cache/canonicalise",
        help="Directory holding per-snapshot stage caches. Default: .cache/canonicalise",
    )
    canon.add_argument(
        "--no-cache", action="store_true",
        help="Skip writing stage caches (cache reads still work for --from-stage).",
    )
    canon.add_argument(
        "--from-stage",
        help="Resume from this stage letter (A,B,C,D,E,F,G,I,J); restores cached "
             "outputs for earlier stages. If --snapshot-id is unset, uses the "
             "latest snapshot under --cache-root. Stage J builds the view layer "
             "(eval_results_view, models_view, evals_view) + writes the JSON "
             "sidecars (manifest, headline, hierarchy).",
    )
    canon.add_argument(
        "--to-stage",
        help="Stop after this stage letter. Stops before warehouse emit if set "
             "earlier than I; cache dir is the result. Set to J (default) to "
             "produce the full view-layer warehouse.",
    )

    _add_datafix_parser(subs)

    args = parser.parse_args(argv)

    if args.command == "canonicalise":
        return _cmd_canonicalise(args)
    if args.command == "datafix":
        return args.func(args)
    return _cmd_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())

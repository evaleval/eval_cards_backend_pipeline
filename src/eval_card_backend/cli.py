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

    eee_root: Path | None = eee_local if (eee_local / "data").exists() else None
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
        "--registry-local-dir", default=".cache/entity_registry",
        help="Local cache dir for the evaleval/entity-registry-data dataset.",
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

    args = parser.parse_args(argv)

    if args.command == "canonicalise":
        return _cmd_canonicalise(args)
    return _cmd_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())

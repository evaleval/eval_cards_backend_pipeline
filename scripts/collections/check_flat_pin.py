#!/usr/bin/env python3
"""Pre-flight: is this EEE revision safe to extract a collection at?

The collection extractors read the datastore through `ensure_snapshot`, which
builds its file listing from the repo's `flat/` index rather than walking
`data/`. `flat/` is rebuilt by a cron in the datastore repo, so a commit that
merged records only minutes ago can resolve to a snapshot that does not contain
them. The extractor warns about this and carries on, which is how a run can
spend an hour re-deriving the corpus it already had and report success.

This check turns that warning into a refusal, before anything is downloaded.

    python scripts/collections/check_flat_pin.py --revision <sha> \
        --require-collection aisi-cyber-ctfs --require-collection aisi-the-last-ones

The decisive test is whether the flat index at this pin actually CONTAINS the
collections being extracted — a flat-rebuild commit always post-dates its own
descriptor by however long the rebuild took, so "created_at < commit date" on
its own rejects the very commits you are supposed to pin. Lag is reported as
context and only fails the check when it exceeds --max-lag-hours.

Exit 0 when every required collection is present and the lag is within
tolerance. Exit 1 otherwise, naming the remedy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

DEFAULT_REPO = "evaleval/EEE_datastore"
FLAT_MANIFEST = "flat/latest_manifest.json"
BY_COLLECTION = "flat/indexes/by_collection"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument(
        "--require-collection", action="append", default=[], metavar="NAME",
        help="collection that must already be in the flat index at this pin",
    )
    ap.add_argument(
        "--max-lag-hours", type=float, default=2.0,
        help="fail when the flat index pre-dates the pin by more than this. "
             "A flat-rebuild commit self-lags by its own rebuild duration "
             "(observed 8-67 min), so the default tolerates that and still "
             "catches a pin taken hours after the last rebuild.",
    )
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    import os

    from huggingface_hub import HfApi, hf_hub_download

    token = args.hf_token or os.environ.get("HF_TOKEN")
    api = HfApi()

    try:
        info = api.dataset_info(args.repo, revision=args.revision, token=token)
    except Exception as exc:
        print(f"FAIL: cannot resolve {args.repo}@{args.revision}: "
              f"{type(exc).__name__}: {exc}")
        return 1

    committed = getattr(info, "last_modified", None)
    if committed is not None and committed.tzinfo is None:
        committed = committed.replace(tzinfo=timezone.utc)

    try:
        path = hf_hub_download(
            repo_id=args.repo, repo_type="dataset", revision=args.revision,
            filename=FLAT_MANIFEST, token=token,
        )
    except Exception as exc:
        print(f"FAIL: no {FLAT_MANIFEST} at {args.revision}: "
              f"{type(exc).__name__}: {exc}")
        return 1

    with open(path, encoding="utf-8") as handle:
        descriptor = json.load(handle)
    created = _parse_iso(descriptor.get("created_at"))

    print(f"repo            : {args.repo}")
    print(f"revision        : {args.revision}")
    print(f"commit dated    : {committed.isoformat() if committed else 'unknown'}")
    print(f"flat created_at : {descriptor.get('created_at')}")
    print(f"flat records    : {descriptor.get('aggregate_file_count')}")

    failures: list[str] = []

    if created is None or committed is None:
        failures.append(
            "cannot compare flat index age with the commit date "
            "(missing created_at or commit timestamp)"
        )
    else:
        hours = max((committed - created).total_seconds() / 3600, 0.0)
        print(f"flat lag        : {hours:.1f}h (tolerance {args.max_lag_hours}h)")
        if hours > args.max_lag_hours:
            failures.append(
                f"flat index lags the pin by {hours:.1f}h — records merged in "
                f"that window are NOT in the snapshot this extraction would "
                f"read. Pin a flat-rebuild commit instead (the datastore's "
                f"'cron: flat rebuild' commits), or trigger one: gh workflow "
                f"run flat-rebuild.yml --repo evaleval/every_eval_ever"
            )

    if args.require_collection:
        try:
            present = {
                entry.path.rsplit("/", 1)[-1].removesuffix(".jsonl")
                for entry in api.list_repo_tree(
                    args.repo, repo_type="dataset", revision=args.revision,
                    path_in_repo=BY_COLLECTION, token=token,
                )
            }
        except Exception as exc:
            failures.append(f"cannot list {BY_COLLECTION}: {exc}")
            present = set()
        for name in args.require_collection:
            mark = "ok" if name in present else "MISSING"
            print(f"required collection {name}: {mark}")
            if name not in present:
                failures.append(
                    f"collection {name!r} is not in the flat index at this pin"
                )

    if failures:
        print("\nPIN REJECTED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPin OK: required collections are in the flat index at this pin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

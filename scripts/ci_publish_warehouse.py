"""CI helper: upload the latest warehouse snapshot to a HF dataset.

Reads `HF_TARGET_DATASET` (e.g. `evaleval/card_backend`) and
`HF_TOKEN` from env. Picks the most-recent snapshot under `warehouse/`
and uploads it twice:

  - `warehouse/<snapshot_id>/` — immutable historical pin; consumers that
    want reproducibility set `SNAPSHOT_URL=.../warehouse/<id>`.
  - `warehouse/latest/` — mirror of the same snapshot, refreshed every
    run, so the frontend can fetch from a stable URL without knowing the
    snapshot ID. `delete_patterns="*"` makes the latest/ contents
    replace rather than accumulate across runs.

Idempotent: re-running over an existing snapshot path no-ops the
timestamped upload; the latest/ upload always rewrites.

Shrink guard: before the first upload the new `snapshot_meta.json` is
compared with the published `warehouse/latest/snapshot_meta.json` on three
measures (`row_counts.eee_records`, `row_counts.fact_results`, number of
`configs`). If any new value is below 90% of the published one, or the
published file is missing or unreadable, nothing is uploaded and the script
exits non-zero. `ALLOW_SNAPSHOT_SHRINK=1` publishes anyway, and that
snapshot becomes the baseline for the next run. The guard stops a grossly
smaller snapshot replacing the published one; it does not inspect records.

`--check-only` runs the guard, prints the comparison and uploads nothing.
`HF_TOKEN` is optional in that mode when the target dataset is public.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

PUBLISHED_META_PATH = "warehouse/latest/snapshot_meta.json"
SHRINK_THRESHOLD = 0.9

# HF throttles the LFS preupload endpoint under bursty load (merge + Space
# rebuild + other syncs), returning 429. huggingface_hub's built-in backoff
# gives up after ~5 tries in ~25s, which a multi-minute throttle outlasts.
# Retry the whole upload_folder on 429 with a longer backoff; upload_folder is
# idempotent (already-uploaded files no-op on preupload) so re-calling is safe.
_RETRY_BACKOFF_SECONDS = (30, 60, 120, 240, 300)


def _upload_with_retry(api: HfApi, **kwargs) -> None:
    for attempt, delay in enumerate((*_RETRY_BACKOFF_SECONDS, None)):
        try:
            api.upload_folder(**kwargs)
            return
        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status != 429 or delay is None:
                raise
            print(
                f"HF 429 on {kwargs.get('path_in_repo')} "
                f"(attempt {attempt + 1}); retrying in {delay}s.",
                file=sys.stderr,
            )
            time.sleep(delay)


def _measures(meta: dict) -> dict[str, int | None]:
    row_counts = meta.get("row_counts")
    if not isinstance(row_counts, dict):
        row_counts = {}
    configs = meta.get("configs")
    return {
        "eee_records": row_counts.get("eee_records"),
        "fact_results": row_counts.get("fact_results"),
        "configs": len(configs) if isinstance(configs, list) else None,
    }


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def compare_snapshots(new_meta: dict, published_meta: dict) -> tuple[bool, str]:
    """Return (ok, report). The report names all three new/published pairs."""
    new = _measures(new_meta)
    published = _measures(published_meta)
    ok = True
    lines = []
    for name, pub in published.items():
        cur = new[name]
        if not _is_count(pub):
            verdict = "FAIL (published value unreadable)"
            ok = False
        elif not _is_count(cur):
            verdict = "FAIL (new value missing)"
            ok = False
        elif cur < SHRINK_THRESHOLD * pub:
            verdict = f"FAIL (below {SHRINK_THRESHOLD:.0%} of published)"
            ok = False
        else:
            verdict = "ok"
        lines.append(f"  {name}: new={cur} published={pub} {verdict}")
    return ok, "\n".join(lines)


def _fetch_published_meta(api: HfApi, target: str) -> dict:
    path = api.hf_hub_download(
        repo_id=target,
        filename=PUBLISHED_META_PATH,
        repo_type="dataset",
    )
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("published snapshot_meta.json is not a JSON object")
    return meta


def shrink_guard(api: HfApi, target: str, latest: Path, *, allow_shrink: bool) -> bool:
    """Compare `latest` with the published snapshot; True means publish may go ahead."""
    problem = None
    try:
        new_meta = json.loads((latest / "snapshot_meta.json").read_text(encoding="utf-8"))
        if not isinstance(new_meta, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        new_meta = None
        problem = f"cannot read {latest / 'snapshot_meta.json'}: {exc}"

    if new_meta is not None:
        try:
            published_meta = _fetch_published_meta(api, target)
        except EntryNotFoundError:
            problem = f"hf://{target}/{PUBLISHED_META_PATH} does not exist"
        except Exception as exc:  # noqa: BLE001 - any download or parse error blocks
            problem = (
                f"cannot read hf://{target}/{PUBLISHED_META_PATH}: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            ok, report = compare_snapshots(new_meta, published_meta)
            report = (
                f"{latest.name} against published "
                f"{published_meta.get('snapshot_id')}\n{report}"
            )
            if ok:
                print(f"Shrink guard passed: {report}")
                return True
            problem = report

    if allow_shrink:
        print(f"Shrink guard overridden (ALLOW_SNAPSHOT_SHRINK=1): {problem}")
        return True
    print(
        f"Shrink guard failed: {problem}\n"
        "Nothing was uploaded. Set ALLOW_SNAPSHOT_SHRINK=1 to publish anyway.",
        file=sys.stderr,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the shrink guard, print the comparison and upload nothing",
    )
    args = parser.parse_args(argv)

    target = os.environ.get("HF_TARGET_DATASET")
    token = os.environ.get("HF_TOKEN") or None
    allow_shrink = os.environ.get("ALLOW_SNAPSHOT_SHRINK") == "1"
    if not target:
        print("HF_TARGET_DATASET unset; refusing to upload.", file=sys.stderr)
        return 1
    if not token and not args.check_only:
        print("HF_TOKEN unset; refusing to upload.", file=sys.stderr)
        return 1

    warehouse = Path("warehouse")
    if not warehouse.exists():
        print("No warehouse/ dir on disk; nothing to publish.", file=sys.stderr)
        return 1

    snapshots = sorted(d for d in warehouse.iterdir() if d.is_dir())
    if not snapshots:
        print("warehouse/ has no snapshot subdirectories.", file=sys.stderr)
        return 1
    latest = snapshots[-1]

    api = HfApi(token=token)
    if not shrink_guard(api, target, latest, allow_shrink=allow_shrink):
        return 1
    if args.check_only:
        print(f"--check-only: nothing uploaded for {latest.name}.")
        return 0

    _upload_with_retry(
        api,
        folder_path=str(latest),
        path_in_repo=f"warehouse/{latest.name}",
        repo_id=target,
        repo_type="dataset",
        commit_message=f"snapshot {latest.name}",
    )
    print(f"Uploaded {latest.name} → hf://{target}/warehouse/{latest.name}")

    _upload_with_retry(
        api,
        folder_path=str(latest),
        path_in_repo="warehouse/latest",
        repo_id=target,
        repo_type="dataset",
        commit_message=f"refresh latest → {latest.name}",
        delete_patterns="*",
    )
    print(f"Refreshed hf://{target}/warehouse/latest → {latest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

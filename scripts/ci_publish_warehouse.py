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
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

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


def main() -> int:
    target = os.environ.get("HF_TARGET_DATASET")
    token = os.environ.get("HF_TOKEN")
    if not target:
        print("HF_TARGET_DATASET unset; refusing to upload.", file=sys.stderr)
        return 1
    if not token:
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

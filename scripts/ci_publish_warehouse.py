"""CI helper: upload the latest warehouse snapshot to a HF dataset.

Reads `HF_TARGET_DATASET` (e.g. `j-chim/temp_evalcard_backend`) and
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
from pathlib import Path

from huggingface_hub import HfApi


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
    api.upload_folder(
        folder_path=str(latest),
        path_in_repo=f"warehouse/{latest.name}",
        repo_id=target,
        repo_type="dataset",
        commit_message=f"snapshot {latest.name}",
    )
    print(f"Uploaded {latest.name} → hf://{target}/warehouse/{latest.name}")

    api.upload_folder(
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

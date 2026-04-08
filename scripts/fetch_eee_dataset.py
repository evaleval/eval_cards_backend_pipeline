import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_REPO = "evaleval/EEE_datastore"
DEFAULT_TARGET = ".cache/eee_datastore"


def main() -> int:
    target_dir = Path(os.environ.get("EEE_LOCAL_DATASET_DIR", DEFAULT_TARGET)).resolve()
    token = os.environ.get("HF_TOKEN")

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset snapshot to {target_dir}")

    try:
        local_path = snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=["data/**"],
            token=token,
        )
    except Exception as error:
        print(f"Failed to download dataset snapshot: {error}", file=sys.stderr)
        return 1

    print(f"Dataset ready at {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_REPO = "evaleval/EEE_datastore"
DEFAULT_TARGET = ".cache/eee_datastore"


def main() -> int:
    target_dir = Path(os.environ.get("EEE_LOCAL_DATASET_DIR", DEFAULT_TARGET)).resolve()
    token = os.environ.get("HF_TOKEN")

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing dataset mirror at {target_dir}")
    repo_url = f"https://huggingface.co/datasets/{DATASET_REPO}"

    try:
        if (target_dir / ".git").exists():
            subprocess.run(["git", "-C", str(target_dir), "fetch", "origin", "main"], check=True)
            subprocess.run(["git", "-C", str(target_dir), "checkout", "main"], check=True)
            subprocess.run(["git", "-C", str(target_dir), "pull", "--ff-only", "origin", "main"], check=True)
            subprocess.run(["git", "-C", str(target_dir), "sparse-checkout", "set", "data"], check=True)
            local_path = str(target_dir)
        else:
            if any(target_dir.iterdir()):
                shutil.rmtree(target_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    repo_url,
                    str(target_dir),
                ],
                check=True,
            )
            subprocess.run(["git", "-C", str(target_dir), "sparse-checkout", "set", "data"], check=True)
            local_path = str(target_dir)
    except Exception as error:
        print(f"Git mirror failed, falling back to snapshot_download: {error}", file=sys.stderr)
        try:
            local_path = snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                allow_patterns=["data/**"],
                token=token,
                resume_download=True,
            )
        except Exception as fallback_error:
            print(f"Failed to download dataset snapshot: {fallback_error}", file=sys.stderr)
            return 1

    print(f"Dataset ready at {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

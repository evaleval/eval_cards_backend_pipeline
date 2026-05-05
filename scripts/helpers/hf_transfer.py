import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, snapshot_download


def has_cached_benchmark_metadata(cards_dir: Path, flat_metadata_path: Path) -> bool:
    return flat_metadata_path.exists() or (
        cards_dir.exists() and any(cards_dir.glob("*.json"))
    )


def ensure_local_benchmark_metadata_snapshot(
    local_metadata_dir: str,
    hf_token: str | None,
    force_refresh: bool,
    benchmark_metadata_dataset_repo: str,
) -> str | None:
    target_dir = Path(local_metadata_dir).resolve()
    cards_dir = target_dir / "cards"
    flat_metadata_path = target_dir / "benchmark-metadata.json"

    if force_refresh and target_dir.exists():
        shutil.rmtree(target_dir)

    if has_cached_benchmark_metadata(cards_dir, flat_metadata_path):
        return str(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=benchmark_metadata_dataset_repo,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=["benchmark-metadata.json", "cards/**"],
            token=hf_token,
        )
    except Exception:
        if has_cached_benchmark_metadata(cards_dir, flat_metadata_path):
            return str(target_dir)
        return None

    return str(target_dir)


def ensure_local_dataset_snapshot(
    local_dataset_dir: str,
    hf_token: str | None,
    force_refresh: bool,
    eee_dataset_repo: str,
) -> str:
    target_dir = Path(local_dataset_dir).resolve()
    data_dir = target_dir / "data"
    target_dir.mkdir(parents=True, exist_ok=True)

    if force_refresh and target_dir.exists():
        shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        data_dir = target_dir / "data"

    if data_dir.exists() and any(data_dir.iterdir()):
        return str(target_dir)

    snapshot_download(
        repo_id=eee_dataset_repo,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=["data/**"],
        token=hf_token,
    )

    return str(target_dir)


def discover_configs(
    local_dataset_dir: str | None, hf_token: str | None, eee_dataset_repo: str
) -> list[str]:
    if local_dataset_dir:
        data_root = Path(local_dataset_dir) / "data"
        configs = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
        return configs

    fs = HfFileSystem(token=hf_token)
    entries = fs.ls(f"datasets/{eee_dataset_repo}/data", detail=True)
    configs = []
    for entry in entries:
        name = entry.get("name", "")
        config = name.split("/")[-1]
        if config:
            configs.append(config)
    return sorted(set(configs))


def list_json_files_for_config(
    config: str,
    local_dataset_dir: str | None,
    hf_token: str | None,
    eee_dataset_repo: str,
) -> list[str]:
    if local_dataset_dir:
        root = Path(local_dataset_dir) / "data" / config
        return sorted(
            str(p.relative_to(local_dataset_dir)).replace(os.sep, "/")
            for p in root.rglob("*.json")
            if p.is_file() and not p.name.endswith(".jsonl")
        )

    fs = HfFileSystem(token=hf_token)
    pattern = f"datasets/{eee_dataset_repo}/data/{config}/**/*.json"
    paths = [p for p in fs.glob(pattern) if not p.endswith(".jsonl")]
    prefix = f"datasets/{eee_dataset_repo}/"
    return sorted(p[len(prefix) :] for p in paths)


def read_dataset_json(
    dataset_path: str,
    local_dataset_dir: str | None,
    hf_token: str | None,
    eee_dataset_repo: str,
) -> dict:
    if local_dataset_dir:
        local_path = Path(local_dataset_dir) / dataset_path
        return json.loads(local_path.read_text(encoding="utf-8"))

    local_path = hf_hub_download(
        repo_id=eee_dataset_repo,
        filename=dataset_path,
        repo_type="dataset",
        token=hf_token,
    )
    return json.loads(Path(local_path).read_text(encoding="utf-8"))


def resolve_upload_target(production_dataset_repo: str) -> str:
    target = (os.environ.get("CARD_BACKEND_OUTPUT_REPO") or "").strip()
    allow_production = (
        os.environ.get("CARD_BACKEND_ALLOW_PRODUCTION") == "1"
        or os.environ.get("GITHUB_ACTIONS") == "true"
    )
    if not target:
        if not allow_production:
            raise RuntimeError(
                "CARD_BACKEND_OUTPUT_REPO is required for local uploads. "
                "Set it to a non-production dataset (e.g. `j-chim/temp_evalcard_backend`); "
                "intentional local prod uploads also need CARD_BACKEND_ALLOW_PRODUCTION=1."
            )
        return production_dataset_repo
    if target == production_dataset_repo and not allow_production:
        raise RuntimeError(
            f"Refusing to upload to production target {production_dataset_repo}. "
            "Set CARD_BACKEND_ALLOW_PRODUCTION=1 to override."
        )
    return target


def _iter_output_relative_files(root_dir: Path) -> list[str]:
    files = []
    if not root_dir.exists():
        return files
    for path in sorted(root_dir.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(root_dir)).replace(os.sep, "/"))
    return files


def delete_stale_remote_files(
    api: HfApi,
    token: str,
    output_dir: Path,
    repo_id: str,
) -> None:
    local_files = set(_iter_output_relative_files(output_dir))
    remote_files = set(api.list_repo_files(repo_id, repo_type="dataset", token=token))
    stale_files = sorted(remote_files - local_files)
    if not stale_files:
        return

    chunk_size = 200
    for index in range(0, len(stale_files), chunk_size):
        chunk = stale_files[index : index + chunk_size]
        api.delete_files(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            delete_patterns=chunk,
            commit_message=f"Remove stale pipeline artifacts ({index + 1}-{index + len(chunk)})",
        )


def upload_output(output_dir: Path, production_dataset_repo: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required unless --dry-run is used")

    upload_target = resolve_upload_target(production_dataset_repo)
    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=upload_target, repo_type="dataset", private=False, exist_ok=True
        )
    except Exception as error:
        print(f"create_repo warning: {error}", file=sys.stderr)

    delete_stale_remote_files(api, token, output_dir, repo_id=upload_target)

    api.upload_large_folder(
        repo_id=upload_target,
        repo_type="dataset",
        folder_path=str(output_dir),
    )

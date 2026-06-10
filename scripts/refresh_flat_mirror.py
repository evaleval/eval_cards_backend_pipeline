"""Rebuild the EEE flat view from upstream data/ and publish it to a mirror.

Stopgap while `evaleval/EEE_datastore` does not regenerate flat/ on data/
merges (its flat/ is a one-shot 2026-06-04 build; submissions land only in
data/). On each run:

1. Skip when upstream HEAD matches the mirror's recorded source revision
   (`source_revision.json`), unless --force.
2. Shallow-clone upstream with LFS smudge skipped, then prune every
   `*_samples.jsonl` (the producer never reads instance-level files, and the
   LFS-skipped copies would be pointer stubs anyway; pruning before the build
   keeps the lean manifest self-consistent — entries simply carry no
   instance fields).
3. Run the upstream builder + validator (`tools/` inside the datastore repo).
4. Upload to the mirror: objects/manifest/indexes first, then
   `latest_manifest.json` + `source_revision.json` as a final atomic commit,
   so a reader resolving the moving pointer never sees missing objects.

Point the producer at the mirror via `EEE_DATASET_REPO` (+ `EEE_REVISION`
pin). Retire this script once upstream rebuilds flat/ on every merge.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SOURCE_REPO = "evaleval/EEE_datastore"
MIRROR_REPO = "evaleval/EEE_datastore-flat-temp"
MARKER = "source_revision.json"

MIRROR_README = """\
---
viewer: false
---
# EEE_datastore flat-view mirror (temporary)

Machine-generated mirror of the `flat/` view of
[evaleval/EEE_datastore](https://huggingface.co/datasets/evaleval/EEE_datastore),
rebuilt from its `data/` tree because the upstream flat view is not yet
regenerated on every merge. Instance-level `*_samples.jsonl` companions are
not mirrored (manifest entries carry no instance fields here).

Built by `scripts/refresh_flat_mirror.py` in
[evaleval/eval_cards_backend_pipeline](https://github.com/evaleval/eval_cards_backend_pipeline).
`source_revision.json` records the upstream commit each build came from.
This repo will be retired once upstream maintains `flat/` itself.
"""


def upstream_head(api: HfApi, repo: str) -> str:
    return api.dataset_info(repo).sha


def mirror_source_revision(api: HfApi, repo: str) -> str | None:
    try:
        path = hf_hub_download(
            repo_id=repo, filename=MARKER, repo_type="dataset",
            token=api.token, force_download=True,
        )
    except Exception:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8")).get("source_revision")


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def clone_and_build(source_repo: str, work: Path, head: str) -> Path:
    src = work / "src"
    if src.exists():
        shutil.rmtree(src)
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    run(
        ["git", "clone", "--depth", "1",
         f"https://huggingface.co/datasets/{source_repo}", str(src)],
        env=env,
    )
    got = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True,
        check=True,
    ).stdout.strip()
    if got != head:
        print(f"note: upstream moved during clone ({head} -> {got}); building {got}")

    pruned = 0
    for p in list(src.rglob("*_samples.jsonl")):
        p.unlink()
        pruned += 1
    print(f"pruned {pruned} instance-level sample file(s)")
    # Old manifest dirs are LFS pointer stubs in this clone and the validator
    # checks every manifest dir; the mirror only ships the new one anyway.
    shutil.rmtree(src / "flat" / "manifests", ignore_errors=True)

    run([sys.executable, "tools/build_flat_datastore.py", "--datastore", "."], cwd=src)
    run([sys.executable, "tools/validate_flat_datastore.py", "--datastore", "."], cwd=src)
    return src


def export_upload_set(src: Path, work: Path) -> tuple[Path, Path]:
    """Hardlink exactly the files the new manifest references into a clean
    dir; returns (bulk_dir, pointer_dir). The moving pointer + marker ship
    separately so they land last."""
    bulk = work / "export"
    pointer = work / "export_pointer"
    for d in (bulk, pointer):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    manifest = json.loads((src / "flat/latest_manifest.json").read_text(encoding="utf-8"))
    entries_path = manifest["entries_path"]

    def link(rel: str, dest_root: Path) -> None:
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(src / rel, dest)

    n = 0
    for line in (src / entries_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        link(json.loads(line)["object_path"], bulk)
        n += 1
    manifest_dir = Path(entries_path).parent
    for p in (src / manifest_dir).iterdir():
        link(str(manifest_dir / p.name), bulk)
    for p in (src / "flat/indexes").rglob("*"):
        if p.is_file():
            link(str(p.relative_to(src)), bulk)
    print(f"export: {n} objects + manifest dir + indexes -> {bulk}")

    link("flat/latest_manifest.json", pointer)
    return bulk, pointer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--mirror-repo", default=MIRROR_REPO)
    parser.add_argument("--work-dir", default=None,
                        help="Scratch dir (default: a fresh temp dir).")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even when the mirror is current.")
    args = parser.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    head = upstream_head(api, args.source_repo)
    current = mirror_source_revision(api, args.mirror_repo)
    if current == head and not args.force:
        print(f"mirror is current (source {head}); nothing to do")
        return 0
    print(f"drift: mirror has {current}, upstream is {head} — rebuilding")

    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="eee-flat-"))
    work.mkdir(parents=True, exist_ok=True)
    src = clone_and_build(args.source_repo, work, head)
    bulk, pointer = export_upload_set(src, work)

    api.create_repo(args.mirror_repo, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        repo_id=args.mirror_repo, repo_type="dataset", folder_path=str(bulk),
    )
    marker = {
        "source_repo": args.source_repo,
        "source_revision": head,
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    (pointer / MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    (pointer / "README.md").write_text(MIRROR_README, encoding="utf-8")
    api.upload_folder(
        repo_id=args.mirror_repo, repo_type="dataset", folder_path=str(pointer),
        commit_message=f"Flat rebuild from {args.source_repo}@{head[:12]}",
    )
    print(f"mirror updated: {args.mirror_repo} @ {api.dataset_info(args.mirror_repo).sha}")
    print(f"pin EEE_REVISION to that SHA (and EEE_DATASET_REPO={args.mirror_repo}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

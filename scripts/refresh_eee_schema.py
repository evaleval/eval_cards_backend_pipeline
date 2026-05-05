"""Refresh vendored EEE schema artifacts from upstream evaleval/every_eval_ever.

Pulls the JSON Schema and the upstream codegen-output Pydantic models at a
pinned (or HEAD) commit, byte-compares against our vendored copies, and
rewrites them if changed. Updates the pin file
(`registry/eee.schema.SOURCE.json`) to record what we're now on.

Usage:
    uv run python scripts/refresh_eee_schema.py            # pull main HEAD
    uv run python scripts/refresh_eee_schema.py --ref <sha>  # pull a specific commit
    uv run python scripts/refresh_eee_schema.py --check    # exit 1 if upstream differs (dry-run)

Why a script instead of a git submodule or a published package: every_eval_ever
isn't on PyPI, and the schema/codegen output are two small files. Pinning by
SHA in a JSON sidecar gives reproducible vendoring without submodule overhead.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "evaleval/every_eval_ever"

# (upstream path, local destination relative to repo root)
# The repo's root `eval.schema.json` is a symlink — pull from the canonical
# path under every_eval_ever/schemas/ so raw.githubusercontent gives us the
# file contents instead of the symlink target string.
FILES_TO_VENDOR: list[tuple[str, str]] = [
    ("every_eval_ever/schemas/eval.schema.json", "src/eval_card_backend/registry/eee.schema.json"),
    ("every_eval_ever/eval_types.py", "src/eval_card_backend/schemas/eee_types.py"),
]

PIN_PATH = "src/eval_card_backend/registry/eee.schema.SOURCE.json"


def _gh_api(path: str) -> dict:
    url = f"https://api.github.com/{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _fetch_raw(ref: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{ref}/{path}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def _resolve_ref(ref: str) -> str:
    """Resolve a ref (branch / tag / sha) to a full commit SHA."""
    data = _gh_api(f"repos/{UPSTREAM_REPO}/commits/{ref}")
    return data["sha"]


def _read_schema_version(schema_bytes: bytes) -> str | None:
    try:
        return json.loads(schema_bytes).get("version")
    except json.JSONDecodeError:
        return None


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", default="main",
        help="Upstream branch / tag / commit SHA to pull. Default: main.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Dry-run: report drift, exit 1 if any vendored file differs from upstream.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sha = _resolve_ref(args.ref)
    print(f"Resolved {args.ref!r} -> {sha}")

    drift: list[str] = []
    schema_version: str | None = None
    for upstream_path, local_path in FILES_TO_VENDOR:
        print(f"  fetching {upstream_path} ...")
        upstream_bytes = _fetch_raw(sha, upstream_path)
        local_file = repo_root / local_path
        local_bytes = local_file.read_bytes() if local_file.exists() else b""

        if upstream_path.endswith("eval.schema.json"):
            schema_version = _read_schema_version(upstream_bytes)

        if upstream_bytes == local_bytes:
            print(f"    unchanged ({local_path})")
            continue

        drift.append(local_path)
        if args.check:
            print(
                f"    DRIFT: {local_path} (local sha256={_sha256(local_bytes)[:12]} "
                f"upstream sha256={_sha256(upstream_bytes)[:12]})"
            )
            continue

        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_bytes(upstream_bytes)
        print(f"    wrote {local_path}")

    if args.check:
        if drift:
            print(f"\nDrift detected in {len(drift)} file(s). Run without --check to refresh.")
            return 1
        print("\nNo drift; vendored files match upstream.")
        return 0

    pin_file = repo_root / PIN_PATH
    pin = {
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit_sha": sha,
        "schema_version": schema_version,
        "fetched_at": dt.date.today().isoformat(),
        "files": {Path(local).name: upstream for upstream, local in FILES_TO_VENDOR},
        "refresh_with": "uv run python scripts/refresh_eee_schema.py",
    }
    pin_file.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    print(f"\nUpdated pin: {PIN_PATH}")

    if drift:
        print(f"\nRewrote {len(drift)} file(s). Re-run the audit to confirm records still cast cleanly:")
        print("  uv run python scripts/audit_eee_records.py")
    else:
        print("\nNo drift; only pin file refreshed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

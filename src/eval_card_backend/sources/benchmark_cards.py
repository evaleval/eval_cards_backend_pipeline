"""Snapshot and load benchmark cards from `evaleval/auto-benchmarkcards`.

Two on-disk shapes are supported:
  - `cards/<name>.json` per-card files (each wrapping a `benchmark_card` or
    `benchmark_details` payload).
  - A flat `benchmark-metadata.json` mapping name -> card.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import constants as hf_constants

from eval_card_backend.config import BENCHMARK_METADATA_DATASET_REPO
from eval_card_backend.sources._revision_cache import (
    cache_revision_ok as _cache_revision_ok,
    hand_built_snapshot as _hand_built_snapshot,
    write_fetched_commit as _write_fetched_commit,
    cached_revision,  # re-exported: the revision this cache actually holds
    write_cache_revision as _write_cache_revision,
)

__all__ = ["LoadedCard", "cached_revision", "ensure_snapshot", "load_cards"]


@dataclass(frozen=True)
class LoadedCard:
    payload: dict[str, Any]
    source_path: str | None = None


log = logging.getLogger(__name__)

_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def _has_cached(target: Path) -> bool:
    flat = target / "benchmark-metadata.json"
    cards_dir = target / "cards"
    return flat.exists() or (cards_dir.exists() and any(cards_dir.glob("*.json")))


def ensure_snapshot(
    local_dir: str,
    hf_token: str | None,
    force_refresh: bool,
    revision: str | None = None,
) -> Path | None:
    target = Path(local_dir).resolve()
    if _hand_built_snapshot(
        target, revision, "benchmark cards", _has_cached(target), log, force_refresh,
    ):
        return target
    if force_refresh and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    if _has_cached(target) and _cache_revision_ok(target, revision):
        return target
    if _has_cached(target):
        # Revision-mismatched cache — clear and re-download at the pin.
        shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    try:
        _fetch_and_publish(target, hf_token, revision)
    except Exception as exc:
        log.warning("benchmark cards fetch failed: %r", exc)
        return target if _has_cached(target) else None

    return target


def _repo_url() -> str:
    return f"{hf_constants.ENDPOINT}/datasets/{BENCHMARK_METADATA_DATASET_REPO}"


def _fetch_and_publish(target: Path, hf_token: str | None, revision: str | None) -> None:
    """Fetch into a staging dir beside `target`, records included, and swap it
    in with one rename. A partial tree with no marker would read as
    hand-built and never be refetched."""
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".cards-fetch-") as tmp:
        payload = Path(tmp) / "payload"
        payload.mkdir()
        commit = _fetch_cards(payload, hf_token, revision)
        _write_cache_revision(payload, revision or commit)
        _write_fetched_commit(payload, commit)
        shutil.rmtree(target, ignore_errors=True)
        os.replace(payload, target)


def _fetch_cards(dest: Path, hf_token: str | None, revision: str | None) -> str:
    """Check out the card files at `revision` into the empty `dest`; return
    the commit.

    One shallow git fetch pulls every card in a single pack, where
    `snapshot_download` would issue a request per `cards/*.json` file.
    """
    if hf_constants.HF_HUB_OFFLINE:
        raise OSError("HF_HUB_OFFLINE is set")
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    if hf_token:
        # Hub git takes the token as a Basic-auth password (it rejects Bearer),
        # passed via env, not argv, so it stays out of the process list.
        credential = base64.b64encode(f"hf:{hf_token}".encode()).decode()
        env |= {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraHeader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credential}",
        }
    with tempfile.TemporaryDirectory(dir=dest.parent) as repo:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", repo, *args],
                env=env, check=True, capture_output=True, text=True, timeout=300,
            ).stdout

        git("init", "-q")
        git("fetch", "-q", "--depth", "1", _repo_url(), revision or "HEAD")
        commit = git("rev-parse", "FETCH_HEAD").strip()
        paths = git(
            "ls-tree", "--name-only", "FETCH_HEAD", "--", "benchmark-metadata.json", "cards",
        ).split()
        if not paths:
            raise FileNotFoundError(f"no benchmark cards at {commit}")
        git("checkout", "-q", "FETCH_HEAD", "--", *paths)
        for rel in paths:
            os.replace(Path(repo) / rel, dest / rel)
    for path in [dest / "benchmark-metadata.json", *dest.glob("cards/*.json")]:
        if path.is_file() and path.read_bytes().startswith(_LFS_POINTER):
            raise ValueError(f"{path.relative_to(dest)} is a Git LFS pointer at {commit}")
    return commit


def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def load_cards(root: Path) -> dict[str, LoadedCard]:
    """Load payloads and dataset-relative card paths by normalized name.

    Per-card files take precedence over the flat file: upstream edits land in
    `cards/` first and the flat file is only rebuilt in batch updates.
    """
    cards: dict[str, LoadedCard] = {}
    if not root.exists():
        return cards

    flat_path = root / "benchmark-metadata.json"
    if flat_path.exists():
        parsed = json.loads(flat_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            for name, card in parsed.items():
                if isinstance(card, dict) and isinstance(card.get("benchmark_details"), dict):
                    key = _normalize_key(name)
                    if key:
                        cards[key] = LoadedCard(card)

    cards_dir = root / "cards"
    if cards_dir.exists():
        for path in sorted(cards_dir.glob("*.json")):
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                continue
            if isinstance(parsed.get("benchmark_card"), dict):
                payload = parsed["benchmark_card"]
                base = path.stem.replace("benchmark_card_", "")
            elif isinstance(parsed.get("benchmark_details"), dict):
                payload = parsed
                base = path.stem
            else:
                continue
            key = _normalize_key(base)
            if key:
                existing = cards.get(key)
                # Sorted filenames make the first matching path deterministic.
                if existing is None or existing.source_path is None:
                    cards[key] = LoadedCard(payload, path.relative_to(root).as_posix())

    return cards

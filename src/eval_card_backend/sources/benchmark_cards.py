"""Snapshot and load benchmark cards from `evaleval/auto-benchmarkcards`.

Two on-disk shapes are supported:
  - `cards/<name>.json` per-card files (each wrapping a `benchmark_card` or
    `benchmark_details` payload).
  - A flat `benchmark-metadata.json` mapping name -> card.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from eval_card_backend.config import BENCHMARK_METADATA_DATASET_REPO


def _has_cached(target: Path) -> bool:
    flat = target / "benchmark-metadata.json"
    cards_dir = target / "cards"
    return flat.exists() or (cards_dir.exists() and any(cards_dir.glob("*.json")))


def ensure_snapshot(local_dir: str, hf_token: str | None, force_refresh: bool) -> Path | None:
    target = Path(local_dir).resolve()
    if force_refresh and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    if _has_cached(target):
        return target

    try:
        snapshot_download(
            repo_id=BENCHMARK_METADATA_DATASET_REPO,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["benchmark-metadata.json", "cards/**"],
            token=hf_token,
        )
    except Exception:
        return target if _has_cached(target) else None

    return target


def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def load_cards(root: Path) -> dict[str, dict[str, Any]]:
    """Return `{normalized_name: card_payload}`. Empty dict if nothing cached."""
    cards: dict[str, dict[str, Any]] = {}
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
                        cards[key] = card

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
            if key and key not in cards:
                cards[key] = payload

    return cards

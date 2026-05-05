"""Preflight tests.

Preflight has to fail loudly on broken upstream state — empty EEE root, empty
or unreadable alias store, etc. — because a snapshot that runs but produces
all-NULL canonical IDs is the worst kind of regression in a cron job.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eval_card_backend.canonicalise import pipeline


def _write_minimal_eee(eee_root: Path) -> None:
    cfg_dir = eee_root / "data" / "minicfg" / "openai" / "gpt-4o"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "rec.json").write_text("{}")


def test_preflight_passes_when_alias_store_has_rows(tmp_path):
    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    _write_minimal_eee(eee_root)
    (reg_root / "aliases").mkdir(parents=True)
    pd.DataFrame([{
        "id": "1", "raw_value": "x", "entity_type": "model",
        "canonical_id": "x", "source_config": None, "source_field": None,
        "status": "active", "strategy": "exact", "confidence": 1.0,
        "notes": None, "created_at": "", "updated_at": "",
    }]).to_parquet(reg_root / "aliases" / "part-0.parquet")

    pipeline.preflight(eee_root, reg_root, tmp_path / "cards")  # cards missing OK


def test_preflight_fails_when_alias_store_is_empty(tmp_path):
    """File exists but has zero rows — the cold-start registry case."""
    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    _write_minimal_eee(eee_root)
    (reg_root / "aliases").mkdir(parents=True)
    pd.DataFrame(columns=[
        "id", "raw_value", "entity_type", "canonical_id", "source_config",
        "source_field", "status", "strategy", "confidence", "notes",
        "created_at", "updated_at",
    ]).to_parquet(reg_root / "aliases" / "part-0.parquet")

    with pytest.raises(SystemExit):
        pipeline.preflight(eee_root, reg_root, tmp_path / "cards")


def test_preflight_fails_when_alias_store_unreadable(tmp_path):
    """File present but corrupt — e.g. truncated download."""
    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    _write_minimal_eee(eee_root)
    (reg_root / "aliases").mkdir(parents=True)
    (reg_root / "aliases" / "part-0.parquet").write_bytes(b"not a parquet file")

    with pytest.raises(SystemExit):
        pipeline.preflight(eee_root, reg_root, tmp_path / "cards")

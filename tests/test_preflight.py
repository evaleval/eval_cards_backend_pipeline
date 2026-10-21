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
from tests.eee_layout import write_eee_datastore


def _write_minimal_eee(eee_root: Path) -> None:
    write_eee_datastore(eee_root, [("minicfg", "rec.json", "{}")])


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


def _write_alias_store(reg_root: Path) -> None:
    (reg_root / "aliases").mkdir(parents=True)
    pd.DataFrame([{
        "id": "1", "raw_value": "x", "entity_type": "model",
        "canonical_id": "x", "source_config": None, "source_field": None,
        "status": "active", "strategy": "exact", "confidence": 1.0,
        "notes": None, "created_at": "", "updated_at": "",
    }]).to_parquet(reg_root / "aliases" / "part-0.parquet")


def test_preflight_fails_on_an_eee_root_with_no_records(tmp_path, caplog):
    """A data/ tree that exists but holds no record is the shape a failed
    download leaves behind; it must not pass as a corpus."""
    eee_root = tmp_path / "eee"
    (eee_root / "data" / "cfg").mkdir(parents=True)
    _write_alias_store(tmp_path / "reg")

    with pytest.raises(SystemExit, match="Preflight failed"):
        pipeline.preflight(eee_root, tmp_path / "reg", tmp_path / "cards")
    assert "EEE source empty/missing" in caplog.text


def test_preflight_fails_on_a_missing_eee_root(tmp_path):
    _write_alias_store(tmp_path / "reg")
    with pytest.raises(SystemExit, match="Preflight failed"):
        pipeline.preflight(None, tmp_path / "reg", tmp_path / "cards")


def test_preflight_fails_on_a_missing_alias_store(tmp_path, caplog):
    eee_root = tmp_path / "eee"
    _write_minimal_eee(eee_root)
    (tmp_path / "reg").mkdir()

    with pytest.raises(SystemExit, match="Preflight failed"):
        pipeline.preflight(eee_root, tmp_path / "reg", tmp_path / "cards")
    assert "Registry alias store missing" in caplog.text


def test_preflight_fails_on_a_missing_registry_root(tmp_path):
    eee_root = tmp_path / "eee"
    _write_minimal_eee(eee_root)
    with pytest.raises(SystemExit, match="Preflight failed"):
        pipeline.preflight(eee_root, None, tmp_path / "cards")

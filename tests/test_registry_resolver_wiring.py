"""Regression tests for the backend's complete registry resolver bundle."""
from __future__ import annotations

import pandas as pd
import pytest

from eval_card_backend.sources import registry


def _write_table(root, table: str, frame: pd.DataFrame, layout: str) -> None:
    if layout == "flat":
        frame.to_parquet(root / f"{table}.parquet")
        return
    target = root / table
    target.mkdir()
    frame.to_parquet(target / "part-0.parquet")


def _aliases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "alias-1",
                "raw_value": "acme/Widget-7B-Instruct",
                "entity_type": "model",
                "canonical_id": "acme/widget-7b-instruct",
                "source_config": None,
                "source_field": "model_info.id",
                "status": "active",
                "strategy": "exact",
                "confidence": 1.0,
                "notes": None,
                "created_at": "",
                "updated_at": "",
            }
        ]
    )


def _models() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "acme/widget-7b-instruct",
                "display_name": "Widget 7B Instruct",
                "org_id": "acme",
                "parents": "[]",
                "model_group_id": "acme/widget-7b",
                "model_family_id": "acme/widget",
                "lineage_origin_model_id": None,
                "lineage_origin_model_org_id": None,
                "release_date": "2026-09-01",
                "resolution_source": "models_dev",
                "resolution_granularity": "variant",
                "open_weights": True,
                "params_billions": 7.0,
                "review_status": "reviewed",
                "metadata": "{}",
            }
        ]
    )


@pytest.mark.parametrize("layout", ["flat", "parts"])
def test_load_resolver_includes_canonical_enrichment(tmp_path, layout):
    _write_table(tmp_path, "aliases", _aliases(), layout)
    _write_table(tmp_path, "canonical_models", _models(), layout)

    result = registry.load_resolver(tmp_path).resolve(
        "acme/Widget-7B-Instruct", "model"
    )

    assert result.canonical_id == "acme/widget-7b-instruct"
    assert result.resolved_leaf_id == "acme/widget-7b-instruct"
    assert result.model_group_id == "acme/widget-7b"
    assert result.release_date == "2026-09-01"


def test_load_resolver_preserves_curated_alias_over_hf_shaped_raw(tmp_path):
    """Producer wiring must not replace a registered id with a bare HF hit."""
    _write_table(tmp_path, "aliases", _aliases(), "parts")
    _write_table(tmp_path, "canonical_models", _models(), "parts")

    result = registry.load_resolver(tmp_path).resolve(
        "acme/Widget-7B-Instruct", "model"
    )

    assert result.canonical_id == "acme/widget-7b-instruct"
    assert result.hf_attested_unregistered is False


def test_load_resolver_fails_closed_without_canonical_models(tmp_path):
    _write_table(tmp_path, "aliases", _aliases(), "flat")

    with pytest.raises(FileNotFoundError, match="canonical_models"):
        registry.load_resolver(tmp_path)

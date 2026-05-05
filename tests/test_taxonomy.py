"""Composite / family / slice-promotion taxonomy YAML loader tests."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

pytest.importorskip("yaml")

from eval_card_backend.canonicalise import taxonomy


def _write(tmp: Path, name: str, content: str) -> None:
    (tmp / name).write_text(content)


# ---------------------------------------------------------------------------
# Composite loading
# ---------------------------------------------------------------------------


def test_composite_default_when_no_file(tmp_path):
    """Missing composites.yaml → empty mapping; non-curated configs get
    default 1:1 treatment downstream."""
    assert taxonomy.load_composites(tmp_path) == {}


def test_composite_explicit_configs(tmp_path):
    _write(tmp_path, "composites.yaml", """
hal:
  display: HAL Leaderboard
  configs:
    - hal-gaia
    - hal-scicode
""")
    out = taxonomy.load_composites(tmp_path)
    assert out == {
        "hal": {
            "display": "HAL Leaderboard",
            "configs": ["hal-gaia", "hal-scicode"],
        }
    }


def test_composite_display_only_default_config_inferred(tmp_path):
    """Display-only override implicitly maps slug → [kebab, snake] so the
    JOIN matches whichever form upstream EEE uses."""
    _write(tmp_path, "composites.yaml", """
helm-classic:
  display: HELM Classic
""")
    out = taxonomy.load_composites(tmp_path)
    assert out["helm-classic"]["configs"] == ["helm-classic", "helm_classic"]


def test_composite_validation_duplicate_config(tmp_path):
    """A source_config can't map to two composites — Stage D would have
    to pick one arbitrarily."""
    _write(tmp_path, "composites.yaml", """
foo:
  configs: [shared]
bar:
  configs: [shared]
""")
    composites = taxonomy.load_composites(tmp_path)
    con = duckdb.connect()
    with pytest.raises(ValueError, match="shared"):
        taxonomy.materialise_taxonomy_tables(con, composites, {}, set())


def test_kebab_case_helper():
    assert taxonomy.kebab_case("HELM_Classic") == "helm-classic"
    assert taxonomy.kebab_case("foo bar.baz") == "foo-bar-baz"
    assert taxonomy.kebab_case("__weird__") == "weird"


# ---------------------------------------------------------------------------
# Family loading
# ---------------------------------------------------------------------------


def test_family_explicit_membership(tmp_path):
    _write(tmp_path, "families.yaml", """
mmlu:
  display: MMLU family
  benchmarks: [mmlu, mmlu-pro]
""")
    out = taxonomy.load_families(tmp_path)
    assert out == {
        "mmlu": {
            "display": "MMLU family",
            "benchmarks": ["mmlu", "mmlu-pro"],
        }
    }


def test_family_validation_benchmark_in_two_families(tmp_path):
    """A benchmark can belong to at most one curated family — the
    operator must fix the YAML before the pipeline can run."""
    _write(tmp_path, "families.yaml", """
foo:
  benchmarks: [shared]
bar:
  benchmarks: [shared]
""")
    with pytest.raises(ValueError, match="shared"):
        taxonomy.load_families(tmp_path)


# ---------------------------------------------------------------------------
# Slice promotions
# ---------------------------------------------------------------------------


def test_slice_promotions_loaded(tmp_path):
    _write(tmp_path, "slice_overrides.yaml", """
promote_to_benchmark:
  - bfcl-live
  - mmlu-pro
""")
    assert taxonomy.load_slice_promotions(tmp_path) == {"bfcl-live", "mmlu-pro"}


def test_slice_promotions_missing_file(tmp_path):
    assert taxonomy.load_slice_promotions(tmp_path) == set()


# ---------------------------------------------------------------------------
# DuckDB materialisation
# ---------------------------------------------------------------------------


def test_materialise_creates_three_tables(tmp_path):
    _write(tmp_path, "composites.yaml", """
hal:
  display: HAL Leaderboard
  configs: [hal-gaia, hal-scicode]
""")
    _write(tmp_path, "families.yaml", """
mmlu:
  display: MMLU family
  benchmarks: [mmlu, mmlu-pro]
""")
    _write(tmp_path, "slice_overrides.yaml", """
promote_to_benchmark: [bfcl-live]
""")

    con = duckdb.connect()
    composites, families, promotions = taxonomy.load_and_materialise(
        con, registry_root=None, seed_dir_override=tmp_path,
    )

    rows = con.execute(
        "SELECT source_config, composite_slug, composite_display_name "
        "FROM composite_config_map ORDER BY source_config"
    ).fetchall()
    assert rows == [
        ("hal-gaia", "hal", "HAL Leaderboard"),
        ("hal-scicode", "hal", "HAL Leaderboard"),
    ]
    assert composites["hal"]["display"] == "HAL Leaderboard"

    rows = con.execute(
        "SELECT family_id, family_display_name, benchmark_id "
        "FROM family_membership ORDER BY benchmark_id"
    ).fetchall()
    assert rows == [
        ("mmlu", "MMLU family", "mmlu"),
        ("mmlu", "MMLU family", "mmlu-pro"),
    ]

    rows = con.execute(
        "SELECT benchmark_id FROM slice_promotions"
    ).fetchall()
    assert rows == [("bfcl-live",)]
    assert promotions == {"bfcl-live"}


def test_seed_dir_override_takes_precedence(tmp_path, monkeypatch):
    """Override path wins over EVALCARD_REGISTRY_SEED_DIR env."""
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("EVALCARD_REGISTRY_SEED_DIR", str(other))
    chosen = taxonomy.resolve_seed_dir(None, tmp_path)
    assert chosen == tmp_path

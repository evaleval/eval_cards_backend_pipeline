"""Unit tests for scripts/registry.py.

Uses the bundled `tests/fixtures/registry_aliases/aliases.parquet` so the
suite is deterministic and offline. The fixture covers four entries:
``MMLU`` (global → ``mmlu``), ``helm_mmlu`` (scoped to ``helm_mmlu`` →
``mmlu``), ``BBH`` (global → ``bbh``), ``IFEval`` (global → ``ifeval``).
Live HF dataset coverage is exercised by the integration tests; this
file just needs the resolver wiring to behave correctly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_PARQUET_DIR = REPO_ROOT / "tests" / "fixtures" / "registry_aliases"


@pytest.fixture
def fixture_registry(monkeypatch):
    """Point the resolver at the local fixture parquet and reset its cache."""
    from scripts import registry

    monkeypatch.setenv("REGISTRY_LOCAL_PARQUET_DIR", str(FIXTURE_PARQUET_DIR))
    monkeypatch.delenv("REGISTRY_DISABLE", raising=False)
    registry.reset_for_tests()
    yield registry
    registry.reset_for_tests()


def test_resolver_loads_local_parquet(fixture_registry):
    result = fixture_registry.resolve_benchmark("MMLU", None)
    assert result["canonical_id"] == "mmlu"
    # ``MMLU`` is an exact alias raw_value in the fixture parquet, so
    # the patched resolver returns it via exact_match. The pre-patch
    # version returned ``normalized`` because exact_match was broken
    # (NaN/None mismatch on null source_config); see registry repo
    # alias_store._ensure_lookup_index.
    assert result["strategy"] == "exact"
    assert result["raw_value"] == "MMLU"


def test_resolver_returns_stable_dict_for_no_match(fixture_registry):
    result = fixture_registry.resolve_benchmark("DefinitelyNotARealBenchmark", None)
    assert result["canonical_id"] is None
    assert result["strategy"] == "no_match"
    assert result["confidence"] == 0.0
    # Caller-side audit fields preserved regardless of outcome.
    assert result["raw_value"] == "DefinitelyNotARealBenchmark"


def test_empty_input_short_circuits(fixture_registry):
    result = fixture_registry.resolve_benchmark(None, "any-config")
    assert result["canonical_id"] is None
    assert result["strategy"] == "empty_input"


def test_scoped_alias_uses_source_config(fixture_registry):
    # The fixture has ``helm_mmlu`` as a scoped alias under
    # source_config="helm_mmlu" mapping to canonical "mmlu".
    result = fixture_registry.resolve_benchmark("helm_mmlu", "helm_mmlu")
    assert result["canonical_id"] == "mmlu"
    assert result["strategy"] == "exact"


def test_disable_flag_skips_resolver(monkeypatch):
    from scripts import registry

    monkeypatch.setenv("REGISTRY_DISABLE", "1")
    monkeypatch.delenv("REGISTRY_LOCAL_PARQUET_DIR", raising=False)
    registry.reset_for_tests()
    try:
        result = registry.resolve_benchmark("MMLU", None)
        assert result["canonical_id"] is None
        assert result["strategy"] == "registry_unavailable"
    finally:
        registry.reset_for_tests()


def test_lru_cache_dedupes_repeat_lookups(fixture_registry):
    fixture_registry._resolve_cached.cache_clear()
    fixture_registry.resolve_benchmark("MMLU", "helm_classic")
    fixture_registry.resolve_benchmark("MMLU", "helm_classic")
    fixture_registry.resolve_benchmark("MMLU", "helm_lite")
    info = fixture_registry._resolve_cached.cache_info()
    # 3 calls, 2 unique (raw, sc) tuples → 1 hit, 2 misses
    assert info.hits == 1
    assert info.misses == 2

"""Test-suite conftest.

Provides a session-scoped stub taxonomy seed directory and points
`EVALCARD_REGISTRY_SEED_DIR` at it for every test, so tests that run
the canonicalisation pipeline don't trip the "no seed dir found" guard
in `taxonomy.load_and_materialise`. Tests that intentionally exercise
the no-seed-dir path can `monkeypatch.delenv("EVALCARD_REGISTRY_SEED_DIR")`
themselves.

The stub yaml carries one composites entry so the stage-A check in
`pipeline.run` (which refuses to restore a 0-row composite_config_map
from cache) is satisfied. Tests don't need the contents to match their
fixtures' source_configs — the row count is what's checked.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def _taxonomy_seed_stub(tmp_path_factory) -> Path:
    seed_dir = tmp_path_factory.mktemp("taxonomy_seed_stub")
    (seed_dir / "composites.yaml").write_text(
        "minibench:\n  display: MiniBench\n  configs:\n    - minibench\n"
    )
    return seed_dir


@pytest.fixture(autouse=True)
def _set_taxonomy_seed_env(monkeypatch, _taxonomy_seed_stub) -> None:
    monkeypatch.setenv("EVALCARD_REGISTRY_SEED_DIR", str(_taxonomy_seed_stub))
    # Tests run against local fixtures and must stay hermetic. CI pins the upstream
    # HF dataset revisions via env vars; if those leak into Settings.from_env(),
    # ensure_snapshot treats a fixture dir (which has no `.pinned_revision` marker)
    # as a stale cache, wipes it, and downloads the real dataset — which hangs CI.
    # Clear the pins so no test inherits them regardless of the ambient environment.
    for _rev_var in ("EEE_REVISION", "ENTITY_REGISTRY_REVISION", "BENCHMARK_METADATA_REVISION"):
        monkeypatch.delenv(_rev_var, raising=False)

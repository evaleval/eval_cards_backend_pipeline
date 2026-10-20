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

import os
from pathlib import Path

import pytest

from tests.fixture_corpus import UPSTREAM_ENV_VARS

# Module- and session-scoped pipeline fixtures call Settings.from_env() before
# the per-test guard below runs, so the refresh flags and upstream revision
# pins a shell or CI exports must be gone before any fixture is built.
# Otherwise ensure_snapshot refreshes a fixture dir, or reads it as a stale
# cache under a pin. The private-source settings go too: CI turns the private
# EEE source on, and a test that wants it sets them itself.
_PINNED_ENV_VARS = UPSTREAM_ENV_VARS + ("EEE_PRIVATE_LOCAL_DATASET_DIR",)
for _rev_var in _PINNED_ENV_VARS:
    os.environ.pop(_rev_var, None)

# Keep the suite hermetic: the open-weights backfill reaches the Hugging Face
# Hub, and a pipeline run under test must not depend on network reachability
# (or on what the Hub happens to say today). The backfill's own behaviour is
# covered against an injected double in test_open_weights_backfill.py.
os.environ.setdefault("HF_OPENNESS_PROBE", "0")


@pytest.fixture(scope="session")
def _taxonomy_seed_stub(tmp_path_factory) -> Path:
    seed_dir = tmp_path_factory.mktemp("taxonomy_seed_stub")
    (seed_dir / "composites.yaml").write_text(
        "minibench:\n  display: MiniBench\n  configs:\n    - minibench\n"
    )
    return seed_dir


@pytest.fixture(scope="session")
def _collections_stub(tmp_path_factory) -> Path:
    """Empty curated-collections file + empty vendor dir so tests stay
    hermetic: without these, pipeline runs would load the repo's real
    `collections_curated.yaml` (whose curated-match assertion expects the
    production corpus) and the real `vendor/collections/` extracts."""
    root = tmp_path_factory.mktemp("collections_stub")
    (root / "collections_curated.yaml").write_text("")
    (root / "vendor_collections").mkdir()
    return root


@pytest.fixture(autouse=True)
def _set_taxonomy_seed_env(monkeypatch, _taxonomy_seed_stub, _collections_stub) -> None:
    monkeypatch.setenv("EVALCARD_REGISTRY_SEED_DIR", str(_taxonomy_seed_stub))
    monkeypatch.setenv(
        "COLLECTIONS_CURATED_PATH",
        str(_collections_stub / "collections_curated.yaml"),
    )
    monkeypatch.setenv(
        "COLLECTIONS_VENDOR_DIR",
        str(_collections_stub / "vendor_collections"),
    )
    # Tests run against local fixtures and must stay hermetic: clear the refresh
    # flags and pins so no test inherits them from the ambient environment.
    for _rev_var in _PINNED_ENV_VARS:
        monkeypatch.delenv(_rev_var, raising=False)


@pytest.fixture(scope="session")
def fixture_corpus_build(tmp_path_factory, _taxonomy_seed_stub, _collections_stub):
    """One full pipeline run over `tests/fixtures/`, shared by the tests
    that inspect a whole warehouse (see `tests/fixture_corpus.py`)."""
    from tests.fixture_corpus import build_fixture_corpus

    return build_fixture_corpus(
        tmp_path_factory.mktemp("fixture_corpus"),
        seed_dir=_taxonomy_seed_stub,
        collections_dir=_collections_stub,
    )

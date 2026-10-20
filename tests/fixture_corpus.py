"""Run the full pipeline over the hand-built corpus in `tests/fixtures/`.

Shared by the tests that inspect a whole warehouse (cache shapes, byte
determinism, published counts) rather than one stage's output. The
session-scoped `fixture_corpus_build` fixture in `conftest.py` wraps
`build_fixture_corpus` so those tests share one bake.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT_ID = "2026-04-30T00:00:00Z"

# Every variable through which `Settings.from_env()` turns on a source
# refresh, a revision pin or the private source. The fixture builder and
# conftest strip them all; test_fixtures_offline checks this list against
# what config.py reads.
UPSTREAM_ENV_VARS = (
    "EEE_REFRESH_SNAPSHOT", "ENTITY_REGISTRY_REFRESH", "BENCHMARK_METADATA_REFRESH",
    "EEE_REVISION", "ENTITY_REGISTRY_REVISION", "BENCHMARK_METADATA_REVISION",
    "EEE_INCLUDE_PRIVATE", "EEE_PRIVATE_REVISION",
)


@dataclass(frozen=True)
class CorpusBuild:
    out_dir: Path
    cache_dir: Path
    registry_dir: Path
    snapshot_id: str


def build_fixture_corpus(
    root: Path,
    *,
    seed_dir: Path,
    collections_dir: Path,
    registry_dir: Path | None = None,
    snapshot_id: str = SNAPSHOT_ID,
) -> CorpusBuild:
    """Bake the fixture corpus into `root/warehouse` with its stage cache
    under `root/cache`. `registry_dir` swaps in a modified copy of the
    fixture registry; `seed_dir` / `collections_dir` are the hermetic
    stubs from conftest, set here because a session-scoped caller runs
    before the per-test autouse fixture that normally exports them.
    """
    registry_dir = registry_dir or FIXTURES / "entity_registry"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EVALCARD_REGISTRY_SEED_DIR", str(seed_dir))
        mp.setenv(
            "COLLECTIONS_CURATED_PATH",
            str(collections_dir / "collections_curated.yaml"),
        )
        mp.setenv(
            "COLLECTIONS_VENDOR_DIR", str(collections_dir / "vendor_collections")
        )
        mp.setenv("EEE_LOCAL_DATASET_DIR", str(FIXTURES / "eee"))
        mp.setenv(
            "BENCHMARK_METADATA_LOCAL_DIR", str(FIXTURES / "auto_benchmarkcards")
        )
        for var in UPSTREAM_ENV_VARS:
            mp.delenv(var, raising=False)

        from eval_card_backend.canonicalise import pipeline
        from eval_card_backend.canonicalise.cache import StageCache
        from eval_card_backend.config import Settings

        out_dir = pipeline.run(
            Settings.from_env(),
            snapshot_id=snapshot_id,
            warehouse_dir=str(root / "warehouse"),
            registry_local_dir=str(registry_dir),
            cache_root=str(root / "cache"),
        )
    assert out_dir is not None
    return CorpusBuild(
        out_dir=out_dir,
        cache_dir=StageCache(root / "cache", snapshot_id).dir,
        registry_dir=registry_dir,
        snapshot_id=snapshot_id,
    )

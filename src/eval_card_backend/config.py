from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# src/eval_card_backend/config.py -> repo root. Same anchor the other
# repo-relative data paths already use (see `sources/collections.py`).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Env-overridable for testing against a fork/mirror. The same repo serves
# both the snapshot download and the eee_record_url deep-links.
EEE_DATASET_REPO = os.environ.get("EEE_DATASET_REPO", "evaleval/EEE_datastore")
BENCHMARK_METADATA_DATASET_REPO = "evaleval/auto-benchmarkcards"
ENTITY_REGISTRY_DATASET_REPO = "evaleval/entity-registry-data"

# Anchored at the repo root, not the launch directory. These were plain
# relative strings, so a bake started from a subdirectory silently forked a
# second empty cache (and a second warehouse) next to itself instead of
# reusing the repo's. CI runs from the repo root, so the resolved paths are
# unchanged there. Explicit overrides (the env vars below, --warehouse-dir,
# --registry-local-dir) are still honoured exactly as given.
DEFAULT_EEE_LOCAL_DIR = str(REPO_ROOT / ".cache" / "eee_datastore")
DEFAULT_BENCHMARK_METADATA_LOCAL_DIR = str(REPO_ROOT / ".cache" / "auto_benchmarkcards")
DEFAULT_REGISTRY_LOCAL_DIR = str(REPO_ROOT / ".cache" / "entity_registry")
DEFAULT_WAREHOUSE_DIR = str(REPO_ROOT / "warehouse")

# Configs unconditionally excluded due to upstream data-quality issues.
# Filter applies even when a user explicitly passes the config name via
# --configs — these are not user-overridable.
#
# alphaxiv: paper-checkpoint leaderboard publishes models without proper
# developer/org attribution (`unknown__<x>` patterns); causes systematic
# provenance/resolution noise. Re-include when upstream cleans up.
IGNORED_CONFIGS: frozenset[str] = frozenset({"alphaxiv"})


@dataclass(frozen=True)
class Settings:
    hf_token: str | None
    eee_local_dir: str
    benchmark_metadata_local_dir: str
    registry_local_dir: str
    warehouse_dir: str
    refresh_eee: bool
    refresh_benchmark_metadata: bool
    refresh_registry: bool
    # Optional upstream revision pins (HF dataset commit SHA, branch, or tag).
    # When set, the source is downloaded at that exact revision instead of
    # latest HEAD. None = latest (default, unchanged behaviour). Used to run
    # the pipeline against a known upstream state so code changes can be
    # validated in isolation from upstream data changes (separate "did my
    # code change the output" from "did the data change").
    eee_revision: str | None
    benchmark_metadata_revision: str | None
    registry_revision: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            hf_token=os.environ.get("HF_TOKEN"),
            eee_local_dir=(
                os.environ.get("EEE_LOCAL_DATASET_DIR") or DEFAULT_EEE_LOCAL_DIR
            ),
            benchmark_metadata_local_dir=(
                os.environ.get("BENCHMARK_METADATA_LOCAL_DIR")
                or DEFAULT_BENCHMARK_METADATA_LOCAL_DIR
            ),
            registry_local_dir=(
                os.environ.get("ENTITY_REGISTRY_LOCAL_DIR") or DEFAULT_REGISTRY_LOCAL_DIR
            ),
            warehouse_dir=(
                os.environ.get("WAREHOUSE_DIR") or DEFAULT_WAREHOUSE_DIR
            ),
            refresh_eee=os.environ.get("EEE_REFRESH_SNAPSHOT") == "1",
            refresh_benchmark_metadata=(
                os.environ.get("BENCHMARK_METADATA_REFRESH") == "1"
            ),
            refresh_registry=os.environ.get("ENTITY_REGISTRY_REFRESH") == "1",
            eee_revision=os.environ.get("EEE_REVISION") or None,
            benchmark_metadata_revision=(
                os.environ.get("BENCHMARK_METADATA_REVISION") or None
            ),
            registry_revision=os.environ.get("ENTITY_REGISTRY_REVISION") or None,
        )

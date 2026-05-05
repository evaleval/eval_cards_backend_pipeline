from __future__ import annotations

import os
from dataclasses import dataclass

EEE_DATASET_REPO = "evaleval/EEE_datastore"
BENCHMARK_METADATA_DATASET_REPO = "evaleval/auto-benchmarkcards"
ENTITY_REGISTRY_DATASET_REPO = "evaleval/entity-registry-data"

DEFAULT_EEE_LOCAL_DIR = ".cache/eee_datastore"
DEFAULT_BENCHMARK_METADATA_LOCAL_DIR = ".cache/auto_benchmarkcards"
DEFAULT_REGISTRY_LOCAL_DIR = ".cache/entity_registry"
DEFAULT_WAREHOUSE_DIR = "warehouse"

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
        )

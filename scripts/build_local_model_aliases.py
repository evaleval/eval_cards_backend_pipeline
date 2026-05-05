"""Build a local registry parquet that includes MODEL aliases.

The registry's `seed` CLI doesn't currently process `models.yaml` —
fixtures/aliases.parquet only has benchmark/metric/harness entries. For our
local provenance investigation we need model resolution working, so this
script:

  1. Reads `seed/models.yaml` + `seed/_overrides/models.yaml` from the
     registry repo (overrides win on canonical_id collision).
  2. Loads the existing fixtures/aliases.parquet (benchmark/metric/harness).
  3. Generates new model alias rows for every (id, alias) pair.
  4. Writes the merged aliases parquet to
     `.cache/local_registry_with_models/aliases.parquet` along with copies
     of the canonical_*.parquet tables (resolver loads from a directory).

Run from repo root:

    uv run --with pyyaml --with pandas --with pyarrow --no-project \
      python scripts/build_local_model_aliases.py

Output dir is hardcoded; set REGISTRY_LOCAL_PARQUET_DIR to that path when
running pipeline / queries to use it.
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REGISTRY_REPO = Path("/Users/jchim/projects/evaleval/evalcard-registry")
SEED_MAIN = REGISTRY_REPO / "seed" / "models.yaml"
SEED_OVERRIDES = REGISTRY_REPO / "seed" / "_overrides" / "models.yaml"
FIXTURES = REGISTRY_REPO / "fixtures"

LOCAL_ADDITIONS = Path("notes/local_alias_additions.yaml")
OUTPUT_DIR = Path(".cache/local_registry_with_models")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml_models(path: Path) -> list[dict]:
    """Returns a flat list of model entries.

    Two shapes supported:
      - Flat list (seed/models.yaml — generated from models.dev)
      - {skip_ids: [...], entries: [...]} (seed/_overrides/models.yaml)
    """
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "entries" in data:
        return data["entries"] or []
    raise ValueError(f"Unexpected YAML shape at {path}: {type(data)}")


def _alias_row(raw_value: str, canonical_id: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "raw_value": raw_value,
        "entity_type": "model",
        "canonical_id": canonical_id,
        "source_config": None,
        "source_field": "seed",
        "status": "confirmed",
        "strategy": "seed",
        "confidence": 1.0,
        "notes": None,
        "created_at": _now(),
        "updated_at": _now(),
    }


def _load_local_additions() -> tuple[dict[str, list[str]], list[dict]]:
    if not LOCAL_ADDITIONS.exists():
        return {}, []
    with open(LOCAL_ADDITIONS) as f:
        data = yaml.safe_load(f) or {}
    return (data.get("extend") or {}, data.get("new") or [])


def main() -> None:
    main_models = _load_yaml_models(SEED_MAIN)
    override_models = _load_yaml_models(SEED_OVERRIDES)
    extend_aliases, new_models = _load_local_additions()
    print(f"  models.yaml: {len(main_models)} entries")
    print(f"  _overrides/models.yaml: {len(override_models)} entries")
    print(f"  local additions — extend: {len(extend_aliases)} canonicals, new: {len(new_models)}")

    # Overrides win on id collision
    by_id: dict[str, dict] = {}
    for entry in main_models:
        by_id[entry["id"]] = entry
    for entry in override_models:
        by_id[entry["id"]] = entry
    for entry in new_models:
        if entry["id"] in by_id:
            print(f"  WARN: local 'new:' entry {entry['id']} already exists; using local version")
        by_id[entry["id"]] = entry

    # Apply extend additions — merge into existing aliases without replacing
    extend_unmatched: list[str] = []
    for canonical_id, extras in extend_aliases.items():
        if canonical_id not in by_id:
            extend_unmatched.append(canonical_id)
            continue
        existing = list(by_id[canonical_id].get("aliases") or [])
        existing.extend(extras)
        by_id[canonical_id] = {**by_id[canonical_id], "aliases": existing}
    if extend_unmatched:
        print(f"  WARN: {len(extend_unmatched)} extend: ids not found in main/overrides:")
        for uid in extend_unmatched:
            print(f"    {uid}")

    print(f"  merged: {len(by_id)} unique canonical model ids")

    rows: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()  # (raw_value_lower, canonical_id)
    for canonical_id, entry in by_id.items():
        display_name = entry.get("display_name") or ""
        family = entry.get("family") or ""
        aliases = entry.get("aliases") or []

        # Add canonical_id itself + display_name + family + each alias
        candidates = {canonical_id, display_name, family} | set(aliases)
        for raw in candidates:
            if not raw:
                continue
            key = (raw.lower(), canonical_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(_alias_row(raw, canonical_id))

    print(f"  generated {len(rows)} model alias rows")

    new_aliases = pd.DataFrame(rows)
    existing_aliases = pd.read_parquet(FIXTURES / "aliases.parquet")
    print(f"  existing fixture aliases: {len(existing_aliases)}")
    merged = pd.concat([existing_aliases, new_aliases], ignore_index=True)
    print(f"  total merged aliases: {len(merged)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy canonical_*.parquet so the resolver has the full store to load
    for fname in [
        "canonical_models.parquet",
        "canonical_benchmarks.parquet",
        "canonical_metrics.parquet",
        "canonical_orgs.parquet",
        "eval_harnesses.parquet",
        "eval_results.parquet",
        "resolution_log.parquet",
        "sync_runs.parquet",
    ]:
        src = FIXTURES / fname
        dst = OUTPUT_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  WARN: missing {src}")

    merged.to_parquet(OUTPUT_DIR / "aliases.parquet", index=False)
    print(f"  wrote {OUTPUT_DIR / 'aliases.parquet'}")
    print(f"\nUsage: REGISTRY_LOCAL_PARQUET_DIR={OUTPUT_DIR.resolve()} <command>")


if __name__ == "__main__":
    main()

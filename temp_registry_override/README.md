# temp_registry_override

**Local taxonomy override for the eval-card-backend pipeline.**
**TODO: registry team — fold these into `evaleval/entity-registry-data` so this folder can be deleted.**

## Why this exists

The production pipeline (`eval-card-backend canonicalise`) reads its
family / composite curation from one of:

1. `<registry_root>/canonical_families.parquet` + `canonical_composites.parquet`
   — preferred, single source of truth.
2. Seed YAMLs (`families.yaml`, `composites.yaml`, `slice_overrides.yaml`)
   loaded from `EVALCARD_REGISTRY_SEED_DIR`, `--taxonomy-seed-dir`, or a
   sibling `evalcard-registry/seed/` checkout.

As of the v2 hierarchy work (May 2026) the registry parquets ship the
canonical benchmark dim but **not yet** `canonical_families.parquet` or
`canonical_composites.parquet` with the `family_id` linkage. Without
either source, the pipeline falls back to "every composite is its own
singleton family", which shatters HELM into 7 families, HAL into 9,
tau-bench into 3, fibble-arena into 6, etc.

This folder ships those YAMLs locally so:

- the standalone audit tool [scripts/build_hierarchy_v2.py](../scripts/build_hierarchy_v2.py)
  reads from here directly, and
- the production pipeline can be pointed at the same files via
  `EVALCARD_REGISTRY_SEED_DIR=temp_registry_override`.

Both produce the family bucketing this team has reviewed and signed off on.

## What's in here

| file                       | purpose                                                                              |
|----------------------------|--------------------------------------------------------------------------------------|
| `composites.yaml`          | composite_slug → `{display, configs: [eee_folder_name, ...]}`                       |
| `families.yaml`            | family_id → `{display, benchmarks: [composite_slug, ...]}` (multi-composite groups) |
| `slice_overrides.yaml`     | `promote_to_benchmark: [...]` — slices that should surface as standalone benchmarks |
| `display_overrides.yaml`   | `acronyms: [...]` — slugs that prettify_display should render uppercase             |

Singleton families (one composite, no shared family) live in `composites.yaml`
only — they don't need a `families.yaml` entry; the loader treats the
composite slug as the family id.

## Multi-composite families currently encoded

| family_id      | composites                                                                                          |
|----------------|-----------------------------------------------------------------------------------------------------|
| `helm`         | helm-classic, helm-lite, helm-capabilities, helm-instruct, helm-mmlu, helm-air-bench, helm-safety   |
| `hal`          | hal-gaia, hal-assistantbench, hal-corebench-hard, hal-online-mind2web, hal-scicode, hal-scienceagentbench, hal-swebench-mini, hal-taubench-airline, hal-usaco |
| `tau-bench`    | tau-bench-2-airline, tau-bench-2-retail, tau-bench-2-telecom                                        |
| `fibble-arena` | fibble-arena, fibble-1-arena, fibble-2-arena, fibble-3-arena, fibble-4-arena, fibble-5-arena         |
| `mmmu`         | mmmu-multiple-choice, mmmu-open-ended                                                               |
| `reward-bench` | reward-bench, reward-bench-2                                                                        |
| `cyse2`        | cyse2-interpreter-abuse, cyse2-prompt-injection, cyse2-vulnerability-exploit                        |

## Action for the registry developer

When folding this into the registry:

1. Mirror `composites.yaml` into `canonical_composites.parquet` rows, with
   each row's `family_id` populated from the inverse of `families.yaml`
   (composite_slug → family_id).
2. Mirror `families.yaml` into `canonical_families.parquet` (id, display_name,
   benchmark_ids[], composite_keys[]).
3. Mirror `slice_overrides.yaml` either as a parquet column on canonical_benchmarks
   (`promote_as_benchmark: bool`) or keep it as a YAML alongside the parquets —
   the pipeline's `load_slice_promotions_from_registry` already reads either form.
4. Mirror `display_overrides.yaml` into a single-file YAML alongside the parquets
   so `prettify_display` can fetch curated acronyms from the registry cache.
5. Delete this folder once the parquet path is live.

A reasonable test for completion: running the pipeline with no
`EVALCARD_REGISTRY_SEED_DIR` set and no `temp_registry_override/` on disk
should still produce a `hierarchy.json` whose family bucketing matches
the structure encoded here.

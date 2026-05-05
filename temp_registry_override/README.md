# temp_registry_override

**Local taxonomy override for the eval-card-backend pipeline.**
**TODO: registry team — fold these into `evaleval/entity-registry-data` so this folder can be deleted.**

> **CI tests are paused while the v2 hierarchy emit and the curated taxonomy
> in this folder are iterating quickly.** The `pytest` step in
> `.github/workflows/sync.yml` is short-circuited so a failed test doesn't
> block the warehouse upload. Re-enable it once the schema settles.
>
> When you flip tests back on, expect to update:
> - `tests/test_stage_j_sidecars.py` — the hierarchy assertions need to
>   match the v2 shape (`src/eval_card_backend/canonicalise/hierarchy_v2.py`),
>   not the legacy registry-driven one (`sidecars.write_hierarchy`).
> - `tests/test_stage_j_models_view.py` — `category_stats` STRUCT keys
>   are now the 18-value enum from `categorized.json`.
> - any test that hardcodes `{General, Reasoning, Agentic, Safety, Knowledge}`
>   — replace with `set(categorisation.categories())` so the source of
>   truth lives in one place.

## What's in here, and who reads it

The pipeline has two consumers of curated taxonomy:

1. **Stages A–I** (warehouse build) — read `composites.yaml`,
   `families.yaml`, `slice_overrides.yaml`, `display_overrides.yaml`
   via `taxonomy.load_and_materialise`. These YAMLs drive
   `composite_config_map`, `family_membership`, `slice_promotions` —
   the warehouse-side dim/fact rows that downstream parquets
   (`models_view.parquet`, `comparison-index.json`,
   `benchmark_index.parquet`) all key on. Without them, multi-config
   leaderboards (HELM, Vals.ai, RewardBench) silently split into per-
   config singleton composites.
2. **Stage J — hierarchy emit** — `hierarchy_v2.py` reads the EEE
   datastore directly and applies the same family/composite mappings
   as Python constants (`EXPLICIT_FAMILY_MAP`, `EXPLICIT_COMPOSITE_MAP`,
   `FAMILY_PRIMARY_OVERRIDE`, `ACRONYMS` baked into the module). The
   YAMLs are NOT used here — the v2 logic is the source of truth for
   the navigation tree.
3. **Categorisation** — both `categorisation.py` (the per-row UDF) and
   `hierarchy_v2.py` read `categorized.json` (display name →
   curated category list). 18-value taxonomy:
   ```
   agentic, applied_reasoning, commonsense_reasoning, finance, general,
   hallucination, humanities_and_social_sciences, knowledge, law,
   linguistic_core, logical_reasoning, mathematics, multimodal,
   natural_sciences, other, robustness, safety, software_engineering
   ```

## CI wiring

`.github/workflows/sync.yml` sets:

```yaml
EVALCARD_REGISTRY_SEED_DIR: ${{ github.workspace }}/temp_registry_override
```

so all three consumers find what they need on every run.

## Multi-composite families currently encoded (hierarchy_v2.py)

| family_id      | composite slugs                                                                                     |
|----------------|-----------------------------------------------------------------------------------------------------|
| `helm`         | helm-classic, helm-lite, helm-capabilities, helm-instruct, helm-mmlu, helm-air-bench, helm-safety   |
| `hal`          | hal-gaia, hal-assistantbench, hal-corebench-hard, hal-online-mind2web, hal-scicode, hal-scienceagentbench, hal-swebench-mini, hal-taubench-airline, hal-usaco |
| `tau-bench`    | tau-bench-2-airline, tau-bench-2-retail, tau-bench-2-telecom, hal-taubench-airline                  |
| `fibble-arena` | fibble-arena, fibble-1-arena, fibble-2-arena, fibble-3-arena, fibble-4-arena, fibble-5-arena         |
| `mmmu`         | mmmu-multiple-choice, mmmu-open-ended                                                               |
| `reward-bench` | reward-bench, reward-bench-2                                                                        |
| `cyse2`        | cyse2-interpreter-abuse, cyse2-prompt-injection, cyse2-vulnerability-exploit                        |

(Same set the YAMLs encode — kept in sync by hand for now; long-term
both move into the registry.)

## Action for the registry developer

When folding this into the registry:

1. Mirror `composites.yaml` into `canonical_composites.parquet` rows
   with `family_id` populated from the inverse of `families.yaml`.
2. Mirror `families.yaml` into `canonical_families.parquet`
   (id, display_name, benchmark_ids[], composite_keys[]).
3. Mirror `slice_overrides.yaml` either as a column on
   `canonical_benchmarks.parquet` (`promote_as_benchmark: bool`) or
   keep it as a YAML alongside the parquets.
4. Mirror `display_overrides.yaml` (`acronyms: [...]`) similarly.
5. Mirror `categorized.json` as a column on `canonical_benchmarks`
   (`categories: VARCHAR[]`) or as a YAML alongside.
6. Drop the `EVALCARD_REGISTRY_SEED_DIR` env var override from
   `.github/workflows/sync.yml`. Delete this folder.

A reasonable test for completion: running the pipeline with no
`temp_registry_override/` on disk should still produce the same
`hierarchy.json` and `category_stats` shape.

## File map

| file                       | who reads it                                                |
|----------------------------|-------------------------------------------------------------|
| `categorized.json`         | `categorisation.py`, `hierarchy_v2.py`                      |
| `composites.yaml`          | `taxonomy.load_and_materialise` (warehouse stages)          |
| `families.yaml`            | `taxonomy.load_and_materialise` (warehouse stages)          |
| `slice_overrides.yaml`     | `taxonomy.load_and_materialise` (warehouse stages)          |
| `display_overrides.yaml`   | `categorisation` / `taxonomy` acronym lookup                |
| `README.md`                | This file.                                                  |

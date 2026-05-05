"""Composite / family / slice taxonomy loader.

Reads curated taxonomy data from the registry data cache and
materialises three small in-memory tables on the DuckDB connection:

- `composite_config_map(source_config, composite_slug, composite_display_name)`
- `family_membership(family_id, family_display_name, benchmark_id)`
- `slice_promotions(benchmark_id)`

**Source preference** (per `notes/hierarchy-alignment.md` §4 — the
registry curation home):

  1. Parquet from `<registry_local_dir>/canonical_composites.parquet`
     and `canonical_families.parquet`. Single source of truth. Shipped
     alongside the rest of the registry data via
     `eval-card-registry/scripts/publish_registry_data.py`.
  2. YAML fallback at `<registry_local_dir>/../{eval-card-registry,
     evalcard-registry}/seed/`. Used when the registry's published
     dataset predates the canonical_families / canonical_composites
     tables (back-compat) or when running against a sibling registry
     checkout in development.

`slice_overrides.yaml` is shipped as YAML alongside the parquets (see
the publish script's SHIPPED_YAML list). The producer reads it from
the registry cache first, then from the seed dir as fallback.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


_KEBAB_RE = re.compile(r"[^a-z0-9-]+")


def kebab_case(s: str) -> str:
    """Lowercase, replace non-alphanumeric runs with `-`, strip leading/
    trailing `-`. Used as the default composite_slug derivation when a
    source_config isn't curated.
    """
    s = s.lower().replace("_", "-")
    s = _KEBAB_RE.sub("-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _load_yaml(path: Path) -> dict | None:
    """Parse a YAML file, returning the top-level mapping or None when
    the file is absent. Returns None on parse failure with a warning;
    callers treat None and {} the same (no curated entries).
    """
    if not path.exists():
        return None
    try:
        import yaml
    except ImportError:
        log.warning(
            "PyYAML not installed; cannot read taxonomy seed at %s. "
            "Install with `uv add pyyaml` or skip curated taxonomy.", path,
        )
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        log.warning("taxonomy: failed to parse %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        log.warning(
            "taxonomy: expected top-level mapping in %s, got %s",
            path, type(data).__name__,
        )
        return None
    return data


def resolve_seed_dir(registry_root: Path | None, override: Path | None = None) -> Path | None:
    """Return the directory holding the taxonomy YAMLs.

    Resolution order:
      1. `override` if supplied (test injection / explicit path).
      2. `EVALCARD_REGISTRY_SEED_DIR` env var if set.
      3. Sibling of `registry_root`: `<registry_root>/../seed/` —
         correct when eval-card-backend and eval-card-registry are
         checked out side-by-side via the uv workspace dep.

    Returns None when no candidate exists.
    """
    import os

    if override is not None:
        return Path(override) if Path(override).exists() else None
    env = os.environ.get("EVALCARD_REGISTRY_SEED_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p
    if registry_root is not None:
        # registry_root is typically `<repo>/.cache/registry/` — the seed
        # YAMLs live next to the registry source repo, not in the data
        # cache. Walk up to find a sibling `eval-card-registry/seed/`.
        for ancestor in [registry_root, *registry_root.parents]:
            candidate = ancestor / "eval-card-registry" / "seed"
            if candidate.exists():
                return candidate
            candidate = ancestor / "evalcard-registry" / "seed"
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Loaders — return plain Python data; the SQL materialiser is below.
# ---------------------------------------------------------------------------


def load_composites(seed_dir: Path) -> dict[str, dict]:
    """Return `{composite_slug: {display: str, configs: [str, ...]}}`.

    `configs` is normalised to a list. Default-display compositional
    entries (key = slug, value = {display: ...} only) get
    `configs = [slug]` so a 1:1 mapping still produces a usable map.
    """
    data = _load_yaml(seed_dir / "composites.yaml")
    if not data:
        return {}
    out: dict[str, dict] = {}
    for slug, entry in data.items():
        if not isinstance(slug, str):
            continue
        if not isinstance(entry, dict):
            log.warning("composites.yaml: entry %r is not a mapping; skipping", slug)
            continue
        display = entry.get("display") or slug
        configs = entry.get("configs")
        if configs is None:
            # Display-only override (no explicit `configs:`): emit the
            # slug in both kebab and snake forms so the JOIN matches
            # whichever the upstream EEE folder uses (e.g. `arc-agi` is
            # kebab, `helm_classic` is snake).
            kebab = slug
            snake = slug.replace("-", "_")
            configs = [kebab] if kebab == snake else [kebab, snake]
        if not isinstance(configs, list):
            log.warning(
                "composites.yaml: %r.configs must be a list, got %s",
                slug, type(configs).__name__,
            )
            continue
        out[slug] = {"display": str(display), "configs": [str(c) for c in configs]}
    return out


def load_families(seed_dir: Path) -> dict[str, dict]:
    """Return `{family_id: {display: str, benchmarks: [str, ...]}}`.

    Validation: a benchmark must appear in at most one family. Raises
    `ValueError` on conflict — the operator must fix the YAML before the
    pipeline can run.
    """
    data = _load_yaml(seed_dir / "families.yaml")
    if not data:
        return {}
    out: dict[str, dict] = {}
    seen_benchmarks: dict[str, str] = {}
    for fid, entry in data.items():
        if not isinstance(fid, str):
            continue
        if not isinstance(entry, dict):
            log.warning("families.yaml: entry %r is not a mapping; skipping", fid)
            continue
        display = entry.get("display") or fid
        benchmarks = entry.get("benchmarks") or []
        if not isinstance(benchmarks, list):
            log.warning(
                "families.yaml: %r.benchmarks must be a list, got %s",
                fid, type(benchmarks).__name__,
            )
            continue
        bench_ids = [str(b) for b in benchmarks]
        for bid in bench_ids:
            prior = seen_benchmarks.get(bid)
            if prior is not None and prior != fid:
                raise ValueError(
                    f"families.yaml: benchmark {bid!r} appears in both "
                    f"{prior!r} and {fid!r} — a benchmark can belong to at "
                    f"most one curated family."
                )
            seen_benchmarks[bid] = fid
        out[fid] = {"display": str(display), "benchmarks": bench_ids}
    return out


def load_slice_promotions(seed_dir: Path) -> set[str]:
    """Return the set of benchmark ids in `promote_to_benchmark`."""
    data = _load_yaml(seed_dir / "slice_overrides.yaml")
    if not data:
        return set()
    promote = data.get("promote_to_benchmark") or []
    if not isinstance(promote, list):
        log.warning(
            "slice_overrides.yaml: promote_to_benchmark must be a list, "
            "got %s", type(promote).__name__,
        )
        return set()
    return {str(b) for b in promote}


# ---------------------------------------------------------------------------
# DuckDB materialisation
# ---------------------------------------------------------------------------


def materialise_taxonomy_tables(
    con,
    composites: dict[str, dict],
    families: dict[str, dict],
    slice_promotions: set[str],
) -> None:
    """Create three small tables on the connection.

    Validation: if any source_config appears in two composites,
    raise — Stage D would otherwise pick an arbitrary one.
    """
    # Validate composites: each source_config in at most one composite.
    seen_configs: dict[str, str] = {}
    for slug, entry in composites.items():
        for cfg in entry["configs"]:
            prior = seen_configs.get(cfg)
            if prior is not None and prior != slug:
                raise ValueError(
                    f"composites.yaml: source_config {cfg!r} appears in "
                    f"both {prior!r} and {slug!r}."
                )
            seen_configs[cfg] = slug

    con.execute(
        "CREATE OR REPLACE TABLE composite_config_map ("
        "source_config VARCHAR, composite_slug VARCHAR, "
        "composite_display_name VARCHAR)"
    )
    composite_rows: list[tuple[str, str, str]] = []
    for slug, entry in composites.items():
        for cfg in entry["configs"]:
            composite_rows.append((cfg, slug, entry["display"]))
    if composite_rows:
        con.executemany(
            "INSERT INTO composite_config_map VALUES (?, ?, ?)",
            composite_rows,
        )

    con.execute(
        "CREATE OR REPLACE TABLE family_membership ("
        "family_id VARCHAR, family_display_name VARCHAR, benchmark_id VARCHAR)"
    )
    family_rows: list[tuple[str, str, str]] = []
    for fid, entry in families.items():
        for bid in entry["benchmarks"]:
            family_rows.append((fid, entry["display"], bid))
    if family_rows:
        con.executemany(
            "INSERT INTO family_membership VALUES (?, ?, ?)",
            family_rows,
        )

    con.execute(
        "CREATE OR REPLACE TABLE slice_promotions (benchmark_id VARCHAR)"
    )
    if slice_promotions:
        con.executemany(
            "INSERT INTO slice_promotions VALUES (?)",
            [(b,) for b in sorted(slice_promotions)],
        )


def load_composites_from_parquet(registry_root: Path) -> dict[str, dict] | None:
    """Read composites from the registry's `canonical_composites.parquet`
    in the data cache. Returns None when the parquet is missing — caller
    falls back to the YAML loader.

    Output shape matches `load_composites`:
        {composite_slug: {display: str, configs: [str, ...]}}
    `source_configs` on the parquet is JSON-encoded; we decode tolerantly.
    """
    import json as _json

    candidates = [
        registry_root / "canonical_composites.parquet",
        registry_root / "canonical_composites" / "part-0.parquet",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        log.warning("taxonomy: failed to read %s: %s", path, exc)
        return None

    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        slug = row.get("id")
        if not isinstance(slug, str):
            continue
        display = row.get("display_name") or slug
        raw = row.get("source_configs")
        configs: list[str]
        if isinstance(raw, list):
            configs = [str(c) for c in raw]
        elif isinstance(raw, str) and raw.strip() and raw.strip() not in ("[]", "null"):
            try:
                decoded = _json.loads(raw)
                configs = [str(c) for c in decoded] if isinstance(decoded, list) else []
            except (ValueError, TypeError):
                configs = []
        else:
            configs = []
        if not configs:
            # Display-only entries default to slug-as-config (replaces the
            # YAML loader's `slug.replace("-", "_")` heuristic).
            configs = [slug.replace("-", "_")]
        out[slug] = {"display": str(display), "configs": configs}
    return out


def load_families_from_parquet(registry_root: Path) -> dict[str, dict] | None:
    """Read families from the registry's `canonical_families.parquet`.
    Returns None on missing file. Output matches `load_families`.

    `benchmark_ids` is JSON-encoded on the parquet.
    """
    import json as _json

    candidates = [
        registry_root / "canonical_families.parquet",
        registry_root / "canonical_families" / "part-0.parquet",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        log.warning("taxonomy: failed to read %s: %s", path, exc)
        return None

    out: dict[str, dict] = {}
    seen_benchmarks: dict[str, str] = {}
    for _, row in df.iterrows():
        fid = row.get("id")
        if not isinstance(fid, str):
            continue
        display = row.get("display_name") or fid
        raw = row.get("benchmark_ids")
        bench_ids: list[str]
        if isinstance(raw, list):
            bench_ids = [str(b) for b in raw]
        elif isinstance(raw, str) and raw.strip() and raw.strip() not in ("[]", "null"):
            try:
                decoded = _json.loads(raw)
                bench_ids = [str(b) for b in decoded] if isinstance(decoded, list) else []
            except (ValueError, TypeError):
                bench_ids = []
        else:
            bench_ids = []
        for bid in bench_ids:
            prior = seen_benchmarks.get(bid)
            if prior is not None and prior != fid:
                raise ValueError(
                    f"canonical_families.parquet: benchmark {bid!r} appears "
                    f"in both {prior!r} and {fid!r}."
                )
            seen_benchmarks[bid] = fid
        out[fid] = {"display": str(display), "benchmarks": bench_ids}
    return out


def load_slice_promotions_from_registry(registry_root: Path) -> set[str] | None:
    """Read slice_promotions from `<registry_root>/slice_overrides.yaml`
    (shipped alongside the parquets by the registry's publish script).
    Returns None when the file isn't there — caller falls back to the
    seed-dir YAML loader."""
    path = registry_root / "slice_overrides.yaml"
    if not path.exists():
        return None
    return load_slice_promotions(registry_root)


def load_and_materialise(
    con,
    registry_root: Path | None,
    seed_dir_override: Path | None = None,
) -> tuple[dict[str, dict], dict[str, dict], set[str]]:
    """One-shot helper: resolve sources, load curation, materialise tables.

    Source preference (parquet → YAML fallback): see module docstring.

    Returns the parsed data so Stage A's slice-grouping pass can see the
    promotion set without re-reading.
    """
    composites: dict[str, dict] | None = None
    families: dict[str, dict] | None = None
    promotions: set[str] | None = None

    # 1. Try the registry data cache first (canonical_*.parquet +
    #    YAML overrides shipped alongside).
    if registry_root is not None and registry_root.exists():
        composites = load_composites_from_parquet(registry_root)
        families = load_families_from_parquet(registry_root)
        promotions = load_slice_promotions_from_registry(registry_root)
        loaded_from = []
        if composites is not None:
            loaded_from.append("canonical_composites.parquet")
        if families is not None:
            loaded_from.append("canonical_families.parquet")
        if promotions is not None:
            loaded_from.append("slice_overrides.yaml")
        if loaded_from:
            log.info(
                "taxonomy: loaded from registry cache %s — %s",
                registry_root, ", ".join(loaded_from),
            )

    # 2. YAML fallback for any source the registry cache didn't have.
    needs_yaml = composites is None or families is None or promotions is None
    if needs_yaml:
        seed_dir = resolve_seed_dir(registry_root, seed_dir_override)
        if seed_dir is None:
            raise RuntimeError(
                "taxonomy: no taxonomy source found. Tried registry cache "
                f"({registry_root!r}) for canonical_*.parquet and "
                f"slice_overrides.yaml; tried seed dir resolution for fallback "
                f"YAMLs (override={seed_dir_override!r}). The producer needs at "
                f"least one of: a registry data snapshot with the new "
                f"canonical_families/canonical_composites tables, OR a sibling "
                f"eval-card-registry checkout with seed/, OR an explicit "
                f"--taxonomy-seed-dir / EVALCARD_REGISTRY_SEED_DIR override. "
                f"Without curated taxonomy, composite_slug falls back to "
                f"kebab-case(source_config) and silently splits multi-config "
                f"leaderboards (Vals.ai, RewardBench, WASP)."
            )
        if composites is None:
            composites = load_composites(seed_dir)
            log.info("taxonomy: loaded composites from YAML at %s", seed_dir)
        if families is None:
            families = load_families(seed_dir)
            log.info("taxonomy: loaded families from YAML at %s", seed_dir)
        if promotions is None:
            promotions = load_slice_promotions(seed_dir)
            log.info("taxonomy: loaded slice_promotions from YAML at %s", seed_dir)

    # composites / families / promotions are guaranteed non-None here
    # (parquet paths return {} or set() for empty rather than None;
    # YAML paths similarly default to empty containers).
    composites = composites or {}
    families = families or {}
    promotions = promotions or set()

    log.info(
        "taxonomy: %d composite(s), %d curated family(ies), "
        "%d slice promotion(s)",
        len(composites), len(families), len(promotions),
    )
    materialise_taxonomy_tables(con, composites, families, promotions)
    return composites, families, promotions

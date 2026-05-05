"""Migrate notes/local_alias_additions.yaml → registry's seed/_overrides/models.yaml.

**Strategy: append, don't rewrite.** The existing override file has curated
section comments that yaml.safe_dump would destroy. Instead we append our
entries at the end of the `entries:` list as a clearly-marked spike section.
The seed CLI's `_load_models_merged()` uses last-write-wins on id collision,
so any append-time entry with the same id as an earlier entry replaces it.

For each `extend: <canonical_id>: [aliases]`:
  - Construct a full entry: existing canonical fields (from main or
    existing override) PLUS the merged alias list (existing + ours).
  - Append at the end. The merge in CLI ensures it wins.

For each `new:` entry:
  - Append at the end as-is.

Idempotent: re-running just appends a new spike section. Safe to run multiple
times (the latest section wins). To clean up, delete the spike section by
hand and re-run.

Run from pipeline repo root:
  uv run --with pyyaml --no-project python scripts/migrate_aliases_to_registry.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

REGISTRY = Path("/Users/jchim/projects/evaleval/evalcard-registry")
SEED_MAIN = REGISTRY / "seed" / "models.yaml"
SEED_OV = REGISTRY / "seed" / "_overrides" / "models.yaml"
LOCAL_ADDS = Path("notes/local_alias_additions.yaml")


SPIKE_HEADER = """
  # ============================================================
  # Provenance investigation spike additions (2026-05-01)
  #
  # Migrated from eval_cards_backend_pipeline/notes/local_alias_additions.yaml.
  # Each entry below either (a) repeats an id from earlier in this file with
  # an extended alias list, or (b) introduces a brand-new canonical. The seed
  # CLI's _load_models_merged() uses last-write-wins on id collision, so
  # entries here win over earlier ones in this file and over main models.yaml.
  # Re-running scripts/migrate_aliases_to_registry.py replaces this whole
  # section idempotently.
  # ============================================================
"""

SPIKE_BEGIN_MARKER = "# === SPIKE-BEGIN: do not hand-edit ==="
SPIKE_END_MARKER = "# === SPIKE-END ==="


def main() -> None:
    main_models = yaml.safe_load(SEED_MAIN.read_text()) or []
    main_by_id: dict[str, dict] = {e["id"]: e for e in main_models}

    ov_text = SEED_OV.read_text()
    ov_doc = yaml.safe_load(ov_text) or {}
    ov_entries = ov_doc.get("entries") or []
    ov_by_id: dict[str, dict] = {e["id"]: e for e in ov_entries}

    local_doc = yaml.safe_load(LOCAL_ADDS.read_text()) or {}
    local_extend = local_doc.get("extend") or {}
    local_new = local_doc.get("new") or []

    print(f"main models.yaml: {len(main_models)} entries")
    print(f"_overrides/models.yaml: {len(ov_entries)} entries")
    print(f"local_alias_additions.yaml: {len(local_extend)} extends, {len(local_new)} news")
    print()

    # ---- Build spike entries ----
    # For each `extend`, build a full entry by merging existing aliases.
    spike_entries: list[dict] = []
    extends_unmatched: list[str] = []

    for canonical_id, extra_aliases in local_extend.items():
        if not extra_aliases:
            continue
        existing_ov = ov_by_id.get(canonical_id)
        existing_main = main_by_id.get(canonical_id)
        source = existing_ov or existing_main
        if source is None:
            extends_unmatched.append(canonical_id)
            continue
        # Construct a full entry by copying the source and merging aliases
        merged = dict(source)
        existing_aliases = list(merged.get("aliases") or [])
        seen_lc = {a.lower() for a in existing_aliases if a}
        for a in extra_aliases:
            if a and a.lower() not in seen_lc:
                existing_aliases.append(a)
                seen_lc.add(a.lower())
        merged["aliases"] = existing_aliases
        spike_entries.append(merged)

    # `new:` entries — append as-is. If the id exists in main or override,
    # the spike entry will win via last-write semantics in the CLI loader.
    spike_entries.extend(local_new)

    if extends_unmatched:
        print(f"WARN: {len(extends_unmatched)} extend ids not found in main or overrides:")
        for x in extends_unmatched:
            print(f"  {x}")

    print(f"\nGenerated {len(spike_entries)} spike entries to append.")

    # ---- Strip any prior spike section ----
    if SPIKE_BEGIN_MARKER in ov_text:
        before, _, rest = ov_text.partition(SPIKE_BEGIN_MARKER)
        _, _, after = rest.partition(SPIKE_END_MARKER)
        ov_text = before.rstrip() + "\n" + after.lstrip()
        print("(replacing prior spike section)")

    # ---- Append the new spike section ----
    # Render entries inside the existing top-level `entries:` list
    spike_block_lines = [
        f"  {SPIKE_BEGIN_MARKER}",
        SPIKE_HEADER.rstrip(),
    ]
    for entry in spike_entries:
        # Render each entry as a list item under entries:. Indent by 2 spaces
        # to match the existing list level (entries: at root, items at 2-space).
        rendered = yaml.safe_dump(
            [entry], sort_keys=False, allow_unicode=True, width=200, default_flow_style=False
        )
        # yaml.safe_dump writes "- key: value\n  key2: value2\n..." — already
        # at the right indent for a list item. We need to add 2 leading spaces
        # to align with the existing "  - id:" style in the file.
        for line in rendered.rstrip().splitlines():
            spike_block_lines.append("  " + line)
        spike_block_lines.append("")  # blank line between entries
    spike_block_lines.append(f"  {SPIKE_END_MARKER}")
    spike_block = "\n".join(spike_block_lines) + "\n"

    new_text = ov_text.rstrip() + "\n\n" + spike_block
    SEED_OV.write_text(new_text)
    print(f"\nWrote {SEED_OV}")
    print(f"  spike entries appended: {len(spike_entries)}")
    print(f"  total override entries (after CLI merge): unchanged + {len(spike_entries)} new wins")


if __name__ == "__main__":
    main()

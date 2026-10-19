# CLAUDE.md

Guidance for coding agents working in `eval_cards_backend_pipeline`. The
README covers install, run, and stage caching; this file covers how work is
checked and the rules that past mistakes taught us.

## Checks

Every check runs the same way locally and in CI.

```bash
uv run pre-commit install                      # once per clone
uv run pre-commit run --all-files              # ruff and file checks over the tree; hygiene over staged lines
python3 scripts/check_code_hygiene.py --diff main...HEAD   # hygiene over a commit range
HF_HUB_OFFLINE=1 uv run pytest tests/ -q --timeout=120
```

`.github/workflows/ci.yml` runs the hooks, the hygiene gate over the added
lines of the pull request or the push, and the tests on every pull request
and push to main. Its dependency setup is copied from `sync.yml` and a test
fails if the two drift.

Pipeline output verification is separate from the test suite: the daily sync
run compares the new snapshot with the published one (`scripts/ci_publish_warehouse.py`)
and `snapshot_meta.json` records row counts, Stage A drops and per-config
dedup counts (`dedup_by_config`). Treat a change in those numbers as something to explain, not
something to accept.

## Rules and what enforces them

Each rule exists because the mistake happened at least twice. A rule with
nothing enforcing it is a known gap; when you hit one, fix it at the highest
level that works (architecture, then types, then a lint or CI check, then a
test) rather than adding more prose here.

| Rule | Enforced by |
| --- | --- |
| Comments state the rule in place. No spec sections, phase or session tags, no "previously did X" tombstones. A false positive is marked inline with `hygiene: ok`. | `scripts/check_code_hygiene.py` (pre-commit and CI) |
| Committed code never cites a gitignored path. `notes/` is deprecated and stays ignored. | `check_code_hygiene.py` rule `ignored-path-ref` |
| Data hotfixes live in `canonicalise/resolution_hotfixes.py`, `canonicalise/hierarchy_hotfixes.py` or `metric_meta_hotfix.py`, each with an upstream pointer and a "Remove when" condition. Existing exceptions are grandfathered with a content hash: the list may only shrink and the tables do not grow. | `tests/test_hotfix_lifecycle.py` |
| A cached stage table that changes shape bumps `CACHE_SCHEMA_VERSION`. Regenerate the fingerprint with `UPDATE_CACHE_FINGERPRINT=1 HF_HUB_OFFLINE=1 uv run pytest tests/test_cache_schema_fingerprint.py -q`. | `tests/test_cache_schema_fingerprint.py` |
| Two bakes of the same inputs produce byte-identical outputs. Aggregations are ordered or provably exact. | `tests/test_double_build_determinism.py` |
| Published counts are distinct counts, never sums over a view that repeats rows. | `tests/test_count_grain.py` |
| Every gate (`--check` script, `assert_*`, pre-flight) has a test that feeds it bad input and asserts the refusal. A check never seen failing does not ship. | `tests/test_derive_scoring_modes.py`, `tests/test_ci_publish_warehouse.py`, `tests/test_check_flat_pin.py`, `tests/test_preflight.py`, `tests/test_collections.py` |
| Test fixtures never reach the network. A directory holding the corpus but no record of a download is hand-built: never fetched into, never deleted. | `sources/_revision_cache.is_hand_built`, used by all three loaders; `tests/test_fixtures_offline.py` |
| PR CI and the sync workflow install dependencies identically. | `tests/test_ci_workflow.py` |
| Minimal edits. Fix the cause in the same change; follow-ups only when blocked by a third party. | judgment |
| Prod data flips to the Space are manual. Hand over exact values; never trigger a rebuild. | judgment |

## Working here

- Push strategy varies; confirm before the first push of a series.
- Stage output and view-layer column names are a contract with the frontend
  (`general-eval-card`). Changing one means changing both, with the TS
  interfaces updated in the same series.
- Hotfixes are being refactored away. Do not add to the pattern; make an
  existing one unnecessary when you can.

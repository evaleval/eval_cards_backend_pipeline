"""Stage A cards loading tests.

Card collisions happen when two card files (`card_key`s) resolve through
the registry to the same canonical `benchmark_id`. Examples in production:
  - `mmlu-pro.json` accidentally aliased to `mmlu` in the alias table
  - `swe-bench-verified.json` and `swebench_verified.json` shipping
    side-by-side mid-rename
  - typo'd filename fuzzy-matched to a real benchmark

Stage A dedupes (first-by-card_key wins) so the LEFT JOIN in Stage D
doesn't fan out fact rows. The WARN tells the operator content was
silently dropped from the benchmark dim.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import duckdb
import pytest

from eval_card_backend.canonicalise import stages
from eval_card_backend.canonicalise.resolver_setup import register_udfs


class _FixedMappingResolver:
    """Resolver double that maps any card_key from `mapping` to a fixed
    canonical_id. Unmapped raws return no_match."""

    def __init__(self, mapping):
        self._m = mapping

    def resolve(self, raw, entity_type, source_config=None):
        canonical = self._m.get(raw)
        return SimpleNamespace(
            canonical_id=canonical,
            strategy="exact" if canonical else "no_match",
        )


def _con_with_resolver(mapping):
    con = duckdb.connect()
    register_udfs(con, _FixedMappingResolver(mapping))
    return con


def test_card_collision_warns_and_first_by_key_wins(caplog):
    """Two cards resolving to the same canonical benchmark_id:
      - cards_resolved retains both rows so an operator can inspect them
      - cards_raw retains one (first-by-card_key alphabetically)
      - WARN log surfaces the collision count
    """
    con = _con_with_resolver({
        "mmlu":       "mmlu",
        "mmlu_alias": "mmlu",
    })
    cards = {
        "mmlu": {
            "benchmark_details": {"name": "MMLU (canonical card)"},
        },
        "mmlu_alias": {
            "benchmark_details": {"name": "MMLU (aliased duplicate)"},
        },
    }

    with caplog.at_level(logging.WARNING, logger="eval_card_backend.canonicalise.stages"):
        n = stages.stage_a_load_cards(con, cards)

    # Both keys resolved → cards_resolved has both rows
    resolved_count = con.execute(
        "SELECT COUNT(*) FROM cards_resolved WHERE benchmark_id = 'mmlu'"
    ).fetchone()[0]
    assert resolved_count == 2

    # Dedup keeps one — first-by-card_key alphabetically (mmlu < mmlu_alias)
    raw_count = con.execute(
        "SELECT COUNT(*) FROM cards_raw WHERE benchmark_id = 'mmlu'"
    ).fetchone()[0]
    assert raw_count == 1

    # The winning row is mmlu (lower card_key wins lexicographically)
    winner = con.execute(
        "SELECT card_key FROM cards_raw WHERE benchmark_id = 'mmlu'"
    ).fetchone()[0]
    assert winner == "mmlu"

    # WARN log fired with the collision count and the count is correct
    collision_warns = [
        r for r in caplog.records
        if "had multiple cards resolve" in r.message
    ]
    assert len(collision_warns) == 1, "expected exactly one collision WARN"
    assert "1 benchmark_id" in collision_warns[0].message

    # Return value reflects the deduped count (not cards_resolved's count).
    # Both keys resolved → 1 winner + 0 orphans = 1 row in cards_raw.
    assert n == 1

    con.close()


def test_no_collision_no_warn(caplog):
    """Sanity: when card_keys resolve to distinct benchmark_ids, no WARN."""
    con = _con_with_resolver({
        "mmlu":             "mmlu",
        "swebench-verified": "swebench-verified",
    })
    cards = {
        "mmlu": {"benchmark_details": {"name": "MMLU"}},
        "swebench-verified": {"benchmark_details": {"name": "SWE-bench Verified"}},
    }

    with caplog.at_level(logging.WARNING, logger="eval_card_backend.canonicalise.stages"):
        stages.stage_a_load_cards(con, cards)

    assert not [
        r for r in caplog.records
        if "had multiple cards resolve" in r.message
    ]
    con.close()


def test_orphan_cards_preserved_no_warn(caplog):
    """Cards whose key doesn't resolve are kept as orphans (benchmark_id
    NULL) — they don't trigger the collision warn."""
    con = _con_with_resolver({"mmlu": "mmlu"})  # 'unknown' won't resolve
    cards = {
        "mmlu": {"benchmark_details": {"name": "MMLU"}},
        "unknown_benchmark": {"benchmark_details": {"name": "Unknown"}},
    }

    with caplog.at_level(logging.WARNING, logger="eval_card_backend.canonicalise.stages"):
        n = stages.stage_a_load_cards(con, cards)

    # cards_raw: 1 resolved (mmlu) + 1 orphan (unknown_benchmark with NULL bid).
    assert n == 2

    orphan_count = con.execute(
        "SELECT COUNT(*) FROM cards_raw WHERE benchmark_id IS NULL"
    ).fetchone()[0]
    assert orphan_count == 1

    # No collision WARN — distinct benchmark_ids (mmlu and NULL).
    assert not [
        r for r in caplog.records
        if "had multiple cards resolve" in r.message
    ]
    con.close()

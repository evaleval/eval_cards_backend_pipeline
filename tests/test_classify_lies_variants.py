"""Unit tests covering the fibble lies-variant rollup.

EEE upstream packs the five fibble lies-variants under one benchmark
`fibble_arena` with `evaluation_name` shaped like `fibble_arena_<N>lie(s)`.
The pipeline's per-record `lookup_benchmark_card` used to miss for fibble
(the ABC card's name is `fibble_arena_daily`, so neither `fibble_arena`
nor its family_key matched), so `top_level_benchmark_owns_slices`
returned False and each variant ended up as its own leaf eval — users
saw one card per "lie" instead of one Fibble Arena card with five
slices.

The fix at `pipeline.py:3947` keeps the existing direct lookup, then —
**only when the direct lookup misses** — falls back to a
`shared_dataset_name` drawn from `evaluation_results[*].source_data`.
Two guards prevent misfires on aggregator records:
  1. The dataset_name must be shared across **every** result in the
     evaluation (skips aggregator records like `llm-stats` whose results
     scatter across many distinct scraped benchmarks).
  2. The dataset_name must be compact-key related to the EEE benchmark
     (substring either direction). `fibble_arena` ⊂ `fibble_arena_daily`
     ✓; `llmstats` vs `arcagiv2` ✗.
Once the card resolves, classify_evaluation_result takes the existing
`top_level_benchmark_owns_slices` branch and emits the leaf as the
family with the lies-variant carried as a slice.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline import (
    canonical_benchmark_family_key,
    classify_evaluation_result,
    load_benchmark_metadata,
    lookup_benchmark_card,
)

ABC_CACHE_DIR = REPO_ROOT / ".cache" / "auto_benchmarkcards"


def _fibble_record(evaluation_name: str) -> tuple[dict, dict]:
    evaluation = {"benchmark": "fibble_arena"}
    result = {
        "evaluation_name": evaluation_name,
        "metric_config": {"metric_name": "Win Rate", "metric_id": "win_rate"},
        "source_data": {"dataset_name": "fibble_arena_daily"},
    }
    return evaluation, result


LIES_VARIANTS = (
    "fibble_arena_1lie",
    "fibble_arena_2lies",
    "fibble_arena_3lies",
    "fibble_arena_4lies",
    "fibble_arena_5lies",
)


def test_per_evaluation_lookup_finds_fibble_card_via_dataset_name():
    # The fix: pipeline.py's per-evaluation lookup_benchmark_card now
    # passes `first_dataset_name` as a fourth value, so the fibble card
    # (named `fibble_arena_daily`) resolves at classify time. Without
    # dataset_name in the candidate set, the lookup misses.
    if not ABC_CACHE_DIR.exists():
        import pytest

        pytest.skip("ABC cache not present; run pipeline once to populate it")
    _, metadata_lookup, _ = load_benchmark_metadata(str(ABC_CACHE_DIR))

    benchmark = "fibble_arena"
    family_key = canonical_benchmark_family_key(benchmark)

    miss = lookup_benchmark_card(metadata_lookup, benchmark, family_key)
    hit = lookup_benchmark_card(
        metadata_lookup, benchmark, family_key, "fibble_arena_daily"
    )

    assert miss is None, (
        "regression guard: without dataset_name the per-evaluation lookup "
        "still misses the fibble card — confirming the asymmetry the fix "
        "addresses"
    )
    assert hit is not None
    assert hit["benchmark_details"]["name"] == "fibble_arena_daily"


def test_lies_variants_collapse_to_family_with_card_resolved():
    # End-to-end through classify with the actual ABC card, mimicking
    # what pipeline.py:3931 now produces for fibble.
    if not ABC_CACHE_DIR.exists():
        import pytest

        pytest.skip("ABC cache not present; run pipeline once to populate it")
    _, metadata_lookup, _ = load_benchmark_metadata(str(ABC_CACHE_DIR))
    card = lookup_benchmark_card(
        metadata_lookup,
        "fibble_arena",
        canonical_benchmark_family_key("fibble_arena"),
        "fibble_arena_daily",
    )
    assert card is not None, "card prerequisite missing"

    for raw in LIES_VARIANTS:
        evaluation, result = _fibble_record(raw)
        normalized = classify_evaluation_result(evaluation, result, card)
        assert normalized["benchmark_family_key"] == "fibble_arena", raw
        assert normalized["benchmark_parent_key"] == "fibble_arena", raw
        # All five variants must share one leaf — the family — so they
        # roll into a single Fibble Arena eval card.
        assert normalized["benchmark_leaf_key"] == "fibble_arena", raw
        # The lies-variant becomes the slice (subtask) discriminator.
        assert normalized["slice_key"] == raw, raw


def test_classify_falls_through_when_card_missing():
    # Pin the pre-fix fall-through behavior so a future regression to
    # the lookup path is caught loudly. If `benchmark_card=None`,
    # classify still produces per-leaf output (each variant its own
    # leaf) — that is the bug; the fix lives upstream in the per-record
    # lookup at pipeline.py:3931, not inside classify itself.
    evaluation, result = _fibble_record("fibble_arena_4lies")
    normalized = classify_evaluation_result(evaluation, result, None)
    assert normalized["benchmark_leaf_key"] == "fibble_arena_4lies"
    assert normalized["slice_key"] is None


def test_aggregator_with_single_dataset_does_not_collapse():
    # Edge case caught by the diff audit: some aggregator records
    # (`llm-stats`) only carry results for ONE scraped benchmark
    # (e.g. only `ARC-AGI v2`). With just the "all results share one
    # dataset_name" guard, those records would still trigger the
    # collapse — `dataset_name="ARC-AGI v2"` would resolve to ARC-AGI's
    # card, flipping the result onto the `owns_slices` path and folding
    # arc_agi_v2's models into a synthetic `llm_stats` rollup.
    #
    # The compact-key relatedness guard catches this: `llmstats` vs
    # `arcagiv2` are unrelated → don't pass dataset_name to the lookup
    # → benchmark_card stays None → ARC-AGI keeps its own leaf under
    # the `llm_stats` parent.
    if not ABC_CACHE_DIR.exists():
        import pytest

        pytest.skip("ABC cache not present; run pipeline once to populate it")
    _, metadata_lookup, _ = load_benchmark_metadata(str(ABC_CACHE_DIR))
    from scripts.pipeline import compact_benchmark_key, lookup_benchmark_card

    eee_benchmark = "llm-stats"
    shared_ds = "ARC-AGI v2"
    eee_compact = compact_benchmark_key(eee_benchmark)
    ds_compact = compact_benchmark_key(shared_ds)
    related = eee_compact and ds_compact and (
        eee_compact in ds_compact or ds_compact in eee_compact
    )
    assert not related, (
        "compact-key relatedness must reject unrelated aggregator/dataset "
        "pairings — fix's correctness depends on this discriminator"
    )

    # Mirror the exact lookup the pipeline does when relatedness fails
    # (lookup_dataset_name=None).
    card = lookup_benchmark_card(
        metadata_lookup,
        eee_benchmark,
        canonical_benchmark_family_key(eee_benchmark),
        None,
    )
    assert card is None, (
        "with relatedness guard rejecting the dataset_name, the lookup "
        "must miss — preserving the existing per-benchmark leaf shape"
    )


def test_fibble_relatedness_passes():
    # Counterpart to the aggregator test: fibble's compact-key
    # relatedness MUST hold so the fix actually fires for the bug it
    # targets.
    from scripts.pipeline import compact_benchmark_key

    eee_compact = compact_benchmark_key("fibble_arena")
    ds_compact = compact_benchmark_key("fibble_arena_daily")
    assert eee_compact in ds_compact, (
        "fibble must satisfy the relatedness guard or the fix never fires"
    )


def test_aggregator_record_keeps_per_benchmark_leaves():
    # Regression guard for the llm-stats class of aggregator records:
    # ONE EEE benchmark (`llm-stats`) wraps MANY scraped benchmarks
    # (MMLU, GPQA, BBH, …) — each result has its own dataset_name. The
    # per-evaluation lookup at pipeline.py:3931 must NOT pass any one of
    # those dataset_names as a card-resolution signal, because it would
    # incorrectly resolve a card describing only one of the wrapped
    # benchmarks (e.g. MMLU) and flip every other result in the record
    # (e.g. GPQA, BBH) to the `top_level_benchmark_owns_slices` path —
    # collapsing them all into a single misleading "llm_stats" leaf and
    # losing each benchmark's standalone card. The fix uses
    # `shared_dataset_name` (only set when all results agree on one
    # dataset_name) precisely to skip aggregators.
    #
    # We assert the post-classify shape for an llm-stats result with
    # `benchmark_card=None` (which it must remain, because
    # `shared_dataset_name` is None when results scatter): each scraped
    # benchmark stays its own leaf, parented to `llm_stats`.
    evaluation = {"benchmark": "llm-stats"}
    result = {
        "evaluation_name": "llm_stats.mmlu",
        "metric_config": {
            "metric_name": "Score",
            "metric_id": "llm_stats.mmlu.score",
        },
        "source_data": {"dataset_name": "MMLU"},
    }
    normalized = classify_evaluation_result(evaluation, result, None)
    assert normalized["benchmark_parent_key"] == "llm_stats"
    # MMLU stays a leaf (not a slice of llm_stats) — preserves the
    # per-benchmark eval card.
    assert normalized["benchmark_leaf_key"] == "mmlu"
    assert normalized["slice_key"] is None

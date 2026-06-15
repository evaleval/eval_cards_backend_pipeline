"""Regression tests for config-aware slice-promotion resolution.

`compute_overrides` buckets EEE rows by folder and, for "promoted"
buckets (≥2 slice candidates that resolve to canonicals), overrides each
row's `benchmark_id` with the slice candidate's canonical. The alias
lookup that resolves those slice candidates MUST be config-scoped:
the registry scopes aliases per `source_config`, so the same surface form
(e.g. "Investment Banking") can map to different canonicals in different
EEE folders (apex-agents vs apex-v1). A config-blind global map would
pick whichever scoped row it happened to hold last, leaking apex-v1 into
the apex-agents composite (the deployed-snapshot bug this guards).

Mirrors the resolver's `AliasStore.lookup` precedence: config-scoped
before global; a surface form with only a global (unscoped) alias must
resolve identically regardless of the bucket's folder.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise import slice_promotion


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def _make_canonical_benchmarks(con, rows):
    """rows: list of (id, parent_benchmark_id)."""
    con.execute(
        "CREATE TABLE canonical_benchmarks ("
        "  id VARCHAR, parent_benchmark_id VARCHAR"
        ")"
    )
    for cid, pid in rows:
        con.execute(
            "INSERT INTO canonical_benchmarks VALUES (?, ?)", [cid, pid]
        )


def _make_aliases(con, rows):
    """rows: list of (raw_value, canonical_id, entity_type, status,
    source_config). source_config=None means a global/unscoped alias."""
    con.execute(
        "CREATE TABLE aliases ("
        "  raw_value VARCHAR, canonical_id VARCHAR, entity_type VARCHAR, "
        "  status VARCHAR, source_config VARCHAR"
        ")"
    )
    for raw, cid, et, status, sc in rows:
        con.execute(
            "INSERT INTO aliases VALUES (?, ?, ?, ?, ?)",
            [raw, cid, et, status, sc],
        )


def _make_results_exploded(con, rows):
    """rows: list of (evaluation_id, result_idx, source_config, dataset_name,
    evaluation_name) or (..., evaluation_result_id) as a 6th element.

    When evaluation_result_id is omitted it is synthesised as
    `<evaluation_id>#<result_idx>` (the same fallback the pipeline uses),
    which is unique whenever each input tuple has a distinct
    (evaluation_id, result_idx) — true for every test below except the
    explicit collision case, which passes its own ids.
    """
    con.execute(
        "CREATE TABLE results_exploded ("
        "  evaluation_id VARCHAR, result_idx INTEGER, source_config VARCHAR, "
        "  source_data VARCHAR, evaluation_name VARCHAR, "
        "  evaluation_result_id VARCHAR"
        ")"
    )
    import json
    for row in rows:
        if len(row) == 6:
            eid, ridx, sc, ds, ename, erid = row
        else:
            eid, ridx, sc, ds, ename = row
            erid = f"{eid}#{ridx}"
        sd = json.dumps({"dataset_name": ds}) if ds is not None else None
        con.execute(
            "INSERT INTO results_exploded VALUES (?, ?, ?, ?, ?, ?)",
            [eid, ridx, sc, sd, ename, erid],
        )


def _overrides(con):
    """Map evaluation_result_id -> canonical_id from the override table."""
    return {
        erid: cid
        for erid, cid in con.execute(
            "SELECT evaluation_result_id, canonical_id "
            "FROM slice_promotion_overrides"
        ).fetchall()
    }


def _ov_by_key(con):
    """Convenience: map the synthesised `<evaluation_id>#<result_idx>` key
    back to canonical_id for tests that key rows by (evaluation_id,
    result_idx). Only valid when evaluation_result_id was synthesised."""
    ov = _overrides(con)
    return {tuple(k.rsplit("#", 1)): v for k, v in ov.items()}


def test_scoped_alias_resolves_per_bucket_source_config(con):
    """"Investment Banking" is scoped to apex-agents under the apex-agents
    folder and to apex-v1 under the apex-v1 folder (two scoped rows, no
    global one). A promoted bucket in the apex-agents folder must override
    to apex-agents; the apex-v1 folder to apex-v1. Config-blind resolution
    would flip the apex-agents rows to apex-v1 (the deployed-snapshot bug).
    """
    _make_canonical_benchmarks(con, [
        ("apex-agents", None),
        ("apex-v1", None),
        ("corporate-lawyer", "apex-agents"),
        ("investment-banking-v1", "apex-v1"),
    ])
    _make_aliases(con, [
        # Two scoped rows for the same surface form, different canonicals.
        ("Investment Banking", "apex-agents", "benchmark", "confirmed", "apex-agents"),
        ("Investment Banking", "apex-v1", "benchmark", "confirmed", "apex-v1"),
        # A second distinct slice in each folder so each bucket promotes
        # (needs >=2 resolving slice candidates).
        ("Corporate Lawyer", "corporate-lawyer", "benchmark", "confirmed", "apex-agents"),
        ("Investment Banking V1", "investment-banking-v1", "benchmark", "confirmed", "apex-v1"),
        # Folder names themselves resolve to the composite canonical.
        ("apex-agents", "apex-agents", "benchmark", "confirmed", None),
        ("apex-v1", "apex-v1", "benchmark", "confirmed", None),
    ])
    _make_results_exploded(con, [
        # apex-agents folder
        ("ev_a1", 0, "apex-agents", "apex-agents", "Investment Banking"),
        ("ev_a2", 0, "apex-agents", "apex-agents", "Corporate Lawyer"),
        # apex-v1 folder
        ("ev_v1", 0, "apex-v1", "apex-v1", "Investment Banking"),
        ("ev_v2", 0, "apex-v1", "apex-v1", "Investment Banking V1"),
    ])

    n = slice_promotion.compute_overrides(con)
    assert n > 0
    ov = _overrides(con)

    # The crux: the apex-agents "Investment Banking" row resolves to
    # apex-agents, NOT apex-v1.
    assert ov["ev_a1#0"] == "apex-agents"
    # The apex-v1 "Investment Banking" row resolves to apex-v1.
    assert ov["ev_v1#0"] == "apex-v1"


def test_global_only_slice_candidate_unaffected_by_folder(con):
    """A surface form with ONLY a global (unscoped) alias must resolve to
    the same canonical regardless of the bucket's folder — the fix must
    not change behaviour for global-only aliases (no regression)."""
    _make_canonical_benchmarks(con, [
        ("aa-lcr", None),
        ("aa-other", None),
    ])
    _make_aliases(con, [
        # Global-only aliases — no source_config.
        ("LCR", "aa-lcr", "benchmark", "confirmed", None),
        ("Other Slice", "aa-other", "benchmark", "confirmed", None),
    ])
    _make_results_exploded(con, [
        # Same global surface forms appear under two different folders.
        ("ev_f1a", 0, "folder-one", "folder-one", "LCR"),
        ("ev_f1b", 0, "folder-one", "folder-one", "Other Slice"),
        ("ev_f2a", 0, "folder-two", "folder-two", "LCR"),
        ("ev_f2b", 0, "folder-two", "folder-two", "Other Slice"),
    ])

    slice_promotion.compute_overrides(con)
    ov = _overrides(con)

    # "LCR" resolves to aa-lcr in both folders — folder-independent.
    assert ov["ev_f1a#0"] == "aa-lcr"
    assert ov["ev_f2a#0"] == "aa-lcr"


def test_scoped_alias_does_not_leak_into_other_folder(con):
    """A slice candidate scoped only to folder-A must NOT resolve when the
    bucket is folder-B and no global alias exists — scoped aliases don't
    leak across configs (mirrors AliasStore: scoped-then-global only)."""
    _make_canonical_benchmarks(con, [
        ("bench-x", None),
        ("bench-y", None),
        ("bench-z", None),
    ])
    _make_aliases(con, [
        # "Special" only scoped to folder-a.
        ("Special", "bench-x", "benchmark", "confirmed", "folder-a"),
        ("Another", "bench-y", "benchmark", "confirmed", "folder-a"),
        # folder-b only has a global slice (so the bucket needs another
        # resolving candidate to promote — give it one global).
        ("Plain", "bench-z", "benchmark", "confirmed", None),
        ("Another", "bench-y", "benchmark", "confirmed", None),
    ])
    _make_results_exploded(con, [
        # folder-a: "Special" + "Another" both resolve (scoped) -> promoted
        ("ev_pa1", 0, "folder-a", "folder-a", "Special"),
        ("ev_pa2", 0, "folder-a", "folder-a", "Another"),
        # folder-b: "Special" has no global + no folder-b scope -> must NOT
        # override; "Plain"/"Another" resolve via global.
        ("ev_pb1", 0, "folder-b", "folder-b", "Special"),
        ("ev_pb2", 0, "folder-b", "folder-b", "Plain"),
        ("ev_pb3", 0, "folder-b", "folder-b", "Another"),
    ])

    slice_promotion.compute_overrides(con)
    ov = _overrides(con)

    # folder-a "Special" resolves via its scoped alias.
    assert ov.get("ev_pa1#0") == "bench-x"
    # folder-b "Special" does NOT resolve to bench-x (no leak). It may fall
    # back to the bucket's ds_canonical if folder-b resolves as a dataset,
    # but it must never be bench-x.
    assert ov.get("ev_pb1#0") != "bench-x"


def test_evaluation_result_id_collision_keeps_distinct_overrides(con):
    """Two distinct physical EEE records can share one (evaluation_id,
    result_idx) — and therefore one fact_id — while each carries its own
    evaluation_result_id and its own evaluation_name (the deployed
    LiveBench case: one evaluation_id, two source records, each with its
    own evaluation_results[] array landing on the same result_idx).

    Overrides MUST be keyed by evaluation_result_id, not (evaluation_id,
    result_idx): otherwise the two records emit two override rows for the
    same key with different canonicals, and the apply-time UPDATE picks
    whichever the (unordered) scan reached last — the run-to-run flip
    this guards. With evaluation_result_id keying, each physical record
    keeps its own correct slice canonical and the result is deterministic.
    """
    _make_canonical_benchmarks(con, [
        ("livebench", None),
        ("livebench-coding", "livebench"),
        ("livebench-language", "livebench"),
    ])
    _make_aliases(con, [
        ("livebench", "livebench", "benchmark", "confirmed", None),
        # The slice candidate is the full "livebench/<slice>" surface form
        # (extract_slice_candidate returns it verbatim), so alias on that.
        ("livebench/coding", "livebench-coding", "benchmark", "confirmed", "live_bench"),
        ("livebench/language", "livebench-language", "benchmark", "confirmed", "live_bench"),
    ])
    # Same evaluation_id + result_idx, two physical records (distinct
    # evaluation_result_id), pointing at different slices.
    eid = "livebench/m/1777649819.489959"
    _make_results_exploded(con, [
        (eid, 0, "live_bench", "LiveBench", "livebench/coding",
         "rec-coding/coding"),
        (eid, 0, "live_bench", "LiveBench", "livebench/language",
         "rec-language/language"),
        # A second slice per record so the bucket promotes (>=2 distinct
        # resolving slice canonicals accumulate in the shared bucket).
        (eid, 1, "live_bench", "LiveBench", "livebench/coding",
         "rec-coding/coding-2"),
        (eid, 1, "live_bench", "LiveBench", "livebench/language",
         "rec-language/language-2"),
    ])

    slice_promotion.compute_overrides(con)
    ov = _overrides(con)

    # The coding record resolves to livebench-coding; the language record
    # to livebench-language — distinct overrides for the colliding key.
    assert ov["rec-coding/coding"] == "livebench-coding"
    assert ov["rec-language/language"] == "livebench-language"
    # Determinism is re-running and getting byte-identical override rows.
    first = sorted(_overrides(con).items())
    slice_promotion.compute_overrides(con)
    assert sorted(_overrides(con).items()) == first

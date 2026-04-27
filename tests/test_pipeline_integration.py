"""End-to-end pipeline integration tests against synthetic EEE + ABC fixtures.

Each test asserts on a specific invariant of the published JSON outputs.
The fixture is constructed in `conftest.py:pipeline_output` and reused
across tests (session-scoped) so the pipeline runs once.

These tests catch integration bugs that pure-function unit tests in
`test_signals.py` cannot:
  - row-build: per-result generation_args attached to the right row
  - grouping: rows with the same model_route_id within one metric_summary
    land in one group
  - signal application: per-row vs group-level annotations attached to the
    correct surface
  - rollups: per-eval and per-model summary counts match underlying data
  - strip pass: no internal underscore fields leak to disk
  - list-view consistency: lite + non-lite views carry the same summary
    blocks
"""
import json
from pathlib import Path


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_metric_summaries(eval_summary: dict) -> list[dict]:
    """Flatten root + subtask metric_summaries for a per-eval JSON."""
    out: list[dict] = list(eval_summary.get("metrics", []))
    for subtask in eval_summary.get("subtasks", []):
        out.extend(subtask.get("metrics", []))
    return out


def _walk_rows(eval_summary: dict) -> list[dict]:
    return [
        row
        for metric in _walk_metric_summaries(eval_summary)
        for row in metric.get("model_results", [])
    ]


# ---------------------------------------------------------------------------
# Smoke: pipeline ran and produced the expected files
# ---------------------------------------------------------------------------


def test_pipeline_emits_expected_top_level_files(pipeline_output):
    expected = {
        "model-cards.json",
        "model-cards-lite.json",
        "eval-list.json",
        "eval-list-lite.json",
        "peer-ranks.json",
        "comparison-index.json",
        "benchmark-metadata.json",
        "developers.json",
        "manifest.json",
        "README.md",
    }
    actual = {p.name for p in pipeline_output.iterdir() if p.is_file()}
    assert expected <= actual, f"missing: {expected - actual}"


def test_pipeline_emits_one_eval_json_per_fixture_benchmark(pipeline_output):
    eval_files = {p.stem for p in (pipeline_output / "evals").glob("*.json")}
    # eval_summary_id is benchmark name slugified; our fixtures emit one
    # metric_summary per benchmark (single evaluation_name "accuracy"), so
    # exactly one eval JSON per benchmark.
    assert {"bench_variant", "bench_multiorg", "bench_orgcollapse", "bench_agentic"} <= eval_files


def test_pipeline_emits_one_model_json_per_fixture_model(pipeline_output):
    model_files = {p.stem for p in (pipeline_output / "models").glob("*.json")}
    assert {"openai__gpt-5", "anthropic__claude-opus-4-5"} <= model_files


# ---------------------------------------------------------------------------
# Variant divergence — bench_variant fixture (3 rows, differing max_tokens)
# ---------------------------------------------------------------------------


def test_variant_divergence_fires_when_setups_diverge(pipeline_output):
    summary = _read(pipeline_output / "evals" / "bench_variant.json")
    rows = _walk_rows(summary)
    assert len(rows) == 3, f"expected 3 rows; got {len(rows)}"
    for row in rows:
        vd = row["evalcards"]["annotations"]["variant_divergence"]
        assert vd is not None, "variant_divergence should be populated"
        assert vd["has_variant_divergence"] is True
        assert vd["triple_count_in_group"] == 3
        assert any(f["field"] == "max_tokens" for f in vd["differing_setup_fields"])


def test_variant_divergence_this_triple_score_matches_row_score(pipeline_output):
    summary = _read(pipeline_output / "evals" / "bench_variant.json")
    for row in _walk_rows(summary):
        vd = row["evalcards"]["annotations"]["variant_divergence"]
        if vd is None:
            continue
        assert vd["this_triple_score"] == row["score"], (
            "this_triple_score must equal the row's own score (per-row injection)"
        )


def test_variant_divergence_scores_in_group_lists_all_group_scores(pipeline_output):
    summary = _read(pipeline_output / "evals" / "bench_variant.json")
    rows = _walk_rows(summary)
    expected_scores = sorted(row["score"] for row in rows)
    for row in rows:
        vd = row["evalcards"]["annotations"]["variant_divergence"]
        assert sorted(vd["scores_in_group"]) == expected_scores


# ---------------------------------------------------------------------------
# Cross-party divergence — bench_multiorg (2 orgs, identical setups)
# ---------------------------------------------------------------------------


def test_cross_party_divergence_fires_for_two_distinct_orgs(pipeline_output):
    summary = _read(pipeline_output / "evals" / "bench_multiorg.json")
    rows = _walk_rows(summary)
    assert len(rows) == 2
    for row in rows:
        cpd = row["evalcards"]["annotations"]["cross_party_divergence"]
        assert cpd is not None
        assert cpd["has_cross_party_divergence"] is True
        assert cpd["organization_count"] == 2
        assert set(cpd["scores_by_organization"].keys()) == {"OpenAI", "Scale AI"}


def test_variant_divergence_null_when_setups_identical(pipeline_output):
    """bench_multiorg has identical setups → variant_divergence is null."""
    summary = _read(pipeline_output / "evals" / "bench_multiorg.json")
    for row in _walk_rows(summary):
        assert row["evalcards"]["annotations"]["variant_divergence"] is None


def test_cross_party_divergence_null_when_orgs_collapse_via_whitespace(pipeline_output):
    """bench_orgcollapse: two rows whose org names differ only in whitespace
    → normalize_org_name collapses them → cross_party returns null."""
    summary = _read(pipeline_output / "evals" / "bench_orgcollapse.json")
    rows = _walk_rows(summary)
    assert len(rows) == 2
    for row in rows:
        cpd = row["evalcards"]["annotations"]["cross_party_divergence"]
        assert cpd is None, "whitespace-only org diff should NOT trigger cross-party"
        prov = row["evalcards"]["annotations"]["provenance"]
        assert prov["is_multi_source"] is False
        assert prov["distinct_reporting_organizations"] == 1


def test_cross_party_dict_is_identical_across_group_rows(pipeline_output):
    """Per spec §6.2.5 the annotation is group-level; we duplicate it onto
    every row. Verify every row in the group carries an identical dict."""
    summary = _read(pipeline_output / "evals" / "bench_multiorg.json")
    rows = _walk_rows(summary)
    cpds = [row["evalcards"]["annotations"]["cross_party_divergence"] for row in rows]
    assert all(c == cpds[0] for c in cpds)


# ---------------------------------------------------------------------------
# Reproducibility gap — bench_agentic (agentic record missing eval_plan)
# ---------------------------------------------------------------------------


def test_reproducibility_gap_fires_for_agentic_missing_eval_plan(pipeline_output):
    summary = _read(pipeline_output / "evals" / "bench_agentic.json")
    rows = _walk_rows(summary)
    assert len(rows) == 1
    rg = rows[0]["evalcards"]["annotations"]["reproducibility_gap"]
    assert rg["has_reproducibility_gap"] is True
    assert "eval_plan" in rg["missing_fields"]
    assert "eval_limits" in rg["missing_fields"]


# ---------------------------------------------------------------------------
# Provenance — source_type maps from evaluator_relationship per row
# ---------------------------------------------------------------------------


def test_provenance_source_type_maps_evaluator_relationship(pipeline_output):
    summary = _read(pipeline_output / "evals" / "bench_multiorg.json")
    rows = _walk_rows(summary)
    by_org = {row["source_metadata"]["source_organization_name"]:
              row["evalcards"]["annotations"]["provenance"]["source_type"]
              for row in rows}
    assert by_org == {"OpenAI": "first_party", "Scale AI": "third_party"}


# ---------------------------------------------------------------------------
# Strip pass — no internal underscore fields leak
# ---------------------------------------------------------------------------


def test_no_internal_underscore_fields_leak_to_evals_json(pipeline_output):
    leaks: list[str] = []
    for path in (pipeline_output / "evals").glob("*.json"):
        summary = _read(path)
        for metric in _walk_metric_summaries(summary):
            if "_signal_groups" in metric:
                leaks.append(f"{path.name}::metric._signal_groups")
            for row in metric.get("model_results", []):
                if "_generation_args" in row:
                    leaks.append(f"{path.name}::row._generation_args")
    assert leaks == []


def test_no_internal_underscore_fields_leak_to_models_json(pipeline_output):
    leaks: list[str] = []
    for path in (pipeline_output / "models").glob("*.json"):
        model = _read(path)
        for cat_summaries in model.get("hierarchy_by_category", {}).values():
            for summary in cat_summaries:
                for metric in _walk_metric_summaries(summary):
                    if "_signal_groups" in metric:
                        leaks.append(f"{path.name}::metric._signal_groups")
                    for row in metric.get("model_results", []):
                        if "_generation_args" in row:
                            leaks.append(f"{path.name}::row._generation_args")
    assert leaks == []


# ---------------------------------------------------------------------------
# Rollup invariants
# ---------------------------------------------------------------------------


def test_per_eval_provenance_summary_counts_match_underlying_data(pipeline_output):
    """provenance_summary.total_results == row count;
    provenance_summary.total_groups == sum-over-metrics of distinct
    model_route_ids."""
    for path in (pipeline_output / "evals").glob("*.json"):
        summary = _read(path)
        ps = summary["provenance_summary"]
        rows = _walk_rows(summary)
        assert ps["total_results"] == len(rows), f"{path.name}: row mismatch"
        expected_groups = sum(
            len({row["model_route_id"] for row in metric.get("model_results", [])})
            for metric in _walk_metric_summaries(summary)
        )
        assert ps["total_groups"] == expected_groups, f"{path.name}: groups mismatch"


def test_per_eval_source_type_distribution_sums_to_total_results(pipeline_output):
    for path in (pipeline_output / "evals").glob("*.json"):
        summary = _read(path)
        ps = summary["provenance_summary"]
        assert sum(ps["source_type_distribution"].values()) == ps["total_results"], (
            f"{path.name}: distribution doesn't sum to total_results"
        )
        # Always 4 buckets, even when zero
        assert set(ps["source_type_distribution"].keys()) == {
            "first_party", "third_party", "collaborative", "unspecified"
        }


def test_per_eval_comparability_summary_counts_match_signal_attachments(pipeline_output):
    """variant_divergent_count == # groups where variant_divergence is non-null
    AND has_variant_divergence: true. Same for cross_party. Walking rows
    undercounts (same group on multiple rows) — instead enumerate groups by
    (model_route_id, metric_summary_id)."""
    for path in (pipeline_output / "evals").glob("*.json"):
        summary = _read(path)
        cs = summary["comparability_summary"]
        seen_groups: set[tuple[str, str]] = set()
        variant_divergent = 0
        cross_party_divergent = 0
        variant_eligible = 0
        cross_party_eligible = 0
        for metric in _walk_metric_summaries(summary):
            for row in metric.get("model_results", []):
                key = (row["model_route_id"], metric["metric_summary_id"])
                if key in seen_groups:
                    continue
                seen_groups.add(key)
                ann = row["evalcards"]["annotations"]
                if ann.get("variant_divergence") is not None:
                    variant_eligible += 1
                    if ann["variant_divergence"]["has_variant_divergence"]:
                        variant_divergent += 1
                if ann.get("cross_party_divergence") is not None:
                    cross_party_eligible += 1
                    if ann["cross_party_divergence"]["has_cross_party_divergence"]:
                        cross_party_divergent += 1
        assert cs["total_groups"] == len(seen_groups), f"{path.name}: total_groups"
        assert cs["groups_with_variant_check"] == variant_eligible
        assert cs["variant_divergent_count"] == variant_divergent
        assert cs["groups_with_cross_party_check"] == cross_party_eligible
        assert cs["cross_party_divergent_count"] == cross_party_divergent


def test_per_model_total_groups_equals_appearance_count(pipeline_output):
    """For each model JSON, provenance_summary.total_groups equals the number
    of metric_summaries this model appears in (each contributes one group)."""
    for path in (pipeline_output / "models").glob("*.json"):
        model = _read(path)
        route = model["model_route_id"]
        appearance_count = sum(
            1
            for cat_summaries in model.get("hierarchy_by_category", {}).values()
            for summary in cat_summaries
            for metric in _walk_metric_summaries(summary)
            if any(row["model_route_id"] == route for row in metric.get("model_results", []))
        )
        assert model["provenance_summary"]["total_groups"] == appearance_count, (
            f"{path.name}: per-model total_groups mismatch"
        )


def test_model_detail_nested_rollups_are_model_scoped(pipeline_output):
    """Nested model-detail eval summaries should roll up only filtered rows."""
    for path in (pipeline_output / "models").glob("*.json"):
        model = _read(path)
        route = model["model_route_id"]
        for cat_summaries in model.get("hierarchy_by_category", {}).values():
            for summary in cat_summaries:
                rows = _walk_rows(summary)
                assert rows, f"{path.name} {summary.get('eval_summary_id')} has no rows"
                assert all(row["model_route_id"] == route for row in rows)
                group_count = sum(
                    1
                    for metric in _walk_metric_summaries(summary)
                    if metric.get("model_results")
                )
                assert summary["reproducibility_summary"]["results_total"] == len(rows)
                assert summary["provenance_summary"]["total_results"] == len(rows)
                assert summary["provenance_summary"]["total_groups"] == group_count
                assert summary["comparability_summary"]["total_groups"] == group_count
                assert sum(
                    summary["provenance_summary"]["source_type_distribution"].values()
                ) == len(rows)


# ---------------------------------------------------------------------------
# benchmark_comparability rollup — only divergent groups appear
# ---------------------------------------------------------------------------


def test_benchmark_comparability_lists_only_divergent_groups(pipeline_output):
    """Per spec §6.1.5: the benchmark-level summary lists `all
    variant-divergent groups`. Verify each entry corresponds to a real group
    that has its respective `has_*_divergence` flag set to true."""
    for path in (pipeline_output / "evals").glob("*.json"):
        summary = _read(path)
        bc = summary["evalcards"]["annotations"]["benchmark_comparability"]
        # Build the set of (group_id) for groups whose row annotation flags
        # the matching `has_*_divergence` true.
        divergent_variant_group_ids: set[str] = set()
        divergent_cross_party_group_ids: set[str] = set()
        for row in _walk_rows(summary):
            ann = row["evalcards"]["annotations"]
            vd = ann.get("variant_divergence")
            if vd and vd.get("has_variant_divergence"):
                divergent_variant_group_ids.add(vd["group_id"])
            cpd = ann.get("cross_party_divergence")
            if cpd and cpd.get("has_cross_party_divergence"):
                divergent_cross_party_group_ids.add(cpd["group_id"])

        listed_variant = {g["group_id"] for g in bc["variant_divergence_groups"]}
        listed_cross_party = {g["group_id"] for g in bc["cross_party_divergence_groups"]}
        assert listed_variant == divergent_variant_group_ids, f"{path.name}"
        assert listed_cross_party == divergent_cross_party_group_ids, f"{path.name}"


# ---------------------------------------------------------------------------
# List-view consistency — all four views carry the three summary blocks
# ---------------------------------------------------------------------------


def test_eval_list_carries_three_summaries_per_entry(pipeline_output):
    eval_list = _read(pipeline_output / "eval-list.json")
    for entry in eval_list["evals"]:
        assert "reproducibility_summary" in entry
        assert "provenance_summary" in entry
        assert "comparability_summary" in entry


def test_eval_list_lite_carries_three_summaries_per_entry(pipeline_output):
    eval_list_lite = _read(pipeline_output / "eval-list-lite.json")
    for entry in eval_list_lite["evals"]:
        assert "reproducibility_summary" in entry
        assert "provenance_summary" in entry
        assert "comparability_summary" in entry


def test_model_cards_carry_three_summaries_per_entry(pipeline_output):
    model_cards = _read(pipeline_output / "model-cards.json")
    for entry in model_cards:
        assert "reproducibility_summary" in entry
        assert "provenance_summary" in entry
        assert "comparability_summary" in entry


def test_model_cards_lite_carry_three_summaries_per_entry(pipeline_output):
    model_cards_lite = _read(pipeline_output / "model-cards-lite.json")
    for entry in model_cards_lite:
        assert "reproducibility_summary" in entry
        assert "provenance_summary" in entry
        assert "comparability_summary" in entry


# ---------------------------------------------------------------------------
# Per-row / per-eval surface contract — schema stability for null returns
# ---------------------------------------------------------------------------


def test_every_row_has_all_four_signal_keys(pipeline_output):
    """Schema stability: variant_divergence and cross_party_divergence are
    explicit `null` when N/A, never absent."""
    expected = {"reproducibility_gap", "provenance", "variant_divergence", "cross_party_divergence"}
    for path in (pipeline_output / "evals").glob("*.json"):
        summary = _read(path)
        for row in _walk_rows(summary):
            actual = set(row["evalcards"]["annotations"].keys())
            assert expected <= actual, f"{path.name}: missing {expected - actual}"


def test_every_eval_has_benchmark_comparability_block(pipeline_output):
    for path in (pipeline_output / "evals").glob("*.json"):
        summary = _read(path)
        bc = summary["evalcards"]["annotations"]["benchmark_comparability"]
        assert "variant_divergence_groups" in bc
        assert "cross_party_divergence_groups" in bc


# ---------------------------------------------------------------------------
# Model identity normalization (existing behavior, scripts/pipeline.py:1474+)
# ---------------------------------------------------------------------------


def test_versioned_raw_id_collapses_to_family_with_distinct_variant_keys(pipeline_output):
    """openai/gpt-5-2024-01-01-{high,low} → both rows share family_id /
    model_route_id (openai__gpt-5) but expose distinct variant_keys via
    VERSION_SUFFIX_REGEX parsing."""
    summary = _read(pipeline_output / "evals" / "bench_versioned.json")
    rows = _walk_rows(summary)
    assert {row["model_route_id"] for row in rows} == {"openai__gpt-5"}
    assert {row["variant_key"] for row in rows} == {"2024-01-01-high", "2024-01-01-low"}


def test_versioned_raw_model_id_preserved_per_row(pipeline_output):
    """raw_model_id should round-trip the upstream EEE model_info.id verbatim
    (the only way a downstream consumer can recover the un-collapsed identity)."""
    summary = _read(pipeline_output / "evals" / "bench_versioned.json")
    raw_ids = {row["raw_model_id"] for row in _walk_rows(summary)}
    assert raw_ids == {"openai/gpt-5-2024-01-01-high", "openai/gpt-5-2024-01-01-low"}


def test_versioned_variants_appear_in_unified_model_card(pipeline_output):
    """The model card for openai__gpt-5 should aggregate every variant_key
    seen across configs (default from the un-versioned records, plus the
    two parsed variants from bench_versioned)."""
    model = _read(pipeline_output / "models" / "openai__gpt-5.json")
    variant_keys = {v["variant_key"] for v in model["variants"]}
    assert {"default", "2024-01-01-high", "2024-01-01-low"} <= variant_keys


# ---------------------------------------------------------------------------
# Per-result iteration (existing behavior — guards against the deprecated
# evaluation_results[0]-only read)
# ---------------------------------------------------------------------------


def test_multiresult_record_emits_one_row_per_result(pipeline_output):
    """1 EEE record × 3 results → 3 distinct metric_summaries × 1 row each."""
    summary = _read(pipeline_output / "evals" / "bench_multiresult.json")
    metrics = _walk_metric_summaries(summary)
    assert len(metrics) == 3
    rows = _walk_rows(summary)
    assert len(rows) == 3
    metric_names = {m["metric_name"] for m in metrics}
    assert metric_names == {"metric_a", "metric_b", "metric_c"}


def test_multiresult_per_result_generation_args_extracted_correctly(pipeline_output):
    """Each result has its own (temperature, max_tokens). All three rows
    must have populated_field_count=2 (active runtime subset). If the
    pipeline reverted to evaluation_results[0]-only, results [1] and [2]
    would carry the [0]th's args — the test would still pass for [0] but
    we'd get inconsistent variant_divergence behavior. Stronger guard:
    each row's reproducibility_gap reflects ITS OWN populated fields."""
    summary = _read(pipeline_output / "evals" / "bench_multiresult.json")
    rows = _walk_rows(summary)
    for row in rows:
        rg = row["evalcards"]["annotations"]["reproducibility_gap"]
        assert rg["has_reproducibility_gap"] is False, (
            f"row should have temp+max_tokens populated; missing: {rg['missing_fields']}"
        )
        assert rg["populated_field_count"] == rg["required_field_count"]


# ---------------------------------------------------------------------------
# Sort / rank — lower_is_better flip (scripts/pipeline.py:3592)
# ---------------------------------------------------------------------------


def test_lower_is_better_metric_top_score_is_minimum(pipeline_output):
    summary = _read(pipeline_output / "evals" / "bench_lower_better.json")
    metrics = _walk_metric_summaries(summary)
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric["lower_is_better"] is True
    rows = metric["model_results"]
    scores = [r["score"] for r in rows]
    # Best (lowest) score is at index 0 of the sorted list
    assert metric["top_score"] == min(scores)
    assert rows[0]["score"] == min(scores)
    # And the worst is at the end
    assert rows[-1]["score"] == max(scores)


def test_higher_is_better_metric_top_score_is_maximum(pipeline_output):
    """Sanity counter-test: bench_variant uses lower_is_better=False."""
    summary = _read(pipeline_output / "evals" / "bench_variant.json")
    metric = _walk_metric_summaries(summary)[0]
    assert metric["lower_is_better"] is False
    rows = metric["model_results"]
    scores = [r["score"] for r in rows]
    assert metric["top_score"] == max(scores)
    assert rows[0]["score"] == max(scores)


# ---------------------------------------------------------------------------
# Score summary aggregation on model card (scripts/pipeline.py:3936-3944)
# ---------------------------------------------------------------------------


def test_model_card_score_summary_aggregates_across_all_benchmarks(pipeline_output):
    """The score_summary on a model card aggregates across every score
    seen for that model in every evaluation_results. We compute the same
    aggregate from the per-eval JSONs and assert equality."""
    model_cards = _read(pipeline_output / "model-cards.json")
    card = next(c for c in model_cards if c["model_route_id"] == "openai__gpt-5")
    expected_scores: list[float] = []
    for path in (pipeline_output / "evals").glob("*.json"):
        for row in _walk_rows(_read(path)):
            if row["model_route_id"] == "openai__gpt-5":
                expected_scores.append(row["score"])
    ss = card["score_summary"]
    assert ss["count"] == len(expected_scores)
    assert ss["min"] == min(expected_scores)
    assert ss["max"] == max(expected_scores)
    assert abs(ss["average"] - sum(expected_scores) / len(expected_scores)) < 1e-9


# ---------------------------------------------------------------------------
# Per-row record-level field propagation (commit 9090cc5 — source_metadata
# carried onto each model_result row so consumers don't have to rejoin)
# ---------------------------------------------------------------------------


def test_per_row_source_metadata_propagated_from_record(pipeline_output):
    """Every model_result row should carry source_metadata.evaluator_relationship
    and source_organization_name."""
    summary = _read(pipeline_output / "evals" / "bench_multiorg.json")
    for row in _walk_rows(summary):
        sm = row.get("source_metadata")
        assert sm is not None
        assert sm.get("evaluator_relationship") in {"first_party", "third_party"}
        assert sm.get("source_organization_name") in {"OpenAI", "Scale AI"}


# ---------------------------------------------------------------------------
# Manifest correctness
# ---------------------------------------------------------------------------


def test_manifest_counts_match_published_artifacts(pipeline_output):
    manifest = _read(pipeline_output / "manifest.json")
    model_cards = _read(pipeline_output / "model-cards.json")
    eval_list = _read(pipeline_output / "eval-list.json")
    assert manifest["model_count"] == len(model_cards)
    assert manifest["eval_count"] == len(eval_list["evals"])
    eval_files = list((pipeline_output / "evals").glob("*.json"))
    assert manifest["eval_count"] == len(eval_files)
    model_files = list((pipeline_output / "models").glob("*.json"))
    assert manifest["model_count"] == len(model_files)


def test_manifest_artifact_sizes_have_path_and_byte_count(pipeline_output):
    """Each artifact_sizes entry should be {path, bytes} with a real byte count
    matching the on-disk file size. Catches `collect_artifact_sizes` regressions
    that drop fields or stop walking files."""
    manifest = _read(pipeline_output / "manifest.json")
    sizes = manifest["artifact_sizes"]
    assert isinstance(sizes, list) and len(sizes) > 0
    for entry in sizes:
        assert "path" in entry and "bytes" in entry
        assert isinstance(entry["bytes"], int) and entry["bytes"] > 0
        # Round-trip at least one entry against actual file size
    sample = sizes[0]
    on_disk = (pipeline_output / sample["path"]).stat().st_size
    assert sample["bytes"] == on_disk


# ---------------------------------------------------------------------------
# List-view shape — eval-list cherry-picks (does not carry full
# model_results); lite is structurally smaller than full
# ---------------------------------------------------------------------------


def test_eval_list_metrics_omit_full_model_results_array(pipeline_output):
    """eval-list.json metric entries are summary stats (top_score, models_count)
    — they must not carry the full model_results array (size optimization)."""
    eval_list = _read(pipeline_output / "eval-list.json")
    for entry in eval_list["evals"]:
        for metric in entry.get("metrics", []):
            assert "model_results" not in metric
        for subtask in entry.get("subtasks", []):
            for metric in subtask.get("metrics", []):
                assert "model_results" not in metric


def test_eval_list_lite_drops_benchmark_card_and_evalcards(pipeline_output):
    """Lite view drops the bulky `benchmark_card` (full ABC card object) and
    the `evalcards` annotations block. This is a structural contract — a
    byte-size check would pass even if a single byte shrunk."""
    full = _read(pipeline_output / "eval-list.json")
    lite = _read(pipeline_output / "eval-list-lite.json")
    assert "benchmark_card" in full["evals"][0]
    assert "benchmark_card" not in lite["evals"][0]
    assert "evalcards" in full["evals"][0]
    assert "evalcards" not in lite["evals"][0]


def test_model_cards_lite_truncates_benchmark_lists(pipeline_output):
    """Lite view truncates `benchmark_names` to <= 8 and `top_benchmark_scores`
    to <= 6 (size optimization for the frontend's hot path)."""
    full = _read(pipeline_output / "model-cards.json")
    lite = _read(pipeline_output / "model-cards-lite.json")
    assert "benchmark_names" in lite[0]
    assert "top_benchmark_scores" in lite[0]
    for full_entry, lite_entry in zip(full, lite):
        assert len(lite_entry["benchmark_names"]) <= 8
        assert len(lite_entry["top_benchmark_scores"]) <= 6
        # Truncation is the rule: lite ≤ full
        assert len(lite_entry["benchmark_names"]) <= len(full_entry["benchmark_names"])
        assert len(lite_entry["top_benchmark_scores"]) <= len(full_entry["top_benchmark_scores"])


# ---------------------------------------------------------------------------
# Reporting completeness varies by ABC card population
# ---------------------------------------------------------------------------


def test_reporting_completeness_present_with_valid_score_on_every_eval(pipeline_output):
    for path in (pipeline_output / "evals").glob("*.json"):
        summary = _read(path)
        rc = summary["evalcards"]["annotations"]["reporting_completeness"]
        assert "completeness_score" in rc
        assert 0.0 <= rc["completeness_score"] <= 1.0
        assert isinstance(rc.get("field_scores"), list)
        # All evals were given the same fixture ABC card → same field count
        assert rc["total_fields_evaluated"] > 0


# ---------------------------------------------------------------------------
# Developer aggregation
# ---------------------------------------------------------------------------


def test_developers_json_includes_each_unique_developer(pipeline_output):
    devs = _read(pipeline_output / "developers.json")
    dev_names = {d["developer"] for d in devs}
    # All developers we created in fixtures should appear (case-insensitive
    # via slugify_developer; canonical casing is the first one seen).
    expected = {"openai", "anthropic", "meta", "google", "mistral"}
    assert expected <= {d.lower() for d in dev_names}


# ---------------------------------------------------------------------------
# Ranking — peer-ranks file populates and is consistent with sorted order
# ---------------------------------------------------------------------------


def test_peer_ranks_file_populated_for_multi_model_evals(pipeline_output):
    """bench_lower_better has 3 models → peer_ranks should have an entry
    keyed by eval_summary_id, with one rank per model."""
    peer_ranks = _read(pipeline_output / "peer-ranks.json")
    assert "bench_lower_better" in peer_ranks
    ranks = peer_ranks["bench_lower_better"]
    assert len(ranks) == 3  # 3 distinct model_ids
    assert all(r["total"] == 3 for r in ranks.values())


# ---------------------------------------------------------------------------
# corpus-aggregates.json — §8 corpus-level aggregates
# ---------------------------------------------------------------------------


def test_corpus_aggregates_artifact_emitted_with_expected_top_level_keys(pipeline_output):
    ca = _read(pipeline_output / "corpus-aggregates.json")
    expected_top = {
        "generated_at",
        "signal_version",
        "stratification_dimensions",
        "reproducibility",
        "completeness",
        "provenance",
        "comparability",
    }
    assert expected_top <= set(ca.keys())
    # Stratification scope — by_source_type was deferred; only category should be listed
    assert ca["stratification_dimensions"] == ["category"]


def test_corpus_aggregates_each_signal_has_overall_plus_by_category(pipeline_output):
    """Schema contract for the four signal blocks. by_source_type intentionally
    absent (deferred 2026-04-26 per user decision)."""
    ca = _read(pipeline_output / "corpus-aggregates.json")
    for signal in ("reproducibility", "completeness", "provenance", "comparability"):
        block = ca[signal]
        assert "overall" in block, f"{signal} missing overall"
        assert "by_category" in block, f"{signal} missing by_category"
        assert "by_source_type" not in block, (
            f"{signal} unexpectedly has by_source_type — scope decision was overall+by_category only"
        )


def test_corpus_reproducibility_totals_match_row_sum_across_evals(pipeline_output):
    """Hard invariant: reproducibility.overall.total_triples must equal
    the actual count of rows across all per-eval JSONs."""
    ca = _read(pipeline_output / "corpus-aggregates.json")
    total_rows = 0
    for path in (pipeline_output / "evals").glob("*.json"):
        total_rows += len(_walk_rows(_read(path)))
    assert ca["reproducibility"]["overall"]["total_triples"] == total_rows


def test_corpus_by_category_triples_sum_to_overall(pipeline_output):
    """Each by_category bucket's total_triples should sum to overall's
    total_triples — guards against double-counting or stratification drift."""
    ca = _read(pipeline_output / "corpus-aggregates.json")
    overall_triples = ca["reproducibility"]["overall"]["total_triples"]
    cat_triples = sum(
        block["total_triples"] for block in ca["reproducibility"]["by_category"].values()
    )
    assert cat_triples == overall_triples


def test_corpus_provenance_source_type_distribution_always_four_buckets(pipeline_output):
    """Standardized-fields principle: distribution always has all 4 buckets,
    even when zero. Sums must equal total_triples."""
    ca = _read(pipeline_output / "corpus-aggregates.json")
    overall = ca["provenance"]["overall"]
    expected_buckets = {"first_party", "third_party", "collaborative", "unspecified"}
    assert set(overall["source_type_distribution"].keys()) == expected_buckets
    assert sum(overall["source_type_distribution"].values()) == overall["total_triples"]
    # Same for each by_category bucket
    for cat, block in ca["provenance"]["by_category"].items():
        assert set(block["source_type_distribution"].keys()) == expected_buckets, (
            f"{cat} category provenance distribution missing buckets"
        )


def test_corpus_reproducibility_per_field_missingness_has_correct_denominator_labels(pipeline_output):
    """Spec §8.3.2: agentic fields use agentic-only denominator; base fields
    use all-triples denominator. Catches a wiring regression where a base
    field accidentally uses the agentic denominator (or vice versa)."""
    ca = _read(pipeline_output / "corpus-aggregates.json")
    pf = ca["reproducibility"]["overall"]["per_field_missingness"]
    # Active runtime fields are temperature + max_tokens (base) and
    # eval_plan + eval_limits (agentic).
    assert pf["temperature"]["denominator"] == "all_triples"
    assert pf["max_tokens"]["denominator"] == "all_triples"
    assert pf["eval_plan"]["denominator"] == "agentic_only"
    assert pf["eval_limits"]["denominator"] == "agentic_only"


def test_corpus_comparability_rates_use_eligibility_denominator(pipeline_output):
    """variant_divergence_rate must be computed against variant_eligible_groups,
    not total_groups. Same for cross_party. Avoids conflating ineligible
    groups (no signal applicable) with eligible-but-passed groups."""
    ca = _read(pipeline_output / "corpus-aggregates.json")
    overall = ca["comparability"]["overall"]
    if overall["variant_eligible_groups"] > 0:
        expected = overall["variant_divergent_groups"] / overall["variant_eligible_groups"]
        assert abs(overall["variant_divergence_rate"] - expected) < 1e-9
    else:
        assert overall["variant_divergence_rate"] is None
    if overall["cross_party_eligible_groups"] > 0:
        expected = (
            overall["cross_party_divergent_groups"] / overall["cross_party_eligible_groups"]
        )
        assert abs(overall["cross_party_divergence_rate"] - expected) < 1e-9
    else:
        assert overall["cross_party_divergence_rate"] is None


def test_corpus_completeness_total_matches_eval_count(pipeline_output):
    """completeness.overall.total_benchmarks must equal the number of
    per-eval JSONs we emit (one completeness annotation per eval)."""
    ca = _read(pipeline_output / "corpus-aggregates.json")
    eval_files = list((pipeline_output / "evals").glob("*.json"))
    assert ca["completeness"]["overall"]["total_benchmarks"] == len(eval_files)


def test_corpus_aggregates_bench_agentic_drives_agentic_denominator(pipeline_output):
    """Positive control: bench_agentic fixture has 1 row with
    agentic_eval_config but no eval_plan/eval_limits → on the corpus-wide
    aggregate, eval_plan / eval_limits should have at least 1 missing on
    an agentic_only denominator. Exercises the agentic-aware split end-to-end."""
    ca = _read(pipeline_output / "corpus-aggregates.json")
    pf = ca["reproducibility"]["overall"]["per_field_missingness"]
    assert pf["eval_plan"]["denominator_count"] >= 1, (
        "expected at least 1 agentic triple from bench_agentic fixture"
    )
    assert pf["eval_plan"]["missing_count"] >= 1


# ---------------------------------------------------------------------------
# eval-hierarchy.json — §7 runtime-generated hierarchy with rollups
# ---------------------------------------------------------------------------


def test_hierarchy_artifact_top_level_structure(pipeline_output):
    h = _read(pipeline_output / "eval-hierarchy.json")
    expected = {"generated_at", "signal_version", "schema_note", "families"}
    assert expected <= set(h.keys())
    assert isinstance(h["families"], list) and len(h["families"]) > 0


def test_hierarchy_schema_note_documents_registry_deferral(pipeline_output):
    """The published artifact must surface the family-collapse-deferred
    caveat in its schema_note so downstream consumers understand why
    helm_classic / helm_lite / helm_air_bench don't roll up under `helm`.
    Catches a regression that drops the migration breadcrumb."""
    h = _read(pipeline_output / "eval-hierarchy.json")
    note = h["schema_note"]
    # Don't pin the exact wording; check the load-bearing concepts are mentioned
    assert "family" in note.lower()
    assert "registry" in note.lower() or "evalcard" in note.lower()


def test_hierarchy_every_family_carries_three_rollups(pipeline_output):
    """Each family node must have all three signal rollups (consistent
    with per-eval / per-model rollup contract)."""
    h = _read(pipeline_output / "eval-hierarchy.json")
    for fam in h["families"]:
        for key in ("reproducibility_summary", "provenance_summary", "comparability_summary"):
            assert key in fam, f"family {fam.get('key')} missing {key}"
        assert "leaves" in fam and isinstance(fam["leaves"], list)
        assert "evals_count" in fam
        assert "eval_summary_ids" in fam


def test_hierarchy_every_leaf_carries_three_rollups(pipeline_output):
    h = _read(pipeline_output / "eval-hierarchy.json")
    for fam in h["families"]:
        for leaf in fam["leaves"]:
            for key in ("reproducibility_summary", "provenance_summary", "comparability_summary"):
                assert key in leaf, f"family {fam['key']} leaf {leaf.get('key')} missing {key}"
            assert "eval_summary_ids" in leaf
            assert "evals_count" in leaf


def test_hierarchy_family_evals_count_equals_sum_of_leaf_evals(pipeline_output):
    """Tree consistency: family.evals_count == sum of its leaves' evals_count.
    Catches a bug where evals get dropped or double-counted between levels."""
    h = _read(pipeline_output / "eval-hierarchy.json")
    for fam in h["families"]:
        leaves_sum = sum(leaf["evals_count"] for leaf in fam["leaves"])
        assert fam["evals_count"] == leaves_sum, (
            f"family {fam['key']}: evals_count={fam['evals_count']} != "
            f"sum-of-leaves={leaves_sum}"
        )


def test_hierarchy_total_evals_match_manifest(pipeline_output):
    """Sum of evals_count across all families equals the published eval_count."""
    h = _read(pipeline_output / "eval-hierarchy.json")
    manifest = _read(pipeline_output / "manifest.json")
    total = sum(fam["evals_count"] for fam in h["families"])
    assert total == manifest["eval_count"]


def test_hierarchy_each_eval_summary_id_appears_in_exactly_one_family(pipeline_output):
    """No eval_summary should be double-listed across families. Catches
    grouping bugs where a benchmark falls under two parents."""
    h = _read(pipeline_output / "eval-hierarchy.json")
    seen: dict[str, str] = {}
    for fam in h["families"]:
        for esid in fam["eval_summary_ids"]:
            assert esid not in seen, (
                f"eval_summary_id {esid!r} appears in both families "
                f"{seen[esid]!r} and {fam['key']!r}"
            )
            seen[esid] = fam["key"]


def test_hierarchy_leaf_rollups_consistent_with_per_eval_summaries(pipeline_output):
    """A leaf with exactly one eval_summary_id should have a
    reproducibility_summary equal to the per-eval JSON's
    reproducibility_summary. Strong consistency check across two artifacts."""
    h = _read(pipeline_output / "eval-hierarchy.json")
    matched_at_least_one = False
    for fam in h["families"]:
        for leaf in fam["leaves"]:
            if len(leaf["eval_summary_ids"]) != 1:
                continue
            esid = leaf["eval_summary_ids"][0]
            per_eval_path = pipeline_output / "evals" / f"{esid}.json"
            if not per_eval_path.exists():
                continue
            per_eval = _read(per_eval_path)
            assert leaf["reproducibility_summary"] == per_eval["reproducibility_summary"], (
                f"leaf rollup for {esid} doesn't match per-eval JSON's reproducibility_summary"
            )
            matched_at_least_one = True
    assert matched_at_least_one, "test fixture had no single-eval leaves to verify against"


def test_hierarchy_setup_alias_collapse_landed_in_one_family(pipeline_output):
    """Positive control on Task 10's runtime hierarchy: bench_setupalias has
    2 records that collapse to model_route_id=openai__gpt-5; both rows fall
    under one eval_summary (bench_setupalias). Family node should have
    evals_count=1 with that single eval underneath."""
    h = _read(pipeline_output / "eval-hierarchy.json")
    fam = next((f for f in h["families"] if f["key"] == "bench_setupalias"), None)
    assert fam is not None, "bench_setupalias family missing from runtime hierarchy"
    assert fam["evals_count"] == 1
    assert fam["eval_summary_ids"] == ["bench_setupalias"]


def test_peer_ranks_lower_is_better_assigns_rank_one_to_lowest_score(pipeline_output):
    """For lower_is_better metrics, the model with the LOWEST score should
    be at rank 1. mistral/mistral-large = 0.10 (best); meta/llama-3 = 0.50
    (worst). Catches a sort-direction regression in the rank computation."""
    peer_ranks = _read(pipeline_output / "peer-ranks.json")
    ranks = peer_ranks["bench_lower_better"]
    # Ranks are keyed by model_id (family_id with `/`) per pipeline:3537
    assert ranks["mistral/mistral-large"]["position"] == 1
    assert ranks["meta/llama-3"]["position"] == 3


# ---------------------------------------------------------------------------
# ABC card join — fuzzy lookup (kebab-case EEE / snake-case ABC)
# ---------------------------------------------------------------------------


def test_abc_card_attaches_when_eee_name_kebab_case_and_card_snake_case(pipeline_output):
    """The fixture has EEE benchmark `bench-fuzzy` (kebab) and ABC card
    `bench_fuzzy.json` (snake). `lookup_benchmark_card` should normalize both
    sides and attach the card to the eval summary. CLAUDE.md notes ~11/34
    real benchmarks rely on this fuzzy matching today."""
    summary = _read(pipeline_output / "evals" / "bench_fuzzy.json")
    card = summary.get("benchmark_card")
    assert card is not None, "fuzzy-lookup ABC card failed to attach"
    assert card["benchmark_details"]["name"] == "bench_fuzzy"


def test_abc_card_tags_propagate_to_eval_summary(pipeline_output):
    """When an ABC card attaches, its domain/task tags should populate the
    eval summary's `tags` field. Verifies `extract_benchmark_tags`."""
    summary = _read(pipeline_output / "evals" / "bench_fuzzy.json")
    tags = summary.get("tags") or {}
    assert "domains" in tags
    # Our fixture card sets domains=["fixture"]
    assert "fixture" in tags["domains"]


# ---------------------------------------------------------------------------
# Setup-alias collapsing — gpt-5-fc + gpt-5-prompt → same family
# ---------------------------------------------------------------------------


def test_setup_alias_suffixes_collapse_to_same_family(pipeline_output):
    """Records with raw IDs `openai/gpt-5-fc` and `openai/gpt-5-prompt`
    should land under model_route_id=openai__gpt-5 with variant_key=default.
    The setup-alias suffix is stripped from family_slug and is NOT preserved
    as a variant qualifier (commits 6e705ba, 9866766)."""
    summary = _read(pipeline_output / "evals" / "bench_setupalias.json")
    rows = _walk_rows(summary)
    assert len(rows) == 2
    assert {row["model_route_id"] for row in rows} == {"openai__gpt-5"}
    assert {row["variant_key"] for row in rows} == {"default"}
    # Raw IDs preserved per row so the un-collapsed identity is recoverable
    assert {row["raw_model_id"] for row in rows} == {
        "openai/gpt-5-fc", "openai/gpt-5-prompt"
    }


def test_setup_alias_rows_share_one_signal_group(pipeline_output):
    """Both setup-alias rows are in the same model_route_id, so signals 3+4
    treat them as ONE group (size 2). This is a real-world consequence of
    the collapse: variant divergence may fire across call-style variants."""
    summary = _read(pipeline_output / "evals" / "bench_setupalias.json")
    cs = summary["comparability_summary"]
    # One model_route_id → one group
    assert cs["total_groups"] == 1


# ---------------------------------------------------------------------------
# Provenance — collaborative source_type (spec §5.4)
# ---------------------------------------------------------------------------


def test_collaborative_source_type_propagates_to_distribution(pipeline_output):
    """bench_orgcollapse has one collaborative row + one third_party row.
    The eval-level provenance_summary should reflect both buckets."""
    summary = _read(pipeline_output / "evals" / "bench_orgcollapse.json")
    dist = summary["provenance_summary"]["source_type_distribution"]
    assert dist["collaborative"] == 1
    assert dist["third_party"] == 1


def test_collaborative_row_has_first_party_only_false(pipeline_output):
    """Per spec §5.4, a collaborative-relationship row never reports
    first_party_only=true even when the group has a single org."""
    summary = _read(pipeline_output / "evals" / "bench_orgcollapse.json")
    for row in _walk_rows(summary):
        sm = row.get("source_metadata") or {}
        if sm.get("evaluator_relationship") == "collaborative":
            prov = row["evalcards"]["annotations"]["provenance"]
            assert prov["source_type"] == "collaborative"
            assert prov["first_party_only"] is False
            return
    raise AssertionError("expected at least one collaborative row in bench_orgcollapse")


# ---------------------------------------------------------------------------
# comparison-index.json shape (load-bearing frontend artifact)
# ---------------------------------------------------------------------------


def test_comparison_index_has_evals_and_by_model_indexes(pipeline_output):
    ci = _read(pipeline_output / "comparison-index.json")
    assert "evals" in ci and isinstance(ci["evals"], dict)
    assert "by_model" in ci and isinstance(ci["by_model"], dict)
    assert "metric_group_order" in ci  # tab strip ordering for the frontend


def test_comparison_index_per_eval_metrics_carry_scores_array(pipeline_output):
    """Each metric inside an eval entry has a `scores` array — the histogram
    rows the frontend renders. One score per scoring model_route_id."""
    ci = _read(pipeline_output / "comparison-index.json")
    eval_entry = ci["evals"]["bench_lower_better"]
    assert eval_entry["metrics"], "expected at least one metric"
    metric = eval_entry["metrics"][0]
    assert "scores" in metric
    assert len(metric["scores"]) == 3  # 3 distinct models in fixture


def test_comparison_index_lower_is_better_metric_sorts_lowest_first(pipeline_output):
    """For lower_is_better metrics the `scores` array is best-first =
    ascending. Catches a sort-direction regression."""
    ci = _read(pipeline_output / "comparison-index.json")
    metric = ci["evals"]["bench_lower_better"]["metrics"][0]
    assert metric["lower_is_better"] is True
    score_values = [entry["score"] for entry in metric["scores"]]
    assert score_values == sorted(score_values)


def test_comparison_index_by_model_includes_each_scoring_model(pipeline_output):
    """The inverse `by_model` index should let a model detail page look up
    its rows in O(1). Each scoring model_route_id appears as a top-level key."""
    ci = _read(pipeline_output / "comparison-index.json")
    by_model = ci["by_model"]
    assert "openai__gpt-5" in by_model
    assert "anthropic__claude-opus-4-5" in by_model
    assert "mistral__mistral-large" in by_model


# ---------------------------------------------------------------------------
# benchmark-metadata.json content correctness
# ---------------------------------------------------------------------------


def test_benchmark_metadata_round_trips_card_content(pipeline_output):
    """Every fixture benchmark should appear in benchmark-metadata.json with
    its purpose_and_intended_users.tasks preserved. Catches lossy aggregation
    in `load_benchmark_metadata`."""
    bm = _read(pipeline_output / "benchmark-metadata.json")
    # The flat aggregator is keyed by normalized benchmark name
    assert "bench_variant" in bm
    assert "bench_agentic" in bm
    # Agentic fixture card was given tasks-literal "agentic" tokens
    agentic_tasks = bm["bench_agentic"]["purpose_and_intended_users"]["tasks"]
    assert "agentic" in agentic_tasks


# ---------------------------------------------------------------------------
# Reproducibility-summary count correctness
# ---------------------------------------------------------------------------


def test_reproducibility_summary_count_matches_actual_gap_rows(pipeline_output):
    """For each eval, has_reproducibility_gap_count must equal the number
    of rows whose per-row reproducibility_gap.has_reproducibility_gap is true.
    Catches a rollup that miscounts (e.g., counts groups instead of rows)."""
    for path in (pipeline_output / "evals").glob("*.json"):
        summary = _read(path)
        rs = summary["reproducibility_summary"]
        expected_total = len(_walk_rows(summary))
        expected_gap_count = sum(
            1
            for row in _walk_rows(summary)
            if row["evalcards"]["annotations"]["reproducibility_gap"][
                "has_reproducibility_gap"
            ]
        )
        assert rs["results_total"] == expected_total, f"{path.name}: results_total"
        assert rs["has_reproducibility_gap_count"] == expected_gap_count, (
            f"{path.name}: has_reproducibility_gap_count"
        )


def test_reproducibility_summary_bench_agentic_flags_one_row(pipeline_output):
    """Direct positive: the agentic fixture's row is missing eval_plan +
    eval_limits, so has_reproducibility_gap_count must be 1."""
    summary = _read(pipeline_output / "evals" / "bench_agentic.json")
    assert summary["reproducibility_summary"]["has_reproducibility_gap_count"] == 1

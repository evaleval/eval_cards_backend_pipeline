"""Every published count equals a distinct count at its stated grain.

The warehouse's headline numbers (manifest.json, hierarchy.json stats,
headline.json, benchmark_index.json, models_view) are read off views that
repeat a fact many times: one row per source config, per reporting org,
per slice, per composite appearance. Counting rows there instead of
distinct keys over-counts. This test bakes a corpus in which the same
(model, benchmark, metric) triple is reported by several source configs
and the same slice benchmark appears under two composites, then checks each
published count against an independent COUNT(DISTINCT ...) over the
lowest-grain table that can express it (fact_results, else
eval_results_view).

Corpus: `tests/fixtures/` as is, plus two slice-child benchmarks
(`mmlu-anatomy`, `mmlu-astronomy`, parent `mmlu`) added to the fixture
registry so the `Anatomy` / `Astronomy` observations in `fixtures_slices`
resolve to slice rows instead of folding into `mmlu`. The duplicated triple
is `openai/gpt-4o` x `mmlu` x `accuracy`, reported by `fixtures_clean`,
`fixtures_variant` and `fixtures_xparty`.

Covered: manifest model_count / eval_count / metric_eval_count /
composite_count / source_config_count; hierarchy stats benchmark_count;
headline total_benchmarks / reporting_org_count / comparability
total_triples / composites[].model_count;
benchmark_index benchmark_count and appearances[].models_count;
models_view benchmarks_count and benchmark_names.

Left out, with the reason:
- hierarchy stats slice_count counts tree slices, which include slice_key
  partitions of one benchmark as well as slice-child benchmarks; no single
  fact column names that grain.
- headline developers[] and model_families[] and tags[] are rollups of
  models_view / eval_results_view columns (developer, model_group_id,
  derived_tags) that fact_results does not carry.
- models_view evaluations_count and headline composites[].evaluation_count
  count eval_results_view cells, whose grain is defined by the view itself.
- headline composites[].benchmark_count counts view rows per composite,
  where a parent row is synthesised only when it has more than one slice
  child; that rule lives in the view, not in fact_results.
"""
from __future__ import annotations

import json
import shutil

import duckdb
import pandas as pd
import pytest

from tests.fixture_corpus import FIXTURES, build_fixture_corpus

SLICE_CHILDREN = {
    "Anatomy": ("mmlu-anatomy", "MMLU Anatomy"),
    "Astronomy": ("mmlu-astronomy", "MMLU Astronomy"),
}


def _registry_with_slice_children(root):
    reg = root / "entity_registry"
    shutil.copytree(FIXTURES / "entity_registry", reg)

    benchmarks = pd.read_parquet(reg / "canonical_benchmarks.parquet")
    parent = benchmarks.loc[benchmarks["id"] == "mmlu"].iloc[0]
    children = []
    for child_id, display in SLICE_CHILDREN.values():
        row = parent.copy()
        row["id"] = child_id
        row["display_name"] = display
        row["parent_benchmark_id"] = "mmlu"
        children.append(row)
    pd.concat([benchmarks, pd.DataFrame(children)], ignore_index=True).to_parquet(
        reg / "canonical_benchmarks.parquet", index=False
    )

    aliases = pd.read_parquet(reg / "aliases.parquet")
    for raw, (child_id, _) in SLICE_CHILDREN.items():
        mask = (aliases["entity_type"] == "benchmark") & (aliases["raw_value"] == raw)
        assert mask.sum() == 1, raw
        aliases.loc[mask, "canonical_id"] = child_id
    aliases.to_parquet(reg / "aliases.parquet", index=False)
    return reg


@pytest.fixture(scope="module")
def build(tmp_path_factory, _taxonomy_seed_stub, _collections_stub):
    root = tmp_path_factory.mktemp("count_grain")
    return build_fixture_corpus(
        root,
        seed_dir=_taxonomy_seed_stub,
        collections_dir=_collections_stub,
        registry_dir=_registry_with_slice_children(root),
    )


@pytest.fixture(scope="module")
def con(build):
    con = duckdb.connect()
    for table in (
        "fact_results", "eval_results_view", "benchmarks", "composites",
        "models_view",
    ):
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM "
            f"read_parquet('{build.out_dir / table}.parquet')"
        )
    yield con
    con.close()


@pytest.fixture(scope="module")
def sidecar(build):
    def load(name: str) -> dict:
        return json.loads((build.out_dir / name).read_text())
    return load


def _scalar(con, sql: str, params=None) -> int:
    return int(con.execute(sql, params or []).fetchone()[0] or 0)


# ---------------------------------------------------------------------------
# The corpus has the duplication this test is about
# ---------------------------------------------------------------------------


def test_corpus_repeats_a_triple_across_source_configs(con):
    configs = con.execute(
        """
        SELECT DISTINCT source_config FROM fact_results
        WHERE model_aggregation_key = 'openai/gpt-4o'
          AND benchmark_key = 'mmlu' AND metric_key = 'accuracy'
        ORDER BY 1
        """
    ).fetchall()
    assert [c for (c,) in configs] == [
        "fixtures_clean", "fixtures_variant", "fixtures_xparty",
    ]


def test_corpus_has_one_slice_under_two_composites(con):
    rows = con.execute(
        """
        SELECT composite_slug FROM benchmarks
        WHERE benchmark_id = 'mmlu-anatomy' AND is_slice
        ORDER BY 1
        """
    ).fetchall()
    assert [c for (c,) in rows] == [
        "fixtures-slices--openai", "fixtures-slices--scale-ai",
    ]


# ---------------------------------------------------------------------------
# manifest.json
# ---------------------------------------------------------------------------


def test_manifest_model_count_is_distinct_model_keys(con, sidecar):
    expected = _scalar(
        con, "SELECT COUNT(DISTINCT model_aggregation_key) FROM fact_results"
    )
    assert sidecar("manifest.json")["model_count"] == expected


def test_manifest_eval_count_is_distinct_composite_benchmark_pairs(con, sidecar):
    expected = _scalar(
        con,
        "SELECT COUNT(DISTINCT (composite_slug, benchmark_key)) FROM fact_results",
    )
    assert sidecar("manifest.json")["eval_count"] == expected


def test_manifest_metric_eval_count_is_distinct_triples(con, sidecar):
    expected = _scalar(
        con,
        """
        SELECT COUNT(DISTINCT (model_aggregation_key, composite_slug,
                               benchmark_key, metric_key))
        FROM fact_results
        """,
    )
    assert sidecar("manifest.json")["metric_eval_count"] == expected


def test_manifest_composite_count_is_distinct_composites(con, sidecar):
    in_dim = _scalar(con, "SELECT COUNT(DISTINCT composite_slug) FROM composites")
    in_facts = _scalar(con, "SELECT COUNT(DISTINCT composite_slug) FROM fact_results")
    assert in_dim == in_facts, "composites dim and fact_results disagree on slugs"
    assert sidecar("manifest.json")["composite_count"] == in_dim


def test_manifest_source_config_count_is_distinct_configs(con, sidecar):
    expected = _scalar(con, "SELECT COUNT(DISTINCT source_config) FROM fact_results")
    assert sidecar("manifest.json")["source_config_count"] == expected


# ---------------------------------------------------------------------------
# Benchmarks: slice children never count as benchmarks of their own
# ---------------------------------------------------------------------------

# A slice child's root is its parent; a root benchmark is its own root.
_ROOT = "COALESCE(parent_benchmark_id, benchmark_key)"


def test_hierarchy_benchmark_count_is_distinct_root_pairs(con, sidecar):
    expected = _scalar(
        con,
        f"SELECT COUNT(DISTINCT (composite_slug, {_ROOT})) FROM fact_results",
    )
    assert sidecar("hierarchy.json")["stats"]["benchmark_count"] == expected


def test_headline_total_benchmarks_is_distinct_roots(con, sidecar):
    expected = _scalar(con, f"SELECT COUNT(DISTINCT {_ROOT}) FROM fact_results")
    assert sidecar("headline.json")["total_benchmarks"] == expected


def test_models_view_benchmarks_count_is_distinct_roots_per_model(con):
    mismatches = con.execute(
        f"""
        WITH expected AS (
            SELECT model_aggregation_key AS model_key,
                   COUNT(DISTINCT {_ROOT}) AS n
            FROM fact_results GROUP BY 1
        )
        SELECT mv.model_key, mv.benchmarks_count, len(mv.benchmark_names), e.n
        FROM models_view mv JOIN expected e USING (model_key)
        WHERE mv.benchmarks_count != e.n OR len(mv.benchmark_names) != e.n
        ORDER BY 1
        """
    ).fetchall()
    assert not mismatches, (
        "models_view (model_key, benchmarks_count, len(benchmark_names), "
        f"distinct root benchmarks in fact_results): {mismatches}"
    )


def test_benchmark_index_count_is_distinct_benchmark_ids(con, sidecar):
    index = sidecar("benchmark_index.json")
    expected = _scalar(con, "SELECT COUNT(DISTINCT benchmark_key) FROM fact_results")
    assert index["benchmark_count"] == expected
    assert len(index["benchmarks"]) == expected


def test_benchmark_index_models_count_is_distinct_models_per_appearance(con, sidecar):
    mismatches = []
    for benchmark_id, entry in sidecar("benchmark_index.json")["benchmarks"].items():
        for appearance in entry["appearances"]:
            expected = _scalar(
                con,
                """
                SELECT COUNT(DISTINCT model_key) FROM eval_results_view
                WHERE composite_slug = ? AND benchmark_id = ? AND metric_id = ?
                  AND is_headline
                """,
                [appearance["composite_slug"], benchmark_id,
                 appearance["primary_metric_id"]],
            )
            if appearance["models_count"] != expected:
                mismatches.append(
                    (benchmark_id, appearance["composite_slug"],
                     appearance["models_count"], expected)
                )
    assert not mismatches, (
        "benchmark_index (benchmark, composite, models_count, distinct "
        f"headline models): {mismatches}"
    )


# ---------------------------------------------------------------------------
# headline.json
# ---------------------------------------------------------------------------


def test_headline_reporting_org_count_is_distinct_orgs(con, sidecar):
    expected = _scalar(
        con,
        "SELECT COUNT(DISTINCT org_display) FROM fact_results "
        "WHERE org_display IS NOT NULL",
    )
    assert sidecar("headline.json")["reporting_org_count"] == expected


def test_headline_comparability_total_is_distinct_groups(con, sidecar):
    expected = _scalar(
        con,
        "SELECT COUNT(DISTINCT comparability_group_id) FROM fact_results "
        "WHERE comparability_group_id IS NOT NULL",
    )
    overall = sidecar("headline.json")["comparability"]["overall"]
    assert overall["total_triples"] == expected


def test_headline_composites_model_count_is_distinct_models(con, sidecar):
    mismatches = []
    for entry in sidecar("headline.json")["composites"]:
        expected = _scalar(
            con,
            "SELECT COUNT(DISTINCT model_aggregation_key) FROM fact_results "
            "WHERE composite_slug = ?",
            [entry["composite_slug"]],
        )
        if entry["model_count"] != expected:
            mismatches.append((entry["composite_slug"], entry["model_count"], expected))
    assert not mismatches, (
        f"headline.composites (slug, model_count, distinct models): {mismatches}"
    )

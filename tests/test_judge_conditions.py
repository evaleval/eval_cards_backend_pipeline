"""Judge conditions + canonical score before comparability (issue #47).

Runs the pipeline over `fixtures_judges`, a corpus shaped after the OpenEval
records that motivated the work: WildBench published as three single-judge
channels, a three-judge aggregate and the source's own already-rescaled copy;
the six GPT-5.x raw-vs-rescaled collisions; two Omni-MATH judge cohorts;
CNN/DailyMail's encoder backbone; a malformed judge list; and a LEXam-shaped
record carrying both judge sources at once.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

GPT4O = "openai/gpt-4o-2024-05-13"
CLAUDE = "anthropic/claude-3-5-sonnet-20241022"
LLAMA = "meta/llama-3.1-405b-instruct-turbo"

# The registry rule the fixture registry carries for OpenEval's WildBench:
# canonical = published * factor + offset, i.e. (x - 1) / 9.
FACTOR, OFFSET = 0.1111111111111111, -0.1111111111111111


def _condition(judges, label) -> str:
    return json.dumps({"judges": sorted(judges), "label": label},
                      separators=(",", ":"))


def _run(tmp_path, monkeypatch, registry_dir=FIXTURES / "entity_registry"):
    """Run the pipeline over `fixtures_judges`; return the snapshot dir. The
    registry dir is a parameter so a test can run the same corpus against an
    edited curation without disturbing the shared fixture."""
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(FIXTURES / "eee"))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR",
                       str(FIXTURES / "auto_benchmarkcards"))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        configs=["fixtures_judges"],
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(tmp_path / "warehouse"),
        registry_local_dir=str(registry_dir),
        cache_root=str(tmp_path / "cache"),
    )
    assert out_dir is not None
    return out_dir


@pytest.fixture
def snapshot(tmp_path, monkeypatch, caplog):
    """Run the pipeline over `fixtures_judges` once; return the snapshot dir."""
    # INFO so the tests that read a run's own log lines see them: the records
    # are emitted during fixture setup, which is too early for a test body to
    # raise the level itself.
    caplog.set_level(logging.INFO)
    return _run(tmp_path, monkeypatch)


@pytest.fixture
def facts(snapshot):
    con = duckdb.connect()
    for table in ("fact_results", "eval_results_view"):
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM "
            f"read_parquet('{snapshot}/{table}.parquet')"
        )
    return con


def _restage_j(snapshot, mutate=None, full=False):
    """Re-run Stage J over the snapshot's canonical parquets, optionally
    injecting fact rows first. Lets a test build a condition shape the EEE
    fixtures cannot carry (protocol arms come from a collection adapter, not
    from a source record)."""
    from eval_card_backend.canonicalise import stages
    from eval_card_backend.canonicalise.resolver_setup import register_udfs
    from eval_card_backend.sources import registry as registry_src
    from eval_entity_resolver import Resolver

    con = duckdb.connect()
    register_udfs(
        con, Resolver(registry_src.load_alias_store(FIXTURES / "entity_registry"))
    )
    for table in ("fact_results", "benchmarks", "composites", "families",
                  "models", "canonical_metrics"):
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM "
            f"read_parquet('{snapshot}/{table}.parquet')"
        )
    if mutate is not None:
        mutate(con)
    stages.stage_j_eval_results_view(con, "2026-04-30T00:00:00Z")
    if full:
        # the page-level views fan out from eval_results_view; they also read
        # the registry's benchmark dim for the preferred-metric rule
        reg = FIXTURES / "entity_registry"
        # stage_j_eval_results_view leaves an empty stand-in behind; replace
        # it with the fixture registry's real rows
        con.execute("DROP TABLE IF EXISTS canonical_benchmarks")
        stages._load_dim(con, "canonical_benchmarks",
                         {"canonical_benchmarks": reg / "canonical_benchmarks.parquet"})
        stages.stage_j_models_view(con, "2026-04-30T00:00:00Z")
        stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")
    return con


def _rows(con, sql, params=None):
    return con.execute(sql, params or []).fetchall()


# ---------------------------------------------------------------------------
# Rename at resolution
# ---------------------------------------------------------------------------


def test_every_wildbench_channel_lands_on_the_renamed_metric(facts):
    """All four channels of one record are the same measurement under four
    source labels; the rename rule is what makes them one page."""
    rows = _rows(facts, """
        SELECT metric_source_label, metric_id, metric_key
        FROM fact_results
        WHERE benchmark_key = 'wildbench' AND model_raw = 'openai/gpt-4o'
        ORDER BY metric_source_label
    """)
    assert [r[0] for r in rows] == [
        "claude_score", "gpt_score", "llama_score", "wildbench_score_rescaled",
    ]
    assert {r[2] for r in rows} == {"wb-score"}
    # the pre-rename id still says how each channel resolved: the three raw
    # channels through the generic `score`, the rescaled one straight to
    # wb-score via the source-scoped whole-id alias
    assert [r[1] for r in rows] == ["score", "score", "score", "wb-score"]


def test_a_source_scoped_conversion_needs_no_rename(tmp_path, monkeypatch):
    """The registry allows a conversion row whose from- and to-metric are the
    same, with no unscoped naming row beside it: a source publishing the
    benchmark's own metric on its own scale needs the affine, not a rename.
    The rename step ignores source-scoped rows, so `metric_id_effective` stays
    `metric_id` — and the Stage D conversion join, which keys on
    (benchmark_id, PRE-rename metric_id, source_config), still finds it."""
    import shutil

    import pandas as pd

    registry = tmp_path / "entity_registry"
    shutil.copytree(FIXTURES / "entity_registry", registry)
    folds_path = registry / "benchmark_metric_folds.parquet"
    folds = pd.read_parquet(folds_path)
    folds.loc[len(folds)] = {
        "benchmark_id": "omni-math",
        "from_metric_id": "correctness",
        "to_metric_id": "correctness",
        "scale_factor": 0.01,
        "scale_offset": None,
        "source_config": "fixtures_judges",
        "note": "conversion: this source publishes correctness as a percentage",
    }
    folds.to_parquet(folds_path, index=False)

    out = _run(tmp_path, monkeypatch, registry)
    con = duckdb.connect()
    rows = _rows(con, f"""
        SELECT metric_id, metric_key, scale_conversion,
               score_canonical = score * 0.01
        FROM read_parquet('{out}/fact_results.parquet')
        WHERE benchmark_key = 'omni-math'
    """)
    assert rows
    assert all(r == ("correctness", "correctness", "curated", True)
               for r in rows)


def test_judge_conditions_are_canonical_json(facts):
    rows = dict(_rows(facts, """
        SELECT metric_source_label, judge_condition FROM fact_results
        WHERE benchmark_key = 'wildbench' AND model_raw = 'openai/gpt-4o'
    """))
    assert rows["gpt_score"] == _condition([GPT4O], "gpt_score")
    # this one discloses its judge as a typed llm_scoring struct, not a
    # stringified list, and reads the same
    assert rows["claude_score"] == _condition([CLAUDE], "claude_score")
    assert rows["llama_score"] == _condition([LLAMA], "llama_score")
    # the aggregate's list arrives unsorted and with a repeat
    assert rows["wildbench_score_rescaled"] == _condition(
        [CLAUDE, LLAMA, GPT4O], "wildbench_score_rescaled"
    )


def test_one_judge_under_two_labels_stays_two_conditions(facts):
    """The new Omni-MATH cohort publishes GPT-4o twice under `gpt_correctness`
    and `gpt-4o_correctness`. Same judge, same metric, different published
    numbers: the label is what keeps them apart."""
    rows = dict(_rows(facts, """
        SELECT metric_source_label, judge_condition FROM fact_results
        WHERE benchmark_key = 'omni-math' AND model_raw = 'openai/gpt-5-mini'
          AND metric_source_label LIKE '%correctness'
    """))
    assert rows["gpt_correctness"] == _condition([GPT4O], "gpt_correctness")
    # the short id resolves to the same canonical model
    assert rows["gpt-4o_correctness"] == _condition([GPT4O], "gpt-4o_correctness")


def test_typed_llm_scoring_outranks_metric_models_json(facts):
    """The LEXam-shaped record discloses GPT-4o in the typed struct and Claude
    in the stringified fallback. The typed disclosure is the specific one."""
    [(condition,)] = _rows(facts, """
        SELECT judge_condition FROM fact_results WHERE benchmark_key = 'lexam'
    """)
    assert condition == _condition([GPT4O], "judge_score")


# ---------------------------------------------------------------------------
# Gating and malformed input
# ---------------------------------------------------------------------------


def test_encoder_backbone_on_an_unjudged_benchmark_is_no_condition(facts):
    """CNN/DailyMail names a DeBERTa entailment model in the same field an
    LLM-judged benchmark uses for judges. The registry says the benchmark is
    not LLM-judged, so it is not a judge — and the model resolver is never
    asked about it."""
    from eval_card_backend.canonicalise import udfs

    rows = _rows(facts, """
        SELECT judge_condition FROM fact_results
        WHERE benchmark_key = 'cnn-dailymail'
    """)
    assert rows == [(None,)]
    assert "microsoft/deberta-large-mnli" not in udfs.miss_examples["model"]


def test_a_fallback_judge_off_the_preferred_metric_is_no_condition(facts):
    """The registry flag is about the benchmark's PREFERRED metric, not about
    every channel published under the benchmark. Disinfo-bench prefers the
    `rating` channel; the `accuracy` one names a model in the same field, and
    the gate does not read it as a judge."""
    rows = _rows(facts, """
        SELECT DISTINCT metric_key, judge_condition IS NOT NULL
        FROM fact_results WHERE benchmark_key = 'disinfo-bench'
        ORDER BY metric_key
    """)
    assert rows == [("accuracy", False), ("rating", True)]


def test_malformed_metric_models_json_is_counted_not_fatal(facts, caplog):
    """Two unreadable shapes and an empty list. None of them aborts the run,
    none of them invents a judge, and the two unreadable ones are counted so a
    source that starts emitting a shape we can't parse is visible."""
    rows = _rows(facts, """
        SELECT judge_condition FROM fact_results
        WHERE model_raw = 'synthetic/malformed'
    """)
    assert rows == [(None,), (None,), (None,)]
    [message] = [
        r.getMessage() for r in caplog.get_records("setup")
        if "malformed metric_models_json" in r.getMessage()
    ]
    assert message.startswith("stage D: 2 row(s)")


def test_rename_rule_logs_the_source_labels_it_swept_up(facts, caplog):
    """A rule keyed on the catch-all `score` renames every channel that failed
    to resolve on its own, so the set of labels it sweeps up grows silently as
    new sources land. The log line is what makes that visible before a new
    channel joins someone else's metric page."""
    messages = sorted(
        r.getMessage() for r in caplog.get_records("setup")
        if r.getMessage().startswith("stage C: rename ")
    )
    assert messages == [
        # the offset-only rule: one channel, named by its raw_metric_name
        "stage C: rename cnn-dailymail score\u2192summarization-score renamed "
        "1 row(s); source labels: nli_entailment=1",
        "stage C: rename wildbench score\u2192wb-score renamed 14 row(s); "
        "source labels: gpt_score=10, claude_score=2, llama_score=2",
    ]


# ---------------------------------------------------------------------------
# Canonical score
# ---------------------------------------------------------------------------


def test_curated_affine_lands_exactly_on_the_metric_endpoints(facts):
    """A raw 1 and a raw 10 are the ends of the published 1-10 rating. Under
    (x-1)/9 they must be the ends of wb-score's [0,1] too — the factor alone
    would put 10 at 1.11 and flag it."""
    rows = dict(_rows(facts, """
        SELECT model_raw, (score, score_canonical, scale_conversion)
        FROM fact_results WHERE model_raw LIKE 'synthetic/endpoint-%'
    """))
    assert rows["synthetic/endpoint-lo"][2] == "curated"
    assert rows["synthetic/endpoint-lo"][1] == pytest.approx(0.0, abs=1e-9)
    assert rows["synthetic/endpoint-hi"][2] == "curated"
    assert rows["synthetic/endpoint-hi"][1] == pytest.approx(1.0, abs=1e-9)
    curated = _rows(facts, """
        SELECT score, score_canonical FROM fact_results
        WHERE benchmark_key = 'wildbench' AND scale_conversion = 'curated'
    """)
    assert curated
    assert all(
        canon == pytest.approx(raw * FACTOR + OFFSET, abs=1e-9)
        for raw, canon in curated
    )


def test_the_sources_own_rescaled_copy_is_not_converted_again(facts):
    """The rescaled channel resolves straight to wb-score, so the rename
    rule's conversion — which keys on the pre-rename `score` — never matches
    it. It is already canonical and must pass through untouched."""
    rows = _rows(facts, """
        SELECT score, score_canonical, scale_conversion FROM fact_results
        WHERE metric_source_label = 'wildbench_score_rescaled'
    """)
    assert rows
    assert all(conv == "none" and canon == raw for raw, canon, conv in rows)


def test_uncertainty_rides_the_same_conversion_as_the_score(facts):
    """A confidence interval is a pair of scores and takes the whole affine
    map; a standard error is a width and takes the factor's magnitude only.
    Sample size is not a score and is untouched."""
    [(se, se_c, lo, hi, n)] = _rows(facts, """
        SELECT score_se, score_se_canonical, score_ci_lower_canonical,
               score_ci_upper_canonical, n_samples
        FROM fact_results
        WHERE benchmark_key = 'wildbench' AND model_raw = 'openai/gpt-4o'
          AND metric_source_label = 'gpt_score'
    """)
    assert se == 0.9
    assert se_c == pytest.approx(0.9 * FACTOR)
    assert lo == pytest.approx(4.5 * FACTOR + OFFSET)
    assert hi == pytest.approx(6.5 * FACTOR + OFFSET)
    assert n == 1024


def test_three_of_the_six_scale_collisions_agree(facts):
    """Six GPT-5.x records publish the raw rating AND the source's rescaled
    copy. Only three are the same measurement; the others differ in response
    count (1023 vs 1024) or are plainly a different run (509 vs 1024)."""
    rows = _rows(facts, """
        WITH raw AS (
            SELECT model_raw, score_canonical AS converted
            FROM fact_results
            WHERE benchmark_key = 'wildbench'
              AND metric_source_label = 'gpt_score'
              AND model_raw LIKE 'openai/gpt-5%'
        ),
        rescaled AS (
            SELECT model_raw, score_canonical AS published
            FROM fact_results
            WHERE metric_source_label = 'wildbench_score_rescaled'
              AND model_raw LIKE 'openai/gpt-5%'
        )
        SELECT raw.model_raw, abs(raw.converted - rescaled.published) <= 1e-6
        FROM raw JOIN rescaled USING (model_raw)
        ORDER BY raw.model_raw
    """)
    assert len(rows) == 6
    agreeing = sorted(m for m, same in rows if same)
    assert agreeing == [
        "openai/gpt-5.4-2026-03-05",
        "openai/gpt-5.4-mini-2026-03-05",
        "openai/gpt-5.4-nano-2026-03-05",
    ]


# ---------------------------------------------------------------------------
# Comparability grouping, status and rollup
# ---------------------------------------------------------------------------


def test_each_judge_gets_its_own_comparability_group(facts):
    """`synthetic/two-judge` is scored on Omni-MATH by two judges. OpenEval
    reports both (0.90 GPT, 0.50 Claude); Scale AI reports the GPT one only
    (0.88). Pooled across judges the two organisations read 0.70 vs 0.88 and
    the group flags; per judge they read 0.90 vs 0.88 and agree. The judge is
    part of the measurement, so it is part of the key."""
    rows = _rows(facts, """
        SELECT metric_source_label, comparability_group_id,
               has_cross_party_divergence
        FROM fact_results
        WHERE benchmark_key = 'omni-math' AND model_raw = 'synthetic/two-judge'
        ORDER BY metric_source_label, score
    """)
    gpt_group = {r[1] for r in rows if r[0] == "gpt_correctness"}
    claude_group = {r[1] for r in rows if r[0] == "claude_correctness"}
    assert len(gpt_group) == 1 and len(claude_group) == 1
    assert gpt_group != claude_group
    # both organisations report the GPT judge and agree within 0.05
    assert [r[2] for r in rows if r[0] == "gpt_correctness"] == [False, False]
    # the Claude judge has one reporter, so cross-party does not apply — and
    # it is emphatically not "the other judge disagreed"
    assert [r[2] for r in rows if r[0] == "claude_correctness"] == [None]
    # the pooled view the old key would have produced
    [(pooled,)] = _rows(facts, """
        SELECT max(m) - min(m) FROM (
            SELECT median(score_canonical) AS m FROM fact_results
            WHERE benchmark_key = 'omni-math'
              AND model_raw = 'synthetic/two-judge'
            GROUP BY org_raw)
    """)
    assert pooled > 0.05


def test_canonical_scores_make_one_homogeneous_group_per_condition(facts):
    """The four WildBench channels of one record publish three 1-10 ratings
    and the source's own 0-1 rescale. After the curated (x-1)/9 they are four
    conditions on one [0,1] metric — every group assessable, none of them
    pooling a raw rating with a rescaled one."""
    rows = _rows(facts, """
        SELECT comparability_group_id, judge_condition, comparability_status,
               min(score_canonical), max(score_canonical)
        FROM fact_results
        WHERE benchmark_key = 'wildbench' AND model_raw = 'openai/gpt-4o'
        GROUP BY 1, 2, 3
    """)
    assert len(rows) == 4
    assert len({r[0] for r in rows}) == 4
    assert {r[2] for r in rows} == {"ok"}
    assert all(0.0 <= lo <= hi <= 1.0 for *_, lo, hi in rows)


def test_two_record_bounds_on_an_unbounded_metric_is_mixed_scale(facts):
    """`rating` has no registry bounds, so the record's own declaration is all
    there is — and the Claude channel declares [0,10] on one number and
    [0,100] on the next. Two scales in one pool is not a disagreement about
    the model, so nothing is compared."""
    rows = _rows(facts, """
        SELECT score, comparability_status, has_variant_divergence,
               has_cross_party_divergence, variant_divergence_threshold,
               variant_divergence_magnitude
        FROM fact_results
        WHERE model_raw = 'synthetic/mixed-bounds'
          AND metric_source_label = 'claude_rating'
        ORDER BY score
    """)
    assert [r[0] for r in rows] == [0.5, 40.0]
    assert all(r[1] == "mixed_scale" for r in rows)
    assert all(r[2] is None and r[3] is None for r in rows)
    assert all(r[4] is None and r[5] is None for r in rows)
    # the GPT channel of the same record agrees on [0,10] and IS assessed
    [(status, flag, threshold, basis)] = _rows(facts, """
        SELECT DISTINCT comparability_status, has_variant_divergence,
               variant_divergence_threshold, variant_threshold_basis
        FROM fact_results
        WHERE model_raw = 'synthetic/mixed-bounds'
          AND metric_source_label = 'gpt_rating'
    """)
    assert (status, flag, basis) == ("ok", False, "range_5pct")
    assert threshold == pytest.approx(0.5)   # 5% of the group's own [0,10]


def test_no_declared_bounds_anywhere_is_no_bounds(facts):
    """Same unbounded metric, and this record declares no range either. The
    two rows differ on temperature and would otherwise be a variant-divergence
    candidate; with no scale to measure against there is no threshold to
    compare a difference to."""
    rows = _rows(facts, """
        SELECT comparability_status, has_variant_divergence,
               variant_divergence_threshold
        FROM fact_results WHERE model_raw = 'synthetic/unbounded'
    """)
    assert len(rows) == 2
    assert all(r == ("no_bounds", None, None) for r in rows)


def test_one_declared_pair_beside_a_bare_row_is_not_ok(facts):
    """Partial bounds are not bounds. Same unbounded metric again, and this
    record declares [0,10] on one row and nothing on the next. A missing pair
    is missing, not agreement by silence: `isfinite(NULL)` is NULL, so without
    the COALESCE guard the bare row would vanish from the missing count and
    the group would be assessed against the one pair that happens to be there.
    """
    rows = _rows(facts, """
        SELECT min_score, max_score, comparability_status,
               has_variant_divergence, variant_divergence_threshold
        FROM fact_results WHERE model_raw = 'synthetic/partial-bounds'
        ORDER BY score
    """)
    assert len(rows) == 2
    assert [(lo, hi) for lo, hi, *_ in rows] == [(0.0, 10.0), (None, None)]
    assert all(r[2] == "mixed_scale" for r in rows)
    assert all(r[3] is None and r[4] is None for r in rows)


def test_registry_bounds_win_over_differing_record_bounds(facts):
    """Two organisations report `accuracy` on the same model: OpenEval
    declares [0,1] and Scale AI [0,100], but the registry bounds the metric
    itself, so the group is assessed on [0,1] and their 0.9 vs 0.5 is a real
    disagreement."""
    rows = _rows(facts, """
        SELECT org_raw, score, min_score, max_score, comparability_status,
               has_cross_party_divergence, cross_party_divergence_threshold,
               cross_party_threshold_basis, comparability_group_id
        FROM fact_results
        WHERE benchmark_key = 'disinfo-bench' AND metric_key = 'accuracy'
        ORDER BY org_raw
    """)
    assert [r[0] for r in rows] == ["OpenEval", "Scale AI"]
    assert [r[1] for r in rows] == [0.9, 0.5]
    # the record-declared [0,100] never reaches the row: registry bounds win
    assert all((r[2], r[3]) == (0.0, 1.0) for r in rows)
    assert len({r[8] for r in rows}) == 1
    assert all(r[4] == "ok" and r[5] is True for r in rows)
    assert all(r[7] == "proportion" and r[6] == pytest.approx(0.05) for r in rows)


def test_a_not_assessable_group_nulls_the_triple_rollup(facts):
    """`synthetic/mixed-bounds` contributes one assessed-and-agreeing group
    (the GPT channel) and one mixed-scale one (the Claude channel, which
    declares [0,10] then [0,100]). The view now has a judge axis, so each is
    its own row, and neither may report agreement about numbers nobody could
    compare."""
    # Keyed on the judge label inside the condition, not on
    # `metric_source_label`: these cells are two runs with no submitted
    # aggregate, so they carry no value and therefore no label for one.
    rows = dict(_rows(facts, """
        SELECT json_extract_string(judge_condition, '$.label'),
               (comparability_status, has_variant_divergence,
                has_cross_party_divergence)
        FROM eval_results_view WHERE model_key = 'synthetic/mixed-bounds'
    """))
    assert rows["gpt_rating"] == ("ok", False, None)
    status, variant, cross = rows["claude_rating"]
    assert status == "mixed_scale"
    assert variant is None and cross is None
    # the contributing groups: one FALSE, one NULL — a plain BOOL_OR would
    # have dropped the NULL and published FALSE
    flags = _rows(facts, """
        SELECT DISTINCT comparability_group_id, has_variant_divergence
        FROM fact_results WHERE model_raw = 'synthetic/mixed-bounds'
    """)
    assert sorted(f for _, f in flags if f is not None) == [False]
    assert any(f is None for _, f in flags)


def test_no_comparability_group_mixes_judge_conditions(facts):
    """A comparability group is one measurement: one model, benchmark,
    slice, metric, protocol point and judge. Checked over the whole corpus."""
    [(violations,)] = _rows(facts, """
        SELECT count(*) FROM (
            SELECT comparability_group_id FROM fact_results
            WHERE comparability_group_id IS NOT NULL
            GROUP BY 1 HAVING count(DISTINCT judge_condition) > 1
               OR (count(DISTINCT judge_condition) = 1
                   AND count(*) FILTER (WHERE judge_condition IS NULL) > 0
                   AND count(*) FILTER (WHERE judge_condition IS NOT NULL) > 0)
        )
    """)
    assert violations == 0


# ---------------------------------------------------------------------------
# Dedupe, headline mapping and ranking
# ---------------------------------------------------------------------------


def test_three_of_the_six_collisions_collapse_in_the_view(facts, caplog):
    """The three agreeing raw-vs-rescaled pairs are one measurement each, so
    the converted copy is dropped before the view is built. The three that
    disagree keep both rows."""
    rows = dict(_rows(facts, """
        SELECT model_key, ARRAY_AGG(metric_source_label ORDER BY metric_source_label)
        FROM eval_results_view
        WHERE benchmark_id = 'wildbench' AND model_key LIKE 'openai/gpt-5%'
        GROUP BY 1
    """))
    collapsed = sorted(m for m, labels in rows.items() if len(labels) == 1)
    assert collapsed == [
        "openai/gpt-5.4-2026-03-05",
        "openai/gpt-5.4-mini-2026-03-05",
        "openai/gpt-5.4-nano-2026-03-05",
    ]
    # the survivor is the copy that needed no conversion
    assert all(labels == ["wildbench_score_rescaled"]
               for labels in rows.values() if len(labels) == 1)
    # and the facts themselves are untouched — dedupe is a view-layer rule
    [(n_facts,)] = _rows(facts, """
        SELECT count(*) FROM fact_results
        WHERE benchmark_key = 'wildbench' AND model_raw LIKE 'openai/gpt-5%'
    """)
    assert n_facts == 12
    # the collapse is counted, not silent
    assert any("dedupe dropped 3 converted scale-copy fact row(s)"
               in r.getMessage() for r in caplog.get_records("setup"))


def test_the_panel_is_the_headline_and_its_members_stay_unranked(facts):
    """Standard four-channel WildBench: three single judges plus the
    three-judge aggregate. The panel is the page's reading of the model; the
    singles stay visible and unranked."""
    rows = dict(_rows(facts, """
        SELECT metric_source_label, (is_headline, position)
        FROM eval_results_view
        WHERE benchmark_id = 'wildbench' AND model_key = 'openai/gpt-4o'
    """))
    assert rows["wildbench_score_rescaled"][0] is True
    assert rows["wildbench_score_rescaled"][1] is not None
    for label in ("gpt_score", "claude_score", "llama_score"):
        assert rows[label] == (False, None)


def test_the_two_judge_cohort_aggregate_wins_its_page(facts):
    """Omni-MATH's newer cohort publishes a two-judge aggregate beside two
    single-judge channels — and the same judge twice under two labels, which
    stay two distinct rows."""
    rows = dict(_rows(facts, """
        SELECT metric_source_label, (is_headline, score)
        FROM eval_results_view
        WHERE benchmark_id = 'omni-math' AND model_key = 'openai/gpt-5-mini'
    """))
    assert set(rows) == {"omni_math_correctness", "gpt_correctness",
                         "gpt-4o_correctness"}
    assert rows["omni_math_correctness"][0] is True
    assert rows["gpt_correctness"][0] is False
    assert rows["gpt-4o_correctness"][0] is False
    # same judge, two labels, two published numbers — never merged
    assert rows["gpt_correctness"][1] != rows["gpt-4o_correctness"][1]


def test_the_judge_with_the_widest_coverage_headlines_the_safety_page(facts):
    """Every model on the OpenEval safety page is judged by GPT-4o; only one
    is also judged by Llama. The judge that reads for the whole page is the
    page's reading; the Llama row stays visible and unranked."""
    rows = dict(_rows(facts, """
        SELECT (model_key, metric_source_label), (is_headline, position, total)
        FROM eval_results_view
        WHERE benchmark_id = 'harmbench'
          AND composite_slug = 'fixtures-judges--openeval'
    """))
    assert rows[("safety/model-a", "gpt_refusal")] == (True, 1, 2)
    assert rows[("safety/model-a", "llama_refusal")] == (False, None, 2)
    assert rows[("safety/model-b", "gpt_refusal")] == (True, 2, 2)


def test_an_undisclosed_judge_is_still_a_headline(facts):
    """HELM reports the same benchmark without naming a judge. It is its own
    page, so the NULL condition has nothing to lose to."""
    [(judge, headline, position)] = _rows(facts, """
        SELECT judge_condition, is_headline, position
        FROM eval_results_view
        WHERE benchmark_id = 'harmbench'
          AND composite_slug = 'fixtures-judges--helm'
    """)
    assert judge is None
    assert headline is True and position == 1


def test_coverage_is_per_judge_set_not_per_label(snapshot):
    """One judge under two labels is one judge's coverage of the page. When
    the raw channel is published for MORE models than the source's own
    rescaled copy, counting the label with the judge would hand the headline
    to the raw channel; counting the judge set leaves the two tied, and the
    number that needed no conversion wins."""
    def add_raw_only_models(con):
        # two models that only ever get the raw gpt_score channel, so the
        # `gpt_score` LABEL out-covers `wildbench_score_rescaled`
        for i in range(2):
            con.execute(
                "INSERT INTO fact_results SELECT * REPLACE ("
                f"  'synthetic/raw-only-{i}' AS model_aggregation_key,"
                f"  'synthetic/raw-only-{i}' AS model_raw,"
                f"  'synthetic/raw-only-{i}' AS model_key,"
                f"  'raw-only-{i}-fact' AS fact_id"
                ") FROM fact_results"
                " WHERE benchmark_key = 'wildbench'"
                "   AND model_raw = 'synthetic/endpoint-hi'"
            )

    con = _restage_j(snapshot, add_raw_only_models)
    by_label = dict(_rows(con, """
        SELECT metric_source_label, COUNT(DISTINCT model_key)
        FROM eval_results_view
        WHERE benchmark_id = 'wildbench' AND score IS NOT NULL
          AND metric_source_label IN ('gpt_score', 'wildbench_score_rescaled')
          AND json_array_length(json_extract(judge_condition, '$.judges')) = 1
        GROUP BY 1
    """))
    assert by_label["gpt_score"] > by_label["wildbench_score_rescaled"]

    # the two channels of a near-miss model: one judge, two labels, one
    # converted and one not
    rows = dict(_rows(con, """
        SELECT metric_source_label,
               (json_extract(judge_condition, '$.judges')::VARCHAR,
                scale_conversion, is_headline)
        FROM eval_results_view
        WHERE benchmark_id = 'wildbench'
          AND model_key = 'openai/gpt-5.2-2025-11-04'
    """))
    assert (rows["gpt_score"][0]
            == rows["wildbench_score_rescaled"][0]
            == f'["{GPT4O}"]')
    assert rows["gpt_score"][1:] == ("curated", False)
    assert rows["wildbench_score_rescaled"][1:] == ("none", True)


def test_protocol_arms_and_judges_resolve_to_one_headline(snapshot):
    """Collection-shaped rows: three protocol arms, one of them assisted, and
    two judge conditions inside the winning arm. The assisted arm is never the
    headline, the arms are settled first, and the panel wins inside the arm
    that represents the model."""
    arm_full = json.dumps({"feedback": "none", "variant": "full"},
                          sort_keys=True, separators=(",", ":"))
    arm_narrow = json.dumps({"feedback": "none", "variant": "narrow"},
                            sort_keys=True, separators=(",", ":"))
    arm_assisted = json.dumps({"feedback": "answer_feedback", "variant": "full"},
                              sort_keys=True, separators=(",", ":"))
    panel = _condition([GPT4O, CLAUDE], "panel_refusal")

    def add_protocol_arms(con):
        for fact_id, protocol, label, judge, score in (
            ("arm-full-gpt", arm_full, "gpt_refusal",
             _condition([GPT4O], "gpt_refusal"), 0.5),
            ("arm-full-panel", arm_full, "panel_refusal", panel, 0.45),
            ("arm-narrow-gpt", arm_narrow, "gpt_refusal",
             _condition([GPT4O], "gpt_refusal"), 0.4),
            ("arm-assisted-gpt", arm_assisted, "gpt_refusal",
             _condition([GPT4O], "gpt_refusal"), 0.9),
        ):
            con.execute(
                "INSERT INTO fact_results SELECT * REPLACE ("
                "  'safety/protocol-model' AS model_aggregation_key,"
                "  'safety/protocol-model' AS model_raw,"
                "  'safety/protocol-model' AS model_key,"
                f"  '{fact_id}' AS fact_id,"
                f"  '{protocol}' AS protocol_condition,"
                f"  '{judge}' AS judge_condition,"
                f"  '{label}' AS metric_source_label,"
                f"  {score} AS score,"
                f"  {score} AS score_canonical"
                ") FROM fact_results WHERE fact_id = ("
                "  SELECT min(fact_id) FROM fact_results"
                "  WHERE benchmark_key = 'harmbench'"
                "    AND model_raw = 'safety/model-b')"
            )
    con = _restage_j(snapshot, add_protocol_arms)
    rows = dict(_rows(con, """
        SELECT (protocol_condition, metric_source_label),
               (is_headline, position)
        FROM eval_results_view
        WHERE model_key = 'safety/protocol-model'
    """))
    assert len(rows) == 4
    assert sum(1 for h, _ in rows.values() if h) == 1
    # the assisted arm is never the headline and is never ranked
    assert rows[(arm_assisted, "gpt_refusal")] == (False, None)
    # the full arm represents the model (0.5 beats the narrow arm's 0.4) and
    # the two-judge panel is the reading inside it
    assert rows[(arm_full, "panel_refusal")][0] is True
    assert rows[(arm_full, "panel_refusal")][1] is not None
    assert rows[(arm_full, "gpt_refusal")] == (False, None)
    assert rows[(arm_narrow, "gpt_refusal")] == (False, None)


# ---------------------------------------------------------------------------
# Corpus invariants
# ---------------------------------------------------------------------------


def test_exactly_one_headline_per_page_cell(facts):
    """One reading per (composite, benchmark, metric, model) — counted over
    every cell, not only the scored ones: a cell nobody scored keeps a
    representative row so coverage rollups still see it."""
    [(violations,)] = _rows(facts, """
        SELECT count(*) FROM (
            SELECT composite_slug, benchmark_id, metric_id, model_key
            FROM eval_results_view
            GROUP BY 1, 2, 3, 4
            HAVING count(*) FILTER (WHERE is_headline) <> 1
        )
    """)
    assert violations == 0


def test_non_headline_rows_are_never_ranked(facts):
    [(violations,)] = _rows(facts, """
        SELECT count(*) FROM eval_results_view
        WHERE NOT is_headline
          AND (position IS NOT NULL OR percentile IS NOT NULL)
    """)
    assert violations == 0


# ---------------------------------------------------------------------------
# Review follow-ups (notes/issue47/REVIEW-backend-diff.md)
# ---------------------------------------------------------------------------


def test_an_offset_only_conversion_is_curated(facts):
    """The registry contract allows a conversion that is factor OR offset.
    CNN/DailyMail's scoped rule carries an offset and no factor: the factor
    defaults to 1 rather than the rule falling through to scale detection."""
    [(published, canonical, conversion, metric)] = _rows(facts, """
        SELECT score, score_canonical, scale_conversion, metric_key
        FROM fact_results WHERE benchmark_key = 'cnn-dailymail'
    """)
    assert metric == "summarization-score"
    assert conversion == "curated"
    assert published == 0.62
    assert canonical == pytest.approx(0.82)


def test_standard_deviation_is_extracted_and_converted(facts):
    """EEE publishes a standard deviation beside the standard error. It is a
    width, so it takes the factor's magnitude and no offset — and it must not
    be dropped on the way through Stage D."""
    [(sd, sd_c)] = _rows(facts, """
        SELECT score_sd, score_sd_canonical FROM fact_results
        WHERE benchmark_key = 'wildbench' AND model_raw = 'openai/gpt-4o'
          AND metric_source_label = 'gpt_score'
    """)
    assert sd == 2.7
    assert sd_c == pytest.approx(2.7 * FACTOR)
    # and it reaches the view, on the same displayed scale as the score: the
    # WildBench conversion is curated, so score_details shows the canonical SD
    [(details, sd_view)] = _rows(facts, """
        SELECT score_details, score_sd_canonical FROM eval_results_view
        WHERE benchmark_id = 'wildbench' AND model_key = 'openai/gpt-4o'
          AND metric_source_label = 'gpt_score'
    """)
    assert details["standard_deviation"] == pytest.approx(2.7 * FACTOR)
    assert sd_view == pytest.approx(2.7 * FACTOR)


def test_a_pair_with_one_missing_response_count_is_not_collapsed(snapshot):
    """Dedupe is strict: two absent counts are one measurement reported
    twice, but exactly one absent means we cannot tell the runs apart. A
    wildcard there would delete an independent evaluation that merely
    agrees."""
    def drop_one_response_count(con):
        # gpt-5.4-nano is one of the three agreeing pairs; strip the count
        # from its raw channel only.
        con.execute(
            "UPDATE fact_results SET metric_additional_details = "
            "  CAST(json_merge_patch(metric_additional_details, "
            "                        '{\"response_count\": null}') AS VARCHAR) "
            "WHERE benchmark_key = 'wildbench' "
            "  AND model_raw = 'openai/gpt-5.4-nano-2026-03-05' "
            "  AND metric_source_label = 'gpt_score'"
        )

    con = _restage_j(snapshot, drop_one_response_count)
    labels = sorted(r[0] for r in _rows(con, """
        SELECT metric_source_label FROM eval_results_view
        WHERE benchmark_id = 'wildbench'
          AND model_key = 'openai/gpt-5.4-nano-2026-03-05'
    """))
    assert labels == ["gpt_score", "wildbench_score_rescaled"]
    # the two pairs that still carry matching counts do collapse
    still_collapsed = _rows(con, """
        SELECT count(*) FROM eval_results_view
        WHERE benchmark_id = 'wildbench'
          AND model_key = 'openai/gpt-5.4-mini-2026-03-05'
    """)
    assert still_collapsed == [(1,)]


def test_both_response_counts_absent_still_collapses(snapshot):
    """The other half of the same rule: neither row names a count, so nothing
    distinguishes them and the converted copy is dropped."""
    def drop_both_response_counts(con):
        con.execute(
            "UPDATE fact_results SET metric_additional_details = "
            "  CAST(json_merge_patch(metric_additional_details, "
            "                        '{\"response_count\": null}') AS VARCHAR) "
            "WHERE benchmark_key = 'wildbench' "
            "  AND model_raw = 'openai/gpt-5.4-nano-2026-03-05'"
        )

    con = _restage_j(snapshot, drop_both_response_counts)
    labels = [r[0] for r in _rows(con, """
        SELECT metric_source_label FROM eval_results_view
        WHERE benchmark_id = 'wildbench'
          AND model_key = 'openai/gpt-5.4-nano-2026-03-05'
    """)]
    assert labels == ["wildbench_score_rescaled"]


def test_one_disclosed_judge_does_not_outrank_wider_undisclosed_coverage(
    snapshot
):
    """Judge cardinality is a preference for PANELS. One disclosed judge and
    an undisclosed condition say equally little about comparability, so the
    tie goes to whichever reads for more of the page."""
    def add_two_conditions(con):
        # five models carry the undisclosed condition, two carry a single
        # disclosed judge, all on one (composite, benchmark, metric) page
        for i in range(5):
            con.execute(
                "INSERT INTO fact_results SELECT * REPLACE ("
                f"  'cover/model-{i}' AS model_aggregation_key,"
                f"  'cover/model-{i}' AS model_raw,"
                f"  'cover/model-{i}' AS model_key,"
                f"  'cover-null-{i}' AS fact_id,"
                "   CAST(NULL AS VARCHAR) AS judge_condition,"
                "   'undisclosed' AS metric_source_label,"
                "   0.5 AS score, 0.5 AS score_canonical"
                ") FROM fact_results WHERE fact_id = ("
                "  SELECT min(fact_id) FROM fact_results"
                "  WHERE benchmark_key = 'harmbench'"
                "    AND model_raw = 'safety/model-b')"
            )
        for i in range(2):
            con.execute(
                "INSERT INTO fact_results SELECT * REPLACE ("
                f"  'cover/model-{i}' AS model_aggregation_key,"
                f"  'cover/model-{i}' AS model_raw,"
                f"  'cover/model-{i}' AS model_key,"
                f"  'cover-gpt-{i}' AS fact_id,"
                f"  '{_condition([GPT4O], 'gpt_refusal')}' AS judge_condition,"
                "   'gpt_refusal' AS metric_source_label,"
                "   0.9 AS score, 0.9 AS score_canonical"
                ") FROM fact_results WHERE fact_id = ("
                "  SELECT min(fact_id) FROM fact_results"
                "  WHERE benchmark_key = 'harmbench'"
                "    AND model_raw = 'safety/model-b')"
            )

    con = _restage_j(snapshot, add_two_conditions)
    rows = dict(_rows(con, """
        SELECT metric_source_label, (is_headline, position)
        FROM eval_results_view WHERE model_key = 'cover/model-0'
    """))
    assert rows["undisclosed"][0] is True
    assert rows["undisclosed"][1] is not None
    assert rows["gpt_refusal"] == (False, None)


def test_a_disclosed_panel_still_beats_wider_single_judge_coverage(snapshot):
    """The panel preference survives the fix: cardinality > 1 is ordered on
    before coverage, so a two-judge panel represents the model even when a
    single judge covers more of the page."""
    panel = _condition([GPT4O, CLAUDE], "panel_refusal")

    def add_panel_and_singles(con):
        for i in range(4):
            con.execute(
                "INSERT INTO fact_results SELECT * REPLACE ("
                f"  'panel/model-{i}' AS model_aggregation_key,"
                f"  'panel/model-{i}' AS model_raw,"
                f"  'panel/model-{i}' AS model_key,"
                f"  'panel-single-{i}' AS fact_id,"
                f"  '{_condition([GPT4O], 'gpt_refusal')}' AS judge_condition,"
                "   'gpt_refusal' AS metric_source_label,"
                "   0.5 AS score, 0.5 AS score_canonical"
                ") FROM fact_results WHERE fact_id = ("
                "  SELECT min(fact_id) FROM fact_results"
                "  WHERE benchmark_key = 'harmbench'"
                "    AND model_raw = 'safety/model-b')"
            )
        con.execute(
            "INSERT INTO fact_results SELECT * REPLACE ("
            "  'panel/model-0' AS model_aggregation_key,"
            "  'panel/model-0' AS model_raw,"
            "  'panel/model-0' AS model_key,"
            "  'panel-panel-0' AS fact_id,"
            f"  '{panel}' AS judge_condition,"
            "   'panel_refusal' AS metric_source_label,"
            "   0.4 AS score, 0.4 AS score_canonical"
            ") FROM fact_results WHERE fact_id = ("
            "  SELECT min(fact_id) FROM fact_results"
            "  WHERE benchmark_key = 'harmbench'"
            "    AND model_raw = 'safety/model-b')"
        )

    con = _restage_j(snapshot, add_panel_and_singles)
    rows = dict(_rows(con, """
        SELECT metric_source_label, is_headline
        FROM eval_results_view WHERE model_key = 'panel/model-0'
    """))
    assert rows["panel_refusal"] is True
    assert rows["gpt_refusal"] is False


def test_every_safety_identity_headlines_the_wider_judge(facts):
    """The coverage rule is not HarmBench-specific: all four benchmark-scoped
    safety identities carry GPT-4o for both models and Llama for one, and on
    each page the judge that reads for the whole page is the page's reading."""
    rows = _rows(facts, """
        SELECT benchmark_id, model_key, metric_source_label, is_headline
        FROM eval_results_view
        WHERE composite_slug = 'fixtures-judges--openeval'
          AND benchmark_id IN ('harmbench', 'xstest', 'simplesafetytests',
                               'anthropic-red-team')
        ORDER BY 1, 2, 3
    """)
    by_bench: dict[str, dict] = {}
    for bench, model, label, headline in rows:
        by_bench.setdefault(bench, {})[(model, label)] = headline
    assert sorted(by_bench) == ["anthropic-red-team", "harmbench",
                                "simplesafetytests", "xstest"]
    for bench, cells in by_bench.items():
        assert cells[("safety/model-a", "gpt_refusal")] is True, bench
        assert cells[("safety/model-a", "llama_refusal")] is False, bench
        assert cells[("safety/model-b", "gpt_refusal")] is True, bench


def test_the_old_omni_math_cohort_headlines_its_three_judge_aggregate(facts):
    """The 2024 cohort: Claude, GPT-4o and Llama singles plus the three-judge
    aggregate. The panel is the page's reading of the model and the three
    singles stay visible and unranked."""
    rows = dict(_rows(facts, """
        SELECT metric_source_label, (is_headline, position, score)
        FROM eval_results_view
        WHERE benchmark_id = 'omni-math' AND model_key = 'openai/gpt-4o-mini'
    """))
    assert sorted(rows) == ["claude_correctness", "gpt_correctness",
                            "llama_correctness", "omni_math_correctness"]
    assert rows["omni_math_correctness"][0] is True
    assert rows["omni_math_correctness"][1] is not None
    assert rows["omni_math_correctness"][2] == 0.42
    for label in ("claude_correctness", "gpt_correctness", "llama_correctness"):
        assert rows[label][:2] == (False, None), label
    [(judges,)] = _rows(facts, """
        SELECT json_extract(judge_condition, '$.judges')::VARCHAR
        FROM eval_results_view
        WHERE benchmark_id = 'omni-math' AND model_key = 'openai/gpt-4o-mini'
          AND metric_source_label = 'omni_math_correctness'
    """)
    assert json.loads(judges) == sorted([CLAUDE, GPT4O, LLAMA])


def test_the_condition_grain_table_projects_each_column_once(snapshot):
    """A repeated `alias.*` binds without complaint and DuckDB auto-renames
    the copy (`a` → `a_1`), doubling the materialised wide aggregate. The
    final view masks it, so the check has to be on the intermediate."""
    con = _restage_j(snapshot)
    for table in ("_erv_tri", "eval_results_view"):
        cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
        auto_renamed = [
            c for c in cols
            if c.rsplit("_", 1)[-1].isdigit() and c.rsplit("_", 1)[0] in cols
        ]
        assert auto_renamed == [], table
        assert len(cols) == len(set(cols)), table


def test_a_sliced_curated_score_shows_on_the_canonical_scale(snapshot):
    """A subtask scalar comes straight off the fact rows, so it has to apply
    the same displayed-score rule the view does: a curated 1-10 rating must
    not surface as `9` beneath a renamed [0, 1] metric."""
    def add_a_slice(con):
        # its own model, so the sliced row is the model's only condition and
        # therefore its headline; the slice sits BELOW the condition grain
        con.execute(
            "INSERT INTO fact_results SELECT * REPLACE ("
            "  'sliced/model' AS model_aggregation_key,"
            "  'sliced/model' AS model_raw,"
            "  'sliced/model' AS model_key,"
            "  'wb-slice-fact' AS fact_id,"
            "  'creative' AS slice_key,"
            "  'Creative writing' AS slice_name,"
            "  9.0 AS score,"
            f"  {9.0 * FACTOR + OFFSET} AS score_canonical,"
            "  'curated' AS scale_conversion"
            ") FROM fact_results WHERE benchmark_key = 'wildbench'"
            "   AND model_raw = 'openai/gpt-4o'"
            "   AND metric_source_label = 'gpt_score'"
        )

    con = _restage_j(snapshot, add_a_slice, full=True)
    [(subtasks,)] = _rows(con, """
        SELECT subtasks FROM evals_view WHERE benchmark_id = 'wildbench'
    """)
    creative = [s for s in subtasks if s["subtask_key"] == "creative"]
    assert len(creative) == 1
    [metric] = creative[0]["metrics"]
    assert metric["metric_key"] == "wb-score"
    assert metric["top_score"] == pytest.approx(9.0 * FACTOR + OFFSET)


def test_the_pre_pivot_leaderboard_is_headline_only(snapshot):
    """A cell whose only row is an assisted run has no headline, and the
    pre-pivoted leaderboard must not publish it: the ranking, the summaries
    and the comparison index all exclude it, so a value here would be a score
    that exists nowhere else on the page."""
    assisted = json.dumps({"feedback": "answer_feedback"},
                          sort_keys=True, separators=(",", ":"))

    def add_an_assisted_only_model(con):
        con.execute(
            "INSERT INTO fact_results SELECT * REPLACE ("
            "  'assisted/only' AS model_aggregation_key,"
            "  'assisted/only' AS model_raw,"
            "  'assisted/only' AS model_key,"
            "  'assisted-only-fact' AS fact_id,"
            f"  '{assisted}' AS protocol_condition,"
            "  0.99 AS score, 0.99 AS score_canonical"
            ") FROM fact_results WHERE fact_id = ("
            "  SELECT min(fact_id) FROM fact_results"
            "  WHERE benchmark_key = 'harmbench'"
            "    AND model_raw = 'safety/model-b')"
        )

    con = _restage_j(snapshot, add_an_assisted_only_model, full=True)
    # the row is in the view, unranked and not a headline
    assert _rows(con, """
        SELECT is_headline, position FROM eval_results_view
        WHERE model_key = 'assisted/only'
    """) == [(False, None)]
    [(rows,)] = _rows(con, """
        SELECT leaderboard_rows FROM evals_view
        WHERE benchmark_id = 'harmbench'
          AND composite_slug = 'fixtures-judges--openeval'
    """)
    assert "assisted/only" not in {
        (r["model_info"] or {}).get("id") for r in rows
    }


def test_a_rollup_mixing_ok_and_mixed_scale_slices_is_null(snapshot):
    """One condition, two slices: one group agrees, the other could not be
    compared at all. They roll up into ONE view row, and a rollup containing
    a non-`ok` group never publishes FALSE."""
    def add_two_slices(con):
        con.execute(
            "INSERT INTO fact_results SELECT * REPLACE ("
            "  'rollup/model' AS model_aggregation_key,"
            "  'rollup/model' AS model_raw,"
            "  'rollup/model' AS model_key,"
            "  'rollup-ok-fact' AS fact_id,"
            "  'slice-ok' AS slice_key, 'Slice OK' AS slice_name,"
            "  'ok' AS comparability_status,"
            "  FALSE AS has_variant_divergence,"
            "  'rollup-group-ok' AS comparability_group_id"
            ") FROM fact_results WHERE fact_id = ("
            "  SELECT min(fact_id) FROM fact_results"
            "  WHERE benchmark_key = 'harmbench'"
            "    AND model_raw = 'safety/model-b')"
        )
        con.execute(
            "INSERT INTO fact_results SELECT * REPLACE ("
            "  'rollup/model' AS model_aggregation_key,"
            "  'rollup/model' AS model_raw,"
            "  'rollup/model' AS model_key,"
            "  'rollup-mixed-fact' AS fact_id,"
            "  'slice-mixed' AS slice_key, 'Slice mixed' AS slice_name,"
            "  'mixed_scale' AS comparability_status,"
            "  CAST(NULL AS BOOLEAN) AS has_variant_divergence,"
            "  'rollup-group-mixed' AS comparability_group_id"
            ") FROM fact_results WHERE fact_id = ("
            "  SELECT min(fact_id) FROM fact_results"
            "  WHERE benchmark_key = 'harmbench'"
            "    AND model_raw = 'safety/model-b')"
        )

    con = _restage_j(snapshot, add_two_slices)
    rows = _rows(con, """
        SELECT comparability_status, has_variant_divergence
        FROM eval_results_view WHERE model_key = 'rollup/model'
    """)
    # both slices sit under one (composite, benchmark, metric, model,
    # protocol, judge) row — that is what makes the rollup non-trivial
    assert len(rows) == 1
    assert rows[0] == ("mixed_scale", None)


def test_the_fact_parquet_carries_the_headline_flag(snapshot):
    """A consumer reading the facts directly (the frontend's build-time
    matrix) cannot re-derive the headline pick, so Stage J re-emits the facts
    with the flag attached. The pick is at CONDITION grain, so a cell's
    flagged facts are exactly the fact rows of its one headline condition —
    more than one row when a condition pooled reruns, never more than one
    condition."""
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE facts AS SELECT * FROM "
        f"read_parquet('{snapshot}/fact_results.parquet')"
    )
    [(nulls,)] = _rows(con, "SELECT count(*) FROM facts WHERE is_headline IS NULL")
    assert nulls == 0
    [(violations,)] = _rows(con, """
        SELECT count(*) FROM (
            SELECT composite_slug, benchmark_key, metric_key,
                   model_aggregation_key
            FROM facts WHERE score IS NOT NULL
            GROUP BY 1, 2, 3, 4
            HAVING count(DISTINCT (protocol_condition, judge_condition))
                   FILTER (WHERE is_headline) > 1
               OR count(*) FILTER (WHERE is_headline) = 0
        )
    """)
    assert violations == 0
    # the wildbench panel row is the flagged one, its members are not
    rows = dict(_rows(con, """
        SELECT metric_source_label, is_headline FROM facts
        WHERE benchmark_key = 'wildbench' AND model_raw = 'openai/gpt-4o'
    """))
    assert rows["wildbench_score_rescaled"] is True
    assert rows["gpt_score"] is False

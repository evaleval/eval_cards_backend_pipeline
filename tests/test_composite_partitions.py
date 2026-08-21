"""Composite org-partition pass (notes/composite-partition-spec.md).

Unit tests drive `stages._apply_composite_partitions` directly over
hand-built `fact_results_signaled` / `composite_config_map` tables — the
function only touches the composite columns plus its helper inputs, so a
narrow table exercises every branch without a pipeline run.

The end-to-end wiring (Stage D precedence joins → Stage E re-key →
warehouse) is covered by the fixture-corpus tests
(`fixtures_xparty` in test_stage_j_models_view) and the curated-merge
e2e test at the bottom of this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from eval_card_backend.canonicalise import stages


def _mk_tables(con, fact_rows, map_rows=()):
    """fact_rows: (source_config, org_token, source_slug, curated,
    composite_slug, composite_display_name, org_display, payload)."""
    con.execute(
        "CREATE TABLE fact_results_signaled ("
        "source_config VARCHAR, org_token VARCHAR, "
        "_curated_source_slug VARCHAR, _composite_curated BOOLEAN, "
        "composite_slug VARCHAR, composite_display_name VARCHAR, "
        "org_display VARCHAR, payload VARCHAR)"
    )
    con.executemany(
        "INSERT INTO fact_results_signaled VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        fact_rows,
    )
    con.execute(
        "CREATE TABLE composite_config_map ("
        "source_config VARCHAR, org_token VARCHAR, source_slug VARCHAR, "
        "specificity TINYINT, composite_slug VARCHAR, "
        "composite_display_name VARCHAR)"
    )
    if map_rows:
        con.executemany(
            "INSERT INTO composite_config_map VALUES (?, ?, ?, ?, ?, ?)",
            map_rows,
        )


def _assignments(con):
    return con.execute(
        "SELECT source_config, NULL, composite_slug, "
        "composite_display_name, payload "
        "FROM fact_results_signaled ORDER BY payload"
    ).fetchall()


def _cols(con):
    return {
        r[1] for r in con.execute(
            "PRAGMA table_info('fact_results_signaled')"
        ).fetchall()
    }


def test_single_org_config_untouched():
    """Zero-churn: a single-org config keeps today's derivation exactly,
    junk unknown-org rows included (they don't trip the predicate)."""
    con = duckdb.connect()
    _mk_tables(con, [
        ("cfg", "acme", "acme-board", False, "cfg", "Acme Board", "Acme", "r1"),
        ("cfg", "acme", "acme-board", False, "cfg", "Acme Board", "Acme", "r2"),
        ("cfg", "unknown-org", "unlabeled", False, "cfg", "cfg", None, "r3"),
    ])
    stages._apply_composite_partitions(con, strict=True)
    rows = _assignments(con)
    assert [(r[2], r[3]) for r in rows] == [
        ("cfg", "Acme Board"), ("cfg", "Acme Board"), ("cfg", "cfg"),
    ]
    # Helper columns are dropped; the rest of the shape survives.
    assert _cols(con) == {
        "source_config", "composite_slug", "composite_display_name",
        "org_display", "payload",
    }


def test_multi_org_config_splits_no_bare_slug():
    """Every partition of a multi-org config gets the suffixed slug —
    no partition keeps the bare config slug (no majority-flapping)."""
    con = duckdb.connect()
    _mk_tables(con, [
        ("open_ai_cfg", "openai", "openai-page", False,
         "open-ai-cfg", "OpenAI Page", "OpenAI", "r1"),
        ("open_ai_cfg", "anthropic", "anthropic-card", False,
         "open-ai-cfg", "Anthropic Card", "Anthropic", "r2"),
    ])
    stages._apply_composite_partitions(con, strict=True)
    rows = _assignments(con)
    assert [(r[2], r[3]) for r in rows] == [
        ("open-ai-cfg--openai", "OpenAI Page"),
        ("open-ai-cfg--anthropic", "Anthropic Card"),
    ]


def test_unknown_org_partition_kept_visible():
    """In a config that is multi-org by NAMED tokens, unresolvable-org
    rows form their own `--unknown-org` partition — never silently
    lumped into a named org's page."""
    con = duckdb.connect()
    _mk_tables(con, [
        ("cfg", "a-org", "a-page", False, "cfg", "A Page", "A Org", "r1"),
        ("cfg", "b-org", "b-page", False, "cfg", "B Page", "B Org", "r2"),
        ("cfg", "unknown-org", "unlabeled", False, "cfg", "cfg", None, "r3"),
    ])
    stages._apply_composite_partitions(con, strict=True)
    rows = _assignments(con)
    assert [r[2] for r in rows] == [
        "cfg--a-org", "cfg--b-org", "cfg--unknown-org",
    ]


def test_display_fallback_for_label_heterogeneous_partition():
    """A partition with several distinct source labels (one org, several
    publications) has no true name in the data — display falls back to
    `<org display> — <source_config>`."""
    con = duckdb.connect()
    _mk_tables(con, [
        ("mrcr", "anthropic", "card-46", False, "mrcr", "Opus 4.6 Card",
         "Anthropic", "r1"),
        ("mrcr", "anthropic", "card-47", False, "mrcr", "Opus 4.7 Card",
         "Anthropic", "r2"),
        ("mrcr", "openai", "launch", False, "mrcr", "GPT Launch Page",
         "OpenAI", "r3"),
    ])
    stages._apply_composite_partitions(con, strict=True)
    rows = _assignments(con)
    assert [(r[2], r[3]) for r in rows] == [
        ("mrcr--anthropic", "Anthropic — mrcr"),
        ("mrcr--anthropic", "Anthropic — mrcr"),
        ("mrcr--openai", "GPT Launch Page"),
    ]


def test_curated_org_scope_absorbs_partition():
    """AISI-style merge: an org-scoped curated member claims its rows
    into the curated composite (already assigned by Stage D); the
    remaining org still auto-splits — its partition never keeps the
    bare slug."""
    con = duckdb.connect()
    _mk_tables(
        con,
        [
            ("hle", "uk-aisi", "aisi-study", True,
             "aisi-inference-scaling", "AISI study", "UK AISI", "r1"),
            ("hle", "scale-ai", "seal-page", False,
             "hle", "Scale SEAL HLE", "Scale", "r2"),
        ],
        map_rows=[
            ("hle", "uk-aisi", None, 2, "aisi-inference-scaling", "AISI study"),
        ],
    )
    stages._apply_composite_partitions(con, strict=True)
    rows = _assignments(con)
    assert [(r[2], r[3]) for r in rows] == [
        ("aisi-inference-scaling", "AISI study"),
        ("hle--scale-ai", "Scale SEAL HLE"),
    ]


def test_curated_config_wide_entry_prevents_split():
    """A config-wide curated entry absorbs all partitions: rows carry
    `_composite_curated` from Stage D, so a multi-org config with a
    config-level claim never enters the automatic split."""
    con = duckdb.connect()
    _mk_tables(
        con,
        [
            ("cfg", "a-org", "a-page", True, "curated", "Curated", "A", "r1"),
            ("cfg", "b-org", "b-page", True, "curated", "Curated", "B", "r2"),
        ],
        map_rows=[("cfg", None, None, 1, "curated", "Curated")],
    )
    stages._apply_composite_partitions(con, strict=True)
    rows = _assignments(con)
    assert {r[2] for r in rows} == {"curated"}


def test_scoped_member_zero_match_live_config_fails_strict():
    """A drifted org key on a live config is a hard error on full runs —
    it must not silently dump the study's rows back onto the automatic
    rule while the build passes."""
    con = duckdb.connect()
    _mk_tables(
        con,
        [("hle", "uk-aisi", "aisi-study", False, "hle", "Study", "AISI", "r1"),
         ("hle", "scale-ai", "seal", False, "hle", "SEAL", "Scale", "r2")],
        map_rows=[
            ("hle", "uk-aisl-typo", None, 2, "aisi-study", "AISI study"),
        ],
    )
    with pytest.raises(RuntimeError, match="drifted"):
        stages._apply_composite_partitions(con, strict=True)


def test_scoped_member_zero_match_non_strict_warns(caplog):
    """--configs subset runs legitimately omit configs; the guard
    downgrades to warnings."""
    con = duckdb.connect()
    _mk_tables(
        con,
        [("hle", "uk-aisi", "aisi-study", False, "hle", "Study", "AISI", "r1"),
         ("hle", "scale-ai", "seal", False, "hle", "SEAL", "Scale", "r2")],
        map_rows=[
            ("hle", "uk-aisl-typo", None, 2, "aisi-study", "AISI study"),
        ],
    )
    stages._apply_composite_partitions(con, strict=False)  # no raise


def test_scoped_member_config_left_corpus_warns_only():
    """Transition tolerance: when the config itself is gone, a detached
    scoped member is a warning, not an error — unless the whole entry
    matches nothing (fully-detached study)."""
    con = duckdb.connect()
    _mk_tables(
        con,
        [("healthbench", "uk-aisi", "aisi-study", True,
          "aisi-study", "AISI study", "AISI", "r1")],
        map_rows=[
            ("healthbench", "uk-aisi", None, 2, "aisi-study", "AISI study"),
            # hle left the corpus entirely: warning only, because the
            # entry still matches via healthbench.
            ("hle", "uk-aisi", None, 2, "aisi-study", "AISI study"),
        ],
    )
    stages._apply_composite_partitions(con, strict=True)  # no raise


def test_entry_with_no_matching_members_always_fails():
    con = duckdb.connect()
    _mk_tables(
        con,
        [("other-cfg", "acme", "x", False, "other-cfg", "X", "Acme", "r1")],
        map_rows=[
            ("hle", "uk-aisi", None, 2, "aisi-study", "AISI study"),
            ("healthbench", "uk-aisi", None, 2, "aisi-study", "AISI study"),
        ],
    )
    with pytest.raises(RuntimeError, match="fully detached"):
        stages._apply_composite_partitions(con, strict=True)


# ---------------------------------------------------------------------------
# End-to-end: curated merge through the full pipeline
# ---------------------------------------------------------------------------


def test_e2e_curated_org_scoped_merge(tmp_path, monkeypatch):
    """Two configs, each with a `Study Org` partition claimed by an
    org-scoped curated entry into one study composite (AISI idiom), plus
    an unclaimed org per config that auto-splits. Verifies Stage D
    precedence wiring, Stage E re-key, and the composites dim."""
    pytest.importorskip("yaml")
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from eee_layout import write_eee_datastore

    def _record(config, eval_id, org, source_name, score):
        return json.dumps({
            "evaluation_id": eval_id,
            "schema_version": "0.2.2",
            "retrieved_timestamp": "2026-04-30T00:00:00Z",
            "model_info": {
                "developer": "openai", "name": "GPT-4o",
                "id": "openai/gpt-4o", "inference_platform": "test",
            },
            "source_metadata": {
                "source_name": source_name, "source_type": "documentation",
                "source_organization_name": org,
                "evaluator_relationship": "third_party",
            },
            "eval_library": {"name": "minibench", "version": "1.0"},
            "evaluation_results": [{
                "evaluation_name": "minibench",
                "source_data": {"dataset_name": "minibench",
                                "source_type": "other"},
                "metric_config": {
                    "metric_id": "minibench.acc", "metric_name": "Accuracy",
                    "evaluation_description": "Accuracy on minibench",
                    "lower_is_better": False,
                },
                "score_details": {"score": score},
                "generation_config": {"generation_args": {
                    "temperature": 0.0, "max_tokens": 128}},
            }],
        })

    eee_root = tmp_path / "eee"
    write_eee_datastore(eee_root, [
        ("cfg_alpha", "ev_a1.json",
         _record("cfg_alpha", "ev_a1", "Study Org", "The Grand Study", 0.9)),
        ("cfg_alpha", "ev_a2.json",
         _record("cfg_alpha", "ev_a2", "Alpha Board", "Alpha Leaderboard", 0.8)),
        ("cfg_beta", "ev_b1.json",
         _record("cfg_beta", "ev_b1", "Study Org", "The Grand Study", 0.7)),
        ("cfg_beta", "ev_b2.json",
         _record("cfg_beta", "ev_b2", "Beta Board", "Beta Leaderboard", 0.6)),
    ])

    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "composites.yaml").write_text("""
grand-study:
  display: The Grand Study
  configs:
    - {config: cfg_alpha, org: study-org}
    - {config: cfg_beta, org: study-org}
""")
    monkeypatch.setenv("EVALCARD_REGISTRY_SEED_DIR", str(seed))
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv(
        "BENCHMARK_METADATA_LOCAL_DIR",
        str(Path(__file__).parent / "fixtures" / "auto_benchmarkcards"),
    )

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out = pipeline.run(
        Settings.from_env(),
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(tmp_path / "warehouse"),
        registry_local_dir=str(Path(__file__).parent / "fixtures" / "entity_registry"),
        cache_root=str(tmp_path / "cache"),
    )

    con = duckdb.connect()
    facts = con.execute(
        f"SELECT source_config, composite_slug, composite_display_name "
        f"FROM read_parquet('{out}/fact_results.parquet') "
        f"ORDER BY evaluation_id"
    ).fetchall()
    assert facts == [
        ("cfg_alpha", "grand-study", "The Grand Study"),
        ("cfg_alpha", "cfg-alpha--alpha-board", "Alpha Leaderboard"),
        ("cfg_beta", "grand-study", "The Grand Study"),
        ("cfg_beta", "cfg-beta--beta-board", "Beta Leaderboard"),
    ]
    comps = con.execute(
        f"SELECT composite_slug, source_configs "
        f"FROM read_parquet('{out}/composites.parquet') ORDER BY 1"
    ).fetchall()
    by_slug = {slug: sorted(cfgs) for slug, cfgs in comps}
    assert by_slug["grand-study"] == ["cfg_alpha", "cfg_beta"]
    assert by_slug["cfg-alpha--alpha-board"] == ["cfg_alpha"]
    assert by_slug["cfg-beta--beta-board"] == ["cfg_beta"]

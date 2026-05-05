"""UDF round-trip tests.

For each Python UDF registered against DuckDB, verify that the SQL
invocation matches a direct Python call. Catches:
  - Type-coercion bugs in the UDF wrapper (JSON-typed params arriving as
    strings, missing `_coerce_json`).
  - DuckDB ↔ Python conversion regressions across versions.
  - Struct-return-type schema mismatches.

Each test registers UDFs against a fresh in-memory DuckDB connection
(with a stub resolver) and runs SQL `SELECT udf(args)` against an inline
literal.
"""
from __future__ import annotations

import json

import duckdb
import pytest

from eval_card_backend.canonicalise import udfs
from eval_card_backend.canonicalise.resolver_setup import register_udfs
from eval_card_backend.metric_meta_hotfix import (
    derive_metric_meta,
    reset_provenance_counter,
)
from eval_card_backend.signals.completeness import compute_completeness_py
from eval_card_backend.signals.reproducibility import is_agentic_py
from eval_card_backend.signals.setup import (
    canonical_json,
    fact_id_py,
    reset_json_coerce_counter,
    variant_key_py,
)


class _StubResolver:
    """Minimal Resolver double for UDF registration. Each test patches
    `responses` for the specific (raw, entity_type) pairs it cares about."""

    def __init__(self, responses=None):
        self.responses = responses or {}

    def resolve(self, raw, entity_type, source_config=None):
        from types import SimpleNamespace
        canonical_id, strategy = self.responses.get(
            (raw, entity_type), (None, "no_match")
        )
        return SimpleNamespace(canonical_id=canonical_id, strategy=strategy)


@pytest.fixture
def con():
    """Fresh DuckDB connection with all UDFs registered + counters reset."""
    udfs.reset_resolver_counters()
    reset_json_coerce_counter()
    reset_provenance_counter()
    c = duckdb.connect()
    register_udfs(c, _StubResolver())
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Resolver UDFs
# ---------------------------------------------------------------------------


def test_resolve_canonical_id_round_trip():
    udfs.reset_resolver_counters()
    reset_json_coerce_counter()
    reset_provenance_counter()
    c = duckdb.connect()
    register_udfs(c, _StubResolver({
        ("openai/gpt-4o", "model"): ("openai/gpt-4o", "exact"),
        ("MMLU", "benchmark"): ("mmlu", "normalized"),
    }))

    rows = c.execute(
        "SELECT resolve_canonical_id('openai/gpt-4o', 'model', NULL),"
        "       resolve_canonical_id('MMLU',           'benchmark', NULL),"
        "       resolve_canonical_id('unknown',        'model', NULL),"
        "       resolve_canonical_id(NULL,             'model', NULL)"
    ).fetchone()
    assert rows == ("openai/gpt-4o", "mmlu", None, None)
    c.close()


def test_resolve_strategy_round_trip():
    udfs.reset_resolver_counters()
    c = duckdb.connect()
    register_udfs(c, _StubResolver({
        ("openai/gpt-4o", "model"): ("openai/gpt-4o", "exact"),
    }))

    rows = c.execute(
        "SELECT resolve_strategy('openai/gpt-4o', 'model', NULL),"
        "       resolve_strategy('unknown',       'model', NULL),"
        "       resolve_strategy(NULL,            'model', NULL)"
    ).fetchone()
    assert rows == ("exact", "no_match", "no_match")
    c.close()


# ---------------------------------------------------------------------------
# Identity / setup helpers
# ---------------------------------------------------------------------------


def test_fact_id_udf_round_trip(con):
    sql_value = con.execute(
        "SELECT fact_id_udf('eval_x', 0)"
    ).fetchone()[0]
    py_value = fact_id_py("eval_x", 0)
    assert sql_value == py_value
    assert len(sql_value) == 16


def test_fact_id_udf_null_evaluation_id(con):
    assert con.execute("SELECT fact_id_udf(NULL, 0)").fetchone()[0] is None


def test_variant_key_udf_round_trip(con):
    payload = '{"temperature": 0.7, "max_tokens": 100}'
    sql_value = con.execute(
        f"SELECT variant_key_udf('{payload}'::JSON)"
    ).fetchone()[0]
    py_value = variant_key_py(json.loads(payload))
    assert sql_value == py_value


def test_variant_key_udf_null(con):
    sql_value = con.execute("SELECT variant_key_udf(NULL)").fetchone()[0]
    py_value = variant_key_py(None)
    assert sql_value == py_value


def test_clean_eval_name_udf_smoke(con):
    """The wrapper is from eval_entity_resolver.eee; we don't redefine its
    behaviour here, just verify the wiring (str → str, NULL → NULL)."""
    out = con.execute("SELECT clean_eval_name_udf('Some  Eval')").fetchone()[0]
    assert isinstance(out, str)
    assert con.execute("SELECT clean_eval_name_udf(NULL)").fetchone()[0] is None


def test_extract_metric_udf_smoke(con):
    out = con.execute(
        "SELECT extract_metric_udf('Accuracy on MMLU')"
    ).fetchone()[0]
    assert isinstance(out, (str, type(None)))
    assert con.execute("SELECT extract_metric_udf(NULL)").fetchone()[0] is None


# ---------------------------------------------------------------------------
# Reproducibility / agentic
# ---------------------------------------------------------------------------


def test_is_agentic_udf_round_trip_card_tasks(con):
    card_json = json.dumps({"purpose_and_intended_users": {"tasks": ["agentic"]}})
    sql_value = con.execute(
        f"SELECT is_agentic_udf('mmlu', '{card_json}'::JSON, NULL)"
    ).fetchone()[0]
    py_value = is_agentic_py("mmlu", json.loads(card_json), None)
    assert sql_value == py_value is True


def test_is_agentic_udf_round_trip_config_presence(con):
    ga_json = json.dumps({"agentic_eval_config": {"k": 1}})
    sql_value = con.execute(
        f"SELECT is_agentic_udf('mmlu', NULL, '{ga_json}'::JSON)"
    ).fetchone()[0]
    py_value = is_agentic_py("mmlu", None, json.loads(ga_json))
    assert sql_value == py_value is True


def test_is_agentic_udf_round_trip_name_regex(con):
    sql_value = con.execute(
        "SELECT is_agentic_udf('swe-bench-verified', NULL, NULL)"
    ).fetchone()[0]
    assert sql_value is True


def test_is_agentic_udf_round_trip_negative(con):
    sql_value = con.execute(
        "SELECT is_agentic_udf('mmlu', NULL, NULL)"
    ).fetchone()[0]
    assert sql_value is False


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def test_compute_completeness_udf_round_trip_empty(con):
    sql = con.execute(
        "SELECT compute_completeness_udf(NULL, NULL, NULL, NULL, NULL, NULL)"
    ).fetchone()[0]
    py = compute_completeness_py(None, None, None, None, None, None)
    assert sql["completeness_score"] == py["completeness_score"] == 0.0
    assert sql["total_fields_evaluated"] == py["total_fields_evaluated"] == 28
    assert sql["populated_count"] == py["populated_count"] == 0.0
    # missing_required_fields is the same set
    assert set(sql["missing_required_fields"]) == set(py["missing_required_fields"])


def test_compute_completeness_udf_round_trip_with_source_metadata(con):
    sql = con.execute(
        "SELECT compute_completeness_udf("
        "  NULL, 'evaluation_run', 'OpenAI', 'first_party', NULL, NULL)"
    ).fetchone()[0]
    py = compute_completeness_py(
        None, "evaluation_run", "OpenAI", "first_party", None, None
    )
    assert sql["populated_count"] == py["populated_count"] == 3
    assert abs(sql["completeness_score"] - py["completeness_score"]) < 1e-9


def test_compute_completeness_udf_partial_field(con):
    """Verify partial_fields struct list flows through correctly."""
    card_json = json.dumps({"data": {"source": "s", "size": "z"}})
    sql = con.execute(
        f"SELECT compute_completeness_udf("
        f"  '{card_json}'::JSON, NULL, NULL, NULL, NULL, NULL)"
    ).fetchone()[0]
    partials = sql["partial_fields"]
    assert len(partials) == 1
    assert partials[0]["score"] == 0.5
    assert partials[0]["populated_subitems"] == 2
    assert partials[0]["total_subitems"] == 4


# ---------------------------------------------------------------------------
# canonical_json_udf
# ---------------------------------------------------------------------------


def test_canonical_json_udf_round_trip(con):
    """`canonical_json_udf('{...}'::JSON)` should produce the same
    canonical-JSON string as a direct `canonical_json(parsed_dict)` call."""
    payload = '{"b": 1, "a": 2}'
    sql = con.execute(
        f"SELECT canonical_json_udf('{payload}'::JSON)"
    ).fetchone()[0]
    py = canonical_json(json.loads(payload))
    assert sql == py == '{"a":2,"b":1}'


def test_canonical_json_udf_null(con):
    sql = con.execute("SELECT canonical_json_udf(NULL)").fetchone()[0]
    assert sql is None


# ---------------------------------------------------------------------------
# Org-name normalisation parity (SQL Stage F vs Python cross-party UDF)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    None,
    "",
    "   ",
    "\t\n  \r",
    "OpenAI",
    "openai",
    "  OpenAI  ",
    "Open  AI",
    "Open\tAI",
    "Open\nAI",
    "Anthropic, Inc.",
    "  Anthropic,  Inc.  ",
    "META PLATFORMS",
    "meta	platforms",
])
def test_org_normalize_sql_python_parity(con, raw):
    """SQL `org_normalize_sql(...)` and Python `normalize_org_name` must agree
    on every input. The two run in different code paths — Stage F's distinct-
    org window vs. the per-row cross-party UDF — so a divergence would mis-
    align `distinct_reporting_orgs` against the UDF's view of "different
    orgs"."""
    from eval_card_backend.canonicalise.stages import org_normalize_sql
    from eval_card_backend.signals.comparability import normalize_org_name

    sql = con.execute(
        f"SELECT {org_normalize_sql('?')}",
        [raw],
    ).fetchone()[0]
    py = normalize_org_name(raw)
    assert sql == py, f"raw={raw!r}: sql={sql!r}, py={py!r}"


# ---------------------------------------------------------------------------
# Comparability divergence UDFs
# ---------------------------------------------------------------------------


def _build_group_rows_sql(rows: list[dict]) -> str:
    """Convert a list of row dicts into a DuckDB STRUCT[]-typed VALUES literal."""
    parts = []
    for r in rows:
        ga_json = json.dumps(r.get("generation_args") or {})
        parts.append(
            "{"
            f"'fact_id': '{r['fact_id']}', "
            f"'evaluation_id': '{r['evaluation_id']}', "
            f"'score': {r['score'] if r['score'] is not None else 'NULL'}, "
            f"'generation_args': '{ga_json}', "
            f"'evaluator_relationship': '{r['evaluator_relationship']}', "
            f"'source_organization_name': '{r['source_organization_name']}'"
            "}"
        )
    inner = ", ".join(parts)
    cast = (
        "STRUCT(fact_id VARCHAR, evaluation_id VARCHAR, score DOUBLE, "
        "generation_args VARCHAR, evaluator_relationship VARCHAR, "
        "source_organization_name VARCHAR)[]"
    )
    return f"CAST([{inner}] AS {cast})"


def _metric_config_sql(metric_unit: str | None = None) -> str:
    unit = f"'{metric_unit}'" if metric_unit else "NULL"
    return (
        "{'metric_kind': NULL, 'metric_unit': "
        f"{unit}, 'min_score': 0.0, 'max_score': 1.0"
        "}::STRUCT(metric_kind VARCHAR, metric_unit VARCHAR, min_score DOUBLE, max_score DOUBLE)"
    )


def test_compute_variant_divergence_udf_returns_struct_with_nulls_when_inapplicable(con):
    """Group with <2 rows → underlying Python returns None; wrapper turns
    that into a struct of all-NULL fields so DuckDB destructuring sees NULL,
    not literal False."""
    rows_sql = _build_group_rows_sql([
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {}, "evaluator_relationship": "first_party",
         "source_organization_name": "X"},
    ])
    out = con.execute(
        f"SELECT compute_variant_divergence_udf({rows_sql}, {_metric_config_sql()})"
    ).fetchone()[0]
    assert out["has_variant_divergence"] is None
    assert out["divergence_magnitude"] is None
    assert out["threshold_used"] is None


def test_compute_variant_divergence_udf_positive(con):
    rows_sql = _build_group_rows_sql([
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {"temperature": 0.0, "max_tokens": 100},
         "evaluator_relationship": "first_party", "source_organization_name": "X"},
        {"fact_id": "2", "evaluation_id": "b", "score": 0.7,
         "generation_args": {"temperature": 0.7, "max_tokens": 100},
         "evaluator_relationship": "first_party", "source_organization_name": "X"},
    ])
    out = con.execute(
        f"SELECT compute_variant_divergence_udf({rows_sql}, {_metric_config_sql('proportion')})"
    ).fetchone()[0]
    assert out["has_variant_divergence"] is True
    assert out["threshold_basis"] == "proportion"
    fields = {f["field"] for f in out["differing_setup_fields"]}
    assert fields == {"temperature"}


def test_compute_cross_party_divergence_udf_returns_nulls_when_single_org(con):
    rows_sql = _build_group_rows_sql([
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {}, "evaluator_relationship": "first_party",
         "source_organization_name": "Foo"},
    ])
    out = con.execute(
        f"SELECT compute_cross_party_divergence_udf({rows_sql}, {_metric_config_sql()})"
    ).fetchone()[0]
    assert out["has_cross_party_divergence"] is None
    assert out["organization_count"] is None


def test_compute_cross_party_divergence_udf_positive(con):
    rows_sql = _build_group_rows_sql([
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {}, "evaluator_relationship": "first_party",
         "source_organization_name": "Foo"},
        {"fact_id": "2", "evaluation_id": "b", "score": 0.7,
         "generation_args": {}, "evaluator_relationship": "third_party",
         "source_organization_name": "Bar"},
    ])
    out = con.execute(
        f"SELECT compute_cross_party_divergence_udf({rows_sql}, {_metric_config_sql('proportion')})"
    ).fetchone()[0]
    assert out["has_cross_party_divergence"] is True
    assert out["organization_count"] == 2
    # MAP type: keys may surface as a Python dict or as a list of pairs depending
    # on DuckDB version; normalise to a dict for the assertion.
    sbo = out["scores_by_organization"]
    if isinstance(sbo, list):
        sbo = dict(sbo)
    assert set(sbo.keys()) == {"Foo", "Bar"}


# ---------------------------------------------------------------------------
# Metric-meta hotfix
# ---------------------------------------------------------------------------


def test_derive_metric_meta_udf_round_trip_proportion_heuristic(con):
    """min=0, max=1, score_type=continuous → metric_unit='proportion' via heuristic."""
    sql = con.execute(
        "SELECT derive_metric_meta_udf("
        "  '{}'::JSON,"
        "  NULL, NULL, 0.0, 1.0, FALSE,"
        "  'Accuracy', 'continuous')"
    ).fetchone()[0]
    py = derive_metric_meta(
        {}, None, None, 0.0, 1.0, False, "Accuracy", registry_score_type="continuous"
    )
    assert sql["metric_kind"] == py["metric_kind"] == "accuracy"
    assert sql["metric_unit"] == py["metric_unit"] == "proportion"
    assert sql["lower_is_better"] == py["lower_is_better"] is False
    assert sql["min_score"] == py["min_score"] == 0.0
    assert sql["max_score"] == py["max_score"] == 1.0


def test_derive_metric_meta_udf_registry_wins(con):
    """Registry metric_kind beats EEE record's metric_kind."""
    sql = con.execute(
        "SELECT derive_metric_meta_udf("
        "  '{\"metric_kind\": \"score\"}'::JSON,"
        "  'accuracy', NULL, NULL, NULL, NULL,"
        "  'Accuracy', NULL)"
    ).fetchone()[0]
    assert sql["metric_kind"] == "accuracy"


def test_derive_metric_meta_udf_synonym_normalisation(con):
    """metric_unit='percentage' → 'percent' from synonym map."""
    sql = con.execute(
        "SELECT derive_metric_meta_udf("
        "  '{\"metric_unit\": \"percentage\"}'::JSON,"
        "  NULL, NULL, NULL, NULL, NULL,"
        "  NULL, NULL)"
    ).fetchone()[0]
    assert sql["metric_unit"] == "percent"

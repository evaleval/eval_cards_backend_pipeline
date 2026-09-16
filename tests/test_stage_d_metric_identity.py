"""Stage D — metric identity, direction and who ran the evaluation.

Three rules that decide what a fact row claims:

  * a structured metric id that spells a scoring variant after the metric
    keeps that variant in the observation key, so two readings of one metric
    are two observations rather than one median;
  * the registry's direction wins, except on a catch-all bucket that carries
    no direction claim, where the submitter's own declaration stands;
  * the submitter says who ran the evaluation.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise.resolver_setup import register_udfs
from eval_card_backend.canonicalise import udfs
from eval_card_backend.metric_meta_hotfix import derive_metric_meta


class _Stub:
    """Minimal resolver: only the structured-metric method the Stage C
    pre-step binds to."""

    def __init__(self, table):
        self.table = table

    def resolve_structured_metric(self, raw_id, source_config=None,
                                  catch_all_ids=frozenset()):
        hit = self.table.get(raw_id)
        if hit is None:
            return None
        from types import SimpleNamespace
        canonical, qualifier = hit
        return SimpleNamespace(canonical_id=canonical,
                               matched_segment=canonical, qualifier=qualifier)

    def resolve(self, raw, entity_type=None, source_config=None, check_hf=True):
        from types import SimpleNamespace
        return SimpleNamespace(canonical_id=None, strategy="no_match",
                               confidence=0.0, resolved_leaf_id=None)


# ---------------------------------------------------------------------------
# The qualifier joins the observation key
# ---------------------------------------------------------------------------


def _metric_keys(qualifier):
    """Replay Stage D's key construction for one qualifier value."""
    con = duckdb.connect()
    return con.execute(
        """
        SELECT 'accuracy' || COALESCE('::' || ?, '') AS metric_key,
               'accuracy'                            AS metric_base_key
        """,
        [qualifier],
    ).fetchone()


def test_qualifier_joins_the_key_and_leaves_the_base_alone():
    """`gpqa.accuracy.diamond` and `gpqa.accuracy.extended` become two keys;
    both still look their bounds and display name up under `accuracy`."""
    assert _metric_keys("diamond") == ("accuracy::diamond", "accuracy")
    assert _metric_keys("extended") == ("accuracy::extended", "accuracy")


def test_no_qualifier_leaves_the_key_unchanged():
    """The overwhelming majority of rows carry no tail, and their key must be
    byte-identical to what it was before the qualifier existed."""
    assert _metric_keys(None) == ("accuracy", "accuracy")


def test_qualifier_udf_only_fires_where_the_structured_match_did():
    udfs.reset_resolver_counters()
    con = duckdb.connect()
    register_udfs(
        con,
        _Stub({
            "gpqa.accuracy.diamond": ("accuracy", "diamond"),
            "lmarena.elo.overall": ("elo", None),
        }),
        frozenset({"score"}),
    )
    assert con.execute(
        "SELECT resolve_structured_metric_id('gpqa.accuracy.diamond', NULL),"
        "       resolve_structured_metric_qualifier('gpqa.accuracy.diamond', NULL),"
        "       resolve_structured_metric_qualifier('lmarena.elo.overall', NULL),"
        "       resolve_structured_metric_qualifier('nothing.here', NULL)"
    ).fetchone() == ("accuracy", "diamond", None, None)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


def _catch_all_gate(metadata, registry_direction):
    """Replay Stage D's gate on the registry direction."""
    con = duckdb.connect()
    return con.execute(
        """
        SELECT CASE WHEN COALESCE(
                   CAST(json_extract(?, '$.catch_all') AS BOOLEAN), FALSE)
               THEN NULL ELSE ? END
        """,
        [metadata, registry_direction],
    ).fetchone()[0]


def test_registry_direction_wins_on_an_ordinary_metric():
    assert _catch_all_gate('{"role": "outcome"}', True) is True
    assert derive_metric_meta(
        {"lower_is_better": False}, None, None, 0, 1, True, "Refusal rate"
    )["lower_is_better"] is True


def test_catch_all_direction_falls_through_to_the_row():
    """The generic `score` bucket's higher-is-better is a default, not a claim
    about the measurement — a toxicity rate submitted as lower-is-better must
    not come out reading as if more toxicity were better."""
    gated = _catch_all_gate('{"catch_all": true}', False)
    assert gated is None
    assert derive_metric_meta(
        {"lower_is_better": True}, None, None, 0, 1, gated, "Toxicity"
    )["lower_is_better"] is True


def test_null_registry_direction_falls_through_to_the_row():
    """Same path for a metric whose direction is deliberately unset because it
    depends on the benchmark (refusal is good on harmful prompts, bad on an
    over-refusal suite)."""
    assert derive_metric_meta(
        {"lower_is_better": True}, None, None, 0, 1, None, "Refusal rate"
    )["lower_is_better"] is True


# ---------------------------------------------------------------------------
# Who ran the evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_config,raw_verified,submitted,expected",
    [
        # every source but llm-stats: the submitter's own declaration
        ("swissai_apertus_evals", None, "first_party", "first_party"),
        ("swissai_apertus_evals", None, "third_party", "third_party"),
        ("swissai_apertus_evals", None, "collaborative", "collaborative"),
        ("swissai_apertus_evals", None, "  ", None),
        # llm-stats carries the field from the aggregator's perspective, so
        # the row-level flag decides there and only there
        ("llm-stats", "false", "third_party", "first_party"),
        ("llm-stats", "true", "third_party", "third_party"),
        ("llm-stats", "true", "first_party", "third_party"),
    ],
)
def test_evaluator_relationship(source_config, raw_verified, submitted, expected):
    con = duckdb.connect()
    got = con.execute(
        """
        SELECT CASE
            WHEN ? = 'llm-stats'
            THEN CASE WHEN ? = 'false' THEN 'first_party' ELSE 'third_party' END
            ELSE NULLIF(TRIM(?), '')
        END
        """,
        [source_config, raw_verified, submitted],
    ).fetchone()[0]
    assert got == expected

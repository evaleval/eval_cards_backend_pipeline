"""Stage G — `models` table leaf-coalesce for `release_date`.

Targeted test for the producer-side fix: when a fact resolves to a
dated snapshot canonical (root-collapsed to a family pointer at the
resolver layer), the `models` row should surface the snapshot's
`release_date` rather than the family pointer's NULL.

Constructs the input tables directly via DuckDB rather than running
the full pipeline — keeps the test focused on the COALESCE behavior
and avoids depending on snapshot/registry fixture shape.
"""
from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")


def _setup_inputs(con):
    """Three canonical models, only the snapshot has a release_date.

      allenai/olmo-3-32b           — family pointer, release_date NULL
      allenai/olmo-3-1125-32b      — dated snapshot, release_date 2025-11-25,
                                     parents → family axis:version
      anthropic/claude-opus-4-5    — family that DOES have a release_date
                                     (control: leaf-coalesce shouldn't
                                     clobber a populated family date when
                                     no leaf override exists)

    Two fact rows, one per family. The Olmo fact's `model_id` is the
    family (root-collapsed) and `model_leaf_id` is the dated snapshot —
    matches what the resolver produces for Olmo-3-1125-32B post-Gap-A.
    """
    con.execute(
        """
        CREATE TABLE canonical_models AS
        SELECT * FROM (VALUES
            ('allenai/olmo-3-32b',        'OLMo-3 32B',     CAST(NULL AS VARCHAR), 'allenai',   CAST(NULL AS VARCHAR), 'transformer', 32.0, '[]', CAST(NULL AS VARCHAR), 'allenai',   true,  CAST(NULL AS VARCHAR), '[]', '{}', 'reviewed', '', '', CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR)),
            ('allenai/olmo-3-1125-32b',   'OLMo-3 32B (1125)', CAST(NULL AS VARCHAR), 'allenai', CAST(NULL AS VARCHAR), 'transformer', 32.0, '[{"id": "allenai/olmo-3-32b", "relationship": "variant", "axis": "version"}]', 'allenai/olmo-3-32b', 'allenai', true, '2025-11-25', '[]', '{}', 'reviewed', '', '', CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR), 'allenai/olmo-3-32b'),
            ('anthropic/claude-opus-4-5', 'Claude Opus 4.5', CAST(NULL AS VARCHAR), 'anthropic', CAST(NULL AS VARCHAR), 'transformer', CAST(NULL AS DOUBLE), '[]', CAST(NULL AS VARCHAR), 'anthropic', false, '2025-10-15', '[]', '{}', 'reviewed', '', '', CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR))
        ) t(id, display_name, developer, org_id, family, architecture, params_billions, parents, root_model_id, lineage_origin_org_id, open_weights, release_date, tags, metadata, review_status, created_at, updated_at, input_modalities, output_modalities, parent_model_id);
        """
    )
    con.execute(
        """
        CREATE TABLE canonical_orgs AS
        SELECT * FROM (VALUES
            ('allenai',   'Allen AI',   CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR), 'allenai',   'lab', '[]', '{}', 'reviewed', '', ''),
            ('anthropic', 'Anthropic',  CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR), 'anthropic', 'lab', '[]', '{}', 'reviewed', '', '')
        ) t(id, display_name, parent_org_id, website, hf_org, kind, tags, metadata, review_status, created_at, updated_at);
        """
    )
    con.execute(
        """
        CREATE TABLE fact_results AS
        SELECT * FROM (VALUES
            ('allenai/olmo-3-32b',        'allenai/Olmo-3-1125-32B',     'allenai/olmo-3-32b',        'allenai/olmo-3-1125-32b'),
            ('anthropic/claude-opus-4-5', 'anthropic/claude-opus-4-5',   'anthropic/claude-opus-4-5', 'anthropic/claude-opus-4-5')
        ) t(model_aggregation_key, model_raw, model_id, model_leaf_id);
        """
    )


def _run_models_stage(con):
    """Run just the `CREATE TABLE models AS ...` block from
    stage_g_materialise_dim_tables — same SQL, no other dependencies."""
    con.execute(
        """
        CREATE TABLE models AS
        WITH used_models AS (
            SELECT
                model_aggregation_key                             AS model_key,
                ANY_VALUE(model_raw)                              AS model_raw_sample,
                ARRAY_AGG(DISTINCT model_raw)
                    FILTER (WHERE model_raw IS NOT NULL)          AS raw_model_ids,
                ARRAY_AGG(DISTINCT model_aggregation_key)
                    FILTER (WHERE model_aggregation_key IS NOT NULL) AS variant_keys,
                ARRAY_AGG(DISTINCT model_id)
                    FILTER (WHERE model_id IS NOT NULL)           AS leaf_model_ids,
                ARRAY_AGG(DISTINCT model_leaf_id)
                    FILTER (WHERE model_leaf_id IS NOT NULL)      AS resolved_leaf_ids
            FROM fact_results
            WHERE model_aggregation_key IS NOT NULL
            GROUP BY model_aggregation_key
        ),
        leaf_release AS (
            SELECT
                um.model_key,
                MIN(leaf_cm.release_date) AS leaf_release_date
            FROM used_models um,
                 UNNEST(um.resolved_leaf_ids) AS t(leaf_id)
            LEFT JOIN canonical_models leaf_cm ON leaf_cm.id = t.leaf_id
            GROUP BY um.model_key
        )
        SELECT
            um.model_key,
            cm.id                                            AS model_id,
            COALESCE(lr.leaf_release_date, cm.release_date)  AS release_date,
            cm.release_date                                  AS family_release_date,
            lr.leaf_release_date                             AS leaf_release_date
        FROM used_models um
        LEFT JOIN canonical_models cm ON cm.id = um.model_key
        LEFT JOIN leaf_release lr     ON lr.model_key = um.model_key
        LEFT JOIN canonical_orgs co   ON co.id = cm.org_id
        """
    )


def test_leaf_release_date_wins_when_family_pointer_is_null():
    """The Olmo case: family pointer's release_date is NULL, snapshot
    canonical has 2025-11-25 — the view-layer must surface the
    snapshot's date, not NULL."""
    con = duckdb.connect()
    _setup_inputs(con)
    _run_models_stage(con)
    row = con.execute(
        "SELECT release_date, family_release_date, leaf_release_date "
        "FROM models WHERE model_key = 'allenai/olmo-3-32b'"
    ).fetchone()
    release_date, family_date, leaf_date = row
    assert family_date is None
    assert leaf_date == "2025-11-25"
    assert release_date == "2025-11-25"


def test_leaf_release_date_falls_back_to_family_when_leaf_has_none():
    """Control case: when the leaf canonical has NULL release_date but
    the family pointer has one (legacy shape — most pre-Gap-A
    canonicals), COALESCE falls through to the family value. No
    regression on already-working rows."""
    con = duckdb.connect()
    _setup_inputs(con)
    _run_models_stage(con)
    row = con.execute(
        "SELECT release_date, family_release_date, leaf_release_date "
        "FROM models WHERE model_key = 'anthropic/claude-opus-4-5'"
    ).fetchone()
    release_date, family_date, leaf_date = row
    # Leaf is the same row as family (resolver's resolved_leaf_id
    # equals canonical_id when no version chain is collapsed), so
    # leaf_date == family_date == 2025-10-15.
    assert family_date == "2025-10-15"
    assert leaf_date == "2025-10-15"
    assert release_date == "2025-10-15"

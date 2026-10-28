"""Compatibility exports for the designated source-metadata hotfix."""

from eval_card_backend.canonicalise.resolution_hotfixes import (
    _ARC_AGI_FIX_ID as FIX_ID,
    _ARC_AGI_SOURCE_CONFIG as SOURCE_CONFIG,
    _ARC_AGI_TABLE_NAME as TABLE_NAME,
    _ARC_AGI_SOURCE_REVISION as SOURCE_REVISION,
    _arc_agi_evidence_sha256 as evidence_sha256,
    _ARC_AGI_PATCHED_RESULT_NAMES as PATCHED_RESULT_NAMES,
    _ARC_AGI_SCORE_METRIC_ID as SCORE_METRIC_ID,
    _ARC_AGI_FIELDS as FIELDS,
    _arc_agi_raw_model_id as raw_model_id,
    _ArcAgiLoggedRequestArgs as ArcAgiLoggedRequestArgs,
    build_arc_agi_datafix as build,
)
from eval_card_backend.datafix.core import load_table as load_table

__all__ = ['FIX_ID', 'SOURCE_CONFIG', 'TABLE_NAME', 'SOURCE_REVISION', 'evidence_sha256', 'PATCHED_RESULT_NAMES', 'SCORE_METRIC_ID', 'FIELDS', 'raw_model_id', 'ArcAgiLoggedRequestArgs', 'build', 'load_table']

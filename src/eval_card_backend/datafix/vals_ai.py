"""Compatibility exports for the designated source-metadata hotfix."""

from eval_card_backend.canonicalise.resolution_hotfixes import (
    _VALS_AI_FIX_ID as FIX_ID,
    _VALS_AI_SOURCE_CONFIG as SOURCE_CONFIG,
    _VALS_AI_TABLE_NAME as TABLE_NAME,
    _VALS_AI_FIELD_TYPES as FIELD_TYPES,
    _VALS_AI_FIELDS as FIELDS,
    _VALS_AI_SOURCE_REVISION as SOURCE_REVISION,
    _vals_ai_evidence_sha256 as evidence_sha256,
    _vals_ai_benchmark_slug as benchmark_slug,
    _ValsAiMaxTokens as ValsAiMaxTokens,
    build_vals_ai_datafix as build,
)
from eval_card_backend.datafix.core import load_table as load_table

__all__ = ['FIX_ID', 'SOURCE_CONFIG', 'TABLE_NAME', 'FIELD_TYPES', 'FIELDS', 'SOURCE_REVISION', 'evidence_sha256', 'benchmark_slug', 'ValsAiMaxTokens', 'build', 'load_table']

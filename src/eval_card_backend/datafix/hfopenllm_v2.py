"""Compatibility exports for the designated source-metadata hotfix."""

from eval_card_backend.canonicalise.resolution_hotfixes import (
    _HFOPENLLM_V2_FIX_ID as FIX_ID,
    _HFOPENLLM_V2_SOURCE_CONFIG as SOURCE_CONFIG,
    _HFOPENLLM_V2_TABLE_NAME as TABLE_NAME,
    _HFOPENLLM_V2_SOURCE_REVISION as SOURCE_REVISION,
    _hfopenllm_v2_evidence_sha256 as evidence_sha256,
    _HFOPENLLM_V2_GENERATIVE_BENCHMARKS as GENERATIVE_BENCHMARKS,
    _HFOPENLLM_V2_FIELDS as FIELDS,
    _HfOpenLlmHarnessConfigs as HfOpenLlmHarnessConfigs,
    build_hfopenllm_v2_datafix as build,
)
from eval_card_backend.datafix.core import load_table as load_table

__all__ = ['FIX_ID', 'SOURCE_CONFIG', 'TABLE_NAME', 'SOURCE_REVISION', 'evidence_sha256', 'GENERATIVE_BENCHMARKS', 'FIELDS', 'HfOpenLlmHarnessConfigs', 'build', 'load_table']

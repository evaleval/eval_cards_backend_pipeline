"""Compatibility exports for the designated source-metadata hotfix."""

from eval_card_backend.canonicalise.resolution_hotfixes import (
    _INSPECT_PLAN_CONFIG_FIX_ID as FIX_ID,
    _INSPECT_PLAN_CONFIG_SOURCE_CONFIG as SOURCE_CONFIG,
    _INSPECT_PLAN_CONFIG_LIBRARY as LIBRARY,
    _INSPECT_PLAN_CONFIG_CHECKED_AT_REVISION as CHECKED_AT_REVISION,
    _INSPECT_PLAN_CONFIG_VERSIONS as VERSIONS,
    _INSPECT_PLAN_CONFIG_TERMINAL_SOLVERS as TERMINAL_SOLVERS,
    _INSPECT_PLAN_CONFIG_KNOWN_SOLVERS as KNOWN_SOLVERS,
    _INSPECT_PLAN_CONFIG_OVERRIDE_KEYS as OVERRIDE_KEYS,
    _INSPECT_PLAN_CONFIG_STEP_PARAM_KEYS as STEP_PARAM_KEYS,
    _INSPECT_PLAN_CONFIG_FIELDS as FIELDS,
    _inspect_plan_config_number as _number,
    _inspect_plan_config_temperature as _temperature,
    _inspect_plan_config_max_tokens as _max_tokens,
    _INSPECT_PLAN_CONFIG_PARSERS as PARSERS,
    _INSPECT_PLAN_CONFIG_CONSTANTS as CONSTANTS,
    _INSPECT_PLAN_CONFIG_SOURCE_REVISION as SOURCE_REVISION,
    _inspect_plan_config_evidence_sha256 as evidence_sha256,
    _inspect_plan_config_eval_plan as eval_plan,
    _inspect_plan_config_names_an_override as _names_an_override,
    _inspect_plan_config_model_param as _model_param,
    _inspect_plan_config_library_rejection as library_rejection,
    _inspect_plan_config_plan_rejection as plan_rejection,
    _InspectPlanConfig as InspectPlanConfig,
    build_inspect_plan_config_datafix as build,
)
from eval_card_backend.datafix.core import load_table as load_table

__all__ = ['FIX_ID', 'SOURCE_CONFIG', 'LIBRARY', 'CHECKED_AT_REVISION', 'VERSIONS', 'TERMINAL_SOLVERS', 'KNOWN_SOLVERS', 'OVERRIDE_KEYS', 'STEP_PARAM_KEYS', 'FIELDS', '_number', '_temperature', '_max_tokens', 'PARSERS', 'CONSTANTS', 'SOURCE_REVISION', 'evidence_sha256', 'eval_plan', '_names_an_override', '_model_param', 'library_rejection', 'plan_rejection', 'InspectPlanConfig', 'build', 'load_table']

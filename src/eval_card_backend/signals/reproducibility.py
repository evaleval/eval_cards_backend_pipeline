"""Per-row reproducibility-gap signal.

Active production rule: temperature + max_tokens for non-agentic; agentic adds
eval_plan + eval_limits. The 4-field super-set (temperature, top_p, max_tokens,
prompt_template) is kept as a separate constant so the rule can be widened
without code changes; the fact_results schema stores raw values + has_* flags
for all 4 base fields, so widening is a single CTAS over the existing parquet.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from eval_card_backend.signals.setup import _coerce_json

log = logging.getLogger(__name__)


# Per-row counter incremented when `purpose_and_intended_users` parses cleanly
# but isn't the expected dict shape (e.g. a list, string, scalar). Surfaced
# in the run summary so silent agentic mis-classification is visible.
_purpose_shape_counter: Counter[str] = Counter()


def reset_purpose_shape_counter() -> None:
    _purpose_shape_counter.clear()


def log_purpose_shape_summary() -> None:
    if _purpose_shape_counter:
        log.warning("--- is_agentic: purpose_and_intended_users shape mismatches ---")
        for kind, count in _purpose_shape_counter.most_common():
            log.warning(
                "  %s: %d rows skipped agentic Rule 1 (non-dict purpose)",
                kind, count,
            )


SPEC_BASE_REPRODUCIBILITY_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "max_tokens",
    "prompt_template",
)
BASE_REPRODUCIBILITY_FIELDS: tuple[str, ...] = ("temperature", "max_tokens")
AGENTIC_REPRODUCIBILITY_FIELDS: tuple[str, ...] = ("eval_plan", "eval_limits")

AGENTIC_TASK_TOKENS = frozenset({"agentic", "tool_use", "multi_step_agent"})


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalise_benchmark_slug(s: str | None) -> str:
    if not s:
        return ""
    return _NON_ALNUM_RE.sub("_", s.lower()).strip("_")


_AGENTIC_REGEX_PATH = (
    Path(__file__).resolve().parent.parent / "registry" / "agentic_name_regex.json"
)


def _load_agentic_regex() -> re.Pattern[str]:
    data = json.loads(_AGENTIC_REGEX_PATH.read_text(encoding="utf-8"))
    return re.compile(data["pattern"])


AGENTIC_NAME_REGEX = _load_agentic_regex()


def is_agentic_py(
    benchmark_id: str | None,
    benchmark_card: object,
    generation_args: object,
) -> bool:
    """Three-rule union for agentic classification:

    Rule 1: card.purpose_and_intended_users.tasks ∩ AGENTIC_TASK_TOKENS.
    Rule 2: generation_args.agentic_eval_config is not None.
    Rule 3: AGENTIC_NAME_REGEX matches the normalised benchmark slug.
    """
    card = _coerce_json(benchmark_card, caller="is_agentic_py.card")
    ga = _coerce_json(generation_args, caller="is_agentic_py.gen_args")

    if isinstance(card, dict):
        purpose = card.get("purpose_and_intended_users")
        if purpose is not None and not isinstance(purpose, dict):
            _purpose_shape_counter[type(purpose).__name__] += 1
            purpose = {}
        elif purpose is None:
            purpose = {}
        tasks = purpose.get("tasks")
        if isinstance(tasks, list):
            for task in tasks:
                if (
                    isinstance(task, str)
                    and task.strip().lower() in AGENTIC_TASK_TOKENS
                ):
                    return True

    if isinstance(ga, dict) and ga.get("agentic_eval_config") is not None:
        return True

    slug = _normalise_benchmark_slug(benchmark_id)
    if slug and AGENTIC_NAME_REGEX.search(slug):
        return True

    return False


def required_repro_fields(is_agentic: bool) -> tuple[str, ...]:
    if is_agentic:
        return BASE_REPRODUCIBILITY_FIELDS + AGENTIC_REPRODUCIBILITY_FIELDS
    return BASE_REPRODUCIBILITY_FIELDS

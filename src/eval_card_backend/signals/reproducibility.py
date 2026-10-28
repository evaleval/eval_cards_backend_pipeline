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


# ---------------------------------------------------------------------------
# Non-generative declarations
#
# The base fields describe how text was sampled. A result that sampled no
# text has no value to report for them, and the only trustworthy way to
# know that is a statement by the evaluation's own source. These are the
# sources that make one, and the exact values that mean nothing was
# sampled. Nothing here is inferred from a benchmark name, a task format,
# a metric name or a token count.
#
# This table is the single definition: the Stage E SQL builder and the
# Python helper below both read it, so the two readings cannot drift.
# ---------------------------------------------------------------------------

DECLARATION_SOURCES: tuple[str, ...] = (
    # The lm-evaluation-harness `output_type` the Open LLM v2 leaderboard's
    # published result dumps state per model and benchmark.
    "lm_eval_output_type",
    # The reward-bench leaderboard's own per-model type, carried in each
    # record's `model_info.additional_details.model_type`.
    "rewardbench_model_type",
)

NON_GENERATIVE_VALUES: dict[str, frozenset[str]] = {
    # Candidate continuations are scored by log-likelihood under the model;
    # `generate_until` is the harness's generative mode and is not here.
    "lm_eval_output_type": frozenset({
        "loglikelihood",
        "loglikelihood_rolling",
        "multiple_choice",
    }),
    # Classifier heads and log-probability ratios from a forward pass.
    # `Generative` and `Generative RM` sample text and are not here.
    "rewardbench_model_type": frozenset({
        "Seq. Classifier",
        "Custom Classifier",
        "DPO",
    }),
}


def parse_declaration_sources(raw: str | None) -> frozenset[str]:
    """Parse the comma-separated list of declaration sources to honour.

    Unset or empty honours none, which is the rule exactly as it stands.
    An unrecognised name is a typo rather than a silent no-op, so it
    fails here instead of quietly changing nothing.
    """
    if not raw:
        return frozenset()
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(names) - set(DECLARATION_SOURCES))
    if unknown:
        raise ValueError(
            f"unknown non-generative declaration source(s): "
            f"{', '.join(unknown)}. Known: {', '.join(DECLARATION_SOURCES)}."
        )
    return frozenset(names)


def is_non_generative_py(
    honoured: frozenset[str] | set[str],
    declaration_source: str | None,
    declared_value: str | None,
) -> bool:
    """Whether a result is known to have sampled no text.

    True only when a source declared an output type for this result, that
    source is one the run honours, and the declared value is on that
    source's list. Anything undeclared is False.
    """
    if declaration_source is None or declaration_source not in honoured:
        return False
    return declared_value in NON_GENERATIVE_VALUES[declaration_source]

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


def required_repro_fields(
    is_agentic: bool, base_applicable: bool = True
) -> tuple[str, ...]:
    """The fields a row has to state to be reproducible.

    The base fields drop out when the row sampled no text; the agentic
    fields describe the run's plan and limits, which exist either way, so
    a non-generative agentic row still owes them. A row that owes nothing
    is not complete, it is outside the rule.
    """
    fields = BASE_REPRODUCIBILITY_FIELDS if base_applicable else ()
    if is_agentic:
        fields = fields + AGENTIC_REPRODUCIBILITY_FIELDS
    return fields

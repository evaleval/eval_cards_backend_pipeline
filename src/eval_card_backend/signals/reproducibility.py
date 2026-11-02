"""Reproducibility: the eight setup items.

A result has a reproducibility gap when it leaves any applicable setup item
undisclosed. Each item is one decision somebody re-running the evaluation
has to pin down, and any one of its accepted fields discloses it. Stage J
applies the rule in SQL (`stages.stage_j_reproducibility`), reading the
table below; nothing else defines it.
"""
from __future__ import annotations


# The setup items, in display order: (id, applicability, accepted fields).
# A field is an evidence path (see `stages._repro_namespaces_sql`); the first one
# that is disclosed is the one reported for the item.
#
# Applicability, with what each one asks of the row:
#   always              - every result
#   sampling_controlled - every result not scored from log-probabilities,
#   generative_run        where nothing is sampled or generated
#   reasoning_run       - results with an affirmative reasoning marker;
#                         without one the item is unknown, not absent
#   agentic_run         - results with an affirmative agentic marker; likewise
# An item that does not apply, or whose applicability is unknown and that
# nothing discloses, leaves the denominator. A disclosed value settles
# applicability on its own: `compaction: false` answers context handling.
SETUP_ITEMS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("decoding", "sampling_controlled", (
        "generation_args.temperature",
        "generation_args.top_p",
        "generation_args.top_k",
        "protocol.temperature",
        "protocol.top_p",
        "generation_config.additional_details.temperature_raw",
        "generation_config.additional_details.temperature",
    )),
    ("reasoning_budget", "reasoning_run", (
        "protocol.reasoning_effort",
        "protocol.reasoning_tokens",
        "generation_args.reasoning",
        "generation_config.additional_details.effort",
        "generation_config.additional_details.reasoning_effort",
    )),
    ("output_length", "generative_run", (
        "generation_args.max_tokens",
        "protocol.token_limit",
        "limits.token_limit",
        "limits.max_tokens",
        "limits.message_limit",
        "limits.max_messages",
    )),
    # Harness is two items: which one, and which version or task config of
    # it. The library name alone is on almost every row and cannot be re-run.
    ("harness", "always", (
        "protocol.scaffold",
        "row.eval_library",
        "agent.eval_plan",
        "agent.agentic_eval_config",
        "generation_config.additional_details.lm_eval_task",
        "generation_config.additional_details.harness",
        "generation_config.additional_details.framework",
    )),
    ("harness_version", "always", (
        "row.eval_library_version",
        "generation_config.additional_details.eval_library_version",
        "generation_config.additional_details.harness_version",
        "generation_config.additional_details.lm_eval_task",
        "protocol.harness_version",
    )),
    ("context_handling", "agentic_run", (
        "protocol.compaction",
        "protocol.context_management",
        "limits.compaction",
        "limits.truncation",
    )),
    ("test_set", "always", (
        "score.sample_size",
        "protocol.n_tasks",
        "protocol.split",
        "row.split",
        "generation_config.additional_details.eval_split",
        "generation_config.additional_details.subset",
        "generation_config.additional_details.n_samples",
    )),
    ("attempts", "agentic_run", (
        "protocol.attempts",
        "protocol.epochs",
        "protocol.n_trajectories",
        "protocol.num_runs",
        "protocol.samples",
        "limits.attempts",
        "limits.epochs",
        "generation_config.additional_details.repeats",
        "generation_config.additional_details.max_attempts",
        "generation_config.additional_details.num_runs",
    )),
)

REASONING_MARKERS: tuple[str, ...] = (
    "protocol.reasoning_effort",
    "protocol.reasoning_tokens",
    "generation_args.reasoning",
)
AGENTIC_MARKERS: tuple[str, ...] = (
    "protocol.scaffold",
    "protocol.compaction",
    "agent.eval_plan",
    "agent.agentic_eval_config",
    "limits.message_limit",
    "limits.max_messages",
)

# Scoring mode is read from the view's `scoring_mode` first; a row it leaves
# NULL falls back to an output_type the row itself states.
OUTPUT_TYPE_PATHS: tuple[str, ...] = (
    "generation_config.additional_details.output_type",
    "protocol.output_type",
)
LOG_PROB_OUTPUT_TYPES: tuple[str, ...] = ("multiple_choice", "loglikelihood", "loglikelihood_rolling")
GENERATIVE_OUTPUT_TYPES: tuple[str, ...] = ("generate_until", "generate", "generation")

# Strings that state nothing. Compared trimmed and lower-cased.
PLACEHOLDER_VALUES: tuple[str, ...] = (
    "", "unknown", "n/a", "na", "null", "nil", "default", "not applicable",
    "not reported", "not stated", "not specified", "unspecified",
    "source does not state", "tbd", "?", "-",
)
# Disclosures of an absence. They answer an item but cannot be the marker
# that a run reasoned or was agentic.
NEGATIVE_VALUES: tuple[str, ...] = ("false", "no", "none", "off", "disabled")

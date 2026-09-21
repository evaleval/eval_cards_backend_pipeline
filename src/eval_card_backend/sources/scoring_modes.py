"""Loader for `scoring_modes.yaml` — how a result was scored.

Generated text or log-probabilities. The distinction decides which setup
fields can exist for a run at all: a log-prob result scores the model's
likelihood over fixed answer choices, so temperature and max_tokens are not
undisclosed, they are ABSENT. Consumers that score reproducibility need it or
they mark a quarter of the corpus down for knobs it never had.

The file is a fallback, keyed by (composite_slug, benchmark_id), for sources
whose records do not carry `output_type` themselves. It is derived from the
harness's own dumps by `scripts/derive_scoring_modes.py`, never from a
benchmark's name.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from eval_card_backend.config import REPO_ROOT

log = logging.getLogger(__name__)

DEFAULT_SCORING_MODES_PATH = REPO_ROOT / "scoring_modes.yaml"

VALID_MODES = frozenset({"log_prob", "generative"})

#: Values a harness may report for `output_type`, mapped to our two modes.
OUTPUT_TYPE_MODES: dict[str, str] = {
    "multiple_choice": "log_prob",
    "loglikelihood": "log_prob",
    "loglikelihood_rolling": "log_prob",
    "generate_until": "generative",
    "generate": "generative",
    "generation": "generative",
}


def _not_a_mapping(target: Path, what: str, value: object) -> list[tuple[str, str, str]]:
    """Warn once and give up on the whole file.

    A wrong shape is not a wrong cell: it says the file is not the table we
    think it is, so nothing in it can be trusted. A typo'd mode below costs
    one entry; this costs the mapping.
    """
    log.warning(
        "scoring modes: %s has %s as %s, expected a mapping; mapping skipped",
        target, what, type(value).__name__,
    )
    return []


def load_scoring_modes(path: Path | None = None) -> list[tuple[str, str, str]]:
    """Return (composite_slug, benchmark_id, mode) rows.

    A malformed or missing file yields an empty mapping and a warning rather
    than an exception: an unknown scoring mode degrades to "cannot say", which
    every consumer already handles, whereas failing the bake over a fallback
    table would take the whole warehouse down with it.
    """
    target = path or DEFAULT_SCORING_MODES_PATH
    if not target.exists():
        log.warning("scoring modes: %s not found; every row's mode stays NULL", target)
        return []

    try:
        payload = yaml.safe_load(target.read_text())
    except yaml.YAMLError as exc:
        log.warning("scoring modes: %s is not valid YAML (%s); mapping skipped", target, exc)
        return []

    if not isinstance(payload, dict):
        return _not_a_mapping(target, "its root", payload)
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        return _not_a_mapping(target, "`sources`", sources)

    rows: list[tuple[str, str, str]] = []
    for composite_slug, source in sources.items():
        if not isinstance(source, dict):
            return _not_a_mapping(target, f"source {composite_slug!r}", source)
        benchmarks = source.get("benchmarks")
        if not isinstance(benchmarks, dict):
            return _not_a_mapping(target, f"{composite_slug!r}'s `benchmarks`", benchmarks)
        for benchmark_id, entry in benchmarks.items():
            if not isinstance(entry, dict):
                return _not_a_mapping(
                    target, f"entry {composite_slug!r}/{benchmark_id!r}", entry
                )
            mode = entry.get("mode")
            if not isinstance(mode, str) or mode not in VALID_MODES:
                # Naming the bad entry matters more than guessing past it: a
                # typo here silently withholds the fact for a whole source.
                log.warning(
                    "scoring modes: %s/%s has mode %r, expected one of %s; entry ignored",
                    composite_slug, benchmark_id, mode, sorted(VALID_MODES),
                )
                continue
            rows.append((str(composite_slug), str(benchmark_id), mode))

    log.info("scoring modes: %d (source, benchmark) entries loaded from %s", len(rows), target.name)
    return rows

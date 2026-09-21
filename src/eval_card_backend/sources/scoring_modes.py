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

log = logging.getLogger(__name__)

# Package data, like the other checked-in lookup tables under `registry/`:
# resolved next to the module rather than from the repo root so an installed
# wheel carries it and cannot silently classify every row as unknown.
DEFAULT_SCORING_MODES_PATH = (
    Path(__file__).resolve().parent.parent / "registry" / "scoring_modes.yaml"
)

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


class _NoDuplicateKeys(yaml.SafeLoader):
    """SafeLoader that refuses a repeated mapping key.

    PyYAML takes the last one. Two `bbh:` entries under one source would
    therefore change a benchmark's mode with nothing to show for it, which is
    the failure this whole file exists to prevent.
    """


def _construct_unique_mapping(loader, node, deep=False):
    # Membership by equality rather than a set: a key can be unhashable here,
    # and the shape check rejects those later anyway.
    seen: list = []
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.append(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_NoDuplicateKeys.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _malformed(target: Path, complaint: str) -> list[tuple[str, str, str]]:
    """Warn once and give up on the whole file.

    A wrong shape is not a wrong cell: it says the file is not the table we
    think it is, so nothing in it can be trusted. A typo'd mode below costs
    one entry; this costs the mapping.
    """
    log.warning("scoring modes: %s %s; mapping skipped", target, complaint)
    return []


def _not_a_mapping(target: Path, what: str, value: object) -> list[tuple[str, str, str]]:
    return _malformed(
        target, f"has {what} as {type(value).__name__}, expected a mapping"
    )


def load_scoring_modes(path: Path | None = None) -> list[tuple[str, str, str]]:
    """Return (composite_slug, benchmark_id, mode) rows.

    A malformed, unreadable or missing file yields an empty mapping and a
    warning rather than an exception: an unknown scoring mode degrades to
    "cannot say", which every consumer already handles, whereas failing the
    bake over a fallback table would take the whole warehouse down with it.

    A file with `sources: {}` is not malformed. It is the mapping switched
    off on purpose, so it loads to nothing without complaint.
    """
    target = path or DEFAULT_SCORING_MODES_PATH
    if not target.exists():
        log.warning("scoring modes: %s not found; every row's mode stays NULL", target)
        return []

    try:
        payload = yaml.load(target.read_text(), Loader=_NoDuplicateKeys)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        # Not just a parse: the path can be a directory, and bad bytes fail
        # at the decode. Both used to end the bake from inside the stage.
        log.warning(
            "scoring modes: %s could not be read as the mapping (%s: %s); mapping skipped",
            target, type(exc).__name__, exc,
        )
        return []

    if not isinstance(payload, dict):
        return _not_a_mapping(target, "its root", payload)
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        return _not_a_mapping(target, "`sources`", sources)

    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for composite_slug, source in sources.items():
        # Keys are read, never coerced. YAML 1.1 turns a bare `yes` or `on`
        # into a boolean, and str() would file it under "True" next to any
        # other such key; a quoted "1" beside a bare 1 collapses the same way.
        # Two rows under one key make the fallback lookup ambiguous.
        if not isinstance(composite_slug, str):
            return _malformed(
                target,
                f"names a source with the {type(composite_slug).__name__} key "
                f"{composite_slug!r}, expected a string",
            )
        if not isinstance(source, dict):
            return _not_a_mapping(target, f"source {composite_slug!r}", source)
        benchmarks = source.get("benchmarks")
        if not isinstance(benchmarks, dict):
            return _not_a_mapping(target, f"{composite_slug!r}'s `benchmarks`", benchmarks)
        for benchmark_id, entry in benchmarks.items():
            if not isinstance(benchmark_id, str):
                return _malformed(
                    target,
                    f"names a benchmark under {composite_slug!r} with the "
                    f"{type(benchmark_id).__name__} key {benchmark_id!r}, "
                    f"expected a string",
                )
            if (composite_slug, benchmark_id) in seen:
                return _malformed(
                    target, f"gives {composite_slug}/{benchmark_id} more than once"
                )
            seen.add((composite_slug, benchmark_id))
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
            rows.append((composite_slug, benchmark_id, mode))

    log.info("scoring modes: %d (source, benchmark) entries loaded from %s", len(rows), target.name)
    return rows

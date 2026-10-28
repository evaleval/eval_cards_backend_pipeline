"""EEE datafix overlay: corrections to known upstream gaps in raw records.

The overlay sits at the Stage A record boundary (`sources/eee.py`): every
record is validated as it always was, then the fixes registered for its
config run over it, and a patched copy is validated again before it casts
into `eee_raw`. Interpretive signals are therefore computed from corrected
data without any stage below Stage A knowing the overlay exists.

Each fix is a vendored patch table plus a pure applier, so the same
correction can be replayed outside the pipeline (`datafix export`) and
eventually be pushed upstream. See `core` for the doctrine every applier
follows.

What a run's overlay did is accumulated in a `DatafixSummary` belonging to
that run, which the caller creates and hands to the loader.
"""
from __future__ import annotations

from eval_card_backend.datafix.core import (
    OVERLAY_REVISION,
    DatafixSummary,
    PatchEvent,
    apply_fixes,
    fingerprint,
)
def annotation_log():
    from eval_card_backend.datafix.annotations import annotations as impl
    return impl()


def annotating_configs():
    from eval_card_backend.datafix.annotations import annotating_configs as impl
    return impl()


def reset_annotations():
    from eval_card_backend.datafix.annotations import reset_annotations as impl
    return impl()


def all_fixes():
    from eval_card_backend.datafix.registry import all_fixes as impl
    return impl()


def covers(source_config):
    from eval_card_backend.datafix.registry import covers as impl
    return impl(source_config)


def covers_every_config():
    from eval_card_backend.datafix.registry import covers_every_config as impl
    return impl()


def fixes_for(source_config):
    from eval_card_backend.datafix.registry import fixes_for as impl
    return impl(source_config)


def named_configs():
    from eval_card_backend.datafix.registry import named_configs as impl
    return impl()


__all__ = [
    "annotation_log",
    "annotating_configs",
    "reset_annotations",
    "OVERLAY_REVISION",
    "DatafixSummary",
    "PatchEvent",
    "all_fixes",
    "apply_fixes",
    "covers",
    "covers_every_config",
    "fingerprint",
    "fixes_for",
    "named_configs",
]

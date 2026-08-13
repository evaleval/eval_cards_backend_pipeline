"""Non-finite numeric handling in the Stage A Arrow cast.

Upstream converters sometimes serialize IEEE non-finite values as strings
("Infinity", "-Infinity", "NaN") because JSON has no literal for them —
first seen in llm-stats `metric_config.max_score`, where 4 such records
failed the pyarrow cast for the whole 622-record config. `_pad` normalizes
them to NULL for float-typed schema fields only.
"""
from __future__ import annotations

import math

import pyarrow as pa

from eval_card_backend.schemas.eee_arrow import _pad


def test_string_infinity_in_float_field_becomes_null():
    assert _pad("Infinity", pa.float64()) is None
    assert _pad("-Infinity", pa.float64()) is None
    assert _pad("NaN", pa.float64()) is None


def test_float_nonfinite_becomes_null():
    assert _pad(float("inf"), pa.float64()) is None
    assert _pad(float("-inf"), pa.float64()) is None
    assert _pad(float("nan"), pa.float64()) is None


def test_finite_floats_pass_through():
    assert _pad(0.97, pa.float64()) == 0.97
    assert _pad(0.0, pa.float64()) == 0.0


def test_string_field_named_infinity_untouched():
    # dtype guard: only FLOAT fields normalize; a genuine string value that
    # happens to read "Infinity" (e.g. a display name) must survive.
    assert _pad("Infinity", pa.string()) == "Infinity"


def test_struct_with_nonfinite_bound_normalizes_in_place():
    dtype = pa.struct([
        pa.field("max_score", pa.float64()),
        pa.field("unit", pa.string()),
    ])
    out = _pad({"max_score": "Infinity", "unit": "proportion"}, dtype)
    assert out == {"max_score": None, "unit": "proportion"}
    assert not any(
        isinstance(v, float) and not math.isfinite(v) for v in out.values() if v
    )

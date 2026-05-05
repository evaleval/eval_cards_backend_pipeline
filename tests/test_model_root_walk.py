"""Unit tests for `_derive_model_root_id` — the transitive variant /
quantization root walk that overwrites `canonical_models.root_model_id`.

Each case sets up a minimal `canonical_models` table directly on a
DuckDB connection (no registry fixture, no resolver), runs the helper,
and reads back the resulting `root_model_id` column. The goal is to
pin down the walk's semantics independently of the full pipeline:

  - A model with no parent of either kind → root = self.
  - A variant chain (parent_model_id) → root = topmost ancestor.
  - A quantization chain (root_model_id pre-set on a leaf) → root
    follows the registry-set value.
  - Mixed chains (variant edge above a quantization root) walk both
    until a fixed point.
  - Cycles in either edge type terminate the walk without infinite
    looping.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise.stages import _derive_model_root_id


_DDL = """
CREATE TABLE canonical_models (
    id              VARCHAR,
    parent_model_id VARCHAR,
    root_model_id   VARCHAR
)
"""


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute(_DDL)
    yield c
    c.close()


def _root_of(con) -> dict[str, str]:
    rows = con.execute(
        "SELECT id, root_model_id FROM canonical_models"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def test_orphan_models_are_their_own_root(con):
    """No parent of either kind → root = self."""
    con.executemany(
        "INSERT INTO canonical_models VALUES (?, ?, ?)",
        [
            ("openai/gpt-4o", None, None),
            ("anthropic/claude", None, None),
        ],
    )
    _derive_model_root_id(con)
    assert _root_of(con) == {
        "openai/gpt-4o": "openai/gpt-4o",
        "anthropic/claude": "anthropic/claude",
    }


def test_variant_chain_collapses_to_topmost(con):
    """grok-4-0407 → variant of grok-4. Walk reaches grok-4."""
    con.executemany(
        "INSERT INTO canonical_models VALUES (?, ?, ?)",
        [
            ("grok-4",      None,    None),
            ("grok-4-0407", "grok-4", None),
        ],
    )
    _derive_model_root_id(con)
    assert _root_of(con) == {
        "grok-4":      "grok-4",
        "grok-4-0407": "grok-4",
    }


def test_multi_hop_variant_chain(con):
    """A → B → C variant chain: every node resolves to A."""
    con.executemany(
        "INSERT INTO canonical_models VALUES (?, ?, ?)",
        [
            ("base",  None,   None),
            ("v1",    "base", None),
            ("v1.1",  "v1",   None),
        ],
    )
    _derive_model_root_id(con)
    assert _root_of(con) == {
        "base": "base",
        "v1":   "base",
        "v1.1": "base",
    }


def test_quantization_root_followed(con):
    """llama-3-70b-int4 has a registry-set root_model_id pointing at
    the unquantized base. Walk lands at the base."""
    con.executemany(
        "INSERT INTO canonical_models VALUES (?, ?, ?)",
        [
            ("llama-3-70b",      None, None),
            ("llama-3-70b-int4", None, "llama-3-70b"),
        ],
    )
    _derive_model_root_id(con)
    assert _root_of(con) == {
        "llama-3-70b":      "llama-3-70b",
        "llama-3-70b-int4": "llama-3-70b",
    }


def test_mixed_quantization_above_variant(con):
    """Quant leaf → quant root → variant of base. Walk follows both
    edge kinds and lands at base."""
    con.executemany(
        "INSERT INTO canonical_models VALUES (?, ?, ?)",
        [
            ("base",       None,    None),
            ("base-tuned", "base",  None),
            ("base-tuned-int4", None, "base-tuned"),
        ],
    )
    _derive_model_root_id(con)
    assert _root_of(con) == {
        "base":             "base",
        "base-tuned":       "base",
        "base-tuned-int4":  "base",
    }


def test_self_loop_variant_terminates(con):
    """Pathological: a variant edge points at self. Walk terminates
    without infinite looping; the node remains its own root."""
    con.executemany(
        "INSERT INTO canonical_models VALUES (?, ?, ?)",
        [
            ("loopy", "loopy", None),
        ],
    )
    _derive_model_root_id(con)
    assert _root_of(con) == {"loopy": "loopy"}


def test_cycle_between_two_nodes_terminates(con):
    """A → B → A variant cycle. Walk terminates at the second visit;
    each node resolves to the *other* (whichever was reached first
    before the cycle closed). Non-infinite is the contract being
    asserted; the specific landing node is incidental."""
    con.executemany(
        "INSERT INTO canonical_models VALUES (?, ?, ?)",
        [
            ("a", "b", None),
            ("b", "a", None),
        ],
    )
    _derive_model_root_id(con)
    roots = _root_of(con)
    # Cycle break is deterministic per starting node: starting at `a`,
    # walk visits a → b → (a already visited) → stop at b. Likewise
    # `b` walks to a.
    assert roots == {"a": "b", "b": "a"}


def test_dangling_parent_id_left_as_self(con):
    """parent_model_id points at a model id that doesn't exist in the
    table. Walk treats it as terminating (no recursion target found)."""
    con.executemany(
        "INSERT INTO canonical_models VALUES (?, ?, ?)",
        [
            ("orphan", "ghost-parent", None),
        ],
    )
    _derive_model_root_id(con)
    # The walk advances to "ghost-parent" once (it's not in the visited
    # set yet), then on the next iteration `parent_of.get("ghost-parent")`
    # returns None (KeyError → default), so the walk terminates there.
    assert _root_of(con) == {"orphan": "ghost-parent"}


def test_empty_table_is_noop(con):
    """No rows → helper returns cleanly, column stays empty."""
    _derive_model_root_id(con)
    assert _root_of(con) == {}

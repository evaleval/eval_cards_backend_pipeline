"""A flat benchmark name is a verified whole when a byte-exact alias names
the benchmark the row resolved to.

The structured path reads dotted names only. Most of HELM is flat (`MMLU All
Subjects`, `Anatomy`, `AIRBench 2024`) and resolves through the plain alias
index, whose `exact` answer is a registry statement about that spelling AS
WRITTEN. Before this rule every such row was `unknown`, so a source's own
total pooled with whatever sat beside it and the cell was labelled a pooling
of unread rows. `normalized` and fuzzy hits say two spellings probably mean
one benchmark and nothing about what was measured, so they stay `unknown`.

The alias set is the fixture registry's plus the real HELM / Global-MMLU-Lite
spellings, written to a temporary alias table, so the tests exercise the
resolver's own strategy chain rather than a stub.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"

EXTRA_ALIASES = [
    # (raw_value, canonical_id, source_config)
    ("MMLU All Subjects", "mmlu", None),
    ("Anatomy", "mmlu-anatomy", "helm_mmlu"),
    ("Astronomy", "mmlu-astronomy", "helm_mmlu"),
    ("AIRBench 2024", "air-bench-2024", None),
    ("Global MMLU Lite", "global-mmlu-lite", None),
    ("Arabic", "global-mmlu-lite-arabic", "global_mmlu_lite"),
]


@pytest.fixture(scope="module")
def role(tmp_path_factory):
    """`resolve_benchmark_observation_role_py` bound to a resolver over the
    fixture aliases plus `EXTRA_ALIASES`."""
    from eval_card_backend.canonicalise import udfs
    from eval_card_backend.sources import registry as registry_src
    from eval_entity_resolver import Resolver

    root = tmp_path_factory.mktemp("aliases")
    shutil.copy(FIXTURES / "entity_registry" / "aliases.parquet",
                root / "aliases.parquet")
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE a AS SELECT * FROM read_parquet('{root / 'aliases.parquet'}')"
    )
    for i, (raw, canonical, cfg) in enumerate(EXTRA_ALIASES):
        con.execute(
            "INSERT INTO a BY NAME SELECT ? AS id, ? AS raw_value, "
            "'benchmark' AS entity_type, ? AS canonical_id, ? AS source_config, "
            "'confirmed' AS status, 'seed' AS strategy, 1.0 AS confidence",
            [f"flat-exact-{i}", raw, canonical, cfg],
        )
    con.execute(f"COPY a TO '{root / 'aliases.parquet'}' (FORMAT PARQUET)")
    con.close()
    resolver = Resolver(registry_src.load_alias_store(root))
    fns = udfs.make_resolver_udfs(resolver)
    # position 11 in the returned tuple, by name to stay robust
    by_name = {f.__name__: f for f in fns}
    return by_name["resolve_benchmark_observation_role_py"]


def test_flat_exact_alias_on_the_canonical_is_a_whole(role):
    """HELM's `MMLU All Subjects` is the registry's own spelling of `mmlu`,
    so the row measured MMLU."""
    assert role("MMLU All Subjects", "helm_mmlu", "mmlu") == "whole"


def test_flat_exact_alias_on_a_slice_child_is_the_whole_of_the_child(role):
    """`Anatomy`, scoped to HELM, hits the slice child `mmlu-anatomy` and
    resolves INTO that child's cell, where it is the child's own whole
    reading (the same answer a structured `bfcl_v3.simple` gets). Relative
    to `mmlu` it is a part, and it never reaches the parent's cell: asked
    whether it is a whole of `mmlu`, the answer is no."""
    assert role("Anatomy", "helm_mmlu", "mmlu-anatomy") == "whole"
    assert role("Anatomy", "helm_mmlu", "mmlu") == "unknown"
    assert role("Arabic", "global_mmlu_lite", "global-mmlu-lite-arabic") == "whole"


def test_other_real_source_totals(role):
    assert role("AIRBench 2024", "helm_air_bench", "air-bench-2024") == "whole"
    assert role("Global MMLU Lite", "global_mmlu_lite", "global-mmlu-lite") == "whole"


def test_normalized_flat_alias_stays_unknown(role):
    """Case and spacing differences resolve through the normalized index,
    which is a guess about spelling, not a statement about what was
    measured."""
    assert role("mmlu all subjects", "helm_mmlu", "mmlu") == "unknown"
    assert role("MMLU  All Subjects", "helm_mmlu", "mmlu") == "unknown"


def test_scoped_alias_outside_its_source_is_not_a_whole(role):
    """The HELM scoping is part of the statement: `Anatomy` from another
    source is nobody's exact alias."""
    assert role("Anatomy", "some_other_source", "mmlu-anatomy") == "unknown"


def test_exact_hit_must_name_the_resolved_benchmark(role):
    """A hotfix or slice promotion can move a row to another canonical after
    resolution; the alias statement was about the old one, so the row is not
    a verified whole of the new one."""
    assert role("MMLU All Subjects", "helm_mmlu", "mmlu-pro") == "unknown"
    assert role("MMLU All Subjects", "helm_mmlu", None) == "unknown"


def test_structured_names_are_unaffected(role):
    """The dotted path still answers first: a real subset is a part, a bare
    dotted total a whole, whatever the plain index would say."""
    assert role("omni-math.category.alpha", "fixtures_tasks", "omni-math") == "part"
    assert role("omni-math.overall", "fixtures_tasks", "omni-math") == "whole"

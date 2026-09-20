"""Instance sample-file pointers: resolution to one repo-relative shape,
and the fetchable URL Stage J builds from it.

Upstream `detailed_evaluation_results.file_path` is written three ways
(bare filename, `./`-relative, already rooted). Only the rooted form can
be fetched as-is; the other two are meaningless without the directory of
the record that declared them. Stage A resolves all three against
`source_record_path`, so consumers get one addressable shape.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise import stages

from tests.test_stage_j_eval_results_view import (
    _materialise_view,
    _run_through_stage_i,
)


RECORD = "data/mbpp/qwen/qwen-2.5-coder-32b/81426a71.json"
RESOLVED = "data/mbpp/qwen/qwen-2.5-coder-32b/81426a71_samples.jsonl"


def _resolve(record_path, file_path):
    """Evaluate the Stage A resolution SQL on one (record, file) pair."""
    con = duckdb.connect()
    sql = stages.instance_file_path_sql("fp", "srp")
    con.execute("CREATE TABLE t(srp VARCHAR, fp VARCHAR)")
    con.execute("INSERT INTO t VALUES (?, ?)", [record_path, file_path])
    return con.execute(f"SELECT {sql} FROM t").fetchone()[0]


# ---------------------------------------------------------------------------
# Stage A — pointer resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, file_path",
    [
        ("bare filename", "81426a71_samples.jsonl"),
        ("dot-relative", "./81426a71_samples.jsonl"),
    ],
)
def test_relative_pointers_resolve_against_record_dir(shape, file_path):
    """The two relative shapes land beside the record that declared them."""
    assert _resolve(RECORD, file_path) == RESOLVED, shape


def test_rooted_pointer_passes_through_unchanged():
    """An already repo-relative pointer is addressable — don't touch it."""
    rooted = "data/hle/anthropic/claude-opus-4/d52f5c7d_samples.jsonl"
    assert _resolve("data/hle/anthropic/claude-opus-4/d52f5c7d.json", rooted) == rooted


def test_rooted_pointer_is_not_reresolved_against_a_different_record():
    """Rooting wins over the record dir: a rooted pointer is never rewritten
    to sit beside its record, even when the two disagree."""
    rooted = "data/other/org/model/abc_samples.jsonl"
    assert _resolve(RECORD, rooted) == rooted


@pytest.mark.parametrize(
    "reason, record_path, file_path",
    [
        ("no pointer", RECORD, None),
        ("no record path", None, "abc_samples.jsonl"),
        ("record path has no directory", "toplevel.json", "abc_samples.jsonl"),
        ("dot-relative, record path has no directory", "toplevel.json", "./abc.jsonl"),
    ],
)
def test_unresolvable_pointers_yield_null(reason, record_path, file_path):
    """A half-resolved path 404s as silently as a raw one — emit nothing."""
    assert _resolve(record_path, file_path) is None, reason


def test_dot_prefix_is_stripped_not_merely_trimmed():
    """`./` is removed, and a leading dot in the filename itself survives."""
    assert _resolve(RECORD, "./.hidden_samples.jsonl").endswith("/.hidden_samples.jsonl")


# ---------------------------------------------------------------------------
# Stage J — the fetchable URL
# ---------------------------------------------------------------------------


def _with_instance_pointer(path: str):
    """Dim mutation: give every fact row the same instance pointer, so the
    view has something to build a URL from without new fixture data."""

    def mutate(con):
        con.execute("UPDATE fact_results SET instance_file_path = ?", [path])

    return mutate


def test_instance_file_url_is_built_from_the_resolved_path(tmp_path, monkeypatch):
    """The URL addresses the EEE repo at the same revision `eee_record_url`
    uses, so a snapshot's record and sample links never disagree."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out, mutate=_with_instance_pointer(RESOLVED))

    urls = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT instance_file_url FROM eval_results_view "
            "WHERE instance_file_url IS NOT NULL"
        ).fetchall()
    ]
    assert urls, "no instance_file_url emitted"
    for url in urls:
        assert url == (
            "https://huggingface.co/datasets/evaleval/EEE_datastore/resolve/main/"
            + RESOLVED
        )


def test_instance_file_url_is_null_without_a_pointer(tmp_path, monkeypatch):
    """No pointer means no link — never a URL to the repo root."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out, mutate=_with_instance_pointer(None))

    assert con.execute(
        "SELECT count(*) FROM eval_results_view WHERE instance_file_url IS NOT NULL"
    ).fetchone()[0] == 0


def test_sample_urls_carry_links_not_paths(tmp_path, monkeypatch):
    """`instance_data.sample_urls` is consumed as a link list; it must hold
    URLs, not the repo-relative paths they are built from."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_view(out, mutate=_with_instance_pointer(RESOLVED))
    stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")

    rows = con.execute(
        "SELECT instance_data FROM evals_view "
        "WHERE instance_data.available"
    ).fetchall()
    assert rows, "fixture produced no eval with instance data"
    for (data,) in rows:
        assert data["url_count"] > 0
        assert data["sample_urls"], "available instance data with no sample_urls"
        for url in data["sample_urls"]:
            assert url.startswith("https://huggingface.co/datasets/")

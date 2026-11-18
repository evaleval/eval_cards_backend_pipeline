"""Float means and sums that reach a published file do not depend on the
order DuckDB meets the rows in.

Double addition is not associative. `AVG(x)` and `SUM(x)` over doubles add
in storage and thread order, so two builds of the same rows can publish
neighbouring doubles, and rounding to 12 places only hides that until a
mean sits next to a rounding boundary. `stages.order_free_avg_sql` and
`stages.order_free_sum_sql` sort the values first.

The last test keeps a bare double `AVG(` or weighted `SUM(` from coming
back into the package.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import duckdb
import pytest

from eval_card_backend.canonicalise import stages

SRC = Path(__file__).parent.parent / "src" / "eval_card_backend"


def _scores() -> list[float]:
    rng = random.Random(27)
    return [round(rng.uniform(0, 100), 4) for _ in range(383)]


@pytest.mark.parametrize("threads", [1, 4])
def test_result_is_the_same_for_every_row_order(threads):
    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    con.execute("CREATE TABLE v AS SELECT unnest(?::DOUBLE[]) AS x", [_scores()])
    avg, total = stages.order_free_avg_sql("x"), stages.order_free_sum_sql("x")
    seen_plain, seen = set(), set()
    for order in ("x ASC", "x DESC", "hash(x)", "hash(x + 1)"):
        con.execute(f"CREATE OR REPLACE TABLE t AS SELECT x FROM v ORDER BY {order}")
        seen_plain.add(con.execute("SELECT AVG(x), SUM(x) FROM t").fetchone())
        seen.add(con.execute(f"SELECT {avg}, {total} FROM t").fetchone())
    assert len(seen_plain) > 1, "these values no longer show the hazard"
    assert len(seen) == 1


def test_nulls_filters_and_empty_groups_behave_like_avg_and_sum():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES "
        "('a', 0.5, TRUE), ('a', NULL, TRUE), ('a', 1.5, TRUE), ('a', 8.0, FALSE), "
        "('b', NULL, TRUE), ('c', 2.0, FALSE)) AS t(g, x, keep)"
    )
    rows = con.execute(
        f"SELECT g, "
        f"{stages.order_free_avg_sql('x')}, AVG(x), "
        f"{stages.order_free_sum_sql('x')}, SUM(x), "
        f"{stages.order_free_avg_sql('x', 'keep')}, AVG(x) FILTER (WHERE keep), "
        f"{stages.order_free_sum_sql('x * 2', 'keep')}, SUM(x * 2) FILTER (WHERE keep) "
        f"FROM t GROUP BY g ORDER BY g"
    ).fetchall()
    assert rows == [
        ("a", 10.0 / 3, 10.0 / 3, 10.0, 10.0, 1.0, 1.0, 4.0, 4.0),
        ("b", None, None, None, None, None, None, None, None),
        ("c", 2.0, 2.0, 2.0, 2.0, None, None, None, None),
    ]


# AVG over a DECIMAL is an exact integer sum: the 0/1 rate idiom and an
# explicit DECIMAL cast.
_EXACT_AVG = re.compile(
    r"AVG\(\s*(?:CASE\s+WHEN\b[^;]*?\bTHEN\s+1\.0\s+ELSE\s+0\.0\s+END\s*\)|CAST\([^;]*?\bAS\s+DECIMAL)",
    re.S,
)


def float_aggregates_outside_helpers(source: str) -> list[str]:
    """Each `AVG(` that is not one of the exact forms, and each `SUM(` of a
    product, as the text that follows it."""
    found = []
    for match in re.finditer(r"\bAVG\(", source):
        if not _EXACT_AVG.match(source, match.start()):
            found.append(source[match.start():match.start() + 60].split("\n")[0])
    for match in re.finditer(r"\bSUM\([^()]*\*", source):
        found.append(source[match.start():match.start() + 60].split("\n")[0])
    return found


def test_the_scan_flags_a_double_average_and_a_weighted_sum():
    assert float_aggregates_outside_helpers(
        "SELECT ROUND(AVG(score), 12), SUM(score * n_samples), "
        "AVG(CASE WHEN gap THEN 1.0 ELSE 0.0 END), "
        "AVG(CAST(x AS DECIMAL(38, 18))), SUM(n_samples)"
    ) == [
        "AVG(score), 12), SUM(score * n_samples), AVG(CASE WHEN gap T",
        "SUM(score * n_samples), AVG(CASE WHEN gap THEN 1.0 ELSE 0.0 ",
    ]


def test_no_double_average_or_weighted_sum_bypasses_the_helpers():
    offenders = {
        str(path.relative_to(SRC)): found
        for path in sorted(SRC.rglob("*.py"))
        if (found := float_aggregates_outside_helpers(path.read_text()))
    }
    assert not offenders, (
        "use stages.order_free_avg_sql / order_free_sum_sql, or average a "
        f"DECIMAL: {offenders}"
    )

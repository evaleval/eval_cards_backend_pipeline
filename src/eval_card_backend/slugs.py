"""Producer-owned URL slug helpers.

Slugs in the view layer are RFC 3986 percent-encoded forms of canonical
IDs, used directly as URL path segments by the frontend. The frontend
decodes via standard `decodeURIComponent` on `<Link>` href construction
and otherwise treats slugs as opaque.

`safe=''` means every reserved character (including `/`, `:`, `?`) gets
percent-encoded — the slug is path-safe with no per-frontend escape
helpers and no `__`-style collision risk.
"""
from __future__ import annotations

from urllib.parse import quote


def url_encode(value: str | None) -> str | None:
    if value is None:
        return None
    return quote(value, safe="")


def metric_summary_id(
    benchmark_id: str | None, metric_id: str | None
) -> str | None:
    if benchmark_id is None or metric_id is None:
        return None
    return quote(f"{benchmark_id}:{metric_id}", safe="")


# Summary-score keyword set. Conservative starting set; extend when surfaced.
SUMMARY_METRIC_KEYWORDS: frozenset[str] = frozenset(
    {"overall", "aggregate", "total", "all"}
)


def is_summary_score(
    metric_id: str | None,
    parent_benchmark_id: str | None,
    benchmark_id: str | None,
) -> bool:
    """True when this row is a suite rollup metric (drives the 'Rollup'
    badge and the coverage-matrix top-M filter). Requires the benchmark
    to have a real parent that's distinct from itself, so a standalone
    benchmark whose metric happens to be named 'overall' is NOT a summary.
    """
    if metric_id is None or parent_benchmark_id is None or benchmark_id is None:
        return False
    if parent_benchmark_id == benchmark_id:
        return False
    return metric_id.lower() in SUMMARY_METRIC_KEYWORDS

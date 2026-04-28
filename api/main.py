"""FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from shared.config import settings
from shared.duckdb_backend import DuckDBBackend

app = FastAPI(
    title="Eval Cards Backend Pipeline",
    description="Ingestion and query backend for Eval Cards evaluation data",
    version="0.1.0",
)

backend = DuckDBBackend(settings.duckdb_path)

_BASE_DIR = Path(settings.ingestion_base_dir).resolve()


def _resolve_safe_path(raw_path: str) -> Path:
    """Resolve *raw_path* and ensure it is inside the configured base directory.

    Symlinks are rejected when the final ingestion path itself is a symlink, and any
    path whose resolved location is outside the base directory is also rejected.
    """
    candidate = Path(raw_path)
    if candidate.is_symlink():
        raise HTTPException(
            status_code=400,
            detail="Symlinks are not permitted as the final ingestion path.",
        )
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to resolve path: {exc}",
        )
    if not resolved.is_relative_to(_BASE_DIR):
        raise HTTPException(
            status_code=400,
            detail=f"Path must be inside the configured ingestion base directory: {_BASE_DIR}",
        )
    return resolved


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"ok": True}


@app.post("/ingest/aggregate")
async def ingest_aggregate(jsonl_path: str | None = None) -> dict:
    """Ingest aggregate JSONL records (eval.schema.json shape) into DuckDB."""
    raw = jsonl_path or settings.default_aggregate_jsonl_path
    path = _resolve_safe_path(raw)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Aggregate file not found: {path}")

    try:
        stats = backend.ingest_aggregate_jsonl(path)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc

    return {"ok": True, "path": str(path), **stats}


@app.post("/ingest/instance")
async def ingest_instance(jsonl_path: str | None = None) -> dict:
    """Ingest instance-level JSONL records (instance_level_eval.schema.json shape) into DuckDB."""
    raw = jsonl_path or settings.default_instance_jsonl_path
    path = _resolve_safe_path(raw)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Instance file not found: {path}")

    try:
        stats = backend.ingest_instance_jsonl(path)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc

    return {"ok": True, "path": str(path), **stats}


@app.get("/stats")
async def get_stats() -> dict:
    """Return backend table-level stats."""
    return {"ok": True, "stats": backend.stats()}


@app.get("/sources")
async def sources(
    limit: int = Query(default=100, ge=1, le=5000),
) -> dict:
    """Return source-level coverage and identifier completeness summary."""
    return {"ok": True, "rows": backend.sources(limit=limit)}


@app.get("/models")
async def models(
    source_name: str | None = Query(
        default=None,
        description=(
            "Optional source filter. Use __missing_source__ for NULL source rows."
        ),
    ),
    benchmark_name: str | None = Query(
        default=None,
        description=(
            "Optional benchmark filter using source_data.dataset_name fallback "
            "to evaluation_name."
        ),
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Return model-level aggregates for dashboard displays."""
    rows = backend.models(
        source_name=source_name,
        benchmark_name=benchmark_name,
        limit=limit,
        offset=offset,
    )
    return {"ok": True, "rows": rows}


@app.get("/benchmarks")
async def benchmarks(
    source_name: str | None = Query(
        default=None,
        description=(
            "Optional source filter. Use __missing_source__ for NULL source rows."
        ),
    ),
    benchmark_name: str | None = Query(
        default=None,
        description=(
            "Optional benchmark filter using source_data.dataset_name fallback "
            "to evaluation_name."
        ),
    ),
    metric_kind: str | None = Query(default=None, description="Optional metric_kind filter."),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict:
    """Return benchmark+metric slices with joined instance coverage."""
    rows = backend.benchmarks(
        source_name=source_name,
        benchmark_name=benchmark_name,
        metric_kind=metric_kind,
        limit=limit,
    )
    return {"ok": True, "rows": rows}


@app.get("/benchmarks/{benchmark_name}/models")
async def benchmark_models(
    benchmark_name: str,
    source_name: str | None = Query(
        default=None,
        description=(
            "Optional source filter. Use __missing_source__ for NULL source rows."
        ),
    ),
    metric_identity: str | None = Query(
        default=None,
        description=(
            "Optional metric identity filter using metric_id fallback "
            "to metric_name."
        ),
    ),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict:
    """Return ranked model rows for a benchmark, grouped by metric identity."""
    rows = backend.benchmark_model_rankings(
        benchmark_name=benchmark_name,
        source_name=source_name,
        metric_identity=metric_identity,
        limit=limit,
    )
    return {"ok": True, "rows": rows}


@app.get("/metrics/top-models")
async def top_model_metrics(
    metric_kind: str | None = Query(default=None, description="Filter by metric_kind"),
    metric_name: str | None = Query(default=None, description="Filter by metric_name"),
    source_name: str | None = Query(default=None, description="Filter by source_name"),
    limit: int = Query(default=20, ge=1, le=500),
) -> dict:
    """Return top model averages for a given metric slice."""
    rows = backend.top_model_metrics(
        metric_kind=metric_kind,
        metric_name=metric_name,
        source_name=source_name,
        limit=limit,
    )
    return {"ok": True, "rows": rows}


@app.get("/join-integrity")
async def join_integrity() -> dict:
    """Report deterministic linkage coverage across aggregate and instance tables."""
    return {"ok": True, "join_integrity": backend.join_integrity()}


@app.get("/quality/orphan-runs")
async def quality_orphan_runs(
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    """Report runs with detailed instance references that remain missing, partial, or repaired."""
    return {"ok": True, "rows": backend.orphan_runs(limit=limit)}


@app.get("/quality/identifier-issues")
async def quality_identifier_issues(
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    """Report identifier sparsity plus repaired or unvalidated instance linkage examples."""
    return {"ok": True, "issues": backend.identifier_issues(limit=limit)}

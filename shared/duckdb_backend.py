"""DuckDB-backed storage and query helpers for aggregate + instance eval data."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip().lower())
    return cleaned


def _name_join_id(evaluation_name: str) -> str:
    return f"name:{_normalize_name(evaluation_name)}"


def _slugify(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


_UUID_FILE_RE = re.compile(
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})(?:_samples)?(?:\.jsonl?)?$",
    re.IGNORECASE,
)


def _file_link_key(file_path: str | Path | None) -> str | None:
    if file_path is None:
        return None

    filename = Path(str(file_path)).name.strip()
    if not filename:
        return None

    match = _UUID_FILE_RE.search(filename)
    if match:
        return f"uuid:{match.group('uuid').lower()}"

    return f"name:{filename.lower()}"


_METRIC_LIKE_EVALUATION_RE = re.compile(
    r"(?:^|[_\s.-])(accuracy|acc|win[_\s-]?rate|pass@?\d+|avg|average|latency|"
    r"attempts|score|elo|exact[_\s-]?match|f1|precision|recall)(?:$|[_\s.-])",
    re.IGNORECASE,
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_dumps_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class DuckDBBackend:
    """Simple backend storage optimized for ingestion/idempotency and SQL analytics."""

    def __init__(self, db_path: str = "data/backend.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                evaluation_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                retrieved_timestamp TEXT NOT NULL,
                source_name TEXT,
                source_type TEXT,
                source_organization_name TEXT,
                source_organization_url TEXT,
                evaluator_relationship TEXT,
                model_id TEXT NOT NULL,
                model_name TEXT,
                model_developer TEXT,
                eval_library_name TEXT,
                eval_library_version TEXT,
                run_fingerprint TEXT,
                content_hash TEXT,
                hash_algorithm TEXT,
                canonicalization_version TEXT,
                detailed_results_file_path TEXT,
                detailed_results_file_key TEXT,
                detailed_results_total_rows INTEGER,
                detailed_results_observed_rows INTEGER,
                source_record_url TEXT,
                instance_artifact_path TEXT,
                instance_artifact_url TEXT,
                benchmark_folder TEXT,
                raw_model_id TEXT,
                model_route_id TEXT,
                harness_id TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_detailed_results_key
            ON evaluation_runs (detailed_results_file_key)
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                evaluation_id TEXT NOT NULL,
                metric_index INTEGER NOT NULL,
                evaluation_result_id TEXT,
                result_join_id TEXT NOT NULL,
                name_join_id TEXT,
                join_key_source TEXT NOT NULL,
                evaluation_name TEXT NOT NULL,
                metric_id TEXT,
                metric_name TEXT,
                metric_kind TEXT,
                metric_unit TEXT,
                metric_parameters_json TEXT,
                lower_is_better BOOLEAN,
                score_type TEXT,
                min_score DOUBLE,
                max_score DOUBLE,
                score DOUBLE NOT NULL,
                source_dataset_name TEXT,
                source_type TEXT,
                source_ref TEXT,
                metric_config_json TEXT,
                score_details_json TEXT,
                canonical_metric_tuple TEXT,
                run_id TEXT,
                canonical_model_id TEXT,
                benchmark_family_id TEXT,
                eval_slice_id TEXT,
                canonical_metric_id TEXT,
                harness_id TEXT,
                result_index INTEGER,
                raw_evaluation_name TEXT,
                raw_metric_id TEXT,
                raw_metric_name TEXT,
                raw_metric_kind TEXT,
                raw_harness_name TEXT,
                model_route_id TEXT,
                eval_summary_id TEXT,
                metric_summary_id TEXT,
                benchmark_family_key TEXT,
                benchmark_parent_key TEXT,
                benchmark_leaf_key TEXT,
                slice_key TEXT,
                category TEXT,
                metric_key TEXT,
                source_record_url TEXT,
                detailed_results_url TEXT,
                PRIMARY KEY (evaluation_id, metric_index)
            )
            """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_join
            ON evaluation_metrics (evaluation_id, result_join_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_name_join
            ON evaluation_metrics (evaluation_id, name_join_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_kind
            ON evaluation_metrics (metric_kind)
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instance_evaluations (
                evaluation_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                evaluation_name TEXT NOT NULL,
                evaluation_result_id TEXT,
                result_join_id TEXT NOT NULL,
                name_join_id TEXT,
                join_key_source TEXT NOT NULL,
                original_evaluation_id TEXT,
                evaluation_id_validation_status TEXT,
                ingest_source_path TEXT,
                model_id TEXT NOT NULL,
                model_route_id TEXT,
                eval_summary_id TEXT,
                metric_summary_id TEXT,
                benchmark_family_key TEXT,
                eval_slice_id TEXT,
                metric_key TEXT,
                score DOUBLE,
                is_correct BOOLEAN,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, sample_id, result_join_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_instance_join
            ON instance_evaluations (evaluation_id, result_join_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_instance_name_join
            ON instance_evaluations (evaluation_id, name_join_id)
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_quality_issues (
                issue_id TEXT PRIMARY KEY,
                issue_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                evaluation_id TEXT,
                metric_index INTEGER,
                sample_id TEXT,
                eval_summary_id TEXT,
                metric_summary_id TEXT,
                message TEXT NOT NULL,
                details_json TEXT
            )
            """
        )

        # Backward-compatible migration path for databases created before name_join_id existed.
        self.conn.execute(
            """
            ALTER TABLE evaluation_runs
            ADD COLUMN IF NOT EXISTS detailed_results_file_path TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE evaluation_runs
            ADD COLUMN IF NOT EXISTS detailed_results_file_key TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE evaluation_runs
            ADD COLUMN IF NOT EXISTS detailed_results_total_rows INTEGER
            """
        )
        self.conn.execute(
            """
            ALTER TABLE evaluation_metrics
            ADD COLUMN IF NOT EXISTS name_join_id TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE instance_evaluations
            ADD COLUMN IF NOT EXISTS name_join_id TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE instance_evaluations
            ADD COLUMN IF NOT EXISTS original_evaluation_id TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE instance_evaluations
            ADD COLUMN IF NOT EXISTS evaluation_id_validation_status TEXT
            """
        )
        self.conn.execute(
            """
            ALTER TABLE instance_evaluations
            ADD COLUMN IF NOT EXISTS ingest_source_path TEXT
            """
        )
        for column, column_type in [
            ("detailed_results_observed_rows", "INTEGER"),
            ("source_record_url", "TEXT"),
            ("instance_artifact_path", "TEXT"),
            ("instance_artifact_url", "TEXT"),
            ("benchmark_folder", "TEXT"),
            ("raw_model_id", "TEXT"),
            ("model_route_id", "TEXT"),
            ("harness_id", "TEXT"),
        ]:
            self.conn.execute(
                f"ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS {column} {column_type}"
            )

        for column, column_type in [
            ("canonical_metric_tuple", "TEXT"),
            ("run_id", "TEXT"),
            ("canonical_model_id", "TEXT"),
            ("benchmark_family_id", "TEXT"),
            ("eval_slice_id", "TEXT"),
            ("canonical_metric_id", "TEXT"),
            ("harness_id", "TEXT"),
            ("result_index", "INTEGER"),
            ("raw_evaluation_name", "TEXT"),
            ("raw_metric_id", "TEXT"),
            ("raw_metric_name", "TEXT"),
            ("raw_metric_kind", "TEXT"),
            ("raw_harness_name", "TEXT"),
            ("model_route_id", "TEXT"),
            ("eval_summary_id", "TEXT"),
            ("metric_summary_id", "TEXT"),
            ("benchmark_family_key", "TEXT"),
            ("benchmark_parent_key", "TEXT"),
            ("benchmark_leaf_key", "TEXT"),
            ("slice_key", "TEXT"),
            ("category", "TEXT"),
            ("metric_key", "TEXT"),
            ("source_record_url", "TEXT"),
            ("detailed_results_url", "TEXT"),
        ]:
            self.conn.execute(
                f"ALTER TABLE evaluation_metrics ADD COLUMN IF NOT EXISTS {column} {column_type}"
            )

        for column, column_type in [
            ("model_route_id", "TEXT"),
            ("eval_summary_id", "TEXT"),
            ("metric_summary_id", "TEXT"),
            ("benchmark_family_key", "TEXT"),
            ("eval_slice_id", "TEXT"),
            ("metric_key", "TEXT"),
        ]:
            self.conn.execute(
                f"ALTER TABLE instance_evaluations ADD COLUMN IF NOT EXISTS {column} {column_type}"
            )

        runs_missing_detailed_results = self.conn.execute(
            """
            SELECT evaluation_id, raw_json
            FROM evaluation_runs
            WHERE detailed_results_file_key IS NULL
              AND raw_json LIKE '%detailed_evaluation_results%'
            """
        ).fetchall()
        for evaluation_id, raw_json in runs_missing_detailed_results:
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                continue

            (
                detailed_results_file_path,
                detailed_results_file_key,
                detailed_results_total_rows,
            ) = self._detailed_results_info(payload)
            self.conn.execute(
                """
                UPDATE evaluation_runs
                SET detailed_results_file_path = ?,
                    detailed_results_file_key = ?,
                    detailed_results_total_rows = ?
                WHERE evaluation_id = ?
                """,
                [
                    detailed_results_file_path,
                    detailed_results_file_key,
                    detailed_results_total_rows,
                    evaluation_id,
                ],
            )

        metrics_missing_name_join = self.conn.execute(
            """
            SELECT evaluation_id, metric_index, evaluation_name
            FROM evaluation_metrics
            WHERE name_join_id IS NULL
            """
        ).fetchall()
        for evaluation_id, metric_index, evaluation_name in metrics_missing_name_join:
            self.conn.execute(
                """
                UPDATE evaluation_metrics
                SET name_join_id = ?
                WHERE evaluation_id = ? AND metric_index = ?
                """,
                [_name_join_id(evaluation_name), evaluation_id, metric_index],
            )

        instances_missing_name_join = self.conn.execute(
            """
            SELECT evaluation_id, sample_id, result_join_id, evaluation_name
            FROM instance_evaluations
            WHERE name_join_id IS NULL
            """
        ).fetchall()
        for (
            evaluation_id,
            sample_id,
            result_join_id,
            evaluation_name,
        ) in instances_missing_name_join:
            self.conn.execute(
                """
                UPDATE instance_evaluations
                SET name_join_id = ?
                WHERE evaluation_id = ? AND sample_id = ? AND result_join_id = ?
                """,
                [_name_join_id(evaluation_name), evaluation_id, sample_id, result_join_id],
            )

        self.conn.execute(
            """
            UPDATE instance_evaluations
            SET original_evaluation_id = evaluation_id
            WHERE original_evaluation_id IS NULL
            """
        )
        self.conn.execute(
            """
            UPDATE instance_evaluations
            SET evaluation_id_validation_status = 'legacy_unvalidated'
            WHERE evaluation_id_validation_status IS NULL
            """
        )

    def _source_reference(self, source_data: dict[str, Any]) -> str | None:
        source_type = source_data.get("source_type")
        if source_type == "url":
            urls = source_data.get("url")
            if isinstance(urls, list):
                return ", ".join(str(x) for x in urls)
        if source_type == "hf_dataset":
            repo = source_data.get("hf_repo")
            split = source_data.get("hf_split")
            if repo and split:
                return f"{repo}:{split}"
            if repo:
                return str(repo)
        dataset_name = source_data.get("dataset_name")
        return str(dataset_name) if dataset_name is not None else None

    def _join_ids(
        self,
        evaluation_result_id: str | None,
        evaluation_name: str,
    ) -> tuple[str, str, str]:
        name_join_id = _name_join_id(evaluation_name)
        if evaluation_result_id:
            return evaluation_result_id, name_join_id, "evaluation_result_id"
        return name_join_id, name_join_id, "evaluation_name_fallback"

    def _detailed_results_info(
        self, payload: dict[str, Any]
    ) -> tuple[str | None, str | None, int | None]:
        detailed_results = payload.get("detailed_evaluation_results") or {}
        if not isinstance(detailed_results, dict):
            return None, None, None

        file_path = detailed_results.get("file_path")
        if file_path is not None:
            file_path = str(file_path).strip() or None

        return (
            file_path,
            _file_link_key(file_path),
            _to_int(detailed_results.get("total_rows")),
        )

    def _expected_evaluation_id_for_instance_path(
        self, path: Path
    ) -> tuple[str | None, str]:
        file_link_key = _file_link_key(path)
        if file_link_key is None:
            return None, "no_matching_detailed_results"

        rows = self.conn.execute(
            """
            SELECT evaluation_id
            FROM evaluation_runs
            WHERE detailed_results_file_key = ?
            ORDER BY evaluation_id
            """,
            [file_link_key],
        ).fetchall()
        if len(rows) == 1:
            return str(rows[0][0]), "matched_detailed_results"
        if len(rows) > 1:
            return None, "ambiguous_detailed_results"
        return None, "no_matching_detailed_results"

    def _record_quality_issue(
        self,
        *,
        issue_type: str,
        severity: str,
        evaluation_id: str | None,
        metric_index: int | None = None,
        sample_id: str | None = None,
        eval_summary_id: str | None = None,
        metric_summary_id: str | None = None,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        issue_id = "|".join(
            [
                issue_type,
                evaluation_id or "",
                "" if metric_index is None else str(metric_index),
                sample_id or "",
                metric_summary_id or "",
            ]
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO semantic_quality_issues (
                issue_id,
                issue_type,
                severity,
                evaluation_id,
                metric_index,
                sample_id,
                eval_summary_id,
                metric_summary_id,
                message,
                details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                issue_id,
                issue_type,
                severity,
                evaluation_id,
                metric_index,
                sample_id,
                eval_summary_id,
                metric_summary_id,
                message,
                _json_dumps_or_none(details),
            ],
        )

    def _count_jsonl_rows(self, path: Path) -> int | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _pipeline_instance_path(
        self, evaluation: dict[str, Any], output_dir: Path | None
    ) -> Path | None:
        artifact = evaluation.get("instance_artifact") or {}
        if not isinstance(artifact, dict):
            return None
        artifact_path = artifact.get("path")
        if not artifact_path:
            return None
        path = Path(str(artifact_path))
        if path.is_absolute():
            return path
        root = output_dir or Path(".")
        return (root / path).resolve()

    def ingest_pipeline_evaluations(
        self, evaluations: list[dict[str, Any]], output_dir: str | Path | None = None
    ) -> dict[str, Any]:
        """Ingest enriched pipeline evaluation objects and their owned instance JSONL files."""
        output_root = Path(output_dir).resolve() if output_dir is not None else None
        runs = 0
        metrics = 0
        instance_files = 0
        instance_rows = 0

        for evaluation in evaluations:
            stats = self.ingest_pipeline_evaluation(evaluation, output_root)
            runs += stats["runs_ingested"]
            metrics += stats["metrics_ingested"]
            instance_path = self._pipeline_instance_path(evaluation, output_root)
            if instance_path is not None and instance_path.exists():
                ingest_stats = self.ingest_instance_jsonl(instance_path)
                instance_files += 1
                instance_rows += int(ingest_stats["instance_rows_ingested"])

        return {
            "runs_ingested": runs,
            "metrics_ingested": metrics,
            "instance_files_ingested": instance_files,
            "instance_rows_ingested": instance_rows,
            "semantic_quality_issues": int(
                self.conn.execute("SELECT COUNT(*) FROM semantic_quality_issues").fetchone()[0]
            ),
        }

    def ingest_pipeline_evaluation(
        self, evaluation: dict[str, Any], output_dir: Path | None = None
    ) -> dict[str, int]:
        evaluation_id = str(evaluation.get("evaluation_id", "")).strip()
        if not evaluation_id:
            return {"runs_ingested": 0, "metrics_ingested": 0}

        source_meta = evaluation.get("source_metadata") or {}
        if not isinstance(source_meta, dict):
            source_meta = {}
        model_info = evaluation.get("model_info") or {}
        if not isinstance(model_info, dict):
            model_info = {}
        eval_library = evaluation.get("eval_library") or {}
        if not isinstance(eval_library, dict):
            eval_library = {}

        canonical_model_id = str(
            model_info.get("family_id") or model_info.get("id") or ""
        ).strip()
        raw_model_id = str(model_info.get("id") or canonical_model_id).strip()
        if not canonical_model_id:
            raise ValueError(f"Pipeline evaluation {evaluation_id!r} is missing model identity")

        artifact = evaluation.get("instance_artifact") or {}
        if not isinstance(artifact, dict):
            artifact = {}
        instance_artifact_path = artifact.get("path")
        instance_artifact_url = artifact.get("url")
        instance_path = self._pipeline_instance_path(evaluation, output_dir)
        observed_instance_rows = (
            self._count_jsonl_rows(instance_path) if instance_path is not None else None
        )
        detailed_meta = evaluation.get("detailed_evaluation_results_meta") or {}
        if not isinstance(detailed_meta, dict):
            detailed_meta = {}
        detailed_results_total_rows = _to_int(detailed_meta.get("total_rows"))
        detailed_results_file_path = (
            str(instance_artifact_path).strip()
            if instance_artifact_path
            else None
        )
        detailed_results_file_key = _file_link_key(detailed_results_file_path)

        result_items = evaluation.get("evaluation_results")
        if not isinstance(result_items, list):
            result_items = []

        self.conn.begin()
        try:
            self.conn.execute(
                "DELETE FROM semantic_quality_issues WHERE evaluation_id = ?",
                [evaluation_id],
            )
            self.conn.execute(
                "DELETE FROM evaluation_metrics WHERE evaluation_id = ?",
                [evaluation_id],
            )
            self.conn.execute(
                "DELETE FROM evaluation_runs WHERE evaluation_id = ?",
                [evaluation_id],
            )
            self.conn.execute(
                """
                INSERT INTO evaluation_runs (
                    evaluation_id,
                    schema_version,
                    retrieved_timestamp,
                    source_name,
                    source_type,
                    source_organization_name,
                    source_organization_url,
                    evaluator_relationship,
                    model_id,
                    model_name,
                    model_developer,
                    eval_library_name,
                    eval_library_version,
                    run_fingerprint,
                    content_hash,
                    hash_algorithm,
                    canonicalization_version,
                    detailed_results_file_path,
                    detailed_results_file_key,
                    detailed_results_total_rows,
                    detailed_results_observed_rows,
                    source_record_url,
                    instance_artifact_path,
                    instance_artifact_url,
                    benchmark_folder,
                    raw_model_id,
                    model_route_id,
                    harness_id,
                    raw_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    evaluation_id,
                    str(evaluation.get("schema_version", "")),
                    str(evaluation.get("retrieved_timestamp", "")),
                    source_meta.get("source_name"),
                    source_meta.get("source_type"),
                    source_meta.get("source_organization_name"),
                    source_meta.get("source_organization_url"),
                    source_meta.get("evaluator_relationship"),
                    canonical_model_id,
                    model_info.get("family_name") or model_info.get("name"),
                    model_info.get("developer"),
                    eval_library.get("name"),
                    eval_library.get("version"),
                    None,
                    None,
                    None,
                    "pipeline-enriched-v1",
                    detailed_results_file_path,
                    detailed_results_file_key,
                    detailed_results_total_rows,
                    observed_instance_rows,
                    evaluation.get("source_record_url"),
                    instance_artifact_path,
                    instance_artifact_url,
                    evaluation.get("benchmark"),
                    raw_model_id,
                    model_info.get("model_route_id"),
                    eval_library.get("name"),
                    _json_dumps(evaluation),
                ],
            )

            if (
                detailed_results_total_rows is not None
                and observed_instance_rows is not None
                and detailed_results_total_rows != observed_instance_rows
            ):
                self._record_quality_issue(
                    issue_type="detailed_results_row_count_mismatch",
                    severity="warning",
                    evaluation_id=evaluation_id,
                    message="Declared detailed result row count differs from observed JSONL rows.",
                    details={
                        "declared_total_rows": detailed_results_total_rows,
                        "observed_rows": observed_instance_rows,
                        "instance_artifact_path": instance_artifact_path,
                    },
                )

            metrics = 0
            for metric_index, result in enumerate(result_items):
                if not isinstance(result, dict):
                    continue
                score_details = result.get("score_details") or {}
                if not isinstance(score_details, dict):
                    score_details = {}
                score = _to_float(score_details.get("score"))
                if score is None:
                    continue

                metric_cfg = result.get("metric_config") or {}
                if not isinstance(metric_cfg, dict):
                    metric_cfg = {}
                normalized = result.get("normalized_result") or {}
                if not isinstance(normalized, dict):
                    normalized = {}
                source_data = result.get("source_data") or evaluation.get("source_data") or {}
                if not isinstance(source_data, dict):
                    source_data = {}

                evaluation_name = str(result.get("evaluation_name", "")).strip()
                if not evaluation_name:
                    continue
                evaluation_result_id = result.get("evaluation_result_id")
                if evaluation_result_id is not None:
                    evaluation_result_id = str(evaluation_result_id)
                result_join_id, name_join_id, join_key_source = self._join_ids(
                    evaluation_result_id,
                    evaluation_name,
                )

                metric_id = (
                    normalized.get("metric_id")
                    or metric_cfg.get("metric_id")
                    or normalized.get("metric_key")
                )
                metric_name = normalized.get("metric_name") or metric_cfg.get("metric_name")
                metric_kind = metric_cfg.get("metric_kind")
                metric_key = normalized.get("metric_key") or metric_id or metric_name
                eval_summary_id = normalized.get("eval_summary_id")
                if not eval_summary_id:
                    parent_key = (
                        normalized.get("benchmark_parent_key")
                        or evaluation.get("benchmark")
                        or source_data.get("dataset_name")
                    )
                    benchmark_key = (
                        normalized.get("benchmark_leaf_key")
                        or normalized.get("benchmark_family_key")
                        or evaluation.get("benchmark")
                        or source_data.get("dataset_name")
                        or evaluation_name
                    )
                    pieces = []
                    if parent_key:
                        pieces.append(parent_key)
                    if benchmark_key and str(benchmark_key) != str(parent_key):
                        pieces.append(benchmark_key)
                    eval_summary_id = _slugify("__".join(str(piece) for piece in pieces if piece))
                metric_summary_id = normalized.get("metric_summary_id")
                if not metric_summary_id:
                    pieces = [eval_summary_id, normalized.get("slice_key"), metric_key]
                    metric_summary_id = _slugify(
                        "__".join(str(piece) for piece in pieces if piece)
                    )

                benchmark_family_id = (
                    normalized.get("benchmark_family_key")
                    or normalized.get("benchmark_parent_key")
                    or evaluation.get("benchmark")
                    or ""
                )
                eval_slice_id = (
                    normalized.get("slice_key")
                    or normalized.get("benchmark_leaf_key")
                    or normalized.get("benchmark_family_key")
                    or evaluation_name
                )
                harness_id = str(eval_library.get("name") or "unknown").strip() or "unknown"
                canonical_metric_id = str(metric_key or metric_id or evaluation_name)
                canonical_tuple = "|".join(
                    [
                        evaluation_id,
                        canonical_model_id,
                        str(benchmark_family_id),
                        str(eval_slice_id),
                        canonical_metric_id,
                        harness_id,
                        str(metric_index),
                    ]
                )

                self.conn.execute(
                    """
                    INSERT INTO evaluation_metrics (
                        evaluation_id,
                        metric_index,
                        evaluation_result_id,
                        result_join_id,
                        name_join_id,
                        join_key_source,
                        evaluation_name,
                        metric_id,
                        metric_name,
                        metric_kind,
                        metric_unit,
                        metric_parameters_json,
                        lower_is_better,
                        score_type,
                        min_score,
                        max_score,
                        score,
                        source_dataset_name,
                        source_type,
                        source_ref,
                        metric_config_json,
                        score_details_json,
                        canonical_metric_tuple,
                        run_id,
                        canonical_model_id,
                        benchmark_family_id,
                        eval_slice_id,
                        canonical_metric_id,
                        harness_id,
                        result_index,
                        raw_evaluation_name,
                        raw_metric_id,
                        raw_metric_name,
                        raw_metric_kind,
                        raw_harness_name,
                        model_route_id,
                        eval_summary_id,
                        metric_summary_id,
                        benchmark_family_key,
                        benchmark_parent_key,
                        benchmark_leaf_key,
                        slice_key,
                        category,
                        metric_key,
                        source_record_url,
                        detailed_results_url
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        evaluation_id,
                        metric_index,
                        evaluation_result_id,
                        result_join_id,
                        name_join_id,
                        join_key_source,
                        evaluation_name,
                        metric_id,
                        metric_name,
                        metric_kind,
                        metric_cfg.get("metric_unit"),
                        _json_dumps_or_none(metric_cfg.get("metric_parameters")),
                        bool(metric_cfg.get("lower_is_better")),
                        metric_cfg.get("score_type"),
                        _to_float(metric_cfg.get("min_score")),
                        _to_float(metric_cfg.get("max_score")),
                        score,
                        source_data.get("dataset_name"),
                        source_data.get("source_type"),
                        self._source_reference(source_data),
                        _json_dumps(metric_cfg),
                        _json_dumps(score_details),
                        canonical_tuple,
                        evaluation_id,
                        canonical_model_id,
                        benchmark_family_id,
                        eval_slice_id,
                        canonical_metric_id,
                        harness_id,
                        metric_index,
                        evaluation_name,
                        metric_cfg.get("metric_id"),
                        metric_cfg.get("metric_name"),
                        metric_cfg.get("metric_kind"),
                        eval_library.get("name"),
                        model_info.get("model_route_id"),
                        eval_summary_id,
                        metric_summary_id,
                        normalized.get("benchmark_family_key"),
                        normalized.get("benchmark_parent_key"),
                        normalized.get("benchmark_leaf_key"),
                        normalized.get("slice_key"),
                        normalized.get("category"),
                        metric_key,
                        evaluation.get("source_record_url"),
                        evaluation.get("detailed_evaluation_results"),
                    ],
                )
                metrics += 1

                if evaluation_result_id is None:
                    self._record_quality_issue(
                        issue_type="missing_evaluation_result_id",
                        severity="warning",
                        evaluation_id=evaluation_id,
                        metric_index=metric_index,
                        eval_summary_id=eval_summary_id,
                        metric_summary_id=metric_summary_id,
                        message="Aggregate metric is missing evaluation_result_id; name fallback may be required.",
                    )
                if not metric_cfg.get("metric_id") or not metric_cfg.get("metric_name"):
                    self._record_quality_issue(
                        issue_type="missing_metric_identity",
                        severity="warning",
                        evaluation_id=evaluation_id,
                        metric_index=metric_index,
                        eval_summary_id=eval_summary_id,
                        metric_summary_id=metric_summary_id,
                        message="Aggregate metric is missing explicit metric_id or metric_name.",
                    )
                if metric_cfg.get("metric_name") and _normalize_name(evaluation_name) == _normalize_name(
                    str(metric_cfg.get("metric_name"))
                ):
                    self._record_quality_issue(
                        issue_type="evaluation_metric_name_collision",
                        severity="warning",
                        evaluation_id=evaluation_id,
                        metric_index=metric_index,
                        eval_summary_id=eval_summary_id,
                        metric_summary_id=metric_summary_id,
                        message="evaluation_name and metric_name are identical after normalization.",
                    )
                if (
                    not metric_cfg.get("metric_id")
                    and _METRIC_LIKE_EVALUATION_RE.search(evaluation_name)
                ):
                    self._record_quality_issue(
                        issue_type="metric_like_evaluation_name",
                        severity="warning",
                        evaluation_id=evaluation_id,
                        metric_index=metric_index,
                        eval_summary_id=eval_summary_id,
                        metric_summary_id=metric_summary_id,
                        message="evaluation_name appears to carry metric semantics without explicit metric_id.",
                    )

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return {"runs_ingested": 1, "metrics_ingested": metrics}

    def ingest_aggregate_jsonl(self, jsonl_path: str | Path) -> dict[str, int]:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(path)

        runs = 0
        metrics = 0

        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue

                payload = json.loads(raw_line)
                evaluation_id = str(payload.get("evaluation_id", "")).strip()
                if not evaluation_id:
                    continue

                dedupe = payload.get("dedupe_identity") or {}
                if not isinstance(dedupe, dict):
                    dedupe = {}

                source_meta = payload.get("source_metadata") or {}
                if not isinstance(source_meta, dict):
                    source_meta = {}

                model_info = payload.get("model_info") or {}
                if not isinstance(model_info, dict):
                    model_info = {}

                eval_library = payload.get("eval_library") or {}
                if not isinstance(eval_library, dict):
                    eval_library = {}

                model_id = str(model_info.get("id", "")).strip()
                if not model_id:
                    raise ValueError(
                        f"Row with evaluation_id={evaluation_id!r} is missing required"
                        " field model_info.id"
                    )

                try:
                    (
                        detailed_results_file_path,
                        detailed_results_file_key,
                        detailed_results_total_rows,
                    ) = self._detailed_results_info(payload)
                    self.conn.begin()
                    self.conn.execute(
                        "DELETE FROM evaluation_metrics WHERE evaluation_id = ?",
                        [evaluation_id],
                    )
                    self.conn.execute(
                        "DELETE FROM evaluation_runs WHERE evaluation_id = ?",
                        [evaluation_id],
                    )

                    self.conn.execute(
                        """
                        INSERT INTO evaluation_runs (
                            evaluation_id,
                            schema_version,
                            retrieved_timestamp,
                            source_name,
                            source_type,
                            source_organization_name,
                            source_organization_url,
                            evaluator_relationship,
                            model_id,
                            model_name,
                            model_developer,
                            eval_library_name,
                            eval_library_version,
                            run_fingerprint,
                            content_hash,
                            hash_algorithm,
                            canonicalization_version,
                            detailed_results_file_path,
                            detailed_results_file_key,
                            detailed_results_total_rows,
                            raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            evaluation_id,
                            str(payload.get("schema_version", "")),
                            str(payload.get("retrieved_timestamp", "")),
                            source_meta.get("source_name"),
                            source_meta.get("source_type"),
                            source_meta.get("source_organization_name"),
                            source_meta.get("source_organization_url"),
                            source_meta.get("evaluator_relationship"),
                            model_id,
                            model_info.get("name"),
                            model_info.get("developer"),
                            eval_library.get("name"),
                            eval_library.get("version"),
                            dedupe.get("run_fingerprint"),
                            dedupe.get("content_hash"),
                            dedupe.get("hash_algorithm"),
                            dedupe.get("canonicalization_version"),
                            detailed_results_file_path,
                            detailed_results_file_key,
                            detailed_results_total_rows,
                            json.dumps(payload, ensure_ascii=False),
                        ],
                    )

                    result_items = payload.get("evaluation_results")
                    if not isinstance(result_items, list):
                        result_items = []

                    for metric_index, result in enumerate(result_items):
                        if not isinstance(result, dict):
                            continue

                        metric_cfg = result.get("metric_config") or {}
                        if not isinstance(metric_cfg, dict):
                            metric_cfg = {}

                        score_details = result.get("score_details") or {}
                        if not isinstance(score_details, dict):
                            score_details = {}

                        score = _to_float(score_details.get("score"))
                        if score is None:
                            continue

                        evaluation_name = str(result.get("evaluation_name", "")).strip()
                        if not evaluation_name:
                            continue

                        evaluation_result_id = result.get("evaluation_result_id")
                        if evaluation_result_id is not None:
                            evaluation_result_id = str(evaluation_result_id)

                        result_join_id, name_join_id, join_key_source = self._join_ids(
                            evaluation_result_id,
                            evaluation_name,
                        )

                        source_data = result.get("source_data") or {}
                        if not isinstance(source_data, dict):
                            source_data = {}

                        metric_parameters = metric_cfg.get("metric_parameters")
                        metric_parameters_json = None
                        if isinstance(metric_parameters, dict):
                            metric_parameters_json = json.dumps(
                                metric_parameters,
                                sort_keys=True,
                                ensure_ascii=False,
                            )

                        self.conn.execute(
                            """
                            INSERT INTO evaluation_metrics (
                                evaluation_id,
                                metric_index,
                                evaluation_result_id,
                                result_join_id,
                                name_join_id,
                                join_key_source,
                                evaluation_name,
                                metric_id,
                                metric_name,
                                metric_kind,
                                metric_unit,
                                metric_parameters_json,
                                lower_is_better,
                                score_type,
                                min_score,
                                max_score,
                                score,
                                source_dataset_name,
                                source_type,
                                source_ref,
                                metric_config_json,
                                score_details_json
                            ) VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                            """,
                            [
                                evaluation_id,
                                metric_index,
                                evaluation_result_id,
                                result_join_id,
                                name_join_id,
                                join_key_source,
                                evaluation_name,
                                metric_cfg.get("metric_id"),
                                metric_cfg.get("metric_name"),
                                metric_cfg.get("metric_kind"),
                                metric_cfg.get("metric_unit"),
                                metric_parameters_json,
                                metric_cfg.get("lower_is_better"),
                                metric_cfg.get("score_type"),
                                _to_float(metric_cfg.get("min_score")),
                                _to_float(metric_cfg.get("max_score")),
                                score,
                                source_data.get("dataset_name"),
                                source_data.get("source_type"),
                                self._source_reference(source_data),
                                json.dumps(metric_cfg, ensure_ascii=False),
                                json.dumps(score_details, ensure_ascii=False),
                            ],
                        )
                        metrics += 1

                    runs += 1
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise

        return {"runs_ingested": runs, "metrics_ingested": metrics}

    def ingest_instance_jsonl(self, jsonl_path: str | Path) -> dict[str, Any]:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(path)

        rows = 0
        validated_rows = 0
        repaired_rows = 0
        filled_rows = 0
        unvalidated_rows = 0
        expected_evaluation_id, file_link_lookup_status = (
            self._expected_evaluation_id_for_instance_path(path)
        )
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue

                payload = json.loads(raw_line)
                original_evaluation_id = str(payload.get("evaluation_id", "")).strip()
                sample_id = str(payload.get("sample_id", "")).strip()
                evaluation_name = str(payload.get("evaluation_name", "")).strip()
                model_id = str(payload.get("model_id", "")).strip()

                evaluation_id = original_evaluation_id
                if expected_evaluation_id is not None:
                    if original_evaluation_id == expected_evaluation_id:
                        evaluation_id_validation_status = "validated_from_detailed_results"
                    elif original_evaluation_id:
                        evaluation_id = expected_evaluation_id
                        evaluation_id_validation_status = "repaired_from_detailed_results"
                    else:
                        evaluation_id = expected_evaluation_id
                        evaluation_id_validation_status = "filled_from_detailed_results"
                elif file_link_lookup_status == "ambiguous_detailed_results":
                    evaluation_id_validation_status = "payload_only_ambiguous_file_link"
                else:
                    evaluation_id_validation_status = "payload_only_unvalidated"

                if not all([evaluation_id, sample_id, evaluation_name, model_id]):
                    continue

                evaluation_result_id = payload.get("evaluation_result_id")
                if evaluation_result_id is not None:
                    evaluation_result_id = str(evaluation_result_id)

                result_join_id, name_join_id, join_key_source = self._join_ids(
                    evaluation_result_id,
                    evaluation_name,
                )

                evaluation_obj = payload.get("evaluation") or {}
                if not isinstance(evaluation_obj, dict):
                    evaluation_obj = {}

                score = _to_float(evaluation_obj.get("score"))
                is_correct = evaluation_obj.get("is_correct")
                if not isinstance(is_correct, bool):
                    is_correct = None

                hierarchy = payload.get("hierarchy") or {}
                if not isinstance(hierarchy, dict):
                    hierarchy = {}

                delete_evaluation_ids = {evaluation_id}
                if original_evaluation_id and original_evaluation_id != evaluation_id:
                    delete_evaluation_ids.add(original_evaluation_id)
                for delete_evaluation_id in delete_evaluation_ids:
                    self.conn.execute(
                        """
                        DELETE FROM instance_evaluations
                        WHERE evaluation_id = ?
                          AND sample_id = ?
                          AND result_join_id = ?
                          AND name_join_id = ?
                        """,
                        [delete_evaluation_id, sample_id, result_join_id, name_join_id],
                    )

                self.conn.execute(
                    """
                    INSERT INTO instance_evaluations (
                        evaluation_id,
                        sample_id,
                        evaluation_name,
                        evaluation_result_id,
                        result_join_id,
                        name_join_id,
                        join_key_source,
                        original_evaluation_id,
                        evaluation_id_validation_status,
                        ingest_source_path,
                        model_id,
                        model_route_id,
                        eval_summary_id,
                        metric_summary_id,
                        benchmark_family_key,
                        eval_slice_id,
                        metric_key,
                        score,
                        is_correct,
                        raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        evaluation_id,
                        sample_id,
                        evaluation_name,
                        evaluation_result_id,
                        result_join_id,
                        name_join_id,
                        join_key_source,
                        original_evaluation_id or None,
                        evaluation_id_validation_status,
                        str(path),
                        model_id,
                        hierarchy.get("model_route_id"),
                        hierarchy.get("eval_summary_id"),
                        hierarchy.get("metric_summary_id"),
                        hierarchy.get("benchmark_family_key"),
                        hierarchy.get("slice_key"),
                        hierarchy.get("metric_key"),
                        score,
                        is_correct,
                        _json_dumps(payload),
                    ],
                )
                if evaluation_id_validation_status == "validated_from_detailed_results":
                    validated_rows += 1
                elif evaluation_id_validation_status == "repaired_from_detailed_results":
                    repaired_rows += 1
                elif evaluation_id_validation_status == "filled_from_detailed_results":
                    filled_rows += 1
                else:
                    unvalidated_rows += 1
                rows += 1

        return {
            "instance_rows_ingested": rows,
            "instance_rows_validated": validated_rows,
            "instance_rows_repaired": repaired_rows,
            "instance_rows_filled_from_file_link": filled_rows,
            "instance_rows_unvalidated": unvalidated_rows,
            "file_link_lookup_status": file_link_lookup_status,
            "expected_evaluation_id": expected_evaluation_id,
        }

    def stats(self) -> dict[str, int]:
        runs_count = int(self.conn.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0])
        metrics_count = int(
            self.conn.execute("SELECT COUNT(*) FROM evaluation_metrics").fetchone()[0]
        )
        models_count = int(
            self.conn.execute("SELECT COUNT(DISTINCT model_id) FROM evaluation_runs").fetchone()[0]
        )
        metric_kind_count = int(
            self.conn.execute(
                """
                SELECT COUNT(DISTINCT metric_kind)
                FROM evaluation_metrics
                WHERE metric_kind IS NOT NULL
                """
            ).fetchone()[0]
        )
        instance_count = int(
            self.conn.execute("SELECT COUNT(*) FROM instance_evaluations").fetchone()[0]
        )
        semantic_issue_count = int(
            self.conn.execute("SELECT COUNT(*) FROM semantic_quality_issues").fetchone()[0]
        )
        return {
            "evaluation_runs": runs_count,
            "evaluation_metrics": metrics_count,
            "models": models_count,
            "metric_kinds": metric_kind_count,
            "instance_rows": instance_count,
            "semantic_quality_issues": semantic_issue_count,
        }

    def top_model_metrics(
        self,
        metric_kind: str | None = None,
        metric_name: str | None = None,
        source_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            WITH agg AS (
                SELECT
                    r.model_id,
                    r.source_name,
                    COALESCE(m.metric_name, m.evaluation_name) AS metric_name,
                    COALESCE(m.metric_kind, 'unknown') AS metric_kind,
                    AVG(m.score) AS avg_score,
                    COUNT(*) AS observations,
                    COALESCE(BOOL_OR(m.lower_is_better), FALSE) AS lower_is_better
                FROM evaluation_metrics m
                JOIN evaluation_runs r ON r.evaluation_id = m.evaluation_id
                WHERE (? IS NULL OR m.metric_kind = ?)
                  AND (? IS NULL OR COALESCE(m.metric_name, m.evaluation_name) = ?)
                  AND (? IS NULL OR r.source_name = ?)
                GROUP BY 1, 2, 3, 4
            )
            SELECT *
            FROM agg
            ORDER BY CASE WHEN lower_is_better THEN avg_score ELSE -avg_score END ASC NULLS LAST
            LIMIT ?
            """,
            [metric_kind, metric_kind, metric_name, metric_name, source_name, source_name, limit],
        ).fetchall()

        return [
            {
                "model_id": row[0],
                "source_name": row[1],
                "metric_name": row[2],
                "metric_kind": row[3],
                "avg_score": float(row[4]),
                "observations": int(row[5]),
                "lower_is_better": bool(row[6]),
            }
            for row in rows
        ]

    def join_integrity(self) -> dict[str, int]:
        metrics_total = int(
            self.conn.execute("SELECT COUNT(*) FROM evaluation_metrics").fetchone()[0]
        )
        metrics_with_result_id = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM evaluation_metrics WHERE evaluation_result_id IS NOT NULL"
            ).fetchone()[0]
        )
        instance_total = int(
            self.conn.execute("SELECT COUNT(*) FROM instance_evaluations").fetchone()[0]
        )
        instance_with_result_id = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM instance_evaluations WHERE evaluation_result_id IS NOT NULL"
            ).fetchone()[0]
        )
        runs_with_detailed_results = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM evaluation_runs
                WHERE detailed_results_file_path IS NOT NULL
                """
            ).fetchone()[0]
        )
        instance_rows_validated = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM instance_evaluations
                WHERE evaluation_id_validation_status = 'validated_from_detailed_results'
                """
            ).fetchone()[0]
        )
        instance_rows_repaired = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM instance_evaluations
                WHERE evaluation_id_validation_status IN (
                    'repaired_from_detailed_results',
                    'filled_from_detailed_results'
                )
                """
            ).fetchone()[0]
        )
        instance_rows_unvalidated = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM instance_evaluations
                WHERE evaluation_id_validation_status IN (
                    'payload_only_unvalidated',
                    'payload_only_ambiguous_file_link',
                    'legacy_unvalidated'
                )
                """
            ).fetchone()[0]
        )

        matched_instances_exact = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT i.evaluation_id, i.sample_id, i.result_join_id
                    FROM instance_evaluations i
                    JOIN evaluation_metrics m
                      ON i.evaluation_id = m.evaluation_id
                     AND i.evaluation_result_id IS NOT NULL
                     AND m.evaluation_result_id = i.evaluation_result_id
                )
                """
            ).fetchone()[0]
        )

        matched_instances_name_fallback = int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT i.evaluation_id, i.sample_id, i.result_join_id
                    FROM instance_evaluations i
                    JOIN evaluation_metrics m
                      ON i.evaluation_id = m.evaluation_id
                     AND i.name_join_id = m.name_join_id
                     AND NOT (
                        i.evaluation_result_id IS NOT NULL
                        AND m.evaluation_result_id IS NOT NULL
                     )
                )
                """
            ).fetchone()[0]
        )

        return {
            "aggregate_metrics_total": metrics_total,
            "aggregate_metrics_with_evaluation_result_id": metrics_with_result_id,
            "aggregate_runs_with_detailed_results": runs_with_detailed_results,
            "instance_rows_total": instance_total,
            "instance_rows_with_evaluation_result_id": instance_with_result_id,
            "instance_rows_validated_from_detailed_results": instance_rows_validated,
            "instance_rows_repaired_via_detailed_results": instance_rows_repaired,
            "instance_rows_without_trusted_file_link_validation": instance_rows_unvalidated,
            "instance_rows_joined_by_evaluation_result_id": matched_instances_exact,
            "instance_rows_joined_by_name_fallback": matched_instances_name_fallback,
            "instance_rows_joined_to_aggregate": (
                matched_instances_exact + matched_instances_name_fallback
            ),
        }

    def orphan_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            WITH coverage AS (
                SELECT
                    r.evaluation_id,
                    r.source_name,
                    r.model_id,
                    r.detailed_results_file_path,
                    r.detailed_results_total_rows,
                    COUNT(i.sample_id) AS ingested_instance_rows,
                    COALESCE(
                        SUM(
                            CASE
                                WHEN i.evaluation_id_validation_status IN (
                                    'repaired_from_detailed_results',
                                    'filled_from_detailed_results'
                                ) THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    ) AS repaired_instance_rows
                FROM evaluation_runs r
                LEFT JOIN instance_evaluations i
                  ON i.evaluation_id = r.evaluation_id
                WHERE r.detailed_results_file_path IS NOT NULL
                GROUP BY 1, 2, 3, 4, 5
            ),
            issues AS (
                SELECT
                    evaluation_id,
                    source_name,
                    model_id,
                    detailed_results_file_path,
                    detailed_results_total_rows,
                    ingested_instance_rows,
                    repaired_instance_rows,
                    CASE
                        WHEN detailed_results_total_rows IS NULL THEN NULL
                        ELSE GREATEST(detailed_results_total_rows - ingested_instance_rows, 0)
                    END AS missing_instance_rows,
                    CASE
                        WHEN ingested_instance_rows = 0 THEN 'missing_ingested_instances'
                        WHEN repaired_instance_rows > 0 THEN 'repaired_instance_evaluation_id'
                        WHEN detailed_results_total_rows IS NOT NULL
                             AND ingested_instance_rows < detailed_results_total_rows
                            THEN 'partial_instance_ingest'
                        ELSE NULL
                    END AS issue_type
                FROM coverage
            )
            SELECT
                evaluation_id,
                source_name,
                model_id,
                detailed_results_file_path,
                detailed_results_total_rows,
                ingested_instance_rows,
                missing_instance_rows,
                repaired_instance_rows,
                issue_type
            FROM issues
            WHERE issue_type IS NOT NULL
            ORDER BY
                CASE issue_type
                    WHEN 'missing_ingested_instances' THEN 0
                    WHEN 'partial_instance_ingest' THEN 1
                    WHEN 'repaired_instance_evaluation_id' THEN 2
                    ELSE 3
                END,
                evaluation_id
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        return [
            {
                "evaluation_id": row[0],
                "source_name": row[1],
                "model_id": row[2],
                "detailed_results_file_path": row[3],
                "detailed_results_total_rows": (
                    int(row[4]) if row[4] is not None else None
                ),
                "ingested_instance_rows": int(row[5]),
                "missing_instance_rows": int(row[6]) if row[6] is not None else None,
                "repaired_instance_rows": int(row[7]),
                "issue_type": row[8],
            }
            for row in rows
        ]

    def identifier_issues(self, limit: int = 100) -> dict[str, Any]:
        summary = {
            "aggregate_metrics_missing_evaluation_result_id": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM evaluation_metrics
                    WHERE evaluation_result_id IS NULL
                    """
                ).fetchone()[0]
            ),
            "aggregate_metrics_missing_metric_id": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM evaluation_metrics
                    WHERE metric_id IS NULL
                    """
                ).fetchone()[0]
            ),
            "aggregate_metrics_missing_metric_kind": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM evaluation_metrics
                    WHERE metric_kind IS NULL
                    """
                ).fetchone()[0]
            ),
            "instance_rows_missing_evaluation_result_id": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM instance_evaluations
                    WHERE evaluation_result_id IS NULL
                    """
                ).fetchone()[0]
            ),
            "instance_rows_repaired_evaluation_id": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM instance_evaluations
                    WHERE evaluation_id_validation_status = 'repaired_from_detailed_results'
                    """
                ).fetchone()[0]
            ),
            "instance_rows_filled_evaluation_id_from_file_link": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM instance_evaluations
                    WHERE evaluation_id_validation_status = 'filled_from_detailed_results'
                    """
                ).fetchone()[0]
            ),
            "instance_rows_without_trusted_file_link_validation": int(
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM instance_evaluations
                    WHERE evaluation_id_validation_status IN (
                        'payload_only_unvalidated',
                        'payload_only_ambiguous_file_link',
                        'legacy_unvalidated'
                    )
                    """
                ).fetchone()[0]
            ),
            "semantic_quality_issues": int(
                self.conn.execute("SELECT COUNT(*) FROM semantic_quality_issues").fetchone()[0]
            ),
        }

        repaired_rows = self.conn.execute(
            """
            SELECT
                ingest_source_path,
                sample_id,
                evaluation_name,
                model_id,
                original_evaluation_id,
                evaluation_id,
                evaluation_id_validation_status
            FROM instance_evaluations
            WHERE evaluation_id_validation_status IN (
                'repaired_from_detailed_results',
                'filled_from_detailed_results'
            )
            ORDER BY ingest_source_path, sample_id
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        unvalidated_rows = self.conn.execute(
            """
            SELECT
                ingest_source_path,
                sample_id,
                evaluation_name,
                model_id,
                original_evaluation_id,
                evaluation_id,
                evaluation_id_validation_status
            FROM instance_evaluations
            WHERE evaluation_id_validation_status IN (
                'payload_only_unvalidated',
                'payload_only_ambiguous_file_link',
                'legacy_unvalidated'
            )
            ORDER BY
                CASE evaluation_id_validation_status
                    WHEN 'payload_only_ambiguous_file_link' THEN 0
                    WHEN 'payload_only_unvalidated' THEN 1
                    ELSE 2
                END,
                ingest_source_path,
                sample_id
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        aggregate_metric_rows = self.conn.execute(
            """
            SELECT
                evaluation_id,
                metric_index,
                evaluation_name,
                metric_id,
                metric_name,
                metric_kind
            FROM evaluation_metrics
            WHERE evaluation_result_id IS NULL
               OR metric_id IS NULL
               OR metric_kind IS NULL
            ORDER BY evaluation_id, metric_index
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        return {
            "summary": summary,
            "repaired_instance_examples": [
                {
                    "ingest_source_path": row[0],
                    "sample_id": row[1],
                    "evaluation_name": row[2],
                    "model_id": row[3],
                    "original_evaluation_id": row[4],
                    "stored_evaluation_id": row[5],
                    "evaluation_id_validation_status": row[6],
                }
                for row in repaired_rows
            ],
            "unvalidated_instance_examples": [
                {
                    "ingest_source_path": row[0],
                    "sample_id": row[1],
                    "evaluation_name": row[2],
                    "model_id": row[3],
                    "original_evaluation_id": row[4],
                    "stored_evaluation_id": row[5],
                    "evaluation_id_validation_status": row[6],
                }
                for row in unvalidated_rows
            ],
            "aggregate_metric_examples": [
                {
                    "evaluation_id": row[0],
                    "metric_index": int(row[1]),
                    "evaluation_name": row[2],
                    "metric_id": row[3],
                    "metric_name": row[4],
                    "metric_kind": row[5],
                }
                for row in aggregate_metric_rows
            ],
        }

    def sources(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            WITH runs AS (
                SELECT
                    COALESCE(source_name, '__missing_source__') AS source_name,
                    COUNT(*) AS run_count,
                    COUNT(DISTINCT model_id) AS model_count,
                    SUM(
                        CASE
                            WHEN source_type = 'documentation' THEN 1
                            ELSE 0
                        END
                    ) AS documentation_runs,
                    SUM(
                        CASE
                            WHEN source_type = 'evaluation_run' THEN 1
                            ELSE 0
                        END
                    ) AS evaluation_run_runs
                FROM evaluation_runs
                GROUP BY 1
            ),
            metrics AS (
                SELECT
                    COALESCE(r.source_name, '__missing_source__') AS source_name,
                    COUNT(*) AS metric_rows,
                    COUNT(DISTINCT m.evaluation_id) AS evaluations_with_metrics,
                    COUNT(
                        DISTINCT COALESCE(
                            m.metric_id,
                            m.metric_name,
                            m.evaluation_name
                        )
                    ) AS metric_identity_count,
                    AVG(m.score) AS average_metric_score
                FROM evaluation_metrics m
                JOIN evaluation_runs r ON r.evaluation_id = m.evaluation_id
                GROUP BY 1
            ),
            instances AS (
                SELECT
                    COALESCE(r.source_name, '__missing_source__') AS source_name,
                    COUNT(*) AS instance_rows,
                    COUNT(DISTINCT i.evaluation_id) AS instance_evaluation_count
                FROM instance_evaluations i
                JOIN evaluation_runs r ON r.evaluation_id = i.evaluation_id
                GROUP BY 1
            )
            SELECT
                runs.source_name,
                runs.run_count,
                runs.model_count,
                runs.documentation_runs,
                runs.evaluation_run_runs,
                COALESCE(metrics.metric_rows, 0) AS metric_rows,
                COALESCE(metrics.evaluations_with_metrics, 0) AS evaluations_with_metrics,
                COALESCE(metrics.metric_identity_count, 0) AS metric_identity_count,
                metrics.average_metric_score,
                COALESCE(instances.instance_rows, 0) AS instance_rows,
                COALESCE(instances.instance_evaluation_count, 0) AS instance_evaluation_count
            FROM runs
            LEFT JOIN metrics USING (source_name)
            LEFT JOIN instances USING (source_name)
            ORDER BY run_count DESC, metric_rows DESC, source_name
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        return [
            {
                "source_name": row[0],
                "run_count": int(row[1]),
                "model_count": int(row[2]),
                "documentation_runs": int(row[3]),
                "evaluation_run_runs": int(row[4]),
                "metric_rows": int(row[5]),
                "evaluations_with_metrics": int(row[6]),
                "metric_identity_count": int(row[7]),
                "average_metric_score": float(row[8]) if row[8] is not None else None,
                "instance_rows": int(row[9]),
                "instance_evaluation_count": int(row[10]),
            }
            for row in rows
        ]

    def models(
        self,
        source_name: str | None = None,
        benchmark_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        run_rows = self.conn.execute(
            """
            SELECT
                model_id,
                COUNT(*) AS run_count,
                COUNT(
                    DISTINCT COALESCE(source_name, '__missing_source__')
                ) AS source_count,
                string_agg(
                    DISTINCT COALESCE(source_name, '__missing_source__'),
                    '||'
                ) AS source_names,
                MAX(retrieved_timestamp) AS latest_retrieved_timestamp
            FROM evaluation_runs
            WHERE (? IS NULL OR COALESCE(source_name, '__missing_source__') = ?)
            GROUP BY 1
            """,
            [source_name, source_name],
        ).fetchall()

        metric_rows = self.conn.execute(
            """
            SELECT
                r.model_id,
                COUNT(*) AS metric_rows,
                COUNT(DISTINCT m.evaluation_id) AS evaluation_count,
                COUNT(
                    DISTINCT COALESCE(
                        m.metric_id,
                        m.metric_name,
                        m.evaluation_name
                    )
                ) AS metric_identity_count,
                COUNT(
                    DISTINCT COALESCE(
                        m.source_dataset_name,
                        m.evaluation_name
                    )
                ) AS benchmark_count,
                AVG(m.score) AS average_score,
                SUM(
                    CASE
                        WHEN m.evaluation_result_id IS NOT NULL THEN 1
                        ELSE 0
                    END
                ) AS metric_rows_with_result_id
            FROM evaluation_metrics m
            JOIN evaluation_runs r ON r.evaluation_id = m.evaluation_id
            WHERE (? IS NULL OR COALESCE(r.source_name, '__missing_source__') = ?)
              AND (
                ? IS NULL
                OR LOWER(COALESCE(m.source_dataset_name, m.evaluation_name)) = LOWER(?)
              )
            GROUP BY 1
            """,
            [source_name, source_name, benchmark_name, benchmark_name],
        ).fetchall()

        instance_rows = self.conn.execute(
            """
            SELECT
                i.model_id,
                COUNT(*) AS instance_rows,
                COUNT(DISTINCT i.evaluation_id) AS instance_evaluation_count
            FROM instance_evaluations i
            JOIN evaluation_runs r ON r.evaluation_id = i.evaluation_id
            WHERE (? IS NULL OR COALESCE(r.source_name, '__missing_source__') = ?)
            GROUP BY 1
            """,
            [source_name, source_name],
        ).fetchall()

        models_by_id: dict[str, dict[str, Any]] = {}
        for row in run_rows:
            source_names = row[3].split("||") if row[3] else []
            models_by_id[str(row[0])] = {
                "model_id": str(row[0]),
                "run_count": int(row[1]),
                "source_count": int(row[2]),
                "source_names": sorted(source_names),
                "latest_retrieved_timestamp": row[4],
                "evaluation_count": 0,
                "metric_rows": 0,
                "metric_identity_count": 0,
                "benchmark_count": 0,
                "average_score": None,
                "metric_rows_with_result_id": 0,
                "instance_rows": 0,
                "instance_evaluation_count": 0,
            }

        for row in metric_rows:
            model_id = str(row[0])
            entry = models_by_id.setdefault(
                model_id,
                {
                    "model_id": model_id,
                    "run_count": 0,
                    "source_count": 0,
                    "source_names": [],
                    "latest_retrieved_timestamp": None,
                    "evaluation_count": 0,
                    "metric_rows": 0,
                    "metric_identity_count": 0,
                    "benchmark_count": 0,
                    "average_score": None,
                    "metric_rows_with_result_id": 0,
                    "instance_rows": 0,
                    "instance_evaluation_count": 0,
                },
            )
            entry["metric_rows"] = int(row[1])
            entry["evaluation_count"] = int(row[2])
            entry["metric_identity_count"] = int(row[3])
            entry["benchmark_count"] = int(row[4])
            entry["average_score"] = float(row[5]) if row[5] is not None else None
            entry["metric_rows_with_result_id"] = int(row[6])

        for row in instance_rows:
            model_id = str(row[0])
            entry = models_by_id.setdefault(
                model_id,
                {
                    "model_id": model_id,
                    "run_count": 0,
                    "source_count": 0,
                    "source_names": [],
                    "latest_retrieved_timestamp": None,
                    "evaluation_count": 0,
                    "metric_rows": 0,
                    "metric_identity_count": 0,
                    "benchmark_count": 0,
                    "average_score": None,
                    "metric_rows_with_result_id": 0,
                    "instance_rows": 0,
                    "instance_evaluation_count": 0,
                },
            )
            entry["instance_rows"] = int(row[1])
            entry["instance_evaluation_count"] = int(row[2])

        rows_out = list(models_by_id.values())
        if benchmark_name is not None:
            rows_out = [row for row in rows_out if row["metric_rows"] > 0]
        rows_out.sort(
            key=lambda row: (
                -row["metric_rows"],
                -row["run_count"],
                row["model_id"],
            )
        )
        return rows_out[offset : offset + limit]

    def benchmarks(
        self,
        source_name: str | None = None,
        benchmark_name: str | None = None,
        metric_kind: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                COALESCE(r.source_name, '__missing_source__') AS source_name,
                COALESCE(m.eval_summary_id, m.source_dataset_name, m.evaluation_name) AS eval_summary_id,
                m.metric_summary_id,
                m.benchmark_family_key,
                m.benchmark_leaf_key,
                m.slice_key,
                m.metric_key,
                COALESCE(m.source_dataset_name, m.evaluation_name) AS benchmark_name,
                COALESCE(m.metric_id, m.metric_name, m.evaluation_name) AS metric_identity,
                m.metric_id,
                COALESCE(m.metric_name, m.evaluation_name) AS metric_name,
                COALESCE(m.metric_kind, 'unknown') AS metric_kind,
                m.metric_unit,
                COALESCE(BOOL_OR(m.lower_is_better), FALSE) AS lower_is_better,
                COUNT(*) AS metric_rows,
                COUNT(DISTINCT r.evaluation_id) AS evaluation_count,
                COUNT(DISTINCT r.model_id) AS model_count,
                AVG(m.score) AS average_score,
                MIN(m.score) AS min_score,
                MAX(m.score) AS max_score,
                SUM(
                    CASE
                        WHEN m.evaluation_result_id IS NOT NULL THEN 1
                        ELSE 0
                    END
                ) AS metric_rows_with_result_id,
                COUNT(
                    DISTINCT CASE
                        WHEN i.sample_id IS NOT NULL THEN r.model_id
                        ELSE NULL
                    END
                ) AS models_with_instances,
                COUNT(i.sample_id) AS joined_instance_rows
            FROM evaluation_metrics m
            JOIN evaluation_runs r ON r.evaluation_id = m.evaluation_id
            LEFT JOIN instance_evaluations i
              ON i.evaluation_id = m.evaluation_id
             AND (
                (
                    i.evaluation_result_id IS NOT NULL
                    AND m.evaluation_result_id IS NOT NULL
                    AND i.evaluation_result_id = m.evaluation_result_id
                )
                OR (
                    (
                        i.evaluation_result_id IS NULL
                        OR m.evaluation_result_id IS NULL
                    )
                    AND i.name_join_id = m.name_join_id
                )
             )
            WHERE (? IS NULL OR COALESCE(r.source_name, '__missing_source__') = ?)
              AND (
                ? IS NULL
                OR LOWER(COALESCE(m.source_dataset_name, m.evaluation_name)) = LOWER(?)
              )
              AND (? IS NULL OR COALESCE(m.metric_kind, 'unknown') = ?)
            GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
            ORDER BY model_count DESC, metric_rows DESC, source_name, benchmark_name
            LIMIT ?
            """,
            [
                source_name,
                source_name,
                benchmark_name,
                benchmark_name,
                metric_kind,
                metric_kind,
                limit,
            ],
        ).fetchall()

        return [
            {
                "source_name": row[0],
                "eval_summary_id": row[1],
                "metric_summary_id": row[2],
                "benchmark_family_key": row[3],
                "benchmark_leaf_key": row[4],
                "slice_key": row[5],
                "metric_key": row[6],
                "benchmark_name": row[7],
                "metric_identity": row[8],
                "metric_id": row[9],
                "metric_name": row[10],
                "metric_kind": row[11],
                "metric_unit": row[12],
                "lower_is_better": bool(row[13]),
                "metric_rows": int(row[14]),
                "evaluation_count": int(row[15]),
                "model_count": int(row[16]),
                "average_score": float(row[17]) if row[17] is not None else None,
                "min_score": float(row[18]) if row[18] is not None else None,
                "max_score": float(row[19]) if row[19] is not None else None,
                "metric_rows_with_result_id": int(row[20]),
                "models_with_instances": int(row[21]),
                "joined_instance_rows": int(row[22]),
            }
            for row in rows
        ]

    def benchmark_model_rankings(
        self,
        benchmark_name: str,
        source_name: str | None = None,
        metric_identity: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                r.model_id,
                COALESCE(m.metric_id, m.metric_name, m.evaluation_name) AS metric_identity,
                m.metric_id,
                COALESCE(m.metric_name, m.evaluation_name) AS metric_name,
                COALESCE(m.metric_kind, 'unknown') AS metric_kind,
                COALESCE(BOOL_OR(m.lower_is_better), FALSE) AS lower_is_better,
                AVG(m.score) AS average_score,
                COUNT(*) AS metric_rows,
                COUNT(DISTINCT r.evaluation_id) AS evaluation_count,
                COUNT(i.sample_id) AS joined_instance_rows,
                AVG(i.score) AS average_instance_score
            FROM evaluation_metrics m
            JOIN evaluation_runs r ON r.evaluation_id = m.evaluation_id
            LEFT JOIN instance_evaluations i
              ON i.evaluation_id = m.evaluation_id
             AND i.model_id = r.model_id
             AND (
                (
                    i.evaluation_result_id IS NOT NULL
                    AND m.evaluation_result_id IS NOT NULL
                    AND i.evaluation_result_id = m.evaluation_result_id
                )
                OR (
                    (
                        i.evaluation_result_id IS NULL
                        OR m.evaluation_result_id IS NULL
                    )
                    AND i.name_join_id = m.name_join_id
                )
             )
            WHERE LOWER(COALESCE(m.source_dataset_name, m.evaluation_name)) = LOWER(?)
              AND (? IS NULL OR COALESCE(r.source_name, '__missing_source__') = ?)
              AND (
                ? IS NULL
                OR LOWER(COALESCE(m.metric_id, m.metric_name, m.evaluation_name)) = LOWER(?)
              )
            GROUP BY 1, 2, 3, 4, 5
            """,
            [
                benchmark_name,
                source_name,
                source_name,
                metric_identity,
                metric_identity,
            ],
        ).fetchall()

        grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped_rows[str(row[1])].append(
                {
                    "model_id": str(row[0]),
                    "metric_identity": str(row[1]),
                    "metric_id": row[2],
                    "metric_name": row[3],
                    "metric_kind": row[4],
                    "lower_is_better": bool(row[5]),
                    "average_score": float(row[6]) if row[6] is not None else None,
                    "metric_rows": int(row[7]),
                    "evaluation_count": int(row[8]),
                    "joined_instance_rows": int(row[9]),
                    "average_instance_score": (
                        float(row[10]) if row[10] is not None else None
                    ),
                    "rank": 0,
                    "total": 0,
                }
            )

        ranked_rows: list[dict[str, Any]] = []
        for metric_rows in grouped_rows.values():
            metric_rows.sort(
                key=lambda row: (
                    (
                        row["average_score"]
                        if row["lower_is_better"]
                        else -row["average_score"]
                    ),
                    -row["metric_rows"],
                    row["model_id"],
                )
            )
            total = len(metric_rows)
            for rank, row in enumerate(metric_rows, start=1):
                row["rank"] = rank
                row["total"] = total
                ranked_rows.append(row)

        ranked_rows.sort(key=lambda row: (row["metric_identity"], row["rank"], row["model_id"]))
        return ranked_rows[:limit]

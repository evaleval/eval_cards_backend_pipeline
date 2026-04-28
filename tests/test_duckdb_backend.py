import json

from shared.duckdb_backend import DuckDBBackend


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _aggregate_row(*, evaluation_id: str, model_id: str, detailed_results_file_path: str) -> dict:
    return {
        "evaluation_id": evaluation_id,
        "schema_version": "0.2.2",
        "retrieved_timestamp": "2026-04-09T12:00:00Z",
        "source_metadata": {
            "source_name": "Test Source",
            "source_type": "documentation",
        },
        "model_info": {
            "id": model_id,
            "name": model_id,
        },
        "eval_library": {
            "name": "pytest",
            "version": "0.1",
        },
        "evaluation_results": [
            {
                "evaluation_name": "theory_of_mind",
                "metric_config": {
                    "metric_name": "Theory of Mind",
                    "metric_kind": "accuracy",
                },
                "score_details": {
                    "score": 0.5,
                },
            }
        ],
        "detailed_evaluation_results": {
            "format": "jsonl",
            "file_path": detailed_results_file_path,
            "hash_algorithm": "sha256",
            "checksum": "unused-in-test",
            "total_rows": 1,
        },
    }


def _instance_row(
    *,
    evaluation_id: str,
    evaluation_name: str = "theory_of_mind",
    sample_id: str = "1",
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
) -> dict:
    return {
        "evaluation_id": evaluation_id,
        "sample_id": sample_id,
        "evaluation_name": evaluation_name,
        "evaluation_result_id": None,
        "model_id": model_id,
        "evaluation": {
            "score": 1.0,
            "is_correct": True,
        },
    }


def _aggregate_row_for_dashboard(
    *,
    evaluation_id: str,
    model_id: str,
    source_name: str,
    benchmark_name: str,
    evaluation_name: str,
    evaluation_result_id: str,
    metric_id: str,
    metric_name: str,
    metric_kind: str,
    metric_unit: str,
    lower_is_better: bool,
    score: float,
) -> dict:
    return {
        "evaluation_id": evaluation_id,
        "schema_version": "0.2.2",
        "retrieved_timestamp": "2026-04-09T12:00:00Z",
        "source_metadata": {
            "source_name": source_name,
            "source_type": "documentation",
        },
        "model_info": {
            "id": model_id,
            "name": model_id,
        },
        "eval_library": {
            "name": "pytest",
            "version": "0.1",
        },
        "evaluation_results": [
            {
                "evaluation_name": evaluation_name,
                "evaluation_result_id": evaluation_result_id,
                "source_data": {
                    "dataset_name": benchmark_name,
                    "source_type": "url",
                    "url": ["https://example.com/dataset"],
                },
                "metric_config": {
                    "metric_id": metric_id,
                    "metric_name": metric_name,
                    "metric_kind": metric_kind,
                    "metric_unit": metric_unit,
                    "lower_is_better": lower_is_better,
                    "score_type": "continuous",
                    "min_score": 0.0,
                    "max_score": 1.0 if not lower_is_better else 1000.0,
                },
                "score_details": {
                    "score": score,
                },
            }
        ],
    }


def test_ingest_repairs_malformed_instance_evaluation_id(tmp_path) -> None:
    db_path = tmp_path / "backend.duckdb"
    backend = DuckDBBackend(str(db_path))

    aggregate_evaluation_id = "theory_of_mind/hf_Qwen_Qwen2.5-3B-Instruct/1772541652.0"
    malformed_instance_evaluation_id = "30ed1a75-5bfd-4405-abce-b0fd5e0165ba_samples"
    instance_filename = "30ed1a75-5bfd-4405-abce-b0fd5e0165ba_samples.jsonl"

    aggregate_jsonl = tmp_path / "aggregate.jsonl"
    instance_jsonl = tmp_path / instance_filename
    _write_jsonl(
        aggregate_jsonl,
        [
            _aggregate_row(
                evaluation_id=aggregate_evaluation_id,
                model_id="Qwen/Qwen2.5-3B-Instruct",
                detailed_results_file_path=f"data/theory_of_mind/Qwen/Qwen2.5-3B-Instruct/{instance_filename}",
            )
        ],
    )
    _write_jsonl(
        instance_jsonl,
        [_instance_row(evaluation_id=malformed_instance_evaluation_id)],
    )

    backend.ingest_aggregate_jsonl(aggregate_jsonl)
    ingest_stats = backend.ingest_instance_jsonl(instance_jsonl)

    assert ingest_stats["instance_rows_ingested"] == 1
    assert ingest_stats["instance_rows_repaired"] == 1
    assert ingest_stats["expected_evaluation_id"] == aggregate_evaluation_id

    join_integrity = backend.join_integrity()
    assert join_integrity["instance_rows_repaired_via_detailed_results"] == 1
    assert join_integrity["instance_rows_joined_by_name_fallback"] == 1
    assert join_integrity["instance_rows_joined_to_aggregate"] == 1

    orphan_runs = backend.orphan_runs(limit=10)
    assert orphan_runs == [
        {
            "evaluation_id": aggregate_evaluation_id,
            "source_name": "Test Source",
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "detailed_results_file_path": (
                "data/theory_of_mind/Qwen/Qwen2.5-3B-Instruct/"
                "30ed1a75-5bfd-4405-abce-b0fd5e0165ba_samples.jsonl"
            ),
            "detailed_results_total_rows": 1,
            "ingested_instance_rows": 1,
            "missing_instance_rows": 0,
            "repaired_instance_rows": 1,
            "issue_type": "repaired_instance_evaluation_id",
        }
    ]

    identifier_issues = backend.identifier_issues(limit=10)
    assert identifier_issues["summary"]["instance_rows_repaired_evaluation_id"] == 1
    assert identifier_issues["repaired_instance_examples"] == [
        {
            "ingest_source_path": str(instance_jsonl),
            "sample_id": "1",
            "evaluation_name": "theory_of_mind",
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "original_evaluation_id": malformed_instance_evaluation_id,
            "stored_evaluation_id": aggregate_evaluation_id,
            "evaluation_id_validation_status": "repaired_from_detailed_results",
        }
    ]


def test_validated_instance_file_does_not_raise_orphan_issue(tmp_path) -> None:
    db_path = tmp_path / "backend.duckdb"
    backend = DuckDBBackend(str(db_path))

    aggregate_evaluation_id = (
        "global-mmlu-lite/alibaba_qwen3-235b-a22b-instruct-2507/"
        "1770822797.839372"
    )
    instance_filename = "879c853a-49a7-476d-82fd-4b280c3dbfac.jsonl"

    aggregate_jsonl = tmp_path / "aggregate.jsonl"
    instance_jsonl = tmp_path / instance_filename
    _write_jsonl(
        aggregate_jsonl,
        [
            _aggregate_row(
                evaluation_id=aggregate_evaluation_id,
                model_id="alibaba/qwen3-235b-a22b-instruct-2507",
                detailed_results_file_path=(
                    "baseline/data/global-mmlu-lite/alibaba/"
                    "qwen3-235b-a22b-instruct-2507/879c853a-49a7-476d-82fd-4b280c3dbfac.jsonl"
                ),
            )
        ],
    )
    _write_jsonl(
        instance_jsonl,
        [
            _instance_row(
                evaluation_id=aggregate_evaluation_id,
                evaluation_name="Global MMLU Lite",
                model_id="alibaba/qwen3-235b-a22b-instruct-2507",
            )
        ],
    )

    backend.ingest_aggregate_jsonl(aggregate_jsonl)
    ingest_stats = backend.ingest_instance_jsonl(instance_jsonl)

    assert ingest_stats["instance_rows_validated"] == 1
    assert ingest_stats["instance_rows_repaired"] == 0
    assert backend.orphan_runs(limit=10) == []


def test_dashboard_join_queries_cover_source_model_benchmark_views(tmp_path) -> None:
    db_path = tmp_path / "backend.duckdb"
    backend = DuckDBBackend(str(db_path))

    aggregate_jsonl = tmp_path / "aggregate.jsonl"
    instance_jsonl = tmp_path / "instance.jsonl"

    _write_jsonl(
        aggregate_jsonl,
        [
            _aggregate_row_for_dashboard(
                evaluation_id="bench_alpha/model_a/1",
                model_id="model/A",
                source_name="Source One",
                benchmark_name="bench_alpha",
                evaluation_name="accuracy",
                evaluation_result_id="bench_alpha/model_a/1#accuracy#overall",
                metric_id="accuracy",
                metric_name="Accuracy",
                metric_kind="accuracy",
                metric_unit="proportion",
                lower_is_better=False,
                score=0.91,
            ),
            _aggregate_row_for_dashboard(
                evaluation_id="bench_alpha/model_b/1",
                model_id="model/B",
                source_name="Source One",
                benchmark_name="bench_alpha",
                evaluation_name="accuracy",
                evaluation_result_id="bench_alpha/model_b/1#accuracy#overall",
                metric_id="accuracy",
                metric_name="Accuracy",
                metric_kind="accuracy",
                metric_unit="proportion",
                lower_is_better=False,
                score=0.72,
            ),
            _aggregate_row_for_dashboard(
                evaluation_id="bench_beta/model_a/1",
                model_id="model/A",
                source_name="Source Two",
                benchmark_name="bench_beta",
                evaluation_name="latency",
                evaluation_result_id="bench_beta/model_a/1#latency_ms#overall",
                metric_id="latency_ms",
                metric_name="Latency (ms)",
                metric_kind="latency",
                metric_unit="ms",
                lower_is_better=True,
                score=120.0,
            ),
            _aggregate_row_for_dashboard(
                evaluation_id="bench_beta/model_b/1",
                model_id="model/B",
                source_name="Source Two",
                benchmark_name="bench_beta",
                evaluation_name="latency",
                evaluation_result_id="bench_beta/model_b/1#latency_ms#overall",
                metric_id="latency_ms",
                metric_name="Latency (ms)",
                metric_kind="latency",
                metric_unit="ms",
                lower_is_better=True,
                score=200.0,
            ),
        ],
    )
    _write_jsonl(
        instance_jsonl,
        [
            _instance_row(
                evaluation_id="bench_alpha/model_a/1",
                evaluation_name="accuracy",
                sample_id="a-1",
                model_id="model/A",
            ),
            _instance_row(
                evaluation_id="bench_alpha/model_b/1",
                evaluation_name="accuracy",
                sample_id="b-1",
                model_id="model/B",
            ),
        ],
    )

    backend.ingest_aggregate_jsonl(aggregate_jsonl)
    backend.ingest_instance_jsonl(instance_jsonl)

    source_rows = backend.sources(limit=10)
    assert [row["source_name"] for row in source_rows] == ["Source One", "Source Two"]
    assert source_rows[0]["model_count"] == 2
    assert source_rows[0]["metric_rows"] == 2

    model_rows = backend.models(
        source_name="Source One",
        benchmark_name="bench_alpha",
        limit=10,
        offset=0,
    )
    assert [row["model_id"] for row in model_rows] == ["model/A", "model/B"]
    assert [row["metric_rows"] for row in model_rows] == [1, 1]
    assert [row["benchmark_count"] for row in model_rows] == [1, 1]

    benchmark_rows = backend.benchmarks(
        source_name="Source One",
        benchmark_name="bench_alpha",
        limit=10,
    )
    assert len(benchmark_rows) == 1
    assert benchmark_rows[0]["metric_identity"] == "accuracy"
    assert benchmark_rows[0]["model_count"] == 2
    assert benchmark_rows[0]["joined_instance_rows"] == 2

    accuracy_rankings = backend.benchmark_model_rankings(
        benchmark_name="bench_alpha",
        source_name="Source One",
        metric_identity="accuracy",
        limit=10,
    )
    assert [row["model_id"] for row in accuracy_rankings] == ["model/A", "model/B"]
    assert [row["rank"] for row in accuracy_rankings] == [1, 2]
    assert [row["total"] for row in accuracy_rankings] == [2, 2]

    latency_rankings = backend.benchmark_model_rankings(
        benchmark_name="bench_beta",
        source_name="Source Two",
        metric_identity="latency_ms",
        limit=10,
    )
    assert [row["model_id"] for row in latency_rankings] == ["model/A", "model/B"]
    assert [row["rank"] for row in latency_rankings] == [1, 2]

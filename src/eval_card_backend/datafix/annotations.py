"""What a source declares about whether one of its results sampled text.

Two sources in the corpus state it, each about results the pipeline
already reads, each pinned to a revision:

- `lm_eval_output_type`: the vendored Open LLM v2 harness-config table
  records the lm-evaluation-harness `output_type` the leaderboard's own
  published dumps state, per model and per benchmark. A declaration
  exists for exactly the (model, benchmark) pairs that table verified.
- `rewardbench_model_type`: reward-bench records carry the
  leaderboard's per-model type in
  `model_info.additional_details.model_type`. It is a property of the
  model, so it covers every result in the record that states it.

Nothing is inferred: not from benchmark names, task formats, metric
names or token counts. Nothing is written into the record either. The
declarations are collected here while Stage A reads the records and
handed to the pipeline as a side table keyed by the physical result,
`(source_record_path, result_idx)`. The loader's record path is unique
per file, which `(evaluation_id, result_idx)` is not: Open LLM v2 ships
float16/bfloat16 record pairs that share an evaluation_id.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterator

import pyarrow as pa

from eval_card_backend.datafix import hfopenllm_v2
from eval_card_backend.datafix.registry import fixes_for

REWARDBENCH_SOURCE_CONFIG = "reward-bench"

# One row per declared physical result. The identity columns beyond the
# key are redundant by design: Stage D asserts they match the fact the
# row joins to, so a table built against one snapshot can never be
# silently joined onto another.
ANNOTATION_SCHEMA = pa.schema([
    pa.field("source_record_path", pa.string(), nullable=False),
    pa.field("result_idx", pa.int64(), nullable=False),
    pa.field("evaluation_id", pa.string()),
    pa.field("evaluation_result_id", pa.string()),
    pa.field("evaluation_name", pa.string()),
    pa.field("model_id", pa.string()),
    pa.field("declaration_source", pa.string(), nullable=False),
    pa.field("declared_value", pa.string(), nullable=False),
    pa.field("source_fix_id", pa.string()),
    pa.field("source_revision", pa.string()),
    pa.field("table_sha256", pa.string()),
])


class _LmEvalOutputType:
    """Declarations from the vendored Open LLM v2 harness-config table.

    The table is the same artifact the overlay's fill reads, so the
    declaration and the values filled from it come from one evidence
    source at one revision. `config_for` returns nothing for a (model,
    benchmark) pair the table did not verify, and those results stay
    undeclared.
    """

    declaration_source = "lm_eval_output_type"
    source_config = hfopenllm_v2.SOURCE_CONFIG

    def __init__(self, fix: Any) -> None:
        self._fix = fix

    @property
    def source_fix_id(self) -> str:
        return self._fix.fix_id

    def source_revision(self, eee_revision: str | None) -> str:
        return self._fix.source_config_revision

    @property
    def table_sha256(self) -> str | None:
        return self._fix.table_sha256

    def declare(self, record: dict[str, Any]) -> Iterator[tuple[int, str]]:
        model_id = (record.get("model_info") or {}).get("id")
        if not model_id:
            return
        for idx, result in enumerate(record.get("evaluation_results") or []):
            if not isinstance(result, dict):
                continue
            declared = self._record_value(result)
            if declared is not None:
                yield idx, declared
                continue
            precision = ((record.get("model_info") or {}).get("additional_details") or {}).get("precision")
            config = self._fix.config_for(model_id, result.get("evaluation_name"), precision)
            if config is None:
                continue
            output_type = config.get("output_type")
            if isinstance(output_type, str) and output_type:
                yield idx, output_type

    @staticmethod
    def _record_value(result: dict[str, Any]) -> str | None:
        value = ((result.get("generation_config") or {}).get("additional_details") or {}).get("output_type")
        return value if isinstance(value, str) and value else None

    def provenance(self, result: dict[str, Any], eee_revision: str | None) -> dict[str, Any]:
        if self._record_value(result) is not None:
            return {"source_fix_id": None, "source_revision": eee_revision, "table_sha256": None}
        return {
            "source_fix_id": self.source_fix_id,
            "source_revision": self.source_revision(eee_revision),
            "table_sha256": self.table_sha256,
        }


class _RewardBenchModelType:
    """Declarations carried inside the reward-bench records themselves.

    There is no table: the record states the type, so the evidence is
    the EEE revision the record was read at.
    """

    declaration_source = "rewardbench_model_type"
    source_config = REWARDBENCH_SOURCE_CONFIG
    source_fix_id = None
    table_sha256 = None

    def source_revision(self, eee_revision: str | None) -> str | None:
        return eee_revision

    def provenance(self, result: dict[str, Any], eee_revision: str | None) -> dict[str, Any]:
        return {"source_fix_id": None, "source_revision": eee_revision, "table_sha256": None}

    def declare(self, record: dict[str, Any]) -> Iterator[tuple[int, str]]:
        details = (record.get("model_info") or {}).get("additional_details") or {}
        value = details.get("model_type")
        if not isinstance(value, str) or not value.strip():
            return
        for idx, result in enumerate(record.get("evaluation_results") or []):
            if isinstance(result, dict):
                yield idx, value


def _declarers() -> dict[str, list[Any]]:
    """Declaration sources by the config whose records they speak about.

    The harness-config declarer is built from the registered fix so the
    vendored table is read once per process, as the fill already does.
    """
    fixes = [fix for fix in fixes_for(hfopenllm_v2.SOURCE_CONFIG)
             if fix.fix_id == hfopenllm_v2.FIX_ID]
    out: dict[str, list[Any]] = {
        REWARDBENCH_SOURCE_CONFIG: [_RewardBenchModelType()],
    }
    if fixes:
        out[hfopenllm_v2.SOURCE_CONFIG] = [_LmEvalOutputType(fixes[0])]
    return out


def annotating_configs() -> frozenset[str]:
    """Source configs some declaration source speaks about. Records of any
    other config are never opened for a declaration."""
    return frozenset({hfopenllm_v2.SOURCE_CONFIG, REWARDBENCH_SOURCE_CONFIG})


class AnnotationLog:
    """The declarations one Stage A pass collected.

    Process-level, like the overlay's own summary: reset before a run,
    read once the records are loaded.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._declarers: dict[str, list[Any]] | None = None

    def reset(self) -> None:
        self._rows.clear()
        self._declarers = None

    def record(
        self,
        record_path: str,
        source_config: str,
        record: dict[str, Any],
        eee_revision: str | None,
    ) -> None:
        if self._declarers is None:
            self._declarers = _declarers()
        evaluation_id = record.get("evaluation_id")
        model_id = (record.get("model_info") or {}).get("id")
        results = record.get("evaluation_results") or []
        for declarer in self._declarers.get(source_config, ()):
            for idx, value in declarer.declare(record):
                result = results[idx]
                self._rows.append({
                    "source_record_path": record_path,
                    "result_idx": idx,
                    "evaluation_id": evaluation_id,
                    # The id as the record carries it, which is what the
                    # exploded row keeps in `evaluation_result_id_raw`;
                    # the pipeline synthesises one further down and the
                    # identity check compares like with like.
                    "evaluation_result_id": result.get("evaluation_result_id"),
                    "evaluation_name": result.get("evaluation_name"),
                    "model_id": model_id,
                    "declaration_source": declarer.declaration_source,
                    "declared_value": value,
                    **declarer.provenance(result, eee_revision),
                })

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def as_arrow(self) -> pa.Table:
        return pa.Table.from_pylist(self._rows, schema=ANNOTATION_SCHEMA)

    def summarise(self) -> dict[str, Any]:
        """Per-source counts, for the run log and `snapshot_meta`."""
        by_value: Counter = Counter()
        for row in self._rows:
            by_value[(row["declaration_source"], row["declared_value"])] += 1
        out: dict[str, Any] = {}
        for (source, value), count in sorted(by_value.items()):
            block = out.setdefault(source, {"results": 0, "by_value": {}})
            block["results"] += count
            block["by_value"][value] = count
        return out


_log = AnnotationLog()


def annotations() -> AnnotationLog:
    return _log


def reset_annotations() -> None:
    _log.reset()

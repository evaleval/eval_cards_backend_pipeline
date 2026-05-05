import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts import registry, signals
from scripts.helpers.benchmark_constants import SUMMARY_SCORE_LEAF_KEYS
from scripts.helpers.benchmark_identity import (
    canonical_benchmark_family_key,
    normalize_benchmark_key as shared_normalize_benchmark_key,
)
from scripts.helpers.presentation import humanize_token_key, join_display_name_parts
from scripts.helpers.slug_utils import (
    slugify,
    slugify_developer,
    slugify_model_segment,
)
from scripts.helpers.hf_transfer import (
    discover_configs,
    ensure_local_benchmark_metadata_snapshot,
    ensure_local_dataset_snapshot,
    list_json_files_for_config,
    read_dataset_json,
    upload_output,
)
from scripts.helpers.entity_resolution import (
    aggregated_display_identity,
    resolve_model_identity_for_pipeline,
)
from scripts.helpers.readme_builder import generate_readme as build_dataset_readme
from scripts.metadata.benchmark_cards import (
    as_string_list,
    benchmark_card_language_keys,
    candidate_benchmark_keys,
    compact_benchmark_key,
    extract_benchmark_tags,
    iter_matching_benchmark_cards,
    load_benchmark_metadata_from_dir,
    lookup_benchmark_card,
    lookup_benchmark_card_for_parent,
)
from scripts.metadata.benchmark_normalization import (
    classify_evaluation_result,
    infer_category_from_benchmark,
)
from scripts.metadata.metrics import (
    METRIC_GROUP_ORDER,
    METRIC_REGISTRY_ALIAS_LOOKUP,
    METRIC_REGISTRY_ENTRIES,
    load_metric_registry,
    metric_group,
    metric_group_order_index,
    pick_primary_metric,
)


PRODUCTION_DATASET_REPO = "evaleval/card_backend"
# `CARD_BACKEND_OUTPUT_REPO` lets local/CI runs target a different upload
# destination. The `resolve_upload_target` guard refuses unintended writes
# to `evaleval/card_backend` from local shells (where `HF_TOKEN` is often
# auto-loaded from a profile); `CARD_BACKEND_ALLOW_PRODUCTION=1` is the
# explicit opt-in for an intentional manual local prod push. CI runs
# (`GITHUB_ACTIONS=true`) deploy to production by default — the gate
# there is owner PR review at merge time, not an env flag.
#
# Both `DATASET_REPO` and `DATASET_RESOLVE_BASE` are resolved at *import*
# time. Tests that need to flip the env after import should call
# `reload_dataset_target()` to re-bind the module-level constants.
EEE_DATASET_REPO = "evaleval/EEE_datastore"
BENCHMARK_METADATA_DATASET_REPO = "evaleval/auto-benchmarkcards"
EEE_DATASET_RAW_BASE = f"https://huggingface.co/datasets/{EEE_DATASET_REPO}/raw/main"


def _resolve_dataset_repo() -> str:
    return os.environ.get("CARD_BACKEND_OUTPUT_REPO") or PRODUCTION_DATASET_REPO


DATASET_REPO = _resolve_dataset_repo()
DATASET_RESOLVE_BASE = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"


def reload_dataset_target() -> None:
    """Re-bind module-level dataset constants from the current env.

    Used by tests + CLI flags that change ``CARD_BACKEND_OUTPUT_REPO``
    after import. Without this, validators / README generators continue
    to reference the import-time value and silently produce stale URLs.
    """
    global DATASET_REPO, DATASET_RESOLVE_BASE
    DATASET_REPO = _resolve_dataset_repo()
    DATASET_RESOLVE_BASE = (
        f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"
    )
CONFIG_VERSION = 1
OUTPUT_DIR = Path("output")


def emit_legacy_json() -> bool:
    """Whether to emit the per-detail JSON artifacts the frontend stopped reading.

    Default off. The frontend reads everything from `duckdb/v1/*.parquet`
    plus a small set of top-level JSONs (manifest, eval-hierarchy,
    benchmark-metadata, comparison-index, corpus-aggregates, peer-ranks).
    The per-model/per-eval/per-developer JSON dirs and their list-view
    aggregators (model-cards*, eval-list*, developers.json) are dead in
    that mode and add ~340 MB + thousands of files to every upload.

    Tests set EMIT_LEGACY_JSON=1 in conftest to preserve their JSON-based
    assertions until they're migrated to read from parquet.
    """
    return os.environ.get("EMIT_LEGACY_JSON", "").strip() == "1"

DEFAULT_LOCAL_DATASET_DIR = ".cache/eee_datastore"
DEFAULT_LOCAL_BENCHMARK_METADATA_DIR = ".cache/auto_benchmarkcards"
DEFAULT_METRIC_REGISTRY_PATH = Path("registry/metric_looking_strings.json")
FILE_READ_MAX_RETRIES = 5
FILE_READ_RETRY_DELAY_SEC = 1.5
# EEE configs to drop entirely — upstream data-quality issues mean these
# rows shouldn't be ingested at all. Filter both discovered and explicit
# `CONFIGS=` overrides so production can never accidentally publish them.
IGNORED_CONFIGS = {"alphaxiv"}
# Adding ``helm_capabilities`` / ``helm_instruct`` here is unsafe:
# ``canonical_benchmark_display_name`` walks every candidate and fires
# PREFERRED on any match, so an entry for ``helm_capabilities`` would
# trigger whenever that key appears as ANY candidate, including
# sub-benchmark rows whose family_key happens to be ``helm_capabilities``
# (e.g. ``helm_capabilities_mmlu_pro``). Override clobbers leaf_name
# for every child eval_summary in those families. For SWE-PolyBench /
# Multi-SWE-bench the same over-firing is desirable (all children
# should display the parent name); for HELM each child has its own
# distinct name and must be preserved. The HELM-duplicate display
# issue requires a more surgical mechanism (or the deferred
# family-collapse work) — left visible until then.
KNOWN_TOP_LEVEL_KEYS = {
    "schema_version",
    "evaluation_id",
    "retrieved_timestamp",
    "source_metadata",
    "eval_library",
    "model_info",
    "evaluation_results",
    "detailed_evaluation_results",
}


def as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_benchmark_key(value: Any) -> str:
    return shared_normalize_benchmark_key(value)


def parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = as_string(value).strip()
    if not text:
        return None

    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_params_billions_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if numeric > 0 else None

    text = as_string(value).strip().lower()
    if not text:
        return None

    text = text.replace(",", "")
    scale = 1.0
    if "trillion" in text or re.search(r"\bt\b", text):
        scale = 1000.0
    elif "million" in text or re.search(r"\bm\b", text):
        scale = 0.001
    elif "thousand" in text or re.search(r"\bk\b", text):
        scale = 0.000001
    elif "billion" in text or re.search(r"\bb\b", text):
        scale = 1.0

    number_match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not number_match:
        return None
    try:
        numeric = float(number_match.group(0)) * scale
    except Exception:
        return None
    return numeric if numeric > 0 else None


def infer_params_billions_from_name(*values: Any) -> float | None:
    patterns = [
        re.compile(
            r"(\d+(?:\.\d+)?)\s*[x*]\s*(\d+(?:\.\d+)?)\s*b", flags=re.IGNORECASE
        ),
        re.compile(r"(\d+(?:\.\d+)?)\s*b", flags=re.IGNORECASE),
        re.compile(r"(\d+(?:\.\d+)?)\s*m(?:\b|illion)", flags=re.IGNORECASE),
    ]

    for value in values:
        text = as_string(value)
        if not text:
            continue

        mo = patterns[0].search(text)
        if mo:
            left = parse_float(mo.group(1))
            right = parse_float(mo.group(2))
            if left and right:
                inferred = left * right
                if inferred > 0:
                    return inferred

        mo = patterns[1].search(text)
        if mo:
            inferred = parse_float(mo.group(1))
            if inferred and inferred > 0:
                return inferred

        mo = patterns[2].search(text)
        if mo:
            inferred_m = parse_float(mo.group(1))
            if inferred_m and inferred_m > 0:
                return inferred_m / 1000.0

    return None


def derive_model_params_billions(model_info: dict) -> float | None:
    additional = model_info.get("additional_details")
    additional_details = additional if isinstance(additional, dict) else {}

    candidate_paths = [
        model_info.get("params_billions"),
        model_info.get("parameters_billions"),
        model_info.get("parameter_count_billions"),
        model_info.get("parameter_count"),
        model_info.get("num_parameters"),
        additional_details.get("params_billions"),
        additional_details.get("parameters_billions"),
        additional_details.get("parameter_count_billions"),
        additional_details.get("parameter_count"),
        additional_details.get("num_parameters"),
        additional_details.get("parameters"),
        additional_details.get("model_size"),
    ]

    for candidate in candidate_paths:
        parsed = parse_params_billions_value(candidate)
        if parsed is not None:
            return parsed

    return infer_params_billions_from_name(model_info.get("name"), model_info.get("id"))


def iso_from_epoch_string(value: Any) -> str | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return (
        datetime.fromtimestamp(numeric, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def max_iso(left: str | None, right: str | None) -> str | None:
    if not left:
        return right
    if not right:
        return left
    return left if left > right else right


def raw_url_for_dataset_path(dataset_path: str) -> str:
    return f"{EEE_DATASET_RAW_BASE}/{dataset_path.lstrip('/')}"


def normalize_detailed_eval_meta(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if isinstance(value.get("entries"), dict):
            return value["entries"]
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        file_path_match = re.search(
            r"file_path'?:\s*'([^']+)'", value, flags=re.IGNORECASE
        ) or re.search(r'"file_path"\s*:\s*"([^"]+)"', value)
        format_match = re.search(
            r"format'?:\s*'([^']+)'", value, flags=re.IGNORECASE
        ) or re.search(r'"format"\s*:\s*"([^"]+)"', value)
        rows_match = re.search(
            r"total_rows'?:\s*(\d+)", value, flags=re.IGNORECASE
        ) or re.search(r'"total_rows"\s*:\s*(\d+)', value)
        if file_path_match or format_match or rows_match:
            return {
                "file_path": file_path_match.group(1) if file_path_match else None,
                "format": format_match.group(1) if format_match else None,
                "total_rows": int(rows_match.group(1)) if rows_match else None,
            }
    return None


def resolve_detailed_results_url(record: dict, source_record_url: str) -> str | None:
    value = record.get("detailed_evaluation_results")
    if isinstance(value, str) and value:
        if value.startswith("http://") or value.startswith("https://"):
            return value
        cleaned = value.lstrip("/")
        if cleaned.startswith("data/"):
            return raw_url_for_dataset_path(cleaned)
    if isinstance(value, dict):
        file_path = as_string(
            value.get("file_path") or value.get("path") or value.get("url")
        )
        if file_path:
            if file_path.startswith("http://") or file_path.startswith("https://"):
                return file_path
            if file_path.startswith("data/"):
                return raw_url_for_dataset_path(file_path)
            if source_record_url:
                base = source_record_url[: source_record_url.rfind("/") + 1]
                return f"{base}{file_path.lstrip('/')}"
    if source_record_url.endswith(".json"):
        return f"{source_record_url[:-5]}_samples.jsonl"
    return None


def infer_interaction_type(instances: list[dict]) -> str:
    if not instances:
        return "unknown"
    first = instances[0]
    if isinstance(first, dict):
        if "interactions" in first or "messages" in first:
            return "multi_turn"
        if "tool_calls" in first or "tool_use" in first:
            return "agentic"
        if "input" in first and "output" in first and "evaluation" in first:
            return "single_turn"
    return "unknown"


def maybe_load_instance_data(
    record: dict, local_dataset_dir: str | None, hf_token: str | None
) -> dict | None:
    candidates: list[str] = []
    explicit = as_string(record.get("detailed_evaluation_results"))
    source_record_url = as_string(record.get("source_record_url"))

    if explicit:
        candidates.append(explicit)
    if source_record_url.endswith(".json"):
        base = source_record_url[:-5]
        candidates.append(f"{base}_samples.jsonl")
        candidates.append(f"{base}.jsonl")

    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    for url in deduped:
        dataset_path = ""
        if url.startswith(f"{EEE_DATASET_RAW_BASE}/"):
            dataset_path = url[len(EEE_DATASET_RAW_BASE) + 1 :]

        try:
            if local_dataset_dir and dataset_path:
                text = (Path(local_dataset_dir) / dataset_path).read_text(
                    encoding="utf-8"
                )
            elif dataset_path:
                local_path = hf_hub_download(
                    repo_id=EEE_DATASET_REPO,
                    filename=dataset_path,
                    repo_type="dataset",
                    token=hf_token,
                )
                text = Path(local_path).read_text(encoding="utf-8")
            else:
                continue
        except Exception:
            continue

        lines = [line for line in text.splitlines() if line.strip()]
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

        if rows:
            examples = rows[:] if len(rows) <= 5 else random.sample(rows, 5)
            return {
                "interaction_type": infer_interaction_type(rows),
                "instance_count": len(rows),
                "source_url": url,
                "instance_examples": examples,
            }

    return None


def read_text_from_dataset_url(
    url: str, local_dataset_dir: str | None, hf_token: str | None
) -> str | None:
    dataset_path = ""
    if url.startswith(f"{EEE_DATASET_RAW_BASE}/"):
        dataset_path = url[len(EEE_DATASET_RAW_BASE) + 1 :]

    try:
        if local_dataset_dir and dataset_path:
            return (Path(local_dataset_dir) / dataset_path).read_text(encoding="utf-8")
        if dataset_path:
            local_path = hf_hub_download(
                repo_id=EEE_DATASET_REPO,
                filename=dataset_path,
                repo_type="dataset",
                token=hf_token,
            )
            return Path(local_path).read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def dataset_resolve_url(relative_path: str) -> str:
    return f"{DATASET_RESOLVE_BASE}/{relative_path.lstrip('/')}"


def build_instance_artifact_relative_path(evaluation: dict) -> str:
    route_id = as_string(
        (evaluation.get("model_info") or {}).get("model_route_id") or "unknown"
    )
    evaluation_key = slugify(
        evaluation.get("evaluation_id")
        or evaluation.get("source_record_url")
        or "instance"
    )
    return f"instances/{route_id}/{evaluation_key}.jsonl"


def build_record_artifact_relative_path(evaluation: dict) -> str:
    route_id = as_string(
        (evaluation.get("model_info") or {}).get("model_route_id") or "unknown"
    )
    evaluation_key = slugify(evaluation.get("evaluation_id") or "record")
    return f"records/{route_id}/{evaluation_key}.json"


def build_evaluation_hierarchy_payload(evaluation: dict) -> dict:
    category = infer_category_from_benchmark(
        as_string(evaluation.get("benchmark")), evaluation.get("benchmark_card")
    )
    model_info = evaluation.get("model_info") or {}
    return {
        "category": category,
        "benchmark": evaluation.get("benchmark"),
        "model_family_id": as_string(model_info.get("family_id")),
        "model_route_id": as_string(model_info.get("model_route_id")),
        "eval_summary_ids": evaluation.get("eval_summary_ids", []),
    }


def build_result_hierarchy_payload(evaluation: dict, result: dict) -> dict:
    normalized = result.get("normalized_result") or {}
    return {
        **build_evaluation_hierarchy_payload(evaluation),
        "eval_summary_id": get_eval_group_id(evaluation, result),
        "metric_summary_id": get_metric_summary_id(evaluation, result),
        "benchmark_family_key": normalized.get("benchmark_family_key"),
        "benchmark_family_name": normalized.get("benchmark_family_name"),
        "benchmark_parent_key": normalized.get("benchmark_parent_key"),
        "benchmark_parent_name": normalized.get("benchmark_parent_name"),
        "benchmark_leaf_key": normalized.get("benchmark_leaf_key"),
        "benchmark_leaf_name": normalized.get("benchmark_leaf_name"),
        "benchmark_component_key": normalized.get("benchmark_component_key"),
        "benchmark_component_name": normalized.get("benchmark_component_name"),
        "slice_key": normalized.get("slice_key"),
        "slice_name": normalized.get("slice_name"),
        "metric_key": normalized.get("metric_key"),
        "metric_name": normalized.get("metric_name"),
        "metric_source": normalized.get("metric_source"),
        "display_name": normalized.get("display_name"),
        "canonical_display_name": normalized.get("canonical_display_name"),
        "is_summary_score": bool(normalized.get("is_summary_score")),
    }


def find_matching_result_for_instance_row(evaluation: dict, row: dict) -> dict | None:
    results = evaluation.get("evaluation_results") or []
    if not results:
        return None

    evaluation_result_id = as_string(row.get("evaluation_result_id")).strip()
    if evaluation_result_id:
        for result in results:
            if (
                as_string(result.get("evaluation_result_id")).strip()
                == evaluation_result_id
            ):
                return result

    evaluation_name = as_string(row.get("evaluation_name")).strip()
    if evaluation_name:
        matches = []
        for result in results:
            normalized = result.get("normalized_result") or {}
            candidate_names = {
                as_string(result.get("evaluation_name")).strip(),
                as_string(normalized.get("raw_evaluation_name")).strip(),
                as_string(normalized.get("display_name")).strip(),
                as_string(normalized.get("canonical_display_name")).strip(),
                as_string(normalized.get("benchmark_leaf_name")).strip(),
            } - {""}
            if evaluation_name in candidate_names:
                matches.append(result)
        if len(matches) == 1:
            return matches[0]

    if len(results) == 1:
        return results[0]
    return None


def annotate_instance_row(evaluation: dict, row: dict) -> dict:
    annotated = dict(row)
    matched_result = find_matching_result_for_instance_row(evaluation, annotated)
    if matched_result is not None:
        annotated["hierarchy"] = build_result_hierarchy_payload(
            evaluation, matched_result
        )
    else:
        annotated["hierarchy"] = build_evaluation_hierarchy_payload(evaluation)
        if len(annotated["hierarchy"].get("eval_summary_ids", [])) == 1:
            annotated["hierarchy"]["eval_summary_id"] = annotated["hierarchy"][
                "eval_summary_ids"
            ][0]
    return annotated


def transform_instance_artifact_text(evaluation: dict, artifact_text: str) -> str:
    transformed_lines = []
    for line in artifact_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except Exception:
            transformed_lines.append(stripped)
            continue
        transformed_lines.append(
            json.dumps(annotate_instance_row(evaluation, row), ensure_ascii=False)
        )
    return "\n".join(transformed_lines) + ("\n" if transformed_lines else "")


def extract_score(result: dict) -> float | None:
    score_details = result.get("score_details") if isinstance(result, dict) else None
    if not isinstance(score_details, dict):
        return None
    score = score_details.get("score")
    try:
        return float(score)
    except Exception:
        return None


def get_eval_group_id(evaluation: dict, result: dict) -> str:
    normalized = result.get("normalized_result") if isinstance(result, dict) else None
    source_data = result.get("source_data") if isinstance(result, dict) else {}
    parent_key = (
        (normalized or {}).get("benchmark_parent_key")
        or evaluation.get("benchmark")
        or (source_data or {}).get("dataset_name")
    )
    benchmark_key = (
        (normalized or {}).get("benchmark_leaf_key")
        or (normalized or {}).get("benchmark_family_key")
        or evaluation.get("benchmark")
        or (source_data or {}).get("dataset_name")
        or result.get("evaluation_name")
    )
    pieces = []
    if as_string(parent_key):
        pieces.append(parent_key)
    if as_string(benchmark_key) and as_string(benchmark_key) != as_string(parent_key):
        pieces.append(benchmark_key)
    return slugify("__".join(as_string(piece) for piece in pieces if as_string(piece)))


def get_metric_summary_id(evaluation: dict, result: dict) -> str:
    normalized = result.get("normalized_result") if isinstance(result, dict) else None
    metric_key = (normalized or {}).get("metric_key")
    pieces = [get_eval_group_id(evaluation, result)]
    slice_key = (normalized or {}).get("slice_key")
    if slice_key:
        pieces.append(slice_key)
    if metric_key:
        pieces.append(metric_key)
    return slugify("__".join(as_string(piece) for piece in pieces if as_string(piece)))


def clean_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "instances").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "records").mkdir(parents=True, exist_ok=True)
    if emit_legacy_json():
        (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "evals").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "developers").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_output_relative_files(root_dir: Path = OUTPUT_DIR) -> list[str]:
    if not root_dir.exists():
        return []
    return sorted(
        str(path.relative_to(root_dir)).replace(os.sep, "/")
        for path in root_dir.rglob("*")
        if path.is_file()
    )


def build_lightweight_model_cards(model_cards: list[dict]) -> list[dict]:
    lite_cards: list[dict] = []

    for card in model_cards:
        lite_cards.append(
            {
                "model_family_id": card.get("model_family_id"),
                "model_route_id": card.get("model_route_id"),
                "model_family_name": card.get("model_family_name"),
                "developer": card.get("developer"),
                "params_billions": card.get("params_billions"),
                "total_evaluations": card.get("total_evaluations"),
                "benchmark_count": card.get("benchmark_count"),
                "benchmark_family_count": card.get("benchmark_family_count"),
                "categories_covered": card.get("categories_covered") or [],
                "last_updated": card.get("last_updated"),
                "variants": [
                    {
                        "variant_key": variant.get("variant_key"),
                        "variant_label": variant.get("variant_label"),
                        "evaluation_count": variant.get("evaluation_count"),
                        "raw_model_ids": [],
                        "last_updated": variant.get("last_updated"),
                    }
                    for variant in (card.get("variants") or [])
                ],
                "score_summary": card.get("score_summary") or {},
                "reproducibility_summary": card.get("reproducibility_summary"),
                "provenance_summary": card.get("provenance_summary"),
                "comparability_summary": card.get("comparability_summary"),
                "benchmark_names": (card.get("benchmark_names") or [])[:8],
                "top_benchmark_scores": (card.get("top_benchmark_scores") or [])[:6],
            }
        )

    return lite_cards


def build_lightweight_eval_list(eval_list: dict) -> dict:
    lite_evals: list[dict] = []

    for summary in eval_list.get("evals") or []:
        instance_data = summary.get("instance_data") or {}
        lite_evals.append(
            {
                "eval_summary_id": summary.get("eval_summary_id"),
                "benchmark": summary.get("benchmark"),
                "benchmark_family_key": summary.get("benchmark_family_key"),
                "benchmark_family_name": summary.get("benchmark_family_name"),
                "benchmark_parent_key": summary.get("benchmark_parent_key"),
                "benchmark_parent_name": summary.get("benchmark_parent_name"),
                "benchmark_leaf_key": summary.get("benchmark_leaf_key"),
                "benchmark_leaf_name": summary.get("benchmark_leaf_name"),
                "benchmark_component_key": summary.get("benchmark_component_key"),
                "benchmark_component_name": summary.get("benchmark_component_name"),
                "evaluation_name": summary.get("evaluation_name"),
                "display_name": summary.get("display_name"),
                "canonical_display_name": summary.get("canonical_display_name"),
                "is_summary_score": summary.get("is_summary_score", False),
                "summary_score_for": summary.get("summary_score_for"),
                "summary_score_for_name": summary.get("summary_score_for_name"),
                "summary_eval_ids": summary.get("summary_eval_ids") or [],
                "category": summary.get("category", "other"),
                "models_count": summary.get("models_count", 0),
                "metrics_count": summary.get("metrics_count"),
                "subtasks_count": summary.get("subtasks_count"),
                "metric_names": summary.get("metric_names") or [],
                "primary_metric_name": summary.get("primary_metric_name"),
                "tags": summary.get("tags")
                or {"domains": [], "languages": [], "tasks": []},
                "source_data": summary.get("source_data"),
                "metrics": summary.get("metrics") or [],
                "top_score": summary.get("top_score"),
                "instance_data": {
                    "available": bool(instance_data.get("available", False)),
                    "url_count": instance_data.get("url_count", 0),
                    "sample_urls": (instance_data.get("sample_urls") or [])[:1],
                    "models_with_loaded_instances": instance_data.get(
                        "models_with_loaded_instances", 0
                    ),
                },
                "reproducibility_summary": summary.get("reproducibility_summary"),
                "provenance_summary": summary.get("provenance_summary"),
                "comparability_summary": summary.get("comparability_summary"),
            }
        )

    return {"evals": lite_evals}


def collect_artifact_sizes(root_dir: Path = OUTPUT_DIR) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for relative_path in iter_output_relative_files(root_dir):
        file_path = root_dir / relative_path
        artifacts.append(
            {
                "path": relative_path,
                "bytes": file_path.stat().st_size,
            }
        )
    return artifacts


# Uniform grouping for every metric the pipeline emits. The group label is
# attached to each metric in comparison-index.json so the frontend can render
# tabs in a consistent order ("Capability" before "Cost" before "Latency"
# etc.) across every benchmark, and is also used internally to choose a
# primary metric for eval-list.json's ``primary_metric_name``.
#
# Rules were derived empirically from the full set of 207 metric occurrences
# across 89 evals (see commit message). ``metric_kind`` from upstream EEE
# takes precedence; for the ~78% of metrics where ``metric_kind`` is absent
# we fall back to name regexes that cover every observed metric name.
# Submission "axes" — the kind of thing that differentiates one row from
# another for the same (model_route_id, eval, metric). Surfaced per row so
# the frontend can label the bar correctly: "Harness: droid" vs "Variant:
# thinking-8k" vs "Re-run: 2026-03-17".
RUN_KIND_HARNESS = "harness"  # different agent / scaffold (terminal_bench, swe_bench)
RUN_KIND_VARIANT = "variant"  # raw_model_id varies (reasoning budget, snapshot)
RUN_KIND_RERUN = "rerun"  # same setup, re-evaluated later
RUN_KIND_DEFAULT = "default"  # the only submission for this peer


def extract_run_descriptor(row: dict) -> tuple[str, str]:
    """Return ``(run_kind, run_label)`` for a single ``model_results`` row.

    These rows are *submissions*, not subtasks. Multiple rows for the same
    ``model_route_id`` on a benchmark generally mean one of three things:

      * **harness**: same model run by different agent scaffolding —
        terminal_bench's droid / letta-code / mux / openhands / claude-code
        agents, browsecompplus's smolagents-code / openai-solo / claude-
        code-cli, swe_bench's openai-solo / claude-code-cli / smolagents-
        code. The harness is encoded as the ``<harness>__<model>`` segment
        of ``evaluation_id``.
      * **variant**: same family but a different ``raw_model_id`` —
        reasoning-budget variants (claude-haiku-4-5 + ``-thinking-1k`` /
        ``-thinking-8k``) or snapshot dates (claude-3-5-sonnet
        ``-20240620`` / ``-20241022``) that the canonical model identity
        collapses to one family.
      * **rerun**: identical model and setup, evaluated at a later date —
        differentiated only by ``retrieved_timestamp`` /
        ``evaluation_timestamp``.

    Returned ``run_kind`` is one of ``RUN_KIND_*``; ``run_label`` is a
    short string the frontend can display ("droid", "thinking-8k",
    "2026-03-17"). Single-submission peers receive
    ``(RUN_KIND_DEFAULT, "")`` from the caller — this function only runs
    when there is something to differentiate.
    """
    eval_id = as_string(row.get("evaluation_id"))
    if eval_id:
        parts = eval_id.split("/")
        if len(parts) >= 2 and "__" in parts[1]:
            harness = parts[1].split("__", 1)[0].strip()
            if harness:
                return RUN_KIND_HARNESS, harness

    raw = as_string(row.get("raw_model_id"))
    family = as_string(row.get("model_id"))
    if raw and family and raw != family:
        if raw.lower().startswith(family.lower()):
            tail = raw[len(family) :].lstrip("-_/").strip()
            if tail:
                return RUN_KIND_VARIANT, tail
        return RUN_KIND_VARIANT, raw

    pt = row.get("passthrough_top_level_fields") or {}
    eval_ts = as_string(pt.get("evaluation_timestamp"))
    if eval_ts:
        return RUN_KIND_RERUN, eval_ts

    retrieved = as_string(row.get("retrieved_timestamp"))
    if retrieved:
        try:
            iso = (
                datetime.fromtimestamp(float(retrieved), tz=timezone.utc)
                .date()
                .isoformat()
            )
            return RUN_KIND_RERUN, iso
        except (TypeError, ValueError):
            return RUN_KIND_RERUN, retrieved
    return RUN_KIND_RERUN, "submission"


def build_comparison_index(eval_summaries: list[dict], generated_at: str) -> dict:
    """Build an exhaustive per-eval, per-metric comparison index.

    For each eval_summary, emits every metric the eval reports — the frontend
    renders one tab per metric in its histogram view, so there is no
    ``primary metric`` here. Each metric carries its full ``scores`` list:
    one entry per scoring model_route_id, ranked best-first respecting
    ``lower_is_better``. Also emits an inverse ``by_model`` index keyed by
    (model_route_id, eval_summary_id, metric_summary_id) so a model detail
    page can look up its peer comparisons in O(1) per benchmark+metric.

    Metrics within an eval are ordered by ``metric_group`` (capability tabs
    first, then robustness / efficiency / cost / latency / rank / other), so
    the tab strip has the same shape across every benchmark.
    """

    evals_out: dict[str, dict] = {}
    by_model: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for summary in eval_summaries:
        eval_summary_id = as_string(summary.get("eval_summary_id"))
        if not eval_summary_id:
            continue

        # Use root metrics if the eval has any; otherwise fall back to the
        # first subtask's metrics. Mirrors the same fallback that
        # ``primary_metric_name`` uses in eval-list.json.
        root_metrics = summary.get("metrics") or []
        subtasks = summary.get("subtasks") or []
        candidate_metrics = (
            root_metrics or (subtasks[0].get("metrics") if subtasks else None) or []
        )
        if not candidate_metrics:
            continue

        # Capability tabs surface first across every eval. Within-group
        # ordering is alphabetical to keep the artifact deterministic.
        ordered_metrics = sorted(
            candidate_metrics,
            key=lambda m: (
                metric_group_order_index(metric_group(m)),
                as_string(m.get("metric_name")),
                as_string(m.get("metric_summary_id")),
            ),
        )

        metrics_out: list[dict] = []
        for metric in ordered_metrics:
            metric_summary_id = as_string(metric.get("metric_summary_id"))
            if not metric_summary_id:
                continue
            lower_is_better = bool(metric.get("lower_is_better"))
            group = metric_group(metric)

            # Group rows by model_route_id. Multiple rows per route are
            # *submissions* (different agent harnesses, reasoning-budget
            # variants, model snapshots, or simple re-runs) — see
            # `extract_run_descriptor`. The headline bar uses the best
            # submission's score; the full submission list is kept so the
            # frontend can drill in.
            rows_by_route: dict[str, list[dict]] = defaultdict(list)
            for row in metric.get("model_results") or []:
                route = as_string(row.get("model_route_id"))
                if not route:
                    continue
                if row.get("score") is None:
                    continue
                rows_by_route[route].append(row)

            # Build a per-route headline entry + its submission tail.
            route_entries: list[tuple[str, dict, list[dict]]] = []
            for route, rows in rows_by_route.items():
                # Sort submissions best-first within the route.
                rows = sorted(
                    rows,
                    key=lambda r: r["score"],
                    reverse=not lower_is_better,
                )
                headline = rows[0]
                # Only label submissions when more than one exists for this
                # route — single-submission peers don't need a setup label.
                if len(rows) > 1:
                    submissions = []
                    for sub in rows:
                        run_kind, run_label = extract_run_descriptor(sub)
                        submissions.append(
                            {
                                "score": sub["score"],
                                "run_kind": run_kind,
                                "run_label": run_label,
                                "raw_model_id": as_string(sub.get("raw_model_id"))
                                or None,
                            }
                        )
                else:
                    submissions = []
                route_entries.append((route, headline, submissions))

            # Two-pass stable sort across routes: route id ascending
            # tiebreak, then headline score in the metric's preferred
            # direction.
            route_entries.sort(key=lambda e: e[0])
            route_entries.sort(key=lambda e: e[1]["score"], reverse=not lower_is_better)

            total = len(route_entries)
            scores_out: list[dict] = []
            position = 0
            previous_score = None
            for idx, (route_id, headline, submissions) in enumerate(
                route_entries, start=1
            ):
                row_score = headline["score"]
                if previous_score is None or row_score != previous_score:
                    position = idx
                    previous_score = row_score
                # Detect the submission axis for this peer (used by the UI to
                # decide how to caption the drill-in: "8 harnesses" vs
                # "3 reasoning budgets" vs "2 re-runs").
                if submissions:
                    kinds = {s["run_kind"] for s in submissions}
                    submission_axis = next(iter(kinds)) if len(kinds) == 1 else "mixed"
                    headline_kind, headline_label = extract_run_descriptor(headline)
                else:
                    submission_axis = RUN_KIND_DEFAULT
                    headline_kind, headline_label = RUN_KIND_DEFAULT, ""

                entry: dict = {
                    "model_route_id": route_id,
                    "model_family_id": as_string(headline.get("model_id")),
                    "model_family_name": as_string(headline.get("model_name")),
                    "developer": as_string(headline.get("developer")),
                    "variant_key": as_string(headline.get("variant_key")) or "default",
                    "score": row_score,
                    "rank": position,
                    "total": total,
                    "submission_count": len(submissions) if submissions else 1,
                    "submission_axis": submission_axis,
                }
                if submissions:
                    entry["headline_run_kind"] = headline_kind
                    entry["headline_run_label"] = headline_label
                    entry["submissions"] = submissions
                scores_out.append(entry)

                by_model[route_id][eval_summary_id][metric_summary_id] = {
                    "score": row_score,
                    "rank": position,
                    "total": total,
                    "submission_count": entry["submission_count"],
                    "submission_axis": submission_axis,
                }

            metric_config = metric.get("metric_config") or {}
            unit = (
                as_string(metric_config.get("unit"))
                or as_string(metric_config.get("metric_unit"))
                or None
            )

            metrics_out.append(
                {
                    "metric_summary_id": metric_summary_id,
                    "metric_name": as_string(metric.get("metric_name")),
                    "metric_id": as_string(metric.get("metric_id")),
                    "metric_key": as_string(metric.get("metric_key")),
                    "group": group,
                    "group_order": metric_group_order_index(group),
                    "lower_is_better": lower_is_better,
                    "unit": unit,
                    "scores": scores_out,
                }
            )

        if not metrics_out:
            continue

        evals_out[eval_summary_id] = {
            "eval_summary_id": eval_summary_id,
            "canonical_benchmark_id": summary.get("canonical_benchmark_id"),
            "benchmark_family_key": summary.get("benchmark_family_key"),
            "benchmark_family_name": summary.get("benchmark_family_name"),
            "benchmark_parent_key": summary.get("benchmark_parent_key"),
            "benchmark_parent_name": summary.get("benchmark_parent_name"),
            "benchmark_leaf_key": summary.get("benchmark_leaf_key"),
            "benchmark_leaf_name": summary.get("benchmark_leaf_name"),
            "display_name": summary.get("display_name"),
            "category": summary.get("category", "other"),
            "is_summary_score": bool(summary.get("is_summary_score")),
            "summary_score_for": summary.get("summary_score_for"),
            "summary_eval_ids": summary.get("summary_eval_ids", []),
            "metrics": metrics_out,
        }

    return {
        "generated_at": generated_at,
        "config_version": CONFIG_VERSION,
        "metric_group_order": list(METRIC_GROUP_ORDER),
        "evals": evals_out,
        "by_model": {
            route: {eid: dict(metric_map) for eid, metric_map in eval_map.items()}
            for route, eval_map in by_model.items()
        },
    }


def validate_output_contract(output_dir: Path = OUTPUT_DIR) -> None:
    errors: list[str] = []

    eval_list_path = output_dir / "eval-list.json"
    model_cards_path = output_dir / "model-cards.json"
    evals_dir = output_dir / "evals"
    models_dir = output_dir / "models"

    eval_summary_ids: set[str] = set()
    if eval_list_path.exists():
        eval_list = json.loads(eval_list_path.read_text(encoding="utf-8"))
        for item in eval_list.get("evals") or []:
            primary = as_string(item.get("eval_summary_id"))
            if primary:
                eval_summary_ids.add(primary)
            # Demoted siblings of canonical-collapsed rows are still emitted
            # as evals/<id>.json for drill-down; track them so the
            # files-vs-list parity check passes.
            for source in item.get("reporting_sources") or []:
                if as_string(source):
                    eval_summary_ids.add(as_string(source))

    published_eval_files = {
        path.stem for path in evals_dir.glob("*.json") if path.is_file()
    }
    if eval_summary_ids and published_eval_files != eval_summary_ids:
        missing_files = sorted(eval_summary_ids - published_eval_files)
        extra_files = sorted(published_eval_files - eval_summary_ids)
        if missing_files:
            errors.append(
                f"Missing eval files for eval-list entries: {missing_files[:10]}"
            )
        if extra_files:
            errors.append(
                f"Extra eval files not present in eval-list: {extra_files[:10]}"
            )

    required_eval_keys = [
        "benchmark_family_key",
        "benchmark_family_name",
        "benchmark_parent_key",
        "benchmark_parent_name",
        "benchmark_leaf_key",
        "benchmark_leaf_name",
        "canonical_display_name",
    ]

    for eval_path in sorted(evals_dir.glob("*.json")):
        parsed = json.loads(eval_path.read_text(encoding="utf-8"))
        missing_keys = [key for key in required_eval_keys if not parsed.get(key)]
        if missing_keys:
            errors.append(
                f"{eval_path.name} missing top-level hierarchy keys: {missing_keys}"
            )

        for metric in parsed.get("metrics", []):
            for row in metric.get("model_results", []):
                record_url = as_string(row.get("source_record_url"))
                if record_url and not record_url.startswith(
                    f"{DATASET_RESOLVE_BASE}/records/"
                ):
                    errors.append(
                        f"{eval_path.name} has non-pipeline source_record_url: {record_url}"
                    )
                detailed_url = as_string(row.get("detailed_evaluation_results"))
                if detailed_url and not detailed_url.startswith(
                    f"{DATASET_RESOLVE_BASE}/instances/"
                ):
                    errors.append(
                        f"{eval_path.name} has non-pipeline detailed_evaluation_results URL: {detailed_url}"
                    )
                instance_data = row.get("instance_level_data") or {}
                source_url = as_string(instance_data.get("source_url"))
                if source_url and not source_url.startswith(
                    f"{DATASET_RESOLVE_BASE}/instances/"
                ):
                    errors.append(
                        f"{eval_path.name} has non-pipeline instance_level_data.source_url: {source_url}"
                    )

        for subtask in parsed.get("subtasks", []):
            for metric in subtask.get("metrics", []):
                for row in metric.get("model_results", []):
                    record_url = as_string(row.get("source_record_url"))
                    if record_url and not record_url.startswith(
                        f"{DATASET_RESOLVE_BASE}/records/"
                    ):
                        errors.append(
                            f"{eval_path.name} has non-pipeline source_record_url: {record_url}"
                        )
                    detailed_url = as_string(row.get("detailed_evaluation_results"))
                    if detailed_url and not detailed_url.startswith(
                        f"{DATASET_RESOLVE_BASE}/instances/"
                    ):
                        errors.append(
                            f"{eval_path.name} has non-pipeline detailed_evaluation_results URL: {detailed_url}"
                        )
                    instance_data = row.get("instance_level_data") or {}
                    source_url = as_string(instance_data.get("source_url"))
                    if source_url and not source_url.startswith(
                        f"{DATASET_RESOLVE_BASE}/instances/"
                    ):
                        errors.append(
                            f"{eval_path.name} has non-pipeline instance_level_data.source_url: {source_url}"
                        )

    for model_path in sorted(models_dir.glob("*.json")):
        parsed = json.loads(model_path.read_text(encoding="utf-8"))
        if "hierarchy_by_category" not in parsed:
            errors.append(f"{model_path.name} missing hierarchy_by_category")

    model_cards_by_route_id: dict[str, dict] = {}
    if model_cards_path.exists():
        parsed_model_cards = json.loads(model_cards_path.read_text(encoding="utf-8"))
        if isinstance(parsed_model_cards, list):
            model_cards_by_route_id = {
                as_string(card.get("model_route_id")): card
                for card in parsed_model_cards
                if as_string(card.get("model_route_id"))
            }

    for model_path in sorted(models_dir.glob("*.json")):
        parsed = json.loads(model_path.read_text(encoding="utf-8"))
        hierarchy_by_category = parsed.get("hierarchy_by_category") or {}
        hierarchy_categories = sorted(
            category
            for category, summaries in hierarchy_by_category.items()
            if summaries
        )
        actual_eval_summary_ids = {
            as_string(summary.get("eval_summary_id"))
            for summaries in hierarchy_by_category.values()
            for summary in summaries or []
            if as_string(summary.get("eval_summary_id"))
        }

        declared_categories = sorted(
            as_string(category)
            for category in parsed.get("categories_covered") or []
            if as_string(category)
        )
        if declared_categories != hierarchy_categories:
            errors.append(
                f"{model_path.name} categories_covered mismatch: declared={declared_categories} actual={hierarchy_categories}"
            )

        route_id = as_string(parsed.get("model_route_id")) or model_path.stem
        model_card = model_cards_by_route_id.get(route_id)
        if not model_card:
            continue

        expected_eval_summary_ids = {
            as_string(entry.get("benchmarkKey"))
            for entry in model_card.get("top_benchmark_scores") or []
            if as_string(entry.get("benchmarkKey")) in eval_summary_ids
        }
        missing_eval_summary_ids = sorted(
            expected_eval_summary_ids - actual_eval_summary_ids
        )
        if missing_eval_summary_ids:
            errors.append(
                f"{model_path.name} missing hierarchy nodes for model-card top_benchmark_scores: {missing_eval_summary_ids[:10]}"
            )

        card_categories = sorted(
            as_string(category)
            for category in model_card.get("categories_covered") or []
            if as_string(category)
        )
        if card_categories != hierarchy_categories:
            errors.append(
                f"{model_path.name} model-card categories_covered mismatch: card={card_categories} actual={hierarchy_categories}"
            )

    # Scan text outputs for accidental upstream-repo URL leakage (the
    # published dataset shouldn't reference evaleval/EEE_datastore in
    # rendered URLs). Parquet emissions are binary and outside this
    # contract's scope; skipping them lets parity parquet files coexist
    # with the validator.
    text_extensions = (".json", ".md", ".txt", ".csv", ".tsv", ".html", ".jsonl")
    for relative_path in iter_output_relative_files(output_dir):
        if not relative_path.endswith(text_extensions):
            continue
        text = (output_dir / relative_path).read_text(encoding="utf-8")
        if EEE_DATASET_REPO in text:
            errors.append(f"{relative_path} still contains {EEE_DATASET_REPO}")

    if errors:
        raise RuntimeError(
            "Output contract validation failed:\n- " + "\n- ".join(errors[:50])
        )


def attach_variant_signals(metric_summary: dict) -> None:
    """Compute variant_divergence per metric_summary group (intra-summary).

    Variant divergence asks: "for the same model+benchmark+metric, did
    different generation_args (temperature, max_tokens, …) produce
    different scores?". The natural unit is intra-source — comparing
    runs from the same evaluator with different setup. Cross-source
    comparison would conflate variant effects with cross-party effects,
    so this signal stays at the per-metric_summary grouping.

    Provenance and cross_party_divergence have moved to
    ``attach_canonical_signals`` which groups by canonical_id (with
    family_key fallback) — that's the (model, benchmark, metric)
    unit the spec actually intends, and unlike per-metric_summary
    grouping it can detect multi-source / cross-party divergence
    across suites. Variant stays here.
    """
    metric_summary_id = as_string(metric_summary.get("metric_summary_id"))
    metric_config = metric_summary.get("metric_config")

    rows = metric_summary.get("model_results") or []
    grouped: dict[str, list[dict]] = defaultdict(list)
    group_order: list[str] = []
    for row in rows:
        route = as_string(row.get("model_route_id"))
        if route not in grouped:
            group_order.append(route)
        grouped[route].append(row)

    variant_signal_groups: list[dict] = []
    for route in group_order:
        group_rows = grouped[route]
        group_id = f"{route}__{metric_summary_id}"

        projected = [
            {
                "score": row.get("score"),
                "evaluation_id": row.get("evaluation_id"),
                "source_metadata": row.get("source_metadata"),
                "generation_args": row.get("_generation_args"),
                "variant_key": row.get("variant_key"),
                "model_route_id": row.get("model_route_id"),
            }
            for row in group_rows
        ]

        variant_signal = signals.compute_variant_divergence(
            projected, metric_config, group_id=group_id
        )

        for row in group_rows:
            annotations = row.setdefault("evalcards", {}).setdefault("annotations", {})
            if variant_signal is None:
                annotations["variant_divergence"] = None
            else:
                row_variant = dict(variant_signal)
                row_variant["this_triple_score"] = row.get("score")
                annotations["variant_divergence"] = row_variant

        variant_signal_groups.append(
            {
                "group_id": group_id,
                "model_route_id": route,
                "variant_divergence": variant_signal,
            }
        )

    metric_summary["_variant_signal_groups"] = variant_signal_groups


# Backwards-compat alias removed in this refactor.


def _iter_summary_metrics(summary: dict):
    """Yield root metrics + subtask metrics for one eval_summary."""
    for metric in summary.get("metrics") or []:
        yield metric
    for subtask in summary.get("subtasks") or []:
        for metric in subtask.get("metrics") or []:
            yield metric


def _summary_canonical_grouping_key(summary: dict) -> str:
    """Return the most-specific identity key available for grouping
    a summary's rows. Prefers the registry's canonical_benchmark_id —
    that's what enables cross-suite collapse (e.g., the four
    ``helm_*_mmlu`` summaries all share canonical ``mmlu`` and bucket
    together). When the registry doesn't resolve this benchmark, fall
    back to the eval_summary_id so rows stay grouped at the
    (suite, leaf) granularity legacy used. The benchmark_family_key
    is too coarse as a fallback (aggregator suites like ``llm-stats``
    have many distinct benchmarks under one family_key — collapsing
    them would invent multi_source where the data has none)."""
    canonical_id = as_string(summary.get("canonical_benchmark_id"))
    if canonical_id:
        return canonical_id
    eval_summary_id = as_string(summary.get("eval_summary_id"))
    if eval_summary_id:
        return f"summary:{eval_summary_id}"
    family_key = as_string(summary.get("benchmark_family_key"))
    if family_key:
        return f"family:{family_key}"
    return ""


def attach_canonical_signals(eval_summaries: list[dict]) -> None:
    """Compute provenance and cross_party_divergence under two
    different (intentional) grouping schemes:

    - **Provenance / multi_source** uses the (model, benchmark) unit:
      ``(canonical_or_family_key, model_route_id)``. Counting parties
      shouldn't depend on which metric they reported — if HELM measures
      gpt-4o on MMLU with ``exact_match`` and HuggingFace measures the
      same with ``accuracy``, that's still 2 parties on the same
      benchmark and should count as multi-source. Frontend definition
      ("(model, benchmark) groups have reports from more than one
      party") matches this grouping.

    - **Cross-party divergence** uses the (model, benchmark, metric)
      unit: ``(canonical_or_family_key, metric_key, model_route_id)``.
      Comparing scores across different metrics is apples-to-oranges;
      keep metric_key in the grouping key.

    Per-row annotations attached:
      - ``provenance`` — from the (model, benchmark) provenance group
      - ``cross_party_divergence`` — from the (model, benchmark, metric) group

    Stashes:
      - ``_signal_groups``: per-(canonical, metric_key, route) carrying
        cross_party_divergence. One entry per metric_summary per route
        (deduped), so per-eval / per-model rollups don't double-count.
      - ``_provenance_signal_groups``: per-(canonical, route) carrying
        is_multi_source / first_party_only_in_group. Multiple
        metric_summaries within the same eval may share a provenance
        group; group_id encodes (canonical, route) so rollups can
        dedupe.
    """
    for summary in eval_summaries:
        for metric in _iter_summary_metrics(summary):
            metric["_signal_groups"] = []
            metric["_provenance_signal_groups"] = []

    # ------------------------------------------------------------------
    # Pass 1 — provenance at (canonical_or_family, model_route_id)
    # ------------------------------------------------------------------
    provenance_buckets: dict[
        tuple[str, str], list[tuple[dict, dict, dict]]
    ] = defaultdict(list)
    for summary in eval_summaries:
        bucket_key = _summary_canonical_grouping_key(summary)
        if not bucket_key:
            continue
        for metric in _iter_summary_metrics(summary):
            for row in metric.get("model_results") or []:
                route = as_string(row.get("model_route_id"))
                if not route:
                    continue
                provenance_buckets[(bucket_key, route)].append((summary, metric, row))

    multi_source_groups_count = 0
    provenance_groups_emitted = 0

    for (bucket_key, route), entries in provenance_buckets.items():
        group_id = f"prov__{route}__{bucket_key}"
        projected = [
            {
                "score": row.get("score"),
                "evaluation_id": row.get("evaluation_id"),
                "source_metadata": row.get("source_metadata"),
                "generation_args": row.get("_generation_args"),
                "variant_key": row.get("variant_key"),
                "model_route_id": row.get("model_route_id"),
            }
            for _summary, _metric, row in entries
        ]
        provenance_per_row = signals.compute_provenance(projected)

        is_multi_source = (
            bool(provenance_per_row[0]["is_multi_source"])
            if provenance_per_row
            else False
        )
        first_party_only_in_group = False
        for (_summary, _metric, row), prov in zip(entries, provenance_per_row):
            annotations = row.setdefault("evalcards", {}).setdefault(
                "annotations", {}
            )
            annotations["provenance"] = prov
            if prov.get("first_party_only"):
                first_party_only_in_group = True

        canonical_id_field = (
            None
            if bucket_key.startswith("family:") or bucket_key.startswith("summary:")
            else bucket_key
        )
        provenance_entry = {
            "group_id": group_id,
            "model_route_id": route,
            "canonical_benchmark_id": canonical_id_field,
            "is_multi_source": is_multi_source,
            "first_party_only_in_group": first_party_only_in_group,
        }
        # Stash on every contributing metric_summary so per-eval rollups
        # can collect; rollup-side dedup by group_id keeps the count
        # correct when multiple metric_summaries share a provenance
        # group (different metrics on the same model+benchmark).
        seen_metric_ids: set[int] = set()
        for _summary, metric, _row in entries:
            if id(metric) in seen_metric_ids:
                continue
            seen_metric_ids.add(id(metric))
            metric["_provenance_signal_groups"].append(provenance_entry)

        provenance_groups_emitted += 1
        if is_multi_source:
            multi_source_groups_count += 1

    # ------------------------------------------------------------------
    # Pass 2 — cross_party at (canonical_or_family, metric_key, route)
    # ------------------------------------------------------------------
    cross_party_buckets: dict[
        tuple[str, str], list[tuple[dict, dict]]
    ] = defaultdict(list)
    for summary in eval_summaries:
        bucket_key = _summary_canonical_grouping_key(summary)
        if not bucket_key:
            continue
        for metric in _iter_summary_metrics(summary):
            metric_key = as_string(metric.get("metric_key"))
            if not metric_key:
                continue
            cross_party_buckets[(bucket_key, metric_key)].append((summary, metric))

    cross_party_groups_emitted = 0
    cross_party_eligible_count = 0
    cross_party_divergent_count = 0

    for (bucket_key, metric_key), entries in cross_party_buckets.items():
        metric_config = entries[0][1].get("metric_config")

        rows_by_route: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        route_order: list[str] = []
        for _summary, metric in entries:
            for row in metric.get("model_results") or []:
                route = as_string(row.get("model_route_id"))
                if not route:
                    continue
                if route not in rows_by_route:
                    route_order.append(route)
                rows_by_route[route].append((metric, row))

        for route in route_order:
            group_rows = rows_by_route[route]
            group_id = f"{route}__{bucket_key}__{metric_key}"
            projected = [
                {
                    "score": row.get("score"),
                    "evaluation_id": row.get("evaluation_id"),
                    "source_metadata": row.get("source_metadata"),
                    "generation_args": row.get("_generation_args"),
                    "variant_key": row.get("variant_key"),
                    "model_route_id": row.get("model_route_id"),
                }
                for _metric, row in group_rows
            ]
            cross_party_signal = signals.compute_cross_party_divergence(
                projected, metric_config, group_id=group_id
            )

            for _metric, row in group_rows:
                annotations = row.setdefault("evalcards", {}).setdefault(
                    "annotations", {}
                )
                annotations["cross_party_divergence"] = cross_party_signal

            canonical_id_field = (
                None
                if bucket_key.startswith("family:") or bucket_key.startswith("summary:")
                else bucket_key
            )
            seen_metric_ids: set[int] = set()
            for metric, _row in group_rows:
                if id(metric) in seen_metric_ids:
                    continue
                seen_metric_ids.add(id(metric))
                metric["_signal_groups"].append(
                    {
                        "group_id": group_id,
                        "model_route_id": route,
                        "canonical_benchmark_id": canonical_id_field,
                        "metric_key": metric_key,
                        "cross_party_divergence": cross_party_signal,
                    }
                )

            cross_party_groups_emitted += 1
            if cross_party_signal and isinstance(cross_party_signal, dict):
                cross_party_eligible_count += 1
                if cross_party_signal.get("has_cross_party_divergence"):
                    cross_party_divergent_count += 1

    print(
        f"[pipeline] {json.dumps({'event': 'registry.canonical_signals', 'provenance_groups': provenance_groups_emitted, 'multi_source_groups': multi_source_groups_count, 'cross_party_groups': cross_party_groups_emitted, 'cross_party_eligible': cross_party_eligible_count, 'cross_party_divergent': cross_party_divergent_count})}"
    )


def collect_signal_rollup_inputs(
    metrics: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Walk metrics and aggregate row + group annotations.

    Returns ``(row_repro, row_provenance, variant_groups, signal_groups, provenance_groups)``:
      - ``row_repro`` / ``row_provenance``: per-row annotation dicts.
      - ``variant_groups``: per-metric_summary intra-source groups
        (carry ``variant_divergence``).
      - ``signal_groups``: per-(canonical, metric_key, route) — carry
        ``cross_party_divergence``. The (model, benchmark, metric) unit.
      - ``provenance_groups``: per-(canonical, route) — carry
        ``is_multi_source`` + ``first_party_only_in_group``. The (model,
        benchmark) unit. May appear on multiple metric_summaries within
        the same eval (one per metric); rollup-side de-duplication by
        ``group_id`` is the caller's responsibility.
    """
    row_repro_annotations: list[dict] = []
    row_provenance_annotations: list[dict] = []
    variant_signal_groups: list[dict] = []
    signal_groups: list[dict] = []
    provenance_signal_groups_by_id: dict[str, dict] = {}
    for metric in metrics:
        variant_signal_groups.extend(metric.get("_variant_signal_groups") or [])
        signal_groups.extend(metric.get("_signal_groups") or [])
        for entry in metric.get("_provenance_signal_groups") or []:
            gid = entry.get("group_id")
            if gid and gid not in provenance_signal_groups_by_id:
                provenance_signal_groups_by_id[gid] = entry
        for row in metric.get("model_results", []):
            annotations = (row.get("evalcards") or {}).get("annotations") or {}
            repro = annotations.get("reproducibility_gap")
            if repro is not None:
                row_repro_annotations.append(repro)
            prov = annotations.get("provenance")
            if prov is not None:
                row_provenance_annotations.append(prov)
    return (
        row_repro_annotations,
        row_provenance_annotations,
        variant_signal_groups,
        signal_groups,
        list(provenance_signal_groups_by_id.values()),
    )


def build_benchmark_comparability(
    variant_groups: list[dict], signal_groups: list[dict]
) -> dict:
    """Build the ``benchmark_comparability`` annotation block for an
    eval_summary header. Variant groups come from the per-metric_summary
    pass (intra-source); signal_groups come from the canonical-or-family
    pass (cross-source / cross-party). Output schema preserves the
    original two-bucket shape (variant_divergence_groups +
    cross_party_divergence_groups)."""
    variant_divergence_groups = [
        {
            "group_id": g["group_id"],
            "model_route_id": g["model_route_id"],
            "divergence_magnitude": g["variant_divergence"]["divergence_magnitude"],
            "threshold_used": g["variant_divergence"]["threshold_used"],
            "threshold_basis": g["variant_divergence"].get("threshold_basis"),
            "differing_setup_fields": g["variant_divergence"]["differing_setup_fields"],
        }
        for g in variant_groups
        if g.get("variant_divergence")
        and g["variant_divergence"].get("has_variant_divergence")
    ]
    cross_party_divergence_groups = [
        {
            "group_id": g["group_id"],
            "model_route_id": g["model_route_id"],
            "canonical_benchmark_id": g.get("canonical_benchmark_id"),
            "metric_key": g.get("metric_key"),
            "divergence_magnitude": g["cross_party_divergence"]["divergence_magnitude"],
            "threshold_used": g["cross_party_divergence"]["threshold_used"],
            "threshold_basis": g["cross_party_divergence"].get("threshold_basis"),
            "scores_by_organization": g["cross_party_divergence"][
                "scores_by_organization"
            ],
            "differing_setup_fields": g["cross_party_divergence"][
                "differing_setup_fields"
            ],
        }
        for g in signal_groups
        if g.get("cross_party_divergence")
        and g["cross_party_divergence"].get("has_cross_party_divergence")
    ]
    return {
        "variant_divergence_groups": variant_divergence_groups,
        "cross_party_divergence_groups": cross_party_divergence_groups,
    }


def summarize_comparability_combined(
    variant_groups: list[dict], signal_groups: list[dict]
) -> dict:
    """Combined rollup over two grouping schemes. Variant counts come
    from the intra-source per-metric_summary groups; cross-party counts
    come from the canonical-or-summary-key groups. The two are
    summarized separately and merged into one shape so the eval_summary
    header keeps a single ``comparability_summary`` field. ``total_groups``
    is the sum of both grouping schemes' counts (matches what
    ``aggregate_comparability`` produces at the corpus level when given
    the same combined input)."""
    variant_summary = signals.summarize_comparability(variant_groups)
    canonical_summary = signals.summarize_comparability(signal_groups)
    return {
        "total_groups": variant_summary.get("total_groups", 0)
        + canonical_summary.get("total_groups", 0),
        "groups_with_variant_check": variant_summary.get(
            "groups_with_variant_check", 0
        ),
        "variant_divergent_count": variant_summary.get("variant_divergent_count", 0),
        "groups_with_cross_party_check": canonical_summary.get(
            "groups_with_cross_party_check", 0
        ),
        "cross_party_divergent_count": canonical_summary.get(
            "cross_party_divergent_count", 0
        ),
    }


def _dedup_signal_groups(
    group_lists: list[list[dict]],
    key_fields: tuple[str, ...] = ("group_id",),
) -> list[dict]:
    """Concat signal-group lists and dedup by the join of ``key_fields``.

    Cross-party (Signal 4) and provenance (Signal 2) groups are keyed by
    ``group_id`` — already canonical-scoped, so the same ``group_id``
    appearing in multiple contributing sources represents the same group
    and should fold to one. Variant (Signal 3) groups are per-source
    (per-metric_summary intra-source); their ``group_id`` differs across
    sources but their ``model_route_id`` is what matters for the merged
    canonical view, so dedup by route.
    """
    seen: dict[str, dict] = {}
    counter = 0
    for groups in group_lists:
        for entry in groups or []:
            parts = [as_string(entry.get(field)) for field in key_fields]
            key = "::".join(parts) if any(parts) else f"_anon::{counter}"
            counter += 1
            if key not in seen:
                seen[key] = entry
    return list(seen.values())


def _merge_metrics_by_key(metric_groups: list[list[dict]]) -> list[dict]:
    """Union metrics across sources, keying by ``metric_key``.

    Same ``metric_key`` from different sources (e.g. both report ``acc`` for
    GPQA) gets concatenated ``model_results`` — no dedup; each evaluator's
    measurement is a real, distinct data point. Different ``metric_key``s
    (e.g. ``acc_cot`` vs ``acc_no_cot``) stay as separate metrics in the
    canonical record so Signal 3 (variant divergence) surfaces them.
    """
    by_key: dict[str, list[dict]] = defaultdict(list)
    for metrics in metric_groups:
        for metric in metrics or []:
            key = (
                as_string(metric.get("metric_key"))
                or as_string(metric.get("metric_id"))
                or as_string(metric.get("metric_name"))
                or "?"
            )
            by_key[key].append(metric)

    merged: list[dict] = []
    for key, contributors in by_key.items():
        # Use first contributor as the structural template; only the per-row
        # data varies by source.
        sample = dict(contributors[0])
        union_results: list[dict] = []
        for metric in contributors:
            union_results.extend(metric.get("model_results") or [])
        sample["model_results"] = union_results
        unique_routes = {
            as_string(r.get("model_route_id"))
            for r in union_results
            if as_string(r.get("model_route_id"))
        }
        sample["models_count"] = len(unique_routes)
        scores = [r.get("score") for r in union_results if r.get("score") is not None]
        sample["top_score"] = max(scores) if scores else None
        sample["_variant_signal_groups"] = _dedup_signal_groups(
            [m.get("_variant_signal_groups") or [] for m in contributors],
            key_fields=("model_route_id",),
        )
        sample["_signal_groups"] = _dedup_signal_groups(
            [m.get("_signal_groups") or [] for m in contributors]
        )
        sample["_provenance_signal_groups"] = _dedup_signal_groups(
            [m.get("_provenance_signal_groups") or [] for m in contributors]
        )
        merged.append(sample)
    return merged


def _merge_subtasks_by_key(subtask_groups: list[list[dict]]) -> list[dict]:
    """Union subtasks across sources by ``subtask_key``. Within each matched
    subtask, recursively union its metrics."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for subtasks in subtask_groups:
        for subtask in subtasks or []:
            key = (
                as_string(subtask.get("subtask_key"))
                or as_string(subtask.get("subtask_name"))
                or as_string(subtask.get("display_name"))
                or "?"
            )
            by_key[key].append(subtask)

    merged: list[dict] = []
    for key, contributors in by_key.items():
        sample = dict(contributors[0])
        sample["metrics"] = _merge_metrics_by_key(
            [st.get("metrics") or [] for st in contributors]
        )
        merged.append(sample)
    return merged


def _is_example_eval_summary(summary: dict) -> bool:
    """Mirror ``parity_outputs._is_example_eval_entry``: treat any record
    whose ``source_data.hf_repo`` starts with ``example://`` as a demo
    fixture (not real eval data). The parity layer has always filtered
    these out of ``eval_list.parquet`` / ``eval_list_lite.parquet`` at
    emission time; we apply the same filter once upstream at
    ``catalog_eval_summaries`` construction so eval-hierarchy and the
    catalog parquets share one view (otherwise hierarchy includes
    fixture rows the catalog can't render → ghost families like
    ``theory_of_mind``)."""
    source_data = summary.get("source_data") or {}
    hf_repo = source_data.get("hf_repo")
    return isinstance(hf_repo, str) and hf_repo.startswith("example://")


def build_canonical_union_eval_summaries(
    eval_summaries: list[dict],
) -> tuple[list[dict], dict[str, str]]:
    """Build canonical-union eval_summaries and a source→canonical id map.

    For each ``canonical_benchmark_id`` with two or more contributing source
    eval_summaries, emit one merged eval_summary record whose ``model_results``
    are the union of all contributing rows (no dedup — each source is a real,
    distinct measurement) and whose signal rollups are recomputed over that
    union. Single-source canonicals don't need merging — there's only one
    contributor — so they're skipped.

    The catalog (``eval-list.json``, ``eval_list.parquet``, the standalone
    ``GPQA``-style tile) reads the canonical-union record. Per-source records
    survive in ``output/evals/<source_id>.json`` for drilldowns and in
    ``model_summaries`` so model cards keep per-source attribution.

    Returns ``(canonical_records, source_to_canonical_id_map)``.
    """
    by_canonical: dict[str, list[dict]] = defaultdict(list)
    for summary in eval_summaries:
        if _is_example_eval_summary(summary):
            continue
        canonical_id = as_string(summary.get("canonical_benchmark_id"))
        if canonical_id:
            by_canonical[canonical_id].append(summary)

    canonical_records: list[dict] = []
    source_to_canonical_id: dict[str, str] = {}
    for canonical_id, contributors in by_canonical.items():
        if len(contributors) < 2:
            continue

        # Use the contributor with the most models as a structural template
        # (carries plausible defaults for benchmark_card, source_data, tags,
        # category, etc.) and overlay the merged data on top.
        base = max(contributors, key=lambda s: s.get("models_count", 0) or 0)
        canonical_summary_id = f"canonical__{canonical_id}"
        display = (
            registry.get_canonical_display_name(canonical_id)
            or as_string(base.get("display_name"))
            or canonical_id
        )

        merged_metrics = _merge_metrics_by_key(
            [s.get("metrics") or [] for s in contributors]
        )
        merged_subtasks = _merge_subtasks_by_key(
            [s.get("subtasks") or [] for s in contributors]
        )

        all_routes: set[str] = set()
        all_scores: list[float] = []
        for metric in merged_metrics:
            for row in metric.get("model_results") or []:
                rid = as_string(row.get("model_route_id"))
                if rid:
                    all_routes.add(rid)
                if row.get("score") is not None:
                    all_scores.append(row["score"])
        for subtask in merged_subtasks:
            for metric in subtask.get("metrics") or []:
                for row in metric.get("model_results") or []:
                    rid = as_string(row.get("model_route_id"))
                    if rid:
                        all_routes.add(rid)

        all_metric_pool: list[dict] = list(merged_metrics)
        for subtask in merged_subtasks:
            all_metric_pool.extend(subtask.get("metrics") or [])
        (
            row_repro,
            row_provenance,
            variant_groups,
            signal_groups,
            provenance_groups,
        ) = collect_signal_rollup_inputs(all_metric_pool)

        canonical = dict(base)
        canonical["eval_summary_id"] = canonical_summary_id
        canonical["benchmark"] = display
        #canonical["benchmark_family_key"] = canonical_id
        #canonical["benchmark_family_name"] = display
        
       # Family-key routing: cross-source canonicals (rebroadcast by 2+ EEE
        # configs — gpqa, mmlu-pro, finance-agent) lift to their canonical id
        # as a standalone family, since no single suite "owns" them. Single-
        # source canonicals (vals-index, sage-vals, vals-multimodal-index —
        # all contributors come from one EEE config) stay routed under that
        # source's suite family so the catalog presents them in their natural
        # context (Vals Index belongs under Vals AI, not as a sibling of GPQA).
        contributor_source_configs = {
            normalize_benchmark_key(as_string(c.get("_eee_source_config")))
            for c in contributors
            if as_string(c.get("_eee_source_config"))
        }
        if len(contributor_source_configs) == 1:
            source_family_key = contributor_source_configs.pop()
            source_family_name = (
                as_string(base.get("benchmark_family_name"))
                if as_string(base.get("benchmark_family_key"))
                == source_family_key
                else humanize_slug(source_family_key)
            )
            canonical["benchmark_family_key"] = source_family_key
            canonical["benchmark_family_name"] = (
                source_family_name or humanize_slug(source_family_key)
            )
        else:
            canonical["benchmark_family_key"] = canonical_id
            canonical["benchmark_family_name"] = display
        canonical["benchmark_parent_key"] = canonical_id
        canonical["benchmark_parent_name"] = display
        canonical["benchmark_leaf_key"] = canonical_id
        canonical["benchmark_leaf_name"] = display
        canonical["benchmark_component_key"] = canonical_id
        canonical["benchmark_component_name"] = display
        canonical["evaluation_name"] = display
        canonical["display_name"] = display
        canonical["canonical_display_name"] = display
        canonical["canonical_benchmark_id"] = canonical_id
        canonical["metrics"] = merged_metrics
        canonical["subtasks"] = merged_subtasks
        canonical["metrics_count"] = len(merged_metrics)
        canonical["subtasks_count"] = len(merged_subtasks)
        canonical["models_count"] = len(all_routes)
        canonical["top_score"] = max(all_scores) if all_scores else None
        canonical["metric_names"] = [
            as_string(m.get("metric_name")) for m in merged_metrics if m.get("metric_name")
        ]
        canonical["source_eval_summary_ids"] = sorted(
            as_string(s.get("eval_summary_id"))
            for s in contributors
            if s.get("eval_summary_id")
        )
        canonical["reporting_sources"] = canonical["source_eval_summary_ids"]
        canonical["reproducibility_summary"] = signals.summarize_reproducibility(row_repro)
        canonical["provenance_summary"] = signals.summarize_provenance(
            row_provenance, provenance_groups
        )
        canonical["comparability_summary"] = summarize_comparability_combined(
            variant_groups, signal_groups
        )
        # ``reporting_completeness`` is benchmark-card-driven and source-
        # invariant within a canonical (the card content is the same), so
        # carry the base contributor's value through unchanged. Downstream
        # consumers iterate every eval_summary and read this field
        # unconditionally; without it the canonical record would crash
        # callers expecting the full annotations contract.
        base_annotations = (base.get("evalcards") or {}).get("annotations") or {}
        canonical["evalcards"] = {
            "annotations": {
                "reporting_completeness": base_annotations.get(
                    "reporting_completeness"
                ),
                "benchmark_comparability": build_benchmark_comparability(
                    variant_groups, signal_groups
                ),
            }
        }
        # source_data is per-source upstream; clear it on the canonical so
        # consumers don't read a misleading single-source attribution.
        canonical["source_data"] = None

        canonical_records.append(canonical)
        for contributor in contributors:
            src_id = as_string(contributor.get("eval_summary_id"))
            if src_id:
                source_to_canonical_id[src_id] = canonical_summary_id

    return canonical_records, source_to_canonical_id


def filter_metric_summary_for_model(
    metric_summary: dict, family_id: str
) -> dict | None:
    model_results = [
        row
        for row in metric_summary.get("model_results", [])
        if as_string(row.get("model_id")) == family_id
    ]
    if not model_results:
        return None

    filtered = {
        key: value for key, value in metric_summary.items() if key != "model_results"
    }
    filtered["model_results"] = model_results
    model_route_ids = {
        as_string(row.get("model_route_id"))
        for row in model_results
        if as_string(row.get("model_route_id"))
    }
    if "_signal_groups" in filtered:
        filtered["_signal_groups"] = [
            group
            for group in filtered.get("_signal_groups") or []
            if as_string(group.get("model_route_id")) in model_route_ids
        ]
    if "_variant_signal_groups" in filtered:
        filtered["_variant_signal_groups"] = [
            group
            for group in filtered.get("_variant_signal_groups") or []
            if as_string(group.get("model_route_id")) in model_route_ids
        ]
    if "_provenance_signal_groups" in filtered:
        filtered["_provenance_signal_groups"] = [
            group
            for group in filtered.get("_provenance_signal_groups") or []
            if as_string(group.get("model_route_id")) in model_route_ids
        ]
    filtered["models_count"] = len(model_results)
    filtered["top_score"] = model_results[0].get("score") if model_results else None
    return filtered


def filter_eval_summary_for_model(summary: dict, family_id: str) -> dict | None:
    root_metrics = []
    for metric in summary.get("metrics", []):
        filtered_metric = filter_metric_summary_for_model(metric, family_id)
        if filtered_metric:
            root_metrics.append(filtered_metric)

    subtasks = []
    for subtask in summary.get("subtasks", []):
        subtask_metrics = []
        for metric in subtask.get("metrics", []):
            filtered_metric = filter_metric_summary_for_model(metric, family_id)
            if filtered_metric:
                subtask_metrics.append(filtered_metric)
        if subtask_metrics:
            subtasks.append(
                {
                    **subtask,
                    "metrics": subtask_metrics,
                    "metrics_count": len(subtask_metrics),
                    "metric_names": [
                        as_string(metric.get("metric_name"))
                        for metric in subtask_metrics
                        if as_string(metric.get("metric_name"))
                    ],
                }
            )

    if not root_metrics and not subtasks:
        return None

    filtered = {
        key: value
        for key, value in summary.items()
        if key
        not in {"metrics", "subtasks", "instance_data", "models_count", "top_score"}
    }
    filtered["metrics"] = root_metrics
    filtered["subtasks"] = subtasks
    filtered["subtasks_count"] = len(subtasks)
    filtered["metrics_count"] = len(root_metrics) + sum(
        len(subtask.get("metrics", [])) for subtask in subtasks
    )
    filtered["metric_names"] = sorted(
        {as_string(metric.get("metric_name")) for metric in root_metrics}
        | {
            as_string(metric.get("metric_name"))
            for subtask in subtasks
            for metric in subtask.get("metrics", [])
        }
        - {""}
    )
    primary_metrics = root_metrics or (
        subtasks[0].get("metrics", []) if subtasks else []
    )
    filtered["primary_metric_name"] = (
        as_string(primary_metrics[0].get("metric_name")) if primary_metrics else None
    )
    filtered["models_count"] = 1
    filtered["top_score"] = (
        primary_metrics[0].get("top_score")
        if len(primary_metrics) == 1 and not subtasks
        else None
    )

    filtered_metrics: list[dict] = list(root_metrics)
    for subtask in subtasks:
        filtered_metrics.extend(subtask.get("metrics", []))
    (
        row_repro_annotations,
        row_provenance_annotations,
        variant_signal_groups,
        signal_groups,
        provenance_signal_groups,
    ) = collect_signal_rollup_inputs(filtered_metrics)
    filtered["reproducibility_summary"] = signals.summarize_reproducibility(
        row_repro_annotations
    )
    filtered["provenance_summary"] = signals.summarize_provenance(
        row_provenance_annotations, provenance_signal_groups
    )
    filtered["comparability_summary"] = summarize_comparability_combined(
        variant_signal_groups, signal_groups
    )
    original_annotations = (summary.get("evalcards") or {}).get("annotations") or {}
    filtered_annotations = dict(original_annotations)
    filtered_annotations["benchmark_comparability"] = build_benchmark_comparability(
        variant_signal_groups, signal_groups
    )
    filtered["evalcards"] = {"annotations": filtered_annotations}

    instance_urls: set[str] = set()
    models_with_instance = 0
    for metric in root_metrics:
        for row in metric.get("model_results", []):
            url = as_string(row.get("detailed_evaluation_results"))
            if url:
                instance_urls.add(url)
            if row.get("instance_level_data") is not None:
                models_with_instance += 1
    for subtask in subtasks:
        for metric in subtask.get("metrics", []):
            for row in metric.get("model_results", []):
                url = as_string(row.get("detailed_evaluation_results"))
                if url:
                    instance_urls.add(url)
                if row.get("instance_level_data") is not None:
                    models_with_instance += 1

    filtered["instance_data"] = {
        "available": bool(instance_urls),
        "url_count": len(instance_urls),
        "sample_urls": sorted(instance_urls)[:3],
        "models_with_loaded_instances": models_with_instance,
    }
    return filtered


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    load_instance_in_dry_run = os.environ.get("LOAD_INSTANCE_IN_DRY_RUN") == "1"
    config_batch_size = parse_positive_int(os.environ.get("CONFIG_BATCH_SIZE"), 4)
    config_limit = os.environ.get("CONFIG_LIMIT")
    explicit_configs = [
        c.strip()
        for c in as_string(
            os.environ.get("CONFIGS") or os.environ.get("CONFIG_NAMES")
        ).split(",")
        if c.strip()
    ]
    configured_local_dataset_dir = (
        as_string(os.environ.get("EEE_LOCAL_DATASET_DIR")).strip()
        or DEFAULT_LOCAL_DATASET_DIR
    )
    configured_local_metadata_dir = (
        as_string(os.environ.get("BENCHMARK_METADATA_LOCAL_DIR")).strip()
        or DEFAULT_LOCAL_BENCHMARK_METADATA_DIR
    )
    force_refresh_snapshot = os.environ.get("EEE_REFRESH_SNAPSHOT") == "1"
    force_refresh_metadata = os.environ.get("BENCHMARK_METADATA_REFRESH") == "1"
    allow_skipped_configs = os.environ.get("ALLOW_SKIPPED_CONFIGS") == "1"
    hf_token = os.environ.get("HF_TOKEN")

    local_dataset_dir = ensure_local_dataset_snapshot(
        configured_local_dataset_dir,
        hf_token,
        force_refresh_snapshot,
        EEE_DATASET_REPO,
    )
    local_metadata_dir = ensure_local_benchmark_metadata_snapshot(
        configured_local_metadata_dir,
        hf_token,
        force_refresh_metadata,
        BENCHMARK_METADATA_DATASET_REPO,
    )
    if not local_metadata_dir:
        raise RuntimeError(
            "Failed to cache benchmark metadata from evaleval/auto-benchmarkcards"
        )

    # Pick up any per-run override of `CARD_BACKEND_OUTPUT_REPO` BEFORE
    # any URL-emitting code runs. Without this re-bind, the README and
    # the URL prefix guard in `validate_output_contract` would silently
    # use the import-time value.
    reload_dataset_target()

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    random.seed(42)
    load_metric_registry(DEFAULT_METRIC_REGISTRY_PATH)

    clean_output_dir()
    print(
        f"[pipeline] {json.dumps({'event': 'metric_registry.loaded', 'registry_path': str(DEFAULT_METRIC_REGISTRY_PATH), 'entry_count': len(METRIC_REGISTRY_ENTRIES), 'alias_count': len(METRIC_REGISTRY_ALIAS_LOOKUP)})}"
    )
    cards, metadata_lookup, benchmark_metadata = load_benchmark_metadata_from_dir(
        Path(local_metadata_dir)
    )
    print(
        f"[pipeline] {json.dumps({'event': 'metadata.loaded', 'benchmark_card_count': len(cards), 'metadata_key_count': len(metadata_lookup), 'metadata_cache_dir': local_metadata_dir, 'metadata_repo': BENCHMARK_METADATA_DATASET_REPO})}"
    )

    all_configs = explicit_configs or discover_configs(
        local_dataset_dir, hf_token, EEE_DATASET_REPO
    )
    ignored_present = [c for c in all_configs if c in IGNORED_CONFIGS]
    if ignored_present:
        print(
            f"[pipeline] {json.dumps({'event': 'config.ignored', 'configs': ignored_present, 'reason': 'upstream_data_quality'})}"
        )
        all_configs = [c for c in all_configs if c not in IGNORED_CONFIGS]
    if config_limit:
        all_configs = all_configs[
            : max(
                1,
                min(
                    parse_positive_int(config_limit, len(all_configs)), len(all_configs)
                ),
            )
        ]

    skipped_configs: list[str] = []
    evaluations: list[dict] = []

    for i in range(0, len(all_configs), config_batch_size):
        batch = all_configs[i : i + config_batch_size]
        print(
            f"[pipeline] {json.dumps({'event': 'config.batch.start', 'batch_index': i // config_batch_size, 'batch_size': len(batch), 'configs': batch})}"
        )

        for config in batch:
            try:
                files = list_json_files_for_config(
                    config, local_dataset_dir, hf_token, EEE_DATASET_REPO
                )
                print(
                    f"[pipeline] {json.dumps({'event': 'config.discovery', 'config': config, 'data_json_files_found': len(files), 'discovery_pages': 1, 'discovery_error': None})}"
                )
                loaded_rows = 0
                failed_files: list[str] = []
                for dataset_path in files:
                    record = None
                    last_error = None
                    for attempt in range(1, FILE_READ_MAX_RETRIES + 1):
                        try:
                            record = read_dataset_json(
                                dataset_path,
                                local_dataset_dir,
                                hf_token,
                                EEE_DATASET_REPO,
                            )
                            break
                        except Exception as error:
                            last_error = error
                            if attempt < FILE_READ_MAX_RETRIES:
                                time.sleep(FILE_READ_RETRY_DELAY_SEC * attempt)

                    if record is None:
                        failed_files.append(dataset_path)
                        print(
                            f"[pipeline] {json.dumps({'event': 'file.load.error', 'config': config, 'path': dataset_path, 'message': str(last_error) if last_error else 'unknown'})}"
                        )
                        continue

                    source_record_url = raw_url_for_dataset_path(dataset_path)
                    eval_results = (
                        record.get("evaluation_results")
                        if isinstance(record.get("evaluation_results"), list)
                        else []
                    )
                    first_result = eval_results[0] if eval_results else None
                    benchmark = (
                        as_string(record.get("evaluation_id")).split("/")[0]
                        if record.get("evaluation_id")
                        else None
                    )
                    passthrough = {
                        k: v for k, v in record.items() if k not in KNOWN_TOP_LEVEL_KEYS
                    }
                    detailed_meta = normalize_detailed_eval_meta(
                        record.get("detailed_evaluation_results")
                    )
                    # TEMPORARY: upstream-data-quality fix — `evaluator_relationship`
                    # is mis-tagged on many EEE records. While upstream is being
                    # corrected, when PARTY_OVERRIDE_LLM_STATS_FIX=1 is set we
                    # override every llm-stats row to first_party (it's a
                    # self-reported aggregator) and every other row to
                    # third_party. Affects per-row provenance.source_type and
                    # third_party_ratio aggregates, but NOT V5 multi-source
                    # rate (which is keyed on source_organization_name, not
                    # relationship). Remove this block once upstream is fixed.
                    raw_source_metadata = record.get("source_metadata")
                    if os.environ.get("PARTY_OVERRIDE_LLM_STATS_FIX") == "1":
                        sm_override = dict(raw_source_metadata or {})
                        sm_override["evaluator_relationship"] = (
                            "first_party" if config == "llm-stats" else "third_party"
                        )
                        raw_source_metadata = sm_override

                    eval_obj = {
                        "schema_version": as_string(record.get("schema_version")),
                        "evaluation_id": as_string(record.get("evaluation_id")),
                        "retrieved_timestamp": as_string(
                            record.get("retrieved_timestamp")
                        ),
                        "benchmark": benchmark,
                        "source_data": (first_result or {}).get("source_data"),
                        "source_metadata": raw_source_metadata,
                        "eval_library": record.get("eval_library"),
                        "model_info": record.get("model_info") or {},
                        "generation_config": (first_result or {}).get(
                            "generation_config"
                        ),
                        "source_record_url": source_record_url,
                        "detailed_evaluation_results_meta": detailed_meta,
                        "detailed_evaluation_results": resolve_detailed_results_url(
                            record, source_record_url
                        ),
                        "passthrough_top_level_fields": passthrough or None,
                        "evaluation_results": eval_results,
                        "benchmark_card": None,
                        "instance_level_data": None,
                        "_raw_record_payload": record,
                        # Raw EEE config name with original punctuation
                        # (e.g. ``tau-bench-2_airline``); the registry's
                        # scoped_aliases keys preserve hyphens, so the
                        # pipeline-normalized ``benchmark_family_key``
                        # cannot be used for resolution.
                        "_eee_source_config": config,
                    }
                    evaluations.append(eval_obj)
                    loaded_rows += 1

                if failed_files:
                    message = (
                        f"Failed to load {len(failed_files)} files for config {config}"
                    )
                    print(
                        f"[pipeline] {json.dumps({'event': 'config.load.partial', 'config': config, 'row_count': loaded_rows, 'failed_files': len(failed_files), 'sample_failed_paths': failed_files[:5]})}"
                    )
                    if not allow_skipped_configs:
                        raise RuntimeError(message)

                print(
                    f"[pipeline] {json.dumps({'event': 'config.load.ok', 'config': config, 'discovered_data_json_files': len(files), 'discovery_pages': 1, 'row_count': loaded_rows})}"
                )
            except Exception as error:
                print(
                    f"[pipeline] {json.dumps({'event': 'config.load.error', 'config': config, 'message': str(error)})}"
                )
                if allow_skipped_configs:
                    print(f"Skipping config {config}: {error}", file=sys.stderr)
                    skipped_configs.append(config)
                else:
                    raise

        print(
            f"[pipeline] {json.dumps({'event': 'config.batch.done', 'batch_index': i // config_batch_size, 'cumulative_evaluations': len(evaluations), 'cumulative_skipped': len(skipped_configs)})}"
        )

    if (not dry_run) or load_instance_in_dry_run:
        with_instance = 0
        missing_instance = 0
        for idx, evaluation in enumerate(evaluations, start=1):
            instance_data = maybe_load_instance_data(
                evaluation, local_dataset_dir, hf_token
            )
            if instance_data:
                evaluation["instance_level_data"] = instance_data
                with_instance += 1
            else:
                missing_instance += 1
            if idx % 100 == 0 or idx == len(evaluations):
                print(
                    f"[pipeline] {json.dumps({'event': 'instance.batch.progress', 'processed': idx, 'total': len(evaluations), 'with_instance_data': with_instance, 'missing_instance_data': missing_instance})}"
                )
        print(
            f"[pipeline] {json.dumps({'event': 'instance.load.summary', 'total': len(evaluations), 'with_instance_data': with_instance, 'missing_instance_data': missing_instance})}"
        )

    for evaluation in evaluations:
        raw_model_info = evaluation.get("model_info") or {}
        bundle = resolve_model_identity_for_pipeline(raw_model_info)
        canon = bundle["canonical"]
        disp = bundle["display_route"]
        model_info = dict(evaluation.get("model_info") or {})
        model_info.update(
            {
                "normalized_id": canon["normalized_id"],
                "family_id": disp["family_id"],
                "family_slug": disp["family_slug"],
                "family_name": disp["family_name"],
                "variant_key": disp["variant_key"],
                "variant_label": disp["variant_label"],
                "model_route_id": disp["model_route_id"],
                "canonical_model_id": canon.get("canonical_model_id"),
                "canonical_resolution_strategy": canon.get(
                    "canonical_resolution_strategy"
                ),
                "canonical_resolution_confidence": canon.get(
                    "canonical_resolution_confidence"
                ),
            }
        )
        model_info["_model_identity_bundle"] = bundle
        evaluation["model_info"] = model_info
        # Direct per-evaluation card lookup using the EEE benchmark name —
        # behavior unchanged for any benchmark whose name matches a card
        # in `metadata_lookup`.
        evaluation["benchmark_card"] = lookup_benchmark_card(
            metadata_lookup,
            evaluation.get("benchmark"),
            canonical_benchmark_family_key(evaluation.get("benchmark")),
        )

        # Fallback ONLY when the direct lookup misses: try the shared
        # `dataset_name` from `evaluation_results[*].source_data`. This
        # covers benchmarks whose ABC card name diverges from the EEE
        # benchmark name (e.g. EEE `fibble_arena` ↔ ABC
        # `fibble_arena_daily`) — without it, `benchmark_card` stays
        # None, `top_level_benchmark_owns_slices` returns False, and
        # `classify_evaluation_result` promotes each sub-variant to its
        # own leaf instead of treating it as a slice of the family.
        #
        # Two guards keep the fallback from misfiring:
        #   1. **Shared across all results.** Aggregator records (e.g.
        #      `llm-stats` wrapping 8-12+ scraped benchmarks per record)
        #      scatter their per-result `dataset_name`s; we only use one
        #      when every result agrees, leaving `shared_dataset_name`
        #      None for aggregators.
        #   2. **Compact-key relatedness.** Even a shared dataset_name
        #      must overlap the EEE benchmark by compact-key substring
        #      (`fibble_arena` ⊂ `fibble_arena_daily` ✓). A single-record
        #      aggregator (e.g. an llm-stats record covering only
        #      `ARC-AGI v2`) has `llmstats` vs `arcagiv2` — unrelated, so
        #      we don't pull in an unrelated child card and accidentally
        #      flip every result onto the `owns_slices` path.
        if evaluation["benchmark_card"] is None:
            unique_dataset_names = {
                (r.get("source_data") or {}).get("dataset_name")
                for r in (evaluation.get("evaluation_results") or [])
                if isinstance(r, dict) and isinstance(r.get("source_data"), dict)
            }
            unique_dataset_names.discard(None)
            unique_dataset_names.discard("")
            shared_dataset_name = (
                next(iter(unique_dataset_names))
                if len(unique_dataset_names) == 1
                else None
            )
            eee_compact = compact_benchmark_key(evaluation.get("benchmark"))
            ds_compact = (
                compact_benchmark_key(shared_dataset_name)
                if shared_dataset_name
                else ""
            )
            if (
                eee_compact
                and ds_compact
                and (eee_compact in ds_compact or ds_compact in eee_compact)
            ):
                evaluation["benchmark_card"] = lookup_benchmark_card(
                    metadata_lookup,
                    evaluation.get("benchmark"),
                    canonical_benchmark_family_key(evaluation.get("benchmark")),
                    shared_dataset_name,
                )

        enriched_results = []
        for result in evaluation.get("evaluation_results") or []:
            enriched = dict(result)
            normalized = classify_evaluation_result(
                evaluation, enriched, evaluation["benchmark_card"]
            )
            enriched["normalized_result"] = normalized
            enriched_results.append(enriched)
        evaluation["evaluation_results"] = enriched_results

        evaluation["eval_summary_ids"] = sorted(
            {
                get_eval_group_id(evaluation, result)
                for result in evaluation.get("evaluation_results") or []
                if extract_score(result) is not None
            }
        )

        raw_record_payload = evaluation.pop("_raw_record_payload", None)
        record_relative_path = build_record_artifact_relative_path(evaluation)
        record_artifact_url = dataset_resolve_url(record_relative_path)
        evaluation["source_record_url"] = record_artifact_url

        raw_instance_url = as_string(evaluation.get("detailed_evaluation_results"))
        if raw_instance_url:
            # Only publish pipeline-owned instance artifact URLs. If we cannot
            # materialize the upstream samples into output/instances, clear the
            # public field instead of leaking raw source-dataset URLs.
            evaluation["detailed_evaluation_results"] = None
            artifact_text = read_text_from_dataset_url(
                raw_instance_url, local_dataset_dir, hf_token
            )
            if artifact_text is not None:
                instance_relative_path = build_instance_artifact_relative_path(
                    evaluation
                )
                instance_output_path = OUTPUT_DIR / instance_relative_path
                instance_output_path.parent.mkdir(parents=True, exist_ok=True)
                transformed_artifact_text = transform_instance_artifact_text(
                    evaluation, artifact_text
                )
                instance_output_path.write_text(
                    transformed_artifact_text, encoding="utf-8"
                )
                pipeline_instance_url = dataset_resolve_url(instance_relative_path)
                evaluation["detailed_evaluation_results"] = pipeline_instance_url
                evaluation["instance_artifact"] = {
                    "path": instance_relative_path,
                    "url": pipeline_instance_url,
                    "eval_summary_ids": evaluation.get("eval_summary_ids", []),
                }
                if evaluation.get("instance_level_data"):
                    evaluation["instance_level_data"] = {
                        **(evaluation.get("instance_level_data") or {}),
                        "source_url": pipeline_instance_url,
                        "instance_examples": [
                            annotate_instance_row(evaluation, row)
                            if isinstance(row, dict)
                            else row
                            for row in (
                                (evaluation.get("instance_level_data") or {}).get(
                                    "instance_examples"
                                )
                                or []
                            )
                        ],
                    }
            elif evaluation.get("instance_level_data"):
                evaluation["instance_level_data"] = {
                    **(evaluation.get("instance_level_data") or {}),
                    "source_url": None,
                }

        if raw_record_payload is not None:
            record_output_payload = dict(raw_record_payload)
            record_output_payload["source_record_url"] = record_artifact_url
            record_output_payload["detailed_evaluation_results"] = evaluation.get(
                "detailed_evaluation_results"
            )
            if evaluation.get("detailed_evaluation_results_meta") is not None:
                record_output_payload["detailed_evaluation_results_meta"] = (
                    evaluation.get("detailed_evaluation_results_meta")
                )
            write_json(OUTPUT_DIR / record_relative_path, record_output_payload)

    benchmark_groups: dict[str, dict] = {}
    model_family_groups: dict[str, list[dict]] = defaultdict(list)

    for evaluation in evaluations:
        family_id = as_string((evaluation.get("model_info") or {}).get("family_id"))
        if family_id:
            model_family_groups[family_id].append(evaluation)

        for result in evaluation.get("evaluation_results") or []:
            score = extract_score(result)
            if score is None:
                continue
            normalized = result.get("normalized_result") or {}
            eval_group_id = get_eval_group_id(evaluation, result)
            metric_summary_id = get_metric_summary_id(evaluation, result)
            group = benchmark_groups.setdefault(
                eval_group_id,
                {
                    "eval_summary_id": eval_group_id,
                    "benchmark": normalized.get("benchmark_parent_name")
                    or evaluation.get("benchmark"),
                    "benchmark_family_key": normalized.get("benchmark_family_key"),
                    "benchmark_family_name": normalized.get("benchmark_family_name"),
                    "benchmark_parent_key": normalized.get("benchmark_parent_key"),
                    "benchmark_parent_name": normalized.get("benchmark_parent_name"),
                    "benchmark_leaf_key": normalized.get("benchmark_leaf_key"),
                    "benchmark_leaf_name": normalized.get("benchmark_leaf_name"),
                    "benchmark_component_key": normalized.get(
                        "benchmark_component_key"
                    ),
                    "benchmark_component_name": normalized.get(
                        "benchmark_component_name"
                    ),
                    "evaluation_name": normalized.get("benchmark_leaf_name")
                    or normalized.get("benchmark_family_name"),
                    "display_name": normalized.get("benchmark_leaf_name")
                    or normalized.get("benchmark_family_name"),
                    "canonical_display_name": normalized.get("benchmark_leaf_name")
                    or normalized.get("benchmark_parent_name")
                    or normalized.get("benchmark_family_name"),
                    "is_summary_score": bool(normalized.get("is_summary_score")),
                    "category": infer_category_from_benchmark(
                        as_string(evaluation.get("benchmark"))
                    ),
                    "source_data": result.get("source_data"),
                    "benchmark_card": None,
                    "tags": {"domains": [], "languages": [], "tasks": []},
                    "subtasks": {},
                    "_source_metadata_aggregate": {},
                    "_eee_source_config": evaluation.get("_eee_source_config"),
                },
            )

            evaluation_source_metadata = evaluation.get("source_metadata")
            if isinstance(evaluation_source_metadata, dict):
                aggregate = group["_source_metadata_aggregate"]
                for sm_key, sm_value in evaluation_source_metadata.items():
                    if sm_value is not None and sm_key not in aggregate:
                        aggregate[sm_key] = sm_value

            # Set benchmark card and tags on first encounter
            if group["benchmark_card"] is None:
                _card = lookup_benchmark_card_for_parent(
                    metadata_lookup,
                    normalized.get("benchmark_leaf_name"),
                    normalized.get("benchmark_leaf_key"),
                    aux_values=(
                        evaluation.get("benchmark"),
                        normalized.get("benchmark_family_key"),
                        (result.get("source_data") or {}).get("dataset_name"),
                    ),
                    parent_values=(
                        normalized.get("benchmark_parent_key"),
                        normalized.get("benchmark_parent_name"),
                        evaluation.get("benchmark"),
                        normalized.get("benchmark_family_key"),
                    ),
                )

                # Last-resort: ABC card slugs match registry canonical_ids,
                # so when name/key/family/dataset_name lookups all miss, the
                # registry's canonical may identify the right card. E.g.
                # `vals_ai`'s benchmark_leaf_name "MMLU-Pro - Health" doesn't
                # match any direct ABC key but resolves to canonical
                # `mmlu-pro` which matches the `mmlu-pro.json` card.
                if _card is None:
                    leaf_candidates = [
                        normalized.get("benchmark_leaf_name"),
                        normalized.get("benchmark_leaf_key"),
                        normalized.get("benchmark_family_key"),
                    ]
                    src = evaluation.get("_eee_source_config")
                    for candidate in leaf_candidates:
                        if not candidate:
                            continue
                        registry_result = registry.resolve_benchmark(
                            candidate, src
                        )
                        canonical_id = (registry_result or {}).get("canonical_id")
                        if canonical_id:
                            _card = lookup_benchmark_card(
                                metadata_lookup, canonical_id
                            )
                            if _card:
                                break

                if _card:
                    group["benchmark_card"] = _card
                    group["tags"] = extract_benchmark_tags(_card)
                    # Re-derive category from card domains (more accurate than name regex)
                    group["category"] = infer_category_from_benchmark(
                        as_string(evaluation.get("benchmark")), _card
                    )

            subtask_key = as_string(normalized.get("slice_key") or "__root__")
            subtask = group["subtasks"].setdefault(
                subtask_key,
                {
                    "subtask_key": None
                    if subtask_key == "__root__"
                    else normalized.get("slice_key"),
                    "subtask_name": normalized.get("slice_name"),
                    "display_name": normalized.get("slice_name")
                    or normalized.get("benchmark_leaf_name")
                    or normalized.get("benchmark_family_name"),
                    "canonical_display_name": join_display_name_parts(
                        normalized.get("benchmark_leaf_name")
                        or normalized.get("benchmark_parent_name")
                        or normalized.get("benchmark_family_name"),
                        normalized.get("slice_name"),
                    ),
                    "metrics": {},
                },
            )

            metric_summary = subtask["metrics"].setdefault(
                metric_summary_id,
                {
                    "metric_summary_id": metric_summary_id,
                    "legacy_eval_summary_id": slugify(
                        f"{evaluation.get('benchmark') or ((result.get('source_data') or {}).get('dataset_name')) or 'unknown'}__{result.get('evaluation_name') or 'unknown'}"
                    ),
                    "evaluation_name": result.get("evaluation_name"),
                    "display_name": " / ".join(
                        [
                            part
                            for part in [
                                normalized.get("benchmark_leaf_name"),
                                normalized.get("slice_name"),
                                normalized.get("metric_name"),
                            ]
                            if part
                        ]
                    ),
                    "canonical_display_name": normalized.get("canonical_display_name"),
                    "benchmark_leaf_key": normalized.get("benchmark_leaf_key"),
                    "benchmark_leaf_name": normalized.get("benchmark_leaf_name"),
                    "slice_key": normalized.get("slice_key"),
                    "slice_name": normalized.get("slice_name"),
                    "lower_is_better": bool(
                        (result.get("metric_config") or {}).get("lower_is_better")
                    ),
                    "metric_name": normalized.get("metric_name"),
                    "metric_id": normalized.get("metric_id"),
                    "metric_key": normalized.get("metric_key"),
                    "metric_source": normalized.get("metric_source"),
                    "metric_config": result.get("metric_config"),
                    "model_results": [],
                },
            )
            generation_args = (result.get("generation_config") or {}).get(
                "generation_args"
            )
            if not isinstance(generation_args, dict):
                generation_args = None
            agentic_for_spec = signals.is_agentic(
                evaluation.get("benchmark"),
                evaluation.get("benchmark_card"),
                generation_args,
            )
            reproducibility_gap = signals.compute_reproducibility_gap(
                generation_args, agentic_for_spec
            )

            metric_summary["model_results"].append(
                {
                    "model_id": as_string(
                        (evaluation.get("model_info") or {}).get("family_id")
                    ),
                    "model_route_id": as_string(
                        (evaluation.get("model_info") or {}).get("model_route_id")
                    ),
                    "model_name": as_string(
                        (evaluation.get("model_info") or {}).get("family_name")
                        or (evaluation.get("model_info") or {}).get("name")
                    ),
                    "developer": as_string(
                        (evaluation.get("model_info") or {}).get("developer")
                    ),
                    "variant_key": as_string(
                        (evaluation.get("model_info") or {}).get("variant_key")
                    )
                    or "default",
                    "raw_model_id": as_string(
                        (evaluation.get("model_info") or {}).get("id")
                    ),
                    "score": score,
                    "evaluation_id": evaluation.get("evaluation_id"),
                    "retrieved_timestamp": evaluation.get("retrieved_timestamp"),
                    # Carry provenance straight onto each per-model row so
                    # downstream consumers don't have to re-join against the
                    # parent evaluation record. Without this, the hierarchy
                    # view loses the evaluator_relationship / source_organization
                    # context and all 1st/3rd-party badges collapse to "other".
                    "source_metadata": evaluation.get("source_metadata"),
                    "source_data": evaluation.get("source_data"),
                    "source_record_url": evaluation.get("source_record_url"),
                    "detailed_evaluation_results": evaluation.get(
                        "detailed_evaluation_results"
                    ),
                    "detailed_evaluation_results_meta": evaluation.get(
                        "detailed_evaluation_results_meta"
                    ),
                    "passthrough_top_level_fields": evaluation.get(
                        "passthrough_top_level_fields"
                    ),
                    "instance_level_data": evaluation.get("instance_level_data"),
                    "normalized_result": normalized,
                    "evalcards": {
                        "annotations": {
                            "reproducibility_gap": reproducibility_gap,
                        }
                    },
                    # Carried per row for variant_divergence comparison,
                    # cross_party representative-setup.
                    "_generation_args": generation_args,
                }
            )

    peer_ranks: dict[str, dict[str, dict[str, int]]] = {}
    eval_summaries: list[dict] = []

    for summary in benchmark_groups.values():
        root_metrics: list[dict] = []
        subtask_summaries: list[dict] = []
        model_ids_for_group: set[str] = set()
        unique_metric_names: set[str] = set()
        total_metric_count = 0

        for subtask in summary["subtasks"].values():
            metric_summaries: list[dict] = []
            for metric_summary in subtask["metrics"].values():
                lower = bool(metric_summary.get("lower_is_better"))
                model_results = sorted(
                    metric_summary["model_results"],
                    key=lambda r: (r["score"], r["model_id"]),
                )
                if not lower:
                    model_results.reverse()
                metric_summary["model_results"] = model_results
                metric_summary["models_count"] = len(model_results)
                metric_summary["top_score"] = (
                    model_results[0]["score"] if model_results else None
                )

                # Compute provenance + variant_divergence + cross_party_divergence per group,
                # and attach annotations to each row.
                # Group-level summaries are stashed on the metric_summary for downstream aggregation
                attach_variant_signals(metric_summary)

                metric_summaries.append(metric_summary)
                total_metric_count += 1
                unique_metric_names.add(as_string(metric_summary.get("metric_name")))
                model_ids_for_group.update(row["model_id"] for row in model_results)

                ranks: dict[str, dict[str, int]] = {}
                position = 0
                previous_score = None
                for idx, row in enumerate(model_results, start=1):
                    if previous_score is None or row["score"] != previous_score:
                        position = idx
                        previous_score = row["score"]
                    rank_entry = {"position": position, "total": len(model_results)}
                    ranks[row["model_id"]] = rank_entry
                    raw_id = row.get("raw_model_id", "")
                    if raw_id and raw_id != row["model_id"]:
                        ranks[raw_id] = rank_entry
                metric_summary["_ranks"] = ranks
                metric_summary["_total"] = len(model_results)

            metric_summaries.sort(
                key=lambda metric: (
                    as_string(metric.get("metric_name")),
                    as_string(metric.get("metric_summary_id")),
                )
            )
            if subtask.get("subtask_key") is None:
                root_metrics = metric_summaries
            else:
                subtask_summaries.append(
                    {
                        "subtask_key": subtask.get("subtask_key"),
                        "subtask_name": subtask.get("subtask_name"),
                        "display_name": subtask.get("display_name"),
                        "metrics": metric_summaries,
                        "metrics_count": len(metric_summaries),
                        "metric_names": [
                            as_string(metric.get("metric_name"))
                            for metric in metric_summaries
                        ],
                    }
                )

        subtask_summaries.sort(
            key=lambda subtask: as_string(subtask.get("display_name"))
        )
        summary["metrics"] = root_metrics
        summary["subtasks"] = subtask_summaries
        summary["subtasks_count"] = len(subtask_summaries)
        summary["metrics_count"] = total_metric_count
        summary["models_count"] = len(model_ids_for_group)
        summary["metric_names"] = sorted(name for name in unique_metric_names if name)
        primary_metrics = root_metrics or (
            subtask_summaries[0]["metrics"] if subtask_summaries else []
        )
        primary_metric = pick_primary_metric(primary_metrics)
        summary["primary_metric_name"] = (
            as_string(primary_metric.get("metric_name")) if primary_metric else None
        )
        summary["top_score"] = (
            primary_metric.get("top_score")
            if primary_metric and len(primary_metrics) == 1 and not subtask_summaries
            else None
        )

        # Peer ranks at single-benchmark level: average rank across all metrics
        all_metric_summaries = list(root_metrics)
        for st in subtask_summaries:
            all_metric_summaries.extend(st.get("metrics", []))
        metrics_with_ranks = [m for m in all_metric_summaries if m.get("_ranks")]
        if metrics_with_ranks:
            # Collect per-model rank positions across all metrics
            model_rank_sums: dict[str, list[float]] = defaultdict(list)
            max_total = 0
            for m in metrics_with_ranks:
                total = m.get("_total", 0)
                if total > max_total:
                    max_total = total
                for model_id, rank_info in m["_ranks"].items():
                    model_rank_sums[model_id].append(rank_info["position"])
            # Average and re-rank
            avg_ranks = []
            for model_id, positions in model_rank_sums.items():
                avg_ranks.append((sum(positions) / len(positions), model_id))
            avg_ranks.sort()
            benchmark_ranks: dict[str, dict[str, int]] = {}
            position = 0
            previous_avg = None
            for idx, (avg, model_id) in enumerate(avg_ranks, start=1):
                if previous_avg is None or avg != previous_avg:
                    position = idx
                    previous_avg = avg
                benchmark_ranks[model_id] = {
                    "position": position,
                    "total": len(avg_ranks),
                }
            peer_ranks[summary["eval_summary_id"]] = benchmark_ranks

        # Summarise instance-level data availability across all model results
        instance_urls: set[str] = set()
        models_with_instance = 0
        models_without_instance = 0
        for ms in all_metric_summaries:
            for row in ms.get("model_results", []):
                url = as_string(row.get("detailed_evaluation_results"))
                has_instance = row.get("instance_level_data") is not None
                if url:
                    instance_urls.add(url)
                if has_instance:
                    models_with_instance += 1
                elif url:
                    models_without_instance += 1
        summary["instance_data"] = {
            "available": bool(instance_urls),
            "url_count": len(instance_urls),
            "sample_urls": sorted(instance_urls)[:3],
            "models_with_loaded_instances": models_with_instance,
        }

        eval_summaries.append(summary)

    # Second pass: link summary scores to their sibling eval groups so the frontend
    # knows which groups roll up into which summary and vice-versa.
    parent_to_summary_eval_ids: dict[str, list[str]] = defaultdict(list)
    for summary in eval_summaries:
        if summary.get("is_summary_score"):
            parent_key = as_string(summary.get("benchmark_parent_key"))
            if parent_key:
                parent_to_summary_eval_ids[parent_key].append(
                    summary["eval_summary_id"]
                )

    for summary in eval_summaries:
        if summary.get("is_summary_score"):
            summary["summary_score_for"] = as_string(
                summary.get("benchmark_parent_key")
            )
            summary["summary_score_for_name"] = as_string(
                summary.get("benchmark_parent_name")
            )
        else:
            parent_key = as_string(summary.get("benchmark_parent_key"))
            sibling_summary_ids = parent_to_summary_eval_ids.get(parent_key, [])
            if sibling_summary_ids:
                summary["summary_eval_ids"] = sibling_summary_ids

    # Resolve each eval_summary against the registry so cross-suite
    # duplicates (the four ``helm_*_mmlu`` summaries, the three MMLU-Pro
    # spellings, etc.) carry a shared ``canonical_benchmark_id``. The
    # registry's scoped_aliases keys preserve original EEE punctuation,
    # so ``_eee_source_config`` (raw config name) is the lookup scope —
    # the pipeline-normalized ``benchmark_family_key`` would miss
    # hyphenated suites like ``tau-bench-2_airline``.
    canonical_resolution_count = 0
    for summary in eval_summaries:
        raw_value = (
            as_string(summary.get("benchmark_leaf_name"))
            or as_string(summary.get("benchmark_leaf_key"))
            or None
        )
        resolution = registry.resolve_benchmark(
            raw_value, summary.get("_eee_source_config")
        )
        summary["canonical_benchmark_id"] = resolution["canonical_id"]
        summary["canonical_benchmark_resolution"] = resolution
        if resolution["canonical_id"]:
            canonical_resolution_count += 1
    print(
        f"[pipeline] {json.dumps({'event': 'registry.eval_summaries_resolved', 'total': len(eval_summaries), 'resolved': canonical_resolution_count})}"
    )

    # Compute provenance + cross_party_divergence at the
    # (canonical_benchmark_id-or-family_key, metric_key, model_route_id)
    # grouping. This is the (model, benchmark, metric) unit the spec
    # intends — variant_divergence stays at the per-metric_summary
    # grouping (it's an intra-source signal). Replaces the legacy
    # canonical-only sidecar pattern; the per-row annotation field
    # names no longer carry a ``_canonical`` suffix.
    attach_canonical_signals(eval_summaries)

    # ------------------------------------------------------------------
    # HOTFIX (2026-04-30): the ABC cards for ``helm_capabilities`` and
    # ``helm-instruct`` both carry ``benchmark_details.name = "Holistic
    # Evaluation of Language Models (HELM)"`` (the umbrella project
    # name, not the suite-specific name). The pipeline propagates this
    # card name into the suite-aggregate eval_summary's display fields,
    # so two top-level eval_summaries end up with byte-identical
    # ``display_name`` and the frontend renders them as visible
    # duplicates.
    #
    # The proper fix is either (a) the deferred family-collapse work
    # that uses registry parent_benchmark_id to collapse helm_* siblings
    # into one HELM family node, or (b) fixing the ABC cards upstream.
    # Targeted post-processing rename gated on the (family_key, current
    # display_name) pair so sub-benchmark rows that share a family_key
    # but already have specific display names (e.g.
    # helm_capabilities_mmlu_pro → "MMLU-Pro") aren't touched.
    #
    # Earlier attempt: adding helm_capabilities / helm_instruct to
    # PREFERRED_BENCHMARK_DISPLAY_NAMES caused a 60-field regression —
    # that lookup is consulted from many call sites. Override lives
    # here post-construction so it applies surgically.
    #
    # See notes/upstream-data-issues.md for upstream defect tracking.
    _ABC_FAMILY_NAME_OVERRIDES = {
        # family_key: (from_string_gate, to_string)
        "helm_classic": ("Helm classic", "HELM Classic"),
        "helm_capabilities": (
            "Holistic Evaluation of Language Models (HELM)",
            "HELM Capabilities",
        ),
        "helm_instruct": (
            "Holistic Evaluation of Language Models (HELM)",
            "HELM Instruct",
        ),
        "helm_lite": ("Helm lite", "HELM Lite"),
        "helm_safety": ("Helm safety", "HELM Safety"),
        "helm_air_bench": ("Helm air bench", "HELM AIR-Bench"),
        # ABC card describes V1 mechanics under V2 canonical IDs.
        "tau_bench_2": (
            "τ-bench (Tool-Agent-User Interaction Benchmark)",
            "Tau2-Bench",
        ),
    }
    for summary in eval_summaries:
        family_key = as_string(summary.get("benchmark_family_key"))
        override = _ABC_FAMILY_NAME_OVERRIDES.get(family_key)
        if override:
            from_string, replacement = override
            # Gate on benchmark_family_name (where the ABC card's umbrella string
            # lands). display_name carries the leaf form (e.g., "Capabilities")
            # so it's the wrong field to gate on.
            if as_string(summary.get("benchmark_family_name")) == from_string:
                for field in (
                    "benchmark_family_name",
                    "benchmark_parent_name",
                    "benchmark_leaf_name",
                    "canonical_display_name",
                    "display_name",
                ):
                    if as_string(summary.get(field)) == from_string:
                        summary[field] = replacement

    # For canonical-resolved leaves on leaderboard-aggregator suites (hfopenllm_v2,
    # vals_ai, llm_stats, openeval, artificial_analysis_llms), the upstream ABC card
    # name often pollutes benchmark_family_name (e.g., "MMLU-Pro leaderboard
    # submissions (TIGER-Lab)") and the leaf-level display fields carry the raw EEE
    # form (e.g., "artificial_analysis.aime_25"). Override the display-surface
    # fields (family_name + display_name + canonical_display_name, at summary and
    # metric level) to the registry's canonical display_name so detail pages, list
    # views, model drill-downs and parquets show "MMLU-Pro" / "AIME 2025" instead.
    # Internal key-ish fields (benchmark_leaf_name, benchmark_component_name,
    # evaluation_name, metric_id) are intentionally left raw — consumers should
    # read the display fields for labels.
    # Curated suites (HELM Classic / Capabilities / Instruct / etc., tau-bench)
    # legitimately own their leaves and keep their suite-named family.
    _LEADERBOARD_AGGREGATOR_FAMILY_KEYS = {
        "hfopenllm_v2",
        "hfopenllm",
        "artificial_analysis_llms",
        "llm_stats",
        "openeval",
        "vals_ai",
    }
    for summary in eval_summaries:
        family_key = as_string(summary.get("benchmark_family_key"))
        if family_key not in _LEADERBOARD_AGGREGATOR_FAMILY_KEYS:
            continue
        canonical_id = as_string(summary.get("canonical_benchmark_id"))
        if not canonical_id:
            continue
        canonical_display = registry.get_canonical_display_name(canonical_id)
        if not canonical_display:
            continue
        # benchmark_family_name is intentionally NOT set here — see family-display
        # normalization below; per-row leaf canonicals would break the
        # per-family agreement that eval-hierarchy bucketing depends on.
        summary["display_name"] = canonical_display
        summary["canonical_display_name"] = canonical_display
        for metric in summary.get("metrics") or []:
            metric["display_name"] = join_display_name_parts(
                canonical_display, metric.get("metric_name")
            )
            metric["canonical_display_name"] = join_display_name_parts(
                canonical_display, metric.get("slice_name"), metric.get("metric_name")
            )

    # Normalize benchmark_family_name once per family_key. Without this,
    # per-row computation in classify_evaluation_result lets per-leaf signals
    # (ABC card name, dataset_name) bleed into a field that must be uniform
    # across the family — eval-hierarchy bucketing picks the first row's value
    # via setdefault, so divergent rows produce non-deterministic, leaf-named
    # family headers (e.g. OpenEval family rendered as "BBQ").
    #
    # Source ranking per family_key:
    #   1. registry canonical display (try snake → kebab forms)
    #   2. suite-aggregate row's benchmark_family_name (post-_ABC_OVERRIDES,
    #      which already cleans suite-row family_name for HELM-style curated
    #      suites). Read family_name not display_name — display_name on the
    #      suite-row is leaf-derived (e.g. "Classic" not "HELM Classic").
    #   3. humanize_token_key fallback
    suite_row_family_name_by_family: dict[str, str] = {}
    for summary in eval_summaries:
        family_key = as_string(summary.get("benchmark_family_key"))
        if not family_key:
            continue
        if as_string(summary.get("eval_summary_id")) == family_key:
            suite_family_name = as_string(summary.get("benchmark_family_name"))
            if suite_family_name:
                suite_row_family_name_by_family[family_key] = suite_family_name

    family_display_by_key: dict[str, str] = {}
    for summary in eval_summaries:
        family_key = as_string(summary.get("benchmark_family_key"))
        if not family_key:
            continue
        if family_key not in family_display_by_key:
            display = registry.get_canonical_display_name(family_key)
            if not display and "_" in family_key:
                display = registry.get_canonical_display_name(
                    family_key.replace("_", "-")
                )
            if not display:
                display = suite_row_family_name_by_family.get(family_key)
            if not display:
                display = humanize_token_key(family_key)
            family_display_by_key[family_key] = display
        summary["benchmark_family_name"] = family_display_by_key[family_key]

    family_name_check: dict[str, set[str]] = defaultdict(set)
    for summary in eval_summaries:
        family_key = as_string(summary.get("benchmark_family_key"))
        family_name = as_string(summary.get("benchmark_family_name"))
        if family_key and family_name:
            family_name_check[family_key].add(family_name)
    divergent_family_names = {
        family_key: sorted(names)
        for family_key, names in family_name_check.items()
        if len(names) > 1
    }
    if divergent_family_names:
        raise RuntimeError(
            "benchmark_family_name divergence within benchmark_family_key — "
            "eval-hierarchy bucketing requires per-family agreement. "
            f"Divergent families: {divergent_family_names}"
        )

    eval_summaries.sort(
        key=lambda s: (-s.get("models_count", 0), as_string(s.get("eval_summary_id")))
    )

    comparison_index = build_comparison_index(eval_summaries, started_at)

    # Strip temporary fields from metric summaries before serialization
    for summary in eval_summaries:
        for metric in summary.get("metrics", []):
            metric.pop("_ranks", None)
            metric.pop("_total", None)
        for subtask in summary.get("subtasks", []):
            for metric in subtask.get("metrics", []):
                metric.pop("_ranks", None)
                metric.pop("_total", None)

    # Attach interpretive-signal annotations to each eval summary:
    # - reporting_completeness at the benchmark level
    # - benchmark_comparability listing divergent groups
    # - reproducibility_summary, provenance_summary, comparability_summary
    #   aggregates over the eval's rows / groups.
    for summary in eval_summaries:
        joined_record = {
            "autobenchmarkcard": summary.get("benchmark_card") or {},
            "eee_eval": {
                "source_metadata": summary.pop("_source_metadata_aggregate", {}) or {}
            },
            "evalcards": {},
        }
        completeness = signals.compute_reporting_completeness(joined_record)

        all_metrics: list[dict] = list(summary.get("metrics", []))
        for subtask in summary.get("subtasks", []):
            all_metrics.extend(subtask.get("metrics", []))

        (
            row_repro_annotations,
            row_provenance_annotations,
            variant_signal_groups,
            signal_groups,
            provenance_signal_groups,
        ) = collect_signal_rollup_inputs(all_metrics)

        repro_summary = signals.summarize_reproducibility(row_repro_annotations)
        provenance_summary = signals.summarize_provenance(
            row_provenance_annotations, provenance_signal_groups
        )
        comparability_summary = summarize_comparability_combined(
            variant_signal_groups, signal_groups
        )

        summary["evalcards"] = {
            "annotations": {
                "reporting_completeness": completeness,
                "benchmark_comparability": build_benchmark_comparability(
                    variant_signal_groups, signal_groups
                ),
            }
        }
        summary["reproducibility_summary"] = repro_summary
        summary["provenance_summary"] = provenance_summary
        summary["comparability_summary"] = comparability_summary

    # Build canonical-union eval_summaries: 
    # for canonicals reported by 2+ sources, union their model_results and 
    # recompute signal rollups over the merged set. 
    # Must run AFTER the per-source signal annotation loop above
    # so the canonical merge can carry forward the contributors' completed
    # ``evalcards.annotations`` and recompute group rollups from each metric's
    # ``_*_signal_groups``. The catalog (eval-list) and detail-page emission
    # read these canonical records; per-source records survive in
    # eval_summaries for model_summaries / model_cards (which need per-source
    # attribution) and for output/evals/<source_id>.json drilldowns.
    canonical_eval_summaries, source_to_canonical_id = (
        build_canonical_union_eval_summaries(eval_summaries)
    )
    canonical_eval_summaries.sort(
        key=lambda s: (-s.get("models_count", 0), as_string(s.get("eval_summary_id")))
    )
    sources_in_canonical: set[str] = set(source_to_canonical_id.keys())
    catalog_eval_summaries: list[dict] = [
        s
        for s in eval_summaries
        if (
            as_string(s.get("eval_summary_id")) not in sources_in_canonical 
            and not _is_example_eval_summary(s)
        )
    ] + canonical_eval_summaries
    catalog_eval_summaries.sort(
        key=lambda s: (-s.get("models_count", 0), as_string(s.get("eval_summary_id")))
    )

    aggregated_model_family_groups: dict[str, list[dict]] = defaultdict(list)
    for family_evals in model_family_groups.values():
        for evaluation in family_evals:
            display_identity = aggregated_display_identity(
                evaluation.get("model_info") or {}
            )
            aggregated_model_family_groups[display_identity["family_id"]].append(
                evaluation
            )

    model_summaries: list[dict] = []
    model_cards: list[dict] = []

    for family_id, family_evals in aggregated_model_family_groups.items():
        family_evals_sorted = sorted(
            family_evals, key=lambda e: as_string(e.get("retrieved_timestamp"))
        )
        latest = family_evals_sorted[-1]
        model_info = latest.get("model_info") or {}
        display_identity = aggregated_display_identity(model_info)
        route_id = as_string(
            display_identity.get("model_route_id") or family_id.replace("/", "__")
        )
        family_name = as_string(
            display_identity.get("family_name")
            or model_info.get("family_name")
            or model_info.get("name")
            or family_id.split("/")[-1]
        )
        params_billions: float | None = None

        by_category: dict[str, list[dict]] = defaultdict(list)
        raw_model_ids = sorted(
            {
                as_string((e.get("model_info") or {}).get("id"))
                for e in family_evals
                if as_string((e.get("model_info") or {}).get("id"))
            }
        )
        variants_map: dict[str, dict] = {}
        score_values: list[float] = []
        last_updated = None

        # ---- FIX 2: collect per-benchmark scores for model card ----
        benchmark_names_set: set[str] = set()
        # key = eval_summary_id, value = best score entry for that metric
        best_per_metric: dict[str, dict] = {}

        for evaluation in family_evals:
            category = infer_category_from_benchmark(
                as_string(evaluation.get("benchmark"))
            )
            by_category[category].append(evaluation)
            iso = iso_from_epoch_string(evaluation.get("retrieved_timestamp"))
            last_updated = max_iso(last_updated, iso)

            if params_billions is None:
                params_billions = derive_model_params_billions(
                    evaluation.get("model_info") or {}
                )

            evaluation_display_identity = aggregated_display_identity(
                evaluation.get("model_info") or {}
            )
            model_variant_key = as_string(
                evaluation_display_identity.get("variant_key") or "default"
            )
            variant = variants_map.setdefault(
                model_variant_key,
                {
                    "variant_key": model_variant_key,
                    "variant_label": as_string(
                        evaluation_display_identity.get("variant_label") or "Default"
                    ),
                    "evaluation_count": 0,
                    "raw_model_ids": set(),
                    "last_updated": None,
                },
            )
            variant["evaluation_count"] += 1
            raw_id = as_string((evaluation.get("model_info") or {}).get("id"))
            if raw_id:
                variant["raw_model_ids"].add(raw_id)
            variant["last_updated"] = max_iso(variant["last_updated"], iso)

            for result in evaluation.get("evaluation_results") or []:
                normalized = result.get("normalized_result") or {}
                benchmark_display_name = as_string(
                    normalized.get("benchmark_parent_name")
                    or normalized.get("benchmark_family_name")
                )
                if benchmark_display_name:
                    benchmark_names_set.add(benchmark_display_name)
                score = extract_score(result)
                if score is not None:
                    score_values.append(score)

                    # Track best score per eval_summary_id for the model card
                    esid = get_eval_group_id(evaluation, result)
                    metric_config = result.get("metric_config") or {}
                    lower_is_better = bool(metric_config.get("lower_is_better"))
                    eval_name = as_string(result.get("evaluation_name"))

                    prev = best_per_metric.get(esid)
                    is_better = (
                        prev is None
                        or (lower_is_better and score < prev["score"])
                        or (not lower_is_better and score > prev["score"])
                    )
                    if is_better:
                        best_per_metric[esid] = {
                            "benchmark": benchmark_display_name
                            or as_string(evaluation.get("benchmark")),
                            "benchmarkKey": esid,
                            "canonical_display_name": as_string(
                                normalized.get("canonical_display_name")
                                or normalized.get("display_name")
                            ),
                            "evaluation_name": eval_name,
                            "score": score,
                            "metric": as_string(
                                metric_config.get("evaluation_description") or eval_name
                            ),
                            "unit": as_string(metric_config.get("unit")) or None,
                            "lower_is_better": lower_is_better,
                        }

        # Build top_benchmark_scores: deduplicate per benchmark (keep best metric),
        # sort by absolute score descending, cap at 15 entries
        top_benchmark_scores = sorted(
            best_per_metric.values(),
            key=lambda s: -abs(s["score"]),
        )[:15]
        # Strip None units to keep JSON compact
        for entry in top_benchmark_scores:
            if entry.get("unit") is None:
                del entry["unit"]

        summary_model_info = dict(model_info)
        summary_model_info.update(
            {
                "id": family_id,
                "name": family_name,
                "family_id": family_id,
                "family_name": family_name,
                "model_route_id": route_id,
                "variant_key": "default",
                "variant_label": "Default",
                "model_version": None,
            }
        )

        summary = {
            "model_info": summary_model_info,
            "model_family_id": family_id,
            "model_route_id": route_id,
            "model_family_name": family_name,
            # Registry-resolved canonical id, mirroring what model-cards.json
            # carries. None when no registry hit on this model.
            "canonical_model_id": display_identity.get("canonical_model_id"),
            "canonical_resolution_strategy": display_identity.get(
                "canonical_resolution_strategy"
            ),
            "canonical_resolution_confidence": display_identity.get(
                "canonical_resolution_confidence"
            ),
            "raw_model_ids": raw_model_ids,
            "evaluations_by_category": dict(by_category),
            "evaluation_summaries_by_category": {},
            "hierarchy_by_category": {},
            "total_evaluations": len(family_evals),
            "last_updated": last_updated,
            "categories_covered": [],
            "variants": [
                {
                    "variant_key": v["variant_key"],
                    "variant_label": v["variant_label"],
                    "evaluation_count": v["evaluation_count"],
                    "raw_model_ids": sorted(v["raw_model_ids"]),
                    "last_updated": v["last_updated"],
                }
                for v in variants_map.values()
            ],
        }
        filtered_eval_summaries = [
            filtered_summary
            for filtered_summary in (
                filter_eval_summary_for_model(eval_summary, family_id)
                for eval_summary in eval_summaries
            )
            if filtered_summary is not None
        ]
        summary_categories = sorted(
            {
                as_string(filtered_summary.get("category") or "other")
                for filtered_summary in filtered_eval_summaries
                if as_string(filtered_summary.get("category") or "other")
            }
        )
        summary["categories_covered"] = summary_categories
        summary["evaluation_summaries_by_category"] = {
            category: [
                filtered_summary
                for filtered_summary in filtered_eval_summaries
                if as_string(filtered_summary.get("category") or "other") == category
            ]
            for category in summary_categories
        }
        summary["hierarchy_by_category"] = summary["evaluation_summaries_by_category"]

        # Aggregate row-level reproducibility_gap + provenance annotations
        # across this model's filtered hierarchy, and per-group signals
        model_repro_annotations: list[dict] = []
        model_provenance_annotations: list[dict] = []
        model_variant_signal_groups: list[dict] = []
        model_signal_groups: list[dict] = []
        # Provenance groups are de-duped by group_id — multiple
        # metric_summaries within the same eval may carry the same
        # entry (one per metric on that (model, benchmark) pair).
        model_provenance_groups_by_id: dict[str, dict] = {}
        for filtered_summary in filtered_eval_summaries:
            filtered_metrics: list[dict] = list(filtered_summary.get("metrics", []))
            for subtask in filtered_summary.get("subtasks", []):
                filtered_metrics.extend(subtask.get("metrics", []))
            for metric in filtered_metrics:
                for entry in metric.get("_variant_signal_groups") or []:
                    if as_string(entry.get("model_route_id")) == route_id:
                        model_variant_signal_groups.append(entry)
                for entry in metric.get("_signal_groups") or []:
                    if as_string(entry.get("model_route_id")) == route_id:
                        model_signal_groups.append(entry)
                for entry in metric.get("_provenance_signal_groups") or []:
                    if as_string(entry.get("model_route_id")) != route_id:
                        continue
                    gid = entry.get("group_id")
                    if gid and gid not in model_provenance_groups_by_id:
                        model_provenance_groups_by_id[gid] = entry
                for row in metric.get("model_results", []):
                    annotations = (row.get("evalcards") or {}).get("annotations") or {}
                    repro = annotations.get("reproducibility_gap")
                    if repro is not None:
                        model_repro_annotations.append(repro)
                    prov = annotations.get("provenance")
                    if prov is not None:
                        model_provenance_annotations.append(prov)
        model_provenance_groups = list(model_provenance_groups_by_id.values())
        model_repro_summary = signals.summarize_reproducibility(model_repro_annotations)
        model_provenance_summary = signals.summarize_provenance(
            model_provenance_annotations, model_provenance_groups
        )
        model_comparability_summary = summarize_comparability_combined(
            model_variant_signal_groups, model_signal_groups
        )
        summary["reproducibility_summary"] = model_repro_summary
        summary["provenance_summary"] = model_provenance_summary
        summary["comparability_summary"] = model_comparability_summary
        model_summaries.append(summary)

        if score_values:
            score_summary = {
                "count": len(score_values),
                "min": min(score_values),
                "max": max(score_values),
                "average": sum(score_values) / len(score_values),
            }
        else:
            score_summary = {"count": 0, "min": None, "max": None, "average": None}

        model_cards.append(
            {
                "model_family_id": family_id,
                "model_route_id": route_id,
                "model_family_name": family_name,
                # Registry-resolved canonical (None when no resolution).
                # `family_id` and `model_route_id` are already canonical-derived
                # via canonical_model_identity, but exposing the canonical id
                # explicitly lets the frontend display "this card represents
                # registry canonical X" without parsing route_id.
                "canonical_model_id": display_identity.get("canonical_model_id"),
                "canonical_resolution_strategy": display_identity.get(
                    "canonical_resolution_strategy"
                ),
                "canonical_resolution_confidence": display_identity.get(
                    "canonical_resolution_confidence"
                ),
                "developer": as_string(model_info.get("developer")),
                "params_billions": params_billions,
                "total_evaluations": len(family_evals),
                "benchmark_count": len(
                    {
                        as_string(e.get("benchmark"))
                        for e in family_evals
                        if as_string(e.get("benchmark"))
                    }
                ),
                "benchmark_family_count": len(
                    {
                        as_string(
                            (
                                (result.get("normalized_result") or {}).get(
                                    "benchmark_family_key"
                                )
                            )
                        )
                        for evaluation in family_evals
                        for result in evaluation.get("evaluation_results") or []
                        if as_string(
                            (
                                (result.get("normalized_result") or {}).get(
                                    "benchmark_family_key"
                                )
                            )
                        )
                    }
                ),
                "categories_covered": summary_categories,
                "last_updated": last_updated,
                "variants": summary["variants"],
                "score_summary": score_summary,
                "reproducibility_summary": model_repro_summary,
                "provenance_summary": model_provenance_summary,
                "comparability_summary": model_comparability_summary,
                # ---- FIX 2 continued: include benchmark names and per-benchmark
                # scores so the frontend compare dialog and domain pills work ----
                "benchmark_names": sorted(benchmark_names_set),
                "top_benchmark_scores": top_benchmark_scores,
            }
        )

    model_cards.sort(
        key=lambda m: (-m["total_evaluations"], as_string(m["model_route_id"]))
    )
    model_summaries.sort(key=lambda m: as_string(m.get("model_route_id")))

    lite_model_cards = build_lightweight_model_cards(model_cards)

    eval_list = {
        "evals": [
            {
                "eval_summary_id": s["eval_summary_id"],
                "canonical_benchmark_id": s.get("canonical_benchmark_id"),
                "benchmark": s["benchmark"],
                "benchmark_family_key": s.get("benchmark_family_key"),
                "benchmark_family_name": s.get("benchmark_family_name"),
                "benchmark_parent_key": s.get("benchmark_parent_key"),
                "benchmark_parent_name": s.get("benchmark_parent_name"),
                "benchmark_leaf_key": s.get("benchmark_leaf_key"),
                "benchmark_leaf_name": s.get("benchmark_leaf_name"),
                "benchmark_component_key": s.get("benchmark_component_key"),
                "benchmark_component_name": s.get("benchmark_component_name"),
                "evaluation_name": s["evaluation_name"],
                "display_name": s.get("display_name"),
                "canonical_display_name": s.get("canonical_display_name"),
                "is_summary_score": s.get("is_summary_score", False),
                "summary_score_for": s.get("summary_score_for"),
                "summary_score_for_name": s.get("summary_score_for_name"),
                "summary_eval_ids": s.get("summary_eval_ids", []),
                "category": s.get("category", "other"),
                "models_count": s["models_count"],
                "metrics_count": s.get("metrics_count"),
                "subtasks_count": s.get("subtasks_count"),
                "metric_names": s.get("metric_names"),
                "primary_metric_name": s.get("primary_metric_name"),
                "benchmark_card": s["benchmark_card"],
                "tags": s.get("tags", {"domains": [], "languages": [], "tasks": []}),
                "source_data": s["source_data"],
                "metrics": [
                    {
                        "metric_summary_id": metric["metric_summary_id"],
                        "metric_name": metric.get("metric_name"),
                        "metric_id": metric.get("metric_id"),
                        "metric_key": metric.get("metric_key"),
                        "metric_source": metric.get("metric_source"),
                        "canonical_display_name": metric.get("canonical_display_name"),
                        "lower_is_better": metric.get("lower_is_better"),
                        "models_count": metric.get("models_count"),
                        "top_score": metric.get("top_score"),
                    }
                    for metric in s.get("metrics", [])
                ],
                "subtasks": [
                    {
                        "subtask_key": subtask.get("subtask_key"),
                        "subtask_name": subtask.get("subtask_name"),
                        "display_name": subtask.get("display_name"),
                        "canonical_display_name": subtask.get("canonical_display_name"),
                        "metrics_count": subtask.get("metrics_count"),
                        "metric_names": subtask.get("metric_names"),
                        "metrics": [
                            {
                                "metric_summary_id": metric["metric_summary_id"],
                                "metric_name": metric.get("metric_name"),
                                "metric_id": metric.get("metric_id"),
                                "metric_key": metric.get("metric_key"),
                                "metric_source": metric.get("metric_source"),
                                "canonical_display_name": metric.get(
                                    "canonical_display_name"
                                ),
                                "lower_is_better": metric.get("lower_is_better"),
                                "models_count": metric.get("models_count"),
                                "top_score": metric.get("top_score"),
                            }
                            for metric in subtask.get("metrics", [])
                        ],
                    }
                    for subtask in s.get("subtasks", [])
                ],
                "top_score": s.get("top_score"),
                "instance_data": s.get(
                    "instance_data",
                    {
                        "available": False,
                        "url_count": 0,
                        "sample_urls": [],
                        "models_with_loaded_instances": 0,
                    },
                ),
                "evalcards": s.get("evalcards"),
                "reproducibility_summary": s.get("reproducibility_summary"),
                "provenance_summary": s.get("provenance_summary"),
                "comparability_summary": s.get("comparability_summary"),
                # ``reporting_sources`` lists the per-source eval_summary_ids
                # whose data was unioned into a ``canonical__<id>`` row. For
                # non-canonical rows this is None. ``validate_output_contract``
                # uses this list to authorize per-source ``evals/<id>.json``
                # drilldown files that don't have a primary catalog row.
                "reporting_sources": s.get("source_eval_summary_ids"),
            }
            for s in catalog_eval_summaries
        ],
        "totalModels": len(model_cards),
    }

    # Catalog already reflects canonical-union semantics: rows that
    # contributed to a multi-source canonical were excluded from
    # catalog_eval_summaries upstream and replaced with the canonical-union
    # record (whose model_results are the cross-source union and whose
    # signal rollups were recomputed over that union). Sort the catalog
    # for deterministic ordering.
    eval_list["evals"].sort(
        key=lambda r: (-(r.get("models_count") or 0), as_string(r.get("eval_summary_id")))
    )

    lite_eval_list = build_lightweight_eval_list(eval_list)

    # ---- FIX 3: group developers by slug to merge case variants ----
    # e.g. "anthropic" and "Anthropic" both slugify to "anthropic"
    dev_group_by_slug: dict[str, list[dict]] = defaultdict(list)
    dev_name_by_slug: dict[str, str] = {}
    for card in model_cards:
        developer = as_string(card.get("developer") or "Unknown")
        slug = slugify_developer(developer)
        dev_group_by_slug[slug].append(card)
        # Keep the most common name variant (or the capitalized one)
        existing_name = dev_name_by_slug.get(slug)
        if existing_name is None or (
            developer[0:1].isupper() and not existing_name[0:1].isupper()
        ):
            dev_name_by_slug[slug] = developer

    developers = [
        {"developer": dev_name_by_slug[slug], "model_count": len(models)}
        for slug, models in dev_group_by_slug.items()
    ]
    developers.sort(key=lambda d: (-d["model_count"], as_string(d["developer"])))

    dev_summaries = []
    for slug, models in dev_group_by_slug.items():
        developer = dev_name_by_slug[slug]
        sorted_models = sorted(
            models, key=lambda m: as_string(m.get("model_family_name"))
        )
        dev_summaries.append(
            {"developer": developer, "slug": slug, "models": sorted_models}
        )

    # Corpus-level aggregates: walks every eval_summary's per-row
    # annotations + per-eval completeness + per-group signal entries to build
    # one summary artifact for paper / dashboard consumption. Must run before
    # the strip pass since it reads `_signal_groups` off the metric_summaries.
    repro_inputs: list[tuple] = []
    provenance_row_inputs: list[tuple] = []
    # Three grouping schemes flow into corpus rollups:
    # - ``variant_group_inputs``: per-metric_summary intra-source groups,
    #   carry variant_divergence
    # - ``group_inputs``: per-(canonical, metric_key, route), carry
    #   cross_party_divergence — the (model, benchmark, metric) unit
    # - ``provenance_group_inputs``: per-(canonical, route), carry
    #   is_multi_source/first_party_only flags — the (model, benchmark)
    #   unit. May include duplicates across metrics in the same
    #   (canonical, route) group; deduped by group_id below.
    group_inputs: list[tuple] = []
    variant_group_inputs: list[tuple] = []
    provenance_group_inputs_by_id: dict[str, tuple] = {}
    completeness_inputs: list[tuple] = []
    base_field_count = len(signals.BASE_REPRODUCIBILITY_FIELDS)

    for eval_summary in eval_summaries:
        category = as_string(eval_summary.get("category")) or None
        completeness = (
            (eval_summary.get("evalcards") or {}).get("annotations") or {}
        ).get("reporting_completeness")
        if isinstance(completeness, dict):
            completeness_inputs.append((completeness, category))

        all_metrics: list[dict] = list(eval_summary.get("metrics", []))
        for subtask in eval_summary.get("subtasks", []):
            all_metrics.extend(subtask.get("metrics", []))
        for metric in all_metrics:
            for group in metric.get("_variant_signal_groups") or []:
                variant_group_inputs.append((group, category))
            for group in metric.get("_signal_groups") or []:
                group_inputs.append((group, category))
            for entry in metric.get("_provenance_signal_groups") or []:
                gid = entry.get("group_id")
                if gid and gid not in provenance_group_inputs_by_id:
                    provenance_group_inputs_by_id[gid] = (entry, category)
            for row in metric.get("model_results", []):
                annotations = (row.get("evalcards") or {}).get("annotations") or {}
                repro = annotations.get("reproducibility_gap")
                if isinstance(repro, dict):
                    # Derive is_agentic from required_field_count: any extras
                    # beyond the base set come from the agentic schema.
                    is_agentic = (
                        repro.get("required_field_count") or 0
                    ) > base_field_count
                    repro_inputs.append(
                        (
                            {"annotation": repro, "is_agentic": is_agentic},
                            category,
                        )
                    )
                provenance = annotations.get("provenance")
                if isinstance(provenance, dict):
                    provenance_row_inputs.append((provenance, category))

    # Comparability stratifies over a combined input so a single block
    # carries both variant_* (intra-source) and cross_party_* (canonical
    # /family grouping) counters. ``total_groups`` is the sum of both
    # axes; the per-axis ``*_eligible_groups`` / ``*_divergent_groups``
    # counts are correct because each entry contributes to at most one
    # axis (variant entries don't carry cross_party; canonical entries
    # don't carry variant).
    combined_comparability_inputs = variant_group_inputs + group_inputs
    provenance_group_inputs = list(provenance_group_inputs_by_id.values())
    corpus_aggregates = {
        "generated_at": started_at,
        "signal_version": signals.SIGNAL_VERSION,
        "stratification_dimensions": ["category"],
        "reproducibility": signals.stratify(
            repro_inputs, signals.aggregate_reproducibility
        ),
        "completeness": signals.stratify(
            completeness_inputs, signals.aggregate_completeness
        ),
        # Provenance uses the (model, benchmark) grouping — drops
        # metric_key from the bucket so HELM-on-MMLU-with-exact_match
        # and HF-on-MMLU-with-accuracy count as one (model, benchmark)
        # pair with two parties → multi-source. Frontend definition
        # ("(model, benchmark) groups have reports from more than one
        # party") matches this grouping.
        "provenance": signals.stratify_provenance(
            provenance_row_inputs, provenance_group_inputs
        ),
        # Comparability cross-party axis uses the (model, benchmark,
        # metric) grouping — keeps metric_key in the bucket so cross-
        # party divergence compares apples to apples. Variant axis is
        # per-metric_summary intra-source.
        "comparability": signals.stratify(
            combined_comparability_inputs, signals.aggregate_comparability
        ),
    }

    manifest = {
        "generated_at": started_at,
        "model_count": len(model_cards),
        # eval_count = total per-eval JSON files on disk = per-source records
        # + canonical-union records. Each canonical-union row in the catalog
        # gets its own ``canonical__<id>.json`` and its source contributors
        # also keep their per-source JSONs as drilldowns.
        "eval_count": len(eval_summaries) + len(canonical_eval_summaries),
        "metric_eval_count": sum(
            len(summary.get("metrics", []))
            + sum(
                len(subtask.get("metrics", []))
                for subtask in summary.get("subtasks", [])
            )
            for summary in eval_summaries
        ),
        "config_version": CONFIG_VERSION,
        "skipped_config_count": len(skipped_configs),
        "skipped_configs": skipped_configs,
        "source_config_count": len(all_configs),
        "summary_artifacts": {
            "comparison_index": "comparison-index.json",
            "corpus_aggregates": "corpus-aggregates.json",
            "eval_hierarchy": "eval-hierarchy.json",
            **(
                {
                    "model_cards": "model-cards.json",
                    "model_cards_lite": "model-cards-lite.json",
                    "eval_list": "eval-list.json",
                    "eval_list_lite": "eval-list-lite.json",
                }
                if emit_legacy_json()
                else {}
            ),
        },
    }

    write_json(OUTPUT_DIR / "corpus-aggregates.json", corpus_aggregates)
    write_json(OUTPUT_DIR / "peer-ranks.json", peer_ranks)
    write_json(OUTPUT_DIR / "comparison-index.json", comparison_index)
    write_json(OUTPUT_DIR / "benchmark-metadata.json", benchmark_metadata)
    if emit_legacy_json():
        write_json(OUTPUT_DIR / "model-cards.json", model_cards)
        write_json(OUTPUT_DIR / "model-cards-lite.json", lite_model_cards)
        write_json(OUTPUT_DIR / "eval-list.json", eval_list)
        write_json(OUTPUT_DIR / "eval-list-lite.json", lite_eval_list)
        write_json(OUTPUT_DIR / "developers.json", developers)

    # eval-hierarchy.json: regenerate at runtime from current eval_summaries
    # (replaces a previous shutil.copy2 of `reports/eval_hierarchy.json`,
    # which was a stale snapshot from a separate generator script and only
    # covered ~7/20 of the live family keys).
    #
    # Two-level structure: family → leaf. `family_key` and `leaf_key` come
    # straight off each eval_summary, so the output always reflects the
    # current corpus. Per-family rollups aggregate across all the family's
    # leaves; per-leaf rollups aggregate the eval_summary(ies) that
    # share that `(family, leaf)` pair (typically one).
    #
    # KNOWN LIMITATION (2026-04-26): family keys are NOT collapsed via
    # `build_eval_hierarchy_report.py`'s FAMILY_RULES. Consequence:
    # `helm_air_bench`, `helm_classic`, `helm_lite`, `helm_instruct`,
    # `helm_capabilities`, and `helm_mmlu` each appear as a top-level
    # family instead of as composites under one `helm` family. Same for
    # `apex_v1` + `apex_agents` etc.
    def _collect_eval_signal_data(
        eval_summary: dict,
    ) -> tuple[list, list, list, list, dict]:
        repro_anns: list[dict] = []
        prov_anns: list[dict] = []
        variant_groups: list[dict] = []
        sig_groups: list[dict] = []
        # Provenance groups are de-duped by group_id within the
        # collection — multiple metric_summaries within one eval may
        # carry the same (canonical, route) provenance entry.
        provenance_groups_by_id: dict[str, dict] = {}
        metrics_pool = list(eval_summary.get("metrics") or [])
        for subtask in eval_summary.get("subtasks") or []:
            metrics_pool.extend(subtask.get("metrics") or [])
        for metric in metrics_pool:
            variant_groups.extend(metric.get("_variant_signal_groups") or [])
            sig_groups.extend(metric.get("_signal_groups") or [])
            for entry in metric.get("_provenance_signal_groups") or []:
                gid = entry.get("group_id")
                if gid and gid not in provenance_groups_by_id:
                    provenance_groups_by_id[gid] = entry
            for row in metric.get("model_results") or []:
                annotations = (row.get("evalcards") or {}).get("annotations") or {}
                repro = annotations.get("reproducibility_gap")
                if isinstance(repro, dict):
                    repro_anns.append(repro)
                prov = annotations.get("provenance")
                if isinstance(prov, dict):
                    prov_anns.append(prov)
        return repro_anns, prov_anns, variant_groups, sig_groups, provenance_groups_by_id

    def _node_rollups(evals_for_node: list[dict]) -> dict:
        repro_all: list[dict] = []
        prov_all: list[dict] = []
        variant_groups_all: list[dict] = []
        groups_all: list[dict] = []
        prov_groups_by_id: dict[str, dict] = {}
        for es in evals_for_node:
            r, p, vg, g, pg = _collect_eval_signal_data(es)
            repro_all.extend(r)
            prov_all.extend(p)
            variant_groups_all.extend(vg)
            groups_all.extend(g)
            for gid, entry in pg.items():
                if gid not in prov_groups_by_id:
                    prov_groups_by_id[gid] = entry
        return {
            "reproducibility_summary": signals.summarize_reproducibility(repro_all),
            "provenance_summary": signals.summarize_provenance(
                prov_all, list(prov_groups_by_id.values())
            ),
            "comparability_summary": summarize_comparability_combined(
                variant_groups_all, groups_all
            ),
        }

    families_in_progress: dict[str, dict] = {}
    for eval_summary in catalog_eval_summaries:
        family_key = (
            as_string(eval_summary.get("benchmark_family_key")) or "unknown"
        )
        family_display = (
            as_string(eval_summary.get("benchmark_family_name")) or family_key
        )
        family = families_in_progress.setdefault(
            family_key,
            {
                "key": family_key,
                "display_name": family_display,
                "category": as_string(eval_summary.get("category")) or "other",
                "_evals": [],
                "_leaves": {},
            },
        )
        family["_evals"].append(eval_summary)
        leaf_key = as_string(eval_summary.get("benchmark_leaf_key")) or as_string(
            eval_summary.get("eval_summary_id")
        )
        family["_leaves"].setdefault(leaf_key, []).append(eval_summary)

    runtime_hierarchy: dict[str, Any] = {
        "generated_at": started_at,
        "signal_version": signals.SIGNAL_VERSION,
        "schema_note": (
            "Runtime-generated 2-level hierarchy (family → leaf). "
            "Replaces the previously-shipped static reports/eval_hierarchy.json snapshot. "
            "Family keys are uncollapsed runtime values: `helm_classic`, `helm_lite`, "
            "`helm_air_bench` etc. each appear as separate top-level families instead "
            "of as composites under one `helm` family."
        ),
        "families": [],
    }
    for family_key in sorted(families_in_progress):
        family = families_in_progress[family_key]
        family_rollups = _node_rollups(family["_evals"])
        leaves: list[dict] = []
        for leaf_key in sorted(family["_leaves"]):
            leaf_evals = family["_leaves"][leaf_key]
            leaf_rollups = _node_rollups(leaf_evals)
            # Within a single (family, leaf) bucket the eval_summaries
            # almost always share one canonical_benchmark_id, but the
            # corpus has at least one case (helm_classic_mmlu vs
            # helm_lite_mmlu both at leaf "mmlu" under different family
            # keys) where they collapse to canonical "mmlu". We surface
            # the leaf-local set so consumers can detect that.
            leaf_canonical_ids = sorted(
                {
                    as_string(es.get("canonical_benchmark_id"))
                    for es in leaf_evals
                    if as_string(es.get("canonical_benchmark_id"))
                }
            )
            leaves.append(
                {
                    "key": leaf_key,
                    "canonical_benchmark_id": (
                        leaf_canonical_ids[0]
                        if len(leaf_canonical_ids) == 1
                        else None
                    ),
                    "canonical_benchmark_ids": leaf_canonical_ids,
                    "display_name": (
                        as_string(leaf_evals[0].get("canonical_display_name"))
                        or as_string(leaf_evals[0].get("benchmark_leaf_name"))
                        or leaf_key
                    ),
                    "category": as_string(leaf_evals[0].get("category")) or "other",
                    "evals_count": len(leaf_evals),
                    "eval_summary_ids": sorted(
                        as_string(es.get("eval_summary_id"))
                        for es in leaf_evals
                        if as_string(es.get("eval_summary_id"))
                    ),
                    **leaf_rollups,
                }
            )
        family_canonical_ids = sorted(
            {
                as_string(es.get("canonical_benchmark_id"))
                for es in family["_evals"]
                if as_string(es.get("canonical_benchmark_id"))
            }
        )
        runtime_hierarchy["families"].append(
            {
                "key": family["key"],
                "display_name": family["display_name"],
                "category": family["category"],
                "canonical_benchmark_ids": family_canonical_ids,
                "evals_count": len(family["_evals"]),
                "eval_summary_ids": sorted(
                    as_string(es.get("eval_summary_id"))
                    for es in family["_evals"]
                    if as_string(es.get("eval_summary_id"))
                ),
                **family_rollups,
                "leaves": leaves,
            }
        )

    # Frontend (`general-eval-card/lib/backend-artifacts.ts:67-79`) declares
    # `EvalHierarchy.stats` as required and `app/evals/page.tsx` reads
    # `hierarchy.stats.family_count` without a guard. The canonical semantics
    # come from `scripts/build_eval_hierarchy_report.py:647-654`. The runtime
    # hierarchy here is a 2-level (family → leaf) view built from
    # `eval_summaries`; we approximate the report-script counters from the
    # current corpus rather than from the report's tree:
    #   - `family_count`         : top-level families emitted
    #   - `composite_count`      : families with >1 leaf (multi-leaf suites)
    #   - `standalone_benchmark_count`: families with exactly one leaf
    #   - `single_benchmark_count`: total leaves under composite (multi-leaf)
    #     families (i.e. the per-benchmark members of those suites)
    #   - `slice_count`          : distinct `slice_key` values across all
    #     metrics in the corpus (root metrics + subtask metrics)
    #   - `metric_count`         : distinct `metric_key` values across the
    #     same metric set
    #   - `metric_rows_scanned`  : total `model_results` rows across all
    #     metrics (each row is one model×metric submission)
    runtime_family_count = len(runtime_hierarchy["families"])
    runtime_composite_count = sum(
        1 for fam in runtime_hierarchy["families"] if len(fam.get("leaves") or []) > 1
    )
    runtime_standalone_benchmark_count = sum(
        1 for fam in runtime_hierarchy["families"] if len(fam.get("leaves") or []) == 1
    )
    runtime_single_benchmark_count = sum(
        len(fam.get("leaves") or [])
        for fam in runtime_hierarchy["families"]
        if len(fam.get("leaves") or []) > 1
    )
    runtime_slice_keys: set[str] = set()
    runtime_metric_keys: set[str] = set()
    runtime_metric_rows_scanned = 0
    for eval_summary in eval_summaries:
        metrics_pool = list(eval_summary.get("metrics") or [])
        for subtask in eval_summary.get("subtasks") or []:
            metrics_pool.extend(subtask.get("metrics") or [])
        for metric in metrics_pool:
            slice_key_value = as_string(metric.get("slice_key"))
            if slice_key_value:
                runtime_slice_keys.add(slice_key_value)
            metric_key_value = as_string(metric.get("metric_key"))
            if metric_key_value:
                runtime_metric_keys.add(metric_key_value)
            runtime_metric_rows_scanned += len(metric.get("model_results") or [])

    runtime_hierarchy["stats"] = {
        "family_count": runtime_family_count,
        "composite_count": runtime_composite_count,
        "standalone_benchmark_count": runtime_standalone_benchmark_count,
        "single_benchmark_count": runtime_single_benchmark_count,
        "slice_count": len(runtime_slice_keys),
        "metric_count": len(runtime_metric_keys),
        "metric_rows_scanned": runtime_metric_rows_scanned,
    }

    write_json(OUTPUT_DIR / "eval-hierarchy.json", runtime_hierarchy)

    hierarchy_path = OUTPUT_DIR / "eval-hierarchy.json"
    readme_text = build_dataset_readme(
        manifest=manifest,
        eval_list=eval_list,
        benchmark_metadata=benchmark_metadata,
        dataset_repo=DATASET_REPO,
        benchmark_metadata_dataset_repo=BENCHMARK_METADATA_DATASET_REPO,
        hierarchy_path=hierarchy_path,
    )
    (OUTPUT_DIR / "README.md").write_text(readme_text, encoding="utf-8")

    # Strip Signals 3+4 internals carried for the rollup passes. After this
    # point the per-summary JSONs serialize cleanly: rows lose
    # `_generation_args` (input plumbing for variant_divergence comparison)
    # and metric_summaries lose `_signal_groups` (group-level summaries used
    # by the per-eval / per-model rollups).
    def _strip_signals_internals(summary_obj: dict) -> None:
        summary_obj.pop("_eee_source_config", None)
        all_metrics: list[dict] = list(summary_obj.get("metrics", []))
        for subtask in summary_obj.get("subtasks", []):
            all_metrics.extend(subtask.get("metrics", []))
        for metric in all_metrics:
            metric.pop("_signal_groups", None)
            metric.pop("_variant_signal_groups", None)
            metric.pop("_provenance_signal_groups", None)
            for row in metric.get("model_results", []):
                row.pop("_generation_args", None)

    def _strip_eval_obj_internals(eval_obj: dict) -> None:
        # Models' ``evaluations_by_category`` carries raw eval_obj
        # records. ``_eee_source_config`` (set at eval_obj construction
        # for registry lookup) leaks into the published model JSONs
        # via this path; strip it before serialization.
        eval_obj.pop("_eee_source_config", None)
        mi = eval_obj.get("model_info")
        if isinstance(mi, dict):
            mi.pop("_model_identity_bundle", None)

    for summary in eval_summaries:
        _strip_signals_internals(summary)
    for summary in canonical_eval_summaries:
        _strip_signals_internals(summary)
    for summary in model_summaries:
        for category_summaries in summary.get("hierarchy_by_category", {}).values():
            for filtered_summary in category_summaries:
                _strip_signals_internals(filtered_summary)
        for category_evals in summary.get("evaluations_by_category", {}).values():
            for eval_obj in category_evals:
                _strip_eval_obj_internals(eval_obj)

    if emit_legacy_json():
        for summary in model_summaries:
            write_json(OUTPUT_DIR / "models" / f"{summary['model_route_id']}.json", summary)
        # Per-source detail JSONs survive as drilldowns from canonical pages
        # and as direct destinations for model_summaries' top_benchmark_scores
        # references (which carry per-source eval_summary_ids).
        for summary in eval_summaries:
            write_json(OUTPUT_DIR / "evals" / f"{summary['eval_summary_id']}.json", summary)
        # Canonical-union detail JSONs: the catalog tile points here for
        # multi-source canonicals (e.g. /evals/canonical__gpqa). Filename
        # matches the eval_summary_id verbatim.
        for summary in canonical_eval_summaries:
            write_json(OUTPUT_DIR / "evals" / f"{summary['eval_summary_id']}.json", summary)
        for summary in dev_summaries:
            write_json(
                OUTPUT_DIR / "developers" / f"{summary['slug']}.json",
                {"developer": summary["developer"], "models": summary["models"]},
            )

    validate_output_contract(OUTPUT_DIR)

    # Parity-layer parquet artifacts under output/duckdb/v1/. Adds a
    # frontend-ready read surface alongside the existing JSON files;
    # run on every pipeline invocation so consumers can rely on the
    # path being present. Cleaning transforms (license, dev names,
    # params, variants, benchmark display names) are applied here per
    # PLAN_20260428.md. Emitted AFTER `validate_output_contract` because
    # the validator scans every output file as UTF-8 text.
    from scripts import parity_outputs

    parity_outputs.write_parity_artifacts(
        model_cards=model_cards,
        lite_model_cards=lite_model_cards,
        eval_list=eval_list,
        lite_eval_list=lite_eval_list,
        # Pass per-source AND canonical-union eval_summaries so the
        # eval_summaries.parquet detail-page surface contains both source
        # drilldowns (referenced by model_summaries.top_benchmark_scores) and
        # canonical-union records (referenced by the catalog tile).
        eval_summaries=eval_summaries + canonical_eval_summaries,
        model_summaries=model_summaries,
        dev_summaries=dev_summaries,
        benchmark_metadata=benchmark_metadata,
        output_dir=OUTPUT_DIR,
    )

    manifest["artifact_sizes"] = collect_artifact_sizes(OUTPUT_DIR)
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    print(
        f"[pipeline] {json.dumps({'event': 'pipeline.summary', 'dry_run': dry_run, 'evaluations_loaded': len(evaluations), 'model_count': len(model_cards), 'eval_count': len(eval_summaries), 'skipped_config_count': len(skipped_configs)})}"
    )

    print(
        json.dumps(
            {
                "dry_run": dry_run,
                "model_count": len(model_cards),
                "eval_count": len(eval_summaries),
                "skipped_configs": skipped_configs,
                "output_dir": str(OUTPUT_DIR.resolve()),
            },
            indent=2,
        )
    )

    if not dry_run:
        upload_output(OUTPUT_DIR, PRODUCTION_DATASET_REPO)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

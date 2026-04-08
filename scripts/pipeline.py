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

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, snapshot_download


DATASET_REPO = "evaleval/card_backend"
EEE_DATASET_REPO = "evaleval/EEE_datastore"
EEE_DATASET_RAW_BASE = f"https://huggingface.co/datasets/{EEE_DATASET_REPO}/raw/main"
CONFIG_VERSION = 1
OUTPUT_DIR = Path("output")
METADATA_DIR = Path("metadata")
DEFAULT_LOCAL_DATASET_DIR = ".cache/eee_datastore"
FILE_READ_MAX_RETRIES = 5
FILE_READ_RETRY_DELAY_SEC = 1.5
VERSION_SUFFIX_REGEX = re.compile(r"^(.*?)-((?:19|20)\d{6})(?:-(.+))?$")
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
    return re.sub(r"[^a-z0-9]+", "_", as_string(value).lower()).strip("_")


def slugify(value: Any) -> str:
    return normalize_benchmark_key(value)


def sanitize_slug_input(value: Any) -> str:
    text = as_string(value)
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r"\\x[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\\u[0-9a-fA-F]{4}", "", text)
    return text


def ensure_safe_slug_segment(value: Any) -> str:
    cleaned = as_string(value).strip()
    if not cleaned:
        return "unknown"
    if re.match(r"^x00", cleaned, flags=re.IGNORECASE):
        trimmed = re.sub(r"^x0+", "", cleaned, flags=re.IGNORECASE)
        return f"safe-{trimmed or 'unknown'}"
    return cleaned


def slugify_developer(value: Any) -> str:
    cleaned = sanitize_slug_input(value).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return ensure_safe_slug_segment(cleaned)


def slugify_model_segment(value: Any) -> str:
    cleaned = sanitize_slug_input(value).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return ensure_safe_slug_segment(cleaned)


def humanize_slug(value: Any) -> str:
    parts = [p for p in re.split(r"[-_/]+", as_string(value)) if p]
    out = []
    for part in parts:
        if len(part) <= 3 and any(c.isdigit() for c in part):
            out.append(part.upper())
        else:
            out.append(part[:1].upper() + part[1:])
    return " ".join(out)


def parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def iso_from_epoch_string(value: Any) -> str | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def max_iso(left: str | None, right: str | None) -> str | None:
    if not left:
        return right
    if not right:
        return left
    return left if left > right else right


def load_benchmark_metadata() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    cards = []
    lookup: dict[str, dict] = {}
    flat_map: dict[str, dict] = {}

    for file_path in sorted(METADATA_DIR.glob("benchmark_card_*.json")):
        parsed = json.loads(file_path.read_text(encoding="utf-8"))
        card = parsed.get("benchmark_card")
        if not card:
            continue
        base_name = file_path.stem.replace("benchmark_card_", "")
        keys = candidate_benchmark_keys(base_name, card.get("benchmark_details", {}).get("name"))
        cards.append({"file_name": file_path.name, "base_name": base_name, "card": card, "keys": keys})
        for key in keys:
            lookup[key] = card
            flat_map[key] = card

    return cards, lookup, flat_map


def candidate_benchmark_keys(*values: Any) -> list[str]:
    keys = set()
    for value in values:
        text = as_string(value)
        if not text:
            continue
        keys.add(normalize_benchmark_key(text))
        keys.add(normalize_benchmark_key(re.sub(r"^benchmark_card_", "", text, flags=re.IGNORECASE)))
        keys.add(normalize_benchmark_key(re.sub(r"[_-]+", " ", text)))
    return [k for k in keys if k]


def lookup_benchmark_card(metadata_lookup: dict[str, dict], *values: Any) -> dict | None:
    for key in candidate_benchmark_keys(*values):
        if key in metadata_lookup:
            return metadata_lookup[key]
    return None


def ensure_local_dataset_snapshot(local_dataset_dir: str, hf_token: str | None, force_refresh: bool) -> str:
    target_dir = Path(local_dataset_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if force_refresh and target_dir.exists():
        shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=EEE_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=["data/**"],
        token=hf_token,
    )

    return str(target_dir)


def discover_configs(local_dataset_dir: str | None, hf_token: str | None) -> list[str]:
    if local_dataset_dir:
        data_root = Path(local_dataset_dir) / "data"
        configs = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
        return configs

    fs = HfFileSystem(token=hf_token)
    entries = fs.ls(f"datasets/{EEE_DATASET_REPO}/data", detail=True)
    configs = []
    for entry in entries:
        name = entry.get("name", "")
        config = name.split("/")[-1]
        if config:
            configs.append(config)
    return sorted(set(configs))


def list_json_files_for_config(config: str, local_dataset_dir: str | None, hf_token: str | None) -> list[str]:
    if local_dataset_dir:
        root = Path(local_dataset_dir) / "data" / config
        return sorted(str(p.relative_to(local_dataset_dir)).replace(os.sep, "/") for p in root.rglob("*.json") if p.is_file() and not p.name.endswith(".jsonl"))

    fs = HfFileSystem(token=hf_token)
    pattern = f"datasets/{EEE_DATASET_REPO}/data/{config}/**/*.json"
    paths = [p for p in fs.glob(pattern) if not p.endswith(".jsonl")]
    prefix = f"datasets/{EEE_DATASET_REPO}/"
    return sorted(p[len(prefix):] for p in paths)


def read_dataset_json(dataset_path: str, local_dataset_dir: str | None, hf_token: str | None) -> dict:
    if local_dataset_dir:
        local_path = Path(local_dataset_dir) / dataset_path
        return json.loads(local_path.read_text(encoding="utf-8"))

    local_path = hf_hub_download(
        repo_id=EEE_DATASET_REPO,
        filename=dataset_path,
        repo_type="dataset",
        token=hf_token,
    )
    return json.loads(Path(local_path).read_text(encoding="utf-8"))


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

        file_path_match = re.search(r"file_path'?:\s*'([^']+)'", value, flags=re.IGNORECASE) or re.search(r'"file_path"\s*:\s*"([^"]+)"', value)
        format_match = re.search(r"format'?:\s*'([^']+)'", value, flags=re.IGNORECASE) or re.search(r'"format"\s*:\s*"([^"]+)"', value)
        rows_match = re.search(r"total_rows'?:\s*(\d+)", value, flags=re.IGNORECASE) or re.search(r'"total_rows"\s*:\s*(\d+)', value)
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
        file_path = as_string(value.get("file_path") or value.get("path") or value.get("url"))
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


def maybe_load_instance_data(record: dict, local_dataset_dir: str | None, hf_token: str | None) -> dict | None:
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
                text = (Path(local_dataset_dir) / dataset_path).read_text(encoding="utf-8")
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


def normalize_model_info(model_info: dict) -> dict:
    raw_id = as_string(model_info.get("id") or model_info.get("name") or "unknown/unknown")
    fallback_developer = as_string(model_info.get("developer") or raw_id.split("/")[0] or "unknown")
    if "/" in raw_id:
        parts = raw_id.split("/")
    else:
        parts = [slugify_developer(fallback_developer), raw_id]
    raw_developer = parts[0] if len(parts) > 1 else slugify_developer(fallback_developer)
    raw_model_name = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
    match = VERSION_SUFFIX_REGEX.match(raw_model_name)
    base_slug = match.group(1) if match else raw_model_name
    version_date = match.group(2) if match else None
    qualifier = match.group(3) if match else None

    return {
        "raw_id": raw_id,
        "developer": as_string(model_info.get("developer") or humanize_slug(raw_developer)),
        "developer_slug": slugify_developer(raw_developer),
        "model_name": as_string(model_info.get("name") or humanize_slug(base_slug)),
        "raw_model_name": raw_model_name,
        "family_slug": slugify_model_segment(base_slug),
        "version_date": version_date,
        "qualifier": qualifier,
    }


def canonical_model_identity(model_info: dict) -> dict:
    normalized = normalize_model_info(model_info)
    family_id = f"{normalized['developer_slug']}/{normalized['family_slug']}"
    normalized_id = (
        f"{normalized['developer_slug']}/{normalized['raw_model_name']}"
        if "/" in normalized["raw_id"]
        else f"{normalized['developer_slug']}/{normalized['raw_id']}"
    )
    variant_parts = [p for p in [normalized["version_date"], normalized["qualifier"]] if p]
    variant_key = "-".join(variant_parts) if variant_parts else "default"
    variant_label = " ".join(variant_parts) if variant_parts else "Default"

    return {
        "normalized_id": normalized_id,
        "family_id": family_id,
        "family_slug": normalized["family_slug"],
        "family_name": normalized["model_name"],
        "model_route_id": family_id.replace("/", "__"),
        "variant_key": variant_key,
        "variant_label": variant_label,
    }


def infer_category_from_benchmark(benchmark_name: str) -> str:
    key = normalize_benchmark_key(benchmark_name)
    if not key:
        return "other"
    if re.search(r"(math|gsm|gpqa|mmlu|medqa|legalbench|boolq|hellaswag|quac|cnn_dailymail|civilcomments|ifeval|musr)", key):
        return "reasoning"
    if re.search(r"(appworld|swe_bench|tau_bench|browsecomp|agent|livecodebench)", key):
        return "agentic"
    if re.search(r"(reward_bench|hfopenllm|helm)", key):
        return "general"
    return "other"


def extract_score(result: dict) -> float | None:
    score_details = result.get("score_details") if isinstance(result, dict) else None
    if not isinstance(score_details, dict):
        return None
    score = score_details.get("score")
    try:
        return float(score)
    except Exception:
        return None


def get_eval_summary_id(evaluation: dict, result: dict) -> str:
    source_data = result.get("source_data") if isinstance(result, dict) else {}
    benchmark_key = evaluation.get("benchmark") or (source_data or {}).get("dataset_name") or result.get("evaluation_name")
    return slugify(f"{benchmark_key}__{result.get('evaluation_name') or 'unknown'}")


def clean_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "evals").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "developers").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_output() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required unless --dry-run is used")

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
    except Exception as error:
        print(f"create_repo warning: {error}", file=sys.stderr)

    api.upload_folder(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=str(OUTPUT_DIR),
        path_in_repo=".",
        commit_message=f"Pipeline sync {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
    )


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    load_instance_in_dry_run = os.environ.get("LOAD_INSTANCE_IN_DRY_RUN") == "1"
    config_batch_size = parse_positive_int(os.environ.get("CONFIG_BATCH_SIZE"), 4)
    config_limit = os.environ.get("CONFIG_LIMIT")
    explicit_configs = [c.strip() for c in as_string(os.environ.get("CONFIGS") or os.environ.get("CONFIG_NAMES")).split(",") if c.strip()]
    configured_local_dataset_dir = as_string(os.environ.get("EEE_LOCAL_DATASET_DIR")).strip() or DEFAULT_LOCAL_DATASET_DIR
    force_refresh_snapshot = os.environ.get("EEE_REFRESH_SNAPSHOT") == "1"
    allow_skipped_configs = os.environ.get("ALLOW_SKIPPED_CONFIGS") == "1"
    hf_token = os.environ.get("HF_TOKEN")

    local_dataset_dir = ensure_local_dataset_snapshot(configured_local_dataset_dir, hf_token, force_refresh_snapshot)

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    random.seed(42)

    clean_output_dir()
    cards, metadata_lookup, benchmark_metadata = load_benchmark_metadata()
    print(f"[pipeline] {json.dumps({'event': 'metadata.loaded', 'benchmark_card_count': len(cards), 'metadata_key_count': len(metadata_lookup)})}")

    all_configs = explicit_configs or discover_configs(local_dataset_dir, hf_token)
    if config_limit:
        all_configs = all_configs[: max(1, min(parse_positive_int(config_limit, len(all_configs)), len(all_configs)))]

    skipped_configs: list[str] = []
    evaluations: list[dict] = []

    for i in range(0, len(all_configs), config_batch_size):
        batch = all_configs[i : i + config_batch_size]
        print(f"[pipeline] {json.dumps({'event': 'config.batch.start', 'batch_index': i // config_batch_size, 'batch_size': len(batch), 'configs': batch})}")

        for config in batch:
            try:
                files = list_json_files_for_config(config, local_dataset_dir, hf_token)
                print(f"[pipeline] {json.dumps({'event': 'config.discovery', 'config': config, 'data_json_files_found': len(files), 'discovery_pages': 1, 'discovery_error': None})}")
                loaded_rows = 0
                failed_files: list[str] = []
                for dataset_path in files:
                    record = None
                    last_error = None
                    for attempt in range(1, FILE_READ_MAX_RETRIES + 1):
                        try:
                            record = read_dataset_json(dataset_path, local_dataset_dir, hf_token)
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
                    eval_results = record.get("evaluation_results") if isinstance(record.get("evaluation_results"), list) else []
                    first_result = eval_results[0] if eval_results else None
                    benchmark = as_string(record.get("evaluation_id")).split("/")[0] if record.get("evaluation_id") else None
                    passthrough = {k: v for k, v in record.items() if k not in KNOWN_TOP_LEVEL_KEYS}
                    detailed_meta = normalize_detailed_eval_meta(record.get("detailed_evaluation_results"))
                    eval_obj = {
                        "schema_version": as_string(record.get("schema_version")),
                        "evaluation_id": as_string(record.get("evaluation_id")),
                        "retrieved_timestamp": as_string(record.get("retrieved_timestamp")),
                        "benchmark": benchmark,
                        "source_data": (first_result or {}).get("source_data"),
                        "source_metadata": record.get("source_metadata"),
                        "eval_library": record.get("eval_library"),
                        "model_info": record.get("model_info") or {},
                        "generation_config": (first_result or {}).get("generation_config"),
                        "source_record_url": source_record_url,
                        "detailed_evaluation_results_meta": detailed_meta,
                        "detailed_evaluation_results": resolve_detailed_results_url(record, source_record_url),
                        "passthrough_top_level_fields": passthrough or None,
                        "evaluation_results": eval_results,
                        "benchmark_card": None,
                        "instance_level_data": None,
                    }
                    evaluations.append(eval_obj)
                    loaded_rows += 1

                if failed_files:
                    message = f"Failed to load {len(failed_files)} files for config {config}"
                    print(
                        f"[pipeline] {json.dumps({'event': 'config.load.partial', 'config': config, 'row_count': loaded_rows, 'failed_files': len(failed_files), 'sample_failed_paths': failed_files[:5]})}"
                    )
                    if not allow_skipped_configs:
                        raise RuntimeError(message)

                print(
                    f"[pipeline] {json.dumps({'event': 'config.load.ok', 'config': config, 'discovered_data_json_files': len(files), 'discovery_pages': 1, 'row_count': loaded_rows})}"
                )
            except Exception as error:
                print(f"[pipeline] {json.dumps({'event': 'config.load.error', 'config': config, 'message': str(error)})}")
                if allow_skipped_configs:
                    print(f"Skipping config {config}: {error}", file=sys.stderr)
                    skipped_configs.append(config)
                else:
                    raise

        print(f"[pipeline] {json.dumps({'event': 'config.batch.done', 'batch_index': i // config_batch_size, 'cumulative_evaluations': len(evaluations), 'cumulative_skipped': len(skipped_configs)})}")

    if (not dry_run) or load_instance_in_dry_run:
        with_instance = 0
        missing_instance = 0
        for idx, evaluation in enumerate(evaluations, start=1):
            instance_data = maybe_load_instance_data(evaluation, local_dataset_dir, hf_token)
            if instance_data:
                evaluation["instance_level_data"] = instance_data
                with_instance += 1
            else:
                missing_instance += 1
            if idx % 100 == 0 or idx == len(evaluations):
                print(f"[pipeline] {json.dumps({'event': 'instance.batch.progress', 'processed': idx, 'total': len(evaluations), 'with_instance_data': with_instance, 'missing_instance_data': missing_instance})}")
        print(f"[pipeline] {json.dumps({'event': 'instance.load.summary', 'total': len(evaluations), 'with_instance_data': with_instance, 'missing_instance_data': missing_instance})}")

    for evaluation in evaluations:
        identity = canonical_model_identity(evaluation.get("model_info") or {})
        model_info = dict(evaluation.get("model_info") or {})
        model_info.update(
            {
                "normalized_id": identity["normalized_id"],
                "family_id": identity["family_id"],
                "family_slug": identity["family_slug"],
                "family_name": identity["family_name"],
                "variant_key": identity["variant_key"],
                "variant_label": identity["variant_label"],
                "model_route_id": identity["model_route_id"],
            }
        )
        evaluation["model_info"] = model_info
        evaluation["benchmark_card"] = lookup_benchmark_card(metadata_lookup, evaluation.get("benchmark"), (evaluation.get("source_data") or {}).get("dataset_name"))

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
            eval_summary_id = get_eval_summary_id(evaluation, result)
            group = benchmark_groups.setdefault(
                eval_summary_id,
                {
                    "eval_summary_id": eval_summary_id,
                    "benchmark": evaluation.get("benchmark"),
                    "evaluation_name": result.get("evaluation_name"),
                    "lower_is_better": bool((result.get("metric_config") or {}).get("lower_is_better")),
                    "metric_config": result.get("metric_config"),
                    "source_data": result.get("source_data"),
                    "benchmark_card": lookup_benchmark_card(
                        metadata_lookup,
                        evaluation.get("benchmark"),
                        result.get("evaluation_name"),
                        (result.get("source_data") or {}).get("dataset_name"),
                    ),
                    "model_results": [],
                },
            )
            group["model_results"].append(
                {
                    "model_id": as_string((evaluation.get("model_info") or {}).get("family_id")),
                    "model_route_id": as_string((evaluation.get("model_info") or {}).get("model_route_id")),
                    "model_name": as_string((evaluation.get("model_info") or {}).get("family_name") or (evaluation.get("model_info") or {}).get("name")),
                    "developer": as_string((evaluation.get("model_info") or {}).get("developer")),
                    "raw_model_id": as_string((evaluation.get("model_info") or {}).get("id")),
                    "score": score,
                    "evaluation_id": evaluation.get("evaluation_id"),
                    "retrieved_timestamp": evaluation.get("retrieved_timestamp"),
                    "source_record_url": evaluation.get("source_record_url"),
                    "detailed_evaluation_results": evaluation.get("detailed_evaluation_results"),
                    "detailed_evaluation_results_meta": evaluation.get("detailed_evaluation_results_meta"),
                    "passthrough_top_level_fields": evaluation.get("passthrough_top_level_fields"),
                    "instance_level_data": evaluation.get("instance_level_data"),
                }
            )

    peer_ranks: dict[str, dict[str, dict[str, int]]] = {}
    eval_summaries: list[dict] = []

    for summary in benchmark_groups.values():
        lower = bool(summary.get("lower_is_better"))
        model_results = sorted(summary["model_results"], key=lambda r: (r["score"], r["model_id"]))
        if not lower:
            model_results.reverse()
        summary["model_results"] = model_results
        summary["models_count"] = len(model_results)
        eval_summaries.append(summary)

        ranks: dict[str, dict[str, int]] = {}
        position = 0
        previous_score = None
        for idx, row in enumerate(model_results, start=1):
            if previous_score is None or row["score"] != previous_score:
                position = idx
                previous_score = row["score"]
            ranks[row["model_id"]] = {"position": position, "total": len(model_results)}
        peer_ranks[summary["eval_summary_id"]] = ranks

    eval_summaries.sort(key=lambda s: (-s.get("models_count", 0), as_string(s.get("eval_summary_id"))))

    model_summaries: list[dict] = []
    model_cards: list[dict] = []

    for family_id, family_evals in model_family_groups.items():
        family_evals_sorted = sorted(family_evals, key=lambda e: as_string(e.get("retrieved_timestamp")))
        latest = family_evals_sorted[-1]
        model_info = latest.get("model_info") or {}
        route_id = as_string(model_info.get("model_route_id"))
        family_name = as_string(model_info.get("family_name") or model_info.get("name") or family_id.split("/")[-1])

        by_category: dict[str, list[dict]] = defaultdict(list)
        raw_model_ids = sorted({as_string((e.get("model_info") or {}).get("id")) for e in family_evals if as_string((e.get("model_info") or {}).get("id"))})
        variants_map: dict[str, dict] = {}
        score_values: list[float] = []
        last_updated = None

        for evaluation in family_evals:
            category = infer_category_from_benchmark(as_string(evaluation.get("benchmark")))
            by_category[category].append(evaluation)
            iso = iso_from_epoch_string(evaluation.get("retrieved_timestamp"))
            last_updated = max_iso(last_updated, iso)

            model_variant_key = as_string((evaluation.get("model_info") or {}).get("variant_key") or "default")
            variant = variants_map.setdefault(
                model_variant_key,
                {
                    "variant_key": model_variant_key,
                    "variant_label": as_string((evaluation.get("model_info") or {}).get("variant_label") or "Default"),
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
                score = extract_score(result)
                if score is not None:
                    score_values.append(score)

        summary = {
            "model_info": model_info,
            "model_family_id": family_id,
            "model_route_id": route_id,
            "model_family_name": family_name,
            "raw_model_ids": raw_model_ids,
            "evaluations_by_category": dict(by_category),
            "total_evaluations": len(family_evals),
            "last_updated": last_updated,
            "categories_covered": sorted(by_category.keys()),
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
                "developer": as_string(model_info.get("developer")),
                "total_evaluations": len(family_evals),
                "benchmark_count": len({as_string(e.get("benchmark")) for e in family_evals if as_string(e.get("benchmark"))}),
                "categories_covered": sorted(by_category.keys()),
                "last_updated": last_updated,
                "variants": summary["variants"],
                "score_summary": score_summary,
            }
        )

    model_cards.sort(key=lambda m: (-m["total_evaluations"], as_string(m["model_route_id"])))
    model_summaries.sort(key=lambda m: as_string(m.get("model_route_id")))

    eval_list = {
        "evals": [
            {
                "eval_summary_id": s["eval_summary_id"],
                "benchmark": s["benchmark"],
                "evaluation_name": s["evaluation_name"],
                "lower_is_better": s["lower_is_better"],
                "models_count": s["models_count"],
                "benchmark_card": s["benchmark_card"],
                "source_data": s["source_data"],
                "metric_config": s["metric_config"],
                "top_score": s["model_results"][0]["score"] if s["model_results"] else None,
            }
            for s in eval_summaries
        ],
        "totalModels": len({r["model_id"] for s in eval_summaries for r in s["model_results"]}),
    }

    dev_group: dict[str, list[dict]] = defaultdict(list)
    for card in model_cards:
        developer = as_string(card.get("developer") or "Unknown")
        dev_group[developer].append(card)

    developers = [{"developer": dev, "model_count": len(models)} for dev, models in dev_group.items()]
    developers.sort(key=lambda d: (-d["model_count"], as_string(d["developer"])))

    dev_summaries = []
    for dev in developers:
        developer = dev["developer"]
        models = sorted(dev_group[developer], key=lambda m: as_string(m.get("model_family_name")))
        dev_summaries.append({"developer": developer, "slug": slugify_developer(developer), "models": models})

    manifest = {
        "generated_at": started_at,
        "model_count": len(model_cards),
        "eval_count": len(eval_summaries),
        "config_version": CONFIG_VERSION,
        "skipped_config_count": len(skipped_configs),
        "skipped_configs": skipped_configs,
        "source_config_count": len(all_configs),
    }

    write_json(OUTPUT_DIR / "model-cards.json", model_cards)
    write_json(OUTPUT_DIR / "eval-list.json", eval_list)
    write_json(OUTPUT_DIR / "peer-ranks.json", peer_ranks)
    write_json(OUTPUT_DIR / "benchmark-metadata.json", benchmark_metadata)
    write_json(OUTPUT_DIR / "developers.json", developers)
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    for summary in model_summaries:
        write_json(OUTPUT_DIR / "models" / f"{summary['model_route_id']}.json", summary)
    for summary in eval_summaries:
        write_json(OUTPUT_DIR / "evals" / f"{summary['eval_summary_id']}.json", summary)
    for summary in dev_summaries:
        write_json(OUTPUT_DIR / "developers" / f"{summary['slug']}.json", {"developer": summary["developer"], "models": summary["models"]})

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
        upload_output()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

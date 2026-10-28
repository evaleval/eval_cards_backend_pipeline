"""Regenerate each vendored patch table from its evidence source.

Every table this package ships is derived, not hand-written: a builder
reads the evidence (a raw EEE snapshot for the Vals.ai run settings, the
leaderboard's published harness dumps for the Open LLM v2 configs) and
writes the table as canonical JSON. Reruns on the same evidence are
byte-identical, so a table can be regenerated and diffed to review what
changed rather than trusted on sight.

Builder output that is useful for triage but not needed to apply a fix
(which files attested what, which were rejected, which gaps stayed
uncorroborated) goes to a separate diagnostics file that is not vendored.

`export_patched_records` is the same appliers run outside the pipeline:
it writes patched copies of the affected records in the upstream
datastore layout plus a manifest of every field written, so the
corrections can be reviewed, handed upstream, or replayed elsewhere.
"""
from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator

from eval_card_backend.datafix import arc_agi, hfopenllm_v2, vals_ai
from eval_card_backend.datafix.core import (
    DATA_DIR,
    apply_fixes,
    atomic_write_text,
    canonical_json,
    generation_arg,
    load_table,
)
from eval_card_backend.datafix.registry import (
    covers,
    covers_every_config,
    named_configs,
)
from eval_card_backend.datafix.table_io import read_json, write_json_bytes

EEE_DATASET = "evaleval/EEE_datastore"
HF_RESULTS_DATASET = "open-llm-leaderboard/results"

# Entries listed per bucket in a rebuild's change summary before it stops
# spelling them out and reports a count instead.
LIST_LIMIT = 20


# ---------------------------------------------------------------------------
# vals_ai_max_tokens: the Vals.ai documented run settings
# ---------------------------------------------------------------------------

# These statements are frozen evidence for the interim overlay. Source adapters
# verify their own methodology at runtime; the builder retains the reviewed copy.
DOCUMENTED_STATEMENTS = load_table(vals_ai.TABLE_NAME)["documented_statements"]
VALS_VERIFIED_ON = next(iter(next(iter(DOCUMENTED_STATEMENTS.values())).values()))["verified_on"]
VALS_PAGES = {slug: {k: v for k, v in next(iter(fields.values())).items()
                     if k not in ("value", "verified_on")}
              for slug, fields in DOCUMENTED_STATEMENTS.items()}
VALS_DOCUMENTED_VALUES = {field: statement["value"]
                          for field, statement in DOCUMENTED_STATEMENTS["sage"].items()}


def iter_config_records(eee_root: Path, config: str) -> Iterator[tuple[str, dict]]:
    """Walk one config's `data/` subtree, yielding (relative path, record).

    Paths are relative to `eee_root` and yielded in sorted order, which is
    what makes every builder and the export deterministic.
    """
    root = Path(eee_root) / "data" / config
    for path in sorted(root.rglob("*.json")):
        yield path.relative_to(eee_root).as_posix(), json.loads(
            path.read_text(encoding="utf-8")
        )


def _stated_values(record: dict, field: str) -> list[Any]:
    """Every value one record actually states for `field`."""
    stated: list[Any] = []
    for result in record.get("evaluation_results") or []:
        if not isinstance(result, dict):
            continue
        status, value = generation_arg(result, field)
        if status == "present":
            stated.append(value)
    return stated


def build_vals_table(
    eee_root: Path, eee_revision: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Patch table + diagnostics for the Vals.ai documented run settings."""
    records: list[tuple[str, str, dict]] = []
    seen: Counter = Counter()
    for path, record in iter_config_records(Path(eee_root), vals_ai.SOURCE_CONFIG):
        for result in record.get("evaluation_results") or []:
            if isinstance(result, dict):
                seen[(record.get("evaluation_id"), result.get("evaluation_result_id"))] += 1
        slug = vals_ai.benchmark_slug(record)
        if slug in DOCUMENTED_STATEMENTS:
            records.append((path, slug, record))
    duplicates = sorted(str(key) for key, n in seen.items() if n > 1)
    if duplicates:
        raise RuntimeError(
            f"{vals_ai.SOURCE_CONFIG}: (evaluation_id, evaluation_result_id) is "
            f"not unique across the config: {len(duplicates)} duplicated key(s), "
            f"first: {duplicates[0]}. The patch table cannot address a result "
            f"unambiguously; resolve upstream before rebuilding."
        )

    patches = []
    for _, slug, record in records:
        for result in record.get("evaluation_results") or []:
            if not isinstance(result, dict):
                continue
            for field, statement in sorted(DOCUMENTED_STATEMENTS[slug].items()):
                if generation_arg(result, field)[0] == "missing":
                    patches.append({
                        "evaluation_id": record.get("evaluation_id"),
                        "evaluation_result_id": result.get("evaluation_result_id"),
                        "field": field, "expected_old": None,
                        "new": statement["value"], "tier": "benchmark_page",
                    })
    uncorroborated = []
    patches.sort(
        key=lambda p: (p["evaluation_id"], p["evaluation_result_id"] or "", p["field"])
    )
    uncorroborated.sort(
        key=lambda u: (u["record_path"], u["field"], u["result_index"])
    )
    table = {
        "fix_id": vals_ai.FIX_ID,
        "source": {"dataset": EEE_DATASET, "revision": eee_revision},
        "documented_statements": DOCUMENTED_STATEMENTS,
        "policy": "benchmark_page_default",
        "patches": patches,
    }
    diagnostics = {
        "fix_id": vals_ai.FIX_ID,
        "source": {"dataset": EEE_DATASET, "revision": eee_revision},
        "records_in_documented_benchmarks": len(records),
        "uncorroborated": uncorroborated,
    }
    return table, diagnostics


# ---------------------------------------------------------------------------
# hfopenllm_v2_harness_configs: the published harness configs
# ---------------------------------------------------------------------------

# Which harness tasks make up each benchmark the leaderboard reports, and
# how many subtasks a dump must carry to attest the benchmark as a whole.
TASK_PREFIXES: dict[str, tuple[str, ...]] = {
    "IFEval": ("leaderboard_ifeval",),
    "MATH Level 5": ("leaderboard_math_",),
    "BBH": ("leaderboard_bbh_",),
    "GPQA": ("leaderboard_gpqa_",),
    "MUSR": ("leaderboard_musr_",),
    "MMLU-PRO": ("leaderboard_mmlu_pro",),
}
MATH_SUFFIX = "_hard"
COMPLETE_SUBTASKS: dict[str, int] = {"IFEval": 1, "MATH Level 5": 7, "MMLU-PRO": 1}

# Fields kept per benchmark. Multiple-choice benchmarks are scored by
# loglikelihood, so they have no decoding settings to record.
GENERATIVE_FIELDS = ("output_type", "temperature", "max_tokens", "num_fewshot", "do_sample")
MULTIPLE_CHOICE_FIELDS = ("output_type", "num_fewshot")


def _tasks_of(benchmark: str, configs: dict[str, Any]) -> dict[str, Any]:
    prefixes = TASK_PREFIXES[benchmark]
    out = {}
    for task, config in configs.items():
        if not any(
            task == p or (p.endswith("_") and task.startswith(p)) for p in prefixes
        ):
            continue
        if benchmark == "MATH Level 5" and not task.endswith(MATH_SUFFIX):
            continue
        out[task] = config
    return out


def _harness_config(task_config: dict[str, Any]) -> dict[str, Any]:
    kwargs = task_config.get("generation_kwargs") or {}
    output_type = task_config.get("output_type")
    fields = (
        GENERATIVE_FIELDS
        if output_type == "generate_until"
        else MULTIPLE_CHOICE_FIELDS
    )
    values = {
        "output_type": output_type,
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_gen_toks"),
        "num_fewshot": task_config.get("num_fewshot"),
        "do_sample": kwargs.get("do_sample"),
    }
    return {f: values[f] for f in fields}


# A run evaluates `pretrained=` directly, or an adapter / delta checkpoint
# loaded on top of it; in the latter case the dump is named after the adapter.
_IDENTITY_ARGS = ("pretrained", "peft", "delta")


def _launched_models(model_args: Any) -> dict[str, str]:
    if not isinstance(model_args, str):
        return {}
    launched = {}
    for part in model_args.split(","):
        key, sep, value = part.partition("=")
        if sep and key in _IDENTITY_ARGS:
            launched[key] = value
    return launched


def _dump_identity(dump: dict[str, Any]) -> tuple[str | None, str | None]:
    """The model the dump is about, and why it can't be trusted.

    Identity is the dump's own `model_name`, cross-checked against the
    model arguments the run was launched with. The directory a dump
    sits in is never used: ids differing only in case share a directory,
    and at least one dump sits under a directory naming another model.

    A run that loads an adapter or a delta checkpoint evaluated those
    weights, not the base model, so the dump has to be named after the
    adapter; one named after `pretrained=` is rejected rather than
    attributed to the base model. A run naming both an adapter and a
    delta says nothing about which weights produced the numbers.
    """
    name = dump.get("model_name")
    if not isinstance(name, str) or not name:
        return None, "no_model_name"
    launched = _launched_models((dump.get("config") or {}).get("model_args"))
    if "pretrained" not in launched:
        return None, "no_pretrained"
    if "peft" in launched and "delta" in launched:
        return None, "ambiguous_adapter_args"
    expected = launched.get("peft", launched.get("delta", launched["pretrained"]))
    if name != expected:
        return None, "identity_contradicts_model_args"
    return name, None


def _legacy_harness_summary(
    results_dir: Path, revision: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Patch table + diagnostics for the Open LLM v2 harness configs."""
    results_root = Path(results_dir)
    # Per (model, benchmark): the configs attested, and the files that did.
    attested: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    conflicts: dict[str, set[str]] = defaultdict(set)
    rejected: list[dict[str, Any]] = []
    n_files = 0

    for path in sorted(results_root.rglob("results_*.json")):
        n_files += 1
        rel = path.relative_to(results_root).as_posix()
        try:
            dump = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            rejected.append({"path": rel, "reason": "unreadable", "detail": str(exc)})
            continue
        model_id, reason = _dump_identity(dump)
        if model_id is None:
            rejected.append({
                "path": rel,
                "reason": reason,
                "model_name": dump.get("model_name"),
                "model_args": (dump.get("config") or {}).get("model_args"),
            })
            continue
        configs = dump.get("configs") or {}
        for benchmark in TASK_PREFIXES:
            tasks = _tasks_of(benchmark, configs)
            if not tasks:
                continue
            stated = {canonical_json(_harness_config(c), indent=None) for c in tasks.values()}
            complete = COMPLETE_SUBTASKS.get(benchmark)
            if len(stated) != 1 or (complete is not None and len(tasks) != complete):
                conflicts[model_id].add(benchmark)
                continue
            attested[model_id][benchmark].append((stated.pop(), rel))

    # A file that disagreed with its peers taints the pair for good, so the
    # cross-file agreement check runs after every file has been read.
    verified: dict[str, list[str]] = {b: [] for b in TASK_PREFIXES}
    per_model: dict[str, dict[str, dict[str, Any]]] = {}
    attesting_files: dict[str, dict[str, list[str]]] = {}
    for model_id, by_benchmark in attested.items():
        for benchmark, entries in by_benchmark.items():
            if benchmark in conflicts.get(model_id, ()):
                continue
            stated = {config for config, _ in entries}
            if len(stated) != 1:
                conflicts[model_id].add(benchmark)
                continue
            verified[benchmark].append(model_id)
            per_model.setdefault(model_id, {})[benchmark] = json.loads(stated.pop())
            attesting_files.setdefault(model_id, {})[benchmark] = sorted(
                rel for _, rel in entries
            )

    defaults: dict[str, dict[str, Any]] = {}
    for benchmark in TASK_PREFIXES:
        tally: Counter = Counter()
        for model_id in verified[benchmark]:
            tally[canonical_json(per_model[model_id][benchmark], indent=None)] += 1
        if not tally:
            continue
        # The majority config is the default; anything else is an override.
        defaults[benchmark] = json.loads(tally.most_common(1)[0][0])

    overrides: dict[str, dict[str, Any]] = {}
    for model_id, by_benchmark in sorted(per_model.items()):
        for benchmark, config in sorted(by_benchmark.items()):
            if config != defaults.get(benchmark):
                overrides.setdefault(model_id, {})[benchmark] = config

    table = {
        "fix_id": hfopenllm_v2.FIX_ID,
        "source": {"dataset": HF_RESULTS_DATASET, "revision": revision},
        "defaults": defaults,
        "overrides": overrides,
        "conflicts": {m: sorted(b) for m, b in sorted(conflicts.items()) if b},
        "verified": {b: sorted(models) for b, models in sorted(verified.items())},
    }
    diagnostics = {
        "fix_id": hfopenllm_v2.FIX_ID,
        "source": {"dataset": HF_RESULTS_DATASET, "revision": revision},
        "files_read": n_files,
        "files_rejected": rejected,
        "attesting_files": attesting_files,
    }
    return table, diagnostics


def build_hfopenllm_table(results_dir: Path, revision: str):
    """Freeze complete run metadata keyed by exact model and precision."""
    from eval_card_backend.datafix.harness_evidence import build_dump_index

    table, diagnostics = _legacy_harness_summary(results_dir, revision)
    index = build_dump_index(results_dir)
    table["by_precision"] = {}
    table["profiles"] = {}
    table["libraries"] = {}
    names = dict(zip(("ifeval", "math", "bbh", "gpqa", "musr", "mmlu_pro"), TASK_PREFIXES))
    for (model, precision), evidence in sorted(index.entries.items()):
        library = {key: getattr(evidence, key) for key in ("git_hash", "transformers_version")
                   if getattr(evidence, key) is not None}
        table["libraries"].setdefault(model, {})[precision] = library
        for key, group in evidence.groups.items():
            profile = {name: value for name, value in vars(group).items()
                       if value is not None and name != "gaps"}
            digest = hashlib.sha256(canonical_json(profile).encode()).hexdigest()[:20]
            table["profiles"][digest] = profile
            table["by_precision"].setdefault(model, {}).setdefault(precision, {})[names[key]] = digest
    diagnostics["precise_index"] = {"files_read": index.files_read,
                                   "rejected": index.rejected,
                                   "entries": len(index.entries)}
    return table, diagnostics


# ---------------------------------------------------------------------------
# arc_agi_logged_request_args: the published run outputs
# ---------------------------------------------------------------------------

# One published dataset per public set, pinned. Nothing is ever carried from
# one set to another: a configuration was observed running under different
# caps on the two, and the semi-private sets publish no outputs at all.
ARC_PUBLIC_RUNS: dict[str, dict[str, str]] = {
    "v1_Public_Eval": {
        "dataset": "arcprize/arc_agi_v1_public_eval",
        "revision": "3e9c9d1a8402aff82356815c106cecaea65cb7d9",
        "directory": "v1",
    },
    "v2_Public_Eval": {
        "dataset": "arcprize/arc_agi_v2_public_eval",
        "revision": "026789c1c12a4c34580a32e84dcaf5630d7e8f31",
        "directory": "v2",
    },
}

# The output cap a request may state. Exactly one of them, and the same one,
# on every request of a run: they are provider-specific spellings, and a run
# that mixes them was not run under one setting.
ARC_CAP_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")

ARC_RESULTS_FILE = "results.json"
ARC_COST_METRIC_ID = "cost_per_task"
_ATTEMPT_KEY = re.compile(r"^attempt_\d+$")


def read_run_history(repo_dir: Path, revision: str) -> dict[str, int]:
    """How many commits touched each path, up to `revision`.

    One `git log` over the whole dataset repository, not a request per file.
    A path a second commit touched was rewritten after it was first
    published, so the directory holding it no longer stands for one run.
    Paths that no longer exist are in here too, which is the point: a file
    that was published and then deleted is a rewrite of the directory just
    as much as one that was published twice.
    """
    log = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "--name-only", "--no-renames",
         "--pretty=format:", revision],
        capture_output=True, text=True, check=True,
    )
    return dict(Counter(line for line in log.stdout.splitlines() if line.strip()))


def _rounds_to(value: float, stated: float) -> bool:
    """Whether `value` rounds to `stated` at the precision `stated` carries.

    The published numbers are rounded, so an exact comparison would reject
    every run; a fixed epsilon would accept runs the number does not
    identify. The tolerance is therefore half a unit in the last place the
    stated number actually spells out, and nothing wider.
    """
    stated_decimal = Decimal(repr(stated))
    half = Decimal(1).scaleb(stated_decimal.as_tuple().exponent) / 2
    return abs(Decimal(repr(value)) - stated_decimal) <= half


def _sums_to(value: float, stated: float, terms: int) -> bool:
    """Whether a sum of `terms` task scores matches one summed elsewhere.

    Adding the same numbers up in a different order lands a few units in the
    last place away, and the published aggregate is written out in full
    precision, so this one comparison carries that allowance. It stays far
    below the smallest real disagreement, which is one task's score.
    """
    spread = terms * sys.float_info.epsilon * max(abs(value), abs(stated), 1.0)
    return abs(value - stated) <= spread


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class _RunRejected(Exception):
    """One published run directory cannot stand for a run."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _attempt_cap(kwargs: dict[str, Any]) -> tuple[str, int]:
    stated = [key for key in ARC_CAP_KEYS if key in kwargs]
    if not stated:
        raise _RunRejected("cap_absent")
    if len(stated) > 1:
        raise _RunRejected("cap_ambiguous", ", ".join(stated))
    value = kwargs[stated[0]]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _RunRejected("cap_not_an_integer", f"{stated[0]}={value!r}")
    return stated[0], value


def _read_run(directory: Path, history: dict[str, int]) -> dict[str, Any]:
    """Reconstruct one published run from its directory.

    Raises `_RunRejected` with the gate that failed. What comes back is the
    run as the logs state it: the score reduction recomputed task by task,
    the request arguments every attempt shares, and when the last attempt
    ended.
    """
    results_path = directory / ARC_RESULTS_FILE
    if not results_path.is_file():
        raise _RunRejected("no_results_file")
    try:
        published = json.loads(results_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _RunRejected("results_file_unreadable", str(exc)) from exc
    task_results = published.get("task_results")
    if not isinstance(task_results, dict):
        raise _RunRejected("results_file_without_task_results")

    task_paths = sorted(
        p for p in directory.glob("*.json") if p.name != ARC_RESULTS_FILE
    )
    if not task_paths:
        raise _RunRejected("no_task_files")

    scores: dict[str, float] = {}
    attempts_seen = 0
    cap: tuple[str, int] | None = None
    temperatures: list[Any] = []
    latest_end: datetime | None = None
    # Every path the directory ever held, not only the ones still in it: a
    # run whose files were replaced or removed after publication is no
    # longer one run, whatever is left on disk.
    prefix = f"{directory.name}/"
    rewritten = sorted(
        path for path, commits in history.items()
        if path.startswith(prefix) and commits > 1
    )
    if rewritten:
        raise _RunRejected("run_rewritten", ", ".join(rewritten[:5]))

    for path in task_paths:
        task_id = path.stem
        try:
            pairs = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise _RunRejected("task_file_unreadable", f"{path.name}: {exc}") from exc
        if not isinstance(pairs, list) or not pairs:
            raise _RunRejected("task_file_not_a_pair_list", path.name)
        solved = []
        for index, pair in enumerate(pairs):
            if not isinstance(pair, dict) or not pair:
                raise _RunRejected("task_file_not_a_pair_list", path.name)
            correct = False
            for key, attempt in pair.items():
                if not _ATTEMPT_KEY.match(key):
                    raise _RunRejected("attempt_key_unknown", f"{path.name}: {key}")
                attempts_seen += 1
                if not isinstance(attempt, dict):
                    raise _RunRejected("attempt_not_recorded", f"{path.name}: {key}")
                metadata = attempt.get("metadata")
                if not isinstance(metadata, dict):
                    raise _RunRejected("attempt_without_metadata", f"{path.name}: {key}")
                if metadata.get("task_id") != task_id:
                    raise _RunRejected(
                        "task_id_mismatch", f"{path.name}: {metadata.get('task_id')!r}"
                    )
                if metadata.get("test_id") != directory.name:
                    raise _RunRejected(
                        "test_id_mismatch", f"{path.name}: {metadata.get('test_id')!r}"
                    )
                pair_index = metadata.get("pair_index")
                if pair_index is None:
                    raise _RunRejected("pair_index_absent", f"{path.name}: {key}")
                if (
                    isinstance(pair_index, bool)
                    or not isinstance(pair_index, int)
                    or pair_index != index
                ):
                    raise _RunRejected(
                        "pair_index_mismatch", f"{path.name}: {pair_index!r} at {index}"
                    )
                end = _parse_timestamp(metadata.get("end_timestamp"))
                if end is None:
                    raise _RunRejected("attempt_without_end_time", f"{path.name}: {key}")
                latest_end = end if latest_end is None else max(latest_end, end)
                kwargs = metadata.get("kwargs")
                if not isinstance(kwargs, dict):
                    raise _RunRejected("attempt_without_kwargs", f"{path.name}: {key}")
                attempt_cap = _attempt_cap(kwargs)
                if cap is None:
                    cap = attempt_cap
                elif attempt_cap != cap:
                    raise _RunRejected("cap_conflict", f"{cap} vs {attempt_cap}")
                temperatures.append(kwargs.get("temperature"))
                if not isinstance(attempt.get("correct"), bool):
                    raise _RunRejected("correct_not_recorded", f"{path.name}: {key}")
                correct = correct or attempt["correct"]
            solved.append(correct)
        scores[task_id] = sum(solved) / len(solved)

    if cap is None:
        raise _RunRejected("no_attempts")
    if set(scores) != set(task_results):
        missing = sorted(set(task_results) - set(scores))
        extra = sorted(set(scores) - set(task_results))
        raise _RunRejected(
            "task_set_mismatch",
            f"{len(missing)} not published as files, {len(extra)} not in results.json",
        )
    for task_id, score in sorted(scores.items()):
        stated = (task_results[task_id] or {}).get("score")
        if not isinstance(stated, (int, float)) or not _rounds_to(score, float(stated)):
            raise _RunRejected("task_score_mismatch", f"{task_id}: {stated!r}")
    for name, recomputed, terms in (
        ("total_tasks", len(scores), 0),
        ("total_attempts", attempts_seen, 0),
        ("score", sum(scores.values()), len(scores)),
    ):
        stated = published.get(name)
        if not isinstance(stated, (int, float)) or not (
            _sums_to(float(recomputed), float(stated), terms) if terms
            else _rounds_to(float(recomputed), float(stated))
        ):
            raise _RunRejected("run_score_mismatch", f"{name}: {stated!r} vs {recomputed}")

    # A temperature the requests disagree on, or that only some of them
    # state, is no single temperature for the run. That says nothing about
    # the cap, so the run still stands and only the temperature is dropped.
    temperature: float | None = None
    temperature_reason: str | None = None
    value = temperatures[0]
    if any(t is None for t in temperatures):
        temperature_reason = (
            "temperature_absent" if all(t is None for t in temperatures)
            else "temperature_partial"
        )
    elif len({repr(t) for t in temperatures}) > 1:
        temperature_reason = "temperature_conflict"
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        temperature_reason = "temperature_not_a_number"
    else:
        temperature = float(value)

    return {
        "cap_key": cap[0],
        "cap_value": cap[1],
        "temperature": temperature,
        "temperature_reason": temperature_reason,
        "score": sum(scores.values()) / len(scores),
        "avg_cost_per_task": published.get("avg_cost_per_task"),
        "latest_end": latest_end,
        "total_tasks": len(scores),
        "total_attempts": attempts_seen,
    }


def _arc_results(record: dict[str, Any], result_name: str) -> dict[str, Any]:
    """The score and cost results one record states for one public set."""
    found: dict[str, Any] = {}
    for idx, result in enumerate(record.get("evaluation_results") or []):
        if not isinstance(result, dict) or result.get("evaluation_name") != result_name:
            continue
        metric = (result.get("metric_config") or {}).get("metric_id")
        if metric in (arc_agi.SCORE_METRIC_ID, ARC_COST_METRIC_ID):
            found[metric] = {"index": idx, "result": result}
    return found


def _stated_score(entry: dict[str, Any] | None) -> float | None:
    if entry is None:
        return None
    score = (entry["result"].get("score_details") or {}).get("score")
    return float(score) if isinstance(score, (int, float)) else None


def build_arc_table(
    eee_root: Path,
    eee_revision: str,
    runs_root: Path,
    histories: dict[str, dict[str, int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Patch table + diagnostics for the ARC Prize published run outputs.

    A directory of logs becomes a patch only when it survives every gate:

    1. the directory names the run: every attempt is logged under the
       directory's own test id, every task file states the task id its name
       carries, and every attempt states, as an integer, the pair index it
       sits at;
    2. the task files reconcile exactly with `results.json`, and the
       leaderboard's reduction recomputes from the logs: a test pair counts
       as solved when any attempt of it is correct, a task scores the
       fraction of its pairs solved, and the run's score is those task
       scores summed, which divided by the task count is the published
       proportion. Task count, attempt count, every task score and the
       aggregate all have to agree with `results.json`;
    3. every request states exactly one output cap, under the same key name,
       as the same integer. Temperature is taken only when every request
       states the same one; stated by some requests and not others, it is
       left missing;
    4. the recomputed proportion equals the record's score, and
       `results.json`'s average cost per task equals the record's paired
       cost result, each at the precision the record spells out;
    5. no attempt ended after the record was retrieved, and no path the
       directory ever held was touched by a second commit of the dataset;
    6. the shas are pinned, and a v1 directory only ever fills a v1 result.

    A model appears in several records as the leaderboard is re-read, so
    gates 4 and 5 are re-checked per record and every record that passes is
    patched.
    """
    root = Path(eee_root)
    by_raw_id: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path, record in iter_config_records(root, arc_agi.SOURCE_CONFIG):
        raw_id = arc_agi.raw_model_id(record)
        if raw_id is not None:
            by_raw_id[raw_id].append((path, record))

    patches: list[dict[str, Any]] = []
    rejected_runs: list[dict[str, Any]] = []
    rejected_results: list[dict[str, Any]] = []
    temperature_not_taken: list[dict[str, Any]] = []
    reported: set[tuple[str, str]] = set()
    runs_read = 0

    for result_name in sorted(ARC_PUBLIC_RUNS):
        source = ARC_PUBLIC_RUNS[result_name]
        dataset_root = Path(runs_root) / source["directory"]
        history = histories.get(result_name, {})
        directories = sorted(
            p for p in dataset_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        for directory in directories:
            runs_read += 1
            records = by_raw_id.get(directory.name, [])
            if not records:
                rejected_runs.append({
                    "result_name": result_name,
                    "directory": directory.name,
                    "reason": "no_record_for_this_model",
                    "detail": "",
                })
                continue
            try:
                run = _read_run(directory, history)
            except _RunRejected as exc:
                rejected_runs.append({
                    "result_name": result_name,
                    "directory": directory.name,
                    "reason": exc.reason,
                    "detail": exc.detail,
                })
                continue

            fields: dict[str, Any] = {"max_tokens": run["cap_value"]}
            if run["temperature"] is not None:
                fields["temperature"] = run["temperature"]
            else:
                temperature_not_taken.append({
                    "result_name": result_name,
                    "directory": directory.name,
                    "reason": run["temperature_reason"],
                })

            for path, record in records:
                results = _arc_results(record, result_name)
                score_entry = results.get(arc_agi.SCORE_METRIC_ID)
                if score_entry is None:
                    continue
                reason = _arc_record_rejection(record, results, run)
                if reason is not None:
                    reported.add((path, result_name))
                    rejected_results.append({
                        "record_path": path,
                        "result_name": result_name,
                        "directory": directory.name,
                        "evaluation_result_id": score_entry["result"].get(
                            "evaluation_result_id"
                        ),
                        "reason": reason,
                    })
                    continue
                reported.add((path, result_name))
                patches.append({
                    "evaluation_id": record.get("evaluation_id"),
                    "evaluation_result_id": score_entry["result"].get(
                        "evaluation_result_id"
                    ),
                    "evaluation_name": result_name,
                    "expected_old": None,
                    "fields": fields,
                    "cap_key": run["cap_key"],
                    "dataset": source["dataset"],
                    "revision": source["revision"],
                    "directory": directory.name,
                    "metadata": {
                        "source_data/additional_details/scored_tasks": str(run["total_tasks"]),
                        "score_details/uncertainty/num_samples": run["total_tasks"],
                        "generation_config/additional_details/cap_key": run["cap_key"],
                        "generation_config/additional_details/request_log":
                            f"{source['dataset']}@{source['revision']}/{directory.name}",
                    },
                })

    # Everything in scope that no published run reached, so the diagnostics
    # account for every public score result, not only the ones a run failed.
    for result_name in sorted(ARC_PUBLIC_RUNS):
        for records in by_raw_id.values():
            for path, record in records:
                entry = _arc_results(record, result_name).get(arc_agi.SCORE_METRIC_ID)
                if entry is None or (path, result_name) in reported:
                    continue
                rejected_results.append({
                    "record_path": path,
                    "result_name": result_name,
                    "directory": arc_agi.raw_model_id(record),
                    "evaluation_result_id": entry["result"].get("evaluation_result_id"),
                    "reason": "no_published_run",
                })

    rejected_patches = []
    accepted = {(p["evaluation_id"], p["evaluation_result_id"]) for p in patches}
    for result_name, source in sorted(ARC_PUBLIC_RUNS.items()):
        for model_id, records in sorted(by_raw_id.items()):
            directory = Path(runs_root) / source["directory"] / model_id
            if not directory.is_dir():
                continue
            try:
                run = _read_run(directory, histories.get(result_name, {}))
                run_reason = None
            except _RunRejected as exc:
                run = None
                run_reason = exc.reason
            for _, record in records:
                results = _arc_results(record, result_name)
                entry = results.get(arc_agi.SCORE_METRIC_ID)
                if entry is None:
                    continue
                result_id = entry["result"].get("evaluation_result_id")
                if (record.get("evaluation_id"), result_id) in accepted:
                    continue
                reason = run_reason or _arc_record_rejection(record, results, run)
                if reason is None:
                    continue
                rejected_patches.append({
                    "evaluation_id": record.get("evaluation_id"),
                    "evaluation_result_id": result_id,
                    "evaluation_name": result_name, "directory": model_id,
                    "expected_old": None, "fields": {},
                    "metadata": {
                        "generation_config/additional_details/request_args_unavailable": reason,
                        "generation_config/additional_details/request_log":
                            f"{source['dataset']}@{source['revision']}/{model_id}",
                    },
                })
    rejected_patches.sort(key=lambda p: (p["evaluation_id"], p["evaluation_result_id"] or ""))
    patches.sort(key=lambda p: (p["evaluation_id"], p["evaluation_result_id"] or ""))
    rejected_runs.sort(key=lambda r: (r["result_name"], r["directory"]))
    temperature_not_taken.sort(key=lambda r: (r["result_name"], r["directory"]))
    rejected_results.sort(
        key=lambda r: (r["record_path"], r["result_name"], r["reason"])
    )
    table = {
        "fix_id": arc_agi.FIX_ID,
        "source": {
            "datasets": {
                name: {"dataset": s["dataset"], "revision": s["revision"]}
                for name, s in sorted(ARC_PUBLIC_RUNS.items())
            },
            "eee": {"dataset": EEE_DATASET, "revision": eee_revision},
            "revision": "+".join(
                ARC_PUBLIC_RUNS[name]["revision"] for name in sorted(ARC_PUBLIC_RUNS)
            ),
        },
        "patches": patches,
        "rejected_patches": rejected_patches,
    }
    diagnostics = {
        "fix_id": arc_agi.FIX_ID,
        "source": table["source"],
        "runs_read": runs_read,
        "rejected_runs": rejected_runs,
        "rejected_results": rejected_results,
        "temperature_not_taken": temperature_not_taken,
    }
    return table, diagnostics


def _arc_record_rejection(
    record: dict[str, Any], results: dict[str, Any], run: dict[str, Any]
) -> str | None:
    """Why this record's score is not the run in the logs, or None."""
    stated_score = _stated_score(results.get(arc_agi.SCORE_METRIC_ID))
    if stated_score is None:
        return "score_not_stated"
    if not _rounds_to(run["score"], stated_score):
        return "score_mismatch"
    stated_cost = _stated_score(results.get(ARC_COST_METRIC_ID))
    if stated_cost is None:
        return "cost_not_stated"
    cost = run["avg_cost_per_task"]
    if not isinstance(cost, (int, float)) or not _rounds_to(float(cost), stated_cost):
        return "cost_mismatch"
    retrieved = _parse_timestamp(record.get("retrieved_timestamp"))
    if retrieved is None:
        return "retrieved_time_unreadable"
    if run["latest_end"] > retrieved:
        return "run_ended_after_retrieval"
    return None


# ---------------------------------------------------------------------------
# Writing tables + exporting patched records
# ---------------------------------------------------------------------------


def write_table(name: str, table: dict[str, Any]) -> Path:
    path = DATA_DIR / name
    write_json_bytes(path, canonical_json(table).encode("utf-8"))
    return path


def vendored_table(name: str) -> dict[str, Any] | None:
    """The table as currently shipped, or None when none is vendored yet."""
    path = DATA_DIR / name
    if not path.exists():
        return None
    return read_json(path)


def table_differs(old: dict[str, Any], new: dict[str, Any]) -> bool:
    return canonical_json(old) != canonical_json(new)


def _capped(items: list[str], limit: int = LIST_LIMIT) -> list[str]:
    if len(items) <= limit:
        return items
    return items[:limit] + [f"  … and {len(items) - limit} more"]


def _source_revision_line(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    before = (old.get("source") or {}).get("revision")
    after = (new.get("source") or {}).get("revision")
    if before == after:
        return []
    return [f"  evidence revision {before} -> {after}"]


def vals_table_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """What a rebuild of the run-settings table would change, patch by patch."""

    def by_key(table: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
        return {
            (p["evaluation_id"], p.get("evaluation_result_id"), p["field"]): p
            for p in table.get("patches") or []
        }

    def named(key: tuple) -> str:
        return f"{key[0]} / {key[1]} {key[2]}"

    before, after = by_key(old), by_key(new)
    added, removed = sorted(set(after) - set(before)), sorted(set(before) - set(after))
    lines = _source_revision_line(old, new)
    for label, keys in (("added", added), ("removed", removed)):
        for field, count in sorted(Counter(key[2] for key in keys).items()):
            lines.append(f"  {label} {field} patches: {count}")
    lines += _capped([
        f"  + {named(key)} (tier {after[key].get('tier')})" for key in added
    ])
    lines += _capped([
        f"  - {named(key)} (tier {before[key].get('tier')})" for key in removed
    ])
    lines += _capped([
        f"  ~ {named(key)}: tier {before[key].get('tier')} -> "
        f"{after[key].get('tier')}, value {before[key].get('new')} -> "
        f"{after[key].get('new')}"
        for key in sorted(set(before) & set(after))
        if before[key] != after[key]
    ])
    if (old.get("documented_statements") or {}) != (
        new.get("documented_statements") or {}
    ):
        lines.append("  documented statements changed")
    return lines


def hfopenllm_table_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """What a rebuild of the harness-config table would change."""
    lines = _source_revision_line(old, new)

    old_defaults = old.get("defaults") or {}
    new_defaults = new.get("defaults") or {}
    lines += _capped([
        f"  default {benchmark}: "
        f"{canonical_json(old_defaults.get(benchmark), indent=None).rstrip()} -> "
        f"{canonical_json(new_defaults.get(benchmark), indent=None).rstrip()}"
        for benchmark in sorted(set(old_defaults) | set(new_defaults))
        if old_defaults.get(benchmark) != new_defaults.get(benchmark)
    ])

    for label, key in (("override", "overrides"), ("conflict", "conflicts")):
        before = old.get(key) or {}
        after = new.get(key) or {}
        lines += _capped([f"  + {label} {model}" for model in sorted(set(after) - set(before))])
        lines += _capped([f"  - {label} {model}" for model in sorted(set(before) - set(after))])
        lines += _capped([
            f"  ~ {label} {model} changed"
            for model in sorted(set(before) & set(after))
            if before[model] != after[model]
        ])

    old_verified = old.get("verified") or {}
    new_verified = new.get("verified") or {}
    for benchmark in sorted(set(old_verified) | set(new_verified)):
        before = set(old_verified.get(benchmark) or ())
        after = set(new_verified.get(benchmark) or ())
        if before == after:
            continue
        lines.append(
            f"  verified {benchmark}: +{len(after - before)} / -{len(before - after)} "
            f"model(s), {len(before)} -> {len(after)}"
        )
    return lines


def arc_table_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """What a rebuild of the logged-request-args table would change."""

    def by_key(table: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
        return {
            (p["evaluation_id"], p.get("evaluation_result_id")): p
            for p in table.get("patches") or []
        }

    def summary(patch: dict[str, Any]) -> str:
        fields = canonical_json(patch.get("fields"), indent=None).rstrip()
        return f"{patch.get('directory')} {fields}"

    before, after = by_key(old), by_key(new)
    lines = _source_revision_line(old, new)
    lines += _capped([
        f"  + {key[0]} / {key[1]} ({summary(after[key])})"
        for key in sorted(set(after) - set(before))
    ])
    lines += _capped([
        f"  - {key[0]} / {key[1]} ({summary(before[key])})"
        for key in sorted(set(before) - set(after))
    ])
    lines += _capped([
        f"  ~ {key[0]} / {key[1]}: {summary(before[key])} -> {summary(after[key])}"
        for key in sorted(set(before) & set(after))
        if before[key] != after[key]
    ])
    return lines


def _export_configs(root: Path, configs: Iterable[str] | None) -> list[str]:
    """Which configs an export walks.

    A fix registered for every config makes the whole datastore the default,
    so the default is read off the tree rather than out of the registry;
    anything asked for that no fix covers is dropped.
    """
    if configs is None:
        configs = (
            sorted(p.name for p in (root / "data").iterdir() if p.is_dir())
            if covers_every_config()
            else sorted(named_configs())
        )
    return sorted({config for config in configs if covers(config)})


def export_patched_records(
    eee_root: Path, out_dir: Path, configs: Iterable[str] | None = None
) -> dict[str, int]:
    """Write patched copies of every affected record, plus a patch manifest.

    Output mirrors the upstream `data/<config>/…` layout and carries only
    the records a fix changed. Manifest rows are ordered by record path,
    then result index, pointer and fix id; paths are relative to the
    export root and nothing records a wall-clock time, so two exports of
    the same inputs are byte-identical.

    The output directory must be empty or absent: an export that wrote
    into a populated one would leave records from an earlier run standing
    next to a manifest that does not list them.
    """
    root = Path(eee_root)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} is not empty. Export writes only the records a fix "
            f"changed, so anything already there would survive as a record "
            f"the fresh manifest does not list. Point --out at a new or "
            f"empty directory."
        )
    wanted = _export_configs(root, list(configs) if configs else None)
    counts = Counter()
    manifest_lines: list[str] = []

    for config in wanted:
        for path, record in iter_config_records(root, config):
            counts["records_scanned"] += 1
            patched, events, _ = apply_fixes(record, config)
            if not events:
                continue
            counts["records_patched"] += 1
            counts["results_patched"] += len({e.result_index for e in events})
            counts["fields_patched"] += len(events)
            target = out / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(patched, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            for event in sorted(events, key=lambda e: e.sort_key):
                manifest_lines.append(
                    canonical_json(event.as_row(path), indent=None).rstrip("\n")
                )

    atomic_write_text(
        out / "patch_manifest.jsonl",
        "".join(line + "\n" for line in manifest_lines),
    )
    return dict(counts)

"""AISI inference-scaling collection extractor (v4).

Reassembles the UK AISI inference-scaling batch (arXiv 2606.17930) from the
EEE datastore's per-sample JSONLs into complete per-setting results:

- every sample row is one COMPLETE, self-contained trajectory (v4 finding:
  zero prefix-containment across all 36,400 rows — there are no installment
  pieces at trajectory level; the installment bias lives in the RECORD
  aggregates, which re-run only unsolved tasks),
- each (benchmark, model, protocol point) cell is aggregated into one
  synthetic EEE-shaped result using the benchmark's DECLARED score source
  (`BENCHMARKS` below, mirrored in collections_curated.yaml) — proven per
  run by recomputing every member record's own aggregate under that
  reading and requiring near-exact agreement,
- writes `vendor/collections/aisi_inference_scaling/`:
    trajectories.parquet  one row per trajectory
    results.parquet       synthetic results at results_exploded grain,
                          plus protocol_condition / n_trajectories
    manifest.json         pinned EEE revision, member enumeration,
                          expected drop count, extraction stats +
                          reconciliation evidence

Run manually when AISI data changes (an EEE pin bump that includes AISI
changes must carry the regenerated vendor files in the same PR.2):

    uv run python scripts/collections/aisi_inference_scaling.py \
        --revision <EEE_REVISION>

Aggregate records are read from the local `.cache/eee_datastore` snapshot
when its listing revision matches (else a per-revision sibling dir is
materialised); sample JSONLs are streamed via `hf_hub_download` (multi-GB;
transcripts are parsed and discarded, never persisted). CI never runs this.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub>=0.24",
#     "duckdb>=1.0",
#     "pyarrow>=16",
#     "pydantic>=2.5",
#     "pyyaml>=6",
# ]
# ///

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval_card_backend.sources.collections import slug  # noqa: E402

COLLECTION_ID = "uk-aisi-inference-scaling"
STUDY_TITLE = "How Inference Compute Shapes Frontier LLM Evaluation"
STUDY_SLUG = slug(STUDY_TITLE)

# Benchmark labels for the EMITTED synthetic rows and trajectories. The
# study's inspect harness names two configs by its internal task names,
# but the paper's methods section (arXiv 2606.17930) names TerminalBench
# 2.0 and SWE-Bench Pro, and the task suite matches TB 2.x (88 of its 89
# tasks, none outside). Emit the canonical names so the data carries the
# real identity; the registry stays free of submission-specific label
# quirks. Internal keying (grouping, score semantics, manifests) stays on
# the raw config names.
BENCHMARK_LABELS = {
    "terminalbench": "terminal-bench-2",
    "swebenchpro": "swe-bench-pro",
}
EEE_REPO = "evaleval/EEE_datastore"
OUT_DIR = REPO_ROOT / "vendor" / "collections" / "aisi_inference_scaling"
DEFAULT_EEE_CACHE = REPO_ROOT / ".cache" / "eee_datastore"

# Per-benchmark score semantics (v4 — measured, see the collections spec):
# - trajectory_outcome: which per-row field is the per-attempt outcome
#   shown in trajectories.parquet (None = no per-attempt score exists).
# - aggregate: how the cell headline is computed. `task_mean_score` =
#   per-task mean of row `score`, then mean over tasks (equal task
#   weighting — cancels the "re-runs oversample hard tasks" bias).
#   `record_summary_mean` = row-count-weighted mean of the member
#   records' own summary aggregates (healthbench: per-row scores are
#   constant 0.0 upstream; only summaries carry the rubric signal).
# - upstream_rule: the arithmetic the UPSTREAM record aggregates used —
#   drives the per-record reconciliation identity check, not our output.
# Mirrored in collections_curated.yaml (`score_semantics`).
BENCHMARKS: dict[str, dict] = {
    "hle": {
        "outcome": "binary", "metric": "accuracy",
        "trajectory_outcome": "score", "aggregate": "task_mean_score",
        "upstream_rule": "task_mean_score",
    },
    "terminalbench": {
        "outcome": "binary", "metric": "accuracy",
        "trajectory_outcome": "score", "aggregate": "task_mean_score",
        "upstream_rule": "task_mean_score",
    },
    "swebenchpro": {
        "outcome": "binary", "metric": "accuracy",
        "trajectory_outcome": "score", "aggregate": "task_mean_score",
        "upstream_rule": "row_mean_score",
    },
    "frontiermath": {
        # Row `score` is a pre-averaged per-task solve rate (constant per
        # (task, record)); the per-attempt outcome is `is_correct`.
        "outcome": "binary", "metric": "accuracy",
        "trajectory_outcome": "is_correct", "aggregate": "task_mean_score",
        "upstream_rule": "task_mean_score",
    },
    "healthbench": {
        # Row `score` is constant 0.0 upstream; the graded per-attempt
        # value lives in metadata.submissions[-1].score (verified: it
        # recomposes the record summaries exactly wherever submission
        # coverage is complete). Headline still comes from the record
        # summaries — records dominated by submission-less rows carry
        # summary means over a population the datastore doesn't ship,
        # so the summaries are the only claim we can restate.
        "outcome": "graded", "metric": "healthbench_score",
        "trajectory_outcome": "last_submission_score",
        "aggregate": "record_summary_mean",
        "upstream_rule": None,
    },
}

# eval_plan submit-wrapper token → feedback arm.
WRAPPER_TO_FEEDBACK = {
    "_never_correct": "none",
    "to_float": "answer_feedback",
    "healthbench_score_value": "answer_feedback",
}

RATE_GATE = 0.10          # generic extraction rates above this block
RECON_TOLERANCE = 0.005   # per-record aggregate identity tolerance
RECON_MIN_EXACT = 0.85    # minimum exact-match rate per benchmark


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


# ---------------------------------------------------------------------------
# Member enumeration + aggregate-record parsing
# ---------------------------------------------------------------------------


@dataclass
class Member:
    path: str
    uuid: str
    config: str
    record: dict
    evaluation_id: str
    n_results: int
    feedback: str | None          # from the eval_plan submit wrapper
    reasoning_effort: str | None
    reasoning_tokens: int | None
    generation_config: dict | None
    source_data: dict | None


def _parse_plan_scalar(v: Any) -> Any:
    """eval_plan.config values arrive as strings, sometimes doubly encoded
    ('\"high\"', '64000'). Unwrap best-effort."""
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except ValueError:
        return v


def parse_member(path: str, record: dict) -> Member:
    results = record.get("evaluation_results") or []
    feedback = None
    reasoning_effort = None
    reasoning_tokens = None
    generation_config = None
    source_data = None
    if results:
        first = results[0]
        generation_config = first.get("generation_config")
        source_data = first.get("source_data")
        args = (generation_config or {}).get("generation_args") or {}
        plan = args.get("eval_plan") or {}
        cfg = plan.get("config") or {}
        eff = _parse_plan_scalar(cfg.get("reasoning_effort"))
        reasoning_effort = str(eff) if eff is not None else None
        rtok = _parse_plan_scalar(cfg.get("reasoning_tokens"))
        try:
            reasoning_tokens = int(float(rtok)) if rtok is not None else None
        except (TypeError, ValueError):
            reasoning_tokens = None
        steps = " ".join(plan.get("steps") or [])
        for token, arm in WRAPPER_TO_FEEDBACK.items():
            if token in steps:
                feedback = arm
                break
    return Member(
        path=path,
        uuid=Path(path).stem,
        config=path.split("/")[1],
        record=record,
        evaluation_id=record["evaluation_id"],
        n_results=len(results),
        feedback=feedback,
        reasoning_effort=reasoning_effort,
        reasoning_tokens=reasoning_tokens,
        generation_config=generation_config,
        source_data=source_data,
    )


def record_summary_score(m: Member) -> float | None:
    """The record's own summary aggregate under the collection's metric:
    the `mean …` entry for healthbench (rubric mean), else the first
    `accuracy`/`mean` entry."""
    for er in (m.record.get("evaluation_results") or []):
        if er["evaluation_name"].startswith(("mean", "accuracy")):
            return er["score_details"]["score"]
    return None


def _pinned_snapshot_root(revision: str, eee_cache: Path, hf_token: str | None) -> Path:
    """A local aggregate-record tree at exactly `revision`.

    Reuses `eee_cache` when its listing was written at that revision;
    otherwise materialises a SEPARATE per-revision dir via the pipeline's
    own `eee.ensure_snapshot` (parallel top-up download) rather than wiping
    the developer's working cache, which may deliberately track a newer
    HEAD."""
    listing_file = eee_cache / ".eee_file_listing.json"
    if listing_file.exists():
        listing = json.loads(listing_file.read_text(encoding="utf-8"))
        if listing.get("revision") == revision:
            return eee_cache

    from eval_card_backend.sources import eee

    pinned_dir = eee_cache.parent / f"{eee_cache.name}_{revision[:12]}"
    print(f"local cache is not at {revision[:12]} — materialising {pinned_dir} …")
    return eee.ensure_snapshot(
        str(pinned_dir), hf_token, force_refresh=False, revision=revision
    )


def enumerate_members(
    revision: str, eee_cache: Path, hf_token: str | None,
    stats: Counter | None = None,
) -> tuple[list[Member], Path]:
    """Read every aggregate record at `revision` and keep the study's
    members. Predicate: slug(source_name) == study slug — org is
    corroboration only (the Institute/Initiative spelling split means
    exact-org matching leaks fragments). Unreadable listing entries are
    counted, not silently skipped — a missing member would undercount
    `expected_drop_count` (the canonicalise leak guard backstops, but the
    miss should be visible here first)."""
    stats = stats if stats is not None else Counter()
    root = _pinned_snapshot_root(revision, eee_cache, hf_token)
    listing = json.loads(
        (root / ".eee_file_listing.json").read_text(encoding="utf-8")
    )
    members: list[Member] = []
    n_org_mismatch = 0
    for path in listing["paths"]:
        p = root / path
        if not p.exists():
            stats["listing_files_missing"] += 1
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            stats["listing_files_unparseable"] += 1
            continue
        if not isinstance(rec, dict):
            stats["listing_files_not_dict"] += 1
            continue
        sm = rec.get("source_metadata") or {}
        if slug(sm.get("source_name")) != STUDY_SLUG:
            continue
        org = sm.get("source_organization_name") or ""
        if "AI Security" not in org:
            n_org_mismatch += 1
            print(f"  NOTE: member by source_name with unexpected org {org!r}: {path}")
        members.append(parse_member(path, rec))
    skipped = (stats["listing_files_missing"] + stats["listing_files_unparseable"]
               + stats["listing_files_not_dict"])
    print(f"members: {len(members)} record(s); org-mismatch notes: "
          f"{n_org_mismatch}; unreadable listing entries: {skipped}")
    return members, root


# ---------------------------------------------------------------------------
# Sample streaming → trajectories (v4: one row = one complete trajectory)
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    member: Member
    config: str
    model_id: str
    sample_id: str
    has_turns: bool
    condition: str | None
    epoch: int | None            # global attempt counter (turns[0].epoch)
    token_limit: int | None
    score: float | None          # raw row score (semantics per BENCHMARKS)
    is_correct: bool | None
    num_turns: int | None
    tool_calls: int | None
    stop_reason: str | None
    total_tokens: int | None     # own turns[-1] cumulative, else token_usage
    output_tokens: int | None
    reasoning_tokens: int | None
    wall_time_s: float | None
    working_time_s: float | None
    last_submission_score: float | None = None
    # Behavioural feedback-arm markers (perfect specificity on all 372
    # known-arm records): a success-stop can only happen when the run
    # actually received correctness feedback; a fully-correct submission
    # FOLLOWED by more submissions can only happen when it did not.
    success_stop: bool = False
    continued_after_correct: bool = False
    cum_seq: tuple = ()          # transient: guards only (duplicates/containment)
    condition_inferred: bool = False
    unstitchable: bool = False   # turns missing AND condition unrecoverable
    included: bool = False
    protocol: dict | None = None

    def outcome(self) -> float | None:
        """Per-attempt outcome under the benchmark's declared source."""
        src = BENCHMARKS.get(self.config, {}).get("trajectory_outcome")
        if src == "score":
            return self.score
        if src == "is_correct":
            return None if self.is_correct is None else float(self.is_correct)
        if src == "last_submission_score":
            return self.last_submission_score
        return None


def _to_int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_sample_row(member: Member, row: dict, stats: Counter) -> Trajectory | None:
    meta = row.get("metadata") or {}
    ev = row.get("evaluation") or {}
    tok = row.get("token_usage") or {}
    if "messages" not in row or row.get("messages") in (None, []):
        stats["rows_missing_messages"] += 1

    turns_raw = meta.get("turns")
    turns: list[dict] = []
    if isinstance(turns_raw, str) and turns_raw:
        try:
            parsed = json.loads(turns_raw)
            if isinstance(parsed, list):
                turns = [t for t in parsed if isinstance(t, dict)]
        except ValueError:
            stats["rows_turns_parse_error"] += 1
    elif isinstance(turns_raw, list):
        turns = [t for t in turns_raw if isinstance(t, dict)]

    has_turns = bool(turns)
    condition = epoch = token_limit = None
    cum_total = cum_output = None
    wall = working = None
    cum_seq: tuple = ()
    if has_turns:
        t0, tN = turns[0], turns[-1]
        condition = t0.get("condition")
        epoch = _to_int(t0.get("epoch"))
        token_limit = _to_int(t0.get("token_limit"))
        if len({(t.get("condition"), _to_int(t.get("epoch"))) for t in turns}) > 1:
            stats["rows_turns_inconsistent_keys"] += 1
        # The row's OWN turn history is the authoritative accounting:
        # `token_usage` undercounts on resumed runs (upstream artifact) —
        # never summed, never mixed.
        cum_total = _to_int(tN.get("total_tokens_target_model"))
        cum_output = _to_int(tN.get("output_tokens_target_model"))
        wall = _to_float(tN.get("wall_time"))
        working = _to_float(tN.get("working_time"))
        cum_seq = tuple(
            _to_int(t.get("total_tokens_target_model")) or 0 for t in turns
        )
        piece = _to_int(tok.get("total_tokens"))
        if piece is not None and cum_total is not None and piece < cum_total * 0.98:
            stats["rows_token_usage_undercount"] += 1
    else:
        stats["rows_missing_turns"] += 1
        epoch = _to_int(meta.get("epoch"))

    subs_raw = meta.get("submissions")
    sub_scores: list[float] = []
    if isinstance(subs_raw, str) and subs_raw:
        try:
            sub_scores = [
                float(s["score"]) for s in json.loads(subs_raw)
                if isinstance(s, dict) and isinstance(s.get("score"), (int, float))
            ]
        except ValueError:
            stats["rows_submissions_parse_error"] += 1
    stop = meta.get("traj_stopping_reason") or meta.get("stop_reason")

    return Trajectory(
        member=member,
        config=member.config,
        model_id=member.record["model_info"]["id"],
        sample_id=str(row.get("sample_id")),
        has_turns=has_turns,
        condition=condition,
        epoch=epoch,
        token_limit=token_limit,
        score=_to_float(ev.get("score")),
        is_correct=ev.get("is_correct") if isinstance(ev.get("is_correct"), bool) else None,
        num_turns=_to_int(ev.get("num_turns")),
        tool_calls=_to_int(ev.get("tool_calls_count")),
        stop_reason=stop,
        total_tokens=cum_total if has_turns else _to_int(tok.get("total_tokens")),
        output_tokens=cum_output if has_turns else _to_int(tok.get("output_tokens")),
        reasoning_tokens=_to_int(tok.get("reasoning_tokens")),
        wall_time_s=wall,
        working_time_s=working,
        last_submission_score=sub_scores[-1] if sub_scores else None,
        success_stop=(stop == "completed_on_successful_submit"),
        continued_after_correct=any(v >= 1.0 for v in sub_scores[:-1]),
        cum_seq=cum_seq,
    )


def stream_trajectories(
    members: list[Member], revision: str, hf_token: str | None, stats: Counter
) -> list[Trajectory]:
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    def _sample_path(m: Member) -> str:
        p = ((m.record.get("detailed_evaluation_results") or {}).get("file_path"))
        return p or m.path.replace(".json", "_samples.jsonl")

    def _fetch(m: Member) -> str | Exception:
        last: Exception | None = None
        for _attempt in range(3):
            try:
                return hf_hub_download(
                    repo_id=EEE_REPO, repo_type="dataset", revision=revision,
                    filename=_sample_path(m), token=hf_token,
                )
            except Exception as exc:  # transient disconnects/429s
                last = exc
        return last if last is not None else RuntimeError("unreachable")

    # Prefetch in parallel (multi-GB total); parse sequentially.
    with ThreadPoolExecutor(max_workers=16) as pool:
        fetched = list(pool.map(_fetch, members))

    trajs: list[Trajectory] = []
    for i, (m, local) in enumerate(zip(members, fetched), 1):
        if isinstance(local, Exception):
            stats["files_download_failed"] += 1
            print(f"  DOWNLOAD FAILED {_sample_path(m)}: "
                  f"{type(local).__name__}: {local}")
            continue
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["rows_total"] += 1
                try:
                    row = json.loads(line)
                except ValueError:
                    stats["rows_parse_error"] += 1
                    continue
                t = parse_sample_row(m, row, stats)
                if t is None:
                    stats["rows_parse_error"] += 1
                    continue
                trajs.append(t)
        if i % 50 == 0 or i == len(members):
            print(f"  samples: {i}/{len(members)} files, {stats['rows_total']} rows")
    return trajs


# ---------------------------------------------------------------------------
# v4 guards: duplicates + independence (no installment pieces)
# ---------------------------------------------------------------------------


def dedupe_and_guard(trajs: list[Trajectory], stats: Counter) -> list[Trajectory]:
    """Drop exact duplicate rows (same identity AND identical turn-history
    fingerprint — re-uploaded batches), then verify the v4 independence
    assumption: within a (benchmark, model, condition, sample) pool, no
    row's turn history may be a strict prefix of another's. A prefix hit
    would mean installment pieces exist after all and the one-row-one-
    trajectory model undercounts — hard evidence to re-diagnose."""
    seen: set = set()
    kept: list[Trajectory] = []
    for t in trajs:
        # Full tuple fingerprint (not a hash — a hash collision would
        # silently drop a genuine trajectory).
        key = (t.config, t.model_id, t.condition, t.sample_id, t.epoch,
               t.cum_seq)
        if t.has_turns and key in seen:
            stats["rows_duplicate_removed"] += 1
            continue
        seen.add(key)
        kept.append(t)

    pools: dict[tuple, list[Trajectory]] = defaultdict(list)
    for t in kept:
        if t.has_turns:
            pools[(t.config, t.model_id, t.condition, t.sample_id)].append(t)
    for pool in pools.values():
        if len(pool) < 2:
            continue
        pool.sort(key=lambda t: len(t.cum_seq))
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                if len(b.cum_seq) > len(a.cum_seq) and \
                        b.cum_seq[:len(a.cum_seq)] == a.cum_seq:
                    stats["containment_pairs"] += 1
    stats["trajectories_total"] = len(kept)
    return kept


# ---------------------------------------------------------------------------
# Protocol points
# ---------------------------------------------------------------------------


def _condition_axes(condition: str | None) -> tuple[str | None, bool | None]:
    if condition is None:
        return None, None
    if condition.endswith("+C"):
        return condition[:-2], True
    return condition, False


def assign_protocols(trajs: list[Trajectory], stats: Counter) -> None:
    """Attach a protocol point to every trajectory. Feedback and reasoning
    settings come from the trajectory's own record's eval_plan; records
    with no aggregates (no eval_plan) inherit from sibling member records
    in the same (benchmark, model, condition, token_limit) cell, else
    `unknown` (kept, labelled).

    Rows missing turns entirely (condition unrecoverable — measured zero
    at the current pin) fall back to their record's unique sibling
    condition, else are flagged unstitchable and excluded."""
    record_conditions: dict[str, set[str]] = defaultdict(set)
    for t in trajs:
        if t.has_turns and t.condition:
            record_conditions[t.member.path].add(t.condition)

    # Record-level feedback arm: eval_plan wrapper first; else behavioural
    # markers (validated with zero false positives on all 372 known-arm
    # records): any success-stop row → the run received correctness
    # feedback; any row that kept submitting after a fully-correct
    # submission → it did not.
    rec_succ: dict[str, bool] = defaultdict(bool)
    rec_cont: dict[str, bool] = defaultdict(bool)
    members_by_path: dict[str, Member] = {}
    for t in trajs:
        rec_succ[t.member.path] |= t.success_stop
        rec_cont[t.member.path] |= t.continued_after_correct
        members_by_path[t.member.path] = t.member
    # Iterate unique RECORDS (stats are per-record); a record with
    # conflicting markers gets the explicit "conflict" sentinel so its
    # trajectories land on `unknown` and never reach sibling inference.
    record_arm: dict[str, str] = {}
    for path, member in members_by_path.items():
        if member.feedback is not None:
            record_arm[path] = member.feedback
        elif rec_succ[path] and rec_cont[path]:
            record_arm[path] = "conflict"
            stats["records_arm_conflict"] += 1
        elif rec_succ[path]:
            record_arm[path] = "answer_feedback"
            stats["records_arm_from_behavior"] += 1
        elif rec_cont[path]:
            record_arm[path] = "none"
            stats["records_arm_from_behavior"] += 1

    fb_pool: dict[tuple, set[str]] = defaultdict(set)
    eff_pool: dict[tuple, set[str]] = defaultdict(set)
    rtok_pool: dict[tuple, set[int]] = defaultdict(set)

    for t in trajs:
        if not t.has_turns:
            conds = record_conditions.get(t.member.path, set())
            if len(conds) == 1:
                t.condition = next(iter(conds))
                t.condition_inferred = True
            else:
                t.unstitchable = True
                stats["trajectories_unstitchable"] += 1
                continue
        key = (t.config, t.model_id, t.condition, t.token_limit)
        if record_arm.get(t.member.path) in ("none", "answer_feedback"):
            fb_pool[key].add(record_arm[t.member.path])
        if t.member.reasoning_effort is not None:
            eff_pool[key].add(t.member.reasoning_effort)
        if t.member.reasoning_tokens is not None:
            rtok_pool[key].add(t.member.reasoning_tokens)

    for t in trajs:
        if t.unstitchable:
            continue
        key = (t.config, t.model_id, t.condition, t.token_limit)
        arm = record_arm.get(t.member.path)
        effort = t.member.reasoning_effort
        rtok = t.member.reasoning_tokens
        if arm == "conflict":
            # Conflicting behavioural evidence: labelled unknown outright,
            # never sibling-inferred.
            feedback = "unknown"
            stats["trajectories_feedback_unknown"] += 1
        elif arm is not None:
            feedback = arm
        else:
            feedback = _unique(fb_pool.get(key))
            if feedback is None:
                feedback = "unknown"
                stats["trajectories_feedback_unknown"] += 1
        if effort is None:
            effort = _unique(eff_pool.get(key))
        if rtok is None:
            rtok = _unique(rtok_pool.get(key))
        scaffold, compaction = _condition_axes(t.condition)
        t.protocol = {
            "scaffold": scaffold,
            "compaction": compaction,
            "feedback": feedback,
            "token_limit": t.token_limit,
            "reasoning_tokens": rtok,
            "reasoning_effort": effort,
        }
        t.included = True


def _unique(values) -> Any:
    if not values:
        return None
    vals = set(values)
    return next(iter(vals)) if len(vals) == 1 else None


# ---------------------------------------------------------------------------
# Cell aggregation (per-benchmark score source)
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    config: str
    model_id: str
    protocol: dict
    protocol_json: str
    trajectories: list[Trajectory]
    score: float = 0.0
    score_se: float | None = None
    n_tasks: int = 0
    aggregate_method: str = ""
    n_summary_records: int | None = None
    base_member: Member | None = None
    contributing: list[Member] = field(default_factory=list)


def build_cells(
    trajs: list[Trajectory], stats: Counter
) -> tuple[list[Cell], list[dict]]:
    """Returns (cells, dropped) — dropped itemises every (benchmark,
    model, protocol) cell that had trajectories but no computable score
    (e.g. healthbench cells whose contributing records carry no
    summaries), so score-less cells are a listed, gated loss rather than
    an opaque counter."""
    grouped: dict[tuple, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        if not t.included or t.protocol is None:
            continue
        grouped[(t.config, t.model_id, canonical_json(t.protocol))].append(t)

    cells: list[Cell] = []
    dropped: list[dict] = []
    for (config, model_id, pjson), rows in sorted(grouped.items()):
        method = BENCHMARKS.get(config, {}).get("aggregate", "task_mean_score")
        if method == "record_summary_mean":
            agg = _aggregate_from_record_summaries(rows, stats)
        else:
            agg = _aggregate_task_mean_score(rows, stats)
        if agg is None:
            stats["cells_no_scores"] += 1
            dropped.append({
                "benchmark": config, "model": model_id,
                "protocol_condition": pjson,
                "n_trajectories": len(rows),
                "reason": "no_computable_score",
            })
            continue
        score, se, n_tasks, n_summary = agg

        contributing_paths: dict[str, Member] = {}
        for t in rows:
            contributing_paths[t.member.path] = t.member
        base_member = sorted(
            contributing_paths.values(), key=lambda m: (not m.n_results > 0, m.path)
        )[0]
        cells.append(Cell(
            config=config, model_id=model_id,
            protocol=json.loads(pjson), protocol_json=pjson,
            trajectories=rows, score=score, score_se=se,
            n_tasks=n_tasks, aggregate_method=method,
            n_summary_records=n_summary, base_member=base_member,
            contributing=sorted(contributing_paths.values(), key=lambda m: m.path),
        ))
    stats["cells_total"] = len(cells)
    return cells, dropped


def _aggregate_task_mean_score(
    rows: list[Trajectory], stats: Counter
) -> tuple[float, float | None, int, None] | None:
    """Equal task weighting: per-task mean of row `score`, then mean over
    tasks; SE over task means (clustered estimator). Equal task
    weighting is what cancels the installment bias (re-run batches
    oversample hard tasks)."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for t in rows:
        if t.score is not None:
            by_task[t.sample_id].append(t.score)
    if not by_task:
        return None
    task_means = [statistics.fmean(v) for v in by_task.values()]
    score = statistics.fmean(task_means)
    se = (
        statistics.stdev(task_means) / (len(task_means) ** 0.5)
        if len(task_means) > 1 else None
    )
    return score, se, len(task_means), None


def _aggregate_from_record_summaries(
    rows: list[Trajectory], stats: Counter
) -> tuple[float, float | None, int, int] | None:
    """healthbench: per-row scores are constant 0.0 upstream — the rubric
    signal exists only in record-level summaries. Cell score = row-count-
    weighted mean of the contributing records' own summary means
    (trajectory-weighted; equal task weighting is unrecoverable from
    summaries — declared in score_details.details)."""
    rows_per_record: dict[str, int] = Counter(t.member.path for t in rows)
    members: dict[str, Member] = {t.member.path: t.member for t in rows}
    weighted: list[tuple[str, float, int]] = []
    for path, n in sorted(rows_per_record.items()):
        m = members[path]
        summary = record_summary_score(m) if m.n_results else None
        if summary is not None:
            weighted.append((path, summary, n))
    if not weighted:
        return None
    total_n = sum(n for _, _, n in weighted)
    score = sum(s * n for _, s, n in weighted) / total_n
    # SE uses the SAME row-count weights as the mean (weighted variance of
    # the record means around the weighted mean, over the number of
    # contributing records).
    se = None
    if len(weighted) > 1:
        var = sum(n * (s - score) ** 2 for _, s, n in weighted) / total_n
        se = (var / len(weighted)) ** 0.5
    # num_samples counts tasks from CONTRIBUTING records only — summary-less
    # records carried zero weight and must not inflate the sample count.
    contributing_paths = {path for path, _, _ in weighted}
    n_tasks = len({t.sample_id for t in rows if t.member.path in contributing_paths})
    return score, se, n_tasks, len(weighted)


# ---------------------------------------------------------------------------
# Reconciliation gate: per-record aggregate identity under the declared rule
# ---------------------------------------------------------------------------


def reconcile_record_aggregates(
    members: list[Member], trajs: list[Trajectory]
) -> dict[str, dict]:
    """For every aggregate-carrying member record, recompute its own
    summary from its own rows under the benchmark's measured upstream rule
    and compare. Same computation on the same data — near-exact agreement
    is the proof the score reading is right; run on every refresh."""
    rows_by_record: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        rows_by_record[t.member.path].append(t)

    out: dict[str, dict] = {}
    for cfg, decl in BENCHMARKS.items():
        rule = decl.get("upstream_rule")
        if rule is None:
            continue
        n = exact = within_05 = 0
        n_carriers = skipped_no_upstream = skipped_no_rows = 0
        mismatches: list[dict] = []
        for m in members:
            if m.config != cfg or not m.n_results:
                continue
            n_carriers += 1
            upstream = record_summary_score(m)
            rows = [t for t in rows_by_record.get(m.path, []) if t.score is not None]
            if upstream is None:
                skipped_no_upstream += 1
                continue
            if not rows:
                skipped_no_rows += 1
                continue
            if rule == "row_mean_score":
                ours = statistics.fmean(t.score for t in rows)
            else:  # task_mean_score
                by_task: dict[str, list[float]] = defaultdict(list)
                for t in rows:
                    by_task[t.sample_id].append(t.score)
                ours = statistics.fmean(
                    statistics.fmean(v) for v in by_task.values()
                )
            d = abs(ours - upstream)
            n += 1
            if d <= RECON_TOLERANCE:
                exact += 1
            if d <= 0.05:
                within_05 += 1
            if d > RECON_TOLERANCE:
                mismatches.append({
                    "path": m.path, "upstream": upstream,
                    "recomputed": round(ours, 6), "delta": round(d, 6),
                })
        out[cfg] = {
            "rule": rule, "records": n, "exact": exact,
            "exact_rate": round(exact / n, 4) if n else None,
            "within_0_05": within_05,
            # Skips shrink the denominator — surfaced so the ≥85% gate
            # can't be gamed by a rule/name drift emptying the check.
            "aggregate_carriers": n_carriers,
            "skipped_no_upstream": skipped_no_upstream,
            "skipped_no_rows": skipped_no_rows,
            "mismatches": sorted(mismatches, key=lambda r: -r["delta"])[:25],
        }

    # healthbench (informational): re-prove the declared per-attempt
    # reading (submissions[-1].score) on records with FULL submission
    # coverage — the only population where the record summary is
    # recomposable from per-attempt values.
    hb_n = hb_match = hb_carriers = 0
    hb_mismatches: list[dict] = []
    for m in members:
        if m.config != "healthbench" or not m.n_results:
            continue
        hb_carriers += 1
        upstream = record_summary_score(m)
        rows = rows_by_record.get(m.path, [])
        if upstream is None or not rows:
            continue
        if any(t.last_submission_score is None for t in rows):
            continue  # partial coverage — summary population not shipped
        by_task: dict[str, list[float]] = defaultdict(list)
        for t in rows:
            by_task[t.sample_id].append(t.last_submission_score)
        task_mean = statistics.fmean(statistics.fmean(v) for v in by_task.values())
        row_mean = statistics.fmean(t.last_submission_score for t in rows)
        d = min(abs(task_mean - upstream), abs(row_mean - upstream))
        hb_n += 1
        if d <= RECON_TOLERANCE:
            hb_match += 1
        else:
            hb_mismatches.append({
                "path": m.path, "upstream": upstream,
                "recomputed_task_mean": round(task_mean, 6),
                "recomputed_row_mean": round(row_mean, 6),
                "delta": round(d, 6),
            })
    out["healthbench"] = {
        "rule": "submission_recompose_full_coverage",
        "informational": True,
        "records": hb_n, "exact": hb_match,
        "exact_rate": round(hb_match / hb_n, 4) if hb_n else None,
        "aggregate_carriers": hb_carriers,
        "mismatches": sorted(hb_mismatches, key=lambda r: -r["delta"])[:25],
    }
    return out


# ---------------------------------------------------------------------------
# Synthetic EEE records
# ---------------------------------------------------------------------------


def _max_ts(values: list[str | None]) -> str | None:
    best_v, best_s = None, None
    for s in values:
        f = _to_float(s)
        if f is not None and (best_v is None or f > best_v):
            best_v, best_s = f, s
    return best_s


def build_synthetic_records(cells: list[Cell], stats: Counter) -> list[dict]:
    """One EEE-shaped record per base member record; its
    evaluation_results[] enumerates the protocol points anchored to it
    (result_idx = position, ordered by canonical protocol JSON)."""
    by_base: dict[str, list[Cell]] = defaultdict(list)
    base_members: dict[str, Member] = {}
    for c in cells:
        by_base[c.base_member.path].append(c)
        base_members[c.base_member.path] = c.base_member

    records: list[dict] = []
    for path in sorted(by_base):
        m = base_members[path]
        cell_list = sorted(by_base[path], key=lambda c: (c.config, c.protocol_json))
        results = []
        protocol_by_idx = []
        for c in cell_list:
            decl = BENCHMARKS.get(c.config, {})
            outcome = decl.get("outcome", "binary")
            metric = decl.get("metric", "accuracy")
            gen_cfg = _synthetic_generation_config(c)
            source_data = None
            for cand in [m] + c.contributing:
                if cand.source_data is not None:
                    source_data = copy.deepcopy(cand.source_data)
                    break
            if source_data is None:
                source_data = {"dataset_name": c.config, "source_type": "other"}
            details = {
                "n_trajectories": str(len(c.trajectories)),
                "n_tasks": str(c.n_tasks),
                "aggregation": c.aggregate_method,
                "protocol_condition": c.protocol_json,
            }
            if c.n_summary_records is not None:
                # healthbench: derived from record summaries, trajectory-
                # weighted (task weighting unrecoverable) — declared so we
                # never imply a computation we couldn't do.
                details["weighting"] = "trajectory_weighted"
                details["n_summary_records"] = str(c.n_summary_records)
            results.append({
                "evaluation_name": BENCHMARK_LABELS.get(c.config, c.config),
                "source_data": source_data,
                "evaluation_timestamp": _max_ts(
                    [mm.record.get("evaluation_timestamp") for mm in c.contributing]
                ),
                "metric_config": {
                    "evaluation_description": metric,
                    "metric_name": metric,
                    "lower_is_better": False,
                    "score_type": "binary" if outcome == "binary" else "continuous",
                    "min_score": 0.0,
                    "max_score": 1.0,
                },
                "score_details": {
                    "score": c.score,
                    "uncertainty": {
                        "standard_error": (
                            {"value": c.score_se,
                             "method": "clustered_task_se"
                                       if c.aggregate_method == "task_mean_score"
                                       else "record_summary_se"}
                            if c.score_se is not None else None
                        ),
                        "num_samples": c.n_tasks,
                    },
                    "details": details,
                },
                "generation_config": gen_cfg,
            })
            protocol_by_idx.append((c.protocol_json, len(c.trajectories)))

        rec = {
            "schema_version": m.record.get("schema_version", "0.3.0"),
            "evaluation_id": m.evaluation_id,
            "evaluation_timestamp": _max_ts(
                [mm.record.get("evaluation_timestamp") for c in cell_list
                 for mm in c.contributing]
            ) or m.record.get("evaluation_timestamp"),
            "retrieved_timestamp": m.record["retrieved_timestamp"],
            "source_metadata": copy.deepcopy(m.record["source_metadata"]),
            "eval_library": copy.deepcopy(m.record["eval_library"]),
            "model_info": copy.deepcopy(m.record["model_info"]),
            "evaluation_results": results,
            "detailed_evaluation_results": copy.deepcopy(
                m.record.get("detailed_evaluation_results")
            ),
        }
        records.append({
            "record": rec,
            "config": m.config,
            "path": m.path,
            "protocols": protocol_by_idx,
        })
    stats["synthetic_records"] = len(records)
    stats["synthetic_results"] = sum(len(r["protocols"]) for r in records)
    return records


def _synthetic_generation_config(c: Cell) -> dict | None:
    """Member base record's generation_config verbatim plus the protocol
    point (in additional_details, so variant_key_udf still reads the real
    generation_args)."""
    src = None
    for cand in [c.base_member] + c.contributing:
        if cand.generation_config is not None:
            src = copy.deepcopy(cand.generation_config)
            break
    if src is None:
        return None
    details = dict(src.get("additional_details") or {})
    for k, v in c.protocol.items():
        if v is not None:
            details[f"protocol_{k}"] = str(v)
    src["additional_details"] = details
    return src


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_trajectories_parquet(trajs: list[Trajectory], out: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for t in sorted(
        trajs,
        key=lambda t: (t.config, t.model_id, t.condition or "", t.sample_id,
                       t.epoch or 0, t.member.path),
    ):
        rows.append({
            "collection_id": COLLECTION_ID,
            # Config name, NOT the corrected label: Stage I joins resolved
            # ids onto trajectories via ids.source_config = t.benchmark_raw.
            "benchmark_raw": t.config,
            "model_raw": t.model_id,
            "task_id": t.sample_id,
            "protocol_condition": canonical_json(t.protocol) if t.protocol else None,
            # v4: one row = one complete trajectory; the index is the
            # upstream global attempt counter.
            "trajectory_idx": t.epoch,
            "score": t.outcome(),
            "is_correct": t.is_correct
            if BENCHMARKS.get(t.config, {}).get("outcome") == "binary" else None,
            "total_tokens": t.total_tokens,
            "output_tokens": t.output_tokens,
            "reasoning_tokens": t.reasoning_tokens,
            "num_turns": t.num_turns,
            "tool_calls": t.tool_calls,
            "n_pieces": 1,
            "wall_time_s": t.wall_time_s,
            "working_time_s": t.working_time_s,
            "stop_reason": t.stop_reason,
            "partial_start": False,
            "unstitchable": t.unstitchable,
            "token_source_cumulative": t.has_turns,
            "source_record_uuids": [t.member.uuid],
        })
    schema = pa.schema([
        ("collection_id", pa.string()),
        ("benchmark_raw", pa.string()),
        ("model_raw", pa.string()),
        ("task_id", pa.string()),
        ("protocol_condition", pa.string()),
        ("trajectory_idx", pa.int32()),
        ("score", pa.float64()),
        ("is_correct", pa.bool_()),
        ("total_tokens", pa.int64()),
        ("output_tokens", pa.int64()),
        ("reasoning_tokens", pa.int64()),
        ("num_turns", pa.int32()),
        ("tool_calls", pa.int32()),
        ("n_pieces", pa.int32()),
        ("wall_time_s", pa.float64()),
        ("working_time_s", pa.float64()),
        ("stop_reason", pa.string()),
        ("partial_start", pa.bool_()),
        ("unstitchable", pa.bool_()),
        ("token_source_cumulative", pa.bool_()),
        ("source_record_uuids", pa.list_(pa.string())),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out)


def write_results_parquet(synthetic: list[dict], out: Path) -> None:
    """Validate the synthetic records against the vendored EEE contract,
    explode them with the SAME Stage B SQL the pipeline runs, attach the
    protocol map, and write results.parquet at results_exploded grain."""
    import duckdb
    import pyarrow as pa
    from pydantic import ValidationError

    from eval_card_backend.canonicalise.stages import explode_select_sql
    from eval_card_backend.schemas.eee_arrow import (
        derive_pyarrow_schema,
        pad_record_for_cast,
    )
    from eval_card_backend.schemas.eee_types import EvaluationLog

    base_schema = derive_pyarrow_schema()
    table_schema = pa.schema(
        list(base_schema)
        + [
            pa.field("source_config", pa.string(), nullable=False),
            pa.field("_record_path", pa.string(), nullable=False),
        ]
    )
    rows = []
    protocol_rows = []
    for entry in synthetic:
        rec = entry["record"]
        try:
            EvaluationLog.model_validate(rec)
        except ValidationError as exc:
            raise SystemExit(
                f"synthetic record for {entry['path']} fails the EEE "
                f"contract: {exc.errors()[0] if exc.errors() else exc}"
            )
        padded = pad_record_for_cast(rec, base_schema)
        padded["source_config"] = entry["config"]
        padded["_record_path"] = entry["path"]
        rows.append(padded)
        for idx, (pjson, n_traj) in enumerate(entry["protocols"]):
            protocol_rows.append({
                "evaluation_id": rec["evaluation_id"],
                "result_idx": idx,
                "protocol_condition": pjson,
                "n_trajectories": n_traj,
            })

    records_table = pa.Table.from_pylist(rows, schema=table_schema)
    protocol_table = pa.Table.from_pylist(
        protocol_rows,
        schema=pa.schema([
            ("evaluation_id", pa.string()),
            ("result_idx", pa.int32()),
            ("protocol_condition", pa.string()),
            ("n_trajectories", pa.int32()),
        ]),
    )

    con = duckdb.connect()
    con.register("synthetic_records", records_table)
    con.register("protocol_map", protocol_table)
    con.execute(
        f"CREATE TABLE exploded AS {explode_select_sql('synthetic_records')}"
    )
    n_exploded = con.execute("SELECT count(*) FROM exploded").fetchone()[0]
    n_protocol = len(protocol_rows)
    if n_exploded != n_protocol:
        raise SystemExit(
            f"explode produced {n_exploded} rows but {n_protocol} protocol "
            f"points were assembled — synthetic record construction bug."
        )
    con.execute(
        f"""
        COPY (
            SELECT e.*, pm.protocol_condition, pm.n_trajectories
            FROM exploded e
            JOIN protocol_map pm
              ON pm.evaluation_id = e.evaluation_id
             AND pm.result_idx    = e.result_idx
            ORDER BY e.evaluation_id, e.result_idx
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--revision", required=True,
        help="EEE datastore revision (must equal the EEE_REVISION the "
             "pipeline will consume)",
    )
    ap.add_argument("--eee-cache", type=Path, default=DEFAULT_EEE_CACHE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument(
        "--allow-rate", action="append", default=[],
        metavar="RATE=EXPLANATION",
        help="accept a >10%% extraction rate with a recorded explanation",
    )
    ap.add_argument(
        "--allow-reconciliation", action="append", default=[],
        metavar="CONFIG=EXPLANATION",
        help="accept a benchmark whose per-record aggregate identity check "
             "falls below the exact-match gate, with a recorded explanation",
    )
    args = ap.parse_args()
    import os

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    stats: Counter = Counter()

    members, _root = enumerate_members(args.revision, args.eee_cache, hf_token, stats)
    if not members:
        raise SystemExit("no member records found — wrong revision?")
    unknown_benchmarks = {m.config for m in members} - set(BENCHMARKS)
    if unknown_benchmarks:
        raise SystemExit(
            f"member records under undeclared benchmark(s) "
            f"{sorted(unknown_benchmarks)} — add their score semantics to "
            f"BENCHMARKS (and collections_curated.yaml) before extracting."
        )

    trajs = stream_trajectories(members, args.revision, hf_token, stats)
    if stats["files_download_failed"]:
        raise SystemExit(
            f"{stats['files_download_failed']} sample file(s) failed to "
            f"download after retries — re-run; a partial corpus would bias "
            f"the reassembled aggregates."
        )
    trajs = dedupe_and_guard(trajs, stats)
    assign_protocols(trajs, stats)
    cells, dropped_cells = build_cells(trajs, stats)
    synthetic = build_synthetic_records(cells, stats)
    reconciliation = reconcile_record_aggregates(members, trajs)

    # ---- gates -----------------------------------------------------
    n_rows = stats["rows_total"]
    accounted = (
        stats["trajectories_total"] + stats["rows_parse_error"]
        + stats["rows_duplicate_removed"]
    )
    if accounted != n_rows:
        raise SystemExit(
            f"row reconciliation failed: {stats['trajectories_total']} "
            f"trajectories + {stats['rows_parse_error']} parse errors + "
            f"{stats['rows_duplicate_removed']} duplicates != {n_rows} rows."
        )

    if stats["containment_pairs"]:
        raise SystemExit(
            f"{stats['containment_pairs']} row pair(s) show prefix-contained "
            f"turn histories — installment pieces exist after all; the "
            f"one-row-one-trajectory model undercounts. Re-diagnose before "
            f"shipping."
        )

    n_traj = max(stats["trajectories_total"], 1)
    rates = {
        "unstitchable": stats["trajectories_unstitchable"] / n_traj,
        "duplicate_rows": stats["rows_duplicate_removed"] / max(n_rows, 1),
        "missing_turns": stats["rows_missing_turns"] / max(n_rows, 1),
        "missing_messages": stats["rows_missing_messages"] / max(n_rows, 1),
        "feedback_unknown": stats["trajectories_feedback_unknown"] / n_traj,
        "cells_no_scores": (
            len(dropped_cells) / max(len(cells) + len(dropped_cells), 1)
        ),
        "listing_unreadable": (
            (stats["listing_files_missing"] + stats["listing_files_unparseable"]
             + stats["listing_files_not_dict"]) / max(n_rows, 1)
        ),
    }
    allowed = dict(kv.split("=", 1) for kv in args.allow_rate)
    blocking = [
        name for name, rate in rates.items()
        if rate > RATE_GATE and name not in allowed
    ]
    if blocking:
        raise SystemExit(
            "extraction rate(s) above the 10% gate without an explanation: "
            + ", ".join(f"{n}={rates[n]:.1%}" for n in blocking)
            + " — re-run with --allow-rate NAME=EXPLANATION after diagnosing."
        )

    allowed_recon = dict(kv.split("=", 1) for kv in args.allow_reconciliation)
    recon_failures = [
        cfg for cfg, r in reconciliation.items()
        if not r.get("informational")
        and cfg not in allowed_recon
        and (
            # gate fails on a low exact-match rate, AND on the check
            # silently emptying out while aggregate carriers exist
            (r["records"] and r["exact_rate"] is not None
             and r["exact_rate"] < RECON_MIN_EXACT)
            or (r["records"] == 0 and r.get("aggregate_carriers", 0) > 0)
        )
    ]
    if recon_failures:
        for cfg in recon_failures:
            r = reconciliation[cfg]
            print(f"RECONCILIATION {cfg}: exact {r['exact']}/{r['records']} "
                  f"under rule {r['rule']}; worst: {r['mismatches'][:3]}")
        raise SystemExit(
            f"per-record aggregate identity below {RECON_MIN_EXACT:.0%} for "
            f"{recon_failures} — the declared score reading does not "
            f"reproduce upstream's own summaries; fix the reading (or "
            f"--allow-reconciliation CONFIG=EXPLANATION after diagnosing)."
        )

    # ---- write vendor outputs -----------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_trajectories_parquet(trajs, args.out_dir / "trajectories.parquet")
    write_results_parquet(synthetic, args.out_dir / "results.parquet")

    manifest = {
        "collection_id": COLLECTION_ID,
        "study_slug": STUDY_SLUG,
        "extractor": "scripts/collections/aisi_inference_scaling.py",
        "eee_revision": args.revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_drop_count": sum(m.n_results for m in members),
        "score_semantics": BENCHMARKS,
        "members": [
            {
                "record_uuid": m.uuid,
                "evaluation_id": m.evaluation_id,
                "path": m.path,
                "config": m.config,
                "n_results": m.n_results,
            }
            for m in sorted(members, key=lambda m: m.path)
        ],
        "cells_dropped": dropped_cells,
        "extraction_stats": {
            **{k: int(v) for k, v in sorted(stats.items())},
            "rates": {k: round(v, 6) for k, v in rates.items()},
            "rate_overrides": allowed,
        },
        "reconciliation": {
            "tolerance": RECON_TOLERANCE,
            "min_exact_rate": RECON_MIN_EXACT,
            "overrides": allowed_recon,
            "per_benchmark": reconciliation,
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False)
    )

    print(f"\nwrote {args.out_dir}/")
    print(f"  members: {len(members)}  expected_drop_count: "
          f"{manifest['expected_drop_count']}")
    print(f"  trajectories: {stats['trajectories_total']} "
          f"(duplicates removed {stats['rows_duplicate_removed']}, "
          f"unstitchable {stats['trajectories_unstitchable']}, "
          f"containment pairs {stats['containment_pairs']})")
    print(f"  cells: {stats['cells_total']}  synthetic records: "
          f"{stats['synthetic_records']} → results: "
          f"{stats['synthetic_results']}")
    for cfg, r in sorted(reconciliation.items()):
        print(f"  reconcile {cfg}: exact {r['exact']}/{r['records']} "
              f"(rule {r['rule']})")
    print("  rates: " + ", ".join(f"{k}={v:.2%}" for k, v in rates.items()))


if __name__ == "__main__":
    main()

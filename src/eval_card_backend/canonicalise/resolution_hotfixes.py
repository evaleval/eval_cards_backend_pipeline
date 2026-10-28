"""Temporary fact-level resolution hot fixes.

Sibling of `hierarchy_hotfixes.py`, but operates one stage earlier: these
functions run inside Stage C on the `results_resolved` table, before slice
keys are derived. They compensate for upstream data issues that can't be
expressed as registry aliases because the fix depends on a *cross-field*
condition the resolver can't see (it resolves each field independently).

Every function's docstring names the upstream defect and the condition
that retires the fix (tests/test_hotfix_lifecycle.py checks).
Stage C functions mutate `results_resolved` through DuckDB. Stage A
metadata builders return pure record appliers for the opt-in datafix overlay.

NOTE: several hotfixes below match on literal ``metric_raw`` values ('mean',
'score', ...). Since the metric_name pre-step in stage C, a row whose
``metric_config.metric_name`` resolves carries that name in ``metric_raw``
rather than the extraction literal, so those predicates only reach rows the
pre-step declined (no name, catch-all name, or a more specific description).
"""
from __future__ import annotations

import copy
import json
import logging
import math
import re
from collections import Counter
from typing import Any

from eval_card_backend.datafix.core import (
    GENERATION_ARGS_POINTER,
    PatchEvent,
    canonical_json,
    generation_arg,
    load_table,
    pointer_value,
    set_pointer,
    set_generation_arg,
    sha256_text,
    vendored_sha256,
)

log = logging.getLogger(__name__)


# ── 1. HELM composite-aggregate rows ─────────────────────────────────
# A registry alias can't repair these rows: benchmark and metric resolve
# independently, so there's no field the metric resolver can read that
# says "this rank is really a mean-win-rate", and aliasing
# `rank` → `mean win rate` would be both fragile (future data) and
# semantically false.
#
# For exactly the malformed rows (benchmark_raw is the aggregate label
# within a HELM composite), correct the field placement — benchmark ← the
# composite-overall canonical, metric ← the shared `mean-win-rate` /
# `mean-score` canonical. The benchmark already carries
# the per-tier namespace (helm-classic-leaderboard vs helm-lite-leaderboard),
# so the metric stays the plain canonical concept. This also kills the
# helm_mmlu → `mmlu` collision (those rows otherwise become a bogus
# "mean win rate" slice on the real MMLU benchmark); it runs before
# `_apply_slice_key` so the collision never forms a slice.

# source_config → (composite-overall benchmark id, canonical metric id).
# Narrow + enumerated on purpose: only these six HELM tiers, only the two
# aggregate labels below. Never touches a genuine rank/score elsewhere.
_HELM_AGGREGATE_MAP = {
    "helm_classic": ("helm-classic-leaderboard", "mean-win-rate"),
    "helm_lite": ("helm-lite-leaderboard", "mean-win-rate"),
    "helm_instruct": ("helm-instruct-leaderboard", "mean-win-rate"),
    "helm_mmlu": ("helm-mmlu-leaderboard", "mean-win-rate"),
    "helm_capabilities": ("helm-capabilities-leaderboard", "mean-score"),
    "helm_safety": ("helm-safety-leaderboard", "mean-score"),
}

_HELM_AGGREGATE_LABELS = ("mean win rate", "mean score")


def fix_helm_composite_aggregates(con) -> None:
    """Reassign benchmark + metric for HELM composite-aggregate rows.

    Fires only on rows where `benchmark_raw` is one of the aggregate
    labels within one of the six HELM composite source_configs.

    TODO(upstream): HELM ingestion emits the composite aggregate with the
    metric name in the benchmark field and a generic rank/score in the
    metric field.
    Remove when: HELM ingestion puts the aggregate metric in the metric
    field; the rows then resolve cleanly on their own.
    """
    mapping_values = ", ".join(
        f"('{sc}', '{bid}', '{mid}')"
        for sc, (bid, mid) in _HELM_AGGREGATE_MAP.items()
    )
    labels_sql = ", ".join(f"'{lbl}'" for lbl in _HELM_AGGREGATE_LABELS)

    con.execute(
        f"""
        UPDATE results_resolved AS r
        SET benchmark_id = m.bid,
            metric_id    = m.mid,
            benchmark_resolution_strategy = 'hotfix_helm_aggregate',
            metric_resolution_strategy    = 'hotfix_helm_aggregate'
        FROM (VALUES {mapping_values}) AS m(sc, bid, mid)
        WHERE r.source_config = m.sc
          AND LOWER(TRIM(r.benchmark_raw)) IN ({labels_sql})
        """
    )
    n = con.execute(
        f"""
        SELECT COUNT(*) FROM results_resolved
        WHERE source_config IN ({", ".join(f"'{sc}'" for sc in _HELM_AGGREGATE_MAP)})
          AND benchmark_resolution_strategy = 'hotfix_helm_aggregate'
        """
    ).fetchone()[0]
    log.info("resolution_hotfixes: reassigned %d HELM composite-aggregate row(s)", n)


# ── 2. Vague / malformed metric labels ───────────────────────────────


def fix_vague_metric_labels(con) -> None:
    """Namespace vague/malformed metric labels by their benchmark.

    Override (not IS-NULL-guarded): these raws ("mean", codegolf's junk
    string) never carry a real canonical, but the registry's normalized/
    fuzzy matching can still mis-resolve them to a same-token namespaced
    placeholder. We overwrite unconditionally so the final id is always the
    correct <benchmark_id>.<suffix> regardless of what the resolver guessed.

    TODO(upstream): the source emits metric labels that don't resolve to a
    canonical, so the metric renders blank. Two kinds handled here:
      - "mean": a real value but vague — its *meaning* differs per benchmark
        (cvebench success vs cyse2 exploit vs swebench resolution), so it must
        not be conflated into one global "mean" metric.
      - codegolf's metric field holds a junk benchmark-name string
        ("Codegolf v2.2 benchmark") — same malformed-field class as HELM.
    Until the source emits real metric names, namespace them by benchmark so
    they (a) display and (b) stay distinct. Matching placeholder canonicals
    live in metrics.yaml (lower_is_better unset → direction stays per-row).
    Remove when: the source emits real metric names; the placeholder
    canonicals go with it.
    """
    # "mean" → "<benchmark_id>.mean" (benchmark must have resolved).
    con.execute(
        """
        UPDATE results_resolved
        SET metric_id = benchmark_id || '.mean',
            metric_resolution_strategy = 'hotfix_vague_metric'
        WHERE LOWER(TRIM(metric_raw)) = 'mean'
          AND benchmark_id IS NOT NULL
        """
    )
    # codegolf junk metric string → "codegolf.score".
    con.execute(
        """
        UPDATE results_resolved
        SET metric_id = 'codegolf.score',
            metric_resolution_strategy = 'hotfix_vague_metric'
        WHERE metric_raw = 'Codegolf v2.2 benchmark'
          AND benchmark_id = 'codegolf'
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE metric_resolution_strategy = 'hotfix_vague_metric'"
    ).fetchone()[0]
    log.info("resolution_hotfixes: namespaced %d vague/malformed metric label(s)", n)


# ── 3. inspect_ai/harbor scorer-wrapper benchmark labels ─────────────
# The converted inspect_ai / harbor logs carry a *task* name that is
# per-model ("full-solver-<model>") or per-run
# ("swebenchpro/S-adaptive/+1ep/<hash>"), so the frontend fans one
# benchmark out into N single-model "benchmarks". A registry alias can't
# repair this: aliases are literal strings and the set is open (every new
# model/run mints a new one).
#
# Scope is deliberately enumerated: only these source_configs, only rows
# whose benchmark_raw matches the wrapper shape. The task-variant suffixes
# (S-adaptive/+Nep/<hash>) collapse into the config's single benchmark;
# they resurface as variant submissions of the same (model, benchmark,
# metric) triple, which is the intended modelling for setup variants.

# source_config → canonical benchmark id.
_SCORER_WRAPPER_MAP = {
    "l2-bench": "l2-bench",
    "terminalbench": "terminal-bench",
    "swebenchpro": "swe-bench-pro",
}

# "<word> on <anything> for scorer <word>" — the converter's label shape.
_SCORER_WRAPPER_REGEX = r"^\S+ on .+ for scorer \S+$"


def fix_scorer_wrapper_benchmarks(con) -> None:
    """Reassign benchmark for scorer-wrapper labelled rows.

    Fires only on rows in the enumerated source_configs whose
    `benchmark_raw` matches the wrapper shape. Runs before
    `fix_vague_metric_labels` so l2-bench's bare "mean" metric gets
    namespaced to `l2-bench.mean` once the benchmark is assigned.

    TODO(upstream): the inspect_ai/harbor converter writes the whole
    "<metric> on <task> for scorer <scorer>" label into evaluation_name,
    so every record mints its own unresolvable benchmark.
    Remove when: the upstream submissions carry the real benchmark name in
    evaluation_name; `_SCORER_WRAPPER_MAP` and the `l2-bench.mean`
    placeholder canonical go with it.
    """
    mapping_values = ", ".join(
        f"('{sc}', '{bid}')" for sc, bid in _SCORER_WRAPPER_MAP.items()
    )
    con.execute(
        f"""
        UPDATE results_resolved AS r
        SET benchmark_id = m.bid,
            benchmark_resolution_strategy = 'hotfix_scorer_wrapper'
        FROM (VALUES {mapping_values}) AS m(sc, bid)
        WHERE r.source_config = m.sc
          AND regexp_matches(r.benchmark_raw, '{_SCORER_WRAPPER_REGEX}')
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE benchmark_resolution_strategy = 'hotfix_scorer_wrapper'"
    ).fetchone()[0]
    log.info(
        "resolution_hotfixes: reassigned %d scorer-wrapper benchmark row(s)", n
    )


# ── 4. HLE leaderboard calibration column mislabelled "score" ────────


def fix_hle_calibration_error(con) -> None:
    """Re-key hle-source rows whose raw metric is 'score' to the
    calibration-error canonical (lower is better).

    Must run before apply_metric_folds — the fold would otherwise sweep
    these calibration rows into the accuracy view. With catch-all-aware
    registry data, the structured metric-id pre-step already resolves
    hle's `hle.calibration_error` ids to calibration-error and this UPDATE
    matches 0 rows; it stays as the fallback for the pre-step's disabled
    mode (registry data without catch_all flags).

    TODO(upstream): the HLE leaderboard feed labels its calibration-error
    column literally "score" (values 34–89, anti-correlated with accuracy),
    so the resolver lands it on the generic `score` canonical. A registry
    alias can't fix a raw called "score" without breaking `score` everywhere
    else.
    Remove when: the EEE hle ingestion relabels the column; the
    (hle, score → accuracy) registry fold stays valid on its own then.
    """
    con.execute(
        """
        UPDATE results_resolved
        SET metric_id = 'calibration-error',
            metric_resolution_strategy = 'hotfix_hle_calibration'
        WHERE source_config = 'hle'
          AND benchmark_id = 'hle'
          AND LOWER(TRIM(metric_raw)) = 'score'
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE metric_resolution_strategy = 'hotfix_hle_calibration'"
    ).fetchone()[0]
    log.info("resolution_hotfixes: re-keyed %d hle calibration row(s)", n)


# ── 5. HAL SciCode main-problem rate labelled generic "score" ────────


def fix_scicode_hal_main_rate(con) -> None:
    """Re-key hal-scicode rows from generic `score` to `scicode.main`.

    TODO(upstream): HAL's SciCode leaderboard reports the MAIN-problem
    solve rate while every other scicode source reports the sub-problem
    rate; both arrive as raw "score". Different counting style = different
    metric (maintainer ruling) — re-key HAL's rows to the scicode.main
    canonical so the merged default table never mixes the two.
    Remove when: the EEE hal-scicode ingestion labels the metric.
    """
    con.execute(
        """
        UPDATE results_resolved
        SET metric_id = 'scicode.main',
            metric_resolution_strategy = 'hotfix_scicode_main'
        WHERE source_config = 'hal-scicode'
          AND benchmark_id = 'scicode'
          AND LOWER(TRIM(metric_raw)) = 'score'
        """
    )
    n = con.execute(
        "SELECT COUNT(*) FROM results_resolved "
        "WHERE metric_resolution_strategy = 'hotfix_scicode_main'"
    ).fetchone()[0]
    log.info("resolution_hotfixes: re-keyed %d hal scicode main-rate row(s)", n)


def _metadata_events(record, idx, fields, fix_id, evidence, counters):
    events = []
    result = record["evaluation_results"][idx]
    for field, value in fields.items():
        if value is None:
            continue
        pointer = f"/evaluation_results/{idx}/{field}"
        status, _ = pointer_value(record, pointer)
        if status == "malformed":
            counters[f"{fix_id}:skipped_malformed"] += 1
        elif status == "missing":
            events.append(PatchEvent(
                fix_id=fix_id, evaluation_id=record.get("evaluation_id"),
                result_index=idx, evaluation_result_id=result.get("evaluation_result_id"),
                pointer=pointer, old=None, new=value, evidence=evidence,
            ))
            counters[f"{fix_id}:field:{field}"] += 1
    return events


def _patched_from_events(record, events):
    if not events:
        return None
    patched = copy.deepcopy(record)
    for event in events:
        set_pointer(patched, event.pointer, event.new)
    return patched


# vals_ai source metadata.

_VALS_AI_FIX_ID = "vals_ai_max_tokens"
_VALS_AI_SOURCE_CONFIG = "vals-ai"
_VALS_AI_TABLE_NAME = "vals_ai_max_tokens.json"

# The fields the pages document, each written as the type the upstream
# contract gives it.
_VALS_AI_FIELD_TYPES: dict[str, Any] = {"max_tokens": int, "temperature": float}
_VALS_AI_FIELDS = tuple(sorted(_VALS_AI_FIELD_TYPES))

# The datastore revision the vendored table was built from, stated here as
# well so a run that never opens the table can still report it.
_VALS_AI_SOURCE_REVISION = "5d022848fa4a13be240835fabf39a7a5dea3da5f"


def _vals_ai_evidence_sha256() -> str:
    return vendored_sha256(_VALS_AI_TABLE_NAME)


def _vals_ai_benchmark_slug(record: dict[str, Any]) -> str | None:
    details = (record.get("source_metadata") or {}).get("additional_details")
    if not isinstance(details, dict):
        return None
    slug = details.get("benchmark_slug")
    return slug if isinstance(slug, str) else None


class _ValsAiMaxTokens:
    fix_id = _VALS_AI_FIX_ID
    source_config = _VALS_AI_SOURCE_CONFIG

    def __init__(self, table: dict[str, Any]) -> None:
        self._table = table
        self._documented: dict[str, dict[str, Any]] = {
            slug: {field: entry["value"] for field, entry in by_field.items()}
            for slug, by_field in table["documented_statements"].items()
        }
        # Keyed per result and field: one patch fills one field of one result.
        self._by_eval: dict[str, dict[tuple[str | None, str], dict[str, Any]]] = {}
        for patch in table.get("patches", []):
            self._by_eval.setdefault(patch["evaluation_id"], {})[
                (patch.get("evaluation_result_id"), patch["field"])
            ] = patch
        self._patch_count = len(table.get("patches", []))

    @property
    def source_config_revision(self) -> str:
        return self._table["source"]["revision"]

    @property
    def table_sha256(self) -> str:
        return self._table["_table_sha256"]

    def apply(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[PatchEvent], Counter]:
        if self._table.get("policy") == "benchmark_page_default":
            return self._apply_page_defaults(record)
        counters: Counter = Counter()
        slug = _vals_ai_benchmark_slug(record)
        documented = self._documented.get(slug)
        if documented is None:
            return None, [], counters

        evaluation_id = record.get("evaluation_id")
        patches = self._by_eval.get(evaluation_id, {})
        results = record.get("evaluation_results") or []
        events: list[PatchEvent] = []
        patched: dict[str, Any] | None = None

        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            result_id = result.get("evaluation_result_id")
            for field in sorted(documented):
                status, _ = generation_arg(result, field)
                if status == "malformed":
                    # The containers are shared, so no field of this result
                    # can be written and the result is counted once.
                    self._count(counters, "skipped_malformed")
                    break
                if self._table.get("policy") == "benchmark_page_default" and status != "malformed":
                    events.extend(_metadata_events(record, idx, {
                        f"generation_config/additional_details/{field}_source":
                        "row" if status == "present" else "page_methodology"
                    }, self.fix_id, "benchmark_page", counters))
                patch = patches.get((result_id, field))
                if self._table.get("policy") == "benchmark_page_default":
                    patch = {"new": documented[field], "tier": "benchmark_page"}
                if status == "present":
                    if patch is not None:
                        self._count(counters, "stale")
                    continue
                if patch is None:
                    self._count(counters, "unpatched_gap")
                    continue
                if patch.get("expected_old") is not None:
                    self._count(counters, "stale")
                    continue

                if patched is None:
                    patched = copy.deepcopy(record)
                value = _VALS_AI_FIELD_TYPES[field](patch["new"])
                set_generation_arg(patched["evaluation_results"][idx], field, value)
                events.append(
                    PatchEvent(
                        fix_id=_VALS_AI_FIX_ID,
                        evaluation_id=evaluation_id,
                        result_index=idx,
                        evaluation_result_id=result_id,
                        pointer=GENERATION_ARGS_POINTER.format(idx=idx, field=field),
                        old=None,
                        new=value,
                        evidence=patch["tier"],
                    )
                )
                self._count(counters, "applied")
                self._count(counters, f"tier:{patch['tier']}")
                self._count(counters, f"field:{field}")
                self._count(counters, f"benchmark:{slug}")

        return _patched_from_events(record, events), events, counters

    def _apply_page_defaults(self, record):
        counters = Counter()
        events = []
        slug = _vals_ai_benchmark_slug(record)
        defaults = self._documented.get(slug, {})
        source_details = (record.get("source_metadata") or {}).get("additional_details") or {}
        statements = self._table["documented_statements"].get(slug, {})
        page_url = source_details.get("leaderboard_page_url") or next(iter(statements.values()), {}).get("url")
        if statements and page_url and record.get("evaluation_results"):
            pointer = "/source_metadata/additional_details/vals_methodology_url"
            if pointer_value(record, pointer)[0] == "missing":
                events.append(PatchEvent(self.fix_id, record.get("evaluation_id"), 0,
                                        record["evaluation_results"][0].get("evaluation_result_id"),
                                        pointer, None, page_url + "#methodology", "benchmark_page"))
        for idx, result in enumerate(record.get("evaluation_results") or []):
            if not isinstance(result, dict):
                continue
            fields = {}
            row = (result.get("score_details") or {}).get("details") or {}
            generation = result.get("generation_config") or {}
            details = generation.get("additional_details") or {} if isinstance(generation, dict) else {}
            for field, raw_key, parse in (("temperature", "temperature", float),
                                          ("max_tokens", "max_output_tokens", int)):
                status, _ = generation_arg(result, field)
                if status == "malformed":
                    self._count(counters, "skipped_malformed")
                    continue
                source = None
                value = None
                if status == "present":
                    source = "row"
                elif row.get(raw_key) is not None:
                    try:
                        value = parse(row[raw_key])
                    except (TypeError, ValueError):
                        pass
                    if value is not None and (field != "max_tokens" or value > 0):
                        source = "row"
                    else:
                        value = None
                elif defaults.get(field) is not None:
                    value, source = defaults[field], "page_methodology"
                if source is not None:
                    fields[f"generation_config/additional_details/{field}_source"] = source
                if value is not None:
                    fields[f"generation_config/generation_args/{field}"] = value
            if row.get("top_p") is not None:
                try:
                    fields["generation_config/generation_args/top_p"] = float(row["top_p"])
                except (TypeError, ValueError):
                    pass
            reasoning = row.get("reasoning", details.get("reasoning"))
            if isinstance(reasoning, str):
                try:
                    reasoning = json.loads(reasoning)
                except ValueError:
                    reasoning = None
            if isinstance(reasoning, bool):
                fields["generation_config/generation_args/reasoning"] = reasoning
                # The source boolean has a typed schema slot; remove its obsolete duplicate.
                old = details.get("reasoning")
                if old is not None:
                    events.append(PatchEvent(self.fix_id, record.get("evaluation_id"), idx,
                                            result.get("evaluation_result_id"),
                                            f"/evaluation_results/{idx}/generation_config/additional_details/reasoning",
                                            old, None, "row_reasoning"))
            harness = row.get("harness")
            if harness is not None:
                label = row.get("harness_label") or source_details.get("harness_label") or ""
                key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "harness"
                fields[f"generation_config/generation_args/agentic_eval_config/additional_details/vals_{key}"] = str(harness)
            for key in ("reasoning_effort", "verbosity", "compute_effort"):
                if row.get(key) is not None:
                    fields[f"generation_config/additional_details/{key}"] = str(row[key])
            added = _metadata_events(record, idx, fields, self.fix_id, "row_or_benchmark_page", counters)
            events.extend(added)
            if added:
                self._count(counters, f"benchmark:{slug}")
        return _patched_from_events(record, events), events, counters

    def _count(self, counters: Counter, name: str) -> None:
        counters[f"{_VALS_AI_FIX_ID}:{name}"] += 1

    def summarise(self, counters: Counter) -> dict[str, Any]:
        def grouped(prefix: str) -> dict[str, int]:
            return {
                key.split(":", 1)[1]: value
                for key, value in sorted(counters.items())
                if key.startswith(prefix)
            }

        matched = counters["applied"] + counters["stale"]
        return {
            "by_tier": grouped("tier:"),
            "by_field": grouped("field:"),
            "by_benchmark": grouped("benchmark:"),
            "patches": self._patch_count,
            "unused": max(0, self._patch_count - matched),
            "stale": counters["stale"],
            "unpatched_gap": counters["unpatched_gap"],
            "skipped_malformed": counters["skipped_malformed"],
        }


def build_vals_ai_datafix() -> _ValsAiMaxTokens:
    """Build the vals_ai missing-metadata hotfix.

    TODO(upstream): https://github.com/evaleval/every_eval_ever/pull/306
    carries source-published generation settings into aggregate records.
    Remove when: the pinned datastore has been re-ingested with those
    adapter changes and a source audit reports zero missing covered fields.
    """
    return _ValsAiMaxTokens(load_table(_VALS_AI_TABLE_NAME))


# hfopenllm_v2 source metadata.

_HFOPENLLM_V2_FIX_ID = "hfopenllm_v2_harness_configs"
_HFOPENLLM_V2_SOURCE_CONFIG = "hfopenllm_v2"
_HFOPENLLM_V2_TABLE_NAME = "hfopenllm_v2_harness_configs.json.gz"

# The results-dataset revision the vendored table was built from, stated
# here as well so a run that never opens the table can still report it.
_HFOPENLLM_V2_SOURCE_REVISION = "aa81ecc38fdc5708254b833923368970efdf5ef5"


def _hfopenllm_v2_evidence_sha256() -> str:
    return vendored_sha256(_HFOPENLLM_V2_TABLE_NAME)

# Result names this fix fills, in the spelling the upstream records use.
_HFOPENLLM_V2_GENERATIVE_BENCHMARKS = ("IFEval", "MATH Level 5")

# Generation args taken from the harness config, in write order.
_HFOPENLLM_V2_FIELDS = ("temperature", "max_tokens")


class _HfOpenLlmHarnessConfigs:
    fix_id = _HFOPENLLM_V2_FIX_ID
    source_config = _HFOPENLLM_V2_SOURCE_CONFIG

    def __init__(self, table: dict[str, Any]) -> None:
        self._table = table
        self._defaults: dict[str, dict[str, Any]] = table["defaults"]
        self._overrides: dict[str, dict[str, Any]] = table["overrides"]
        self._verified: dict[str, set[str]] = {
            benchmark: set(models)
            for benchmark, models in table["verified"].items()
        }
        self._conflicts: dict[str, set[str]] = {
            model: set(benchmarks)
            for model, benchmarks in table["conflicts"].items()
        }

    @property
    def source_config_revision(self) -> str:
        return self._table["source"]["revision"]

    @property
    def table_sha256(self) -> str:
        return self._table["_table_sha256"]

    def config_for(self, model_id: str, benchmark: str, precision: str | None = None) -> dict[str, Any] | None:
        if "by_precision" in self._table:
            profile = self._table["by_precision"].get(model_id, {}).get(precision, {}).get(benchmark)
            return self._table["profiles"].get(profile)
        if model_id not in self._verified.get(benchmark, ()):
            return None
        override = self._overrides.get(model_id, {}).get(benchmark)
        return override if override is not None else self._defaults.get(benchmark)

    def apply(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[PatchEvent], Counter]:
        counters: Counter = Counter()
        model_id = (record.get("model_info") or {}).get("id")
        evaluation_id = record.get("evaluation_id")
        results = record.get("evaluation_results") or []
        events: list[PatchEvent] = []
        patched: dict[str, Any] | None = None

        precision = ((record.get("model_info") or {}).get("additional_details") or {}).get("precision")
        library = self._table.get("libraries", {}).get(model_id, {}).get(precision, {})
        if results:
            for field, value in library.items():
                pointer = f"/eval_library/additional_details/{field}"
                if pointer_value(record, pointer)[0] == "missing":
                    events.append(PatchEvent(self.fix_id, evaluation_id, 0,
                                            results[0].get("evaluation_result_id"),
                                            pointer, None, value, "harness_dump"))
        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            benchmark = result.get("evaluation_name")
            if "by_precision" not in self._table and benchmark not in _HFOPENLLM_V2_GENERATIVE_BENCHMARKS:
                continue
            precision = ((record.get("model_info") or {}).get("additional_details") or {}).get("precision")
            config = self.config_for(model_id, benchmark, precision) if model_id else None
            if config is None:
                if benchmark in self._conflicts.get(model_id, ()):
                    self._count(counters, "conflict")
                else:
                    self._count(counters, "unmatched")
                continue

            if "by_precision" in self._table:
                fields = {}
                for field in _HFOPENLLM_V2_FIELDS:
                    fields[f"generation_config/generation_args/{field}"] = config.get(field)
                for field in ("output_type", "num_fewshot", "do_sample"):
                    value = config.get(field)
                    fields[f"generation_config/additional_details/{field}"] = (
                        value if isinstance(value, str) or value is None else json.dumps(value)
                    )
                for field in ("hf_split", "samples_number"):
                    fields[f"source_data/{field}"] = config.get(field)
                fields["score_details/uncertainty/num_samples"] = config.get("samples_number")
                path = config.get("dataset_path")
                if path is not None and path != (result.get("source_data") or {}).get("hf_repo"):
                    fields["source_data/additional_details/harness_dataset_path"] = path
                if config.get("dataset_size") is not None:
                    fields["source_data/additional_details/dataset_size"] = str(config["dataset_size"])
                added = _metadata_events(record, idx, fields, self.fix_id, benchmark, counters)
                events.extend(added)
                if added:
                    self._count(counters, f"benchmark:{benchmark}")
                continue
            filled = False
            for field in _HFOPENLLM_V2_FIELDS:
                value = config.get(field)
                if value is None:
                    continue
                status, _ = generation_arg(result, field)
                if status == "malformed":
                    self._count(counters, "skipped_malformed")
                    continue
                if status == "present":
                    continue
                if patched is None:
                    patched = copy.deepcopy(record)
                set_generation_arg(patched["evaluation_results"][idx], field, value)
                events.append(
                    PatchEvent(
                        fix_id=_HFOPENLLM_V2_FIX_ID,
                        evaluation_id=evaluation_id,
                        result_index=idx,
                        evaluation_result_id=result.get("evaluation_result_id"),
                        pointer=GENERATION_ARGS_POINTER.format(idx=idx, field=field),
                        old=None,
                        new=value,
                        evidence=benchmark,
                    )
                )
                filled = True
            if filled:
                self._count(counters, f"benchmark:{benchmark}")

        return _patched_from_events(record, events), events, counters

    def _count(self, counters: Counter, name: str) -> None:
        counters[f"{_HFOPENLLM_V2_FIX_ID}:{name}"] += 1

    def summarise(self, counters: Counter) -> dict[str, Any]:
        by_benchmark = {
            key.split(":", 1)[1]: value
            for key, value in sorted(counters.items())
            if key.startswith("benchmark:")
        }
        return {
            "by_benchmark": by_benchmark,
            "verified_models": {
                benchmark: len(models)
                for benchmark, models in sorted(self._verified.items())
            },
            "unmatched": counters["unmatched"],
            "conflict": counters["conflict"],
            "skipped_malformed": counters["skipped_malformed"],
        }


def build_hfopenllm_v2_datafix() -> _HfOpenLlmHarnessConfigs:
    """Build the hfopenllm_v2 missing-metadata hotfix.

    TODO(upstream): https://github.com/evaleval/every_eval_ever/pull/306
    carries source-published generation settings into aggregate records.
    Remove when: the pinned datastore has been re-ingested with those
    adapter changes and a source audit reports zero missing covered fields.
    """
    return _HfOpenLlmHarnessConfigs(load_table(_HFOPENLLM_V2_TABLE_NAME))


# inspect_plan_config source metadata.

_INSPECT_PLAN_CONFIG_FIX_ID = "inspect_plan_config"
# No source config of its own: the evidence is in-record, so the rule holds
# wherever an inspect_ai log was converted.
_INSPECT_PLAN_CONFIG_SOURCE_CONFIG = None

_INSPECT_PLAN_CONFIG_LIBRARY = "inspect_ai"

# The EEE revision the allowlists below were read off.
_INSPECT_PLAN_CONFIG_CHECKED_AT_REVISION = "73761dc1d250913a9326fdc213f7ac2d23d390b6"

# Library strings observed on the affected records, each read against that
# version's own source. A later version is not accepted on its own: it has
# to be checked and added here.
_INSPECT_PLAN_CONFIG_VERSIONS = (
    "inspect_ai:0.3.75",
    "inspect_ai:0.3.77.dev7+gf08c940a.d20250323",
    "inspect_ai:0.3.80.dev25+g91a50728.d20250331",
    "inspect_ai:0.3.90.dev63+g6f7b1737",
)

# A plan's last step has to generate with the run's own model.
_INSPECT_PLAN_CONFIG_TERMINAL_SOLVERS = ("generate", "multiple_choice", "react")

# Every solver name that appears in a plan this rule was checked against.
# Anything else is a scaffold whose calls were not looked at.
_INSPECT_PLAN_CONFIG_KNOWN_SOLVERS = frozenset(
    _INSPECT_PLAN_CONFIG_TERMINAL_SOLVERS
    + (
        "prompt_template",
        "system_message",
        "inspect_evals/set_tool_choice",
        "inspect_evals/setup_tools_from_metadata",
        "inspect_evals/system_prompt_based_on_model",
    )
)

# Parameter names a step could use to generate under its own settings.
_INSPECT_PLAN_CONFIG_OVERRIDE_KEYS = frozenset({
    "temperature", "max_tokens", "max_completion_tokens", "max_output_tokens", "top_p", "top_k",
})

# Where a step records what it was given, in the order the rule reads them.
_INSPECT_PLAN_CONFIG_STEP_PARAM_KEYS = ("params", "params_passed")

# Generation args taken from the plan config, in write order.
_INSPECT_PLAN_CONFIG_FIELDS = ("temperature", "max_tokens", "top_p", "top_k")


def _inspect_plan_config_number(value: Any) -> float | int | None:
    """One plan-config value as a finite number, or None.

    Plan config values are JSON-serialized, so they arrive as strings like
    `"0.0"` and `"4096"`; a value that is already a number is read as it is.
    Booleans are not numbers here, and neither is anything non-finite.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def _inspect_plan_config_temperature(value: Any) -> float | None:
    number = _inspect_plan_config_number(value)
    if number is None or number < 0:
        return None
    return float(number)


def _inspect_plan_config_max_tokens(value: Any) -> int | None:
    number = _inspect_plan_config_number(value)
    if not isinstance(number, int) or number < 1:
        return None
    return number


_INSPECT_PLAN_CONFIG_PARSERS = {"temperature": _inspect_plan_config_temperature, "max_tokens": _inspect_plan_config_max_tokens, "top_p": _inspect_plan_config_temperature, "top_k": _inspect_plan_config_max_tokens}

_INSPECT_PLAN_CONFIG_CONSTANTS = {
    "checked_at_revision": _INSPECT_PLAN_CONFIG_CHECKED_AT_REVISION,
    "fields": list(_INSPECT_PLAN_CONFIG_FIELDS),
    "known_solvers": sorted(_INSPECT_PLAN_CONFIG_KNOWN_SOLVERS),
    "library": _INSPECT_PLAN_CONFIG_LIBRARY,
    "override_keys": sorted(_INSPECT_PLAN_CONFIG_OVERRIDE_KEYS),
    "step_param_keys": list(_INSPECT_PLAN_CONFIG_STEP_PARAM_KEYS),
    "terminal_solvers": list(_INSPECT_PLAN_CONFIG_TERMINAL_SOLVERS),
    "versions": list(_INSPECT_PLAN_CONFIG_VERSIONS),
}

_INSPECT_PLAN_CONFIG_SOURCE_REVISION = _INSPECT_PLAN_CONFIG_CHECKED_AT_REVISION


def _inspect_plan_config_evidence_sha256() -> str:
    """The rule's own evidence, hashed the way a patch table is.

    There is no table to check, so what the overlay fingerprint has to
    follow is the allowlists: a version or a solver added here changes what
    a bake fills just as a new table row would.
    """
    return sha256_text(canonical_json(_INSPECT_PLAN_CONFIG_CONSTANTS))


def _inspect_plan_config_eval_plan(result: dict[str, Any]) -> dict[str, Any] | None:
    """The plan a result carries, or None when it carries none."""
    gen_config = result.get("generation_config")
    if not isinstance(gen_config, dict):
        return None
    gen_args = gen_config.get("generation_args")
    if not isinstance(gen_args, dict):
        return None
    plan = gen_args.get("eval_plan")
    return plan if isinstance(plan, dict) else None


def _inspect_plan_config_names_an_override(node: Any) -> bool:
    """Whether a step's parameters name a temperature or an output cap."""
    if isinstance(node, dict):
        if _INSPECT_PLAN_CONFIG_OVERRIDE_KEYS & set(node):
            return True
        return any(_inspect_plan_config_names_an_override(child) for child in node.values())
    if isinstance(node, list):
        return any(_inspect_plan_config_names_an_override(child) for child in node)
    return False


def _inspect_plan_config_model_param(step: dict[str, Any]) -> Any:
    for key in _INSPECT_PLAN_CONFIG_STEP_PARAM_KEYS:
        params = step.get(key)
        if isinstance(params, dict) and params.get("model") is not None:
            return params["model"]
    return None


def _inspect_plan_config_library_rejection(record: dict[str, Any]) -> str | None:
    """Why the record's library disqualifies it, or None."""
    library = record.get("eval_library")
    if not isinstance(library, dict):
        return "library_absent"
    version = library.get("version")
    if library.get("name") != _INSPECT_PLAN_CONFIG_LIBRARY or not isinstance(version, str):
        return "library_not_inspect_ai"
    if len(version.split(",")) > 1:
        return "library_names_another_package"
    if version not in _INSPECT_PLAN_CONFIG_VERSIONS:
        return "version_not_checked"
    return None


def _inspect_plan_config_plan_rejection(plan: dict[str, Any]) -> str | None:
    """Why the plan disqualifies its result, or None."""
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return "plan_without_steps"
    parsed_steps = []
    for step in steps:
        if not isinstance(step, str):
            return "step_unparsed"
        try:
            parsed = json.loads(step)
        except ValueError:
            return "step_unparsed"
        if not isinstance(parsed, dict):
            return "step_unparsed"
        if parsed.get("solver") not in _INSPECT_PLAN_CONFIG_KNOWN_SOLVERS:
            return "unknown_solver"
        if any(_inspect_plan_config_names_an_override(parsed.get(key)) for key in _INSPECT_PLAN_CONFIG_STEP_PARAM_KEYS):
            return "step_sets_generation_args"
        parsed_steps.append(parsed)
    terminal = parsed_steps[-1]
    if terminal.get("solver") not in _INSPECT_PLAN_CONFIG_TERMINAL_SOLVERS:
        return "terminal_solver_not_checked"
    if _inspect_plan_config_model_param(terminal) is not None:
        return "terminal_solver_names_a_model"
    return None


class _InspectPlanConfig:
    fix_id = _INSPECT_PLAN_CONFIG_FIX_ID
    source_config = _INSPECT_PLAN_CONFIG_SOURCE_CONFIG

    @property
    def source_config_revision(self) -> str:
        return _INSPECT_PLAN_CONFIG_CHECKED_AT_REVISION

    @property
    def table_sha256(self) -> str:
        return _inspect_plan_config_evidence_sha256()

    def apply(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[PatchEvent], Counter]:
        counters: Counter = Counter()
        evaluation_id = record.get("evaluation_id")
        events: list[PatchEvent] = []
        patched: dict[str, Any] | None = None
        library_reason = _inspect_plan_config_library_rejection(record)

        for idx, result in enumerate(record.get("evaluation_results") or []):
            if not isinstance(result, dict):
                continue
            plan = _inspect_plan_config_eval_plan(result)
            if plan is None:
                continue
            config = plan.get("config")
            if not isinstance(config, dict):
                continue

            proposed: dict[str, Any] = {}
            for field in _INSPECT_PLAN_CONFIG_FIELDS:
                # The plan was read through the same two containers, so both
                # are dicts here: the field is present or it is missing.
                if config.get(field) is None:
                    continue
                if generation_arg(result, field)[0] == "missing":
                    proposed[field] = config[field]
            if not proposed:
                continue

            reason = library_reason or _inspect_plan_config_plan_rejection(plan)
            if reason is not None:
                self._count(counters, f"skipped:{reason}")
                continue

            for field, raw in proposed.items():
                value = _INSPECT_PLAN_CONFIG_PARSERS[field](raw)
                if value is None:
                    self._count(counters, f"skipped:malformed_{field}")
                    continue
                if patched is None:
                    patched = copy.deepcopy(record)
                set_generation_arg(patched["evaluation_results"][idx], field, value)
                events.append(
                    PatchEvent(
                        fix_id=_INSPECT_PLAN_CONFIG_FIX_ID,
                        evaluation_id=evaluation_id,
                        result_index=idx,
                        evaluation_result_id=result.get("evaluation_result_id"),
                        pointer=GENERATION_ARGS_POINTER.format(idx=idx, field=field),
                        old=None,
                        new=value,
                        evidence="eval_plan.config",
                    )
                )
                self._count(counters, f"field:{field}")

        return patched, events, counters

    def _count(self, counters: Counter, name: str) -> None:
        counters[f"{_INSPECT_PLAN_CONFIG_FIX_ID}:{name}"] += 1

    def summarise(self, counters: Counter) -> dict[str, Any]:
        def grouped(prefix: str) -> dict[str, int]:
            return {
                key.split(":", 1)[1]: value
                for key, value in sorted(counters.items())
                if key.startswith(prefix)
            }

        return {
            "by_field": grouped("field:"),
            "skipped": grouped("skipped:"),
            "versions": list(_INSPECT_PLAN_CONFIG_VERSIONS),
        }


def build_inspect_plan_config_datafix() -> _InspectPlanConfig:
    """Build the inspect_plan_config missing-metadata hotfix.

    TODO(upstream): https://github.com/evaleval/every_eval_ever/pull/306
    carries source-published generation settings into aggregate records.
    Remove when: the pinned datastore has been re-ingested with those
    adapter changes and a source audit reports zero missing covered fields.
    """
    return _InspectPlanConfig()


# arc_agi source metadata.

_ARC_AGI_FIX_ID = "arc_agi_logged_request_args"
_ARC_AGI_SOURCE_CONFIG = "arc-agi"
_ARC_AGI_TABLE_NAME = "arc_agi_logged_request_args.json"

# The published-output revisions the vendored table was built from, one per
# public set, stated here as well so a run that never opens the table can
# still report them.
_ARC_AGI_SOURCE_REVISION = (
    "3e9c9d1a8402aff82356815c106cecaea65cb7d9"
    "+026789c1c12a4c34580a32e84dcaf5630d7e8f31"
)


def _arc_agi_evidence_sha256() -> str:
    return vendored_sha256(_ARC_AGI_TABLE_NAME)

# Result names this fix fills, in the spelling the upstream records use,
# each paired with the dataset whose logs may attest it.
_ARC_AGI_PATCHED_RESULT_NAMES = ("v1_Public_Eval", "v2_Public_Eval")

# Only the score carries the run's settings; the paired cost is a price.
_ARC_AGI_SCORE_METRIC_ID = "score"

# Generation args taken from the logged request arguments, in write order.
_ARC_AGI_FIELDS = ("temperature", "max_tokens")


def _arc_agi_raw_model_id(record: dict[str, Any]) -> str | None:
    """The leaderboard's own model id, which names the published run."""
    details = (record.get("model_info") or {}).get("additional_details")
    if not isinstance(details, dict):
        return None
    value = details.get("raw_model_id")
    return value if isinstance(value, str) else None


class _ArcAgiLoggedRequestArgs:
    fix_id = _ARC_AGI_FIX_ID
    source_config = _ARC_AGI_SOURCE_CONFIG

    def __init__(self, table: dict[str, Any], result_names: tuple[str, ...] | None = None) -> None:
        self._result_names = result_names or _ARC_AGI_PATCHED_RESULT_NAMES
        self._table = table
        self._by_eval: dict[str, dict[str | None, dict[str, Any]]] = {}
        for patch in table["patches"] + table.get("rejected_patches", []):
            self._by_eval.setdefault(patch["evaluation_id"], {})[
                patch.get("evaluation_result_id")
            ] = patch
        self._patch_count = len(table["patches"]) + len(table.get("rejected_patches", []))

    @property
    def source_config_revision(self) -> str:
        return self._table["source"]["revision"]

    @property
    def table_sha256(self) -> str:
        return self._table["_table_sha256"]

    def apply(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[PatchEvent], Counter]:
        counters: Counter = Counter()
        evaluation_id = record.get("evaluation_id")
        patches = self._by_eval.get(evaluation_id, {})
        events: list[PatchEvent] = []
        patched: dict[str, Any] | None = None

        for idx, result in enumerate(record.get("evaluation_results") or []):
            if not isinstance(result, dict):
                continue
            if result.get("evaluation_name") not in self._result_names:
                continue
            metric = (result.get("metric_config") or {}).get("metric_id")
            if metric != _ARC_AGI_SCORE_METRIC_ID:
                continue

            patch = patches.get(result.get("evaluation_result_id"))
            if patch is None:
                if generation_arg(result, "max_tokens")[0] == "missing":
                    self._count(counters, "unpatched_gap")
                continue
            self._count(counters, "matched")
            if patch.get("expected_old") is not None:
                self._count(counters, "stale")
                continue

            extra = patch.get("metadata", {})
            events.extend(_metadata_events(record, idx, extra, self.fix_id,
                                           patch["directory"], counters))
            filled = False
            stale = False
            for field in _ARC_AGI_FIELDS:
                value = patch["fields"].get(field)
                if value is None:
                    continue
                status, _ = generation_arg(result, field)
                if status == "malformed":
                    self._count(counters, "skipped_malformed")
                    continue
                if status == "present":
                    stale = True
                    continue
                if patched is None:
                    patched = copy.deepcopy(record)
                set_generation_arg(patched["evaluation_results"][idx], field, value)
                events.append(
                    PatchEvent(
                        fix_id=_ARC_AGI_FIX_ID,
                        evaluation_id=evaluation_id,
                        result_index=idx,
                        evaluation_result_id=result.get("evaluation_result_id"),
                        pointer=GENERATION_ARGS_POINTER.format(idx=idx, field=field),
                        old=None,
                        new=value,
                        evidence=patch["directory"],
                    )
                )
                self._count(counters, f"field:{field}")
                filled = True
            if stale:
                self._count(counters, "stale")
            if filled:
                self._count(counters, "applied")
                self._count(counters, f"result_name:{patch['evaluation_name']}")
                self._count(counters, f"cap_key:{patch['cap_key']}")

        return _patched_from_events(record, events), events, counters

    def _count(self, counters: Counter, name: str) -> None:
        counters[f"{_ARC_AGI_FIX_ID}:{name}"] += 1

    def summarise(self, counters: Counter) -> dict[str, Any]:
        def grouped(prefix: str) -> dict[str, int]:
            return {
                key.split(":", 1)[1]: value
                for key, value in sorted(counters.items())
                if key.startswith(prefix)
            }

        return {
            "by_field": grouped("field:"),
            "by_result_name": grouped("result_name:"),
            "by_cap_key": grouped("cap_key:"),
            "patches": self._patch_count,
            "unused": self._patch_count - counters["matched"],
            "stale": counters["stale"],
            "unpatched_gap": counters["unpatched_gap"],
            "skipped_malformed": counters["skipped_malformed"],
        }


def build_arc_agi_datafix() -> _ArcAgiLoggedRequestArgs:
    """Build the arc_agi missing-metadata hotfix.

    TODO(upstream): https://github.com/evaleval/every_eval_ever/pull/306
    carries source-published generation settings into aggregate records.
    Remove when: the pinned datastore has been re-ingested with those
    adapter changes and a source audit reports zero missing covered fields.
    """
    return _ArcAgiLoggedRequestArgs(
        load_table(_ARC_AGI_TABLE_NAME), _ARC_AGI_PATCHED_RESULT_NAMES,
    )


_PUBLISHED_METADATA_FIX_ID = "published_source_metadata"
_PUBLISHED_METADATA_REVISION = "source-record-rules-v1"
_PUBLISHED_METADATA_RULES = {
    "hle_source": "Scale SEAL Humanity's Last Exam Leaderboard",
    "hle_library": "Scale SEAL HLE leaderboard",
    "hle_judge_prefix": "Scale SEAL evaluates Humanity's Last Exam at temperature 0.0. ",
    "hle_policy": "Each model on the leaderboard is evaluated on all public questions of Humanity's Last Exam with temperature 0.0 when configurable or stated otherwise.",
    "hle_note_patterns": [
        r"Sampled at temperature = (?P<temperature>\d+(?:\.\d+)?), top_p = (?P<top_p>\d+(?:\.\d+)?)\.",
        r"Sampled at temperature (?P<temperature>\d+(?:\.\d+)?)\.?",
    ],
    "mmlu_source_repo": "TIGER-Lab/mmlu_pro_leaderboard_submission",
    "mmlu_obsolete_style": "5-shot CoT",
    "mmlu_shot_marker": r"\((\d+)-shot\)",
    "mmlu_obsolete_description": "Overall accuracy across the ~12,000-question MMLU-Pro benchmark, evaluated 5-shot with chain-of-thought.",
    "mmlu_description": "Overall accuracy across the ~12,000-question MMLU-Pro benchmark.",
    "mmlu_subject_description": r"(Accuracy on the MMLU-Pro .+ subset), evaluated 5-shot with chain-of-thought\.",
}


def _published_metadata_sha():
    return sha256_text(canonical_json(_PUBLISHED_METADATA_RULES))


class _PublishedMetadata:
    fix_id = _PUBLISHED_METADATA_FIX_ID
    source_config = None
    source_config_revision = _PUBLISHED_METADATA_REVISION

    @property
    def table_sha256(self):
        return _published_metadata_sha()

    def apply(self, record):
        events = []
        counters = Counter()
        source = record.get("source_metadata") or {}
        library = (record.get("eval_library") or {}).get("name")
        model = record.get("model_info") or {}
        model_details = model.get("additional_details") or {}
        is_hle = (source.get("source_name") == _PUBLISHED_METADATA_RULES["hle_source"]
                  and library == _PUBLISHED_METADATA_RULES["hle_library"])
        note = model_details.get("source_row_note")
        note_values = {}
        if is_hle and isinstance(note, str):
            for pattern in _PUBLISHED_METADATA_RULES["hle_note_patterns"]:
                match = re.fullmatch(pattern, note.strip())
                if match:
                    note_values = {key: float(value) for key, value in match.groupdict().items()}
                    break
        if is_hle and record.get("evaluation_results"):
            pointer = "/source_metadata/additional_details/source_generation_policy"
            if pointer_value(record, pointer)[0] == "missing":
                events.append(PatchEvent(self.fix_id, record.get("evaluation_id"), 0,
                                        record["evaluation_results"][0].get("evaluation_result_id"),
                                        pointer, None, _PUBLISHED_METADATA_RULES["hle_policy"], "published_source_policy"))
        for idx, result in enumerate(record.get("evaluation_results") or []):
            if not isinstance(result, dict):
                continue
            fields = {}
            data = result.get("source_data") or {}
            uncertainty = (result.get("score_details") or {}).get("uncertainty") or {}
            if library in {"inspect_ai", "lm_eval"}:
                count = uncertainty.get("num_samples")
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                    key = "samples_number" if data.get("source_type") == "hf_dataset" else "additional_details/samples_number"
                    path = f"source_data/{key}"
                    value = count if key == "samples_number" else str(count)
                    status, old = pointer_value(record, f"/evaluation_results/{idx}/{path}")
                    if status == "present" and old != value:
                        self._correct(record, idx, path, old, value, events, counters)
                    else:
                        fields[path] = value
                plan = _inspect_plan_config_eval_plan(result) if library == "inspect_ai" else None
                if plan:
                    version = (record.get("eval_library") or {}).get("version") or ""
                    packages = {part.split(":", 1)[0] for part in version.split(",") if ":" in part}
                    unverified = sorted(packages - {"inspect_ai", "inspect_evals"})
                    if unverified:
                        fields["generation_config/additional_details/plan_config_unverified_packages"] = ",".join(unverified)
                    steps = plan.get("steps") or []
                    if steps:
                        try:
                            terminal = json.loads(steps[-1]) if isinstance(steps[-1], str) else steps[-1]
                        except ValueError:
                            terminal = None
                        if isinstance(terminal, dict):
                            fields["generation_config/additional_details/terminal_solver"] = terminal.get("solver")
                    if not unverified:
                        for key, value in (plan.get("config") or {}).items():
                            # Operational concurrency is descriptive even for an unreviewed solver.
                            if key == "max_connections" and value is not None:
                                fields[f"generation_config/additional_details/{key}"] = value if isinstance(value, str) else json.dumps(value)
                if (plan and _inspect_plan_config_library_rejection(record) is None
                        and _inspect_plan_config_plan_rejection(plan) is None):
                    config = plan.get("config") or {}
                    gen = result.get("generation_config") or {}
                    model_config = gen.get("additional_details") or {}
                    typed = gen.get("generation_args") or {}
                    effective = {**model_config, **config}
                    for field in _INSPECT_PLAN_CONFIG_FIELDS:
                        raw = effective.get(field)
                        value = _INSPECT_PLAN_CONFIG_PARSERS[field](raw)
                        if value is not None:
                            fields[f"generation_config/generation_args/{field}"] = value
                    for key, value in config.items():
                        if value is not None:
                            fields[f"generation_config/additional_details/{key}"] = value if isinstance(value, str) else json.dumps(value)
                    steps = plan.get("steps") or []
                    if steps:
                        fields["generation_config/additional_details/terminal_solver"] = json.loads(steps[-1]).get("solver")
                    origins = {"plan_config" if config.get(field) is not None else "model_config"
                               for field in _INSPECT_PLAN_CONFIG_FIELDS
                               if effective.get(field) is not None or typed.get(field) is not None}
                    if origins:
                        fields["generation_config/additional_details/generation_args_source"] = next(iter(origins)) if len(origins) == 1 else "mixed"
            if library == "lm_eval":
                generation = result.get("generation_config") or {}
                output_type = (generation.get("additional_details") or {}).get("output_type")
                if output_type in {"multiple_choice", "loglikelihood", "loglikelihood_rolling"}:
                    for field in ("temperature", "max_tokens", "top_p", "top_k"):
                        path = f"generation_config/generation_args/{field}"
                        status, value = pointer_value(record, f"/evaluation_results/{idx}/{path}")
                        if status == "present":
                            self._correct(record, idx, path, value, None, events, counters)
            if library == "helm" and source.get("source_type") == "evaluation_run":
                score_details = (result.get("score_details") or {}).get("details") or {}
                raw_split = score_details.get("split")
                split = {"train": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}.get(raw_split)
                if data.get("source_type") == "hf_dataset":
                    fields["source_data/hf_split"] = split
                    ids = data.get("sample_ids")
                    if isinstance(ids, list) and all(isinstance(value, str) for value in ids):
                        unique = list(dict.fromkeys(ids))
                        if len(unique) != len(ids):
                            self._correct(record, idx, "source_data/sample_ids", ids, unique, events, counters)
                            old = data.get("samples_number")
                            if old == len(ids):
                                self._correct(record, idx, "source_data/samples_number", old, len(unique), events, counters)
                            elif old is None:
                                fields["source_data/samples_number"] = len(unique)
            if is_hle:
                if note:
                    fields["generation_config/additional_details/source_row_note"] = note
                for field, value in note_values.items():
                    fields[f"generation_config/generation_args/{field}"] = value
                # These exact values and judge prose identify the old adapter defaults.
                judge_path = "metric_config/llm_scoring/input_prompt"
                judge = ((result.get("metric_config") or {}).get("llm_scoring") or {}).get("input_prompt")
                old_adapter = isinstance(judge, str) and judge.startswith(_PUBLISHED_METADATA_RULES["hle_judge_prefix"])
                if old_adapter:
                    self._correct(record, idx, judge_path, judge, judge.removeprefix(_PUBLISHED_METADATA_RULES["hle_judge_prefix"]), events, counters)
                    for field, old_value in (("temperature", 0.0), ("max_attempts", 1)):
                        path = f"generation_config/generation_args/{field}"
                        status, old = pointer_value(record, f"/evaluation_results/{idx}/{path}")
                        if status == "present" and old == old_value:
                            self._correct(record, idx, path, old, note_values.get(field), events, counters)
            if data.get("hf_repo") == _PUBLISHED_METADATA_RULES["mmlu_source_repo"]:
                description_path = "metric_config/evaluation_description"
                _, description = pointer_value(record, f"/evaluation_results/{idx}/{description_path}")
                if description == _PUBLISHED_METADATA_RULES["mmlu_obsolete_description"]:
                    self._correct(record, idx, description_path, description,
                                  _PUBLISHED_METADATA_RULES["mmlu_description"], events, counters)
                elif isinstance(description, str):
                    matched = re.fullmatch(_PUBLISHED_METADATA_RULES["mmlu_subject_description"], description)
                    if matched:
                        self._correct(record, idx, description_path, description,
                                      matched.group(1) + ".", events, counters)
                raw_name = model_details.get("raw_model_name") or model.get("name") or ""
                marker = re.search(_PUBLISHED_METADATA_RULES["mmlu_shot_marker"], raw_name, re.IGNORECASE)
                style = f"{int(marker.group(1))}-shot" if marker else None
                for container in ("source_data", "metric_config"):
                    path = f"{container}/additional_details/prompt_style"
                    status, old = pointer_value(record, f"/evaluation_results/{idx}/{path}")
                    if status == "present" and old == _PUBLISHED_METADATA_RULES["mmlu_obsolete_style"]:
                        self._correct(record, idx, path, old, style, events, counters)
                    elif style is not None:
                        fields[path] = style
            events.extend(_metadata_events(record, idx, fields, self.fix_id, "embedded_source_metadata", counters))
        return _patched_from_events(record, events), events, counters

    def _correct(self, record, idx, path, old, new, events, counters):
        if old == new:
            return
        events.append(PatchEvent(
            fix_id=self.fix_id, evaluation_id=record.get("evaluation_id"), result_index=idx,
            evaluation_result_id=record["evaluation_results"][idx].get("evaluation_result_id"),
            pointer=f"/evaluation_results/{idx}/{path}", old=old, new=new,
            evidence="scorer_sample_count" if path.endswith("samples_number") else "obsolete_adapter_default",
        ))
        counters[f"{self.fix_id}:corrected:{path}"] += 1

    def summarise(self, counters):
        return {"by_field": {key: value for key, value in sorted(counters.items())
                             if key.startswith(("field:", "corrected:"))},
                "skipped_malformed": counters["skipped_malformed"]}


def build_published_metadata_datafix():
    """Carry source metadata and correct exact obsolete adapter defaults.

    TODO(upstream): https://github.com/evaleval/every_eval_ever/pull/306
    removes blanket HLE/MMLU-Pro defaults, reports scorer-specific counts,
    and carries published Inspect/lm_eval/HELM metadata.
    Remove when: corrected adapter/converter records replace historical captures
    and a source audit reports zero missing or obsolete covered fields.
    """
    return _PublishedMetadata()

"""Source metadata rules work on identities and values invented by fixtures."""
import copy
from collections import Counter
import json

from eval_card_backend.canonicalise.resolution_hotfixes import (
    _ArcAgiLoggedRequestArgs,
    _HfOpenLlmHarnessConfigs,
    _PublishedMetadata,
    _ValsAiMaxTokens,
)
from eval_card_backend.datafix.core import apply_fixes


def record(*, library="fixture", source="fixture", result=None):
    return {
        "evaluation_id": "arbitrary-run",
        "model_info": {"id": "maker/model", "name": "Model", "additional_details": {"precision": "float16"}},
        "eval_library": {"name": library, "version": "inspect_ai:0.3.75"},
        "source_metadata": {"source_name": source},
        "evaluation_results": [result or {"evaluation_name": "Task", "source_data": {"source_type": "hf_dataset", "hf_repo": "data/original"}}],
    }


def test_harness_metadata_requires_exact_precision_and_preserves_stated_values():
    profile = {"output_type": "multiple_choice", "num_fewshot": 7, "samples_number": 13,
               "dataset_size": 17, "hf_split": "test", "dataset_path": "data/actual"}
    fix = _HfOpenLlmHarnessConfigs({"defaults": {}, "verified": {}, "overrides": {}, "conflicts": {},
                                   "profiles": {"p": profile}, "by_precision": {"maker/model": {"float16": {"Task": "p"}}},
                                   "libraries": {"maker/model": {"float16": {"git_hash": "fixture-revision"}}}})
    raw = record()
    raw["evaluation_results"][0]["source_data"]["hf_split"] = "validation"
    before = copy.deepcopy(raw)
    patched, events, _ = fix.apply(raw)
    assert raw == before
    result = patched["evaluation_results"][0]
    assert result["source_data"] == {"source_type": "hf_dataset", "hf_repo": "data/original", "hf_split": "validation",
                                     "samples_number": 13, "additional_details": {"dataset_size": "17", "harness_dataset_path": "data/actual"}}
    assert result["score_details"]["uncertainty"]["num_samples"] == 13
    assert result["generation_config"]["additional_details"] == {"output_type": "multiple_choice", "num_fewshot": "7"}
    assert "generation_args" not in result["generation_config"]
    assert patched["eval_library"]["additional_details"]["git_hash"] == "fixture-revision"
    assert events
    raw["model_info"]["additional_details"]["precision"] = "bfloat16"
    assert fix.apply(raw)[0] is None


def test_benchmark_page_defaults_need_no_other_model_and_row_values_win():
    fix = _ValsAiMaxTokens({"policy": "benchmark_page_default", "documented_statements": {
        "fixture-task": {"temperature": {"value": 0.25}, "max_tokens": {"value": 1234}}}})
    raw = record()
    raw["source_metadata"]["additional_details"] = {"benchmark_slug": "fixture-task"}
    raw["evaluation_results"][0]["generation_config"] = {"generation_args": {"temperature": 0.7}}
    patched, _, _ = fix.apply(raw)
    assert patched["evaluation_results"][0]["generation_config"] == {
        "generation_args": {"temperature": 0.7, "max_tokens": 1234},
        "additional_details": {"temperature_source": "row", "max_tokens_source": "page_methodology"}}


def test_inspect_counts_are_per_score_and_do_not_use_sample_ids():
    raw = record(library="inspect_ai")
    raw["evaluation_results"] = [
        {"source_data": {"source_type": "hf_dataset", "sample_ids": ["a", "b"], "samples_number": 2},
         "score_details": {"uncertainty": {"num_samples": 11}}},
        {"source_data": {"source_type": "other"}, "score_details": {"uncertainty": {"num_samples": 5}}},
    ]
    patched, _, _ = _PublishedMetadata().apply(raw)
    first, second = patched["evaluation_results"]
    assert first["source_data"]["samples_number"] == 11
    assert first["source_data"]["sample_ids"] == ["a", "b"]
    assert second["source_data"]["additional_details"]["samples_number"] == "5"


def test_inspect_plan_metadata_survives_generic_pointer_merge():
    raw = record(library="inspect_ai")
    raw["evaluation_results"][0]["generation_config"] = {"generation_args": {"eval_plan": {
        "steps": [json.dumps({"solver": "generate"})], "config": {"temperature": "0.2", "top_p": "0.9", "max_tokens": "321"}}}}
    patched, _, _ = apply_fixes(raw, "unseen-inspect-collection")
    generation = patched["evaluation_results"][0]["generation_config"]
    assert generation["generation_args"]["temperature"] == 0.2
    assert generation["generation_args"]["top_p"] == 0.9
    assert generation["additional_details"]["generation_args_source"] == "plan_config"
    assert generation["additional_details"]["terminal_solver"] == "generate"
    raw["eval_library"]["version"] += ",private_package:1.0"
    patched, _, _ = apply_fixes(raw, "unseen-inspect-collection")
    generation = patched["evaluation_results"][0]["generation_config"]
    assert "max_tokens" not in generation["generation_args"]
    assert generation["additional_details"]["plan_config_unverified_packages"] == "private_package"


def hle_record(temperature=0.0, note=None):
    raw = record(library="Scale SEAL HLE leaderboard", source="Scale SEAL Humanity's Last Exam Leaderboard")
    raw["evaluation_results"][0].update({
        "generation_config": {"generation_args": {"temperature": temperature, "max_attempts": 1}},
        "metric_config": {"llm_scoring": {"input_prompt": "Scale SEAL evaluates Humanity's Last Exam at temperature 0.0. Judge description."}}})
    if note:
        raw["model_info"]["additional_details"]["source_row_note"] = note
    return raw


def test_hle_only_corrects_identified_blanket_defaults_and_records_old_values():
    raw = hle_record(note="Sampled at temperature = 0.6, top_p = 0.8.")
    patched, events, _ = _PublishedMetadata().apply(raw)
    args = patched["evaluation_results"][0]["generation_config"]["generation_args"]
    assert args == {"temperature": 0.6, "top_p": 0.8}
    assert any(event.old == 0.0 and event.new == 0.6 for event in events)
    patched, _, _ = _PublishedMetadata().apply(hle_record(temperature=0.9))
    assert patched["evaluation_results"][0]["generation_config"]["generation_args"]["temperature"] == 0.9
    raw = hle_record()
    raw["source_metadata"]["source_name"] = "unrelated"
    assert _PublishedMetadata().apply(raw)[0] is None


def test_mmlu_pro_name_marker_corrects_only_obsolete_blanket_prompt_style():
    raw = record()
    raw["model_info"]["name"] = "Model (3-shot)"
    raw["evaluation_results"][0]["source_data"].update({"hf_repo": "TIGER-Lab/mmlu_pro_leaderboard_submission", "additional_details": {"prompt_style": "5-shot CoT"}})
    raw["evaluation_results"][0]["metric_config"] = {"additional_details": {"prompt_style": "explicit custom setup"}}
    patched, _, _ = _PublishedMetadata().apply(raw)
    assert patched["evaluation_results"][0]["source_data"]["additional_details"]["prompt_style"] == "3-shot"
    assert patched["evaluation_results"][0]["metric_config"]["additional_details"]["prompt_style"] == "explicit custom setup"


def test_malformed_metadata_containers_are_preserved():
    raw = record(library="inspect_ai")
    raw["evaluation_results"][0]["source_data"]["additional_details"] = "malformed"
    raw["evaluation_results"][0]["source_data"]["source_type"] = "other"
    raw["evaluation_results"][0]["score_details"] = {"uncertainty": {"num_samples": 5}}
    assert _PublishedMetadata().apply(raw)[0] is None


def test_vals_retained_row_metadata_is_used_without_a_documented_page():
    fix = _ValsAiMaxTokens({"policy": "benchmark_page_default", "documented_statements": {}})
    raw = record()
    result = raw["evaluation_results"][0]
    result["score_details"] = {"details": {"temperature": "0.8", "reasoning": "true", "harness": "agent setup"}}
    result["generation_config"] = {"additional_details": {"reasoning": "true"}}
    patched, _, _ = fix.apply(raw)
    generation = patched["evaluation_results"][0]["generation_config"]
    assert generation["generation_args"] == {
        "temperature": 0.8, "reasoning": True,
        "agentic_eval_config": {"additional_details": {"vals_harness": "agent setup"}}}
    assert generation["additional_details"] == {"temperature_source": "row"}


def test_vals_unusable_row_value_does_not_fall_back_to_page():
    fix = _ValsAiMaxTokens({"policy": "benchmark_page_default", "documented_statements": {
        "fixture-task": {"max_tokens": {"value": 30000}}}})
    raw = record()
    raw["source_metadata"]["additional_details"] = {"benchmark_slug": "fixture-task"}
    raw["evaluation_results"][0]["score_details"] = {"details": {"max_output_tokens": "unknown"}}
    assert fix.apply(raw)[0] is None


def test_lm_eval_counts_and_likelihood_metadata_follow_the_record():
    raw = record(library="lm_eval")
    raw["evaluation_results"][0].update({
        "score_details": {"uncertainty": {"num_samples": 19}},
        "generation_config": {"generation_args": {"temperature": 0.4, "max_tokens": 512},
                              "additional_details": {"output_type": "loglikelihood"}}})
    patched, _, _ = _PublishedMetadata().apply(raw)
    result = patched["evaluation_results"][0]
    assert result["source_data"]["samples_number"] == 19
    assert result["generation_config"]["generation_args"] == {}


def test_helm_split_requires_converter_provenance_and_deduplicates_ids():
    raw = record(library="helm")
    raw["source_metadata"]["source_type"] = "evaluation_run"
    raw["evaluation_results"][0].update({
        "source_data": {"source_type": "hf_dataset", "sample_ids": ["a", "a", "b"], "samples_number": 3},
        "score_details": {"details": {"split": "validation"}}})
    patched, _, _ = _PublishedMetadata().apply(raw)
    assert patched["evaluation_results"][0]["source_data"] == {
        "source_type": "hf_dataset", "sample_ids": ["a", "b"], "samples_number": 2, "hf_split": "val"}
    raw["source_metadata"]["source_type"] = "documentation"
    assert _PublishedMetadata().apply(raw)[0] is None


def test_zero_sample_counts_are_evidence_not_missing_values():
    for library in ("inspect_ai", "lm_eval"):
        for source_type in ("hf_dataset", "other"):
            raw = record(library=library)
            result = raw["evaluation_results"][0]
            result["source_data"] = {"source_type": source_type}
            if source_type == "hf_dataset":
                result["source_data"]["samples_number"] = 10
            else:
                result["source_data"]["additional_details"] = {"samples_number": "10"}
            result["score_details"] = {"uncertainty": {"num_samples": 0}}
            patched, _, _ = _PublishedMetadata().apply(raw)
            source = patched["evaluation_results"][0]["source_data"]
            assert (source["samples_number"] if source_type == "hf_dataset"
                    else source["additional_details"]["samples_number"]) == (0 if source_type == "hf_dataset" else "0")


def test_arc_rejected_join_metadata_is_counted_in_patch_accounting():
    raw = record(result={"evaluation_name": "v1_Public_Eval", "evaluation_result_id": "score", "metric_config": {"metric_id": "score"}})
    fix = _ArcAgiLoggedRequestArgs({"patches": [], "rejected_patches": [{
        "evaluation_id": "arbitrary-run", "evaluation_result_id": "score", "directory": "fixture-run", "fields": {},
        "metadata": {"generation_config/additional_details/request_args_unavailable": "no_results_file"}}]})
    patched, events, counters = fix.apply(raw)
    assert patched and len(events) == 1
    summary = fix.summarise(Counter({key.split(":", 1)[1]: value for key, value in counters.items()}))
    assert summary["patches"] == 1
    assert summary["unused"] == 0

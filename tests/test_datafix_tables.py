"""The vendored patch tables: shape, checksum, and headline counts.

These are reviewed data artifacts, so a change to any of them is a change
to what a bake produces. Pinning the checksums here makes that change show
up as a deliberate test edit rather than as a silent shift in the
warehouse; the counts alongside them say, in plain numbers, what the
tables claim. The fix whose evidence is in-record vendors no table, so
what is pinned for it is the rule it is closed under.
"""
from __future__ import annotations

from collections import Counter

from eval_card_backend.datafix import (
    arc_agi,
    hfopenllm_v2,
    inspect_plan_config,
    registry,
    vals_ai,
)
from eval_card_backend.datafix.core import fingerprint, load_table

VALS_TABLE_SHA = "9a7db160a1d790342b152368b28d59cbf38f463fffdee7af8594ed01f8adb889"
HARNESS_TABLE_SHA = "078fea2ac53b27c29a0929e0ada70c34c9b27f53d399b8cebc3d870757d8cfd0"
ARC_TABLE_SHA = "3a8361696b78989930897c47c293fd8ff11532317711ff2d6802cd544b23a29f"
PLAN_RULE_SHA = "5cdb3e44e3dd84b027b2b3814c3386898f39e7bfc71777c53c3cd2e3e4c613aa"


def test_the_run_settings_table_is_the_reviewed_one():
    table = load_table(vals_ai.TABLE_NAME)
    documented = table["documented_statements"]

    assert table["_table_sha256"] == VALS_TABLE_SHA
    assert table["fix_id"] == vals_ai.FIX_ID
    assert table["source"]["dataset"] == "evaleval/EEE_datastore"
    assert sorted(documented) == ["medcode", "medscribe", "mmmu", "sage"]
    # One sentence per page states both settings, so every page documents
    # the same pair and each entry cites the page it was read from.
    for slug, by_field in documented.items():
        assert sorted(by_field) == ["max_tokens", "temperature"], slug
        assert by_field["max_tokens"]["value"] == (8192 if slug == "mmmu" else 30000)
        assert by_field["temperature"]["value"] == (0.0 if slug == "mmmu" else 1.0)
        for statement in by_field.values():
            assert statement["url"] == f"https://www.vals.ai/benchmarks/{slug}"
            assert statement["quote"]
            assert statement["verified_on"] == "2026-10-21"
            assert len(statement["page_sha256"]) == 64


def test_page_defaults_are_independent_of_record_identity():
    table = load_table(vals_ai.TABLE_NAME)
    assert table["policy"] == "benchmark_page_default"
    assert "patches" not in table
    assert table["documented_statements"]["mmmu"]["max_tokens"]["value"] == 8192


def test_the_harness_table_is_the_reviewed_one():
    table = load_table(hfopenllm_v2.TABLE_NAME)

    assert table["_table_sha256"] == HARNESS_TABLE_SHA
    assert table["fix_id"] == hfopenllm_v2.FIX_ID
    assert table["source"]["dataset"] == "open-llm-leaderboard/results"
    assert sorted(table["defaults"]) == [
        "BBH", "GPQA", "IFEval", "MATH Level 5", "MMLU-PRO", "MUSR"
    ]


def test_the_harness_defaults_are_the_documented_settings():
    table = load_table(hfopenllm_v2.TABLE_NAME)

    assert table["defaults"]["IFEval"] == {
        "output_type": "generate_until",
        "temperature": 0.0,
        "max_tokens": 1280,
        "num_fewshot": 0,
        "do_sample": False,
    }
    assert table["defaults"]["MATH Level 5"] == {
        "output_type": "generate_until",
        "temperature": 0.0,
        "max_tokens": 1024,
        "num_fewshot": 4,
        "do_sample": False,
    }
    # Loglikelihood-scored benchmarks generate no text, so they carry no
    # decoding settings and the applier never reads them.
    for benchmark in ("BBH", "GPQA", "MUSR", "MMLU-PRO"):
        assert sorted(table["defaults"][benchmark]) == ["num_fewshot", "output_type"]
        assert table["defaults"][benchmark]["output_type"] == "multiple_choice"


def test_only_two_models_ran_math_with_a_shorter_cap():
    table = load_table(hfopenllm_v2.TABLE_NAME)

    assert sorted(table["overrides"]) == ["databricks/dbrx-base", "internlm/internlm2-7b"]
    for model, by_benchmark in table["overrides"].items():
        assert sorted(by_benchmark) == ["MATH Level 5"]
        assert by_benchmark["MATH Level 5"]["max_tokens"] == 512
    assert table["conflicts"] == {}


def test_the_verified_lists_are_sorted_and_free_of_duplicates():
    table = load_table(hfopenllm_v2.TABLE_NAME)

    for benchmark, models in table["verified"].items():
        assert models == sorted(models), benchmark
        assert len(set(models)) == len(models), benchmark
    # Case-only variants are distinct models and both have to survive.
    ifeval = set(table["verified"]["IFEval"])
    assert {"DeepMount00/Llama-3.1-8b-ITA", "DeepMount00/Llama-3.1-8b-Ita"} <= ifeval


# --- arc_agi_logged_request_args --------------------------------------------


def test_the_arc_table_is_the_reviewed_one():
    table = load_table(arc_agi.TABLE_NAME)

    assert table["_table_sha256"] == ARC_TABLE_SHA
    assert table["fix_id"] == arc_agi.FIX_ID
    assert sorted(table["source"]["datasets"]) == [
        "v1_Public_Eval", "v2_Public_Eval"
    ]
    for name, source in table["source"]["datasets"].items():
        assert source["dataset"].startswith("arcprize/")
        assert source["revision"] in table["source"]["revision"], name


def test_the_arc_table_carries_one_logged_cap_per_public_score():
    table = load_table(arc_agi.TABLE_NAME)
    by_name = Counter(p["evaluation_name"] for p in table["patches"])

    assert len(table["patches"]) == 52
    assert by_name == {"v1_Public_Eval": 28, "v2_Public_Eval": 24}
    assert {p["expected_old"] for p in table["patches"]} == {None}
    assert {p["cap_key"] for p in table["patches"]} <= {
        "max_tokens", "max_completion_tokens", "max_output_tokens"
    }
    keys = [(p["evaluation_id"], p["evaluation_result_id"]) for p in table["patches"]]
    assert len(set(keys)) == len(keys)


def test_every_arc_patch_names_the_run_it_was_read_from():
    table = load_table(arc_agi.TABLE_NAME)

    for patch in table["patches"]:
        source = table["source"]["datasets"][patch["evaluation_name"]]
        assert (patch["dataset"], patch["revision"]) == (
            source["dataset"], source["revision"]
        )
        assert patch["directory"]
        assert sorted(patch["fields"]) in (["max_tokens"], ["max_tokens", "temperature"])


def test_the_arc_caps_are_the_values_the_requests_carried():
    table = load_table(arc_agi.TABLE_NAME)

    caps = {p["fields"]["max_tokens"] for p in table["patches"]}
    assert all(isinstance(cap, int) and cap > 0 for cap in caps)
    # No run in the pinned datasets states the same cap under two names, so
    # each patched run has exactly one key recorded.
    by_directory: dict[tuple, set] = {}
    for patch in table["patches"]:
        key = (patch["evaluation_name"], patch["directory"])
        by_directory.setdefault(key, set()).add(
            (patch["cap_key"], patch["fields"]["max_tokens"])
        )
    assert all(len(values) == 1 for values in by_directory.values())


# --- inspect_plan_config ----------------------------------------------------


def test_the_in_record_rule_is_the_reviewed_one():
    """It vendors no table, so its allowlists are what the fingerprint has
    to follow instead."""
    fix = inspect_plan_config.build()

    assert fix.table_sha256 == PLAN_RULE_SHA
    assert all(v.startswith("inspect_ai:") for v in inspect_plan_config.VERSIONS)
    assert inspect_plan_config.TERMINAL_SOLVERS == (
        "generate", "multiple_choice", "react"
    )
    assert set(inspect_plan_config.TERMINAL_SOLVERS) <= inspect_plan_config.KNOWN_SOLVERS
    assert inspect_plan_config.OVERRIDE_KEYS == {
        "temperature", "max_tokens", "max_completion_tokens", "max_output_tokens", "top_p", "top_k"
    }


def test_the_overlay_fingerprint_follows_the_tables():
    assert fingerprint() == (
        "7bfb4efb80da4daf93213f69551280d745a45ad32863dfa34104c32415a42562"
    )


# --- what the registry can state without opening a table --------------------


def test_the_registry_states_each_table_s_own_checksum():
    """A run that fills nothing reports from the registry alone, so what it
    reports has to be what the vendored bytes say."""
    for module in (vals_ai, hfopenllm_v2, arc_agi):
        assert module.evidence_sha256() == load_table(module.TABLE_NAME)[
            "_table_sha256"
        ], module.FIX_ID
    assert inspect_plan_config.evidence_sha256() == PLAN_RULE_SHA


def test_the_registry_states_each_evidence_revision_the_tables_record():
    assert vals_ai.SOURCE_REVISION == load_table(vals_ai.TABLE_NAME)["source"][
        "revision"
    ]
    assert hfopenllm_v2.SOURCE_REVISION == load_table(hfopenllm_v2.TABLE_NAME)[
        "source"
    ]["revision"]
    assert arc_agi.SOURCE_REVISION == load_table(arc_agi.TABLE_NAME)["source"][
        "revision"
    ]
    assert inspect_plan_config.SOURCE_REVISION == (
        inspect_plan_config.CHECKED_AT_REVISION
    )
    assert {e.fix_id: e.source_revision for e in registry.entries()} == {
        vals_ai.FIX_ID: vals_ai.SOURCE_REVISION,
        hfopenllm_v2.FIX_ID: hfopenllm_v2.SOURCE_REVISION,
        inspect_plan_config.FIX_ID: inspect_plan_config.SOURCE_REVISION,
        arc_agi.FIX_ID: arc_agi.SOURCE_REVISION,
        "published_source_metadata": "source-record-rules-v1",
    }

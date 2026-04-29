"""Unit tests for ``scripts.parity``.

Each test mirrors a rule branch of the corresponding spec doc under
``general-eval-card/notes/transformations/``. Quirks documented in the
specs (TS-as-spec divergences, "current behavior wins") are encoded as
explicit cases — do not "fix" them without first updating the spec and
the migration plan.
"""
from __future__ import annotations

import math

import pytest

from scripts import parity


# ---------------------------------------------------------------------------
# 01 — Identity canonicalization
# ---------------------------------------------------------------------------


def test_identity_namespace_and_handle_split():
    """`claude-opus-4-5` collapses to `claude-opus-4.5` per the digit-dash-digit
    rule (Group D in spec 01)."""
    identity = parity.get_canonical_model_identity(
        {"id": "anthropic/claude-opus-4-5", "name": "Claude Opus 4.5"}
    )
    assert identity["namespace"] == "anthropic"
    assert identity["familyId"] == "anthropic/claude-opus-4.5"
    assert identity["familyName"] == "Claude Opus 4.5"
    assert identity["variantKey"] == "base"
    assert identity["variantLabel"] == "Current"


def test_identity_token_case_map():
    identity = parity.get_canonical_model_identity({"id": "openai/gpt-5"})
    assert identity["familyName"] == "GPT 5"
    identity = parity.get_canonical_model_identity({"id": "meta/llama-3"})
    assert identity["familyName"] == "Llama 3"


def test_identity_num_unit_token_uppercases_size_suffix():
    """Per spec 01 Group A — `70b` → `70B`, `1.5t` → `1.5T`. Lowercased
    parameter-count tokens leak into every familyName otherwise."""
    identity = parity.get_canonical_model_identity({"id": "meta/llama-3-70b-instruct"})
    assert identity["familyName"] == "Llama 3 70B Instruct"
    identity = parity.get_canonical_model_identity({"id": "meta/llama-3-8b"})
    assert identity["familyName"] == "Llama 3 8B"


def test_identity_pure_numeric_token_passthrough():
    """Pure-numeric tokens (`4`, `2.5`) survive title-casing unchanged."""
    identity = parity.get_canonical_model_identity({"id": "anthropic/claude-2.5"})
    # 2.5 is a pure-numeric token, claude is mapped — result: "Claude 2.5"
    assert identity["familyName"] == "Claude 2.5"


def test_identity_v_version_lowercases():
    identity = parity.get_canonical_model_identity({"id": "deepseek/deepseek-v3"})
    assert identity["familyName"] == "Deepseek v3"
    identity = parity.get_canonical_model_identity({"id": "deepseek/deepseek-V3"})
    assert identity["familyName"] == "Deepseek v3"


def test_identity_date_extraction_yyyymmdd_only():
    identity = parity.get_canonical_model_identity(
        {"id": "anthropic/claude-3-5-sonnet-20240620"}
    )
    assert identity["familySlug"] == "claude-3.5-sonnet"
    assert identity["versionDate"] == "2024-06-20"
    assert identity["variantKey"] == "20240620"
    assert identity["variantLabel"] == "2024-06-20"


def test_identity_date_with_qualifier():
    identity = parity.get_canonical_model_identity(
        {"id": "anthropic/claude-3-5-sonnet-20240620-thinking"}
    )
    assert identity["versionQualifier"] == "Thinking"
    assert identity["variantKey"] == "20240620-thinking"
    assert identity["variantLabel"] == "2024-06-20 · Thinking"


def test_identity_dashed_dates_do_not_match():
    """Per spec — `2025-12-11` is 10 chars + dashes, not 8 contiguous digits."""
    identity = parity.get_canonical_model_identity(
        {"id": "openai/gpt-5-2025-12-11-thinking-high"}
    )
    assert identity["variantKey"] == "base"
    assert identity["variantLabel"] == "Current"


def test_identity_route_id():
    assert (
        parity.get_model_family_route_id("openai/gpt-5") == "openai__gpt-5"
    )


# ---------------------------------------------------------------------------
# 02 — Setup-alias merging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qualifier, expected",
    [
        ("prompt", True),
        ("Prompt", True),
        ("fc", True),
        ("function-calling", True),
        ("FUNCTION_CALLING", True),
        ("thinking", True),
        ("thinking-1k", True),
        ("thinking-medium", True),
        ("thinking-32k", True),
        ("base", False),
        ("default", False),
        ("", False),
        (None, False),
    ],
)
def test_is_setup_alias_qualifier(qualifier, expected):
    assert parity.is_setup_alias_qualifier(qualifier) is expected


def test_normalize_variant_base_renames_to_default():
    assert parity.normalize_variant("base", "openai/gpt-5") == ("default", "Default")


def test_normalize_variant_passes_default_through():
    assert parity.normalize_variant("default", "openai/gpt-5") == ("default", "Default")


def test_normalize_variant_collapses_setup_alias_to_iso_date():
    key, label = parity.normalize_variant(
        "20240620-thinking", "anthropic/claude-3-5-sonnet"
    )
    assert key == "2024-06-20"
    assert label == "2024-06-20"


def test_merge_variants_dedups_and_aggregates():
    merged = parity.merge_variants(
        [
            {
                "variant_key": "default",
                "variant_label": "Default",
                "evaluation_count": 2,
                "raw_model_ids": ["openai/gpt-5-fc"],
                "last_updated": "2024-01-01T00:00:00Z",
            },
            {
                "variant_key": "default",
                "variant_label": "Default",
                "evaluation_count": 3,
                "raw_model_ids": ["openai/gpt-5-prompt"],
                "last_updated": "2024-02-01T00:00:00Z",
            },
        ]
    )
    assert len(merged) == 1
    assert merged[0]["evaluation_count"] == 5
    assert merged[0]["raw_model_ids"] == ["openai/gpt-5-fc", "openai/gpt-5-prompt"]
    assert merged[0]["last_updated"] == "2024-02-01T00:00:00Z"


# ---------------------------------------------------------------------------
# 03 — License normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "license_text, expected",
    [
        (None, ""),
        ("", ""),
        ("Not specified", ""),
        ("Creative Commons Attribution 4.0", "CC BY 4.0"),
        ("creative commons attribution 4.0 international", "CC BY 4.0"),
        ("Creative Commons Zero", "CC0"),
        ("Apache License 2.0", "Apache 2.0"),
        ("apache 2.0", "Apache 2.0"),
        ("MIT License", "MIT"),
        ("CC-BY-SA 4.0", "CC BY-SA"),
        ("Custom proprietary license terms apply.", "Custom proprietary lic…"),
        ("Custom shorter text", "Custom shorter text"),
        # apache-2.0 SPDX hyphen-lowercase does NOT match — falls to passthrough
        ("apache-2.0", "apache-2.0"),
    ],
)
def test_shorten_license(license_text, expected):
    assert parity.shorten_license(license_text) == expected


def test_shorten_license_truncation_uses_unicode_ellipsis():
    out = parity.shorten_license("a" * 25)
    assert out.endswith("…")
    assert len(out) == 23


# ---------------------------------------------------------------------------
# 04 — Dataset URL synthesis
# ---------------------------------------------------------------------------


def test_dataset_url_dataset_url_wins():
    assert (
        parity.synthesize_dataset_url({"dataset_url": "https://example.com/x"})
        == "https://example.com/x"
    )


def test_dataset_url_empty_string_returns_empty():
    """``dataset_url: ""`` is not nullish — returned verbatim."""
    assert parity.synthesize_dataset_url({"dataset_url": ""}) == ""


def test_dataset_url_url_list_takes_first():
    assert parity.synthesize_dataset_url(
        {"url": ["https://a.example", "https://b.example"]}
    ) == "https://a.example"


def test_dataset_url_url_string_passthrough():
    assert parity.synthesize_dataset_url({"url": "https://x.example"}) == "https://x.example"


def test_dataset_url_hf_repo_template():
    assert (
        parity.synthesize_dataset_url({"hf_repo": "openai/grade-school-math"})
        == "https://huggingface.co/datasets/openai/grade-school-math"
    )


def test_dataset_url_empty_hf_repo_falls_through_to_none():
    """Per spec — `hf_repo` branch uses truthiness so `""` is falsy."""
    assert parity.synthesize_dataset_url({"hf_repo": ""}) is None


def test_dataset_url_returns_none_for_empty_input():
    assert parity.synthesize_dataset_url({}) is None
    assert parity.synthesize_dataset_url(None) is None


# ---------------------------------------------------------------------------
# 05 — Slug candidates
# ---------------------------------------------------------------------------


def test_pipeline_slugify_basic():
    assert parity.pipeline_slugify("openai/gpt-5.2") == "openai_gpt-5.2"
    assert parity.pipeline_slugify("openai__gpt-5.2") == "openai__gpt-5.2"
    assert parity.pipeline_slugify("01-ai/Yi-1.5") == "01-ai_Yi-1.5"


def test_pipeline_slugify_unknown_fallback():
    assert parity.pipeline_slugify("") == "unknown"
    assert parity.pipeline_slugify("!!!") == "unknown"


def test_get_model_detail_slug_candidates_dedups():
    out = parity.get_model_detail_slug_candidates("openai/gpt-3.5")
    assert "openai__gpt-3.5" in out
    assert "openai__gpt-3-5" in out
    # Already lowercase, so case variants collapse
    assert len(out) >= 2


def test_get_developer_slug_candidates_compact_form():
    out = parity.get_developer_slug_candidates("01-ai")
    assert out == ["01-ai", "01ai"]


def test_get_developer_slug_candidates_underscore_and_hyphen():
    out = parity.get_developer_slug_candidates("Mistral AI")
    # Order: underscore_slug, lowercase_underscore, hyphenated, compact
    assert "Mistral_AI" in out
    assert "mistral_ai" in out
    assert "mistral-ai" in out
    assert "mistralai" in out


# ---------------------------------------------------------------------------
# 06 — Developer name canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("openai", "OpenAI"),
        ("OpenAI", "OpenAI"),
        ("  google  ", "Google"),
        ("mistralai", "Mistral AI"),
        ("deepseek", "DeepSeek"),
        ("deepseek-ai", "DeepSeek"),
        ("xai", "xAI"),
        ("x-ai", "xAI"),
        # title-case fallback
        ("jaspionjader", "Jaspionjader"),
        # mixed-case stays as-is
        ("prithivMLmods", "prithivMLmods"),
        # not in map, doesn't start with [a-z]
        ("01-ai", "01-ai"),
        ("", ""),
    ],
)
def test_normalize_developer_name(raw, expected):
    assert parity.normalize_developer_name(raw) == expected


def test_normalize_developer_name_leading_whitespace_no_titlecase():
    """Title-case test uses the *original* (untrimmed) value."""
    assert parity.normalize_developer_name("  jaspionjader  ") == "  jaspionjader  "


# ---------------------------------------------------------------------------
# 07 — Timestamp normalization
# ---------------------------------------------------------------------------


def test_normalize_eval_timestamp_unix_seconds_string():
    assert parity.normalize_eval_timestamp("1700000000") == 1700000000 * 1000


def test_normalize_eval_timestamp_iso_string():
    """ISO with `-` falls into Date.getTime()."""
    out = parity.normalize_eval_timestamp("2024-06-20T00:00:00Z")
    # 2024-06-20T00:00:00Z = 1718841600 unix seconds = 1718841600000 ms
    assert out == pytest.approx(1718841600 * 1000)


def test_normalize_eval_timestamp_returns_nan_on_garbage():
    out = parity.normalize_eval_timestamp("not a date")
    assert math.isnan(out)


def test_to_comparable_timestamp_parsefloat_extracts_leading_year():
    """TS bug: parseFloat extracts the leading numeric prefix, not the full ISO."""
    assert parity.to_comparable_timestamp("2026-04-13T12:34:56Z") == 2026.0


def test_to_comparable_timestamp_unix_seconds_no_multiplier():
    assert parity.to_comparable_timestamp("1774096306.427425") == pytest.approx(1774096306.427425)


def test_to_comparable_timestamp_empty_returns_neg_inf():
    assert parity.to_comparable_timestamp("") == float("-inf")
    assert parity.to_comparable_timestamp(None) == float("-inf")


def test_to_iso_8601_unix_seconds():
    """Mirrors JS `new Date(ms).toISOString()` — always 3-digit ms precision."""
    assert parity.to_iso_8601("1700000000") == "2023-11-14T22:13:20.000Z"
    assert parity.to_iso_8601("1700000000.5") == "2023-11-14T22:13:20.500Z"


def test_to_iso_8601_truncates_microseconds_does_not_round():
    """JS `toISOString()` truncates sub-millisecond precision — does NOT
    round half-to-even (which would produce `.147Z` from `.146500ms`,
    tripping a 1k-row divergence on the production corpus)."""
    # 1700000000.146500 sec → 22:13:20.146 (truncated), NOT .147
    assert parity.to_iso_8601("1700000000.1465") == "2023-11-14T22:13:20.146Z"
    # 1700000000.146999 sec → 22:13:20.146 (truncated), NOT .147
    assert parity.to_iso_8601("1700000000.146999") == "2023-11-14T22:13:20.146Z"


def test_to_iso_8601_iso_passthrough():
    """Inputs already in ISO form normalize to the millisecond-precision shape."""
    assert parity.to_iso_8601("2024-06-20T00:00:00Z") == "2024-06-20T00:00:00.000Z"
    assert parity.to_iso_8601("2024-06-20T00:00:00.123Z") == "2024-06-20T00:00:00.123Z"


def test_to_iso_8601_invalid_returns_none():
    assert parity.to_iso_8601("not a date") is None
    assert parity.to_iso_8601(None) is None


# ---------------------------------------------------------------------------
# 08 — Benchmark display names
# ---------------------------------------------------------------------------


def test_benchmark_display_name_map_hit():
    assert parity.get_benchmark_display_name("helm_lite") == "HELM Lite"
    assert parity.get_benchmark_display_name("HELM-Lite") == "HELM Lite"
    assert parity.get_benchmark_display_name("helm.lite") == "HELM Lite"


def test_benchmark_display_name_humanize_fallback():
    """humanize_token only first-char-caps each part. NOT acronym-aware."""
    assert parity.get_benchmark_display_name("mmlu_pro") == "Mmlu Pro"
    # MMLU-PRO splits on `-`, returns first-char-cap → loses dash and case
    assert parity.get_benchmark_display_name("MMLU-PRO") == "MMLU PRO"


def test_benchmark_display_name_underscore_runs_fall_to_humanize():
    """TS normalize regex is `[-.\\s]+` only. Underscores aren't collapsed,
    so `___helm___lite___` → `helm___lite` (no map hit) → humanize_token of
    the raw input → `Helm Lite`."""
    assert parity.get_benchmark_display_name("___helm___lite___") == "Helm Lite"


def test_benchmark_display_name_dash_collapses_to_map_hit():
    assert parity.get_benchmark_display_name("HELM-Lite") == "HELM Lite"
    assert parity.get_benchmark_display_name("HELM Lite") == "HELM Lite"


def test_benchmark_display_name_empty():
    assert parity.get_benchmark_display_name("") == ""
    assert parity.get_benchmark_display_name(None) == ""


# ---------------------------------------------------------------------------
# 09 — Metric display name expansion
# ---------------------------------------------------------------------------


def test_get_evaluation_display_name_generic_expansion():
    out = parity.get_evaluation_display_name(
        {"benchmark": "MMLU"}, {"evaluation_name": "Accuracy"}
    )
    assert out == "MMLU - Accuracy"


def test_get_evaluation_display_name_passthrough():
    out = parity.get_evaluation_display_name(
        {"benchmark": "MMLU"}, {"evaluation_name": "BLEU"}
    )
    assert out == "BLEU"


def test_prefers_benchmark_name_accuracy_on_prefix():
    assert parity.prefers_benchmark_name(
        {"benchmark": "MMLU", "evaluation_name": "accuracy on math"}
    )


def test_prefers_benchmark_name_accuracy_onset_does_not_match():
    """`startsWith("accuracy on ")` requires literal trailing space."""
    assert not parity.prefers_benchmark_name(
        {"benchmark": "MMLU", "evaluation_name": "accuracy onset"}
    )


def test_prefers_benchmark_name_for_scorer_substring():
    assert parity.prefers_benchmark_name(
        {"benchmark": "MMLU", "evaluation_name": "Quality for scorer GPT-4"}
    )


def test_prefers_benchmark_name_underscore_strict():
    """`model_graded` is underscore literal; spaces don't count."""
    assert not parity.prefers_benchmark_name(
        {"benchmark": "MMLU", "evaluation_name": "Model Graded"}
    )
    assert parity.prefers_benchmark_name(
        {"benchmark": "MMLU", "evaluation_name": "model_graded eval"}
    )


# ---------------------------------------------------------------------------
# 10 — Params parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (7.0, 7.0),
        (7, 7.0),
        ("7B", 7.0),
        ("7 billion", 7.0),
        ("1.5B", 1.5),
        ("405b", 405.0),
        ("1.2T", 1200.0),
        ("70 mn", 0.07),
        ("123 thousand", 0.000123),
        ("0", None),
        ("-5B", 5.0),  # leading minus dropped by regex
        ("", None),
        (None, None),
        (True, None),  # bool rejected
    ],
)
def test_parse_params_billions_variant_a(value, expected):
    out = parity.parse_params_billions(value)
    if expected is None:
        assert out is None
    else:
        assert out == pytest.approx(expected)


def test_parse_params_billions_from_text_allows_zero():
    """Variant B differs from A — `"0B"` is allowed (no `<= 0` check)."""
    assert parity.parse_params_billions_from_text("0B") == 0.0


def test_parse_params_billions_from_text_rejects_non_string():
    assert parity.parse_params_billions_from_text(7.0) is None


def test_parse_params_billions_from_model_name_evaldetail_last_wins():
    """Variant C — last token wins; reproduces context-window quirk."""
    assert parity.parse_params_billions_from_model_name_evaldetail("Llama-3-8B-Instruct-8K") == pytest.approx(0.000008)
    assert parity.parse_params_billions_from_model_name_evaldetail("Llama-3-8B-70B-Instruct") == 70.0


def test_parse_params_billions_from_model_name_compare_rejects_k_t():
    """Variant E — units {b, m} only."""
    assert parity.parse_params_billions_from_model_name_compare("Llama-3-8B-8K") == 8.0
    assert parity.parse_params_billions_from_model_name_compare("Llama-3-8T") is None


def test_parse_params_billions_from_name_id_concat_first_wins():
    """Variant F — first match, `b|B` only."""
    assert parity.parse_params_billions_from_name_id_concat("Llama-3-8B-70B-Instruct") == 8.0


def test_get_params_billions_from_model_info_orchestrator():
    assert parity.get_params_billions_from_model_info(
        {"additional_details": {"params_billions": 70.0}}
    ) == 70.0
    assert parity.get_params_billions_from_model_info(
        {"additional_details": {"params_billions": "7B"}}
    ) == 7.0
    assert parity.get_params_billions_from_model_info(
        {"parameter_count": "1.5B"}
    ) == 1.5
    assert parity.get_params_billions_from_model_info(
        {"name": "Llama-3-8B-Instruct"}
    ) == 8.0


def test_get_params_billions_orchestrator_falls_through_on_unparseable():
    """Per TS `??`-chain semantics — an unparseable string in
    `additional_details` falls through to `model_info.parameter_count`
    and finally to the model name. Earlier impl returned None as soon
    as the first key resolved to garbage, skipping all fallbacks."""
    out = parity.get_params_billions_from_model_info(
        {
            "additional_details": {"params_billions": "garbage"},
            "parameter_count": "7B",
        }
    )
    assert out == 7.0


def test_get_params_billions_orchestrator_falls_through_to_name():
    """When everything upstream is unparseable, fall through to Variant C
    on `model_info.name`."""
    out = parity.get_params_billions_from_model_info(
        {
            "additional_details": {"params_billions": "not a number"},
            "name": "Llama-3-70B-Instruct",
        }
    )
    assert out == 70.0


def test_get_params_billions_orchestrator_picks_first_non_null():
    """`??`-chain picks the first NON-null key. `parameter_count: "7B"`
    is shadowed by `params_billions: "garbage"` — TS picks `garbage` first,
    then falls through. NOT silently picking `parameter_count`."""
    out = parity.get_params_billions_from_model_info(
        {
            "additional_details": {
                "params_billions": "garbage",
                "parameter_count": "7B",
            },
        }
    )
    # Falls through to `model_info.parameter_count` (NOT
    # `additional_details.parameter_count`) per TS behavior.
    assert out is None


# ---------------------------------------------------------------------------
# 11 — Benchmark-card attachment
# ---------------------------------------------------------------------------


def test_normalize_benchmark_key_attach_strips_prefix():
    assert parity.normalize_benchmark_key_attach("hfopenllm_v2/mmlu") == "mmlu"
    assert parity.normalize_benchmark_key_attach("GPQA / Diamond") == "diamond"
    assert parity.normalize_benchmark_key_attach("HELM-Lite") == "helm lite"


def test_normalize_benchmark_key_attach_empty():
    assert parity.normalize_benchmark_key_attach("") == ""
    assert parity.normalize_benchmark_key_attach(None) == ""


def test_candidate_benchmark_keys_attach_alphabet_only_form():
    keys = parity.candidate_benchmark_keys_attach("HELM-Lite")
    assert "helm lite" in keys
    assert "helmlite" in keys
    assert "helm-lite" in keys


def test_candidate_benchmark_keys_attach_empty_returns_empty_string():
    """Per spec Group B — `""` MUST return `[""]` so the eventual
    `card_map.get("")` lookup runs."""
    assert parity.candidate_benchmark_keys_attach("") == [""]
    assert parity.candidate_benchmark_keys_attach(None) == [""]


def test_build_benchmark_card_map_first_write_wins():
    cards = {
        "first": {"benchmark_details": {"name": "HELM"}, "marker": "first"},
        "second": {"benchmark_details": {"name": "HELM"}, "marker": "second"},
    }
    card_map = parity.build_benchmark_card_map(cards)
    assert card_map["helm"]["marker"] == "first"


def test_attach_benchmark_card_to_summary_default_only_guard():
    summary = {"benchmark_card": {"existing": True}, "evaluation_name": "HELM"}
    card_map = {"helm": {"new": True}}
    out = parity.attach_benchmark_card_to_summary(summary, card_map)
    assert out["benchmark_card"] == {"existing": True}


def test_attach_benchmark_card_summary_vs_list_order_diverges():
    """Summary path candidate order [name, name, key]. List path [name, key, name].

    The card is attached as a SINGLE field (`benchmark_card`) — TS contract
    `{...summary, benchmark_card: card}`, never spread across the summary.
    """
    summary = {"evaluation_name": "ZZZ", "composite_benchmark_name": "AAA", "composite_benchmark_key": "BBB"}
    card_map = {
        "aaa": {"benchmark_details": {"name": "AAA"}, "marker": "aaa-card"},
        "bbb": {"benchmark_details": {"name": "BBB"}, "marker": "bbb-card"},
    }
    summary_attached = parity.attach_benchmark_card_to_summary(summary, card_map)
    list_attached = parity.attach_benchmark_card_to_list_item(summary, card_map)
    assert summary_attached["benchmark_card"]["marker"] == "aaa-card"
    assert list_attached["benchmark_card"]["marker"] == "bbb-card"
    # Summary must NOT have card fields spread across its top level.
    assert "marker" not in summary_attached
    assert "marker" not in list_attached
    # And the original summary is untouched.
    assert summary.get("benchmark_card") is None


def test_attach_benchmark_card_does_not_overwrite_summary_keys():
    """Regression: an earlier impl `dict.update(card)`'d the entire card
    over the summary, clobbering same-named keys (e.g. `category`).
    The TS contract is to only add `benchmark_card`."""
    summary = {
        "evaluation_name": "Foo",
        "category": "Reasoning",  # would be clobbered by dict.update(card)
        "model_results": ["should-survive"],
    }
    card_map = {
        "foo": {
            "benchmark_details": {"name": "Foo"},
            "category": "WRONG",
            "model_results": ["card-junk"],
        }
    }
    out = parity.attach_benchmark_card_to_summary(summary, card_map)
    assert out["category"] == "Reasoning"
    assert out["model_results"] == ["should-survive"]
    assert out["benchmark_card"]["category"] == "WRONG"


# ---------------------------------------------------------------------------
# 12 — Instance-level data
# ---------------------------------------------------------------------------


def test_parse_instance_level_data_canonical_shape():
    raw = {
        "instance_examples": [
            {
                "sample_id": "s-1",
                "input": {"raw": "What is 2+2?", "reference": ["4"]},
                "answer_attribution": [{"extracted_value": "4"}],
                "evaluation": {"is_correct": True},
            }
        ]
    }
    out = parity.parse_instance_level_data(raw)
    assert len(out) == 1
    assert out[0]["sample_id"] == "s-1"
    assert out[0]["input"] == "What is 2+2?"
    assert out[0]["ground_truth"] == "4"
    assert out[0]["response"] == "4"
    assert out[0]["is_correct"] is True


def test_parse_instance_level_data_messages_fallback():
    raw = {
        "instance_examples": [
            {
                "sample_id": "s-2",
                "input": "Hi",
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ],
                "metrics": {"exact_match": 1},
            }
        ]
    }
    out = parity.parse_instance_level_data(raw)
    assert out[0]["response"] == "pong"
    assert out[0]["is_correct"] is True


def test_parse_instance_level_data_returns_empty_for_garbage():
    assert parity.parse_instance_level_data(None) == []
    assert parity.parse_instance_level_data("not a dict") == []
    assert parity.parse_instance_level_data({"foo": "bar"}) == []


def test_parse_instance_level_data_array_input():
    out = parity.parse_instance_level_data([
        {"sample_id": "x", "input": "q", "output": "a"}
    ])
    assert out[0]["sample_id"] == "x"
    assert out[0]["input"] == "q"
    assert out[0]["response"] == "a"


def test_parse_instance_level_data_index_fallback_for_sample_id():
    out = parity.parse_instance_level_data([
        {"input": "first"},
        {"input": "second"},
    ])
    assert out[0]["sample_id"] == "0"
    assert out[1]["sample_id"] == "1"


def test_parse_instance_level_data_metadata_later_source_wins():
    """TS `Object.assign` order: evaluation → performance → metadata → metrics.
    Later sources OVERWRITE earlier ones on key collision."""
    out = parity.parse_instance_level_data([
        {
            "sample_id": "x",
            "input": "q",
            "evaluation": {"score": 0.5, "is_correct": True},
            "metrics": {"score": 0.9, "exact_match": 1},
        }
    ])
    assert out[0]["metadata"]["score"] == 0.9
    assert out[0]["metadata"]["is_correct"] is True


# ---------------------------------------------------------------------------
# Category inference + mapping
# ---------------------------------------------------------------------------


def test_infer_category_from_benchmark_safety_tokens():
    assert parity.infer_category_from_benchmark("helm_safety") == "Safety"
    assert parity.infer_category_from_benchmark("Truthful QA") == "Safety"
    assert parity.infer_category_from_benchmark("simplesafetytests") == "Safety"


def test_infer_category_from_benchmark_agentic_tokens():
    assert parity.infer_category_from_benchmark("swe-bench") == "Agentic"
    assert parity.infer_category_from_benchmark("appworld") == "Agentic"


def test_infer_category_from_benchmark_reasoning_tokens():
    assert parity.infer_category_from_benchmark("gsm8k") == "Reasoning"
    assert parity.infer_category_from_benchmark("MATH") == "Reasoning"
    assert parity.infer_category_from_benchmark("HumanEval") == "Reasoning"


def test_infer_category_from_benchmark_knowledge_tokens():
    assert parity.infer_category_from_benchmark("mmlu_pro") == "Knowledge"
    assert parity.infer_category_from_benchmark("medqa") == "Knowledge"


def test_infer_category_from_benchmark_default_general():
    assert parity.infer_category_from_benchmark("randomthing") == "General"
    assert parity.infer_category_from_benchmark("") == "General"
    assert parity.infer_category_from_benchmark(None) == "General"


def test_map_hf_categories_specific_keys():
    """Mirrors `lib/hf-data.ts:1437 mapHFCategories` exactly."""
    assert parity.map_hf_categories(["agentic"]) == ["Agentic"]
    assert parity.map_hf_categories(["reasoning"]) == ["Reasoning"]
    assert parity.map_hf_categories(["safety"]) == ["Safety"]
    assert parity.map_hf_categories(["knowledge"]) == ["Knowledge"]


def test_map_hf_categories_collapse_to_general():
    """`other`, `coding`, `instruction_following`, `language_understanding`
    all collapse to "General" per `PIPELINE_CATEGORY_MAP` in
    `lib/hf-data.ts:1425`. The TS does NOT run the regex fallback when
    pipeline emits `other` — that was a plan-level misreading; the actual
    code path maps `other → General` directly."""
    assert parity.map_hf_categories(["other"]) == ["General"]
    assert parity.map_hf_categories(["coding"]) == ["General"]
    assert parity.map_hf_categories(["instruction_following"]) == ["General"]
    assert parity.map_hf_categories(["language_understanding"]) == ["General"]


def test_map_hf_categories_dedup_and_default():
    """Empty input returns ["General"] per TS contract."""
    assert parity.map_hf_categories([]) == ["General"]
    assert parity.map_hf_categories([None, ""]) == ["General"]
    assert parity.map_hf_categories(["agentic", "reasoning", "agentic"]) == [
        "Agentic",
        "Reasoning",
    ]
    assert parity.map_hf_categories(["unknown_thing"]) == ["General"]


def test_map_hf_category_single_default():
    assert parity.map_hf_category_single(None) == "General"
    assert parity.map_hf_category_single("") == "General"
    assert parity.map_hf_category_single("safety") == "Safety"
    assert parity.map_hf_category_single("Other") == "General"


# ---------------------------------------------------------------------------
# Reshape — score normalization
# ---------------------------------------------------------------------------


def test_normalize_summary_score_default_range():
    """Default min=0, max=1 → returns score unchanged."""
    assert parity.normalize_summary_score({}, 0.7) == 0.7


def test_normalize_summary_score_explicit_range():
    assert parity.normalize_summary_score(
        {"min_score": 0, "max_score": 100}, 50
    ) == 0.5


def test_normalize_summary_score_zero_range_returns_score():
    """TS contract `lib/model-data.ts:77` — when range is non-positive,
    returns `score` unchanged (NOT zero, despite an out-of-date spec note)."""
    assert parity.normalize_summary_score({"min_score": 5, "max_score": 5}, 5) == 5
    assert parity.normalize_summary_score({"min_score": 1, "max_score": 0}, 7) == 7


# ---------------------------------------------------------------------------
# Reshape 14 — score summary stats
# ---------------------------------------------------------------------------


def test_compute_score_summary_stats_basic():
    rows = [
        {
            "score": 0.8,
            "model_name": "alpha",
            "source_metadata": {"evaluator_relationship": "third_party", "source_type": "lb"},
            "retrieved_timestamp": "1700000000",
        },
        {
            "score": 0.6,
            "model_name": "beta",
            "source_metadata": {"evaluator_relationship": "first_party", "source_type": "lb"},
            "retrieved_timestamp": "1800000000",
        },
    ]
    out = parity.compute_score_summary_stats(rows, {"min_score": 0, "max_score": 1}, False, "Bench")
    assert out["models_count"] == 2
    assert out["avg_score"] == pytest.approx(0.7)
    assert out["best_model"]["model_name"] == "alpha"
    assert out["worst_model"]["model_name"] == "beta"
    assert out["evaluator_names"] == []  # active path always empty
    assert out["third_party_ratio"] == 0.5


def test_compute_score_summary_stats_empty_short_circuit():
    out = parity.compute_score_summary_stats([], {}, False, "Bench Display")
    assert out["models_count"] == 0
    assert out["latest_source_name"] == "Bench Display"


# ---------------------------------------------------------------------------
# Reshape 16 — per-category counts
# ---------------------------------------------------------------------------


def test_category_stats_fake_distribution_last_wins_remainder():
    """For total=13, len=4 → {3, 3, 3, 4} per spec."""
    out = parity.category_stats_fake_distribution(13, ["a", "b", "c", "d"])
    assert list(out.values()) == [3, 3, 3, 4]


def test_category_stats_fake_distribution_underflow():
    """total=2, categories=4 → {1, 1, 0, 0}."""
    out = parity.category_stats_fake_distribution(2, ["a", "b", "c", "d"])
    assert out == {"a": 1, "b": 1, "c": 0, "d": 0}


def test_category_stats_fake_distribution_zero_categories():
    assert parity.category_stats_fake_distribution(10, []) == {}


def test_category_stats_distinct_count_by_benchmark():
    summary = {
        "math": [
            {"benchmark": "GSM8K", "evaluation_results": [{"evaluation_name": "Accuracy"}]},
            {"benchmark": "MATH", "evaluation_results": [{"evaluation_name": "Accuracy"}]},
        ],
        "code": [
            {"benchmark": "HumanEval", "evaluation_results": [{"evaluation_name": "pass@1"}]},
        ],
    }
    out = parity.category_stats_distinct_count(summary, ["math", "code"])
    assert out == {"math": 2, "code": 1}

import pytest

from eval_card_backend.canonicalise.benchpress_relationship import (
    benchpress_relationship,
    publisher_key,
)


@pytest.mark.parametrize(
    "url,key",
    [
        ("https://huggingface.co/deepseek-ai/DeepSeek-R1-0528", "huggingface.co/deepseek-ai"),
        ("https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf", "github.com/moonshotai"),
        ("https://arxiv.org/abs/2501.12948", "arxiv.org/2501.12948"),
        ("https://arxiv.org/pdf/2606.15079v1", "arxiv.org/2606.15079"),
        ("https://deepmind.google/models/model-cards/gemini-3-5-flash/", "deepmind.google"),
        ("", ""),
        (None, ""),
    ],
)
def test_publisher_key(url, key):
    assert publisher_key(url) == key


@pytest.mark.parametrize(
    "developer,url,expected",
    [
        # the developer's own document on a host that does not carry its name
        ("DeepSeek", "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528", "first_party"),
        ("Google", "https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-0-Flash-Model-Card.pdf", "first_party"),
        ("xAI / SpaceXAI", "https://media.x.ai/v1/website/card-4p6-4cd2dc57.pdf", "first_party"),
        # a competitor's score in a developer's document
        ("Anthropic", "https://lf3-static.bytednsdoc.com/obj/eden-cn/seed2/Seed2.0%20Model%20Card.pdf", "third_party"),
        ("OpenAI", "https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf", "third_party"),
        # an independent leaderboard
        ("Anthropic", "https://matharena.ai/?comp=brumo--brumo_2025", "third_party"),
        # unknown publisher or developer, or no citation: undecided
        ("OpenAI", "https://example.org/report.pdf", None),
        ("Some Lab", "https://openai.com/index/gpt-5-6/", None),
        ("OpenAI", "", None),
        (None, None, None),
    ],
)
def test_benchpress_relationship(developer, url, expected):
    assert benchpress_relationship(developer, url) == expected

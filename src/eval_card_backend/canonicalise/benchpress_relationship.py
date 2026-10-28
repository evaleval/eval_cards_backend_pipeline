"""Evaluator relationship for BenchPress scores the adapter labels `other`.

The BenchPress adapter labels a score `other` when it cannot tell who
published the cited document: a provider's own page on a host that does not
spell the provider's name (huggingface.co/deepseek-ai, deepmind.google), or a
competitor's score tabulated in a provider's document (a Gemini score in
Qwen's model card). The cited document's publisher settles it: the result is
first party when the publisher is the model's developer and third party
otherwise. A score with no citation URL keeps `other`.

Remove when the upstream BenchPress adapter derives the relationship from the
publisher of the cited document.
"""

from __future__ import annotations

from urllib.parse import urlparse

# model_info.developer as BenchPress writes it -> organization.
DEVELOPER_ORG: dict[str, str] = {
    "Alibaba": "alibaba",
    "Allen AI": "allen-ai",
    "Amazon": "amazon",
    "Anthropic": "anthropic",
    "ByteDance": "bytedance",
    "Cognition": "cognition",
    "Cohere": "cohere",
    "DeepSeek": "deepseek",
    "Google": "google",
    "Hugging Face": "hugging-face",
    "IBM": "ibm",
    "Inception Labs": "inception-labs",
    "InclusionAI": "inclusionai",
    "inclusionAI": "inclusionai",
    "LG AI Research": "lg-ai-research",
    "Liquid AI": "liquid-ai",
    "Meituan": "meituan",
    "Meituan LongCat": "meituan",
    "Meta": "meta",
    "Microsoft": "microsoft",
    "Microsoft AI": "microsoft",
    "MiniMax": "minimax",
    "Mistral": "mistral",
    "Moonshot AI": "moonshot-ai",
    "NVIDIA": "nvidia",
    "Open Thoughts": "open-thoughts",
    "OpenAI": "openai",
    "Shanghai AI Lab": "shanghai-ai-lab",
    "Shanghai AI Laboratory": "shanghai-ai-lab",
    "StepFun": "stepfun",
    "TII": "tii",
    "Tencent": "tencent",
    "Xiaomi": "xiaomi",
    "Z.ai": "zhipu",
    "Zhipu AI": "zhipu",
    "xAI": "xai",
    "xAI / SpaceXAI": "xai",
}

# Publisher key (see publisher_key) -> organization that published the
# cited document.
PUBLISHER_ORG: dict[str, str] = {
    "lf3-static.bytednsdoc.com": "bytedance",
    "seed.bytedance.com": "bytedance",
    "arxiv.org/2504.13914": "bytedance",
    "github.com/moonshotai": "moonshot-ai",
    "huggingface.co/moonshotai": "moonshot-ai",
    "arxiv.org/2507.20534": "moonshot-ai",
    "huggingface.co/tencent": "tencent",
    "huggingface.co/deepseek-ai": "deepseek",
    "github.com/deepseek-ai": "deepseek",
    "arxiv.org/2412.19437": "deepseek",
    "arxiv.org/2501.12948": "deepseek",
    "www.liquid.ai": "liquid-ai",
    "huggingface.co/liquidai": "liquid-ai",
    "aypchzzf9pftwuto.public.blob.vercel-storage.com": "liquid-ai",
    "z.ai": "zhipu",
    "cdn.bigmodel.cn": "zhipu",
    "media.x.ai": "xai",
    "x.ai": "xai",
    "huggingface.co/internlm": "shanghai-ai-lab",
    "github.com/internlm": "shanghai-ai-lab",
    "research.nvidia.com": "nvidia",
    "huggingface.co/nvidia": "nvidia",
    "deepmind.google": "google",
    "storage.googleapis.com": "google",
    "huggingface.co/google": "google",
    "arxiv.org/2503.19786": "google",
    "arxiv.org/2607.02770": "google",
    "arxiv.org/2608.00146": "google",
    "huggingface.co/minimaxai": "minimax",
    "huggingface.co/xiaomimimo": "xiaomi",
    "mimo.xiaomi.com": "xiaomi",
    "huggingface.co/mistralai": "mistral",
    "mistral.ai": "mistral",
    "cohere.com": "cohere",
    "huggingface.co/qwen": "alibaba",
    "qwenlm.github.io": "alibaba",
    "qwen.ai": "alibaba",
    "www.alibabacloud.com": "alibaba",
    "arxiv.org/2505.09388": "alibaba",
    "openai.com": "openai",
    "github.com/openai": "openai",
    "arxiv.org/2508.10925": "openai",
    "ai.meta.com": "meta",
    "research.meta.ai": "meta",
    "huggingface.co/meta-llama": "meta",
    "microsoft.ai": "microsoft",
    "huggingface.co/microsoft": "microsoft",
    "arxiv.org/2504.21318": "microsoft",
    "arxiv.org/2412.08905": "microsoft",
    "longcat.chat": "meituan",
    "arxiv.org/2608.01662": "meituan",
    "www.anthropic.com": "anthropic",
    "assets.anthropic.com": "anthropic",
    "static.stepfun.com": "stepfun",
    "raw.githubusercontent.com/stepfun-ai": "stepfun",
    "ernie.baidu.com": "baidu",
    "huggingface.co/inclusionai": "inclusionai",
    "raw.githubusercontent.com/inclusionai": "inclusionai",
    "intranetproxy.alipay.com": "inclusionai",
    "mdn.alipayobjects.com": "inclusionai",
    "arxiv.org/2606.15079": "inclusionai",
    "arxiv.org/2501.00656": "allen-ai",
    "arxiv.org/2507.11407": "lg-ai-research",
    "arxiv.org/2506.12103": "amazon",
    "huggingface.co/tiiuae": "tii",
    "cursor.com": "cursor",
    "matharena.ai": "matharena",
    "scale.com": "scale-ai",
    "www.tbench.ai": "terminal-bench",
    "raw.githubusercontent.com/claw-eval": "claw-eval",
}

_ORG_PATH_HOSTS = frozenset({"huggingface.co", "github.com", "raw.githubusercontent.com"})


def publisher_key(url: str | None) -> str:
    """The citation's host; the host plus the org path segment for hosts that
    serve many organizations; the paper id for arXiv."""
    parsed = urlparse(url or "")
    host = parsed.netloc.lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host in _ORG_PATH_HOSTS and parts:
        return f"{host}/{parts[0].lower()}"
    if host == "arxiv.org" and parts:
        return f"arxiv.org/{parts[-1].split('v')[0]}"
    return host


def benchpress_relationship(developer: str | None, reference_url: str | None) -> str | None:
    """`first_party` or `third_party` for a BenchPress `other` score, or None
    when the publisher or the developer is not in the tables."""
    publisher = PUBLISHER_ORG.get(publisher_key(reference_url))
    developer_org = DEVELOPER_ORG.get((developer or "").strip())
    if publisher is None or developer_org is None:
        return None
    return "first_party" if publisher == developer_org else "third_party"

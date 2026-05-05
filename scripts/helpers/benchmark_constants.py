from typing import Final

COMMON_LANGUAGE_SUBSET_KEYS: Final[set[str]] = {
    "albanian",
    "arabic",
    "ar",
    "bengali",
    "bn",
    "burmese",
    "chinese",
    "cy",
    "english",
    "en",
    "french",
    "fr",
    "german",
    "de",
    "hindi",
    "hi",
    "id",
    "indonesian",
    "italian",
    "it",
    "japanese",
    "ja",
    "korean",
    "ko",
    "my",
    "portuguese",
    "pt",
    "spanish",
    "es",
    "sq",
    "sw",
    "swahili",
    "welsh",
    "yo",
    "yoruba",
    "zh",
}

# Tokens that lose distinguishing punctuation under benchmark-key normalization.
LANGUAGE_TOKEN_ALIASES: Final[dict[str, str]] = {
    "c++": "cpp",
    "c#": "csharp",
    "f#": "fsharp",
    ".net": "dotnet",
    "objective-c": "objective_c",
}

# Leaf-key values that indicate a score is a summary across all sub-benchmarks
# rather than an independent benchmark of its own.
SUMMARY_SCORE_LEAF_KEYS: Final[set[str]] = {"overall", "aggregate", "total", "all"}

PREFERRED_BENCHMARK_DISPLAY_NAMES: Final[dict[str, str]] = {
    "ace": "ACE",
    "apex": "APEX",
    "apex_agents": "APEX Agents",
    "apex_v1": "APEX v1",
    # SWE-PolyBench / Multi-SWE-bench: hardcoded because removing them
    # caused regressions in three separate code paths (family_name,
    # parent_name, AND aggregator-row dataset_name normalization that
    # happens to land on the same normalized key). The fully-clean
    # refactor would route family/parent name derivation through the
    # registry's canonical display_name when canonical_benchmark_id
    # resolves — but that requires per-row registry calls during
    # normalization, which is bigger architectural work. Keeping these
    # explicit until that lands; documented in CLAUDE.md.
    "swe_polybench": "SWE-PolyBench",
    "multi_swe_bench": "Multi-SWE-bench",
}

# Domain keywords → high-level category mapping.
# Used when ABC cards include `benchmark_details.domains`.
DOMAIN_CATEGORY_MAP: Final[dict[str, str]] = {
    "safety": "safety",
    "toxic": "safety",
    "bias": "safety",
    "fairness": "safety",
    "harmful": "safety",
    "ethics": "safety",
    "math": "reasoning",
    "mathematics": "reasoning",
    "reasoning": "reasoning",
    "commonsense reasoning": "reasoning",
    "planning": "reasoning",
    "logic": "reasoning",
    "olympiad": "reasoning",
    "coding": "coding",
    "code generation": "coding",
    "software engineering": "coding",
    "programming": "coding",
    "instruction following": "instruction_following",
    "summarization": "language_understanding",
    "reading comprehension": "language_understanding",
    "natural language understanding": "language_understanding",
    "natural language inference": "language_understanding",
    "question answering": "knowledge",
    "open domain qa": "knowledge",
    "multiple choice qa": "knowledge",
    "medical knowledge": "knowledge",
    "legal": "knowledge",
    "stem": "knowledge",
    "humanities": "knowledge",
    "social sciences": "knowledge",
    "dialogue modeling": "language_understanding",
    "text generation": "language_understanding",
    "text classification": "language_understanding",
}

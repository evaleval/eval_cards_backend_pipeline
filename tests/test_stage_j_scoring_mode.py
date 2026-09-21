"""Stage J — `scoring_mode` on `eval_results_view`.

Generated text or log-probabilities. The distinction decides which setup
fields can exist for a run at all: a log-prob result scores the model's
likelihood over fixed answer choices, so temperature and max_tokens are not
undisclosed, they are ABSENT. Around a quarter of the corpus is scored that
way, and without this column every one of those rows reads as a source that
failed to document its setup.

Builds a minimal `eval_results_view` and runs the real stage against it.
"""
from __future__ import annotations

import logging
import textwrap

import pytest

duckdb = pytest.importorskip("duckdb")

from eval_card_backend.canonicalise.stages import stage_j_scoring_mode
from eval_card_backend.sources.scoring_modes import load_scoring_modes


def _view(con, rows: list[tuple[str, str, str | None]]) -> None:
    """(composite_slug, benchmark_id, output_type) -> a view-shaped table."""
    con.execute(
        """
        CREATE TABLE eval_results_view(
            composite_slug VARCHAR,
            benchmark_id VARCHAR,
            generation_config STRUCT(additional_details VARCHAR)
        )
        """
    )
    for composite_slug, benchmark_id, output_type in rows:
        details = (
            "NULL" if output_type is None
            else f"""'{{"output_type": "{output_type}"}}'"""
        )
        con.execute(
            f"INSERT INTO eval_results_view VALUES "
            f"('{composite_slug}', '{benchmark_id}', {{'additional_details': {details}}})"
        )


def _modes(con) -> list[str | None]:
    return [r[0] for r in con.execute(
        "SELECT scoring_mode FROM eval_results_view"
    ).fetchall()]


def test_record_output_type_wins():
    """The source saying so itself beats any table we maintain."""
    con = duckdb.connect()
    _view(con, [
        ("anything", "anything", "multiple_choice"),
        ("anything", "anything", "loglikelihood"),
        ("anything", "anything", "loglikelihood_rolling"),
        ("anything", "anything", "generate_until"),
    ])
    stage_j_scoring_mode(con)
    assert _modes(con) == ["log_prob", "log_prob", "log_prob", "generative"]


def test_falls_back_to_the_harness_derived_map():
    """HF Open LLM v2 keeps output_type in the leaderboard's dumps, not in
    the record, so those rows resolve from scoring_modes.yaml."""
    con = duckdb.connect()
    _view(con, [
        ("hf-open-llm-v2", "bbh", None),
        ("hf-open-llm-v2", "gpqa", None),
        ("hf-open-llm-v2", "musr", None),
        ("hf-open-llm-v2", "mmlu-pro", None),
        ("hf-open-llm-v2", "ifeval", None),
        ("hf-open-llm-v2", "math-level-5", None),
    ])
    stage_j_scoring_mode(con)
    assert _modes(con) == [
        "log_prob", "log_prob", "log_prob", "log_prob", "generative", "generative",
    ]


def test_unknown_stays_null_rather_than_guessing():
    """NULL is a real answer. A consumer must be able to tell "we do not know"
    from "generative" — the whole bug being fixed here came from a rule that
    could not."""
    con = duckdb.connect()
    _view(con, [
        ("some-source", "some-benchmark", None),
        # helm_* runs multiple choice by GENERATING the answer letter, so the
        # name is not evidence and must not be treated as any.
        ("helm_mmlu", "mmlu", None),
        # An output_type we do not recognise is also not a guess.
        ("some-source", "some-benchmark", "something_new"),
    ])
    stage_j_scoring_mode(con)
    assert _modes(con) == [None, None, None]


def test_a_present_record_output_type_is_final_even_when_unrecognised():
    """A mapped benchmark that starts reporting its own output_type has taken
    the mapping's job back. If we cannot read what it now says the answer is
    "cannot say": the older benchmark-wide table is no longer describing that
    row and must not answer for it."""
    con = duckdb.connect()
    _view(con, [
        ("hf-open-llm-v2", "bbh", "something_new"),
        ("hf-open-llm-v2", "bbh", ""),
        ("hf-open-llm-v2", "bbh", None),
        ("hf-open-llm-v2", "bbh", "generate_until"),
    ])
    stage_j_scoring_mode(con)
    assert _modes(con) == [None, "log_prob", "log_prob", "generative"]


def test_an_unreadable_output_type_is_named_once_per_value(caplog):
    """A harness renaming its output types hits tens of thousands of rows at
    once, so the log says it once per value rather than once per row."""
    con = duckdb.connect()
    _view(con, [
        ("some-source", "some-benchmark", "something_new"),
        ("some-source", "other-benchmark", "something_new"),
        ("some-source", "some-benchmark", "another_new"),
        ("some-source", "some-benchmark", "generate_until"),
    ])
    with caplog.at_level(logging.WARNING, logger="eval_card_backend.canonicalise.stages"):
        stage_j_scoring_mode(con)
    named = [r.getMessage() for r in caplog.records if "output_type" in r.getMessage()]
    assert len(named) == 2
    assert "'something_new'" in named[0] and "2 row(s)" in named[0]
    assert "'another_new'" in named[1] and "1 row(s)" in named[1]


def test_survives_a_missing_mapping_file(tmp_path, monkeypatch):
    """A bake must not die because a fallback table is absent: every row
    degrades to "cannot say", which consumers already handle."""
    import eval_card_backend.sources.scoring_modes as mod

    monkeypatch.setattr(mod, "DEFAULT_SCORING_MODES_PATH", tmp_path / "absent.yaml")
    assert mod.load_scoring_modes() == []


def test_ignores_an_entry_with_a_bad_mode(tmp_path):
    """A typo withholds that one entry, loudly, and leaves the rest."""
    import eval_card_backend.sources.scoring_modes as mod

    path = tmp_path / "scoring_modes.yaml"
    path.write_text(textwrap.dedent("""
        version: 1
        sources:
          a-source:
            benchmarks:
              good: {mode: log_prob}
              typo: {mode: logprob}
    """))
    assert mod.load_scoring_modes(path) == [("a-source", "good", "log_prob")]


def test_shipped_mapping_is_loadable_and_covers_hf_open_llm_v2():
    entries = load_scoring_modes()
    hf = {b: m for src, b, m in entries if src == "hf-open-llm-v2"}
    # The four log-prob benchmarks are the ~17.7k rows this exists for.
    assert hf["bbh"] == "log_prob"
    assert hf["gpqa"] == "log_prob"
    assert hf["musr"] == "log_prob"
    assert hf["mmlu-pro"] == "log_prob"
    assert hf["ifeval"] == "generative"
    assert hf["math-level-5"] == "generative"

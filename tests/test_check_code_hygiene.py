"""`scripts/check_code_hygiene.py`.

Every rule is tried against a line of the kind it was written for and a
near-miss that must pass, then both diff modes run the script for real
against a throwaway git repo. The rule cases run inside that repo too,
whose `.gitignore` ignores `notes/`, so `ignored-path-ref` is judged
against a known ignore list rather than this repo's.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_code_hygiene.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_code_hygiene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hygiene = _load_script()

# (rule, a line that must fire it, a near-miss that must not). The firing
# lines marked "committed" were added to this repo before the gate existed
# (found with `git log -p`); the rest are made up, no past commit had one.
CASES = [
    (
        "spec-section-ref",
        "# only aggregate root-scope metrics; spec §8.5b is",  # committed
        "# only aggregate root-scope metrics",
    ),
    (
        "spec-label-ref",
        "# Additive merged-view link (spec P6): the canonical benchmark id when a",  # committed
        "# Additive merged-view link: the canonical benchmark id when one exists",
    ),
    (
        "spec-deixis",
        '"""Materialize ``aggregate__<suite_key>`` rows per spec reshape/05.',  # committed
        '"""Materialize ``aggregate__<suite_key>`` rows, one per suite.',
    ),
    (
        "section-number-ref",
        "# Dedup order follows section 4(b) of the design notes.",  # made up
        "# sections are sorted by display order",
    ),
    (
        "design-point-ref",
        "Grain rules (spec design pt 7): a benchmark with >=1 top-level",  # committed
        "Grain rules: a benchmark with >=1 top-level",
    ),
    (
        "phase-ref",
        "# Phase 1: direct lookup by the primary identifiers. metadata_lookup is",  # committed
        "# two-phase commit across the stage caches",
    ),
    (
        "session-ref",
        "# `notes/provenance-investigation.md` Session 5g/5h). Falls back to",  # committed
        "# one DuckDB session per pipeline run",
    ),
    (
        "tombstone-previously",
        "-- rule as evals_view.source_data). Was previously hard-coded",  # committed
        "-- previously-seen ids are skipped",
    ),
    (
        "tombstone-used-to",
        "# this column used to be nullable",  # made up
        "# the cache is used to skip stages A-D",
    ),
    (
        "tombstone-dated-change",
        "# renamed on 2026-08-16 because the view layer dropped the old key",  # made up
        "# snapshot ids are dated, e.g. 2026-08-16T13-44-51Z",
    ),
    (
        "ignored-path-ref",
        "# Curated collection entries (see notes/collections-spec.md).",  # committed
        "# Curated collection entries (see README.md).",
    ),
]


def test_every_rule_has_a_case():
    assert {name for name, _, _ in CASES} == {rule.name for rule in hygiene.RULES}


@pytest.mark.parametrize("_,line,__", CASES, ids=[c[0] for c in CASES])
def test_the_opt_out_marker_silences_every_rule(_, line, __, in_repo):
    assert hygiene.matching_rules(f"{line}  # hygiene: ok") == []


@pytest.mark.parametrize("rule,line,_", CASES, ids=[c[0] for c in CASES])
def test_rule_fires_on_the_kind_of_line_it_was_written_for(rule, line, _, in_repo):
    assert rule in {r.name for r in hygiene.matching_rules(line)}


@pytest.mark.parametrize("rule,_,line", CASES, ids=[c[0] for c in CASES])
def test_rule_stays_quiet_on_a_near_miss(rule, _, line, in_repo):
    assert hygiene.matching_rules(line) == []


@pytest.mark.parametrize(
    "line,cited",
    [
        ("# see notes/spec.md and notes/sub/more.txt", {"notes/spec.md", "notes/sub/more.txt"}),
        ("# README.md explains it; so does docs/guide.rst", set()),
        ("# ../other-repo/notes/x.md and /abs/notes/y.md are outside the repo", set()),
        ("# https://example.org/notes/z.md is a URL, not a path", set()),
        ("# notes/report.json is data, not a document", set()),
    ],
)
def test_ignored_paths_only_reports_documents_inside_the_repo(line, cited, in_repo):
    assert hygiene.ignored_paths(hygiene.DOC_PATH_RE.findall(line)) == cited


def test_ignored_paths_reads_non_ascii_paths(in_repo):
    paths = ["notes/café.md", "notes/设计.md", "README.md"]
    assert hygiene.ignored_paths(paths) == {"notes/café.md", "notes/设计.md"}
    assert hygiene.cites_ignored_path("# see notes/café.md")


def test_ignored_paths_raises_when_git_fails_and_caches_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    hygiene._ignored_cache.clear()

    with pytest.raises(hygiene.GitUnavailable):
        hygiene.ignored_paths(["notes/x.md"])
    assert hygiene._ignored_cache == {}


def test_ignored_paths_raises_when_git_cannot_run(in_repo, monkeypatch):
    monkeypatch.setenv("PATH", "")

    with pytest.raises(hygiene.GitUnavailable):
        hygiene.ignored_paths(["notes/x.md"])
    assert hygiene._ignored_cache == {}


@pytest.mark.parametrize(
    "path,checked",
    [
        ("src/eval_card_backend/x.py", True),
        ("tests/test_x.py", True),
        (".github/workflows/ci.yml", True),
        ("pyproject.toml", True),
        ("README.md", False),
        ("notes/spec.py", False),
        ("DEPRECATED_old/x.py", False),
        ("scripts/check_code_hygiene.py", False),
        ("tests/test_check_code_hygiene.py", False),
        ("hooks/pre-commit", False),
        ("hooks/pre-commit.py", True),
    ],
)
def test_is_checked(path, checked):
    assert hygiene.is_checked(path) is checked


def test_parse_added_lines_numbers_lines_from_hunk_headers():
    diff = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -3,0 +4,2 @@\n"
        "+one\n"
        "+two\n"
        "@@ -10 +12 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/notes/b.md b/notes/b.md\n"
        "--- a/notes/b.md\n"
        "+++ b/notes/b.md\n"
        "@@ -0,0 +1 @@\n"
        "+# Phase 9\n"
    )
    assert list(hygiene.parse_added_lines(diff)) == [
        ("src/a.py", 4, "one"),
        ("src/a.py", 5, "two"),
        ("src/a.py", 12, "new"),
        ("notes/b.md", 1, "# Phase 9"),
    ]


def test_parse_added_lines_reads_quoted_tabbed_and_dev_null_headers():
    diff = (
        "diff --git a/src/a b.py b/src/a b.py\n"
        "--- a/src/a b.py\t\n"
        "+++ b/src/a b.py\t\n"
        "@@ -0,0 +1 @@\n"
        "+one\n"
        'diff --git "a/src/caf\\303\\251 \\"x\\".py" "b/src/caf\\303\\251 \\"x\\".py"\n'
        "--- /dev/null\n"
        '+++ "b/src/caf\\303\\251 \\"x\\".py"\t\n'
        "@@ -0,0 +1 @@\n"
        "+two\n"
        "diff --git a/src/gone.py b/src/gone.py\n"
        "--- a/src/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-three\n"
        "diff --git a/src/c.py b/src/c.py\n"
        "--- a/src/c.py\n"
        "+++ b/src/c.py\n"
        "@@ -1 +1,2 @@\n"
        "-four\n"
        "+++x  # Phase 2\n"
        "+five\n"
    )
    assert list(hygiene.parse_added_lines(diff)) == [
        ("src/a b.py", 1, "one"),
        ('src/café "x".py', 1, "two"),
        ("src/c.py", 1, "++x  # Phase 2"),
        ("src/c.py", 2, "five"),
    ]


def test_parse_added_lines_reads_header_looking_content_inside_a_hunk_as_content():
    diff = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "--- not a header\n"
        "-+++ removed too\n"
        "+++ also content\n"
        "+--- added, not a header\n"
        "+diff --git looking text\n"
        "diff --git a/src/b.py b/src/b.py\n"
        "--- a/src/b.py\n"
        "+++ b/src/b.py\n"
        "@@ -0,0 +1 @@\n"
        "+in b\n"
    )
    assert list(hygiene.parse_added_lines(diff)) == [
        ("src/a.py", 1, "++ also content"),
        ("src/a.py", 2, "--- added, not a header"),
        ("src/a.py", 3, "diff --git looking text"),
        ("src/b.py", 1, "in b"),
    ]


def test_parse_added_lines_keeps_tabs_and_unicode_across_files():
    diff = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+\tindented\twith tabs\n"
        "+# 设计 café §\n"
        "diff --git a/bin.dat b/bin.dat\n"
        "Binary files a/bin.dat and b/bin.dat differ\n"
        "diff --git a/src/c.py b/src/c.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/c.py\n"
        "@@ -0,0 +1 @@\n"
        "+ünïcode\n"
    )
    assert list(hygiene.parse_added_lines(diff)) == [
        ("src/a.py", 1, "\tindented\twith tabs"),
        ("src/a.py", 2, "# 设计 café §"),
        ("src/c.py", 1, "ünïcode"),
    ]


def test_header_looking_content_and_unicode_in_a_real_diff(repo):
    """A removed `-- old` and an added `++ new` make adjacent `--- old` and
    `+++ new` lines inside the hunk; both stay content."""
    _write(repo, "src/a.py", "-- old\n")
    _write(repo, "src/b.py", "y = 1\n")
    _commit(repo, "files")
    _write(repo, "src/a.py", "++ new  # Phase 2\n\t# 设计 Session 3\n")
    _write(repo, "src/b.py", "y = 1\n++ b/src/z.py  # see spec §2\n")
    _git(repo, "add", "-A")

    proc = _run(repo)

    assert proc.returncode == 1
    assert "src/a.py:1  [phase-ref]" in proc.stderr
    assert "src/a.py:2  [session-ref]" in proc.stderr
    assert "src/b.py:2  [spec-section-ref]" in proc.stderr
    assert "src/z.py:" not in proc.stderr


# ---------------------------------------------------------------------------
# The script against a real repo
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    )


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(
        repo, "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "--no-verify", "-m", message,
    )


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=repo, capture_output=True, text=True,
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path, ".gitignore", "notes/\n")
    # A pre-existing offender: never reported, only added lines are.
    _write(tmp_path, "src/mod.py", "# Phase 1 lookup\nx = 1\n")
    _commit(tmp_path, "base")
    return tmp_path


@pytest.fixture
def in_repo(repo, monkeypatch):
    """Run the loaded module's git calls inside the throwaway repo."""
    monkeypatch.chdir(repo)
    hygiene._ignored_cache.clear()
    yield repo
    hygiene._ignored_cache.clear()


def test_staged_mode_reports_added_lines_in_checked_files_only(repo):
    _write(repo, "src/mod.py", "# Phase 1 lookup\nx = 1\n# see spec §3 for the fold\n")
    _write(repo, "docs/design.md", "# Phase 2\n")
    _git(repo, "add", "-A")

    proc = _run(repo)

    assert proc.returncode == 1
    assert "problem(s) in staged changes" in proc.stderr
    assert "src/mod.py:3  [spec-section-ref]" in proc.stderr
    assert "src/mod.py:3  [spec-deixis]" in proc.stderr
    assert "phase-ref" not in proc.stderr
    assert "docs/" not in proc.stderr
    assert "--no-verify" in proc.stderr


def test_staged_mode_reports_a_cited_document_at_an_ignored_path(repo):
    _write(repo, "README.md", "# tracked\n")
    _write(
        repo, "src/mod.py",
        "# Phase 1 lookup\nx = 1\n# rule from notes/design.md\n# see README.md\n",
    )
    _git(repo, "add", "-A")

    proc = _run(repo)

    assert proc.returncode == 1
    assert "src/mod.py:3  [ignored-path-ref]" in proc.stderr
    assert "src/mod.py:4" not in proc.stderr
    assert "gitignored" in proc.stderr


def test_staged_mode_reports_a_non_ascii_document_at_an_ignored_path(repo):
    _write(repo, "src/mod.py", "# Phase 1 lookup\nx = 1\n# see notes/设计.md\n")
    _git(repo, "add", "-A")

    proc = _run(repo)

    assert proc.returncode == 1, proc.stderr
    assert "src/mod.py:3  [ignored-path-ref]" in proc.stderr


def test_staged_mode_skips_lines_marked_hygiene_ok(repo):
    _write(
        repo, "src/mod.py",
        "# Phase 1 lookup\nx = 1\n# see spec §3 for the fold  (hygiene: ok)\n"
        "# Phase 2 follows\n",
    )
    _git(repo, "add", "-A")

    proc = _run(repo)

    assert proc.returncode == 1
    assert "src/mod.py:3" not in proc.stderr
    assert "src/mod.py:4  [phase-ref]" in proc.stderr
    assert "hygiene: ok" in proc.stderr


def test_outside_a_repository_exits_2_with_one_line(tmp_path):
    proc = _run(tmp_path)

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert proc.stderr.count("\n") == 1
    assert proc.stderr.startswith("code-hygiene: git diff --cached failed")
    assert "Traceback" not in proc.stderr


def test_an_unknown_range_exits_2(repo):
    proc = _run(repo, "--diff", "no-such-ref...HEAD")

    assert proc.returncode == 2
    assert proc.stderr.startswith("code-hygiene: git diff no-such-ref...HEAD failed")


def test_staged_mode_passes_clean_changes_and_ignores_the_working_tree(repo):
    _write(repo, "src/mod.py", "# Phase 1 lookup\nx = 2\n")
    _git(repo, "add", "-A")
    _write(repo, "src/mod.py", "# Phase 1 lookup\nx = 2\n# Session 9 leftover\n")

    proc = _run(repo)

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""


def test_diff_mode_checks_the_commit_range_and_not_the_index(repo):
    _write(repo, "src/mod.py", "# Phase 1 lookup\nx = 1\nY = 2  # was previously 3\n")
    _commit(repo, "offending commit")
    _write(repo, "src/other.py", "z = 1\n")
    _git(repo, "add", "-A")

    proc = _run(repo, "--diff", "HEAD~1...HEAD")

    assert proc.returncode == 1
    assert "problem(s) in diff HEAD~1...HEAD" in proc.stderr
    assert "src/mod.py:3  [tombstone-previously]" in proc.stderr
    assert "phase-ref" not in proc.stderr
    assert "--no-verify" not in proc.stderr
    assert _run(repo).returncode == 0


def test_diff_prefix_and_external_diff_config_cannot_blank_the_parser(repo):
    """`diff.noprefix`, `diff.mnemonicPrefix` and `diff.external` change how
    `git diff` prints headers; the gate must still see every added line."""
    _git(repo, "config", "diff.noprefix", "true")
    _git(repo, "config", "diff.mnemonicPrefix", "true")
    _git(repo, "config", "diff.external", "/usr/bin/true")
    _write(repo, "src/mod.py", "# Phase 1 lookup\nx = 1\nY = 2  # was previously 3\n")
    _git(repo, "add", "-A")

    staged = _run(repo)
    assert staged.returncode == 1, staged.stderr
    assert "src/mod.py:3  [tombstone-previously]" in staged.stderr

    _commit(repo, "offending commit")
    ranged = _run(repo, "--diff", "HEAD~1...HEAD")
    assert ranged.returncode == 1, ranged.stderr
    assert "src/mod.py:3  [tombstone-previously]" in ranged.stderr


def test_paths_with_a_space_or_non_ascii_and_lines_starting_with_plus_plus(repo):
    _write(repo, "src/a b.py", "x = 1\n# Phase 2 here\n")
    _write(repo, "src/café.py", "x = 1\n# see spec §3\n")
    _write(repo, "src/c.py", "ok = 1\n++x  # Phase 2\n")
    _git(repo, "add", "-A")

    proc = _run(repo)

    assert proc.returncode == 1
    assert "src/a b.py:2  [phase-ref]" in proc.stderr
    assert "src/café.py:2  [spec-section-ref]" in proc.stderr
    assert "src/c.py:2  [phase-ref]" in proc.stderr
    assert "src/c.py:2  [spec-section-ref]" not in proc.stderr


def test_diff_mode_passes_when_the_range_only_removes_or_moves_offenders(repo):
    _write(repo, "src/mod.py", "x = 1\n")
    _commit(repo, "drop the comment")

    proc = _run(repo, "--diff", "HEAD~1...HEAD")

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""

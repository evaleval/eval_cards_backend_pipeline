#!/usr/bin/env python3
"""Code-hygiene gate for added lines. Runs from the pre-commit hook and in CI.

Blocks comments/code that reference ephemeral working artifacts: spec
sections, decision labels, phase/session tags, tombstones describing
what the code used to do, and documents at gitignored paths. Rationale:
those references are only meaningful inside the working session (or on
the working machine) that produced them; the committed code and its own
comments must stand alone.

Checks only lines ADDED in the staged diff, so existing occurrences don't
block unrelated commits. `--diff <range>` checks the lines added in a git
revision range instead (CI runs it over the pull request or the pushed
range). A line containing `hygiene: ok` is skipped by every rule; use it
for a false positive (and consider tightening the offending rule below).

Exit status: 0 clean, 1 problems found, 2 when git could not produce the
diff (not a repository, unknown range).

To add a rule, append to RULES. A rule fires when its pattern matches
the line and, if it has one, its `confirm` callback agrees.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Callable, NamedTuple


class Rule(NamedTuple):
    name: str
    pattern: re.Pattern
    hint: str
    confirm: Callable[[str], bool] | None = None


OPT_OUT = "hygiene: ok"


class GitUnavailable(Exception):
    """`git diff` could not run or failed."""


# A path a reader is meant to open: a document, by extension. Runtime data
# paths (`warehouse/<snapshot>/x.json`, `.cache/...`) are also gitignored
# but code describes them rather than citing them, so data extensions and
# bare directory names stay out of this pattern on purpose.
DOC_PATH_RE = re.compile(r"(?<![\w./-])(?!\.\.)[\w.-]*(?:/[\w.-]+)*\.(?:md|txt|rst)\b")

_ignored_cache: dict[str, bool] = {}


def ignored_paths(paths) -> set[str]:
    """Return the subset of `paths` matched by the repo's ignore rules.

    One `git check-ignore --stdin` call covers every path not seen before;
    results are cached for the life of the process. Paths that cannot be
    inside the repo (absolute or `..`-relative) are skipped.
    """
    paths = list(paths)
    pending = sorted({
        p for p in paths
        if p not in _ignored_cache and not p.startswith(("/", ".."))
    })
    if pending:
        proc = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"],
            input="\n".join(pending), capture_output=True, text=True,
        )
        # exit 0: some path is ignored; 1: none is; anything else: git failed.
        hits = set(proc.stdout.splitlines()) if proc.returncode in (0, 1) else set()
        for p in pending:
            _ignored_cache[p] = p in hits
    return {p for p in paths if _ignored_cache.get(p)}


def cites_ignored_path(text: str) -> bool:
    return bool(ignored_paths(DOC_PATH_RE.findall(text)))


RULES = [
    Rule(
        "spec-section-ref",
        re.compile("§"),  # the section sign
        "Spec section references don't survive the session that wrote them. "
        "State the rule itself in the comment instead.",
    ),
    Rule(
        "spec-label-ref",
        re.compile(r"[Ss]pec[.,]?\s*[A-Z]\d+\b"),
        "Decision labels like 'spec D9' / 'spec P3' are one-time artifacts. "
        "Say what the decision was, not where it was numbered.",
    ),
    Rule(
        "spec-deixis",
        re.compile(r"\b(per|as per|see)\s+(the\s+)?spec\b|\bspec\s+section\b", re.I),
        "Don't point readers at 'the spec'. Inline the relevant rule.",
    ),
    Rule(
        "section-number-ref",
        re.compile(r"\bsection\s+(\d|\([a-z]\))", re.I),
        "Numbered-section references rot. Describe the rule in place.",
    ),
    Rule(
        "design-point-ref",
        re.compile(r"\bdesign\s+pt\.?\s*\d", re.I),
        "Design-point numbers are meaningless outside the design doc. "
        "State the design decision directly.",
    ),
    Rule(
        "phase-ref",
        re.compile(r"\bphase[-\s]?\d", re.I),
        "Phase numbers only make sense inside a plan. Name the feature or "
        "component, not the phase that built it.",
    ),
    Rule(
        "session-ref",
        re.compile(r"\bsession\s+\d+[a-z]?\b|\bthis (session|conversation|chat)\b", re.I),
        "Working-session references are unintelligible to later readers.",
    ),
    Rule(
        "tombstone-previously",
        re.compile(r"\b(was|were)\s+(previously|formerly)\b", re.I),
        "Tombstone: git history records what the code used to do; the "
        "comment should only explain what it does now.",
    ),
    Rule(
        "tombstone-used-to",
        re.compile(r"\bused\s+to\s+(be|do|have|return|live|contain)\b", re.I),
        "Tombstone: describe current behaviour; history lives in git.",
    ),
    Rule(
        "tombstone-dated-change",
        re.compile(r"\b(renamed|removed|deleted|moved|dropped|changed)\s+(on|in)\s+20\d\d", re.I),
        "Dated change notes belong in the commit message, not the file.",
    ),
    Rule(
        "ignored-path-ref",
        DOC_PATH_RE,
        "This path is gitignored, so readers and CI cannot follow it. "
        "State the rule inline instead.",
        confirm=cites_ignored_path,
    ),
]

CHECKED_EXTENSIONS = {
    ".py", ".pyi", ".sql", ".sh", ".bash", ".zsh",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts",
    ".css", ".scss", ".yaml", ".yml", ".toml",
}

# Design docs may legitimately carry section numbering; frozen trees are
# not held to current hygiene.
EXCLUDED_DIR_PARTS = {"notes", "specs", "docs", "vendor", "node_modules", "deprecated_refs"}
EXCLUDED_PATHS = {
    "scripts/check_code_hygiene.py",
    "tests/test_check_code_hygiene.py",
}


def is_checked(path: str) -> bool:
    if path in EXCLUDED_PATHS:
        return False
    parts = path.split("/")
    if any(p in EXCLUDED_DIR_PARTS or p.startswith("DEPRECATED") for p in parts[:-1]):
        return False
    dot = parts[-1].rfind(".")
    return dot != -1 and parts[-1][dot:] in CHECKED_EXTENSIONS


def added_lines(diff_target: str = "--cached"):
    """Yield (path, line_number, text) for every line added in a git diff.

    `diff_target` is `--cached` for the staged changes or a revision range
    such as `origin/main...HEAD`. The prefix, quoting and external-diff
    flags pin the header format `parse_added_lines` reads, whatever the
    user's git config.
    """
    try:
        proc = subprocess.run(
            [
                "git", "-c", "core.quotepath=false",
                "diff", "--unified=0", "--no-color", "--no-ext-diff",
                "--src-prefix=a/", "--dst-prefix=b/", "--diff-filter=ACMR",
                diff_target,
            ],
            capture_output=True, text=True,
        )
    except OSError as exc:
        raise GitUnavailable(f"cannot run git: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        raise GitUnavailable(
            f"git diff {diff_target} failed" + (f": {detail[0]}" if detail else "")
        )
    return parse_added_lines(proc.stdout)


_C_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


def _unquote(quoted: str) -> str:
    r"""Undo git's C-style quoting of a path: `"caf\303\251 \"x\".py"`."""
    out = bytearray()
    chars = iter(quoted[1:-1])
    for c in chars:
        if c != "\\":
            out += c.encode()
            continue
        c = next(chars)
        if c in "01234567":
            out.append(int(c + next(chars) + next(chars), 8))
        else:
            out.append(_C_ESCAPES[c])
    return out.decode(errors="surrogateescape")


def _header_path(header: str) -> str | None:
    """The path in a `+++ ` header, `/dev/null` as None. Git quotes a path
    with special characters C-style and ends one containing a space with a
    tab."""
    name = header.rstrip("\t")
    if name.startswith('"'):
        name = _unquote(name)
    return name[2:] if name.startswith("b/") else None


def parse_added_lines(diff: str):
    """Yield (path, line_number, text) for every added line in unified diff text.

    The `+++ ` line right after a file's `--- ` header names the file; a
    `+++` anywhere else is an added line that starts with `++`.
    """
    path, lineno, expect = None, 0, None
    for raw in diff.splitlines():
        if raw.startswith("diff --git "):
            path, expect = None, "--- "
        elif expect == "--- " and raw.startswith(expect):
            expect = "+++ "
        elif expect == "+++ ":
            path, expect = _header_path(raw[4:]) if raw.startswith(expect) else None, None
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            lineno = int(m.group(1)) if m else 0
        elif raw.startswith("+"):
            if path is not None:
                yield path, lineno, raw[1:]
            lineno += 1


def matching_rules(text: str) -> list[Rule]:
    if OPT_OUT in text:
        return []
    return [
        rule for rule in RULES
        if rule.pattern.search(text) and (rule.confirm is None or rule.confirm(text))
    ]


def find_problems(lines) -> list[tuple[str, int, Rule, str]]:
    lines = [(path, lineno, text) for path, lineno, text in lines if is_checked(path)]
    # Warm the ignore cache with one git call instead of one per line.
    ignored_paths(tok for _, _, text in lines for tok in DOC_PATH_RE.findall(text))
    findings = []
    for path, lineno, text in lines:
        for rule in matching_rules(text):
            findings.append((path, lineno, rule, text))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--diff", metavar="RANGE", default=None,
        help="check the lines added in this git revision range (for example "
             "origin/main...HEAD) instead of the staged changes",
    )
    args = parser.parse_args(argv)
    try:
        if args.diff is None:
            scope, bypass = "staged changes", ", or bypass once with: git commit --no-verify"
            findings = find_problems(added_lines())
        else:
            scope, bypass = f"diff {args.diff}", ""
            findings = find_problems(added_lines(args.diff))
    except GitUnavailable as exc:
        print(f"code-hygiene: {exc}", file=sys.stderr)
        return 2
    if not findings:
        return 0
    print(f"code-hygiene: {len(findings)} problem(s) in {scope}\n", file=sys.stderr)
    for path, lineno, rule, text in findings:
        print(f"  {path}:{lineno}  [{rule.name}]", file=sys.stderr)
        print(f"      {text.strip()[:120]}", file=sys.stderr)
        print(f"      -> {rule.hint}\n", file=sys.stderr)
    print(
        f"Fix the lines above, mark a false positive with '{OPT_OUT}'{bypass}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

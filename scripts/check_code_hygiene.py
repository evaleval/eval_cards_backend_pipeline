#!/usr/bin/env python3
"""Code-hygiene gate for added lines. Runs from the pre-commit hook and in CI.

Blocks comments/code that reference ephemeral working artifacts: spec
sections, decision labels, phase/session tags, and tombstones describing
what the code used to do. Rationale: those references are only meaningful
inside the working session that produced them; the committed code and its
own comments must stand alone.

Checks only lines ADDED in the staged diff, so existing occurrences don't
block unrelated commits. `--diff <range>` checks the lines added in a git
revision range instead (CI runs `--diff origin/main...HEAD` on a pull
request). Bypass a false positive with `git commit --no-verify` (and
consider tightening the offending rule below).

To add a rule, append to RULES.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import NamedTuple


class Rule(NamedTuple):
    name: str
    pattern: re.Pattern
    hint: str


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
    "hooks/pre-commit",
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
    such as `origin/main...HEAD`.
    """
    diff = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", "--diff-filter=ACMR", diff_target],
        capture_output=True, text=True, check=True,
    ).stdout
    return parse_added_lines(diff)


def parse_added_lines(diff: str):
    """Yield (path, line_number, text) for every added line in unified diff text."""
    path, lineno = None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            path = raw[6:]
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            lineno = int(m.group(1)) if m else 0
        elif raw.startswith("+") and not raw.startswith("+++"):
            if path is not None:
                yield path, lineno, raw[1:]
            lineno += 1


def matching_rules(text: str) -> list[Rule]:
    return [rule for rule in RULES if rule.pattern.search(text)]


def find_problems(lines) -> list[tuple[str, int, Rule, str]]:
    findings = []
    for path, lineno, text in lines:
        if not is_checked(path):
            continue
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
    if args.diff is None:
        scope, bypass = "staged changes", ", or bypass once with: git commit --no-verify"
        findings = find_problems(added_lines())
    else:
        scope, bypass = f"diff {args.diff}", ""
        findings = find_problems(added_lines(args.diff))
    if not findings:
        return 0
    print(f"code-hygiene: {len(findings)} problem(s) in {scope}\n", file=sys.stderr)
    for path, lineno, rule, text in findings:
        print(f"  {path}:{lineno}  [{rule.name}]", file=sys.stderr)
        print(f"      {text.strip()[:120]}", file=sys.stderr)
        print(f"      -> {rule.hint}\n", file=sys.stderr)
    print(f"Fix the lines above{bypass}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

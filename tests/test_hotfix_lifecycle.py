"""Hotfix lifecycle gate.

Data hotfixes live in three designated modules, and every public function
and override table there carries a lifecycle annotation: a `TODO(upstream):`
line naming the upstream defect and a `Remove when:` line naming the
condition that retires it. The annotation may sit in the symbol's own
docstring, in the comment block that opens its `# ── N. ...` section, or
(for a module that is one hotfix) in the module docstring.

Outside those modules no new hotfix-style table may appear. The tables that
already exist are grandfathered in ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES;
that list may only shrink, so a table that goes away must be struck from it.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "eval_card_backend"

HOTFIX_MODULES = {
    "eval_card_backend.canonicalise.resolution_hotfixes": PACKAGE / "canonicalise" / "resolution_hotfixes.py",
    "eval_card_backend.canonicalise.hierarchy_hotfixes": PACKAGE / "canonicalise" / "hierarchy_hotfixes.py",
    "eval_card_backend.metric_meta_hotfix": PACKAGE / "metric_meta_hotfix.py",
}

UPSTREAM_MARK = "TODO(upstream)"
REMOVAL_MARK = "Remove when"

# `module:name` of every hotfix-style table that predates this gate and still
# lives outside the hotfix modules. Shrink only; never add.
ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES = frozenset({
    "eval_card_backend.canonicalise.sidecars:_FALLBACK_ACRONYMS",
    "eval_card_backend.canonicalise.sidecars:_FAMILY_PRIMARY_OVERRIDE",
    "eval_card_backend.canonicalise.sidecars:_METRIC_KIND_TO_GROUP",
    "eval_card_backend.canonicalise.sidecars:_METRIC_NAME_GROUP_RULES",
})

HOTFIX_WORDS = re.compile(r"hotfix|override", re.IGNORECASE)
SECTION_HEADER = re.compile(r"^# ── ", re.MULTILINE)
TABLE_CALLS = {"dict", "set", "frozenset", "tuple", "list"}


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts)


def _assigned_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def _is_table(node: ast.stmt) -> bool:
    value = getattr(node, "value", None)
    if isinstance(value, (ast.Dict, ast.Set, ast.List, ast.Tuple)):
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in TABLE_CALLS
    )


def _leading_comment(lines: list[str], node: ast.stmt) -> str:
    """The contiguous `#` comment lines directly above `node`."""
    i = node.lineno - 2
    block: list[str] = []
    while i >= 0 and lines[i].lstrip().startswith("#"):
        block.insert(0, lines[i])
        i -= 1
    return "\n".join(block)


# ---------------------------------------------------------------------------
# Lifecycle annotations inside the hotfix modules
# ---------------------------------------------------------------------------


def _sections(source: str) -> list[tuple[int, str]]:
    """(first line number, text) of the module preamble and each `# ──` section."""
    starts = [0] + [source.count("\n", 0, m.start()) + 1 for m in SECTION_HEADER.finditer(source)]
    lines = source.splitlines()
    out = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        out.append((start + 1, "\n".join(lines[start:end])))
    return out


def _annotation_context(source: str, node: ast.stmt) -> str:
    section_text = ""
    for first_line, text in _sections(source):
        if first_line <= node.lineno:
            section_text = text
    docstring = ast.get_docstring(node) if isinstance(node, ast.FunctionDef) else None
    return section_text + "\n" + (docstring or "")


def unannotated_hotfixes(source: str) -> list[tuple[str, str]]:
    """(name, missing marker) for each public function or table lacking a mark."""
    tree = ast.parse(source)
    problems = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue
            name = node.name
        elif _is_table(node) and _assigned_name(node):
            name = _assigned_name(node)
        else:
            continue
        context = _annotation_context(source, node)
        for mark in (UPSTREAM_MARK, REMOVAL_MARK):
            if mark not in context:
                problems.append((name, mark))
    return problems


@pytest.mark.parametrize("module", sorted(HOTFIX_MODULES), ids=lambda m: m.rsplit(".", 1)[-1])
def test_every_hotfix_carries_a_lifecycle_annotation(module):
    problems = unannotated_hotfixes(HOTFIX_MODULES[module].read_text())
    assert not problems, (
        f"{module}: hotfixes without a complete lifecycle annotation:\n"
        + "\n".join(f"  {name}: no '{mark}:' line" for name, mark in problems)
        + "\nEach hotfix needs a 'TODO(upstream):' line naming the upstream defect"
        " and a 'Remove when:' line naming what retires it."
    )


def test_unannotated_hotfixes_finds_each_missing_half():
    source = (
        '"""Module."""\n'
        "# ── 1. Complete ──\n"
        "# TODO(upstream): x is wrong.\n"
        "# Remove when: x is right.\n"
        "_X_MAP = {}\n\n"
        "def fix_x(con):\n    pass\n\n"
        "# ── 2. Only the removal ──\n"
        "# Remove when: y is right.\n"
        "def fix_y(con):\n    pass\n\n"
        "# ── 3. Only the pointer ──\n"
        "# TODO(upstream): z is wrong.\n"
        "_Z_RULES = ()\n\n"
        "def _helper():\n    pass\n"
    )
    assert unannotated_hotfixes(source) == [
        ("fix_y", UPSTREAM_MARK),
        ("_Z_RULES", REMOVAL_MARK),
    ]


def test_a_docstring_annotation_counts():
    source = (
        "def fix_w(con):\n"
        '    """Fix w.\n\n    TODO(upstream): w is wrong.\n    Remove when: w is right.\n    """\n'
    )
    assert unannotated_hotfixes(source) == []


# ---------------------------------------------------------------------------
# Ratchet: no new hotfix-style tables outside the hotfix modules
# ---------------------------------------------------------------------------


def hotfix_tables(source: str) -> list[str]:
    """Module-level table assignments named, or introduced by a comment, as a
    hotfix or override."""
    lines = source.splitlines()
    found = []
    for node in ast.parse(source).body:
        name = _assigned_name(node)
        if not name or not _is_table(node):
            continue
        if HOTFIX_WORDS.search(name) or HOTFIX_WORDS.search(_leading_comment(lines, node)):
            found.append(name)
    return found


def tables_outside_hotfix_modules() -> set[str]:
    hotfix_paths = set(HOTFIX_MODULES.values())
    found = set()
    for path in sorted(SRC.rglob("*.py")):
        if path in hotfix_paths:
            continue
        module = _module_name(path)
        found.update(f"{module}:{name}" for name in hotfix_tables(path.read_text()))
    return found


def test_hotfix_tables_matches_by_name_or_leading_comment():
    source = (
        "import re\n"
        "_PLAIN = {}\n"
        "_FOO_OVERRIDE = {}\n"
        "# Producer-local hotfix for a mislabelled feed.\n"
        "_BAR_MAP: dict[str, str] = {'a': 'b'}\n"
        "\n"
        "# Unrelated comment, separated by a blank line.\n"
        "\n"
        "_BAZ_RULES = (('x', re.compile('x')),)\n"
        "_QUX_SET = frozenset({'override'})\n"
        "def f():\n    _LOCAL_OVERRIDE = {}\n"
    )
    assert hotfix_tables(source) == ["_FOO_OVERRIDE", "_BAR_MAP"]


def test_no_new_hotfix_tables_outside_the_hotfix_modules():
    found = tables_outside_hotfix_modules()
    new = sorted(found - ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES)
    assert not new, (
        "hotfix-style tables outside the hotfix modules:\n"
        + "\n".join(f"  {symbol}" for symbol in new)
        + "\nPut the table in canonicalise/resolution_hotfixes.py or"
        " canonicalise/hierarchy_hotfixes.py under a section with a"
        " 'TODO(upstream):' line and a 'Remove when:' line."
    )


def test_the_allowlist_only_shrinks():
    gone = sorted(ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES - tables_outside_hotfix_modules())
    assert not gone, (
        "grandfathered tables no longer found:\n"
        + "\n".join(f"  {symbol}" for symbol in gone)
        + "\nRemove them from ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES so the list keeps shrinking."
    )

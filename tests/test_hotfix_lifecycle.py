"""Hotfix lifecycle gate.

Data hotfixes live in three designated modules. Every public function there,
and every override table the scanner below recognises, carries a lifecycle
annotation in its own docstring or in the comment block directly above it:
a `TODO(upstream):` line naming the upstream defect and a `Remove when:`
line naming the condition that retires it. The module docstring covers a
module with exactly one public symbol, and nothing else; a `# ──` section
header covers only the symbol it sits directly above.

Outside those modules no new override table may appear. A table is any
assignment, at module level or under `if`/`try`, whose target name reads
like an override (`hotfix`, `override`, `_RULES`, `_FIX`, `_FIXES`, `_PATCH`)
or whose leading comment or docstring says so (`hotfix`, `override`,
`producer-local`, `correction`, `workaround`), whatever the right-hand side
is. Inside a function or class body the same test applies to CONSTANT_CASE
names; a lower-case local variable is not a table. The tables that already
exist are grandfathered in
ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES together with a hash of their source,
so the list may only shrink and an entry added to one of them is a deliberate
change that updates the hash.

This is a tripwire against copying the pattern, not a proof: a table under a
neutral name with a neutral comment passes. The scanner catches the habit of
naming and describing the thing honestly; review catches the rest.
"""
from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "eval_card_backend"

HOTFIX_MODULES = {
    "eval_card_backend.canonicalise.resolution_hotfixes": PACKAGE / "canonicalise" / "resolution_hotfixes.py",
    "eval_card_backend.canonicalise.hierarchy_hotfixes": PACKAGE / "canonicalise" / "hierarchy_hotfixes.py",
    "eval_card_backend.metric_meta_hotfix": PACKAGE / "metric_meta_hotfix.py",
}

UPSTREAM_MARK = "TODO(upstream):"
REMOVAL_MARK = "Remove when:"

# `module:symbol` -> hash of the assignment's AST, for every override table
# that predates this gate and still lives outside the hotfix modules. Shrink
# only; never add. A changed hash means the table's content changed.
ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES: dict[str, str] = {
    "eval_card_backend.canonicalise.evalcard_tags:_FALLBACK_RULES": "3ab1403adbac0e56",
    "eval_card_backend.canonicalise.sidecars:_FALLBACK_ACRONYMS": "e137e973f45681bd",
    "eval_card_backend.canonicalise.sidecars:_FAMILY_PRIMARY_OVERRIDE": "5eccffbef645d91e",
    "eval_card_backend.canonicalise.sidecars:_METRIC_KIND_TO_GROUP": "28d0a39033f4f650",
    "eval_card_backend.canonicalise.sidecars:_METRIC_NAME_GROUP_RULES": "c055711c8b37ed4d",
}

NAME_PATTERN = re.compile(r"hotfix|override|_RULES$|_FIX(?:ES)?$|_PATCH", re.IGNORECASE)
CONSTANT_CASE = re.compile(r"^_*[A-Z][A-Z0-9_]*$")
COMMENT_PATTERN = re.compile(
    r"hotfix|override|producer-local|correction|workaround", re.IGNORECASE
)
# Blank lines allowed between a statement and the comment block above it.
MAX_COMMENT_GAP = 6


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts)


def _target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for elt in target.elts for name in _target_names(elt)]
    return []


def _assigned_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Assign):
        return [name for target in node.targets for name in _target_names(target)]
    if isinstance(node, ast.AnnAssign):
        return _target_names(node.target)
    return []


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _statements(
    body: list[ast.stmt], *, in_definition: bool = False,
) -> Iterator[tuple[ast.stmt, ast.stmt | None, bool]]:
    """Every statement in `body` and in any statement list nested under it
    (`if`/`try`/`for`/`with` branches, function and class bodies), with the
    statement that follows it and whether it sits inside a def or class."""
    for i, node in enumerate(body):
        yield node, body[i + 1] if i + 1 < len(body) else None, in_definition
        inner = in_definition or isinstance(node, _DEFINITIONS)
        for field in ("body", "orelse", "finalbody", "handlers"):
            nested = getattr(node, field, None)
            if isinstance(nested, list):
                if nested and isinstance(nested[0], ast.ExceptHandler):
                    for handler in nested:
                        yield from _statements(handler.body, in_definition=inner)
                elif nested and isinstance(nested[0], ast.stmt):
                    yield from _statements(nested, in_definition=inner)


def _first_line(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", None) or []
    return min([node.lineno] + [d.lineno for d in decorators])


def _leading_comment(lines: list[str], node: ast.stmt) -> str:
    """The `#` comment block above `node`, reached across at most
    MAX_COMMENT_GAP blank lines and ended by any other line."""
    i = _first_line(node) - 2
    block: list[str] = []
    gap = 0
    while i >= 0:
        text = lines[i].strip()
        if text.startswith("#"):
            block.insert(0, text)
            gap = 0
        elif text == "":
            gap += 1
            if gap > MAX_COMMENT_GAP:
                break
        else:
            break
        i -= 1
    return "\n".join(block)


def _docstring(node: ast.stmt, following: ast.stmt | None) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ast.get_docstring(node) or ""
    if (
        isinstance(following, ast.Expr)
        and isinstance(following.value, ast.Constant)
        and isinstance(following.value.value, str)
    ):
        return following.value.value
    return ""


def _node_hash(node: ast.stmt) -> str:
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()[:16]


def _hotfix_tables(tree: ast.Module, lines: list[str]) -> dict[str, ast.stmt]:
    found: dict[str, ast.stmt] = {}
    for node, following, in_definition in _statements(tree.body):
        names = _assigned_names(node)
        if in_definition:
            names = [name for name in names if CONSTANT_CASE.match(name)]
        if not names:
            continue
        described = COMMENT_PATTERN.search(
            _leading_comment(lines, node) + "\n" + _docstring(node, following)
        )
        for name in names:
            if described or NAME_PATTERN.search(name):
                found[name] = node
    return found


def hotfix_tables(source: str) -> dict[str, ast.stmt]:
    """`name -> assignment` for every assignment in `source` that is named,
    or introduced by its comment or docstring, as an override table."""
    return _hotfix_tables(ast.parse(source), source.splitlines())


# ---------------------------------------------------------------------------
# Lifecycle annotations inside the hotfix modules
# ---------------------------------------------------------------------------


def _public_symbol_count(tree: ast.Module) -> int:
    count = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            count += not node.name.startswith("_")
        else:
            count += sum(not n.startswith("_") for n in _assigned_names(node))
    return count


def unannotated_hotfixes(source: str) -> list[tuple[str, str]]:
    """(name, missing mark) for each public function or override table whose
    own docstring or leading comment block lacks a mark."""
    tree = ast.parse(source)
    lines = source.splitlines()
    module_doc = (ast.get_docstring(tree) or "") if _public_symbol_count(tree) == 1 else ""
    tables = _hotfix_tables(tree, lines)
    checked: list[tuple[str, ast.stmt, ast.stmt | None]] = []
    for node, following, in_definition in _statements(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not in_definition and not node.name.startswith("_"):
                checked.append((node.name, node, following))
        else:
            for name in _assigned_names(node):
                if tables.get(name) is node:
                    checked.append((name, node, following))
    problems = []
    for name, node, following in checked:
        context = "\n".join([
            _leading_comment(lines, node), _docstring(node, following), module_doc,
        ])
        for mark in (UPSTREAM_MARK, REMOVAL_MARK):
            if mark not in context:
                problems.append((name, mark))
    return problems


@pytest.mark.parametrize("module", sorted(HOTFIX_MODULES), ids=lambda m: m.rsplit(".", 1)[-1])
def test_every_hotfix_carries_a_lifecycle_annotation(module):
    problems = unannotated_hotfixes(HOTFIX_MODULES[module].read_text())
    assert not problems, (
        f"{module}: hotfixes without a complete lifecycle annotation:\n"
        + "\n".join(f"  {name}: no '{mark}' line" for name, mark in problems)
        + "\nEach hotfix needs a 'TODO(upstream):' line naming the upstream defect"
        " and a 'Remove when:' line naming what retires it, in its own docstring"
        " or in the comment block directly above it."
    )


def test_unannotated_hotfixes_finds_each_missing_half():
    source = (
        '"""Module."""\n'
        "# ── 1. Complete ──\n"
        "# TODO(upstream): x is wrong.\n"
        "# Remove when: x is right.\n"
        "_X_RULES = {}\n\n"
        "def fix_x(con):\n"
        '    """Fix x.\n\n    TODO(upstream): x is wrong.\n    Remove when: x is right.\n    """\n\n'
        "# ── 2. Only the removal ──\n"
        "# Remove when: y is right.\n"
        "def fix_y(con):\n    pass\n\n"
        "# ── 3. Only the pointer ──\n"
        "# TODO(upstream): z is wrong.\n"
        "_Z_RULES = ()\n\n"
        "_PLAIN_MAP = {}\n\n"
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


def test_a_section_header_covers_only_the_symbol_directly_below_it():
    source = (
        "# ── 1. One section, two fixes ──\n"
        "# TODO(upstream): a is wrong.\n"
        "# Remove when: a is right.\n\n\n"
        "def fix_a(con):\n    pass\n\n\n"
        "def fix_b(con):\n    pass\n"
    )
    assert unannotated_hotfixes(source) == [
        ("fix_b", UPSTREAM_MARK),
        ("fix_b", REMOVAL_MARK),
    ]


def test_a_table_between_the_header_and_the_function_breaks_the_cover():
    source = (
        "# ── 1. Section ──\n"
        "# TODO(upstream): a is wrong.\n"
        "# Remove when: a is right.\n\n"
        "_A_MAP = {}\n\n\n"
        "def fix_a(con):\n    pass\n"
    )
    assert unannotated_hotfixes(source) == [
        ("fix_a", UPSTREAM_MARK),
        ("fix_a", REMOVAL_MARK),
    ]


def test_module_docstring_covers_exactly_one_public_symbol():
    marks = "TODO(upstream): m is wrong.\nRemove when: m is right.\n"
    one = f'"""Module.\n\n{marks}"""\n\ndef fix_m(con):\n    pass\n'
    assert unannotated_hotfixes(one) == []
    two = one + "\ndef fix_n(con):\n    pass\n"
    assert unannotated_hotfixes(two) == [
        ("fix_m", UPSTREAM_MARK), ("fix_m", REMOVAL_MARK),
        ("fix_n", UPSTREAM_MARK), ("fix_n", REMOVAL_MARK),
    ]


# ---------------------------------------------------------------------------
# Ratchet: no new override tables outside the hotfix modules
# ---------------------------------------------------------------------------


def tables_outside_hotfix_modules() -> dict[str, str]:
    """`module:symbol -> source hash` over the package, hotfix modules excluded."""
    hotfix_paths = set(HOTFIX_MODULES.values())
    found: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        if path in hotfix_paths:
            continue
        module = _module_name(path)
        for name, node in hotfix_tables(path.read_text()).items():
            found[f"{module}:{name}"] = _node_hash(node)
    return found


@pytest.mark.parametrize(
    "source,expected",
    [
        pytest.param(
            "# Producer-local correction for a mislabelled feed.\n\n"
            "_BAR_MAP: dict[str, str] = {'a': 'b'}\n",
            ["_BAR_MAP"], id="neutral-name-comment-then-blank-line",
        ),
        pytest.param("_FOO_OVERRIDE = _A | _B\n", ["_FOO_OVERRIDE"], id="binop"),
        pytest.param(
            "_FOO_OVERRIDE = {k: v for k, v in _A.items()}\n",
            ["_FOO_OVERRIDE"], id="dict-comprehension",
        ),
        pytest.param(
            "_FOO_OVERRIDE = types.MappingProxyType({})\n",
            ["_FOO_OVERRIDE"], id="mapping-proxy",
        ),
        pytest.param(
            "_FOO_OVERRIDE = json.loads('{}')\n", ["_FOO_OVERRIDE"], id="json-loads",
        ),
        pytest.param(
            "if sys.platform:\n    _FOO_OVERRIDE = {}\n", ["_FOO_OVERRIDE"], id="under-if",
        ),
        pytest.param(
            "try:\n    import x\nexcept ImportError:\n    _FOO_OVERRIDE = {}\n",
            ["_FOO_OVERRIDE"], id="under-except",
        ),
        pytest.param(
            "_X_OVERRIDE, _Y_OVERRIDE = {}, {}\n",
            ["_X_OVERRIDE", "_Y_OVERRIDE"], id="tuple-unpacking",
        ),
        pytest.param(
            "_X_OVERRIDE = _Y_OVERRIDE = {}\n",
            ["_X_OVERRIDE", "_Y_OVERRIDE"], id="chained-targets",
        ),
        pytest.param(
            "def f():\n    _LOCAL_OVERRIDE = {}\n    return _LOCAL_OVERRIDE\n",
            ["_LOCAL_OVERRIDE"], id="inside-a-function",
        ),
        pytest.param(
            "def f(x):\n    # The override wins.\n    override = x.get('o')\n    return override\n",
            [], id="local-variable-inside-a-function",
        ),
        pytest.param(
            "_BAZ_RULES = ()\n_QUX_FIX = ()\n_QUUX_FIXES = ()\n_CORGE_PATCH = ()\n",
            ["_BAZ_RULES", "_QUX_FIX", "_QUUX_FIXES", "_CORGE_PATCH"], id="name-suffixes",
        ),
        pytest.param(
            "_BAR_MAP = {}\n\"\"\"Workaround for the feed's spelling.\"\"\"\n",
            ["_BAR_MAP"], id="attribute-docstring",
        ),
        pytest.param(
            "# Unrelated comment, well above.\n" + "\n" * 7 + "_BAR_MAP = {}\n",
            [], id="comment-beyond-the-gap",
        ),
        pytest.param(
            "import re\n_PLAIN = {}\n_QUX_SET = frozenset({'override'})\n"
            "# Hotfix tables live elsewhere.\nimport os\n_AFTER = {}\n",
            [], id="plain-names-and-contents",
        ),
    ],
)
def test_hotfix_tables_detection(source, expected):
    assert list(hotfix_tables(source)) == expected


def test_hotfix_tables_hash_changes_with_the_content():
    before = hotfix_tables("_FOO_OVERRIDE = {'a': 1}\n")["_FOO_OVERRIDE"]
    after = hotfix_tables("_FOO_OVERRIDE = {'a': 1, 'b': 2}\n")["_FOO_OVERRIDE"]
    same = hotfix_tables("# a comment\n_FOO_OVERRIDE = {'a': 1}\n")["_FOO_OVERRIDE"]
    assert _node_hash(before) != _node_hash(after)
    assert _node_hash(before) == _node_hash(same)


def test_no_new_hotfix_tables_outside_the_hotfix_modules():
    found = tables_outside_hotfix_modules()
    new = sorted(found.keys() - ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES.keys())
    assert not new, (
        "override tables outside the hotfix modules:\n"
        + "\n".join(f"  {symbol}" for symbol in new)
        + "\nPut the table in canonicalise/resolution_hotfixes.py or"
        " canonicalise/hierarchy_hotfixes.py with a 'TODO(upstream):' line"
        " and a 'Remove when:' line directly above it."
    )


def test_the_allowlist_only_shrinks():
    gone = sorted(ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES.keys() - tables_outside_hotfix_modules().keys())
    assert not gone, (
        "grandfathered tables no longer found:\n"
        + "\n".join(f"  {symbol}" for symbol in gone)
        + "\nRemove them from ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES so the list keeps shrinking."
    )


def test_grandfathered_tables_are_unchanged():
    found = tables_outside_hotfix_modules()
    changed = [
        f'    "{symbol}": "{found[symbol]}",'
        for symbol, recorded in sorted(ALLOWED_TABLES_OUTSIDE_HOTFIX_MODULES.items())
        if symbol in found and found[symbol] != recorded
    ]
    assert not changed, (
        "grandfathered tables changed content. They are frozen: the fix goes"
        " upstream or into a hotfix module. If the change is deliberate,"
        " update the entries to:\n"
        + "\n".join(changed)
    )

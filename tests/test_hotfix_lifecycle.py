"""Hotfix lifecycle gate.

Data hotfixes live in three designated modules. Every public function there
is a hotfix unless NOT_HOTFIXES names it as plumbing, and a hotfix carries
its lifecycle annotation in its own docstring and nowhere else: a
`TODO(upstream):` line naming the upstream defect and a `Remove when:` line
naming the condition that retires it. A mark in a comment or in the module
docstring is reported so that the docstring stays the single annotation
site. An override table in those modules is the hotfix's data and carries no
annotation of its own; a table no function in the module reads is reported,
since no hotfix then owns it.

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
import io
import re
import tokenize
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
MARKS = (UPSTREAM_MARK, REMOVAL_MARK)

# `module:function` for each public function in a hotfix module that is
# not a hotfix and so carries no lifecycle annotation.
NOT_HOTFIXES = frozenset({
    # Provenance-counter plumbing that pipeline.py calls around a bake.
    "eval_card_backend.metric_meta_hotfix:reset_provenance_counter",
    "eval_card_backend.metric_meta_hotfix:log_metric_meta_summary",
})

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


def _plumbing(module: str) -> frozenset[str]:
    prefix = module + ":"
    return frozenset(s[len(prefix):] for s in NOT_HOTFIXES if s.startswith(prefix))


def _marks_outside_docstrings(source: str, tree: ast.Module) -> Iterator[tuple[int, str]]:
    """(line, mark) for each lifecycle mark in a comment or the module docstring."""
    module_doc = tree.body[0] if ast.get_docstring(tree) else None
    if module_doc is not None:
        for mark in MARKS:
            if mark in ast.get_docstring(tree):
                yield module_doc.lineno, mark
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            for mark in MARKS:
                if mark in tok.string:
                    yield tok.start[0], mark


def _read_names(tree: ast.Module) -> set[str]:
    """Every name some function in the module reads."""
    return {
        node.id
        for top in tree.body if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(top)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def unannotated_hotfixes(
    source: str, *, plumbing: frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    """(where, problem) for each hotfix whose docstring lacks a mark, each
    mark outside a docstring, and each override table nothing reads.
    `plumbing` names the public functions that are not hotfixes."""
    tree = ast.parse(source)
    lines = source.splitlines()
    problems: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") or node.name in plumbing:
            continue
        docstring = ast.get_docstring(node) or ""
        comment = _leading_comment(lines, node)
        for mark in MARKS:
            if mark in docstring:
                continue
            where = "in the comment above, not" if mark in comment else "missing from"
            problems.append((node.name, f"'{mark}' {where} the docstring"))
    for lineno, mark in _marks_outside_docstrings(source, tree):
        problems.append((f"line {lineno}", f"'{mark}' outside a hotfix docstring"))
    read = _read_names(tree)
    for name in _hotfix_tables(tree, lines):
        if name not in read:
            problems.append((name, "no function in the module reads this table"))
    return problems


@pytest.mark.parametrize("module", sorted(HOTFIX_MODULES), ids=lambda m: m.rsplit(".", 1)[-1])
def test_every_hotfix_carries_a_lifecycle_annotation(module):
    problems = unannotated_hotfixes(
        HOTFIX_MODULES[module].read_text(), plumbing=_plumbing(module),
    )
    assert not problems, (
        f"{module}: lifecycle annotation problems:\n"
        + "\n".join(f"  {where}: {problem}" for where, problem in problems)
        + "\nEach hotfix needs a 'TODO(upstream):' line naming the upstream defect"
        " and a 'Remove when:' line naming what retires it, in its own docstring"
        " and nowhere else. A public function that is not a hotfix goes in"
        " NOT_HOTFIXES."
    )


def test_not_hotfixes_names_public_functions_that_exist():
    for symbol in NOT_HOTFIXES:
        module, name = symbol.split(":")
        tree = ast.parse(HOTFIX_MODULES[module].read_text())
        assert name in {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        }, f"{symbol} is not a public function; drop it from NOT_HOTFIXES"


def test_a_docstring_annotation_counts():
    source = (
        "def fix_w(con):\n"
        '    """Fix w.\n\n    TODO(upstream): w is wrong.\n    Remove when: w is right.\n    """\n'
    )
    assert unannotated_hotfixes(source) == []


def test_unannotated_hotfixes_finds_each_missing_half():
    source = (
        "def fix_y(con):\n"
        '    """Fix y.\n\n    Remove when: y is right.\n    """\n\n'
        "def fix_z(con):\n"
        '    """Fix z.\n\n    TODO(upstream): z is wrong.\n    """\n\n'
        "def fix_v(con):\n    pass\n"
    )
    assert unannotated_hotfixes(source) == [
        ("fix_y", "'TODO(upstream):' missing from the docstring"),
        ("fix_z", "'Remove when:' missing from the docstring"),
        ("fix_v", "'TODO(upstream):' missing from the docstring"),
        ("fix_v", "'Remove when:' missing from the docstring"),
    ]


def test_marks_in_the_section_comment_alone_say_to_move_them():
    source = (
        "# ── 1. Section ──\n"
        "# TODO(upstream): a is wrong.\n"
        "# Remove when: a is right.\n\n\n"
        'def fix_a(con):\n    """Fix a."""\n'
    )
    assert unannotated_hotfixes(source) == [
        ("fix_a", "'TODO(upstream):' in the comment above, not the docstring"),
        ("fix_a", "'Remove when:' in the comment above, not the docstring"),
        ("line 2", "'TODO(upstream):' outside a hotfix docstring"),
        ("line 3", "'Remove when:' outside a hotfix docstring"),
    ]


def test_a_mark_duplicated_in_a_comment_or_the_module_docstring_is_reported():
    annotated = (
        "def fix_a(con):\n"
        '    """Fix a.\n\n    TODO(upstream): a is wrong.\n    Remove when: a is right.\n    """\n'
    )
    with_comment = "# Remove when: a is right.\n" + annotated
    assert unannotated_hotfixes(with_comment) == [
        ("line 1", "'Remove when:' outside a hotfix docstring"),
    ]
    with_module_doc = '"""Module.\n\nTODO(upstream): a is wrong.\n"""\n' + annotated
    assert unannotated_hotfixes(with_module_doc) == [
        ("line 1", "'TODO(upstream):' outside a hotfix docstring"),
    ]
    inline = annotated + "    x = 1  # TODO(upstream): also here\n"
    assert unannotated_hotfixes(inline) == [
        ("line 7", "'TODO(upstream):' outside a hotfix docstring"),
    ]


def test_private_helpers_and_listed_plumbing_pass_and_other_public_functions_fail():
    source = (
        "def _helper():\n    pass\n\n"
        "def reset_counter():\n    pass\n\n"
        "def log_summary(log):\n    pass\n"
    )
    assert unannotated_hotfixes(source, plumbing=frozenset({"reset_counter"})) == [
        ("log_summary", "'TODO(upstream):' missing from the docstring"),
        ("log_summary", "'Remove when:' missing from the docstring"),
    ]


def test_a_table_is_covered_by_the_function_that_reads_it():
    table = "_X_RULES = {}\n\n"
    reader = (
        "def fix_x(con):\n"
        '    """Fix x.\n\n    TODO(upstream): x is wrong.\n    Remove when: x is right.\n    """\n'
        "    return _X_RULES\n"
    )
    assert unannotated_hotfixes(table + reader) == []
    assert unannotated_hotfixes(table + reader.replace("return _X_RULES", "pass")) == [
        ("_X_RULES", "no function in the module reads this table"),
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
        " canonicalise/hierarchy_hotfixes.py beside the hotfix function that"
        " reads it; that function's docstring carries the 'TODO(upstream):'"
        " and 'Remove when:' lines."
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

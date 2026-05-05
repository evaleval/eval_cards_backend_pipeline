"""CI helper: swap the workspace path resolver dep for a pinned git URL.

Local dev installs `eval-entity-resolver` from a sibling repo via
`[tool.uv.sources]`'s `path = "../eval-card-registry/..."`. CI runners
don't have that sibling, so this script rewrites the source to a git URL
in-place before `uv sync` runs. `RESOLVER_REF` env var pins the commit;
defaults to `main`.

Usage:
    RESOLVER_REF=<sha> python scripts/ci_install_resolver.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


REGISTRY_REPO = "https://github.com/evaleval/evalcard-registry.git"
REGISTRY_SUBDIR = "packages/eval-entity-resolver"
PYPROJECT = Path("pyproject.toml")


def main() -> int:
    ref = os.environ.get("RESOLVER_REF", "main")
    text = PYPROJECT.read_text()

    new_source = (
        'eval-entity-resolver = { '
        f'git = "{REGISTRY_REPO}", '
        f'rev = "{ref}", '
        f'subdirectory = "{REGISTRY_SUBDIR}" '
        '}'
    )
    workspace_pattern = re.compile(
        r"^eval-entity-resolver = \{[^\n}]*path[^\n}]*\}$",
        flags=re.MULTILINE,
    )
    git_pattern = re.compile(
        r"^eval-entity-resolver = \{[^\n}]*\bgit\s*=[^\n}]*\}$",
        flags=re.MULTILINE,
    )

    if workspace_pattern.search(text):
        new_text = workspace_pattern.sub(new_source, text, count=1)
        PYPROJECT.write_text(new_text)
        print(f"Pinned eval-entity-resolver to {ref} for CI.")
        return 0
    if git_pattern.search(text):
        # Already pinned (e.g., a re-run of the same job). Idempotent no-op.
        print("eval-entity-resolver is already pinned to a git URL; no-op.")
        return 0

    print(
        "FATAL: eval-entity-resolver source not found in pyproject.toml. "
        "Either the workspace-path source was reformatted (multi-line table?) "
        "or removed entirely. CI cannot continue without a resolvable source.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

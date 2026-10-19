"""`.github/workflows/ci.yml`: its triggers, and dependency setup identical to
`sync.yml`'s so the two cannot drift apart silently.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

# sync.yml steps that ci.yml copies verbatim, in order.
SHARED_SETUP_STEPS = (
    "Resolve upstream revisions",
    "Check out eval-entity-registry (for taxonomy seed YAMLs)",
    "Check the registry checkout has the seed and the resolver",
    "Install uv",
    "Configure resolver dep for CI",
    "Sync dependencies",
)
PRE_COMMIT_STEP = "Pre-commit hooks"
HYGIENE_PR_STEP = "Code hygiene on the pull request's added lines"
HYGIENE_PUSH_STEP = "Code hygiene on the pushed commits' added lines"
TESTS_STEP = "Run tests"


@pytest.fixture(scope="module")
def ci_workflow():
    return yaml.safe_load((WORKFLOWS / "ci.yml").read_text())


@pytest.fixture(scope="module")
def sync_workflow():
    return yaml.safe_load((WORKFLOWS / "sync.yml").read_text())


@pytest.fixture(scope="module")
def pre_commit_config():
    return yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())


def _steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def _index(steps, **match):
    hits = [
        i for i, step in enumerate(steps)
        if all(step.get(k) == v for k, v in match.items())
    ]
    assert len(hits) == 1, f"expected one step matching {match}, got {len(hits)}"
    return hits[0]


def test_workflow_runs_on_pull_requests_and_pushes_to_main(ci_workflow):
    # PyYAML reads the bare `on:` key as boolean True.
    triggers = ci_workflow[True]
    assert set(triggers) == {"pull_request", "push"}
    assert triggers["push"] == {"branches": ["main"]}


def test_workflow_job_shape_concurrency_and_permissions(ci_workflow):
    assert set(ci_workflow["jobs"]) == {"checks"}
    job = ci_workflow["jobs"]["checks"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 20
    assert ci_workflow["concurrency"] == {
        "group": "ci-${{ github.ref }}",
        "cancel-in-progress": True,
    }
    assert ci_workflow["permissions"] == {"contents": "read"}


def test_workflow_checks_out_full_history_first(ci_workflow):
    first = _steps(ci_workflow, "checks")[0]
    assert first["uses"] == "actions/checkout@v6"
    assert first["with"] == {"fetch-depth": 0}


def test_workflow_setup_steps_are_identical_to_sync_workflow(ci_workflow, sync_workflow):
    ci_steps = _steps(ci_workflow, "checks")
    sync_steps = _steps(sync_workflow, "run-pipeline")
    positions = []
    for name in SHARED_SETUP_STEPS:
        i = _index(ci_steps, name=name)
        assert ci_steps[i] == sync_steps[_index(sync_steps, name=name)], name
        positions.append(i)
    assert positions == sorted(positions)
    assert positions[-1] < _index(ci_steps, name=PRE_COMMIT_STEP)


def test_workflow_runs_hooks_then_hygiene_diff_then_tests(ci_workflow):
    steps = _steps(ci_workflow, "checks")
    hooks = _index(steps, name=PRE_COMMIT_STEP)
    hygiene_pr = _index(steps, name=HYGIENE_PR_STEP)
    hygiene_push = _index(steps, name=HYGIENE_PUSH_STEP)
    tests = _index(steps, name=TESTS_STEP)
    assert hooks < hygiene_pr < hygiene_push < tests

    assert steps[hooks]["run"] == "uv run pre-commit run --all-files --show-diff-on-failure"
    assert "if" not in steps[hooks]

    assert steps[hygiene_pr]["if"] == "github.event_name == 'pull_request'"
    assert steps[hygiene_pr]["run"] == (
        "python3 scripts/check_code_hygiene.py "
        "--diff origin/${{ github.base_ref || 'main' }}...HEAD"
    )

    assert steps[hygiene_push]["if"] == "github.event_name == 'push'"
    assert steps[hygiene_push]["env"] == {"BEFORE": "${{ github.event.before }}"}
    push_run = steps[hygiene_push]["run"]
    assert 'range="$BEFORE...${{ github.sha }}"' in push_run
    assert '"$BEFORE" = "0000000000000000000000000000000000000000"' in push_run
    assert 'range="HEAD~1...HEAD"' in push_run
    assert push_run.rstrip().endswith('python3 scripts/check_code_hygiene.py --diff "$range"')

    assert steps[tests]["run"] == "uv run pytest tests/ -q --timeout=120"
    assert steps[tests]["env"] == {"HF_HUB_OFFLINE": "1"}
    assert "if" not in steps[tests]


def test_pre_commit_config_declares_the_expected_hooks(pre_commit_config):
    hooks = {
        hook["id"]: (repo["repo"], hook)
        for repo in pre_commit_config["repos"]
        for hook in repo["hooks"]
    }
    assert set(hooks) == {
        "ruff-check",
        "check-yaml",
        "check-toml",
        "check-merge-conflict",
        "check-added-large-files",
        "trailing-whitespace",
        "code-hygiene",
    }
    repo, hygiene = hooks["code-hygiene"]
    assert repo == "local"
    assert hygiene == {
        **hygiene,
        "entry": "python3 scripts/check_code_hygiene.py",
        "language": "system",
        "pass_filenames": False,
        "always_run": True,
        "stages": ["pre-commit"],
    }
    assert "args" not in hooks["ruff-check"][1]


def test_pre_commit_ruff_rev_matches_the_locked_ruff_version(pre_commit_config):
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = [pkg["version"] for pkg in lock["package"] if pkg["name"] == "ruff"]
    assert len(locked) == 1
    ruff_repo = next(
        repo for repo in pre_commit_config["repos"]
        if repo["repo"] == "https://github.com/astral-sh/ruff-pre-commit"
    )
    assert ruff_repo["rev"] == f"v{locked[0]}"


def test_pre_commit_is_a_dev_dependency():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = pyproject["dependency-groups"]["dev"]
    assert any(dep.split(">=")[0].strip() == "pre-commit" for dep in dev)

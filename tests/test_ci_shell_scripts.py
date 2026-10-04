"""`scripts/ci_resolve_revisions.sh` and `scripts/ci_sync_alert.sh`.

Both scripts run for real under bash with fake `gh` and `curl` executables
first on PATH. The fakes answer from files in a fixture dir and append
every invocation to a log, so the tests assert on the exact commands the
scripts would run and never touch the network. The last section checks the
wiring of `sync.yml` itself.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="needs bash and jq",
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REPO = "org/pipeline"
RUN_ID = "1001"
RUN_URL = f"https://github.com/{REPO}/actions/runs/{RUN_ID}"
SHA_EEE = "a" * 40
SHA_REGISTRY = "b" * 40
SHA_SEED = "c" * 40

# The fakes accept only the exact command lines the scripts are meant to
# send. Anything else (a dropped --paginate, a wrong repo or artifact name,
# an unknown URL) exits non-zero with a message, which fails the script.
FAKE_GH = r"""#!/usr/bin/env python3
import json, os, re, shutil, sys
from pathlib import Path

argv = sys.argv[1:]
fix = Path(os.environ["FAKE_FIXTURES"])
repo = os.environ["FAKE_REPO"]
run_id = os.environ["FAKE_RUN_ID"]
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(" ".join(argv) + "\n")


def die(msg):
    sys.stderr.write(f"fake gh: {msg}: {argv}\n")
    sys.exit(1)


def emit(name):
    path = fix / name
    if not path.exists():
        die(f"no fixture {name}")
    sys.stdout.write(path.read_text())


def emit_pages(name, paginate):
    path = fix / name
    if not path.exists():
        die(f"no fixture {name}")
    pages = json.loads(path.read_text())
    for page in pages if paginate else pages[:1]:
        print(json.dumps(page))


if argv[:1] == ["api"]:
    rest = argv[1:]
    paginate = rest[:1] == ["--paginate"]
    if paginate:
        rest = rest[1:]
    if len(rest) != 1:
        die("unexpected api arguments")
    path = rest[0]
    base = f"repos/{repo}"
    comments = re.fullmatch(re.escape(base) + r"/issues/(\d+)/comments\?per_page=100", path)
    if path == f"{base}/issues?labels=sync-alert&state=all&per_page=100":
        emit_pages("issues.json", paginate)
    elif comments:
        emit_pages(f"comments_{comments.group(1)}.json", paginate)
    elif paginate:
        die("--paginate on a path that is not a list")
    elif path == f"{base}/actions/runs/{run_id}":
        emit("run.json")
    elif path == f"{base}/actions/runs/{run_id}/jobs?per_page=100":
        emit("jobs.json")
    elif path == f"{base}/actions/jobs/55/logs":
        emit("job.log")
    elif path == f"{base}/actions/workflows/sync.yml/runs?branch=main&status=completed&per_page=1":
        emit("newest_runs.json")
    else:
        die("unexpected api path")
elif argv[:2] == ["run", "download"]:
    if len(argv) != 9 or argv[2:7] != [run_id, "-R", repo, "-n", "snapshot-meta"] or argv[7] != "-D":
        die("unexpected run download arguments")
    src = fix / "snapshot_meta.json"
    if not src.exists():
        sys.exit(1)
    dest = Path(argv[8]) / "2026-10-04T00:00:00Z"
    dest.mkdir(parents=True)
    shutil.copy(src, dest / "snapshot_meta.json")
elif argv[:2] == ["label", "create"]:
    if argv[2:5] != ["sync-alert", "-R", repo]:
        die("unexpected label create arguments")
elif argv[:2] == ["issue", "create"]:
    if (len(argv) != 10 or argv[2:4] != ["-R", repo] or argv[4] != "--title"
            or argv[6:8] != ["--label", "sync-alert"] or argv[8] != "--body"):
        die("unexpected issue create arguments")
    print(f"https://github.com/{repo}/issues/7")
elif argv[:2] == ["issue", "comment"]:
    if len(argv) != 7 or not argv[2].isdigit() or argv[3:5] != ["-R", repo] or argv[5] != "--body-file":
        die("unexpected issue comment arguments")
    shutil.copy(argv[6], os.environ["FAKE_BODY"])
elif argv[:2] == ["issue", "close"]:
    if len(argv) != 5 or not argv[2].isdigit() or argv[3:5] != ["-R", repo]:
        die("unexpected issue close arguments")
else:
    die("unexpected command")
"""

FAKE_CURL = r"""#!/usr/bin/env python3
import os, sys
from pathlib import Path

argv = sys.argv[1:]
url = argv[-1]
flags = argv[:-1]
fix = Path(os.environ["FAKE_FIXTURES"])
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(url + "\n")


def die(msg, code=1):
    sys.stderr.write(f"fake curl: {msg}: {argv}\n")
    sys.exit(code)


for flag in ("--fail", "--silent", "--show-error", "--location", "--retry", "--max-time"):
    if flag not in flags:
        die(f"missing {flag}")
if "Authorization" in " ".join(flags):
    die("unexpected Authorization header (no token in the test env)")

hf = "https://huggingface.co"
registry_sha = os.environ.get("FAKE_REGISTRY_SHA", "")
routes = {
    f"{hf}/api/datasets/evaleval/EEE_datastore?expand[]=sha": "eee_head.json",
    f"{hf}/api/datasets/evaleval/entity-registry-data?expand[]=sha": "registry_head.json",
    f"{hf}/datasets/evaleval/entity-registry-data/resolve/{registry_sha}/manifest.json": "manifest.json",
    f"{hf}/datasets/evaleval/card_backend/resolve/main/warehouse/latest/snapshot_meta.json": "published_meta.json",
}
if url not in routes:
    die("unexpected url")
if "[" in url and "--globoff" not in flags:
    die("bracketed url without --globoff")
path = fix / routes[url]
if not path.exists():
    die("404", code=22)
sys.stdout.write(path.read_text())
"""

WRITE_PREFIXES = ("issue create", "issue comment", "issue close", "label create")


class Harness:
    def __init__(self, tmp_path: Path):
        self.fixtures = tmp_path / "fixtures"
        self.fixtures.mkdir()
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.log = tmp_path / "calls.log"
        self.body = tmp_path / "comment_body.md"
        self.github_env = tmp_path / "github_env"
        self.github_output = tmp_path / "github_output"
        self.summary = tmp_path / "summary.md"
        for name, text in (("gh", FAKE_GH), ("curl", FAKE_CURL)):
            exe = self.bin / name
            exe.write_text(text)
            exe.chmod(0o755)

    def put(self, name: str, content) -> None:
        text = content if isinstance(content, str) else json.dumps(content)
        (self.fixtures / name).write_text(text)

    def run(self, script: str, *args: str, env: dict | None = None):
        full_env = {
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_LOG": str(self.log),
            "FAKE_BODY": str(self.body),
            "FAKE_FIXTURES": str(self.fixtures),
            "FAKE_REPO": REPO,
            "FAKE_RUN_ID": RUN_ID,
            "FAKE_REGISTRY_SHA": SHA_REGISTRY,
            "GITHUB_REPOSITORY": REPO,
            "GITHUB_ENV": str(self.github_env),
            "GITHUB_OUTPUT": str(self.github_output),
            "GITHUB_STEP_SUMMARY": str(self.summary),
            **(env or {}),
        }
        return subprocess.run(
            ["bash", str(SCRIPTS / script), *args],
            env=full_env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def writes(self) -> list[str]:
        return [c for c in self.calls() if c.startswith(WRITE_PREFIXES)]

    def posted_body(self) -> str:
        return self.body.read_text()


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


# ---------------------------------------------------------------------------
# ci_resolve_revisions.sh
# ---------------------------------------------------------------------------


def _resolve_fixtures(h, *, manifest=None):
    h.put("eee_head.json", {"id": "evaleval/EEE_datastore", "sha": SHA_EEE})
    h.put("registry_head.json", {"id": "evaleval/entity-registry-data", "sha": SHA_REGISTRY})
    h.put("manifest.json", {"seed_git_sha": SHA_SEED} if manifest is None else manifest)


def test_resolve_follows_heads_when_nothing_is_pinned(h):
    _resolve_fixtures(h)
    proc = h.run("ci_resolve_revisions.sh")
    assert proc.returncode == 0, proc.stderr
    assert h.github_env.read_text().splitlines() == [
        f"EEE_REVISION={SHA_EEE}",
        f"ENTITY_REGISTRY_REVISION={SHA_REGISTRY}",
        f"RESOLVER_REF={SHA_SEED}",
    ]
    assert h.github_output.read_text().splitlines() == [
        f"eee_revision={SHA_EEE}",
        f"entity_registry_revision={SHA_REGISTRY}",
        f"resolver_ref={SHA_SEED}",
    ]
    assert h.calls() == [
        "https://huggingface.co/api/datasets/evaleval/EEE_datastore?expand[]=sha",
        "https://huggingface.co/api/datasets/evaleval/entity-registry-data?expand[]=sha",
        "https://huggingface.co/datasets/evaleval/entity-registry-data"
        f"/resolve/{SHA_REGISTRY}/manifest.json",
    ]
    summary = h.summary.read_text()
    assert summary.count("followed head") == 2
    assert "pinned" not in summary


def test_resolve_pins_skip_head_lookups_and_derive_resolver_from_pinned_manifest(h):
    _resolve_fixtures(h)
    pin_eee, pin_registry = "1" * 40, "2" * 40
    proc = h.run(
        "ci_resolve_revisions.sh",
        env={
            "PINNED_EEE_REVISION": pin_eee,
            "PINNED_ENTITY_REGISTRY_REVISION": pin_registry,
            "FAKE_REGISTRY_SHA": pin_registry,
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert h.github_env.read_text().splitlines() == [
        f"EEE_REVISION={pin_eee}",
        f"ENTITY_REGISTRY_REVISION={pin_registry}",
        f"RESOLVER_REF={SHA_SEED}",
    ]
    assert h.calls() == [
        "https://huggingface.co/datasets/evaleval/entity-registry-data"
        f"/resolve/{pin_registry}/manifest.json",
    ]
    assert h.summary.read_text().count("pinned (repo variable") == 2


@pytest.mark.parametrize(
    "manifest",
    [{"content_hash": "x"}, {"seed_git_sha": "main"}, {"seed_git_sha": None}],
)
def test_resolve_fails_when_manifest_has_no_usable_seed_git_sha(h, manifest):
    _resolve_fixtures(h, manifest=manifest)
    proc = h.run("ci_resolve_revisions.sh")
    assert proc.returncode != 0
    assert "seed_git_sha" in proc.stderr
    assert not h.github_env.exists() or h.github_env.read_text() == ""


@pytest.mark.parametrize(
    "var", ["PINNED_EEE_REVISION", "PINNED_ENTITY_REGISTRY_REVISION"]
)
@pytest.mark.parametrize("value", ["main", "0fe996f", "A" * 40, "a" * 41])
def test_resolve_rejects_a_malformed_pin_before_any_request(h, var, value):
    _resolve_fixtures(h)
    proc = h.run("ci_resolve_revisions.sh", env={var: value})
    assert proc.returncode != 0
    assert "not a 40-hex commit sha" in proc.stderr
    assert h.calls() == []
    assert not h.github_env.exists() or h.github_env.read_text() == ""


def test_resolve_fails_when_a_head_lookup_returns_no_sha(h):
    _resolve_fixtures(h)
    h.put("eee_head.json", {"id": "evaleval/EEE_datastore"})
    proc = h.run("ci_resolve_revisions.sh")
    assert proc.returncode != 0
    assert not h.github_env.exists() or h.github_env.read_text() == ""


# ---------------------------------------------------------------------------
# ci_sync_alert.sh
# ---------------------------------------------------------------------------


PUBLISH_STEP = "Publish warehouse snapshot to HF dataset"
CHECK_ONLY_STEP = "Compare snapshot with the published one (publish is off)"


def _alert_fixtures(
    h,
    *,
    conclusion="success",
    open_issue=None,
    comments=(),
    closed_issues=None,
    meta="default",
    newest=RUN_ID,
    published=True,
):
    """`comments` go on the open issue. `closed_issues` maps a closed issue
    number to its comment bodies."""
    h.put(
        "run.json",
        {"id": int(RUN_ID), "conclusion": conclusion, "event": "schedule", "html_url": RUN_URL},
    )
    issues = []
    if open_issue:
        issues.append({"number": open_issue, "state": "open"})
        h.put(f"comments_{open_issue}.json", [[{"body": b} for b in comments]])
    for number, bodies in (closed_issues or {}).items():
        issues.append({"number": number, "state": "closed"})
        h.put(f"comments_{number}.json", [[{"body": b} for b in bodies]])
    h.put("issues.json", [issues])
    h.put("newest_runs.json", {"workflow_runs": [{"id": int(newest)}]})
    failed = conclusion != "success"
    steps = [
        {"name": "Run tests", "conclusion": "success"},
        {"name": "Run canonicalisation", "conclusion": "failure" if failed else "success"},
        {
            "name": PUBLISH_STEP,
            "conclusion": "success" if published and not failed else "skipped",
        },
        {
            "name": CHECK_ONLY_STEP,
            "conclusion": "skipped" if published or failed else "success",
        },
    ]
    h.put(
        "jobs.json",
        {
            "jobs": [
                {
                    "id": 55,
                    "name": "run-pipeline",
                    "conclusion": "failure" if failed else "success",
                    "steps": steps,
                }
            ]
        },
    )
    log_lines = [f"2026-10-04T11:30:{i % 60:02d}.0000000Z line {i}" for i in range(1, 61)]
    log_lines.append("2026-10-04T11:31:00.0000000Z ##[error]Process completed with exit code 1.")
    log_lines += [f"2026-10-04T11:31:01.0000000Z cleanup {i}" for i in range(5)]
    h.put("job.log", "\n".join(log_lines) + "\n")
    if meta == "default":
        meta = _snapshot_meta()
    if meta is not None:
        h.put("snapshot_meta.json", meta)


def _snapshot_meta(drops=()):
    return {
        "snapshot_id": "2026-10-04T00:00:00Z",
        "upstream_pins": {
            "eee_datastore": {"sha": SHA_EEE},
            "entity_registry": {"sha": SHA_REGISTRY},
        },
        "row_counts": {"dropped_eee_records_stage_a": sum(d["count"] for d in drops)},
        "stage_a_drops": list(drops),
    }


CREATE_ISSUE = [
    f"label create sync-alert -R {REPO} --color D93F0B "
    "--description Sync Pipeline failures, drops and staleness",
]


def _assert_created_then_commented(h):
    writes = h.writes()
    assert writes[0] == CREATE_ISSUE[0]
    assert writes[1].startswith(
        f"issue create -R {REPO} --title Sync Pipeline alert --label sync-alert --body "
    )
    assert writes[2].startswith(f"issue comment 7 -R {REPO} --body-file ")
    assert len(writes) == 3


def _assert_only_comment(h, issue):
    writes = h.writes()
    assert len(writes) == 1
    assert writes[0].startswith(f"issue comment {issue} -R {REPO} --body-file ")


@pytest.mark.parametrize(
    "conclusion", ["failure", "timed_out", "startup_failure", "action_required"]
)
def test_alert_failure_creates_issue_and_comments(h, conclusion):
    _alert_fixtures(h, conclusion=conclusion, meta=None)
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    _assert_created_then_commented(h)

    body = h.posted_body()
    lines = body.splitlines()
    assert lines[0] == f"<!-- sync-alert:failure:{RUN_ID} -->"
    assert f"(conclusion: `{conclusion}`)" in lines[1]
    assert f"- Run: {RUN_URL}" in lines
    assert "- Event: `schedule`" in lines
    assert "- Failed steps: `run-pipeline / Run canonicalisation`" in lines
    assert "- Upstream revisions: not available (the run wrote no snapshot_meta.json)" in lines
    # 40 log lines ending at the error line, timestamps stripped, no cleanup.
    start = lines.index("````") + 1
    log_tail = lines[start : lines.index("````", start)]
    assert len(log_tail) == 40
    assert log_tail[0] == "line 22"
    assert log_tail[-2] == "line 60"
    assert log_tail[-1] == "##[error]Process completed with exit code 1."


def test_alert_failure_comments_on_the_open_issue_with_revisions(h):
    _alert_fixtures(h, conclusion="failure", open_issue=3)
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    _assert_only_comment(h, 3)
    assert (
        f"- Upstream revisions: eee_datastore `{SHA_EEE}`, entity_registry `{SHA_REGISTRY}`"
        in h.posted_body().splitlines()
    )


def test_alert_duplicate_marker_posts_nothing(h):
    _alert_fixtures(
        h,
        conclusion="failure",
        open_issue=3,
        comments=["unrelated", f"<!-- sync-alert:failure:{RUN_ID} -->\nearlier delivery"],
    )
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []
    assert "already has" in proc.stdout


def test_alert_marker_on_a_closed_issue_suppresses_a_replay(h):
    """A failure already reported in a thread that has since been closed must
    not open a second issue when it is replayed or delivered late."""
    _alert_fixtures(
        h,
        conclusion="failure",
        closed_issues={2: [f"<!-- sync-alert:failure:{RUN_ID} -->\nreported"]},
    )
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []
    assert "already has" in proc.stdout


def test_alert_closed_issue_without_the_marker_does_not_suppress(h):
    _alert_fixtures(
        h,
        conclusion="failure",
        closed_issues={2: ["<!-- sync-alert:failure:999 -->"]},
    )
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    _assert_created_then_commented(h)


def test_alert_marker_is_found_on_later_pages_of_issues_and_comments(h):
    """The marker sits on page 2 of the comments of an issue that is itself
    on page 2 of the issue list; pull requests in the list are skipped."""
    _alert_fixtures(h, conclusion="failure", open_issue=3)
    h.put(
        "issues.json",
        [
            [
                {"number": 3, "state": "open"},
                {"number": 9, "state": "closed", "pull_request": {}},
            ],
            [{"number": 2, "state": "closed"}],
        ],
    )
    h.put(
        "comments_2.json",
        [
            [{"body": f"filler {i}"} for i in range(100)],
            [{"body": f"<!-- sync-alert:failure:{RUN_ID} -->\nreported"}],
        ],
    )
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []
    calls = h.calls()
    assert f"api --paginate repos/{REPO}/issues?labels=sync-alert&state=all&per_page=100" in calls
    assert f"api --paginate repos/{REPO}/issues/3/comments?per_page=100" in calls
    assert f"api --paginate repos/{REPO}/issues/2/comments?per_page=100" in calls
    assert not any("/issues/9/" in c for c in calls)


def test_alert_fails_when_the_thread_cannot_be_read(h):
    _alert_fixtures(h, conclusion="failure", open_issue=3)
    (h.fixtures / "comments_3.json").unlink()
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode != 0
    assert h.writes() == []


def test_alert_marker_of_another_run_does_not_suppress(h):
    _alert_fixtures(
        h,
        conclusion="failure",
        open_issue=3,
        comments=["<!-- sync-alert:failure:999 -->\nother run"],
    )
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    _assert_only_comment(h, 3)


def test_alert_clean_success_comments_recovered_and_closes(h):
    _alert_fixtures(h, open_issue=3, comments=["<!-- sync-alert:failure:900 -->"])
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    writes = h.writes()
    assert len(writes) == 2
    assert writes[0].startswith(f"issue comment 3 -R {REPO} --body-file ")
    assert writes[1] == f"issue close 3 -R {REPO}"
    lines = h.posted_body().splitlines()
    assert lines[0] == f"<!-- sync-alert:recovered:{RUN_ID} -->"
    assert lines[1].startswith("**Recovered**")
    assert f"- Run: {RUN_URL}" in lines


def test_alert_out_of_order_success_does_not_close(h):
    _alert_fixtures(h, open_issue=3, newest="1002")
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []
    assert "newest completed run on main is 1002" in proc.stdout


def test_alert_check_only_success_posts_nothing_and_closes_nothing(h):
    """A run with publish off succeeds on main, is the newest run and has no
    drops, but it published nothing, so the issue must stay open."""
    _alert_fixtures(h, open_issue=3, published=False)
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []
    assert "succeeded without publishing" in proc.stdout


def test_alert_check_only_success_with_drops_posts_nothing(h):
    drops = [{"config": "cfg_a", "reason": "not_a_dict", "count": 1, "first_path": "p.json"}]
    _alert_fixtures(h, open_issue=3, published=False, meta=_snapshot_meta(drops))
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []


def test_alert_clean_success_without_open_issue_does_nothing(h):
    _alert_fixtures(h)
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []


def test_alert_success_with_drops_comments_breakdown_and_stays_open(h):
    drops = [
        {"config": "cfg_a", "reason": "validation_error", "count": 2,
         "first_path": "data/cfg_a/dev/model/x.json"},
        {"config": "cfg_b", "reason": "not_a_dict", "count": 1,
         "first_path": "data/cfg_b/dev/model/y.json"},
    ]
    _alert_fixtures(h, open_issue=3, meta=_snapshot_meta(drops))
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    _assert_only_comment(h, 3)
    lines = h.posted_body().splitlines()
    assert lines[0] == f"<!-- sync-alert:drops:{RUN_ID} -->"
    assert "rejected 3 upstream record(s)" in lines[1]
    assert "| cfg_a | validation_error | 2 | `data/cfg_a/dev/model/x.json` |" in lines
    assert "| cfg_b | not_a_dict | 1 | `data/cfg_b/dev/model/y.json` |" in lines


def test_alert_success_with_drops_creates_the_issue_when_none_is_open(h):
    drops = [{"config": "cfg_a", "reason": "not_a_dict", "count": 1, "first_path": "p.json"}]
    _alert_fixtures(h, meta=_snapshot_meta(drops))
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    _assert_created_then_commented(h)


def _raw_meta(scalar, **extra):
    return {"row_counts": {"dropped_eee_records_stage_a": scalar}, **extra}


_ENTRY = {"config": "c", "reason": "r", "first_path": "p.json"}


@pytest.mark.parametrize(
    "meta",
    [
        None,
        "{not json",
        "[]",
        {"snapshot_id": "x"},
        _raw_meta(0, stage_a_drops=None),
        _raw_meta(2, stage_a_drops=[]),
        _raw_meta(0, stage_a_drops=[{**_ENTRY, "count": 1}]),
        _raw_meta(1, stage_a_drops=[{**_ENTRY, "count": 2}]),
        _raw_meta(0, stage_a_drops=[{**_ENTRY, "count": 0}]),
        _raw_meta(1, stage_a_drops=[{**_ENTRY, "count": 1.5}, {**_ENTRY, "count": -0.5}]),
        _raw_meta(1, stage_a_drops=[{**_ENTRY, "count": "1"}]),
        _raw_meta(1, stage_a_drops=["x"]),
        _raw_meta(0, stage_a_drops={}),
        _raw_meta(-1),
        _raw_meta(0.5),
        _raw_meta("0"),
        _raw_meta(None, stage_a_drops=[]),
    ],
)
def test_alert_success_with_missing_or_unreadable_artifact_reports_and_never_closes(h, meta):
    _alert_fixtures(h, open_issue=3, meta=meta)
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    _assert_only_comment(h, 3)
    lines = h.posted_body().splitlines()
    assert lines[0] == f"<!-- sync-alert:artifact:{RUN_ID} -->"
    assert "missing or unreadable" in lines[1]


def test_alert_scalar_only_snapshot_is_judged_on_the_scalar(h):
    """Snapshots written before the breakdown existed have no stage_a_drops."""
    _alert_fixtures(h, open_issue=3, meta=_raw_meta(0))
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes()[-1] == f"issue close 3 -R {REPO}"

    _alert_fixtures(h, open_issue=3, meta=_raw_meta(4))
    h.log.unlink()
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    _assert_only_comment(h, 3)
    assert "rejected 4 upstream record(s)" in h.posted_body()
    assert "no per-config breakdown" in h.posted_body()


@pytest.mark.parametrize("conclusion", ["cancelled", "skipped"])
def test_alert_ignores_cancelled_and_skipped(h, conclusion):
    _alert_fixtures(h, conclusion=conclusion, open_issue=3)
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.calls() == [f"api repos/{REPO}/actions/runs/{RUN_ID}"]


def test_alert_dry_run_prints_action_and_body_and_writes_nothing(h):
    _alert_fixtures(h, conclusion="failure", meta=None)
    proc = h.run("ci_sync_alert.sh", "--dry-run", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []
    assert "ACTION (dry run): create issue 'Sync Pipeline alert'" in proc.stdout
    assert f"<!-- sync-alert:failure:{RUN_ID} -->" in proc.stdout

    _alert_fixtures(h, open_issue=3)
    proc = h.run("ci_sync_alert.sh", "--dry-run", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []
    assert "ACTION (dry run): comment 'recovered' on issue #3 and close it" in proc.stdout


# 2026-10-06T12:00:00Z
NOW = "1791288000"


def _published(snapshot_id):
    return {"snapshot_id": snapshot_id, "configs": [], "row_counts": {}}


def test_staleness_alerts_when_published_snapshot_is_older_than_36_hours(h):
    _alert_fixtures(h, open_issue=3)
    h.put("published_meta.json", _published("2026-10-04T23:00:00Z"))
    proc = h.run("ci_sync_alert.sh", "--staleness", env={"SYNC_ALERT_NOW": NOW})
    assert proc.returncode == 0, proc.stderr
    assert h.calls()[0] == (
        "https://huggingface.co/datasets/evaleval/card_backend"
        "/resolve/main/warehouse/latest/snapshot_meta.json"
    )
    _assert_only_comment(h, 3)
    lines = h.posted_body().splitlines()
    assert lines[0] == "<!-- sync-alert:stale:2026-10-06 -->"
    assert "snapshot `2026-10-04T23:00:00Z`, 37 hours old (limit 36)" in lines[3]


def test_staleness_is_quiet_at_36_hours(h):
    _alert_fixtures(h, open_issue=3)
    h.put("published_meta.json", _published("2026-10-05T00:00:00Z"))
    proc = h.run("ci_sync_alert.sh", "--staleness", env={"SYNC_ALERT_NOW": NOW})
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []
    assert "fresh" in proc.stdout


def test_staleness_posts_once_per_day(h):
    _alert_fixtures(h, open_issue=3, comments=["<!-- sync-alert:stale:2026-10-06 -->\nstale"])
    h.put("published_meta.json", _published("2026-10-01T00:00:00Z"))
    proc = h.run("ci_sync_alert.sh", "--staleness", env={"SYNC_ALERT_NOW": NOW})
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []


@pytest.mark.parametrize(
    "published, problem",
    [
        (None, "could not be fetched"),
        ("<html>rate limited</html>", "no snapshot_id timestamp"),
        (_published("not-a-timestamp"), "no snapshot_id timestamp"),
        ({"configs": []}, "no snapshot_id timestamp"),
    ],
)
def test_staleness_alerts_when_published_meta_cannot_be_read(h, published, problem):
    _alert_fixtures(h, open_issue=3)
    if published is not None:
        h.put("published_meta.json", published)
    proc = h.run("ci_sync_alert.sh", "--staleness", env={"SYNC_ALERT_NOW": NOW})
    assert proc.returncode == 0, proc.stderr
    _assert_only_comment(h, 3)
    lines = h.posted_body().splitlines()
    assert lines[0] == "<!-- sync-alert:stale-unreadable:2026-10-06 -->"
    assert any(line.startswith("- Problem:") and problem in line for line in lines)


def test_staleness_unreadable_creates_the_issue_and_posts_once_per_day(h):
    _alert_fixtures(h)
    proc = h.run("ci_sync_alert.sh", "--staleness", env={"SYNC_ALERT_NOW": NOW})
    assert proc.returncode == 0, proc.stderr
    _assert_created_then_commented(h)

    _alert_fixtures(h, open_issue=7, comments=[h.posted_body()])
    h.log.unlink()
    proc = h.run("ci_sync_alert.sh", "--staleness", env={"SYNC_ALERT_NOW": NOW})
    assert proc.returncode == 0, proc.stderr
    assert h.writes() == []


# ---------------------------------------------------------------------------
# sync.yml wiring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sync_workflow():
    path = SCRIPTS.parent / ".github" / "workflows" / "sync.yml"
    return yaml.safe_load(path.read_text())


def _steps(workflow):
    return workflow["jobs"]["run-pipeline"]["steps"]


def _index(steps, **match):
    hits = [
        i for i, step in enumerate(steps)
        if all(step.get(k) == v for k, v in match.items())
    ]
    assert len(hits) == 1, f"expected one step matching {match}, got {len(hits)}"
    return hits[0]


def test_workflow_resolves_revisions_before_checkout_and_caches(sync_workflow):
    steps = _steps(sync_workflow)
    resolve = _index(steps, id="revisions")
    assert steps[resolve]["run"] == "bash scripts/ci_resolve_revisions.sh"
    assert steps[resolve]["env"]["PINNED_EEE_REVISION"] == "${{ vars.EEE_REVISION }}"
    assert (
        steps[resolve]["env"]["PINNED_ENTITY_REGISTRY_REVISION"]
        == "${{ vars.ENTITY_REGISTRY_REVISION }}"
    )

    registry_checkout = [
        i for i, step in enumerate(steps)
        if step.get("with", {}).get("repository") == "evaleval/evalcard-registry"
    ]
    assert len(registry_checkout) == 1
    caches = [
        i for i, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/cache@")
    ]
    assert len(caches) == 3
    assert resolve < registry_checkout[0]
    assert all(resolve < i for i in caches)

    keys = [steps[i]["with"]["key"] for i in caches]
    assert any(
        k.startswith("hf-bulk-${{ steps.revisions.outputs.eee_revision }}-") for k in keys
    )
    assert "entity-registry-${{ steps.revisions.outputs.entity_registry_revision }}" in keys


def test_workflow_checks_out_the_registry_at_the_derived_resolver_ref(sync_workflow):
    steps = _steps(sync_workflow)
    checkout = next(
        step for step in steps
        if step.get("with", {}).get("repository") == "evaleval/evalcard-registry"
    )
    assert checkout["with"]["ref"] == "${{ env.RESOLVER_REF }}"
    assert checkout["with"]["path"] == "evalcard-registry"


def test_workflow_env_carries_no_hardcoded_followed_revisions(sync_workflow):
    env = sync_workflow["env"]
    for name in ("EEE_REVISION", "ENTITY_REGISTRY_REVISION", "RESOLVER_REF"):
        assert name not in env
        for step in _steps(sync_workflow):
            assert name not in step.get("env", {})
    assert set(env) == {
        "HF_TARGET_DATASET",
        "EEE_INCLUDE_PRIVATE",
        "EEE_PRIVATE_REVISION",
        "BENCHMARK_METADATA_REVISION",
    }


def test_workflow_publish_and_check_only_conditions(sync_workflow):
    steps = _steps(sync_workflow)
    shrink = "${{ github.event_name == 'workflow_dispatch' && inputs.allow_shrink && '1' || '0' }}"

    publish = steps[_index(steps, name=PUBLISH_STEP)]
    assert publish["if"] == "${{ github.event_name != 'workflow_dispatch' || inputs.publish }}"
    assert publish["run"] == "uv run python scripts/ci_publish_warehouse.py"
    assert publish["env"]["ALLOW_SNAPSHOT_SHRINK"] == shrink

    check = steps[_index(steps, name=CHECK_ONLY_STEP)]
    assert check["if"] == "${{ github.event_name == 'workflow_dispatch' && !inputs.publish }}"
    assert check["run"] == "uv run python scripts/ci_publish_warehouse.py --check-only"
    assert check["env"]["ALLOW_SNAPSHOT_SHRINK"] == shrink

    # PyYAML reads the bare `on:` key as boolean True.
    inputs = sync_workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["publish"] == {**inputs["publish"], "type": "boolean", "default": True}
    assert inputs["allow_shrink"] == {
        **inputs["allow_shrink"], "type": "boolean", "default": False,
    }


def test_alert_script_names_the_workflow_publish_step(sync_workflow):
    _index(_steps(sync_workflow), name=PUBLISH_STEP)
    script = (SCRIPTS / "ci_sync_alert.sh").read_text()
    assert f'PUBLISH_STEP_NAME="{PUBLISH_STEP}"' in script


def test_workflow_concurrency_and_permissions(sync_workflow):
    assert sync_workflow["concurrency"] == {
        "group": (
            "${{ github.event_name == 'workflow_dispatch' && !inputs.publish "
            "&& format('sync-pipeline-check-{0}', github.ref) || 'sync-pipeline-publish' }}"
        ),
        "cancel-in-progress": False,
    }
    assert sync_workflow["permissions"] == {"contents": "read"}
    assert sync_workflow[True]["schedule"] == [{"cron": "17 11 * * *"}]

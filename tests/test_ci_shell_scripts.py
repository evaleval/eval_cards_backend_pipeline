"""`scripts/ci_resolve_revisions.sh` and `scripts/ci_sync_alert.sh`.

Both scripts run for real under bash with fake `gh` and `curl` executables
first on PATH. The fakes answer from files in a fixture dir and append
every invocation to a log, so the tests assert on the exact commands the
scripts would run and never touch the network.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

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

FAKE_GH = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_LOG"
fix="$FAKE_FIXTURES"
arg_after() {
  local flag="$1"; shift
  while [ $# -gt 0 ]; do
    if [ "$1" = "$flag" ]; then echo "$2"; return 0; fi
    shift
  done
}
case "$1 ${2:-}" in
  "api "*)
    path=""
    for a in "$@"; do case "$a" in repos/*) path="$a" ;; esac; done
    case "$path" in
      */actions/runs/*/jobs*) cat "$fix/jobs.json" ;;
      */actions/jobs/*/logs) cat "$fix/job.log" ;;
      */actions/workflows/*/runs*) cat "$fix/newest_runs.json" ;;
      */actions/runs/*) cat "$fix/run.json" ;;
      */issues/*/comments*) cat "$fix/comments.json" ;;
      *) echo "fake gh: unexpected api path $path" >&2; exit 1 ;;
    esac
    ;;
  "issue list") cat "$fix/issues.json" ;;
  "issue create") echo "https://github.com/org/pipeline/issues/7" ;;
  "issue comment") cp "$(arg_after --body-file "$@")" "$FAKE_BODY" ;;
  "issue close" | "label create") ;;
  "run download")
    [ -f "$fix/snapshot_meta.json" ] || exit 1
    dest="$(arg_after -D "$@")/2026-10-04T00:00:00Z"
    mkdir -p "$dest"
    cp "$fix/snapshot_meta.json" "$dest/snapshot_meta.json"
    ;;
  *) echo "fake gh: unexpected command $*" >&2; exit 1 ;;
esac
"""

FAKE_CURL = r"""#!/usr/bin/env bash
set -euo pipefail
url="${!#}"
printf '%s\n' "$url" >> "$FAKE_LOG"
fix="$FAKE_FIXTURES"
case "$url" in
  *"/api/datasets/evaleval/EEE_datastore?"*) file="eee_head.json" ;;
  *"/api/datasets/evaleval/entity-registry-data?"*) file="registry_head.json" ;;
  */manifest.json) file="manifest.json" ;;
  */warehouse/latest/snapshot_meta.json) file="published_meta.json" ;;
  *) echo "fake curl: unexpected url $url" >&2; exit 1 ;;
esac
[ -f "$fix/$file" ] || { echo "fake curl: 404 $url" >&2; exit 22; }
cat "$fix/$file"
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


def _alert_fixtures(
    h,
    *,
    conclusion="success",
    open_issue=None,
    comments=(),
    meta="default",
    newest=RUN_ID,
):
    h.put(
        "run.json",
        {"id": int(RUN_ID), "conclusion": conclusion, "event": "schedule", "html_url": RUN_URL},
    )
    h.put("issues.json", [{"number": open_issue}] if open_issue else [])
    h.put("comments.json", [{"body": b} for b in comments])
    h.put("newest_runs.json", {"workflow_runs": [{"id": int(newest)}]})
    h.put(
        "jobs.json",
        {
            "jobs": [
                {
                    "id": 55,
                    "name": "run-pipeline",
                    "conclusion": "failure",
                    "steps": [
                        {"name": "Run tests", "conclusion": "success"},
                        {"name": "Run canonicalisation", "conclusion": "failure"},
                        {"name": "Publish", "conclusion": "skipped"},
                    ],
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


@pytest.mark.parametrize("meta", [None, "{not json", {"snapshot_id": "x"}])
def test_alert_success_with_missing_or_unreadable_artifact_reports_and_never_closes(h, meta):
    _alert_fixtures(h, open_issue=3, meta=meta)
    proc = h.run("ci_sync_alert.sh", "--run-id", RUN_ID)
    assert proc.returncode == 0, proc.stderr
    _assert_only_comment(h, 3)
    lines = h.posted_body().splitlines()
    assert lines[0] == f"<!-- sync-alert:artifact:{RUN_ID} -->"
    assert "missing or unreadable" in lines[1]


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


def test_staleness_fails_when_published_meta_cannot_be_read(h):
    _alert_fixtures(h)
    proc = h.run("ci_sync_alert.sh", "--staleness", env={"SYNC_ALERT_NOW": NOW})
    assert proc.returncode != 0
    assert h.writes() == []

    h.put("published_meta.json", _published("not-a-timestamp"))
    proc = h.run("ci_sync_alert.sh", "--staleness", env={"SYNC_ALERT_NOW": NOW})
    assert proc.returncode != 0
    assert h.writes() == []

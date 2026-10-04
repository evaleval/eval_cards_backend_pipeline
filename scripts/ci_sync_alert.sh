#!/usr/bin/env bash
# CI helper: report Sync Pipeline trouble in one GitHub issue thread.
#
#   ci_sync_alert.sh [--dry-run] --run-id <id>   handle one completed run
#   ci_sync_alert.sh [--dry-run] --staleness     check the published snapshot
#
# The thread is the open issue labelled `sync-alert`; it is created when
# something needs reporting and none is open.
#
#   failure, timed_out, startup_failure, action_required
#       comment: run link, event, failed steps, upstream revisions (when the
#       run got far enough to write them) and the end of the failed job log
#   success, Stage A dropped records
#       comment with the breakdown; the issue stays open
#   success, snapshot-meta artifact missing or unreadable
#       comment; the issue stays open
#   success, nothing dropped
#       if an issue is open and this is the newest completed Sync Pipeline
#       run on main: comment "recovered" and close it
#   cancelled, skipped, anything else
#       nothing
#   --staleness
#       comment when the snapshot_id of the published
#       warehouse/latest/snapshot_meta.json is older than 36 hours, at most
#       once per UTC day
#
# Every comment starts with a hidden marker
# `<!-- sync-alert:<kind>:<run id or date> -->`; when the open issue already
# has a comment with that marker nothing is posted, so re-deliveries and
# replays do not duplicate.
#
# --dry-run makes the same read calls, prints the action and the comment
# body, and writes nothing.
#
# Needs gh (GH_TOKEN with issues: write, actions: read), curl and jq.
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-evaleval/eval_cards_backend_pipeline}"
WORKFLOW_FILE="sync.yml"
LABEL="sync-alert"
ISSUE_TITLE="Sync Pipeline alert"
HF_TARGET_DATASET="${HF_TARGET_DATASET:-evaleval/card_backend}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
STALE_AFTER_HOURS=36
LOG_TAIL_LINES=40

dry_run=0
mode=""
run_id=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --staleness) mode="staleness" ;;
    --run-id)
      mode="run"
      run_id="${2:-}"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done
if [ -z "$mode" ]; then
  echo "usage: ci_sync_alert.sh [--dry-run] (--run-id <id> | --staleness)" >&2
  exit 2
fi
if [ "$mode" = "run" ] && ! [[ "$run_id" =~ ^[0-9]+$ ]]; then
  echo "--run-id needs a numeric run id, got '${run_id}'" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
body_file="$tmp/body.md"

open_issue() {
  gh issue list -R "$REPO" --label "$LABEL" --state open --limit 1 --json number \
    | jq -r '.[0].number // empty'
}

comments_file="$tmp/comments.txt"

# Called as a plain statement so a failed read stops the script instead of
# reading as "no such comment".
fetch_comments() {
  gh api --paginate "repos/${REPO}/issues/$1/comments?per_page=100" \
    | jq -r '.[].body' > "$comments_file"
}

has_marker() {
  grep -qF -- "$1" "$comments_file"
}

# post <marker> <summary>: comment $body_file on the open issue, creating
# the issue first when none is open.
post() {
  local marker="$1" summary="$2" issue
  issue="$(open_issue)"
  if [ -n "$issue" ]; then
    fetch_comments "$issue"
    if has_marker "$marker"; then
      echo "ACTION: none (issue #${issue} already has ${marker})"
      return 0
    fi
  fi
  if [ "$dry_run" = 1 ]; then
    if [ -n "$issue" ]; then
      echo "ACTION (dry run): comment on issue #${issue}: ${summary}"
    else
      echo "ACTION (dry run): create issue '${ISSUE_TITLE}' labelled ${LABEL}, then comment: ${summary}"
    fi
    echo "----- body -----"
    cat "$body_file"
    echo "----- end body -----"
    return 0
  fi
  if [ -z "$issue" ]; then
    gh label create "$LABEL" -R "$REPO" --color D93F0B \
      --description "Sync Pipeline failures, drops and staleness" >/dev/null 2>&1 || true
    local url
    url="$(gh issue create -R "$REPO" --title "$ISSUE_TITLE" --label "$LABEL" \
      --body "Thread for Sync Pipeline alerts, written by .github/workflows/sync-alert.yml. It is closed automatically by the next clean run on main.")"
    issue="${url##*/}"
    echo "ACTION: created issue #${issue}"
  fi
  gh issue comment "$issue" -R "$REPO" --body-file "$body_file" >/dev/null
  echo "ACTION: commented on issue #${issue}: ${summary}"
}

# Download the run's snapshot-meta artifact; print the path of the
# snapshot_meta.json inside it, or nothing when there is none.
fetch_snapshot_meta() {
  local dir="$tmp/artifact"
  mkdir -p "$dir"
  gh run download "$run_id" -R "$REPO" -n snapshot-meta -D "$dir" >/dev/null 2>&1 || return 0
  find "$dir" -name snapshot_meta.json | sort | tail -n 1
}

revisions_line() {
  local meta="$1"
  if [ -n "$meta" ] && jq -e '.upstream_pins | type == "object"' "$meta" >/dev/null 2>&1; then
    jq -r '.upstream_pins | to_entries | map("\(.key) `\(.value.sha // "unknown")`") | join(", ")' "$meta"
  else
    echo "not available (the run wrote no snapshot_meta.json)"
  fi
}

handle_failure() {
  local conclusion="$1" event="$2" url="$3"
  local marker="<!-- sync-alert:failure:${run_id} -->"
  local jobs="$tmp/jobs.json"
  gh api "repos/${REPO}/actions/runs/${run_id}/jobs?per_page=100" > "$jobs" 2>/dev/null \
    || echo '{"jobs":[]}' > "$jobs"

  local steps
  steps="$(jq -r '[.jobs[] | .name as $job | (.steps // [])[]
      | select(.conclusion == "failure" or .conclusion == "timed_out")
      | "`\($job) / \(.name)`"] | join(", ")' "$jobs")"
  [ -n "$steps" ] || steps="none reported"

  # End of the failed job's log, up to its last error line. The lines after
  # that are post-job cleanup and say nothing about the failure.
  local job_id log="$tmp/job.log" tail_file="$tmp/tail.log"
  job_id="$(jq -r '[.jobs[] | select(.conclusion == "failure" or .conclusion == "timed_out") | .id][0] // empty' "$jobs")"
  : > "$tail_file"
  if [ -n "$job_id" ] && gh api "repos/${REPO}/actions/jobs/${job_id}/logs" > "$log" 2>/dev/null; then
    local last
    last="$(grep -n '##\[error\]' "$log" | tail -n 1 | cut -d: -f1 || true)"
    if [ -n "$last" ]; then
      head -n "$last" "$log" > "$tmp/upto.log"
    else
      cp "$log" "$tmp/upto.log"
    fi
    tail -n "$LOG_TAIL_LINES" "$tmp/upto.log" \
      | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z //' | cut -c1-400 > "$tail_file"
  fi
  [ -s "$tail_file" ] || echo "(log not available)" > "$tail_file"

  local meta
  meta="$(fetch_snapshot_meta)"
  {
    echo "$marker"
    echo "**Sync Pipeline run did not succeed** (conclusion: \`${conclusion}\`)"
    echo
    echo "- Run: ${url}"
    echo "- Event: \`${event}\`"
    echo "- Failed steps: ${steps}"
    echo "- Upstream revisions: $(revisions_line "$meta")"
    echo
    echo "Last ${LOG_TAIL_LINES} lines of the failed job log, up to its last error:"
    echo
    echo '````'
    cat "$tail_file"
    echo '````'
  } > "$body_file"
  post "$marker" "run ${run_id} ${conclusion}"
}

handle_success() {
  local event="$1" url="$2"
  local meta
  meta="$(fetch_snapshot_meta)"

  if [ -z "$meta" ] || ! jq -e '(.stage_a_drops | type == "array")
      or (.row_counts.dropped_eee_records_stage_a | type == "number")' "$meta" >/dev/null 2>&1; then
    local marker="<!-- sync-alert:artifact:${run_id} -->"
    {
      echo "$marker"
      echo "**Sync Pipeline run succeeded but its \`snapshot-meta\` artifact is missing or unreadable**"
      echo
      echo "- Run: ${url}"
      echo "- Event: \`${event}\`"
      echo
      echo "Dropped records could not be checked, so this run does not close the issue."
    } > "$body_file"
    post "$marker" "run ${run_id} succeeded without a readable snapshot-meta artifact"
    return 0
  fi

  # Snapshots written before the breakdown existed carry only the count.
  local n_drops
  n_drops="$(jq -r 'if (.stage_a_drops | type) == "array"
      then ([.stage_a_drops[].count] | add // 0)
      else .row_counts.dropped_eee_records_stage_a end' "$meta")"

  if [ "$n_drops" != "0" ]; then
    local marker="<!-- sync-alert:drops:${run_id} -->"
    {
      echo "$marker"
      echo "**Sync Pipeline run succeeded but rejected ${n_drops} upstream record(s) at load time**"
      echo
      echo "- Run: ${url}"
      echo "- Event: \`${event}\`"
      echo "- Upstream revisions: $(revisions_line "$meta")"
      echo
      if jq -e '.stage_a_drops | type == "array"' "$meta" >/dev/null; then
        echo "| Config | Reason | Count | First record |"
        echo "| --- | --- | --- | --- |"
        jq -r '.stage_a_drops[] | "| \(.config) | \(.reason) | \(.count) | `\(.first_path)` |"' "$meta"
      else
        echo "The snapshot records no per-config breakdown."
      fi
      echo
      echo "The rejected records are left out of the published snapshot. This issue stays open until a run drops nothing."
    } > "$body_file"
    post "$marker" "run ${run_id} dropped ${n_drops} record(s)"
    return 0
  fi

  local issue
  issue="$(open_issue)"
  if [ -z "$issue" ]; then
    echo "ACTION: none (run ${run_id} succeeded with no drops; no open issue)"
    return 0
  fi
  local newest
  newest="$(gh api "repos/${REPO}/actions/workflows/${WORKFLOW_FILE}/runs?branch=main&status=completed&per_page=1" \
    | jq -r '.workflow_runs[0].id // empty')"
  if [ "$newest" != "$run_id" ]; then
    echo "ACTION: none (run ${run_id} succeeded, but the newest completed run on main is ${newest:-unknown}; issue #${issue} stays open)"
    return 0
  fi

  local marker="<!-- sync-alert:recovered:${run_id} -->"
  {
    echo "$marker"
    echo "**Recovered**: Sync Pipeline run succeeded with no dropped records."
    echo
    echo "- Run: ${url}"
    echo "- Event: \`${event}\`"
    echo "- Upstream revisions: $(revisions_line "$meta")"
  } > "$body_file"
  fetch_comments "$issue"
  if has_marker "$marker"; then
    echo "ACTION: none (issue #${issue} already has ${marker})"
  elif [ "$dry_run" = 1 ]; then
    echo "ACTION (dry run): comment 'recovered' on issue #${issue} and close it"
    echo "----- body -----"
    cat "$body_file"
    echo "----- end body -----"
    return 0
  else
    gh issue comment "$issue" -R "$REPO" --body-file "$body_file" >/dev/null
    echo "ACTION: commented 'recovered' on issue #${issue}"
  fi
  if [ "$dry_run" = 1 ]; then
    echo "ACTION (dry run): close issue #${issue}"
  else
    gh issue close "$issue" -R "$REPO" >/dev/null
    echo "ACTION: closed issue #${issue}"
  fi
}

handle_run() {
  local run="$tmp/run.json"
  gh api "repos/${REPO}/actions/runs/${run_id}" > "$run"
  local conclusion event url
  conclusion="$(jq -r '.conclusion // "none"' "$run")"
  event="$(jq -r '.event' "$run")"
  url="$(jq -r '.html_url' "$run")"
  echo "Run ${run_id}: conclusion=${conclusion} event=${event}"
  case "$conclusion" in
    failure | timed_out | startup_failure | action_required)
      handle_failure "$conclusion" "$event" "$url"
      ;;
    success)
      handle_success "$event" "$url"
      ;;
    *)
      echo "ACTION: none (conclusion ${conclusion})"
      ;;
  esac
}

handle_staleness() {
  local meta="$tmp/published_meta.json"
  local url="${HF_ENDPOINT}/datasets/${HF_TARGET_DATASET}/resolve/main/warehouse/latest/snapshot_meta.json"
  curl --fail --silent --show-error --location --retry 5 --retry-delay 5 \
    --retry-connrefused --max-time 60 "$url" > "$meta"
  local snapshot_id now age_hours
  snapshot_id="$(jq -r '.snapshot_id // empty' "$meta")"
  now="${SYNC_ALERT_NOW:-$(date -u +%s)}"
  if ! age_hours="$(jq -er --argjson now "$now" \
      '(($now - (.snapshot_id | fromdateiso8601)) / 3600) | floor' "$meta" 2>/dev/null)"; then
    echo "FATAL: cannot read a snapshot_id timestamp from ${url} (got '${snapshot_id}')" >&2
    exit 1
  fi
  echo "Published snapshot ${snapshot_id} is ${age_hours}h old (limit ${STALE_AFTER_HOURS}h)"
  if [ "$age_hours" -le "$STALE_AFTER_HOURS" ]; then
    echo "ACTION: none (published snapshot is fresh)"
    return 0
  fi
  local today
  today="$(jq -rn --argjson now "$now" '$now | strftime("%Y-%m-%d")')"
  local marker="<!-- sync-alert:stale:${today} -->"
  {
    echo "$marker"
    echo "**Published snapshot is stale**"
    echo
    echo "- \`warehouse/latest\` in \`${HF_TARGET_DATASET}\` holds snapshot \`${snapshot_id}\`, ${age_hours} hours old (limit ${STALE_AFTER_HOURS})."
    echo "- Checked on ${today} (UTC)."
    echo "- Runs: https://github.com/${REPO}/actions/workflows/${WORKFLOW_FILE}"
    echo
    echo "No Sync Pipeline run has published since then: the schedule did not fire, or runs are failing or not publishing."
  } > "$body_file"
  post "$marker" "published snapshot is ${age_hours}h old"
}

if [ "$mode" = "run" ]; then
  handle_run
else
  handle_staleness
fi

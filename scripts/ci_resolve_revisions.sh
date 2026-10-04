#!/usr/bin/env bash
# CI helper: decide which upstream commits this run consumes.
#
# Inputs (env, both optional):
#   PINNED_EEE_REVISION              commit sha of evaleval/EEE_datastore
#   PINNED_ENTITY_REGISTRY_REVISION  commit sha of evaleval/entity-registry-data
# An empty input means "follow the head of that dataset", resolved to a sha
# here so the pipeline re-lists against a concrete commit and the cache keys
# change when upstream does.
#
# The resolver ref is never an input. It is the `seed_git_sha` recorded in
# `manifest.json` at the selected registry commit: the registry repo commit
# that produced that data, so resolver code, seed YAMLs and registry data
# always come from the same state.
#
# Writes EEE_REVISION, ENTITY_REGISTRY_REVISION and RESOLVER_REF to
# $GITHUB_ENV, the same values to $GITHUB_OUTPUT (lowercase names), and a
# table to $GITHUB_STEP_SUMMARY. Any value that is not a 40-hex sha fails
# the step; there is no fallback.
set -euo pipefail

EEE_REPO="evaleval/EEE_datastore"
REGISTRY_REPO="evaleval/entity-registry-data"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

fetch() {
  local args=(--fail --silent --show-error --location --globoff
    --retry 5 --retry-delay 5 --retry-connrefused --max-time 60)
  if [ -n "${HF_TOKEN:-}" ]; then
    args+=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi
  curl "${args[@]}" "$1"
}

require_sha() {
  local label="$1" value="$2"
  if ! [[ "$value" =~ ^[0-9a-f]{40}$ ]]; then
    echo "FATAL: ${label} is not a 40-hex commit sha: '${value}'" >&2
    exit 1
  fi
}

head_sha() {
  fetch "${HF_ENDPOINT}/api/datasets/$1?expand[]=sha" | jq -r '.sha // empty'
}

pinned_eee="${PINNED_EEE_REVISION:-}"
pinned_registry="${PINNED_ENTITY_REGISTRY_REVISION:-}"

# Reject a malformed pin before any network call.
if [ -n "$pinned_eee" ]; then
  require_sha "repo variable EEE_REVISION" "$pinned_eee"
fi
if [ -n "$pinned_registry" ]; then
  require_sha "repo variable ENTITY_REGISTRY_REVISION" "$pinned_registry"
fi

if [ -n "$pinned_eee" ]; then
  eee_sha="$pinned_eee"
  eee_mode="pinned (repo variable EEE_REVISION)"
else
  eee_sha="$(head_sha "$EEE_REPO")"
  eee_mode="followed head of ${EEE_REPO}"
  require_sha "head of ${EEE_REPO}" "$eee_sha"
fi

if [ -n "$pinned_registry" ]; then
  registry_sha="$pinned_registry"
  registry_mode="pinned (repo variable ENTITY_REGISTRY_REVISION)"
else
  registry_sha="$(head_sha "$REGISTRY_REPO")"
  registry_mode="followed head of ${REGISTRY_REPO}"
  require_sha "head of ${REGISTRY_REPO}" "$registry_sha"
fi

manifest_url="${HF_ENDPOINT}/datasets/${REGISTRY_REPO}/resolve/${registry_sha}/manifest.json"
resolver_ref="$(fetch "$manifest_url" | jq -r '.seed_git_sha // empty')"
require_sha "seed_git_sha in ${manifest_url}" "$resolver_ref"
resolver_mode="derived from seed_git_sha in the registry manifest"

{
  echo "EEE_REVISION=${eee_sha}"
  echo "ENTITY_REGISTRY_REVISION=${registry_sha}"
  echo "RESOLVER_REF=${resolver_ref}"
} | tee -a "${GITHUB_ENV:-/dev/null}"

{
  echo "eee_revision=${eee_sha}"
  echo "entity_registry_revision=${registry_sha}"
  echo "resolver_ref=${resolver_ref}"
} >> "${GITHUB_OUTPUT:-/dev/null}"

{
  echo "### Upstream revisions"
  echo
  echo "| Input | Commit | Mode |"
  echo "| --- | --- | --- |"
  echo "| EEE datastore | \`${eee_sha}\` | ${eee_mode} |"
  echo "| Entity registry data | \`${registry_sha}\` | ${registry_mode} |"
  echo "| Resolver and seed YAMLs | \`${resolver_ref}\` | ${resolver_mode} |"
} | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"

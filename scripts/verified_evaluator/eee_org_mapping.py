"""Attribute each evaluation record in evaleval/EEE_datastore to the org that
SUBMITTED it, recovering the original submitter even when a maintainer later
re-imported / repaired / consolidated the data under a different merged PR.

Why the naive "merged-PR author" approach is wrong
--------------------------------------------------
Several large submissions were proposed in PRs that were *closed, not merged*,
then re-imported by a maintainer in a separate merged PR. Examples (verified):
  - alphaXiv ~28k records: submitted in closed PR #26 (simpod), re-imported in
    merged PR #79 (yananlong).
  - Multi-SWE/SWE-PolyBench: closed PR #72 (jatinganhotra) -> merged #74.
  - Arcadia Impact (ACL): closed PR #57 (mrpfisher) -> merged #76.
The record UUID is stable across re-import, so the true submitter is recoverable
by finding the earliest PR (ANY status) that introduced each UUID.

Model
-----
universe        = every data/**/*.json record currently on main (HEAD).
adders(uuid)    = every PR, any status, that ADDS that uuid -- evidence is the
                  union of (a) its branch adding it vs its fork point and (b) its
                  merge commit adding it to main. Rename detection is disabled
                  (--no-renames) so delete+add and path moves still count as adds.
submitter(uuid) = earliest (lowest-numbered) adder whose author is NOT a
                  maintainer/bot/deleted account. Maintainers consolidate, they
                  don't submit; if no non-maintainer ever added it (e.g. it was
                  direct-pushed to main, or only a maintainer ever touched it) ->
                  submitter_unknown.
lander(uuid)    = latest MERGED PR whose merge commit put that uuid on main
                  (the consolidation that actually landed it). Recorded for
                  provenance; may legitimately be a maintainer.

Caveat: HF org lookups see only *public* membership ("affiliation," not a roster).

Run with `uv run scripts/verified_evaluator/eee_org_mapping.py` (deps are inline
script metadata). Intermediates land in `.cache/verified_evaluator/`; then run
`apply_affiliation.py` to regenerate the `vendor/` parquets.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "requests", "rapidfuzz"]
# ///

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi
from rapidfuzz import fuzz

HF = "https://huggingface.co"
REPO_ID = "evaleval/EEE_datastore"
REPO_TYPE = "dataset"
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".cache" / "verified_evaluator"
REPO_DIR = CACHE_DIR / "repo"
OUT_DIR = CACHE_DIR / "output"

# A data record file: any uuid-named .json under data/, at any nesting depth
# (closed PRs used over-nested paths later flattened on main; the uuid is stable).
UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.json$"
)
# Canonical on-main layout, for descriptive benchmark/developer/model fields.
RECORD_RE = re.compile(
    r"^data/(?P<benchmark>[^/]+)/(?:(?P<developer>[^/]+)/)?(?P<model>[^/]+)/"
    r"(?P<uuid>[0-9a-fA-F-]{8,})\.json$"
)

# Host/coalition org nearly every contributor belongs to -- never the submitting
# org, so it must not win a reconciliation match.
HOST_ORGS = {"evaleval"}

# Curated denylist of accounts that consolidate/repair/re-import data but do not
# author the evaluations themselves. They are recorded as landers, never credited
# as submitters. Extend as more maintainer/bot accounts are identified.
#   yananlong / EvalEvalBot  -- re-import + parquet/housekeeping (via PRs).
#   deepmage121 (Sree Harsha Nelaturu, also commits as nelaturuharsha) -- bulk
#     direct-pushes external leaderboards ("Upload 5295 files"); coalition member,
#     not the source org. Direct-push only, so this is defensive (no PR today).
MAINTAINERS = {"yananlong", "EvalEvalBot", "deepmage121", "nelaturuharsha"}


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def _env() -> dict:
    import os

    return {"GIT_LFS_SKIP_SMUDGE": "1", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_DIR), *args],
        check=True, capture_output=True, text=True, env=_env(),
    ).stdout


def _git_fetch(refspec: str) -> bool:
    """Fetch one refspec. HF's git endpoint is protocol-flaky (v2 intermittently
    errors 'expected acknowledgments'; v0 sometimes hangs up), so try the default
    protocol first, then force v0. Returns False if the ref is gone."""
    for proto in ([], ["-c", "protocol.version=0"]):
        for attempt in range(2):
            r = subprocess.run(
                ["git", "-C", str(REPO_DIR), *proto,
                 "fetch", "--quiet", "origin", refspec],
                capture_output=True, text=True, env=_env(),
            )
            if r.returncode == 0:
                return True
            if "couldn't find remote ref" in r.stderr or "not our ref" in r.stderr:
                return False  # deleted / never existed
            time.sleep(0.3)
    return False


def ensure_clone() -> None:
    if (REPO_DIR / ".git").exists():
        if not _git_fetch("+refs/heads/main:refs/remotes/origin/main"):
            raise SystemExit(
                "Could not update origin/main from HF. If every protocol fails "
                "(repo negotiation state gone bad), re-clone: delete "
                f"{REPO_DIR} and fetch refs/remotes/pr/* back in from a copy."
            )
        return
    CACHE_DIR.mkdir(exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", f"{HF}/datasets/{REPO_ID}", str(REPO_DIR)],
        check=True, env=_env(),
    )


# --------------------------------------------------------------------------- #
# Stage 1: list ALL PRs (any status) + merge commits for merged ones
# --------------------------------------------------------------------------- #
def list_all_prs(api: HfApi) -> list[dict]:
    """{num, author, title, status} for every PR discussion. Cached."""
    cache = CACHE_DIR / "all_prs.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = []
    for d in api.get_repo_discussions(
        repo_id=REPO_ID, repo_type=REPO_TYPE, discussion_type="pull_request"
    ):
        out.append({"num": d.num, "author": d.author, "title": d.title, "status": d.status})
    out.sort(key=lambda p: p["num"])
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out, indent=2))
    return out


def merged_commit_oids(api: HfApi, prs: list[dict]) -> dict[int, str]:
    """{pr_num: merge_commit_oid} for merged PRs. Cached."""
    cache = CACHE_DIR / "merged_commits.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    out: dict[str, str] = dict(cached)
    for p in prs:
        if p["status"] != "merged" or str(p["num"]) in out:
            continue
        d = api.get_discussion_details(
            repo_id=REPO_ID, discussion_num=p["num"], repo_type=REPO_TYPE
        )
        out[str(p["num"])] = d.merge_commit_oid
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out, indent=2))
    return {int(k): v for k, v in out.items() if v}


# --------------------------------------------------------------------------- #
# Stage 2: fetch every PR branch (cached as refs/remotes/pr/N)
# --------------------------------------------------------------------------- #
def fetch_pr_refs(prs: list[dict]) -> set[int]:
    """Fetch refs/pr/N for every PR; return the nums whose branch is available.
    Skips refs already present locally. Records unavailable nums to disk."""
    cache = CACHE_DIR / "pr_ref_status.json"
    status = json.loads(cache.read_text()) if cache.exists() else {}
    have = set(_git("for-each-ref", "--format=%(refname:short)", "refs/remotes/pr/*").split())
    for p in prs:
        n = p["num"]
        if f"pr/{n}" in have:
            status[str(n)] = True
            continue
        if str(n) in status and status[str(n)] is False:
            continue  # known-gone, don't re-probe every run
        ok = _git_fetch(f"refs/pr/{n}:refs/remotes/pr/{n}")
        status[str(n)] = ok
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(status, indent=2))
    return {int(k) for k, v in status.items() if v}


# --------------------------------------------------------------------------- #
# Stage 3a: per-PR introduced record UUIDs (branch vs its fork point)
# --------------------------------------------------------------------------- #
def pr_introduced_uuids(num: int) -> dict[str, str]:
    """{uuid: path} for record files this PR ADDS relative to its merge-base with
    main. Cached per PR. Matches uuid by the stable filename, any nesting depth."""
    cache = CACHE_DIR / "pr_added" / f"{num}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    ref = f"refs/remotes/pr/{num}"
    mb = _git("merge-base", "origin/main", ref).strip()
    out = _git("diff", "--name-status", "--no-renames", mb, ref)
    uuids: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or parts[0][0] != "A":
            continue
        path = parts[-1]
        if not path.startswith("data/"):
            continue
        m = UUID_RE.search(path)
        if m:
            uuids[m.group(1)] = path
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(uuids))
    return uuids


# --------------------------------------------------------------------------- #
# Stage 3b: per merged-PR records landed on main (merge commit vs first parent)
# --------------------------------------------------------------------------- #
def merged_landed_uuids(num: int, commit: str) -> set[str]:
    """UUIDs the merged PR ADDED to main (so currently-on-main provenance)."""
    cache = CACHE_DIR / "pr_landed" / f"{num}.json"
    if cache.exists():
        return set(json.loads(cache.read_text()))
    try:
        out = _git("diff", "--name-status", "--no-renames", f"{commit}^1", commit)
    except subprocess.CalledProcessError:
        # Merge commit unreachable from current main (history rewrite, e.g. the
        # 2026-06 flat-layout migration). No merge-commit evidence; branch
        # evidence (Stage 3) still attributes the PR's records. Not cached, so
        # a future clone with the old history can still recover it.
        print(f"  WARN: PR #{num} merge commit {commit[:10]} unreachable; skipping landed-scan")
        return set()
    uuids = set()
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or parts[0][0] != "A":
            continue
        path = parts[-1]
        if not path.startswith("data/"):
            continue
        m = UUID_RE.search(path)
        if m:
            uuids.add(m.group(1))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sorted(uuids)))
    return uuids


# --------------------------------------------------------------------------- #
# Universe: every record currently on main + its content
# --------------------------------------------------------------------------- #
def head_records() -> dict[str, str]:
    """{uuid: path} for every data/**/*.json on main HEAD."""
    out = _git("ls-tree", "-r", "--name-only", "origin/main", "data")
    recs: dict[str, str] = {}
    for path in out.splitlines():
        m = UUID_RE.search(path)
        if m and path.startswith("data/"):
            recs[m.group(1)] = path
    return recs


def read_meta_batch(paths: dict[str, str], rev: str = "origin/main") -> dict[str, dict]:
    """source_organization_name/source_type/evaluation_id per uuid, read from
    `rev` via one streaming `git cat-file --batch`. Output is in input order, so
    responses are paired to specs by position (robust to spaces in paths)."""
    items = list(paths.items())  # (uuid, path)
    if not items:
        return {}
    specs = [f"{rev}:{p}".encode() for _, p in items]
    proc = subprocess.Popen(
        ["git", "-C", str(REPO_DIR), "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=_env(),
    )
    import threading

    threading.Thread(
        target=lambda: (proc.stdin.write(b"\n".join(specs) + b"\n"), proc.stdin.close()),
        daemon=True,
    ).start()

    meta: dict[str, dict] = {}
    out = proc.stdout
    for uuid, _ in items:
        header = out.readline()
        if not header:
            break
        if header.rstrip().endswith(b"missing"):
            continue
        try:
            size = int(header.split()[-1])
        except (ValueError, IndexError):
            raise RuntimeError(f"cat-file batch desync at header: {header!r}")
        body = out.read(size)
        out.read(1)
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue
        sm = obj.get("source_metadata") or {}
        meta[uuid] = {
            "source_organization_name": sm.get("source_organization_name"),
            "source_type": sm.get("source_type"),
            "evaluation_id": obj.get("evaluation_id"),
        }
    proc.wait()
    return meta


# --------------------------------------------------------------------------- #
# Stage 4: author -> public HF orgs
# --------------------------------------------------------------------------- #
def user_orgs(username: str, session: requests.Session, token: str | None) -> list[str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = session.get(f"{HF}/api/users/{username}/overview", headers=headers, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return sorted(o["name"] for o in r.json().get("orgs", []))


def build_author_org_map(authors: list[str], token: str | None, delay: float = 0.3) -> dict[str, list[str]]:
    cache = CACHE_DIR / "author_orgs.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    session = requests.Session()
    out: dict[str, list[str]] = {}
    for user in sorted(set(authors)):
        if user == "deleted":
            continue
        if user in cached:
            out[user] = cached[user]
            continue
        out[user] = user_orgs(user, session, token)
        time.sleep(delay)
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps({**cached, **out}, indent=2))
    return out


# --------------------------------------------------------------------------- #
# Stage 5: reconcile submitter org <-> declared source_organization_name
# --------------------------------------------------------------------------- #
def best_org_match(orgs: list[str], source_org: str | None, threshold: int = 80) -> tuple[str | None, float]:
    candidates = [o for o in orgs if o not in HOST_ORGS]
    if not candidates or not source_org or source_org.lower() in {"unknown", "n/a", ""}:
        return None, 0.0
    best, score = max(((o, fuzz.token_sort_ratio(o, source_org)) for o in candidates), key=lambda t: t[1])
    return (best, score) if score >= threshold else (None, score)


def review_reason(orgs: list[str], source_org: str | None, match: str | None, status: str) -> str:
    if status == "submitter_unknown":
        return "submitter_unknown"
    if match:
        return "matched"
    if not [o for o in orgs if o not in HOST_ORGS]:
        return "no_org_info"
    if not source_org or source_org.lower() in {"unknown", "n/a", ""}:
        return "no_source_org"
    return "org_mismatch"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(token: str | None) -> dict:
    api = HfApi(token=token)

    print("Ensuring local clone ...")
    ensure_clone()

    print("Stage 1: listing all PRs ...")
    prs = list_all_prs(api)
    by_num = {p["num"]: p for p in prs}
    import collections
    print(f"  {len(prs)} PRs: {dict(collections.Counter(p['status'] for p in prs))}")
    merged_oids = merged_commit_oids(api, prs)

    print("Stage 2: fetching all PR branches (cached) ...")
    available = fetch_pr_refs(prs)
    print(f"  {len(available)}/{len(prs)} PR branches available")

    print("Stage 3: scanning PR branches for introduced records ...")
    introduced: dict[str, list[int]] = collections.defaultdict(list)  # uuid -> [pr nums]
    n_intro = {}
    for p in prs:
        if p["num"] not in available:
            continue
        uu = pr_introduced_uuids(p["num"])
        n_intro[p["num"]] = len(uu)
        for uuid in uu:
            introduced[uuid].append(p["num"])

    print("Stage 3b: scanning merged PRs for records landed on main ...")
    landed: dict[str, list[int]] = collections.defaultdict(list)  # uuid -> [merged nums]
    for num, oid in merged_oids.items():
        for uuid in merged_landed_uuids(num, oid):
            landed[uuid].append(num)

    print("Universe: enumerating records on main + reading content ...")
    universe = head_records()
    meta = read_meta_batch(universe)
    print(f"  {len(universe)} records on main")

    print("Stage 4: author orgs ...")
    author_org_map = build_author_org_map([p["author"] for p in prs], token)

    print("Stage 5: attributing submitters + landers + reconciling ...")
    rows = []
    for uuid, path in universe.items():
        rm = RECORD_RE.match(path)
        # A PR "added" this uuid if its branch introduced it OR its merge commit
        # landed it on main -- union both, since some merged-PR branches are gone
        # (the merge-commit evidence still proves that PR added the record).
        added_nums = sorted(set(introduced.get(uuid, [])) | set(landed.get(uuid, [])))
        # submitter = earliest non-maintainer, non-deleted PR that added it
        submitter_num = next(
            (n for n in added_nums
             if by_num[n]["author"] not in MAINTAINERS and by_num[n]["author"] != "deleted"),
            None,
        )
        # lander = latest merged PR that put this uuid on main
        land_nums = sorted(landed.get(uuid, []))
        lander_num = land_nums[-1] if land_nums else None

        if submitter_num is not None:
            sp = by_num[submitter_num]
            sub_author, sub_status = sp["author"], "attributed"
            sub_orgs = author_org_map.get(sub_author, [])
        else:
            sub_author, sub_status, sub_orgs = "", "submitter_unknown", []

        m = meta.get(uuid, {})
        source_org = m.get("source_organization_name")
        match, score = (best_org_match(sub_orgs, source_org) if sub_status == "attributed" else (None, 0.0))

        rows.append({
            "record_uuid": uuid,
            "path": path,
            "benchmark": rm["benchmark"] if rm else "",
            "developer": (rm["developer"] or "") if rm else "",
            "model": rm["model"] if rm else "",
            "source_organization_name": source_org,
            "source_type": m.get("source_type"),
            "submitter_status": sub_status,
            "submitter_pr": submitter_num or "",
            "submitter_author": sub_author,
            "submitter_pr_status": by_num[submitter_num]["status"] if submitter_num else "",
            "submitter_orgs": sub_orgs,
            "lander_pr": lander_num or "",
            "lander_author": by_num[lander_num]["author"] if lander_num else "",
            "lander_status": by_num[lander_num]["status"] if lander_num else "",
            "n_adding_prs": len(added_nums),
            "matched_org": match,
            "match_score": round(score, 1),
            "review_reason": review_reason(sub_orgs, source_org, match, sub_status),
            "evaluation_id": m.get("evaluation_id"),
        })
    rows.sort(key=lambda r: (r["benchmark"], r["path"]))

    pr_index = [{
        "num": p["num"], "author": p["author"], "status": p["status"],
        "title": p["title"],
        "branch_available": p["num"] in available,
        "is_maintainer": p["author"] in MAINTAINERS,
        "records_introduced": n_intro.get(p["num"], 0),
    } for p in prs]

    return {"author_org_map": author_org_map, "records": rows, "pr_index": pr_index}


def write_outputs(result: dict, outdir: Path) -> None:
    import csv, collections

    outdir.mkdir(exist_ok=True)
    (outdir / "author_org_map.json").write_text(json.dumps(result["author_org_map"], indent=2))

    def dump(path, records, list_cols=()):
        if not records:
            return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            for row in records:
                w.writerow({k: ("|".join(v) if k in list_cols else v) for k, v in row.items()})

    rows = result["records"]
    dump(outdir / "record_attribution.csv", rows, ("submitter_orgs",))
    dump(outdir / "pr_index.csv", result["pr_index"])

    statuses = collections.Counter(r["submitter_status"] for r in rows)
    reasons = collections.Counter(r["review_reason"] for r in rows)
    top_sub = collections.Counter(
        r["submitter_author"] for r in rows if r["submitter_status"] == "attributed"
    ).most_common(8)
    print(f"\nWrote {len(rows)} records -> {outdir/'record_attribution.csv'}")
    print(f"Wrote PR index -> {outdir/'pr_index.csv'}")
    print(f"Wrote author map -> {outdir/'author_org_map.json'}")
    print(f"  submitter_status: {dict(statuses)}")
    print(f"  review_reason: {dict(reasons)}")
    print(f"  top submitters (by records): {top_sub}")


def load_token() -> str | None:
    import os

    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


if __name__ == "__main__":
    tok = load_token()
    if not tok:
        raise SystemExit("No HF token found. Put HF_TOKEN=... in a .env file.")
    write_outputs(run(tok), OUT_DIR)

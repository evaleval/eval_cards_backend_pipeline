"""Apply the curated submitter<->source-org affiliation decisions onto the
per-record attribution, producing the `is_verified_evaluator` signal that feeds the
"validated evaluator" badge.

Badge rule: `is_verified_evaluator` is TRUE iff the evaluation was submitted by
the organisation that RAN it (`source_organization_name`) -- either because the
recovered SUBMITTER is a member of / named creator behind that org, or because
the org provided the data and a coalition member submitted it on their behalf
(one-off calls like UK AISI; say so in the reason). Volunteers and aggregators
re-hosting another org's official leaderboard are FALSE. Records with no
recoverable submitter (bulk direct-pushes) are FALSE, with the direct-commit
author recovered from git for provenance.

The decisions below are the human-owned source of truth, one entry per distinct
(submitter, source_organization_name) pair. Entries are standing rules: a pair
keeps applying to that submitter's future submissions for that org, and a new
unreviewed pair fails the completeness check until a human adds a call here.

Run AFTER `eee_org_mapping.py` (which writes the per-record attribution to
`.cache/verified_evaluator/output/`). Writes the two parquets straight into
`vendor/`, which Stage D joins at build time.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow"]
# ///

from __future__ import annotations

import collections
import csv
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".cache" / "verified_evaluator"
ATTRIB = CACHE_DIR / "output" / "record_attribution.csv"
REPO_DIR = CACHE_DIR / "repo"
VENDOR_DIR = REPO_ROOT / "vendor"

# (submitter_author, source_organization_name) pairs judged VALIDATED -- the
# submitter belongs to / created the org that ran the eval. Each carries evidence.
VALIDATED: dict[tuple[str, str], str] = {
    ("yifanmai", "crfm"):
        "On HF org stanford-crfm; Yifan Mai is the HELM/CRFM maintainer.",
    ("sanderland", "Writer, Inc."):
        "On HF org Writer; Sander Land is at Writer.",
    ("drchangliu", "Wordle Arena Project"):
        "Authored the Wordle/Fibble Arena eval and its paper.",
    ("drchangliu", "Dr. Chang Liu's Lab"):
        "Source org is his own lab.",
    ("StevenDillmann", "Terminal-Bench"):
        "Terminal-Bench paper author / TB-Science lead (Stanford+Laude); on harborframework.",
    ("Elron", "Exgentic"):
        "On HF org Exgentic; ran the Exgentic Open Agent Leaderboard.",
    ("madhavan113", "Mercor"):
        "On HF org mercor; PR titled [Mercor] APEX.",
    ("mrpfisher", "Arcadia Impact"):
        "Contributed inspect_evals evaluations; Arcadia Impact manages inspect_evals.",
    # One-off: submitter is evaleval coalition, NOT AISI -- but the data is
    # AISI's own inference-scaling eval runs, provided by AISI for submission
    # on their behalf (EEE_datastore PRs #184/#185/#187).
    ("deeplumiere", "UK AI Security Institute"):
        "AISI-provided inference-scaling runs, coalition-submitted on AISI's behalf (PRs #184/#185/#187).",
    ("deeplumiere", "UK AI Security Initiative"):
        "Same AISI-provided batch; one record the PR #206 spelling fix missed.",
    # Established first-party channels stay first-party when a later batch of
    # the same org's data is uploaded by a coalition member (decision 2026-08-20).
    ("evijit", "Terminal-Bench"):
        "Terminal-Bench established first-party (StevenDillmann); TB 2.0 batch coalition-uploaded.",
    ("jboat", "Stanford CRFM"):
        "CRFM established first-party (yifanmai); CRFM-run mmlu-winogrande-afr batch.",
    ("karthikchundi", "Stanford CRFM / Tatsu Lab"):
        "CRFM established first-party (yifanmai); alpaca_eval batch.",
    ("evijit", "Exgentic"):
        "On HF org Exgentic (same test as Elron); Exgentic's own leaderboard data.",
    ("jboat", "Institute for Disease Modeling"):
        "On HF org gatesfoundation; IDM is a Gates institute.",
    ("martin-ku-oup", "OUP"):
        "On HF orgs OUP / oup-ai-lab; OUP's own l2-bench.",
}

# Pairs explicitly reviewed and judged NOT validated (submitter re-hosting another
# org's leaderboard). Kept for the audit trail; the join also defaults any
# reviewed-but-not-validated pair to FALSE.
REVIEWED_UNVALIDATED: dict[tuple[str, str], str] = {
    ("simpod", "alphaXiv"):
        "Simon Podhajsky = EvalEval Coalition, not alphaXiv; re-hosted alphaXiv leaderboard.",
    ("mrshu", "Vals.ai"): "NaiveNeuron/coalition; no Vals.ai affiliation.",
    ("mrshu", "Human-Centered Eval"): "Same submitter; re-host.",
    ("Cerru02", "Artificial Analysis"): "Coalition aggregator re-hosting official leaderboards.",
    ("Cerru02", "LLM Stats"): "Coalition aggregator.",
    ("Cerru02", "ARC Prize"): "Coalition aggregator.",
    ("Cerru02", "UC Berkeley Gorilla"): "Coalition aggregator.",
    ("Cerru02", "Ai2"): "Coalition aggregator.",
    ("Cerru02", "CocoaBench"): "Coalition aggregator.",
    ("ameek", "TIGER-Lab"): "Aggregator; 3 unrelated source orgs.",
    ("ameek", "Scale"): "Aggregator.",
    ("ameek", "LMSYS"): "Aggregator.",
    ("jatinganhotra", "ByteDance-Seed"): "IBM researcher re-hosting official leaderboards.",
    ("jatinganhotra", "SWE-bench"): "Not on SWE-bench team; re-host of official leaderboard.",
    ("jatinganhotra", "AmazonScience"): "Re-host.",
    ("bwingenroth", "Google DeepMind"): "Johns Hopkins; re-host of FACTS Grounding.",
    ("bwingenroth", "CapArena (Cheng et al., 2025)"): "Re-host of CapArena.",
    ("Asaf-Yehudai", "Princeton SAgE Team"): "IBM/HUJI; not a HAL/SAgE author; re-host.",
    ("saki-imai", "LiveBench"): "Not on LiveBench team (Abacus.AI/NYU); coalition re-host.",
    ("lmushro", "La Leaderboard"): "No evidence of affiliation.",
    ("jboat", "WMT Conference"):
        "Re-hosted official WMT25 results; not among the 29 WMT25 organizers.",
    # 2026-08 batch (PRs ~#110-#199): coalition re-imports of orgs previously
    # False or with no established first-party channel.
    ("evijit", "Vals.ai"): "Coalition re-import; org previously False (mrshu).",
    ("evijit", "Human-Centered Eval"): "Coalition re-import; org previously False (mrshu).",
    ("evijit", "Artificial Analysis"): "Coalition re-import; org previously False (Cerru02).",
    ("evijit", "Princeton SAgE Team"): "Coalition re-import; org previously False (Asaf-Yehudai).",
    ("evijit", "ARC Prize Foundation"): "Coalition re-import; org previously False (Cerru02).",
    ("evijit", "Scale"): "Coalition re-import; org previously False (ameek).",
    ("evijit", "LMSYS"): "Coalition re-import; org previously False (ameek).",
    ("evijit", "LEXam-Benchmark"): "Coalition re-import; no first-party channel established.",
    ("evijit", "kaggle"): "Coalition re-import; no first-party channel established.",
    ("muhammadravi251001", "LMSYS"): "No LMSYS affiliation; re-host of arena results.",
    ("muhammadravi251001", "Tatsu Lab"): "No Tatsu Lab affiliation; re-host of alpacaeval results.",
    ("reuank", "LiveBench"): "Re-host; org previously False (saki-imai).",
    ("idoleaf", "Google Research"): "No affiliation evidence; re-host of weatherbench2 results.",
    ("bwingenroth", "Anthropic"): "Johns Hopkins; re-host of official mrcr results.",
    ("bwingenroth", "OpenAI"): "Johns Hopkins; re-host of official mrcr results.",
    ("bwingenroth", "Context Arena (independent project by Dillon Uzar)"):
        "Johns Hopkins; re-host of Context Arena results.",
    ("UsmanGohar", "LMArena (formerly LMSYS)"): "Coalition member; re-host of arena results.",
    ("mokarami", "Vectara"): "No public affiliation evidence; revisit if Vectara-confirmed.",
    ("anikethh", "ResearchGym"): "No public affiliation evidence; revisit if author-confirmed.",
}


UNKNOWN_ORG = {"unknown", "n/a", ""}

# Source orgs with an established first-party channel: any VALIDATED pair.
# Continuation batches of these orgs' data that arrive as bot/cron direct
# pushes (no PR submitter) stay first-party -- the channel was validated, the
# uploader is just plumbing (decision 2026-08-20, the "mercor is always
# first-party to us" rule). Aggregator orgs never get a VALIDATED pair, so
# their cron re-hosts stay False.
VALIDATED_ORGS = {org for (_, org) in VALIDATED}


def direct_commit_authors(paths: set[str]) -> dict[str, str]:
    """For each given data/ path, the author of the most recent commit that ADDED
    it on origin/main. Used to recover provenance for records that have no PR
    submitter (bulk direct-pushes). Read from the pinned local cache with NO fetch,
    so the curated snapshot is unchanged. One `git log` pass over data/."""
    if not (REPO_DIR / ".git").exists():
        return {}
    env = {"GIT_LFS_SKIP_SMUDGE": "1", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    out = subprocess.run(
        ["git", "-C", str(REPO_DIR), "log", "--diff-filter=A", "--no-renames",
         "--format=%x01%an", "--name-only", "origin/main", "--", "data"],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    author = None
    res: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("\x01"):
            author = line[1:]
        elif line in paths and line not in res:  # newest-first => latest add wins
            res[line] = author
    return res


def classify(submitter: str, source_org: str, attributed: bool,
             direct_author: str | None = None) -> tuple[bool, str]:
    """Return (is_verified_evaluator, reason)."""
    if not attributed:
        # No PR submitter. Bot/cron direct-pushes of an established first-party
        # org's data stay first-party; the rest of the bucket is consolidator
        # bulk-imports of external leaderboards -- NOT validated. Recover the
        # direct-commit author for provenance either way.
        if (source_org or "").strip() in VALIDATED_ORGS:
            return True, (
                f"continuation batch of established first-party channel"
                + (f" (direct-push by {direct_author})" if direct_author else "")
            )
        if direct_author:
            return False, f"bulk direct-push by {direct_author} (consolidator); not submitted by the source org"
        return False, "no PR submitter and no direct-commit author recovered"
    if (source_org or "").strip().lower() in UNKNOWN_ORG:
        return False, "source org unknown; cannot confirm affiliation"
    key = (submitter, source_org)
    if key in VALIDATED:
        return True, VALIDATED[key]
    return False, REVIEWED_UNVALIDATED.get(key, "reviewed; submitter not affiliated with the source org")


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = list(csv.DictReader(ATTRIB.open()))

    # Completeness guard: every attributed (submitter, source_org) pair must be
    # one we explicitly reviewed -- otherwise a new PR introduced an unclassified
    # pair and the mapping needs a human decision before we ship a badge for it.
    attributed_pairs = {
        (r["submitter_author"], r["source_organization_name"])
        for r in rows if r["submitter_status"] == "attributed"
        and (r["source_organization_name"] or "").strip().lower() not in UNKNOWN_ORG
    }
    reviewed = set(VALIDATED) | set(REVIEWED_UNVALIDATED)
    unclassified = attributed_pairs - reviewed
    if unclassified:
        raise SystemExit(
            "Unclassified attributed pairs (need a human call before badging):\n  "
            + "\n  ".join(f"{s!r} <- {o!r}" for s, o in sorted(unclassified))
        )

    # Recover the direct-commit author for records with no PR submitter, so the
    # bulk direct-push bucket carries provenance instead of a bare blank.
    unknown_paths = {r["path"] for r in rows if r["submitter_status"] != "attributed"}
    direct_authors = direct_commit_authors(unknown_paths)

    # Per-record: the join lookup (evaluation_id -> is_verified_evaluator) plus the
    # columns needed to audit any single record without rejoining anything.
    out_rows = []
    for r in rows:
        attributed = r["submitter_status"] == "attributed"
        validated, reason = classify(
            r["submitter_author"], r["source_organization_name"], attributed,
            direct_authors.get(r["path"]),
        )
        out_rows.append({
            "evaluation_id": r["evaluation_id"],
            "record_uuid": r["record_uuid"],
            "source_organization_name": r["source_organization_name"],
            "submitter_author": r["submitter_author"],
            "is_verified_evaluator": validated,
            "reason": reason,
        })

    out = VENDOR_DIR / "is_verified_evaluator.parquet"
    pq.write_table(pa.Table.from_pylist(out_rows), out)

    # Per (submitter, source_org): the human decisions with evidence.
    pair_counts = collections.Counter(
        (r["submitter_author"], r["source_organization_name"])
        for r in rows if r["submitter_status"] == "attributed"
    )
    pair_rows = []
    for (sub, org), n in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        validated, reason = classify(sub, org, True)
        pair_rows.append({
            "submitter_author": sub,
            "source_organization_name": org,
            "is_verified_evaluator": validated,
            "records": n,
            "reason": reason,
        })
    aff = VENDOR_DIR / "evaluator_affiliation.parquet"
    pq.write_table(pa.Table.from_pylist(pair_rows), aff)

    counts = collections.Counter(r["is_verified_evaluator"] for r in out_rows)
    print(f"Wrote {len(out_rows)} records -> {out}")
    print(f"Wrote {len(pair_rows)} pairs   -> {aff}")
    print(f"  is_verified_evaluator: {dict(counts)}")
    val_records = sum(n for (s, o), n in pair_counts.items() if (s, o) in VALIDATED)
    print(f"  validated records: {val_records} across {len(VALIDATED)} pairs")
    if direct_authors:
        da = collections.Counter(direct_authors.values())
        print(f"  recovered direct-push authors: {dict(da)}")


if __name__ == "__main__":
    main()

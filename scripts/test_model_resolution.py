"""Spike — test resolver coverage on every unique model_route_id in the corpus.

Reports: how many resolve, by which strategy, and which dupes I expected to
collapse actually do (or don't) under the registry as currently seeded.

Run after `scripts/build_local_model_aliases.py`.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

LOCAL_DIR = Path(".cache/local_registry_with_models").resolve()
os.environ["REGISTRY_LOCAL_PARQUET_DIR"] = str(LOCAL_DIR)

import duckdb

from eval_entity_resolver import AliasStore, Resolver

store = AliasStore.from_parquet(str(LOCAL_DIR), read_only=True)
print(f"loaded {len(store.to_dataframe())} alias rows from {LOCAL_DIR}")
resolver = Resolver(store)

con = duckdb.connect()
PAR = "output/duckdb/v1/evaluations.parquet"
routes = con.execute(
    f"""
    SELECT DISTINCT model_route_id FROM '{PAR}'
    WHERE model_route_id IS NOT NULL
    """
).fetchall()
print(f"{len(routes)} unique model_route_ids in corpus")


def to_hf_form(route_id: str) -> str:
    """`anthropic__claude-3-5-sonnet` -> `anthropic/claude-3-5-sonnet`."""
    return route_id.replace("__", "/", 1)


# Resolve each
results = []
strategies = Counter()
for (route,) in routes:
    raw = to_hf_form(route)
    res = resolver.resolve(raw, "model", None)
    results.append((route, raw, res.canonical_id, res.strategy, res.confidence))
    strategies[res.strategy] += 1

print()
print("=== Resolution strategy distribution ===")
for s, n in strategies.most_common():
    print(f"  {s:20s} {n:>5}")

unresolved = [r for r in results if r[2] is None]
print(f"\n=== UNRESOLVED ({len(unresolved)}) ===")

# Group by row count to prioritize
counts = dict(con.execute(
    f"SELECT model_route_id, COUNT(*) FROM '{PAR}' GROUP BY 1"
).fetchall())
unresolved_by_volume = sorted(unresolved, key=lambda r: -counts.get(r[0], 0))
for route, raw, _, _, _ in unresolved_by_volume[:30]:
    print(f"  {route:60s}  ({counts[route]:>5} rows)  raw={raw}")

# Print specific dupe-pair check
print()
print("=== Dupe verification — do the pairs from notes/model-aliases.yaml collapse? ===")
pairs = [
    ("deepseek-ai__deepseek-v3", "deepseek__deepseek-v3"),
    ("deepseek-ai__deepseek-r1", "deepseek__deepseek-r1"),
    ("mistralai__mistral-large-2407", "mistral__mistral-large-2407"),
    ("ai21__jamba-1-5-mini", "ai21-labs__jamba-1-5-mini"),
    ("ibm__granite-4-0-h-small", "unknown__granite-4-0-h-small"),
    ("anthropic__claude-3-5-sonnet", "anthropic__claude-35-sonnet"),
    ("moonshotai__kimi-k2-instruct", "moonshot-ai__kimi-k2-instruct"),
    ("qwen__qwen3-235b-a22b-instruct-2507", "alibaba__qwen3-235b-a22b-instruct-2507"),
    ("aleph-alpha__luminous-base-13b", "alephalpha__luminous-base"),
    ("tii-uae__falcon3-7b-instruct", "tiiuae__falcon3-7b-instruct"),
    ("google__gemma-3-4b", "google__gemma3-4b"),
    ("z-ai__glm-4-5", "zai__glm-4-5"),
    ("kimi__kimi-k2-5", "moonshotai__kimi-k2-5"),
    ("openchat__openchat-3-5", "openchat__openchat-35"),
    ("microsoft__phi4-reasoning-plus", "microsoft__phi-4-reasoning-plus"),
]
res_by_route = {r[0]: (r[2], r[3]) for r in results}
for a, b in pairs:
    ca, sa = res_by_route.get(a, (None, "missing-from-corpus"))
    cb, sb = res_by_route.get(b, (None, "missing-from-corpus"))
    same = "✓" if ca and cb and ca == cb else "✗"
    print(f"  [{same}]  {a:50s} -> {ca!r:35s} [{sa}]")
    print(f"        {b:50s} -> {cb!r:35s} [{sb}]")
    print()

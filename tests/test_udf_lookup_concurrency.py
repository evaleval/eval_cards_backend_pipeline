"""A UDF lookup answers the same on every DuckDB worker thread.

DuckDB calls a Python UDF from several threads at once. A lookup whose
index is still being built by one thread must not hand another thread an
empty or half-built index: that turns into NULL `known_issues` and
fallback `derived_tags` on whichever rows the other threads were scanning.

Each case runs in a fresh interpreter, because the index is process state.
The first caller is paused at a seam inside its first lookup; the second
caller then has to return the same answer a lone caller gets.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_PROBE = r"""
import importlib, json, pathlib, re, sys, threading, time

module_name, func_name, seam, args = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])

parked, release = threading.Event(), threading.Event()

def pausing(real):
    def wrapper(*a, **k):
        out = real(*a, **k)
        if threading.current_thread().name == "first" and not parked.is_set():
            parked.set()
            release.wait(10)
        return out
    return wrapper

if seam == "read_text":
    pathlib.Path.read_text = pausing(pathlib.Path.read_text)
else:
    re.sub = pausing(re.sub)

func = getattr(importlib.import_module(module_name), func_name)
results = {}

def call():
    results[threading.current_thread().name] = func(*args)

first = threading.Thread(target=call, name="first")
first.start()
# Stop waiting once the first caller is parked, or has returned without
# reaching the seam at all.
deadline = time.monotonic() + 5
while not parked.is_set() and first.is_alive() and time.monotonic() < deadline:
    parked.wait(0.01)
second = threading.Thread(target=call, name="second")
second.start()
second.join(5)
release.set()
first.join()
second.join()
results["alone"] = func(*args)
print(json.dumps(results))
"""

_LOOKUPS = {
    "known_issues": (
        "eval_card_backend.canonicalise.udfs", "lookup_known_issues_py", ["mmlu", "MMLU"],
    ),
    "derived_tags": (
        "eval_card_backend.canonicalise.evalcard_tags", "resolve_benchmark_tags_json", ["MMLU", "mmlu"],
    ),
}


@pytest.mark.parametrize("seam", ["read_text", "re_sub"])
@pytest.mark.parametrize("lookup", sorted(_LOOKUPS))
def test_second_thread_sees_full_index_while_first_is_mid_lookup(lookup, seam):
    module_name, func_name, args = _LOOKUPS[lookup]
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, module_name, func_name, seam, json.dumps(args)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout)
    assert results["alone"] not in (None, '["general"]'), (
        f"{args} no longer matches a registry entry, so this test proves nothing"
    )
    assert results["second"] == results["alone"], (
        f"{lookup}: a second thread calling while the first was paused at {seam} "
        f"got {results['second']!r}, a lone caller gets {results['alone']!r}"
    )
    assert results["first"] == results["alone"]

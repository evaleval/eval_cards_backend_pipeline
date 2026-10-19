"""`scripts/collections/check_flat_pin.py`: every way a pin is refused.

The Hub is a fake that serves one revision: a dataset commit date, a flat
descriptor, the `flat/indexes/by_collection/` listing, each collection's index
lines and the record tree under `data/<collection>/`. Each test removes or
thins one of those and asserts the exit code and the named reason, so an
empty or incomplete index can never read as "Pin OK".
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "collections" / "check_flat_pin.py"
)
REV = "a" * 40
COMMITTED = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def _load_script():
    spec = importlib.util.spec_from_file_location("check_flat_pin", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script = _load_script()


def _records(collection: str, n: int) -> list[str]:
    return [f"data/{collection}/dev/model/rec{i:03d}.json" for i in range(n)]


class FakeHub:
    """`collections` maps a collection name to (indexed paths, paths in the
    tree). `created_at` is the flat descriptor's timestamp."""

    def __init__(self, tmp_path, *, collections, created_at=COMMITTED,
                 committed=COMMITTED, manifest_missing=False, info_error=None):
        self.tmp_path = tmp_path
        self.collections = collections
        self.created_at = created_at
        self.committed = committed
        self.manifest_missing = manifest_missing
        self.info_error = info_error
        self.calls: list[tuple] = []

    def api(self):
        hub = self

        class _Api:
            def dataset_info(self, repo, revision=None, token=None):
                hub.calls.append(("info", revision))
                if hub.info_error is not None:
                    raise hub.info_error
                return SimpleNamespace(sha=revision, last_modified=hub.committed)

            def list_repo_tree(self, repo, repo_type, revision=None,
                               path_in_repo=None, recursive=False, token=None):
                hub.calls.append(("tree", path_in_repo))
                if path_in_repo == script.BY_COLLECTION:
                    return [
                        SimpleNamespace(path=f"{script.BY_COLLECTION}/{name}.jsonl")
                        for name in hub.collections
                    ]
                name = path_in_repo.removeprefix("data/")
                return [
                    SimpleNamespace(path=p) for p in hub.collections[name][1]
                ]

        return _Api()

    def download(self, repo_id, repo_type, revision, filename, token=None):
        self.calls.append(("download", filename))
        out = self.tmp_path / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        if filename == script.FLAT_MANIFEST:
            if self.manifest_missing:
                raise FileNotFoundError(filename)
            out.write_text(json.dumps({
                "created_at": self.created_at.isoformat()
                if self.created_at else None,
                "aggregate_file_count": sum(
                    len(indexed) for indexed, _ in self.collections.values()
                ),
            }))
        else:
            name = filename.rsplit("/", 1)[-1].removesuffix(".jsonl")
            out.write_text("".join(
                json.dumps({"legacy_path": p}) + "\n"
                for p in self.collections[name][0]
            ))
        return str(out)


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    def _run(hub: FakeHub, *args: str):
        monkeypatch.setattr(huggingface_hub, "HfApi", hub.api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.download)
        monkeypatch.setattr(
            sys, "argv", ["check_flat_pin.py", "--revision", REV, *args],
        )
        rc = script.main()
        return rc, capsys.readouterr().out

    return _run


def _complete(tmp_path, **kw):
    paths = _records("livebench", 5)
    return FakeHub(tmp_path, collections={"livebench": (paths, paths)}, **kw)


def test_a_complete_index_at_the_pin_passes(run, tmp_path):
    rc, out = run(_complete(tmp_path), "--require-collection", "livebench")
    assert rc == 0
    assert "Pin OK" in out
    assert "required collection livebench: ok" in out


def test_check_flat_pin_fails_on_an_index_that_leaves_records_out(run, tmp_path):
    """The flat rebuild indexes 2 of the 5 records under data/livebench."""
    paths = _records("livebench", 5)
    hub = FakeHub(tmp_path, collections={"livebench": (paths[:2], paths)})
    rc, out = run(hub, "--require-collection", "livebench")
    assert rc == 1
    assert "PIN REJECTED" in out
    assert "leaves out 3 of 5 record(s) under data/livebench" in out
    assert "Pin OK" not in out


def test_check_flat_pin_fails_on_an_empty_index(run, tmp_path):
    """An index file that exists but holds zero lines is not a listing."""
    paths = _records("livebench", 3)
    hub = FakeHub(tmp_path, collections={"livebench": ([], paths)})
    rc, out = run(hub, "--require-collection", "livebench")
    assert rc == 1
    assert "leaves out 3 of 3 record(s)" in out


def test_check_flat_pin_fails_on_a_missing_collection(run, tmp_path):
    rc, out = run(
        _complete(tmp_path),
        "--require-collection", "livebench",
        "--require-collection", "aisi-cyber-ctfs",
    )
    assert rc == 1
    assert "required collection aisi-cyber-ctfs: MISSING" in out
    assert "'aisi-cyber-ctfs' is not in the flat index at this pin" in out


def test_check_flat_pin_fails_on_an_empty_by_collection_tree(run, tmp_path):
    hub = FakeHub(tmp_path, collections={})
    rc, out = run(hub, "--require-collection", "livebench")
    assert rc == 1
    assert "'livebench' is not in the flat index at this pin" in out


def test_check_flat_pin_fails_on_a_missing_flat_manifest(run, tmp_path):
    hub = _complete(tmp_path, manifest_missing=True)
    rc, out = run(hub, "--require-collection", "livebench")
    assert rc == 1
    assert f"FAIL: no {script.FLAT_MANIFEST} at {REV}" in out
    # Nothing past the descriptor was consulted.
    assert ("tree", script.BY_COLLECTION) not in hub.calls


def test_check_flat_pin_fails_on_an_unresolvable_revision(run, tmp_path):
    hub = _complete(tmp_path, info_error=RuntimeError("404 Client Error"))
    rc, out = run(hub, "--require-collection", "livebench")
    assert rc == 1
    assert f"FAIL: cannot resolve {script.DEFAULT_REPO}@{REV}" in out
    assert hub.calls == [("info", REV)]


def test_check_flat_pin_fails_on_a_stale_flat_index(run, tmp_path):
    hub = _complete(tmp_path, created_at=COMMITTED - timedelta(hours=5))
    rc, out = run(hub, "--require-collection", "livebench")
    assert rc == 1
    assert "flat index lags the pin by 5.0h" in out


def test_check_flat_pin_fails_when_the_index_age_is_unknown(run, tmp_path):
    hub = _complete(tmp_path, created_at=None)
    rc, out = run(hub, "--require-collection", "livebench")
    assert rc == 1
    assert "cannot compare flat index age" in out


def test_lag_within_tolerance_passes(run, tmp_path):
    hub = _complete(tmp_path, created_at=COMMITTED - timedelta(minutes=50))
    rc, out = run(hub, "--require-collection", "livebench")
    assert rc == 0
    assert "flat lag        : 0.8h" in out

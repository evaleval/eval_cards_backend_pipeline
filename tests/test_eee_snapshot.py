"""`eee.ensure_snapshot` index + top-up tests.

The flat view (`flat/latest_manifest.json` -> verified `entries.jsonl`)
is the record index: aggregate rows give `legacy_path` (the `data/`
layout) plus a sha256 per object. The repo tree listing is the fallback
when the descriptor is absent. Top-up verifies objects on disk against
the sha map and never deletes anything, so a revision pin change
re-fetches only changed objects.

All network is faked: a per-revision dict of repo files stands in for
`hf_hub_download` / `HfApi`, mirroring the hub's on-disk side effects
(the object plus its `.cache/huggingface/download/<path>.metadata`).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest
from huggingface_hub.utils import EntryNotFoundError

from eval_card_backend.config import EEE_DATASET_REPO
from eval_card_backend.sources import eee as eee_src
from eval_card_backend.sources._revision_cache import _MARKER

REV1 = "a" * 40
REV2 = "b" * 40
COMMITTED = datetime(2026, 9, 16, 8, 35, 46, tzinfo=timezone.utc)
FLAT_CURRENT = "2026-09-16T09:00:00+00:00"
FLAT_STALE = "2026-09-16T05:59:00+00:00"

A = "data/cfg_a/dev/model/aaaa.json"
B = "data/cfg_b/dev/model/bbbb.json"
X = "data/alphaxiv/dev/model/xxxx.json"
ENTRIES = "flat/manifests/sha256_deadbeef/entries.jsonl"
FLAT = "flat/latest_manifest.json"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _entries(objects: dict[str, bytes], bad_sha: dict[str, str] | None = None) -> bytes:
    rows = []
    for p, b in objects.items():
        uuid = Path(p).stem
        rows.append({
            "object_uuid": uuid,
            "object_path": f"flat/objects/{uuid[:2]}/{uuid[2:4]}/{uuid}.json",
            "sha256": (bad_sha or {}).get(p, _sha(b)),
            "size_bytes": len(b),
            "legacy_path": p,
            "benchmark": p.split("/")[1],
            "record_type": "aggregate",
            "instance_level_available": False,
        })
    # A non-aggregate row must be skipped by the index.
    rows.append({
        "object_uuid": "inst", "object_path": "flat/objects/in/st/inst_samples.jsonl",
        "sha256": "0" * 64, "size_bytes": 1,
        "legacy_path": "data/cfg_a/dev/model/aaaa_samples.jsonl",
        "benchmark": "cfg_a", "record_type": "instance_level",
    })
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def _descriptor(entries: bytes, n_agg: int, created_at: str, *, size=None, sha=None) -> bytes:
    return json.dumps({
        "aggregate_file_count": n_agg,
        "created_at": created_at,
        "entries_path": ENTRIES,
        "entries_sha256": sha or _sha(entries),
        "entries_size_bytes": len(entries) if size is None else size,
        "source": {"path": "data", "type": "legacy_data_tree"},
    }).encode()


class FakeHub:
    """`{revision: {path: bytes}}` plus call recording."""

    def __init__(self, tmp_path: Path):
        self.revs: dict[str, dict[str, bytes]] = {}
        self.head = REV1
        self.hub_cache = tmp_path / "hub_cache"
        self.downloads: list[tuple[str, str | None]] = []
        self.tree_listings = 0
        self.info_calls = 0

    def download(self, repo_id, filename, repo_type, revision=None, local_dir=None, token=None):
        assert repo_id == EEE_DATASET_REPO and repo_type == "dataset"
        rev = revision or self.head
        self.downloads.append((filename, rev))
        try:
            content = self.revs[rev][filename]
        except KeyError:
            raise EntryNotFoundError(f"{filename}@{rev}")
        root = Path(local_dir) if local_dir else self.hub_cache / rev
        out = root / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)
        if local_dir:
            meta = root / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(f"{rev}\netag\n0\n")
        return str(out)

    def api(self):
        hub = self

        class _Info:
            def __init__(self, sha):
                self.sha = sha
                self.last_modified = COMMITTED

        class _Api:
            def dataset_info(self, repo_id, token=None, revision=None):
                hub.info_calls += 1
                return _Info(revision or hub.head)

            def list_repo_files(self, repo_id, repo_type, revision=None, token=None):
                hub.tree_listings += 1
                return sorted(hub.revs[revision or hub.head])

        return _Api()

    def objects_downloaded(self) -> list[str]:
        return [f for f, _ in self.downloads if f.startswith("data/")]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path)
    monkeypatch.setattr(eee_src, "hf_hub_download", fake.download)
    monkeypatch.setattr(eee_src, "HfApi", fake.api)
    for cache in (eee_src._index_cache, eee_src._info_cache, eee_src._listing_cache):
        cache.clear()
    yield fake
    for cache in (eee_src._index_cache, eee_src._info_cache, eee_src._listing_cache):
        cache.clear()


def _seed_rev(
    hub: FakeHub, rev: str, objects: dict[str, bytes], *,
    flat=True, created_at=FLAT_CURRENT, bad_sha=None, entries_size=None, entries_sha=None,
):
    files = dict(objects)
    files["data/cfg_a/dev/model/aaaa_samples.jsonl"] = b"{}\n"
    files["manifest.json"] = b'{"files": {}}'  # frozen root artifact, must never be read
    if flat:
        entries = _entries(objects, bad_sha)
        files[ENTRIES] = entries
        files[FLAT] = _descriptor(
            entries, len(objects), created_at, size=entries_size, sha=entries_sha
        )
    hub.revs[rev] = files


OBJECTS = {A: b'{"a": 1}', B: b'{"b": 1}', X: b'{"x": 1}'}


def _listing(target: Path) -> dict:
    return json.loads((target / ".eee_file_listing.json").read_text())


def _run(hub, tmp_path, revision, force=False, name="snap"):
    return eee_src.ensure_snapshot(str(tmp_path / name), None, force, revision=revision)


def test_listing_derived_from_flat_entries(hub, tmp_path):
    _seed_rev(hub, REV1, OBJECTS)
    target = _run(hub, tmp_path, REV1)

    listing = _listing(target)
    assert listing["revision"] == REV1
    # Non-aggregate row excluded, every config kept (IGNORED_CONFIGS applies at download).
    assert listing["paths"] == sorted(OBJECTS)
    assert listing["sha256"] == {p: _sha(b) for p, b in OBJECTS.items()}
    assert hub.tree_listings == 0
    assert all(f != "manifest.json" for f, _ in hub.downloads)
    # Objects come from legacy_path; alphaxiv is listed but never downloaded.
    assert sorted(hub.objects_downloaded()) == [A, B]
    assert not (target / X).exists()
    assert (target / A).read_bytes() == OBJECTS[A]
    assert eee_src.discover_configs(target, None) == ["alphaxiv", "cfg_a", "cfg_b"]
    assert eee_src.list_json_files("cfg_a", target, None) == [A]
    assert eee_src.cached_revision(target) == REV1
    assert (target / _MARKER).read_text() == REV1


@pytest.mark.parametrize("kw, needle", [
    ({"entries_sha": "f" * 64}, "sha256 mismatch"),
    ({"entries_size": 1}, "size mismatch"),
])
def test_entries_file_verification_failure_raises(hub, tmp_path, kw, needle):
    _seed_rev(hub, REV1, OBJECTS, **kw)
    with pytest.raises(RuntimeError, match=f"{ENTRIES}.*{needle}"):
        _run(hub, tmp_path, REV1)
    assert hub.objects_downloaded() == []


def test_object_sha_mismatch_after_download_raises_naming_path(hub, tmp_path):
    _seed_rev(hub, REV1, OBJECTS, bad_sha={B: "0" * 64})
    with pytest.raises(RuntimeError, match=B):
        _run(hub, tmp_path, REV1)


def test_pin_change_refetches_only_changed_objects(hub, tmp_path):
    _seed_rev(hub, REV1, OBJECTS)
    target = _run(hub, tmp_path, REV1)
    a_mtime = (target / A).stat().st_mtime_ns

    # REV2: B changed, A unchanged, a new object C appears, X dropped upstream.
    C = "data/cfg_c/dev/model/cccc.json"
    _seed_rev(hub, REV2, {A: OBJECTS[A], B: b'{"b": 2}', C: b'{"c": 1}'})
    hub.downloads.clear()

    assert _run(hub, tmp_path, REV2) == target
    assert sorted(hub.objects_downloaded()) == [B, C]
    assert (target / A).stat().st_mtime_ns == a_mtime
    assert (target / B).read_bytes() == b'{"b": 2}'
    assert (target / X).exists() is False  # never downloaded (ignored config)
    assert _listing(target)["revision"] == REV2
    assert (target / _MARKER).read_text() == REV2
    assert hub.tree_listings == 0


def test_stale_object_on_disk_is_replaced_and_nothing_deleted(hub, tmp_path):
    _seed_rev(hub, REV1, OBJECTS)
    target = _run(hub, tmp_path, REV1)
    (target / A).write_bytes(b"corrupted")
    stray = target / "data/cfg_a/dev/model/stray.json"
    stray.write_text("{}")
    hub.downloads.clear()

    _run(hub, tmp_path, REV1)
    assert hub.objects_downloaded() == [A]
    assert (target / A).read_bytes() == OBJECTS[A]
    assert stray.exists()


def test_rerun_same_revision_is_offline(hub, tmp_path):
    _seed_rev(hub, REV1, OBJECTS)
    _run(hub, tmp_path, REV1)
    hub.downloads.clear()
    hub.info_calls = 0
    _run(hub, tmp_path, REV1)
    assert hub.downloads == [] and hub.info_calls == 0


def test_old_listing_without_sha_map_still_loads(hub, tmp_path):
    target = tmp_path / "snap"
    (target / "data/cfg_a/dev/model").mkdir(parents=True)
    (target / A).write_bytes(OBJECTS[A])
    meta = target / ".cache/huggingface/download" / f"{A}.metadata"
    meta.parent.mkdir(parents=True)
    meta.write_text(f"{REV1}\netag\n0\n")
    (target / ".eee_file_listing.json").write_text(
        json.dumps({"revision": REV1, "paths": [A, B]})
    )
    _seed_rev(hub, REV1, OBJECTS)

    out = eee_src.ensure_snapshot(str(target), None, False, revision=None)
    assert out == target.resolve()
    assert eee_src.cached_revision(target) == REV1
    assert eee_src._local_paths(target) == [A, B]
    # No re-index (listing present, unpinned): only the missing object is fetched.
    assert hub.downloads == [(B, REV1)]
    assert (target / B).exists()


def test_fallback_to_repo_tree_when_flat_descriptor_missing(hub, tmp_path, caplog):
    _seed_rev(hub, REV1, OBJECTS, flat=False)
    with caplog.at_level(logging.WARNING, logger=eee_src.__name__):
        target = _run(hub, tmp_path, REV1)
    assert hub.tree_listings == 1
    assert any(f"{FLAT} missing" in r.getMessage() for r in caplog.records)
    listing = _listing(target)
    assert sorted(listing["paths"]) == sorted(OBJECTS)
    assert listing["sha256"] == {}
    assert sorted(hub.objects_downloaded()) == [A, B]


def test_flat_lag_behind_pinned_commit_warns(hub, tmp_path, caplog):
    _seed_rev(hub, REV1, OBJECTS, created_at=FLAT_STALE)
    with caplog.at_level(logging.INFO, logger=eee_src.__name__):
        _run(hub, tmp_path, REV1)
    warn = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn) == 1
    msg = warn[0].getMessage()
    assert msg.startswith("flat index lags the pinned commit by 2.6 hours")
    assert "pin a flat-rebuild commit" in msg
    info = [r.getMessage() for r in caplog.records if "flat view" in r.getMessage()]
    assert info and FLAT_STALE in info[0] and "2026-09-16T08:35:46+00:00" in info[0]


def test_current_flat_index_does_not_warn(hub, tmp_path, caplog):
    _seed_rev(hub, REV1, OBJECTS, created_at=FLAT_CURRENT)
    with caplog.at_level(logging.INFO, logger=eee_src.__name__):
        _run(hub, tmp_path, REV1)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_unverifiable_object_refetched_when_pin_changes(hub, tmp_path):
    _seed_rev(hub, REV1, OBJECTS, flat=False)
    target = _run(hub, tmp_path, REV1)
    _seed_rev(hub, REV2, {A: OBJECTS[A], B: b'{"b": 2}'}, flat=False)
    hub.downloads.clear()

    _run(hub, tmp_path, REV2)
    # No sha to compare: everything fetched at REV1 is refetched at REV2.
    assert sorted(hub.objects_downloaded()) == [A, B]
    assert (target / B).read_bytes() == b'{"b": 2}'


def test_index_fetched_once_per_revision_per_process(hub, tmp_path):
    _seed_rev(hub, REV1, OBJECTS)
    _run(hub, tmp_path, REV1, name="one")
    _run(hub, tmp_path, REV1, name="two")
    assert sum(1 for f, _ in hub.downloads if f == FLAT) == 1
    assert sum(1 for f, _ in hub.downloads if f == ENTRIES) == 1
    assert hub.info_calls == 1


def test_unpinned_head_is_resolved_to_sha_and_marker_cleared(hub, tmp_path):
    _seed_rev(hub, REV1, OBJECTS)
    target = _run(hub, tmp_path, REV1)
    assert (target / _MARKER).exists()

    hub.head = REV2
    _seed_rev(hub, REV2, OBJECTS)
    _run(hub, tmp_path, None, force=True)
    assert _listing(target)["revision"] == REV2
    assert not (target / _MARKER).exists()
    assert (target / A).exists()
    # HEAD lookup answered both the sha resolution and the commit-date lookup.
    assert hub.info_calls == 2


def test_hand_built_tree_never_touches_network(hub, tmp_path):
    target = tmp_path / "snap"
    (target / "data/cfg_a/dev/model").mkdir(parents=True)
    (target / A).write_bytes(OBJECTS[A])
    out = eee_src.ensure_snapshot(str(target), None, False, revision=None)
    assert out == target.resolve()
    assert hub.downloads == [] and hub.tree_listings == 0 and hub.info_calls == 0
    assert eee_src._local_paths(target) == [A]

"""Processed private collection extracts behind `EEE_INCLUDE_PRIVATE`.

The private dataset serves processed extracts under `collections/<name>/`.
Their aggregate member records may also be in the public EEE corpus. Toggle
on drops those public member result rows and injects the complete processed
study. Toggle off does not read the private source and leaves public members
as ordinary records.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import duckdb
import huggingface_hub
import pytest

from tests.eee_layout import write_eee_datastore
from tests.test_canonicalise_e2e import (
    _write_cards_fixture,
    _write_registry_fixture,
)
from tests.test_collections import (
    STUDY_ORG_A,
    _load_extractor_module,
    _ordinary_record,
    _study_record,
    _write_vendor_fixture,
)

from eval_card_backend.config import (
    EEE_DATASET_REPO,
    EEE_PRIVATE_DATASET_REPO,
    Settings,
)
from eval_card_backend.sources import collections as collections_src
from eval_card_backend.sources._revision_cache import _MARKER

PRIVATE_SHA = "c" * 40
OTHER_SHA = "e" * 40
SNAPSHOT = "2026-04-30T00:00:00Z"
STUDY_IDS = {"minibench/model/a", "minibench/model/b", "minibench/model/c"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_toggle_defaults_off():
    s = Settings.from_env()
    assert s.include_private_eee is False
    assert s.eee_private_allow_zero_members is False
    assert s.eee_private_revision is None


def test_zero_member_rollout_setting(monkeypatch):
    monkeypatch.setenv("EEE_PRIVATE_ALLOW_ZERO_MEMBERS", "1")
    assert Settings.from_env().eee_private_allow_zero_members is True


@pytest.mark.parametrize("value", ["main", "v1.0", "c" * 39, "C" * 40])
def test_private_revision_must_be_a_commit_sha(monkeypatch, value):
    monkeypatch.setenv("EEE_PRIVATE_REVISION", value)
    with pytest.raises(ValueError, match="commit sha"):
        Settings.from_env()


def test_private_revision_sha_accepted(monkeypatch):
    monkeypatch.setenv("EEE_INCLUDE_PRIVATE", "1")
    monkeypatch.setenv("EEE_PRIVATE_REVISION", PRIVATE_SHA)
    s = Settings.from_env()
    assert (s.include_private_eee, s.eee_private_revision) == (True, PRIVATE_SHA)


# ---------------------------------------------------------------------------
# Download step
# ---------------------------------------------------------------------------


class _PrivateHub:
    """Fake hub serving `{revision: {path: bytes}}` for the private repo;
    records every call."""

    def __init__(self, revs: dict[str, dict[str, bytes]], *, deny: bool = False):
        self.revs = revs
        self.head = next(iter(revs)) if revs else PRIVATE_SHA
        self.deny = deny
        self.calls: list[tuple] = []

    def download(self, repo_id, filename, repo_type, revision=None,
                 local_dir=None, token=None):
        self.calls.append(("download", repo_id, filename, revision))
        assert repo_id == EEE_PRIVATE_DATASET_REPO and repo_type == "dataset"
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.revs[revision][filename])
        return str(out)

    def api(self):
        hub = self

        class _Api:
            def dataset_info(self, repo_id, token=None, revision=None):
                hub.calls.append(("info", repo_id, revision))
                if hub.deny:
                    raise RuntimeError("401 Client Error: Unauthorized")
                return SimpleNamespace(sha=revision or hub.head, last_modified=None)

            def list_repo_files(self, repo_id, repo_type, revision=None, token=None):
                hub.calls.append(("list", repo_id, revision))
                return sorted(hub.revs[revision])

        return _Api()


@pytest.fixture
def private_hub(monkeypatch):
    def install(revs, **kw):
        hub = _PrivateHub(revs, **kw)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.download)
        monkeypatch.setattr(huggingface_hub, "HfApi", hub.api)
        return hub
    return install


EXTRACT = {
    "collections/study/manifest.json": b"{}",
    "collections/study/results.parquet": b"r",
    "collections/study/trajectories.parquet": b"t",
    "README.md": b"",
}


def test_download_fetches_collections_only_and_marks_revision(tmp_path, private_hub):
    hub = private_hub({PRIVATE_SHA: EXTRACT})
    coll, rev = collections_src.ensure_private_collections(
        str(tmp_path / "priv"), None, False
    )
    assert rev == PRIVATE_SHA
    assert coll == (tmp_path / "priv" / "collections").resolve()
    assert sorted(c[2] for c in hub.calls if c[0] == "download") == [
        "collections/study/manifest.json",
        "collections/study/results.parquet",
        "collections/study/trajectories.parquet",
    ]
    assert {c[1] for c in hub.calls} == {EEE_PRIVATE_DATASET_REPO}
    assert (tmp_path / "priv" / _MARKER).read_text() == PRIVATE_SHA


def test_pinned_rerun_needs_no_network(tmp_path, private_hub):
    private_hub({PRIVATE_SHA: EXTRACT})
    collections_src.ensure_private_collections(
        str(tmp_path / "priv"), None, False, PRIVATE_SHA
    )
    hub = private_hub({})
    collections_src.ensure_private_collections(
        str(tmp_path / "priv"), None, False, PRIVATE_SHA
    )
    assert hub.calls == []


def test_revision_change_replaces_the_collections_dir(tmp_path, private_hub):
    private_hub({PRIVATE_SHA: EXTRACT})
    coll, _ = collections_src.ensure_private_collections(
        str(tmp_path / "priv"), None, False, PRIVATE_SHA
    )
    newer = {
        "collections/renamed/manifest.json": b"{}",
        "collections/renamed/results.parquet": b"r",
    }
    private_hub({OTHER_SHA: newer})
    coll, rev = collections_src.ensure_private_collections(
        str(tmp_path / "priv"), None, False, OTHER_SHA
    )
    assert rev == OTHER_SHA
    assert sorted(p.name for p in coll.iterdir()) == ["renamed"]


def test_hand_built_dir_used_as_is(tmp_path, private_hub):
    hub = private_hub({})
    manifest = tmp_path / "priv" / "collections" / "study" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    coll, rev = collections_src.ensure_private_collections(
        str(tmp_path / "priv"), None, False
    )
    assert coll == manifest.parent.parent.resolve() and rev is None
    assert hub.calls == []


def test_access_error_names_repo_and_toggle(tmp_path, private_hub):
    private_hub({}, deny=True)
    with pytest.raises(RuntimeError, match="EEE_INCLUDE_PRIVATE") as exc:
        collections_src.ensure_private_collections(str(tmp_path / "priv"), None, False)
    assert EEE_PRIVATE_DATASET_REPO in str(exc.value)


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------


def _study_files():
    rec_a = _study_record(
        "minibench/model/a", STUDY_ORG_A,
        "accuracy on minibench/S-adaptive/+1ep for scorer x", 0.2,
    )
    rec_b = _study_record(
        "minibench/model/b", STUDY_ORG_A,
        "accuracy on minibench/S-adaptive/+2ep/abc for scorer x", 1.0,
    )
    rec_b["model_info"] = {
        "developer": "openai",
        "name": "openai/gpt-4o-mini",
        "id": "openai/gpt-4o-mini",
    }
    return [
        ("minibench", "rec_a.json", json.dumps(rec_a)),
        ("minibench", "rec_b.json", json.dumps(rec_b)),
    ]


def _write_curated(path: Path) -> None:
    path.write_text(
        "test-study:\n"
        "  display_name: Test Study\n"
        "  kind: paper_study\n"
        "  curated: true\n"
        "  private_source:\n"
        "    composites: [mini-study]\n"
        "  merge_raw_keys:\n"
        "    - test-aisi-institute/mini-study-paper\n"
        "    - test-aisi-initiative/mini-study-paper\n"
        "  protocol_axes:\n"
        "    - {key: feedback, type: categorical, values: [none, answer_feedback, unknown]}\n"
    )


def _write_seed(seed_root: Path) -> None:
    """A taxonomy composite scoped to the study's org on a config the public
    root also produces: off, it matches nothing while its config is live."""
    seed_root.mkdir(parents=True, exist_ok=True)
    (seed_root / "composites.yaml").write_text(
        "minibench:\n  display: MiniBench\n  configs:\n    - minibench\n"
        "mini-study:\n  display: Mini Study\n  configs:\n"
        "    - {config: minibench, org: test-aisi-institute}\n"
        "    - {config: minibench, org: test-aisi-initiative}\n"
    )


@pytest.fixture
def world(tmp_path):
    write_eee_datastore(
        tmp_path / "eee",
        [("minibench", "ordinary.json", json.dumps(_ordinary_record()))]
        + _study_files(),
    )
    members = [
        {"record_uuid": "rec_a", "evaluation_id": "minibench/model/a",
         "path": "data/minibench/dev/model/rec_a.json",
         "config": "minibench", "n_results": 1},
        {"record_uuid": "rec_b", "evaluation_id": "minibench/model/b",
         "path": "data/minibench/dev/model/rec_b.json",
         "config": "minibench", "n_results": 1},
        {"record_uuid": "rec_c", "evaluation_id": "minibench/model/c",
         "path": "data/minibench/dev/model/rec_c.json",
         "config": "minibench", "n_results": 0},
    ]
    _write_vendor_fixture(
        tmp_path / "eee_private" / "collections",
        manifest_overrides={
            "source_repo": EEE_PRIVATE_DATASET_REPO, "eee_revision": None,
        },
        members=members,
    )
    (tmp_path / "vendor_collections").mkdir()
    _write_curated(tmp_path / "curated.yaml")
    _write_registry_fixture(tmp_path / "reg")
    _write_seed(tmp_path / "seed")
    _write_cards_fixture(tmp_path / "cards")
    return tmp_path


def _run(
    world: Path,
    monkeypatch,
    *,
    include_private: bool,
    allow_zero_members: bool = False,
    from_stage: str | None = None,
    private_dir: Path | None = None,
):
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(world / "eee"))
    monkeypatch.setenv(
        "EEE_PRIVATE_LOCAL_DATASET_DIR", str(private_dir or world / "eee_private")
    )
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(world / "cards"))
    monkeypatch.setenv("COLLECTIONS_VENDOR_DIR", str(world / "vendor_collections"))
    monkeypatch.setenv("COLLECTIONS_CURATED_PATH", str(world / "curated.yaml"))
    for var in ("EEE_REFRESH_SNAPSHOT", "BENCHMARK_METADATA_REFRESH"):
        monkeypatch.delenv(var, raising=False)
    if include_private:
        monkeypatch.setenv("EEE_INCLUDE_PRIVATE", "1")
    else:
        monkeypatch.delenv("EEE_INCLUDE_PRIVATE", raising=False)
    if allow_zero_members:
        monkeypatch.setenv("EEE_PRIVATE_ALLOW_ZERO_MEMBERS", "1")
    else:
        monkeypatch.delenv("EEE_PRIVATE_ALLOW_ZERO_MEMBERS", raising=False)

    from eval_card_backend.canonicalise import pipeline

    return pipeline.run(
        Settings.from_env(),
        snapshot_id=SNAPSHOT,
        warehouse_dir=str(world / "warehouse"),
        registry_local_dir=str(world / "reg"),
        taxonomy_seed_dir=str(world / "seed"),
        cache_root=str(world / "cache"),
        from_stage=from_stage,
    )


def _refuse_private(monkeypatch):
    def _refuse(*_a, **_k):
        raise AssertionError("private collections requested with the toggle off")
    monkeypatch.setattr(collections_src, "ensure_private_collections", _refuse)


@pytest.fixture
def no_private_access(monkeypatch):
    _refuse_private(monkeypatch)


def _facts(out: Path) -> list[tuple]:
    return duckdb.connect().execute(
        f"SELECT evaluation_id, collection_id, score "
        f"FROM read_parquet('{out}/fact_results.parquet') ORDER BY score"
    ).fetchall()


def _urls(out: Path) -> dict[float, tuple]:
    rows = duckdb.connect().execute(
        f"SELECT round(score, 4), eee_record_url, instance_file_url "
        f"FROM read_parquet('{out}/eval_results_view.parquet')"
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _warehouse_text(out: Path) -> str:
    return "\n".join(
        p.read_bytes().decode("latin-1") for p in sorted(out.iterdir()) if p.is_file()
    )


# ---------------------------------------------------------------------------
# Toggle on
# ---------------------------------------------------------------------------


def test_on_drops_public_members_and_injects_complete_extract(
    world, monkeypatch, caplog
):
    with caplog.at_level("INFO", logger="eval_card_backend.sources.collections"):
        out = _run(world, monkeypatch, include_private=True)
    facts = _facts(out)
    # Raw public members (0.2, 1.0) are gone. Both synthetic cells, including
    # the cell represented by absent zero-result member c, are injected.
    assert [r[2] for r in facts] == [0.5, 0.6, 0.95]
    assert {r[1] for r in facts if r[0] in STUDY_IDS} == {"test-study"}
    assert "public members 2/3 present" in caplog.text
    assert "1/1 absent members account for zero rows" in caplog.text
    assert "dropped 2/2 manifest result row(s)" in caplog.text
    traj = duckdb.connect().execute(
        f"SELECT collection_id, count(*) "
        f"FROM read_parquet('{out}/collection_trajectories.parquet') GROUP BY 1"
    ).fetchall()
    assert traj == [("test-study", 2)]
    collections = json.loads((out / "collections.json").read_text())
    assert collections["test-study"]["curated"] is True
    assert "private_source" not in collections["test-study"]


def test_on_links_stay_null_for_private_derived_rows(world, monkeypatch):
    out = _run(world, monkeypatch, include_private=True)
    urls = _urls(out)
    assert urls[0.6][0] == (
        f"https://huggingface.co/datasets/{EEE_DATASET_REPO}/resolve/main/"
        "data/minibench/dev/model/ordinary.json"
    )
    assert urls[0.5] == (None, None)
    assert urls[0.95] == (None, None)


def test_on_zero_public_members_fails_without_rollout_setting(world, monkeypatch):
    for name in ("rec_a.json", "rec_b.json"):
        (world / "eee" / "data" / "minibench" / "dev" / "model" / name).unlink()
    monkeypatch.setenv("EEE_REVISION", OTHER_SHA)
    (world / "eee" / _MARKER).write_text(OTHER_SHA)
    with pytest.raises(RuntimeError, match=(
        f"datastore pin {OTHER_SHA}.*EEE_PRIVATE_ALLOW_ZERO_MEMBERS"
    )):
        _run(world, monkeypatch, include_private=True)


def test_on_zero_public_members_allowed_during_rollout(world, monkeypatch, caplog):
    for name in ("rec_a.json", "rec_b.json"):
        (world / "eee" / "data" / "minibench" / "dev" / "model" / name).unlink()
    with caplog.at_level("WARNING", logger="eval_card_backend.sources.collections"):
        out = _run(
            world,
            monkeypatch,
            include_private=True,
            allow_zero_members=True,
        )
    assert [r[2] for r in _facts(out)] == [0.5, 0.6, 0.95]
    assert _urls(out)[0.5] == (None, None)
    assert _urls(out)[0.95] == (None, None)
    assert "EEE_PRIVATE_ALLOW_ZERO_MEMBERS=1" in caplog.text


def test_on_upstream_pin_records_private_revision(world, monkeypatch):
    from eval_card_backend.canonicalise import pipeline

    monkeypatch.setattr(pipeline, "_hf_dataset_snapshot", lambda *a, **k: None)
    (world / "eee_private" / _MARKER).write_text(PRIVATE_SHA)
    monkeypatch.setenv("EEE_PRIVATE_REVISION", PRIVATE_SHA)
    out = _run(world, monkeypatch, include_private=True)
    for name in ("snapshot_meta.json", "manifest.json"):
        payload = json.loads((out / name).read_text())
        pin = payload["upstream_pins"]["eee_private"]
        assert pin["repo_id"] == EEE_PRIVATE_DATASET_REPO
        assert pin["sha"] == PRIVATE_SHA
        assert payload["collection_adapter_accounting"] == {
            "test-study": {
                "members_total": 3,
                "members_present": 2,
                "members_absent": 1,
                "members_absent_nonzero": 0,
                "expected_drop": 2,
                "dropped": 2,
                "injected": 2,
                "private_revision": PRIVATE_SHA,
                "fact_results": 2,
                "eval_results_view": 2,
            }
        }


def test_on_partial_public_ingest_missing_nonempty_member_fails(world, monkeypatch):
    (world / "eee" / "data" / "minibench" / "dev" / "model" / "rec_b.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete study ingest"):
        _run(world, monkeypatch, include_private=True)


def test_present_members_with_offsetting_count_errors_fail_before_delete(world):
    manifest_path = (
        world / "eee_private" / "collections" / "test_study" / "manifest.json"
    )
    con = duckdb.connect()
    con.execute("CREATE TABLE eee_raw (evaluation_id VARCHAR)")
    con.execute(
        "INSERT INTO eee_raw VALUES ('minibench/model/a'), ('minibench/model/b')"
    )
    con.execute("CREATE TABLE results_exploded (evaluation_id VARCHAR)")
    con.execute(
        "INSERT INTO results_exploded VALUES "
        "('minibench/model/b'), ('minibench/model/b')"
    )
    collections_src.create_collection_tables(con)

    with pytest.raises(RuntimeError, match="result count mismatch") as exc:
        collections_src._apply_one_adapter(
            con,
            manifest_path,
            world / "eee",
            OTHER_SHA,
            private_extract=True,
            private_revision=PRIVATE_SHA,
        )
    assert "minibench/model/a: manifest=1, observed=0" in str(exc.value)
    assert "minibench/model/b: manifest=1, observed=2" in str(exc.value)
    assert con.execute(
        "SELECT evaluation_id FROM results_exploded ORDER BY evaluation_id"
    ).fetchall() == [("minibench/model/b",), ("minibench/model/b",)]


def test_old_and_new_uuid_for_one_evaluation_id_fail_before_delete(world):
    manifest_path = (
        world / "eee_private" / "collections" / "test_study" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["members"] = [manifest["members"][0]]
    manifest["expected_drop_count"] = 1
    manifest_path.write_text(json.dumps(manifest))

    con = duckdb.connect()
    con.execute("CREATE TABLE eee_raw (evaluation_id VARCHAR, record_uuid VARCHAR)")
    con.execute(
        "INSERT INTO eee_raw VALUES "
        "('minibench/model/a', 'old-uuid'), "
        "('minibench/model/a', 'new-uuid')"
    )
    con.execute("CREATE TABLE results_exploded (evaluation_id VARCHAR)")
    con.execute(
        "INSERT INTO results_exploded VALUES "
        "('minibench/model/a'), ('minibench/model/a')"
    )
    collections_src.create_collection_tables(con)

    with pytest.raises(
        RuntimeError, match="minibench/model/a: manifest=1, observed=2"
    ):
        collections_src._apply_one_adapter(
            con,
            manifest_path,
            world / "eee",
            OTHER_SHA,
            private_extract=True,
            private_revision=PRIVATE_SHA,
        )
    assert con.execute("SELECT count(*) FROM results_exploded").fetchone()[0] == 2


def test_present_members_with_matching_counts_pass():
    member_counts = {"a": 1, "b": 3, "c": 0}
    con = duckdb.connect()
    con.execute("CREATE TABLE results_exploded (evaluation_id VARCHAR)")
    rows = [
        (evaluation_id,)
        for evaluation_id, count in member_counts.items()
        for _ in range(count)
    ]
    con.executemany("INSERT INTO results_exploded VALUES (?)", rows)

    assert collections_src._validate_present_member_result_counts(
        con,
        "test-study",
        set(member_counts),
        member_counts,
    ) == 4


def test_on_leak_guard_fires_for_unknown_empty_public_record(world, monkeypatch):
    unknown = _study_record(
        "minibench/model/unknown", STUDY_ORG_A, "minibench", 0.1
    )
    unknown["evaluation_results"] = []
    write_eee_datastore(world / "eee", [
        ("minibench", "unknown.json", json.dumps(unknown)),
    ])
    with pytest.raises(RuntimeError, match="leak guard"):
        _run(world, monkeypatch, include_private=True)


def test_on_cache_resume_matches(world, monkeypatch):
    out = _run(world, monkeypatch, include_private=True)
    first = (out / "eval_results_view.parquet").read_bytes()
    out = _run(world, monkeypatch, include_private=True, from_stage="D")
    assert (out / "eval_results_view.parquet").read_bytes() == first


# ---------------------------------------------------------------------------
# Toggle off
# ---------------------------------------------------------------------------


def test_off_never_touches_private_source(world, monkeypatch, no_private_access):
    missing = world / "never_created"
    out = _run(
        world,
        monkeypatch,
        include_private=False,
        allow_zero_members=True,
        private_dir=missing,
    )
    assert not missing.exists()

    facts = _facts(out)
    assert [r[0] for r in facts] == [
        "minibench/model/a", "minibench/ordinary/1", "minibench/model/b",
    ]
    assert {r[1] for r in facts if r[0] in STUDY_IDS} == {"test-study"}
    assert not (out / "collection_trajectories.parquet").exists()
    assert not (out / "collection_context.json").exists()
    for name in ("snapshot_meta.json", "manifest.json"):
        assert "eee_private" not in json.loads((out / name).read_text())["upstream_pins"]
    text = _warehouse_text(out)
    assert EEE_PRIVATE_DATASET_REPO not in text
    assert "minibench/model/c" not in text
    public_shape = duckdb.connect().execute(
        f"SELECT DISTINCT protocol_condition, composite_slug "
        f"FROM read_parquet('{out}/fact_results.parquet') "
        f"WHERE collection_id = 'test-study'"
    ).fetchall()
    assert public_shape == [(None, "mini-study")]
    public_links = duckdb.connect().execute(
        f"SELECT eee_record_url, instance_file_url "
        f"FROM read_parquet('{out}/eval_results_view.parquet') "
        f"WHERE collection_id = 'test-study'"
    ).fetchall()
    assert public_links
    assert all(record_url is not None for record_url, _ in public_links)
    assert all(instance_url is None for _, instance_url in public_links)


def test_off_composite_exemption_fails_without_rollout_setting(
    world, monkeypatch, no_private_access
):
    with pytest.raises(RuntimeError, match="curated scoped members failed"):
        _run(world, monkeypatch, include_private=False)


def test_off_pre_ingest_revision_fails_without_rollout_setting(
    world, monkeypatch, no_private_access
):
    for name in ("rec_a.json", "rec_b.json"):
        path = world / "eee" / "data" / "minibench" / "dev" / "model" / name
        path.unlink()
    with pytest.raises(RuntimeError, match="curated collection entries"):
        _run(world, monkeypatch, include_private=False)


def test_off_pre_ingest_revision_allowed_during_rollout(
    world, monkeypatch, no_private_access
):
    for name in ("rec_a.json", "rec_b.json"):
        path = world / "eee" / "data" / "minibench" / "dev" / "model" / name
        path.unlink()
    out = _run(
        world,
        monkeypatch,
        include_private=False,
        allow_zero_members=True,
    )
    assert [r[0] for r in _facts(out)] == ["minibench/ordinary/1"]
    assert not (out / "collection_trajectories.parquet").exists()


def test_active_collections_filter(world, monkeypatch):
    monkeypatch.setenv("COLLECTIONS_CURATED_PATH", str(world / "curated.yaml"))
    on = collections_src.active_collections(True)
    assert set(on.curated) == {"test-study"}
    assert on.exempt_composites == frozenset()
    assert on.optional_curated == frozenset()
    off = collections_src.active_collections(False)
    assert set(off.curated) == {"test-study"}
    assert off.exempt_composites == frozenset()
    assert off.optional_curated == frozenset()
    rollout_off = collections_src.active_collections(
        False, allow_zero_private_members=True
    )
    assert set(rollout_off.curated) == {"test-study"}
    assert rollout_off.exempt_composites == frozenset({"mini-study"})
    assert rollout_off.optional_curated == frozenset({"test-study"})


# ---------------------------------------------------------------------------
# Transitions between toggle states
# ---------------------------------------------------------------------------


def test_on_then_off_same_dir_leaves_no_on_only_file(world, monkeypatch):
    on_out = _run(world, monkeypatch, include_private=True)
    assert (on_out / "collection_trajectories.parquet").exists()

    _refuse_private(monkeypatch)
    off_out = _run(
        world,
        monkeypatch,
        include_private=False,
        allow_zero_members=True,
    )
    assert off_out == on_out
    assert not (off_out / "collection_trajectories.parquet").exists()
    assert [r[0] for r in _facts(off_out)] == [
        "minibench/model/a", "minibench/ordinary/1", "minibench/model/b",
    ]
    text = _warehouse_text(off_out)
    assert "minibench/model/c" not in text


@pytest.mark.parametrize("from_stage", ["B", "C", "D", "I", "J"])
def test_on_cache_refused_by_off_resume(world, monkeypatch, from_stage):
    out = _run(world, monkeypatch, include_private=True)
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    _refuse_private(monkeypatch)
    with pytest.raises(RuntimeError, match="private-source change"):
        _run(world, monkeypatch, include_private=False, from_stage=from_stage)
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir()} == before


def test_off_cache_resume_matches(world, monkeypatch, no_private_access):
    out = _run(
        world,
        monkeypatch,
        include_private=False,
        allow_zero_members=True,
    )
    first = (out / "eval_results_view.parquet").read_bytes()
    out = _run(
        world,
        monkeypatch,
        include_private=False,
        allow_zero_members=True,
        from_stage="D",
    )
    assert (out / "eval_results_view.parquet").read_bytes() == first


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def test_extractor_reads_a_local_source_dir(tmp_path, monkeypatch):
    extractor = _load_extractor_module()

    def _no_network(*_a, **_k):
        raise AssertionError("network used with --source-dir")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_network)
    monkeypatch.setattr(huggingface_hub, "HfApi", _no_network)

    member_rec = _study_record("hle/x/1", "UK AI Security Institute", "hle", 0.5)
    member_rec["source_metadata"]["source_name"] = extractor.STUDY_TITLE
    other_rec = _ordinary_record()
    for rel, rec in (
        ("data/hle/openai/gpt/u1.json", member_rec),
        ("data/hle/openai/gpt/u2.json", other_rec),
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(json.dumps(rec))
    (tmp_path / "data/hle/openai/gpt/u1_samples.jsonl").write_text("")

    stats = Counter()
    members, root = extractor.enumerate_members(
        None, None, None, stats, source_dir=tmp_path
    )
    assert root == tmp_path
    assert [m.path for m in members] == ["data/hle/openai/gpt/u1.json"]

    trajs = extractor.stream_trajectories(
        members, None, None, stats, source_dir=tmp_path
    )
    assert trajs == [] and stats["files_download_failed"] == 0

    (tmp_path / "data/hle/openai/gpt/u1_samples.jsonl").unlink()
    extractor.stream_trajectories(members, None, None, stats, source_dir=tmp_path)
    assert stats["files_download_failed"] == 1


def test_extractor_writes_outside_vendor_by_default():
    extractor = _load_extractor_module()
    assert "vendor" not in extractor.OUT_DIR.parts
    assert extractor.OUT_DIR.parts[-3:] == (
        ".cache", "collections_extract", "aisi_inference_scaling",
    )

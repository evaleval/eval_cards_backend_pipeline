"""`scripts/ci_publish_warehouse.py`: the shrink guard and `--check-only`.

The Hub is replaced by a fake `HfApi` that serves a published
`snapshot_meta.json` (or raises) and records every `upload_folder` call.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError

SNAPSHOT_ID = "2026-10-04T00:00:00Z"


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ci_publish_warehouse.py"
    spec = importlib.util.spec_from_file_location("ci_publish_warehouse", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meta(eee_records=1000, fact_results=5000, n_configs=20, snapshot_id=SNAPSHOT_ID):
    return {
        "snapshot_id": snapshot_id,
        "configs": [f"cfg{i}" for i in range(n_configs)],
        "row_counts": {"eee_records": eee_records, "fact_results": fact_results},
    }


class FakeApi:
    def __init__(self, published, tmp_path):
        self.published = published
        self.tmp_path = tmp_path
        self.uploads: list[dict] = []
        self.downloads: list[dict] = []

    def hf_hub_download(self, **kwargs):
        self.downloads.append(kwargs)
        if isinstance(self.published, Exception):
            raise self.published
        out = self.tmp_path / "published_snapshot_meta.json"
        out.write_text(
            self.published
            if isinstance(self.published, str)
            else json.dumps(self.published)
        )
        return str(out)

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run `main` in a temp cwd holding one local snapshot; return (rc, api)."""
    mod = _load_module()

    def _run(new_meta, published, *, argv=(), allow_shrink=None, token="tok"):
        snap = tmp_path / "warehouse" / SNAPSHOT_ID
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "snapshot_meta.json").write_text(json.dumps(new_meta))
        api = FakeApi(published, tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TARGET_DATASET", "org/target")
        if token is None:
            monkeypatch.delenv("HF_TOKEN", raising=False)
        else:
            monkeypatch.setenv("HF_TOKEN", token)
        if allow_shrink is None:
            monkeypatch.delenv("ALLOW_SNAPSHOT_SHRINK", raising=False)
        else:
            monkeypatch.setenv("ALLOW_SNAPSHOT_SHRINK", allow_shrink)
        monkeypatch.setattr(mod, "HfApi", lambda token=None: api)
        return mod.main(list(argv)), api

    return _run


def test_guard_passes_and_publishes_both_paths(run, capsys):
    rc, api = run(_meta(eee_records=950, fact_results=4600, n_configs=19), _meta())
    assert rc == 0
    assert [u["path_in_repo"] for u in api.uploads] == [
        f"warehouse/{SNAPSHOT_ID}",
        "warehouse/latest",
    ]
    assert api.downloads == [
        {
            "repo_id": "org/target",
            "filename": "warehouse/latest/snapshot_meta.json",
            "repo_type": "dataset",
        }
    ]
    out = capsys.readouterr().out
    assert "eee_records: new=950 published=1000 ok" in out
    assert "fact_results: new=4600 published=5000 ok" in out
    assert "configs: new=19 published=20 ok" in out


def test_guard_passes_at_exactly_ninety_percent(run):
    rc, api = run(_meta(eee_records=900, fact_results=4500, n_configs=18), _meta())
    assert rc == 0
    assert len(api.uploads) == 2


@pytest.mark.parametrize(
    "shrunk, failing_line",
    [
        ({"eee_records": 899}, "eee_records: new=899 published=1000 FAIL"),
        ({"fact_results": 4499}, "fact_results: new=4499 published=5000 FAIL"),
        ({"n_configs": 17}, "configs: new=17 published=20 FAIL"),
    ],
)
def test_guard_fails_on_each_measure_without_uploading(run, capsys, shrunk, failing_line):
    rc, api = run(_meta(**shrunk), _meta())
    assert rc != 0
    assert api.uploads == []
    err = capsys.readouterr().err
    assert failing_line in err
    # The message names all three pairs, not only the failing one.
    for name in ("eee_records: new=", "fact_results: new=", "configs: new="):
        assert name in err


def test_guard_fails_when_new_count_is_missing(run):
    rc, api = run(_meta(eee_records=None), _meta())
    assert rc != 0
    assert api.uploads == []


def test_override_publishes_a_shrunk_snapshot(run, capsys):
    rc, api = run(_meta(eee_records=10), _meta(), allow_shrink="1")
    assert rc == 0
    assert len(api.uploads) == 2
    assert "overridden" in capsys.readouterr().out


def test_override_value_other_than_one_does_not_override(run):
    rc, api = run(_meta(eee_records=10), _meta(), allow_shrink="0")
    assert rc != 0
    assert api.uploads == []


def test_first_publish_fails_without_override(run, capsys):
    rc, api = run(_meta(), EntryNotFoundError("404"))
    assert rc != 0
    assert api.uploads == []
    assert "does not exist" in capsys.readouterr().err


def test_first_publish_passes_with_override(run):
    rc, api = run(_meta(), EntryNotFoundError("404"), allow_shrink="1")
    assert rc == 0
    assert len(api.uploads) == 2


def test_download_error_fails_without_uploading(run, capsys):
    rc, api = run(_meta(), ConnectionError("hub unreachable"))
    assert rc != 0
    assert api.uploads == []
    assert "hub unreachable" in capsys.readouterr().err


@pytest.mark.parametrize("published", ["{not json", "[]", json.dumps({"configs": []})])
def test_unreadable_published_meta_fails_without_uploading(run, published):
    rc, api = run(_meta(), published)
    assert rc != 0
    assert api.uploads == []


def test_check_only_prints_comparison_and_uploads_nothing(run, capsys):
    rc, api = run(_meta(), _meta(), argv=["--check-only"], token=None)
    assert rc == 0
    assert api.uploads == []
    assert len(api.downloads) == 1
    out = capsys.readouterr().out
    assert "eee_records: new=1000 published=1000 ok" in out
    assert "nothing uploaded" in out


def test_check_only_fails_on_a_shrunk_snapshot(run):
    rc, api = run(_meta(fact_results=100), _meta(), argv=["--check-only"])
    assert rc != 0
    assert api.uploads == []


def test_publish_without_token_is_refused_before_any_request(run):
    rc, api = run(_meta(), _meta(), token=None)
    assert rc != 0
    assert api.downloads == []
    assert api.uploads == []

"""Hand-built source directories stay offline under an upstream revision pin.

The sync workflow exports EEE_REVISION, ENTITY_REGISTRY_REVISION and
BENCHMARK_METADATA_REVISION before it runs the tests, and module- and
session-scoped fixtures build `Settings` before conftest's per-test cleanup
runs. A loader handed a pinned revision and a directory with no download
marker has to recognise a hand-built corpus and return it untouched: nothing
deleted, nothing fetched into it. Every hub entry point raises here, so a
loader that reaches for the network fails the test rather than hanging on it.

Each loader runs on a copy of its fixture directory under `tmp_path`: a
loader that regresses to wiping or fetching destroys the copy, never the
committed fixtures.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import huggingface_hub
import pytest

from eval_card_backend.config import Settings
from eval_card_backend.sources import benchmark_cards as cards_src
from eval_card_backend.sources import eee as eee_src
from eval_card_backend.sources import registry as registry_src

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PIN = "f" * 40

LOADERS = {
    "eee": (eee_src, FIXTURES / "eee", "EEE_REVISION", "eee_revision"),
    "registry": (
        registry_src, FIXTURES / "entity_registry",
        "ENTITY_REGISTRY_REVISION", "registry_revision",
    ),
    "cards": (
        cards_src, FIXTURES / "auto_benchmarkcards",
        "BENCHMARK_METADATA_REVISION", "benchmark_metadata_revision",
    ),
}


def _network(*args, **kwargs):
    raise AssertionError("network")


class _NoApi:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _network


@pytest.fixture
def no_network(monkeypatch):
    for name in ("snapshot_download", "hf_hub_download"):
        monkeypatch.setattr(huggingface_hub, name, _network)
    monkeypatch.setattr(huggingface_hub, "HfApi", _NoApi)
    monkeypatch.setattr(eee_src, "hf_hub_download", _network)
    monkeypatch.setattr(eee_src, "HfApi", _NoApi)
    monkeypatch.setattr(registry_src, "snapshot_download", _network)
    monkeypatch.setattr(cards_src, "snapshot_download", _network)


def _tree(root: Path) -> dict[str, int]:
    return {
        p.relative_to(root).as_posix(): p.stat().st_size
        for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_fixture_dir_is_returned_untouched_under_a_revision_pin(
    loader, tmp_path, monkeypatch, no_network,
):
    module, fixture_dir, env_var, field = LOADERS[loader]
    monkeypatch.setenv(env_var, PIN)
    settings = Settings.from_env()
    assert getattr(settings, field) == PIN

    copy = tmp_path / fixture_dir.name
    shutil.copytree(fixture_dir, copy)
    before = _tree(copy)
    assert before == _tree(fixture_dir)
    assert before, f"{fixture_dir} holds no fixture files"

    out = module.ensure_snapshot(
        str(copy), None, False, revision=getattr(settings, field),
    )

    assert out == copy.resolve()
    assert _tree(copy) == before
    assert not (copy / ".pinned_revision").exists()


@pytest.mark.parametrize("loader", ["registry", "cards"])
@pytest.mark.parametrize("pinned", [True, False], ids=["pinned", "unpinned"])
def test_hand_built_dir_under_a_pin_is_logged_at_warning(
    loader, pinned, tmp_path, monkeypatch, no_network, caplog,
):
    """A pin the loader cannot apply is worth a warning naming the directory
    and the pin; without a pin the same path is routine and stays at INFO."""
    module, fixture_dir, _, _ = LOADERS[loader]
    copy = tmp_path / fixture_dir.name
    shutil.copytree(fixture_dir, copy)

    with caplog.at_level(logging.INFO, logger=module.__name__):
        module.ensure_snapshot(str(copy), None, False, revision=PIN if pinned else None)

    hand_built = [r for r in caplog.records if "hand-built" in r.getMessage()]
    assert len(hand_built) == 1
    record = hand_built[0]
    assert str(copy.resolve()) in record.getMessage()
    if pinned:
        assert record.levelno == logging.WARNING
        assert PIN in record.getMessage()
    else:
        assert record.levelno == logging.INFO
        assert PIN not in record.getMessage()


def test_upstream_pin_names_a_hand_built_dir_instead_of_claiming_the_pin(monkeypatch):
    from eval_card_backend.canonicalise import pipeline

    monkeypatch.setattr(pipeline, "_hf_dataset_snapshot", lambda *a, **k: None)
    kwargs = dict(pinned=PIN, cached=None, hf_token=None)
    assert pipeline._upstream_pin("r", hand_built=True, **kwargs) == {
        "repo_id": "r", "sha": PIN, "last_modified": None,
        "revision_source": "hand-built",
    }
    assert pipeline._upstream_pin("r", hand_built=False, **kwargs)["revision_source"] == "pin"
    # A pinned download leaves a record; the pin was applied to it.
    assert pipeline._upstream_pin(
        "r", pinned=PIN, cached=PIN, hf_token=None, hand_built=True,
    )["revision_source"] == "pin"


def test_snapshot_meta_records_hand_built_dirs_under_a_pin(monkeypatch):
    from eval_card_backend.canonicalise import pipeline, stages

    monkeypatch.setattr(pipeline, "_hf_dataset_snapshot", lambda *a, **k: None)
    monkeypatch.setenv("ENTITY_REGISTRY_REVISION", PIN)
    monkeypatch.setenv("BENCHMARK_METADATA_REVISION", "e" * 40)
    meta = pipeline._build_snapshot_meta(
        snapshot_id="2026-05-01T00:00:00Z",
        chosen=[],
        settings=Settings.from_env(),
        n_eee=0, n_cards=0, n_exploded=0, n_synth_collisions=0,
        stage_e_stats=stages.StageEStats(0, 0, 0, 0, 0, 0, {}),
        n_unit_inconsistent=0,
        view_layer_emitted=False,
        eee_root=None,
        registry_root=FIXTURES / "entity_registry",
        cards_root=FIXTURES / "auto_benchmarkcards",
    )
    pins = meta["upstream_pins"]
    assert pins["entity_registry"]["sha"] == PIN
    assert pins["entity_registry"]["revision_source"] == "hand-built"
    assert pins["benchmark_metadata"]["sha"] == "e" * 40
    assert pins["benchmark_metadata"]["revision_source"] == "hand-built"
    assert meta["registry_revision"] == PIN
    assert set(pins["entity_registry"]) == {"repo_id", "sha", "last_modified", "revision_source"}


def test_every_pin_the_suite_strips_has_a_loader_here():
    from tests.conftest import _PINNED_ENV_VARS

    pins = {v for v in _PINNED_ENV_VARS if v.endswith("_REVISION")}
    # The private extract's pin names a commit of a different repo; its
    # loader keeps a hand-built directory as-is and is tested with it.
    pins.discard("EEE_PRIVATE_REVISION")
    assert pins == {env_var for _, _, env_var, _ in LOADERS.values()}


@pytest.mark.parametrize("loader", ["registry", "cards"])
def test_a_downloaded_cache_under_another_pin_is_still_refreshed(
    loader, tmp_path, monkeypatch,
):
    """The guard reads the absence of any download record, not of the marker
    alone: a cache the hub wrote without a pin carries its metadata files and
    must still be replaced when a pin arrives, or a stale cache defeats it."""
    module, fixture_dir, _, _ = LOADERS[loader]
    target = tmp_path / "cache"
    for rel, size in _tree(fixture_dir).items():
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        (target / rel).write_bytes(b"x" * size)
    meta = target / ".cache" / "huggingface" / "download" / "README.md.metadata"
    meta.parent.mkdir(parents=True)
    meta.write_text("e" * 40 + "\n")

    calls: list[dict] = []

    def _download(**kwargs):
        calls.append(kwargs)
        for rel in _tree(fixture_dir):
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            (target / rel).write_bytes(b"y")

    monkeypatch.setattr(module, "snapshot_download", _download)
    out = module.ensure_snapshot(str(target), None, False, revision=PIN)

    assert out == target.resolve()
    assert [c["revision"] for c in calls] == [PIN]
    assert (target / ".pinned_revision").read_text() == PIN

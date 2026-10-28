import json
import subprocess
import tempfile

import pytest

from eval_card_backend.sources import benchmark_cards
from eval_card_backend.sources.benchmark_cards import load_cards


def test_load_cards_both_shapes(tmp_path):
    # Shape 1: per-card files under cards/
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    (cards_dir / "benchmark_card_alpha.json").write_text(
        json.dumps({"benchmark_card": {"benchmark_details": {"name": "Alpha"}}})
    )

    # Shape 2: flat benchmark-metadata.json
    (tmp_path / "benchmark-metadata.json").write_text(
        json.dumps({"Beta Bench": {"benchmark_details": {"name": "Beta Bench"}}})
    )

    cards = load_cards(tmp_path)

    assert set(cards.keys()) == {"alpha", "beta_bench"}
    assert cards["alpha"].payload["benchmark_details"]["name"] == "Alpha"
    assert cards["alpha"].source_path == "cards/benchmark_card_alpha.json"
    assert cards["beta_bench"].payload["benchmark_details"]["name"] == "Beta Bench"
    assert cards["beta_bench"].source_path is None


def test_card_file_payload_wins_over_flat_file(tmp_path):
    (tmp_path / "benchmark-metadata.json").write_text(json.dumps({
        "AA LCR": {"benchmark_details": {"name": "Flat AA LCR"}},
    }))
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    file_card = {"benchmark_details": {"name": "Per-file AA LCR"}}
    (cards_dir / "aa-lcr.json").write_text(json.dumps({"benchmark_card": file_card}))

    card = load_cards(tmp_path)["aa_lcr"]
    assert card.payload == file_card
    assert card.source_path == "cards/aa-lcr.json"


def test_normalized_filename_collision_keeps_first_path_and_payload(tmp_path):
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    for filename in ("aa_lcr.json", "aa-lcr.json"):
        (cards_dir / filename).write_text(json.dumps({
            "benchmark_details": {"name": filename},
        }))

    card = load_cards(tmp_path)["aa_lcr"]
    assert card.payload["benchmark_details"]["name"] == "aa-lcr.json"
    assert card.source_path == "cards/aa-lcr.json"


def test_load_cards_missing_dir(tmp_path):
    assert load_cards(tmp_path / "does-not-exist") == {}


def _card_repo(tmp_path):
    repo = tmp_path / "remote"
    (repo / "cards").mkdir(parents=True)
    (repo / "cards" / "aa-lcr.json").write_text(json.dumps({
        "benchmark_card": {"benchmark_details": {"name": "AA LCR"}},
    }))
    (repo / "benchmark-metadata.json").write_text(json.dumps({
        "aa-lcr": {"benchmark_details": {"name": "AA LCR"}},
    }))
    (repo / "README.md").write_text("not a card")
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()
    run("init", "-q")
    run("add", ".")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "cards")
    return repo, run("rev-parse", "HEAD")


def test_ensure_snapshot_fetches_pinned_commit_in_one_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_cards.hf_constants, "HF_HUB_OFFLINE", False)
    repo, sha = _card_repo(tmp_path)
    monkeypatch.setattr(benchmark_cards, "_repo_url", lambda: str(repo))
    # Staging stays beside the cache: a system temp dir on another filesystem
    # would make the final rename fail.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "no-such-tmp"))

    root = benchmark_cards.ensure_snapshot(str(tmp_path / "cache"), None, False, sha)

    assert sorted(p.name for p in root.iterdir()) == [
        ".fetched_commit", ".pinned_revision", "benchmark-metadata.json", "cards",
    ]
    assert list(tmp_path.glob(".cards-fetch-*")) == []
    assert load_cards(root)["aa_lcr"].source_path == "cards/aa-lcr.json"
    assert benchmark_cards.cached_revision(root) == sha


def test_ensure_snapshot_unpinned_records_fetched_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_cards.hf_constants, "HF_HUB_OFFLINE", False)
    repo, sha = _card_repo(tmp_path)
    monkeypatch.setattr(benchmark_cards, "_repo_url", lambda: str(repo))

    root = benchmark_cards.ensure_snapshot(str(tmp_path / "cache"), None, False)

    assert benchmark_cards.cached_revision(root) == sha


def test_ensure_snapshot_branch_pin_records_the_commit_it_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_cards.hf_constants, "HF_HUB_OFFLINE", False)
    repo, sha = _card_repo(tmp_path)
    monkeypatch.setattr(benchmark_cards, "_repo_url", lambda: str(repo))
    subprocess.run(["git", "-C", str(repo), "branch", "-M", "main"], check=True)

    root = benchmark_cards.ensure_snapshot(str(tmp_path / "cache"), None, False, "main")

    assert (root / ".pinned_revision").read_text() == "main"
    assert benchmark_cards.cached_revision(root) == sha


def _seed_stale_cache(cache):
    (cache / "cards").mkdir(parents=True)
    (cache / "cards" / "old.json").write_text("{}")
    (cache / ".pinned_revision").write_text("a" * 40)


@pytest.mark.parametrize("step", ["fetch", "publish"])
def test_ensure_snapshot_failure_leaves_no_partial_cache(tmp_path, monkeypatch, step):
    monkeypatch.setattr(benchmark_cards.hf_constants, "HF_HUB_OFFLINE", False)
    repo, sha = _card_repo(tmp_path)
    monkeypatch.setattr(benchmark_cards, "_repo_url", lambda: str(repo))
    cache = tmp_path / "cache"
    _seed_stale_cache(cache)

    def _fail(*args, **kwargs):
        raise OSError(step)

    if step == "fetch":
        monkeypatch.setattr(benchmark_cards, "_fetch_cards", _fail)
    else:
        monkeypatch.setattr(benchmark_cards, "_write_fetched_commit", _fail)

    assert benchmark_cards.ensure_snapshot(str(cache), None, False, sha) is None
    assert not (cache / "cards").exists() and not (cache / "benchmark-metadata.json").exists()
    assert list(tmp_path.glob(".cards-fetch-*")) == []

    monkeypatch.undo()
    monkeypatch.setattr(benchmark_cards.hf_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(benchmark_cards, "_repo_url", lambda: str(repo))
    root = benchmark_cards.ensure_snapshot(str(cache), None, False, sha)
    assert benchmark_cards.cached_revision(root) == sha


def test_ensure_snapshot_rejects_lfs_pointer_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_cards.hf_constants, "HF_HUB_OFFLINE", False)
    repo, _ = _card_repo(tmp_path)
    (repo / "cards" / "big.json").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n"
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-qm", "lfs"], check=True,
    )
    monkeypatch.setattr(benchmark_cards, "_repo_url", lambda: str(repo))

    assert benchmark_cards.ensure_snapshot(str(tmp_path / "cache"), None, False) is None


def test_ensure_snapshot_offline_skips_fetch(tmp_path, monkeypatch):
    repo, _ = _card_repo(tmp_path)
    monkeypatch.setattr(benchmark_cards, "_repo_url", lambda: str(repo))
    monkeypatch.setattr(benchmark_cards.hf_constants, "HF_HUB_OFFLINE", True)

    assert benchmark_cards.ensure_snapshot(str(tmp_path / "cache"), None, False) is None

import json

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
    assert cards["alpha"]["benchmark_details"]["name"] == "Alpha"
    assert cards["beta_bench"]["benchmark_details"]["name"] == "Beta Bench"


def test_load_cards_missing_dir(tmp_path):
    assert load_cards(tmp_path / "does-not-exist") == {}

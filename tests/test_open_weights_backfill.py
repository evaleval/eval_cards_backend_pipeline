"""Open-weights backfill: Hub presence as evidence for the registry's NULLs.

The registry leaves `canonical_models.open_weights` unset for most models,
which the frontend cannot tell apart from "closed". A published model has a
Hub repo, so a resolvable id is evidence of open weights — but a miss is
NOT evidence of a closed model (renamed, deleted, private, or never spelled
the same upstream), so it must stay NULL.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import duckdb
import pytest

from eval_card_backend.canonicalise import stages
from eval_card_backend.sources.hf_openness import (
    OpennessProbe,
    normalise_repo_name,
)


# --- Hub doubles -----------------------------------------------------------


class _NotFound(Exception):
    """Stands in for huggingface_hub's RepositoryNotFoundError."""
    def __init__(self):
        super().__init__("not found")
        self.response = SimpleNamespace(status_code=404)


class _Gated(Exception):
    """Stands in for huggingface_hub's GatedRepoError (403)."""
    def __init__(self):
        super().__init__("gated")
        self.response = SimpleNamespace(status_code=403)


class _Flaky(Exception):
    """A transient failure — no verdict either way."""
    def __init__(self):
        super().__init__("503 upstream")
        self.response = SimpleNamespace(status_code=503)


class FakeHub:
    """`present` resolve, `gated` raise GatedRepoError, `flaky` raise 503,
    everything else 404s. `org_models` backs the rename pass."""

    def __init__(self, present=(), *, gated=(), flaky=(), org_models=None):
        self.present = set(present)
        self.gated = set(gated)
        self.flaky = set(flaky)
        self.org_models = org_models or {}
        self.lookups = []
        self.listings = []

    def model_info(self, repo_id):
        self.lookups.append(repo_id)
        if repo_id in self.flaky:
            raise _Flaky()
        if repo_id in self.gated:
            raise _Gated()
        if repo_id in self.present:
            return SimpleNamespace(id=repo_id)
        raise _NotFound()

    def list_models(self, author=None):
        self.listings.append(author)
        return [SimpleNamespace(id=m) for m in self.org_models.get(author, [])]


def _probe(hub, **kw):
    return OpennessProbe(hub, **kw)


# --- normalisation ---------------------------------------------------------


def test_normalisation_is_order_and_punctuation_insensitive():
    assert normalise_repo_name("Falcon-Instruct-40B") == normalise_repo_name(
        "falcon-40b-instruct"
    )
    assert normalise_repo_name("Yi_1.5-9B") == normalise_repo_name("yi-1-5-9b")


def test_normalisation_still_separates_genuinely_different_names():
    assert normalise_repo_name("llama-3-8b") != normalise_repo_name("llama-3-70b")


# --- probe verdicts --------------------------------------------------------


def test_present_repo_confirms_open():
    hub = FakeHub(present={"org/model-a"})
    assert _probe(hub).confirm(["org/model-a"]) == {"org/model-a": True}


def test_gated_repo_is_open():
    """A gate is a terms click, not a closed model — the weights are
    published, so the model is open."""
    hub = FakeHub(gated={"meta-llama/Llama-3-8B"})
    assert _probe(hub).confirm(["meta-llama/Llama-3-8B"]) == {
        "meta-llama/Llama-3-8B": True
    }


def test_missing_repo_is_not_reported_closed():
    """The mapping never carries False — a miss is unknown, not closed."""
    hub = FakeHub()
    assert _probe(hub).confirm(["org/gone"]) == {}


def test_transient_failure_yields_no_verdict_and_is_not_cached(tmp_path):
    cache = tmp_path / "openness.json"
    hub = FakeHub(flaky={"org/wobbly"})
    assert _probe(hub, cache_path=cache).confirm(["org/wobbly"]) == {}

    # A later run, with the Hub healthy, must still be able to confirm it.
    hub2 = FakeHub(present={"org/wobbly"})
    assert _probe(hub2, cache_path=cache).confirm(["org/wobbly"]) == {
        "org/wobbly": True
    }
    assert hub2.lookups == ["org/wobbly"], "transient miss was wrongly cached"


def test_unqualified_ids_are_never_looked_up():
    """A bare name cannot address a Hub repo — don't spend a request."""
    hub = FakeHub()
    assert _probe(hub).confirm(["bare-name", "", None]) == {}
    assert hub.lookups == []


# --- rename recovery -------------------------------------------------------


def test_rename_is_recovered_by_token_reorder():
    hub = FakeHub(
        present={"tiiuae/falcon-40b-instruct"},
        org_models={"tiiuae": ["tiiuae/falcon-40b-instruct"]},
    )
    assert _probe(hub).confirm(["tiiuae/Falcon-Instruct-40B"]) == {
        "tiiuae/Falcon-Instruct-40B": True
    }


def test_ambiguous_normalised_names_do_not_recover():
    """`wmt_de_en` and `wmt_en_de` normalise alike and are different models;
    a colliding key must not resolve to either."""
    hub = FakeHub(
        org_models={"google": ["google/bert_wmt_de_en", "google/bert_wmt_en_de"]},
    )
    assert _probe(hub).confirm(["google/bert-wmt-en-de"]) == {}


def test_rename_pass_lists_each_org_once():
    hub = FakeHub(org_models={"acme": ["acme/real-model"]})
    _probe(hub).confirm(["acme/missing-one", "acme/missing-two"])
    assert hub.listings == ["acme"], f"listed more than once: {hub.listings}"


# --- caching ---------------------------------------------------------------


def test_verdicts_are_cached_across_runs(tmp_path):
    cache = tmp_path / "openness.json"
    hub = FakeHub(present={"org/model-a"})
    assert _probe(hub, cache_path=cache).confirm(["org/model-a"])

    hub2 = FakeHub(present={"org/model-a"})
    assert _probe(hub2, cache_path=cache).confirm(["org/model-a"]) == {
        "org/model-a": True
    }
    assert hub2.lookups == [], "cached verdict was re-probed"


def test_stale_cache_version_is_discarded(tmp_path):
    cache = tmp_path / "openness.json"
    cache.write_text(json.dumps({"version": 0, "verdicts": {"org/x": True}}))
    hub = FakeHub()
    assert _probe(hub, cache_path=cache).confirm(["org/x"]) == {}
    assert hub.lookups == ["org/x"], "stale-version cache was trusted"


def test_unreadable_cache_does_not_break_the_probe(tmp_path):
    cache = tmp_path / "openness.json"
    cache.write_text("{not json")
    hub = FakeHub(present={"org/model-a"})
    assert _probe(hub, cache_path=cache).confirm(["org/model-a"]) == {
        "org/model-a": True
    }


# --- the stage ------------------------------------------------------------


def _con_with_models(rows):
    """rows: (id, open_weights)"""
    con = duckdb.connect()
    con.execute("CREATE TABLE canonical_models (id VARCHAR, open_weights BOOLEAN)")
    con.executemany("INSERT INTO canonical_models VALUES (?, ?)", rows)
    return con


def test_stage_fills_only_nulls_and_never_overwrites_curation():
    con = _con_with_models([
        ("org/unset-open", None),
        ("org/curated-closed", False),
        ("org/curated-open", True),
    ])
    # The Hub would say "present" for all three; only the NULL may change.
    hub = FakeHub(present={"org/unset-open", "org/curated-closed", "org/curated-open"})

    filled = stages.stage_a_backfill_open_weights(con, probe=_probe(hub))
    assert filled == 1

    got = dict(con.execute("SELECT id, open_weights FROM canonical_models").fetchall())
    assert got == {
        "org/unset-open": True,
        "org/curated-closed": False,
        "org/curated-open": True,
    }
    assert "org/curated-closed" not in hub.lookups, "curated row was probed"


def test_stage_leaves_unconfirmed_rows_null():
    con = _con_with_models([("org/present", None), ("org/absent", None)])
    hub = FakeHub(present={"org/present"})

    stages.stage_a_backfill_open_weights(con, probe=_probe(hub))

    got = dict(con.execute("SELECT id, open_weights FROM canonical_models").fetchall())
    assert got == {"org/present": True, "org/absent": None}


def test_stage_survives_a_probe_that_raises():
    """An unreachable Hub must not fail the bake."""
    class Exploding:
        def confirm(self, _ids):
            raise RuntimeError("hub down")

    con = _con_with_models([("org/unset", None)])
    assert stages.stage_a_backfill_open_weights(con, probe=Exploding()) == 0
    assert con.execute(
        "SELECT open_weights FROM canonical_models"
    ).fetchone()[0] is None


def test_stage_is_a_noop_without_canonical_models():
    assert stages.stage_a_backfill_open_weights(duckdb.connect(), probe=None) == 0


def test_stage_skips_unqualified_ids():
    con = _con_with_models([("bare", None), (None, None)])
    hub = FakeHub()
    assert stages.stage_a_backfill_open_weights(con, probe=_probe(hub)) == 0
    assert hub.lookups == []

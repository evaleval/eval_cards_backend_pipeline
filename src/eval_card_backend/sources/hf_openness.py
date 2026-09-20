"""Confirm open weights by looking the model up on the Hugging Face Hub.

The entity registry carries `canonical_models.open_weights`, but leaves it
NULL for most models it has not curated — 4,883 of 8,815 in the
2026-09-20 snapshot. A model whose weights are published has a model repo
on the Hub; a proprietary API model does not. So a resolvable repo id is
positive evidence of open weights, and this module turns that into a
backfill for rows the registry left unset.

Only ever returns `True`. A lookup that misses is NOT evidence of a closed
model: the id may name a repo that was renamed, made private or deleted,
or one whose spelling upstream never matched the Hub. Those are
indistinguishable from a genuinely proprietary model at this layer, so
they stay NULL — "unknown", which is what the registry already meant —
rather than being asserted closed. Measured against the rows the registry
*has* curated, presence tracks the curated label closely (97% of
`open_weights = true` models resolve; 98% of `open_weights = false` do
not), which is what makes the positive direction trustworthy and the
negative direction not.

Renames are recovered by a second pass: a miss is re-checked against the
org's own model list, matching on the case-folded, punctuation-split token
multiset of the repo name, so `tiiuae/Falcon-Instruct-40B` finds
`tiiuae/falcon-40b-instruct`. Distinct models in one org collide under
that normalisation about 1% of the time (`wmt_de_en` vs `wmt_en_de`), so
the match is used only as evidence that the org publishes this model's
weights — never recorded as the model's canonical id.

Network failures are not cached and never fail the bake; they leave the
row NULL exactly as if the probe had not run.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["OpennessProbe", "confirm_open_weights", "normalise_repo_name"]

# Bump when the cached verdicts' meaning changes, to invalidate old files.
_CACHE_VERSION = 1

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def normalise_repo_name(name: str) -> tuple[str, ...]:
    """Case-folded, punctuation-split, order-independent token multiset.

    `Falcon-Instruct-40B` and `falcon-40b-instruct` normalise alike; this
    is what lets the rename pass match a reordered or re-punctuated name.
    """
    return tuple(sorted(t for t in _TOKEN_SPLIT.split(name.lower()) if t))


class OpennessProbe:
    """Looks model ids up on the Hub, with an on-disk verdict cache.

    `api` is injected so tests can drive it without network. It needs
    `model_info(repo_id)` (raising on absence) and
    `list_models(author=...)`.
    """

    def __init__(self, api, *, cache_path: Path | None = None, max_workers: int = 8):
        self._api = api
        self._cache_path = cache_path
        self._max_workers = max(1, max_workers)
        self._cache: dict[str, bool] = self._load_cache()
        # Fetched at most once per org, and only when that org has a miss.
        self._org_index: dict[str, dict[tuple[str, ...], str]] = {}

    # -- cache ---------------------------------------------------------

    def _load_cache(self) -> dict[str, bool]:
        if self._cache_path is None or not self._cache_path.exists():
            return {}
        try:
            blob = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.debug("openness cache unreadable; starting empty", exc_info=True)
            return {}
        if blob.get("version") != _CACHE_VERSION:
            return {}
        verdicts = blob.get("verdicts")
        return verdicts if isinstance(verdicts, dict) else {}

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps({"version": _CACHE_VERSION, "verdicts": self._cache}),
                encoding="utf-8",
            )
        except OSError:
            log.debug("could not write openness cache", exc_info=True)

    # -- probing -------------------------------------------------------

    def _exists(self, model_id: str) -> bool | None:
        """True/False if the Hub answered, None if the lookup itself failed.

        A `None` is a transient/unknown condition and is never cached —
        otherwise one flaky run would pin a model to NULL indefinitely.
        """
        try:
            self._api.model_info(model_id)
            return True
        except Exception as exc:  # noqa: BLE001 - hub errors are not a fixed type
            # A gated repo exists and its weights are published — the gate is
            # a terms click, not a closed model. Treat it as present.
            if _is_gated(exc):
                return True
            if _is_absent(exc):
                return False
            log.debug("openness lookup failed for %s: %s", model_id, exc)
            return None

    def _org_names(self, org: str) -> dict[tuple[str, ...], str]:
        """Normalised-name -> repo id for one org. Ambiguous names are
        dropped: a colliding key cannot identify a single repo."""
        if org in self._org_index:
            return self._org_index[org]
        index: dict[tuple[str, ...], str] = {}
        collisions: set[tuple[str, ...]] = set()
        try:
            for info in self._api.list_models(author=org):
                repo_id = getattr(info, "id", None) or ""
                if "/" not in repo_id:
                    continue
                key = normalise_repo_name(repo_id.split("/", 1)[1])
                if key in index and index[key] != repo_id:
                    collisions.add(key)
                index[key] = repo_id
        except Exception as exc:  # noqa: BLE001
            log.debug("could not list models for org %s: %s", org, exc)
            index = {}
        for key in collisions:
            index.pop(key, None)
        self._org_index[org] = index
        return index

    def _recover(self, model_id: str) -> bool:
        """Second pass for a miss: does this org publish the same model
        under a reordered or re-punctuated name?"""
        org, _, name = model_id.partition("/")
        if not org or not name:
            return False
        match = self._org_names(org).get(normalise_repo_name(name))
        if match is None:
            return False
        log.debug("openness: %s recovered via %s", model_id, match)
        return True

    def confirm(self, model_ids) -> dict[str, bool]:
        """Return `{model_id: True}` for every id confirmed on the Hub.

        Ids absent from the result are left for the caller to treat as
        unknown; the mapping never carries `False`.
        """
        # Only org-qualified ids can name a Hub repo. Anything else (a bare
        # name, a NULL) is unaddressable, so don't spend a request on it.
        wanted = sorted({m for m in model_ids if m and "/" in m})
        if not wanted:
            return {}

        cached_open = {m: True for m in wanted if self._cache.get(m) is True}
        todo = [m for m in wanted if m not in self._cache]

        found: dict[str, bool] = {}
        if todo:
            with ThreadPoolExecutor(self._max_workers) as pool:
                results = list(pool.map(self._exists, todo))

            misses = []
            for model_id, present in zip(todo, results):
                if present is True:
                    self._cache[model_id] = True
                    found[model_id] = True
                elif present is False:
                    misses.append(model_id)
                # present is None -> transient, leave uncached and unset

            # Rename pass, serialised per org by the shared index cache.
            for model_id in misses:
                if self._recover(model_id):
                    self._cache[model_id] = True
                    found[model_id] = True
                else:
                    self._cache[model_id] = False

            self._save_cache()

        confirmed = {**cached_open, **found}
        log.info(
            "openness probe: %d/%d model ids confirmed on the Hub "
            "(%d served from cache, %d looked up)",
            len(confirmed), len(wanted), len(cached_open), len(todo),
        )
        return confirmed


def _status_of(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _is_gated(exc: Exception) -> bool:
    """True when the repo exists but is behind a terms gate."""
    return type(exc).__name__ == "GatedRepoError" or _status_of(exc) == 403


def _is_absent(exc: Exception) -> bool:
    """True when the Hub positively said the repo is not there.

    Anything else — rate limit, timeout, DNS, 5xx — is transient and must
    not be read as a verdict. A private repo is indistinguishable from a
    deleted one here (both 404), and neither is published weights.
    """
    if _status_of(exc) == 404:
        return True
    return type(exc).__name__ in {
        "RepositoryNotFoundError",
        "EntryNotFoundError",
    }


def confirm_open_weights(
    model_ids,
    *,
    hf_token: str | None = None,
    cache_path: Path | None = None,
    max_workers: int = 8,
) -> dict[str, bool]:
    """Convenience wrapper building the default Hub-backed probe."""
    from huggingface_hub import HfApi

    probe = OpennessProbe(
        HfApi(token=hf_token), cache_path=cache_path, max_workers=max_workers
    )
    return probe.confirm(model_ids)

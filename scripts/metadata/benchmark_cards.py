import json
import re
from pathlib import Path
from typing import Any

from scripts.helpers.benchmark_identity import (
    canonical_benchmark_family_key,
    normalize_benchmark_key,
)
from scripts.helpers.slug_utils import as_string


def compact_benchmark_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", as_string(value).lower())


def candidate_benchmark_keys(*values: Any) -> list[str]:
    keys = set()
    for value in values:
        text = as_string(value)
        if not text:
            continue
        normalized = normalize_benchmark_key(text)
        stripped = normalize_benchmark_key(
            re.sub(r"^benchmark_card_", "", text, flags=re.IGNORECASE)
        )
        separated = normalize_benchmark_key(re.sub(r"[_-]+", " ", text))
        compact = compact_benchmark_key(text)
        keys.add(normalized)
        keys.add(stripped)
        keys.add(separated)
        keys.add(compact)
        family_key = canonical_benchmark_family_key(text)
        if family_key:
            keys.add(family_key)
    return [k for k in keys if k]


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [as_string(item) for item in value if as_string(item)]
    text = as_string(value)
    return [text] if text else []


def load_benchmark_metadata_from_dir(
    root_dir: Path,
) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    cards = []
    lookup: dict[str, dict] = {}
    flat_map: dict[str, dict] = {}

    if not root_dir.exists():
        return cards, lookup, flat_map

    flat_metadata_path = root_dir / "benchmark-metadata.json"
    if flat_metadata_path.exists():
        parsed = json.loads(flat_metadata_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            for raw_key, raw_card in parsed.items():
                if not isinstance(raw_card, dict) or not isinstance(
                    raw_card.get("benchmark_details"), dict
                ):
                    continue
                card_id = normalize_benchmark_key(raw_key)
                if card_id:
                    flat_map[card_id] = raw_card

    cards_dir = root_dir / "cards"
    if cards_dir.exists():
        for file_path in sorted(cards_dir.glob("*.json")):
            parsed = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict) and isinstance(
                parsed.get("benchmark_card"), dict
            ):
                card = parsed["benchmark_card"]
                base_name = file_path.stem.replace("benchmark_card_", "")
            elif isinstance(parsed, dict) and isinstance(
                parsed.get("benchmark_details"), dict
            ):
                card = parsed
                base_name = file_path.stem
            else:
                continue
            card_id = normalize_benchmark_key(base_name)
            if card_id and card_id not in flat_map:
                flat_map[card_id] = card

    for card_id, card in sorted(flat_map.items()):
        keys = candidate_benchmark_keys(card_id, card.get("benchmark_details", {}).get("name"))
        cards.append(
            {
                "file_name": f"{card_id}.json",
                "base_name": card_id,
                "card": card,
                "keys": keys,
            }
        )
        for key in keys:
            lookup[key] = card

    return cards, lookup, flat_map


def extract_benchmark_tags(benchmark_card: dict | None) -> dict:
    """Extract structured tags from a benchmark card for frontend filtering."""
    if not benchmark_card:
        return {"domains": [], "languages": [], "tasks": []}
    details = benchmark_card.get("benchmark_details") or {}
    purpose = benchmark_card.get("purpose_and_intended_users") or {}
    return {
        "domains": as_string_list(details.get("domains")),
        "languages": as_string_list(details.get("languages")),
        "tasks": as_string_list(purpose.get("tasks")),
    }


def lookup_benchmark_card(
    metadata_lookup: dict[str, dict], *values: Any
) -> dict | None:
    for key in candidate_benchmark_keys(*values):
        if key in metadata_lookup:
            return metadata_lookup[key]
    return None


def iter_matching_benchmark_cards(
    metadata_lookup: dict[str, dict], *values: Any
) -> list[dict]:
    matches: list[dict] = []
    seen: set[str] = set()
    for key in candidate_benchmark_keys(*values):
        card = metadata_lookup.get(key)
        if not card:
            continue
        details = card.get("benchmark_details") if isinstance(card, dict) else {}
        card_id = normalize_benchmark_key((details or {}).get("name")) or key
        if card_id in seen:
            continue
        seen.add(card_id)
        matches.append(card)
    return matches


def lookup_benchmark_card_for_parent(
    metadata_lookup: dict[str, dict],
    *values: Any,
    aux_values: tuple[Any, ...] = (),
    parent_values: tuple[Any, ...] = (),
) -> dict | None:
    """Resolve the benchmark card that best describes the requested benchmark.

    `*values` are the **primary** identifiers of the benchmark we're looking
    up (its leaf name and key). `aux_values` are auxiliary identifiers used
    to cast a wider net when matching (e.g. the parent suite name, family
    key, dataset_name on a per-record source) — they help us *find*
    candidates but on their own they don't make a candidate the right answer.
    `parent_values` carry the same role as before: they describe the
    enclosing parent and are used to pick the most compatible child variant
    when a card has multiple `appears_in` entries.

    A returned card is the one whose own name matches a primary identifier
    ("self"). If none qualifies, we fall back to a child card whose
    `appears_in` matches one of our search keys (this is what allows
    leaf-benchmark lookups to find variant-specific cards). We deliberately
    do **not** fall back to a card that matched only via `aux_values`
    without also being a recorded child of the search — returning such a
    card would mean handing the requester their *parent's* card (e.g. the
    helm_classic suite card when they asked for XSUM).
    """

    # Phase 1: direct lookup by the primary identifiers. metadata_lookup is
    # keyed by normalized card identifiers (file basename / card name), so a
    # match here means the card *is* the thing we asked for. This is the most
    # reliable signal — when it succeeds, return it immediately without
    # consulting aux_values, which would otherwise drag in parent/family cards.
    direct = lookup_benchmark_card(metadata_lookup, *values)
    if direct is not None:
        return direct

    # Phase 2: cast a wider net via aux_values. The candidates picked up here
    # were matched via family or dataset identifiers, so they're typically
    # variant siblings/children. We accept them only when their `appears_in`
    # explicitly lists one of our search keys — otherwise it's an
    # ancestor of the leaf and returning it would mislead.
    all_search_values = (*values, *aux_values)
    candidates = iter_matching_benchmark_cards(metadata_lookup, *all_search_values)
    if not candidates:
        return None

    search_keys = set(candidate_benchmark_keys(*all_search_values))
    parent_keys = set(candidate_benchmark_keys(*parent_values))

    child_cards: list[tuple[dict, set[str]]] = []
    for card in candidates:
        details = card.get("benchmark_details") if isinstance(card, dict) else {}
        appears_in = as_string_list((details or {}).get("appears_in"))
        if not appears_in:
            continue  # skip standalone cards picked up only via aux/parent keys
        appears_in_keys = set(candidate_benchmark_keys(*appears_in))
        if appears_in_keys & search_keys:
            child_cards.append((card, appears_in_keys))

    if not parent_keys:
        return child_cards[0][0] if child_cards else None

    compatible = [c for c, appears_in_keys in child_cards if parent_keys & appears_in_keys]
    if compatible:
        return compatible[0]
    if child_cards:
        return child_cards[0][0]
    return None

def benchmark_card_language_keys(benchmark_card: dict | None) -> set[str]:
    if not benchmark_card:
        return set()

    details = (
        benchmark_card.get("benchmark_details")
        if isinstance(benchmark_card, dict)
        else {}
    )
    languages = as_string_list((details or {}).get("languages"))
    keys = set()
    for language in languages:
        keys.update(candidate_benchmark_keys(language))
        keys.add(compact_benchmark_key(language))
    return {key for key in keys if key}
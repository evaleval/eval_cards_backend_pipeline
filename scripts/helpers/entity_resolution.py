import re
from typing import Any

from scripts import registry
from scripts.helpers.slug_utils import (
    as_string,
    humanize_slug,
    slugify_developer,
    slugify_model_segment,
)

VERSION_SUFFIX_REGEX = re.compile(
    r"^(.*?)-((?:19|20)\d{6}|(?:19|20)\d{2}-\d{2}-\d{2})(?:-(.+))?$"
)


def normalize_setup_alias_qualifier(value: Any) -> str:
    return as_string(value).strip().lower().replace("_", "-").replace(" ", "-")


def is_setup_alias_qualifier(value: Any) -> bool:
    normalized = normalize_setup_alias_qualifier(value)
    return bool(
        normalized
        in {
            "prompt",
            "fc",
            "function-calling",
            "prompt-thinking",
            "fc-thinking",
            "function-calling-thinking",
            "thinking",
        }
        or normalized.startswith("thinking-")
        or normalized.startswith("prompt-thinking-")
        or normalized.startswith("fc-thinking-")
        or normalized.startswith("function-calling-thinking-")
    )


def strip_setup_alias_suffix(value: Any) -> tuple[str, str] | None:
    normalized = slugify_model_segment(value)
    patterns = [
        re.compile(
            r"^(.*?)-((?:prompt|fc|function-calling)-thinking(?:-[a-z0-9.]+)*)$"
        ),
        re.compile(r"^(.*?)-(thinking(?:-[a-z0-9.]+)*)$"),
        re.compile(r"^(.*?)-((?:prompt|fc|function-calling))$"),
    ]

    for pattern in patterns:
        match = pattern.match(normalized)
        if match and match.group(1):
            return match.group(1), match.group(2)

    return None


def normalize_model_info(model_info: dict) -> dict:
    raw_id = as_string(
        model_info.get("id") or model_info.get("name") or "unknown/unknown"
    )
    fallback_developer = as_string(
        model_info.get("developer") or raw_id.split("/")[0] or "unknown"
    )
    if "/" in raw_id:
        parts = raw_id.split("/")
    else:
        parts = [slugify_developer(fallback_developer), raw_id]
    raw_developer = (
        parts[0] if len(parts) > 1 else slugify_developer(fallback_developer)
    )
    raw_model_name = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
    match = VERSION_SUFFIX_REGEX.match(raw_model_name)
    base_slug = match.group(1) if match else raw_model_name
    version_date = match.group(2) if match else None
    qualifier = match.group(3) if match else None

    developer_display = as_string(
        model_info.get("developer") or humanize_slug(raw_developer)
    )

    canonical_org_id = None
    canonical_org_strategy = None
    canonical_org_confidence = None
    for candidate in (developer_display, raw_developer):
        if not candidate:
            continue
        org_result = registry.resolve_org(candidate)
        if org_result and org_result.get("canonical_id"):
            canonical_org_id = org_result["canonical_id"]
            canonical_org_strategy = org_result.get("strategy")
            canonical_org_confidence = org_result.get("confidence")
            break

    return {
        "raw_id": raw_id,
        "developer": developer_display,
        "developer_slug": slugify_developer(raw_developer),
        "canonical_org_id": canonical_org_id,
        "canonical_org_resolution_strategy": canonical_org_strategy,
        "canonical_org_resolution_confidence": canonical_org_confidence,
        "model_name": as_string(model_info.get("name") or humanize_slug(base_slug)),
        "raw_model_name": raw_model_name,
        "family_slug": slugify_model_segment(base_slug),
        "version_date": version_date,
        "qualifier": qualifier,
    }


def canonical_model_identity(model_info: dict) -> dict:
    normalized = normalize_model_info(model_info)
    slug_family_id = f"{normalized['developer_slug']}/{normalized['family_slug']}"
    normalized_id = (
        f"{normalized['developer_slug']}/{normalized['raw_model_name']}"
        if "/" in normalized["raw_id"]
        else f"{normalized['developer_slug']}/{normalized['raw_id']}"
    )
    variant_parts = [
        p for p in [normalized["version_date"], normalized["qualifier"]] if p
    ]
    variant_key = "-".join(variant_parts) if variant_parts else "default"
    variant_label = " ".join(variant_parts) if variant_parts else "Default"

    canonical_family_id = slug_family_id
    canonical_model_id = None
    canonical_strategy = None
    canonical_confidence = None
    for candidate in (normalized["raw_id"], normalized_id, slug_family_id):
        if not candidate:
            continue
        result = registry.resolve_model(candidate)
        if result and result.get("canonical_id"):
            canonical_model_id = result["canonical_id"]
            canonical_family_id = canonical_model_id
            canonical_strategy = result.get("strategy")
            canonical_confidence = result.get("confidence")
            break

    return {
        "normalized_id": normalized_id,
        "family_id": canonical_family_id,
        "family_slug": canonical_family_id.split("/", 1)[-1]
        if "/" in canonical_family_id
        else normalized["family_slug"],
        "family_name": normalized["model_name"],
        "model_route_id": canonical_family_id.replace("/", "__"),
        "variant_key": variant_key,
        "variant_label": variant_label,
        "canonical_model_id": canonical_model_id,
        "canonical_resolution_strategy": canonical_strategy,
        "canonical_resolution_confidence": canonical_confidence,
    }


def is_setup_alias_mode(model_info: dict) -> bool:
    additional_details = model_info.get("additional_details") or {}
    raw_mode = as_string(additional_details.get("mode")).strip()
    if not raw_mode:
        return False
    return is_setup_alias_qualifier(raw_mode)


def aggregated_display_identity(model_info: dict) -> dict:
    identity = canonical_model_identity(model_info)
    if not is_setup_alias_mode(model_info):
        return {**identity, "merged_setup_alias": False}

    normalized = normalize_model_info(model_info)
    family_slug = normalized["family_slug"]
    stripped_alias = strip_setup_alias_suffix(family_slug)
    if stripped_alias:
        family_slug = stripped_alias[0]

    family_id = f"{normalized['developer_slug']}/{family_slug}"
    family_name = humanize_slug(family_slug)
    if normalized["version_date"]:
        variant_key = normalized["version_date"]
        variant_label = normalized["version_date"]
    else:
        variant_key = "default"
        variant_label = "Default"

    return {
        **identity,
        "family_id": family_id,
        "family_slug": family_slug,
        "family_name": family_name,
        "model_route_id": family_id.replace("/", "__"),
        "variant_key": variant_key,
        "variant_label": variant_label,
        "merged_setup_alias": True,
    }


def resolve_model_identity_for_pipeline(model_info: dict) -> dict[str, Any]:
    """Canonical registry identity plus display/route identity for stamping ``model_info``.

    Stable routing uses ``display_route`` (post setup-alias merge). Registry audit
    fields live under ``canonical``.
    """

    canonical = canonical_model_identity(model_info)
    display_route = aggregated_display_identity(model_info)
    return {
        "canonical": {
            "normalized_id": canonical["normalized_id"],
            "family_id": canonical["family_id"],
            "model_route_id": canonical["model_route_id"],
            "canonical_model_id": canonical.get("canonical_model_id"),
            "canonical_resolution_strategy": canonical.get(
                "canonical_resolution_strategy"
            ),
            "canonical_resolution_confidence": canonical.get(
                "canonical_resolution_confidence"
            ),
            "variant_key": canonical["variant_key"],
            "variant_label": canonical["variant_label"],
        },
        "display_route": {
            "family_id": display_route["family_id"],
            "family_slug": display_route["family_slug"],
            "family_name": display_route["family_name"],
            "variant_key": display_route["variant_key"],
            "variant_label": display_route["variant_label"],
            "model_route_id": display_route["model_route_id"],
            "merged_setup_alias": display_route.get("merged_setup_alias", False),
        },
    }

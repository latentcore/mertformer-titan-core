from __future__ import annotations

from typing import Any, Dict, List

CANONICAL_CHESS_PROFILE = "strength_4060_24h"
PORTABLE_BASELINE_PROFILE = "production_5080"

PROFILE_SUPPORT_LEVELS = {
    CANONICAL_CHESS_PROFILE: "baseline_supported",
    PORTABLE_BASELINE_PROFILE: "supported_portable_baseline",
    "strength_4060_24h_all_on_experimental": "experimental",
    "strength_4060_24h_omni_max": "experimental_high_risk",
}

NON_RELEASE_BUNDLES = {"all_on_experimental"}


def profile_support_rows(active_profile: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, support_level in PROFILE_SUPPORT_LEVELS.items():
        rows.append(
            {
                "label": label,
                "support_level": support_level,
                "active": label == active_profile,
                "release_candidate_eligible": label == CANONICAL_CHESS_PROFILE,
            }
        )
    return rows


def is_release_candidate_profile(profile: str) -> bool:
    return profile == CANONICAL_CHESS_PROFILE


def is_release_candidate_configuration(profile: str, feature_bundle: str | None) -> bool:
    bundle = (feature_bundle or "").strip()
    return profile == CANONICAL_CHESS_PROFILE and bundle not in NON_RELEASE_BUNDLES


def release_candidate_reason(profile: str) -> str:
    if profile == CANONICAL_CHESS_PROFILE:
        return "Canonical 24h RTX 4060 profile selected."
    if profile == PORTABLE_BASELINE_PROFILE:
        return "Portable baseline is supported, but the frozen chess release-candidate path is the 24h RTX 4060 profile."
    if profile == "strength_4060_24h_all_on_experimental":
        return "All-on experimental profile is research-only and cannot satisfy the frozen release gate."
    if profile == "strength_4060_24h_omni_max":
        return "Omni-max profile is high-risk research-only and cannot satisfy the frozen release gate."
    return "Unknown profile is not eligible for the frozen release gate."


def release_candidate_configuration_reason(profile: str, feature_bundle: str | None) -> str:
    bundle = (feature_bundle or "").strip()
    if profile != CANONICAL_CHESS_PROFILE:
        return release_candidate_reason(profile)
    if bundle in NON_RELEASE_BUNDLES:
        return "Experimental feature bundles cannot satisfy the frozen release gate, even on the canonical 24h RTX 4060 profile."
    return "Canonical 24h RTX 4060 profile selected with a release-eligible bundle configuration."

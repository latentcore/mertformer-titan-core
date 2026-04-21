# MertFormer 5080 Final Onefile Truth

This document tracks the repo-side truth boundary for the promoted final 5080 onefile lane.

## Current Position

- Canonical repo script: `scripts/mertformer_5080_final_onefile.py`
- Delivery helper: `scripts/build_mertformer_5080_final_delivery.py`
- Result decrypt helper: `scripts/decrypt_mertformer_result_package.py`
- Default operator profile: `safe_5080`
- Optional aggressive profile: `challenge_5080`

## Active Model Path

- The active runtime model class is `RepoParityMertFormerModel`.
- Compatibility code from the older onecell scaffold is preserved as `LegacyOnecellMertFormerTiny`.
- Compatibility code is kept for fallback/reference only; it is not the default active training path.
- The active architecture path is the embedded repo-backed stack built from `bitlinear`, `mla`, `moe`, `liquid`, `mertformer_block`, and `model.transformers`.

## Experimental Policy

Experimental/cognitive components remain in the codebase, but they are treated with explicit honesty:

- preserved, not silently stripped
- documented as experimental/feature-flag driven where appropriate
- not used as evidence for frontier-quality or Gemma-beating claims without measured benchmark proof

This applies in particular to layers such as:

- `GlobalWorkspaceBroadcast`
- `HebbianPlasticityLayer`
- `NeuroSymbolicLayer`
- `ContinuousLatentODEStateChannel`
- `NeuromodulatoryGainLayer`
- `LifelongSafetyLayer`
- `world_model_head`

## Claim Boundary

Allowed repo-side claims:

- the repo contains a canonical general-purpose 5080 onefile lane
- the lane has syntax/test/parity coverage and delivery helpers
- smoke/evidence/package flows can be exercised locally

Blocked claims unless measured evidence exists:

- "Gemma-2B was beaten"
- frontier-grade quality superiority
- release-grade strength from smoke-only runs
- impossible reverse engineering

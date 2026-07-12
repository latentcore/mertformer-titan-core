from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.check_facts_drift as drift

OVERLAY_DIR = ROOT / "config" / "model"


def test_measure_overlay_params_matches_the_real_pilot_yaml_docstring() -> None:
    measured = drift.measure_overlay_params("mertformer_pilot_stabilization.yaml")
    assert measured == 172_668_035


def test_check_small_overlays_flags_a_stale_docstring_claim() -> None:
    stale_name = "_test_only_stale_docstring_overlay.yaml"
    stale_path = OVERLAY_DIR / stale_name
    stale_path.write_text(
        "# total parameters ....... 999,999,999  (~1.00B)\n"
        "hidden_size: 512\n"
        "intermediate_size: 1408\n"
        "num_layers: 9\n"
        "num_heads: 8\n"
        "num_kv_heads: 2\n"
        "head_dim: 64\n"
        "use_moe: true\n"
        "num_experts: 8\n"
        "moe_every_n_layers: 3\n"
        "moe_intermediate: 2048\n"
        "use_liquid: true\n"
        "liquid_layers_idx: [1, 4, 7]\n"
        "max_steps: 3000\n",
        encoding="utf-8",
    )
    try:
        offenders = drift.check_small_overlays()
    finally:
        stale_path.unlink(missing_ok=True)
    matches = [o for o in offenders if stale_name in o]
    assert matches, f"expected a stale-docstring finding for {stale_name}; offenders={offenders}"
    assert "999,999,999" in matches[0]


def test_check_canonical_gross_drift_passes_within_tolerance() -> None:
    offenders = drift.check_canonical_gross_drift()
    assert offenders == []

"""Tests for the rough-order FLOPs estimator, including the MoE-aware active-params path."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from economics.flops_estimator import (  # noqa: E402
    DEFAULT_PARAMS,
    estimate_inference_flops,
    estimate_training_flops,
)


def test_training_and_inference_flops_formulas():
    params, tokens = 1.0e9, 2.0e9
    assert estimate_training_flops(params, tokens) == 6.0 * params * tokens
    assert estimate_inference_flops(params, tokens) == 2.0 * params * tokens


def test_default_params_is_design_target():
    # Design-target default is intentionally preserved (not the measured 3.67B).
    assert DEFAULT_PARAMS == 2.64e9


def test_active_params_yields_lower_training_estimate():
    """The MoE-corrected active-params path must give a strictly lower TRAINING
    estimate than the dense params, since fewer params are active per token."""
    tokens = 23.6e9
    dense = 3.70e9
    active = 1.886e9  # MoE-corrected (8 experts, top-2 + shared)
    assert estimate_training_flops(active, tokens) < estimate_training_flops(dense, tokens)
    # ratio tracks the active/dense param ratio exactly (linear in params)
    ratio = estimate_training_flops(active, tokens) / estimate_training_flops(dense, tokens)
    assert abs(ratio - active / dense) < 1e-9

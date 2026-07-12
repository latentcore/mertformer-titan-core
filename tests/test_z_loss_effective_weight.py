"""Regression test for the z-loss double-multiply fix (config/config.py:329-346).

``layers/moe.py`` scales the router z-loss by ``cfg.z_loss_coef`` and folds it into
``aux_loss``; ``train/train.py`` then scales the whole ``aux_loss`` (including that
already-scaled z-loss) again by ``cfg.router_aux_loss_coef``. The effective z-loss
weight is therefore the PRODUCT of the two coefficients, not ``z_loss_coef`` alone.
Before the 2026-07-12 fix this product was 1e-4 * 0.02 = 2e-6 (~50x below the ~1e-3
Switch-Transformer/ST-MoE convention) — a real, unintentional double-multiply, not a
deliberately tiny coefficient. This test locks the corrected effective weight so the
double-multiply cannot silently regress back to a near-zero value.
"""
from config.config import MertFormerConfig


def test_effective_z_loss_weight_is_not_negligible():
    cfg = MertFormerConfig()
    effective = cfg.z_loss_coef * cfg.router_aux_loss_coef
    # Pre-fix value was 2e-6; assert we are at least an order of magnitude above that,
    # so the double-multiply regressing back to "effectively disabled" fails loudly.
    assert effective > 2e-5, (
        f"effective z-loss weight {effective:.2e} is back near the pre-fix 2e-6 "
        "double-multiply value (z_loss_coef * router_aux_loss_coef too small)"
    )


def test_effective_z_loss_weight_matches_documented_target():
    cfg = MertFormerConfig()
    effective = cfg.z_loss_coef * cfg.router_aux_loss_coef
    # Target is 1e-3 (Switch-Transformer/ST-MoE convention), chosen so the existing
    # double-multiply structure lands on it without restructuring moe.py/train.py.
    assert abs(effective - 1e-3) < 1e-6, (
        f"effective z-loss weight {effective:.2e} does not match the documented "
        "1e-3 target (config/config.py:329-346 comment)"
    )

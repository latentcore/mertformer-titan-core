from __future__ import annotations

import importlib

from config import config as config_module


def test_finalize_fills_grad_accum_without_overwriting_explicit_micro_batch(monkeypatch):
    """Partial-fill semantics: an explicitly-set field is preserved, a None one is solved.

    [2026-07-29] This assertion used to live against `__post_init__`. The auto-batch
    computation deliberately MOVED to `_finalize_config()` so that it runs AFTER
    `_apply_overrides()` (see test_overlay_batch_size_reaches_micro_and_accum below and
    the `__post_init__` docstring). The guarantee being pinned here is unchanged; only
    the call site moved.
    """
    monkeypatch.setattr(config_module, "auto_configure_batch_size", lambda target_global_batch, conf=None: (7, 11))
    fresh = config_module.MertFormerConfig(micro_batch_size=2, grad_accum_steps=None)
    # Construction alone no longer auto-fills -- that is the whole point of the move.
    assert fresh.micro_batch_size == 2
    assert fresh.grad_accum_steps is None
    config_module._finalize_config(fresh)
    assert fresh.micro_batch_size == 2, "explicit micro_batch_size must survive finalize"
    assert fresh.grad_accum_steps == 11, "None grad_accum_steps must be auto-solved"


def test_construction_does_not_autofill_batch_shape(monkeypatch):
    """`MertFormerConfig()` must leave the batch shape unresolved until finalize.

    If construction pre-fills micro/accum, `_finalize_config`'s `is None` guard can
    never fire, and any overlay that changes `batch_size` is silently ignored.
    """
    monkeypatch.setattr(config_module, "auto_configure_batch_size", lambda target_global_batch, conf=None: (7, 11))
    fresh = config_module.MertFormerConfig()
    assert fresh.micro_batch_size is None
    assert fresh.grad_accum_steps is None


def test_overlay_batch_size_reaches_micro_and_accum(monkeypatch):
    """Regression: a YAML overlay changing batch_size must reshape micro x accum.

    Before 2026-07-29 `__post_init__` solved micro/accum from the PRE-overlay
    batch_size, so `_finalize_config`'s recompute guard was already satisfied and an
    overlay setting `batch_size: 1024` trained at the 128-shaped micro/accum instead --
    1/8 of the intended global batch, invisible to the TITAN_STRICT_TOKEN_BUDGET guard
    (which reads cfg.batch_size, not the realized product).
    """
    monkeypatch.setattr(
        config_module,
        "auto_configure_batch_size",
        lambda target_global_batch, conf=None: (1, target_global_batch),
    )
    fresh = config_module.MertFormerConfig()
    config_module._apply_overrides(fresh, {"batch_size": 1024})
    config_module._finalize_config(fresh)
    assert fresh.batch_size == 1024
    assert fresh.micro_batch_size * fresh.grad_accum_steps == 1024


def test_use_precomputed_logits_respects_env_override(monkeypatch):
    monkeypatch.setenv("TITAN_USE_PRECOMPUTED_LOGITS", "0")
    module = importlib.reload(config_module)
    try:
        fresh = module.MertFormerConfig()
        assert fresh.use_precomputed_logits is False
    finally:
        monkeypatch.delenv("TITAN_USE_PRECOMPUTED_LOGITS", raising=False)
        importlib.reload(module)

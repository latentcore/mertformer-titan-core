from __future__ import annotations

import importlib

from config import config as config_module


def test_post_init_fills_grad_accum_without_overwriting_explicit_micro_batch(monkeypatch):
    monkeypatch.setattr(config_module, "auto_configure_batch_size", lambda target_global_batch, conf=None: (7, 11))
    fresh = config_module.MertFormerConfig(micro_batch_size=2, grad_accum_steps=None)
    assert fresh.micro_batch_size == 2
    assert fresh.grad_accum_steps == 11


def test_use_precomputed_logits_respects_env_override(monkeypatch):
    monkeypatch.setenv("TITAN_USE_PRECOMPUTED_LOGITS", "0")
    module = importlib.reload(config_module)
    try:
        fresh = module.MertFormerConfig()
        assert fresh.use_precomputed_logits is False
    finally:
        monkeypatch.delenv("TITAN_USE_PRECOMPUTED_LOGITS", raising=False)
        importlib.reload(module)

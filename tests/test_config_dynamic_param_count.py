"""Regression test for D5: dynamic_param_count was a dead getattr fallback.

[2026-07-11] ``auto_configure_batch_size`` read
``getattr(conf, "dynamic_param_count", 3.673e9)`` for its VRAM-budgeting param count,
but no code anywhere in the repo ever set a ``dynamic_param_count`` attribute on any
config object — so the ``getattr`` always fell through to the hardcoded constant
regardless of which model size (171M pilot, 36M smoke, canonical 3.67B, ...) was
actually configured. The code "read config" in appearance only; behavior never
changed. Fixed by computing an analytical estimate from the config's own real
architecture fields (``_estimate_total_params``). These tests pin that the estimate,
and the batch-size solver built on top of it, actually respond to config size.
"""
from types import SimpleNamespace
from unittest import mock

import config.config as config_module


def _stub_conf(**overrides):
    base = dict(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=5632,
        num_layers=18,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        use_moe=True,
        num_experts=8,
        moe_every_n_layers=3,
        moe_intermediate=8192,
        use_8bit_adam=True,
        max_seq_len=4096,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_estimate_total_params_scales_with_model_size():
    small = _stub_conf(hidden_size=512, intermediate_size=1408, num_layers=9,
                        num_heads=8, num_kv_heads=2, head_dim=64, moe_intermediate=2048)
    canonical = _stub_conf()
    small_params = config_module._estimate_total_params(small)
    canonical_params = config_module._estimate_total_params(canonical)
    assert small_params != canonical_params
    assert small_params < canonical_params


def test_estimate_total_params_is_not_the_old_dead_constant():
    # The old code always returned exactly 3.673e9 for ANY conf. A real formula must
    # not coincidentally reproduce that exact constant for a differently-sized config.
    small = _stub_conf(hidden_size=512, intermediate_size=1408, num_layers=9,
                        num_heads=8, num_kv_heads=2, head_dim=64, moe_intermediate=2048)
    assert config_module._estimate_total_params(small) != 3.673 * 10**9


class _FakeCudaProps:
    total_memory = 40 * 1024**3  # 40GB


def test_auto_configure_batch_size_responds_to_config_size_on_cuda_path():
    """End-to-end: two differently-sized configs must not produce identical
    (micro_batch, grad_accum_steps) on the CUDA-gated physics-based solver path."""
    small = _stub_conf(hidden_size=512, intermediate_size=1408, num_layers=9,
                        num_heads=8, num_kv_heads=2, head_dim=64, moe_intermediate=2048,
                        max_seq_len=256)
    canonical = _stub_conf()

    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.device_count", return_value=1), \
         mock.patch("torch.cuda.get_device_properties", return_value=_FakeCudaProps()):
        out_small = config_module.auto_configure_batch_size(target_global_batch=128, conf=small)
        out_canonical = config_module.auto_configure_batch_size(target_global_batch=128, conf=canonical)

    assert out_small != out_canonical, (
        "auto_configure_batch_size produced IDENTICAL output for a 9-layer/512-hidden "
        "config and the canonical 18-layer/2048-hidden config — the dead-getattr "
        "regression (D5) is back."
    )

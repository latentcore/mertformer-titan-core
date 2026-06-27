import sys
from contextlib import contextmanager
from pathlib import Path

import torch

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from config.config import cfg

# NOTE: The legacy "mla" name (file name, test names, local vars) is a fossil.
# The real attention class is GQA (grouped-query attention), NOT latent-MLA;
# latent-MLA (low-rank KV bottleneck) is intentionally NOT implemented. The
# module/file rename (mla.py -> gqa.py) is a frozen-path policy decision (see
# DECISIONS.md MLA->GQA rename) and is deliberately left untouched here.
from layers.mla import GQA, RotaryEmbedding


@contextmanager
def _cfg_patch(**overrides):
    _missing = object()
    original = {}
    for key, value in overrides.items():
        original[key] = getattr(cfg, key, _missing)
        setattr(cfg, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is _missing:
                delattr(cfg, key)
            else:
                setattr(cfg, key, value)


def _mla_tiny_overrides(rope_dim):
    return {
        "hidden_size": 128,
        "num_heads": 8,
        "head_dim": 16,
        "num_kv_heads": 8,
        "max_seq_len": 64,
        "rope_dim": rope_dim,
        "attention_dropout": 0.0,
    }


def test_rotary_cache_growth_no_exception():
    emb = RotaryEmbedding(dim=16, max_seq_len=8, base=100000.0)
    x = torch.zeros(1, 1, 16, 16)
    cos, sin = emb(x, seq_len=16, offset=0)
    assert cos.shape == (1, 1, 16, 16)
    assert sin.shape == (1, 1, 16, 16)


def test_decoupled_rope_with_none_rope_dim_is_safe():
    with _cfg_patch(**_mla_tiny_overrides(rope_dim=None)):
        mla = GQA()
        x = torch.randn(1, 4, 128)
        y, _ = mla(x, decoupled_rope=True, past_key_value=None, use_cache=False)
        assert y.shape == (1, 4, 128)
        assert torch.isfinite(y).all()


def test_decoupled_rope_with_small_custom_rope_dim_is_safe():
    with _cfg_patch(**_mla_tiny_overrides(rope_dim=4)):
        mla = GQA()
        x = torch.randn(1, 4, 128)
        y, _ = mla(x, decoupled_rope=True, past_key_value=None, use_cache=False)
        assert y.shape == (1, 4, 128)
        assert torch.isfinite(y).all()


def test_mla_has_no_static_causal_mask_buffer():
    with _cfg_patch(**_mla_tiny_overrides(rope_dim=None)):
        mla = GQA()
        buffer_names = dict(mla.named_buffers()).keys()
        assert "causal_mask" not in buffer_names


def test_mla_rope_base_fallback_matches_config_default():
    original = getattr(cfg, "rope_base")
    try:
        delattr(cfg, "rope_base")
        with _cfg_patch(**_mla_tiny_overrides(rope_dim=None)):
            mla = GQA()
            assert mla.rope_base == 100000.0
            assert mla.rotary_emb.base == 100000.0
    finally:
        setattr(cfg, "rope_base", original)


def test_mla_kv_cache_offset_path_shape_and_finite():
    with _cfg_patch(**_mla_tiny_overrides(rope_dim=None)):
        mla = GQA()
        x_prefill = torch.randn(1, 6, 128)
        y_prefill, past = mla(x_prefill, decoupled_rope=False, past_key_value=None, use_cache=True)
        assert y_prefill.shape == (1, 6, 128)
        assert torch.isfinite(y_prefill).all()
        assert past is not None

        x_next = torch.randn(1, 2, 128)
        y_next, present = mla(x_next, decoupled_rope=False, past_key_value=past, use_cache=True)
        assert y_next.shape == (1, 2, 128)
        assert torch.isfinite(y_next).all()
        assert present is not None
        k, v = present
        assert k.shape[2] == 8 and v.shape[2] == 8


def test_flash_attn_disabled_fallback_still_runs(monkeypatch):
    import layers.mla as mla_mod

    monkeypatch.setattr(mla_mod, "FLASH_ATTN_AVAILABLE", False)
    with _cfg_patch(**_mla_tiny_overrides(rope_dim=None)):
        mla = mla_mod.GQA()
        mla.train()
        x = torch.randn(1, 4, 128)
        y, _ = mla(x, decoupled_rope=False, past_key_value=None, use_cache=False)
        assert y.shape == (1, 4, 128)
        assert torch.isfinite(y).all()

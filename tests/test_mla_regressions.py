import sys
from contextlib import contextmanager
from pathlib import Path

import torch

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from config.config import cfg
from layers.mla import MLA, RotaryEmbedding


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
        mla = MLA()
        x = torch.randn(1, 4, 128)
        y, _ = mla(x, decoupled_rope=True, past_key_value=None, use_cache=False)
        assert y.shape == (1, 4, 128)
        assert torch.isfinite(y).all()


def test_decoupled_rope_with_small_custom_rope_dim_is_safe():
    with _cfg_patch(**_mla_tiny_overrides(rope_dim=4)):
        mla = MLA()
        x = torch.randn(1, 4, 128)
        y, _ = mla(x, decoupled_rope=True, past_key_value=None, use_cache=False)
        assert y.shape == (1, 4, 128)
        assert torch.isfinite(y).all()


def test_mla_has_no_static_causal_mask_buffer():
    with _cfg_patch(**_mla_tiny_overrides(rope_dim=None)):
        mla = MLA()
        buffer_names = dict(mla.named_buffers()).keys()
        assert "causal_mask" not in buffer_names

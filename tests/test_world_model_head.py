from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from config.config import cfg
from layers.world_model_head import CausalWorldModelHead
from model.transformers import MertFormer


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


def test_world_model_head_shapes():
    h = CausalWorldModelHead(hidden_size=64, horizon=2)
    x = torch.randn(3, 5, 64)
    out = h(x)
    assert out.dynamics_logits.shape == (3, 2, 64)
    assert out.latent_state.shape == (3, 64)
    assert out.uncertainty.shape == (3,)
    assert torch.isfinite(out.dynamics_logits).all()


def test_mertformer_world_model_side_output_non_breaking_signature():
    overrides = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 16,
        "num_layers": 2,
        "num_hidden_layers": 2,
        "vocab_size": 256,
        "max_seq_len": 32,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "use_moe": False,
        "use_liquid": False,
        "use_world_model_head": True,
        "world_model_horizon": 2,
    }
    with _cfg_patch(**overrides):
        model = MertFormer().eval()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, aux, pkv = model(x, use_cache=False)
        assert logits.shape == (2, 8, cfg.vocab_size)
        assert aux.ndim == 0
        assert pkv is None
        wm = model.get_last_world_model_outputs()
        assert wm is not None
        assert wm["world_dynamics_logits"].shape == (2, 2, cfg.hidden_size)


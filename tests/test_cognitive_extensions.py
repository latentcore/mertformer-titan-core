import sys
from contextlib import contextmanager
from pathlib import Path

import torch

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from config.config import cfg
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


def _tiny_all_extensions_cfg():
    return {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 32,
        "num_layers": 3,
        "num_hidden_layers": 3,
        "vocab_size": 512,
        "max_seq_len": 64,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "use_gradient_checkpointing": False,
        "use_moe": True,
        "moe_every_n_layers": 1,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "active_experts": 2,
        "use_liquid": True,
        "liquid_layers_idx": [0, 1, 2],
        "liquid_every_n_layers": 0,
        "use_hierarchical_kv_cache": True,
        "hkv_short_window": 8,
        "hkv_long_stride": 2,
        "hkv_max_long_blocks": 4,
        "use_global_workspace_broadcast": True,
        "workspace_blend": 0.6,
        "use_cross_expert_sync_bus": True,
        "cross_expert_sync_gain": 0.05,
        "use_latent_ode_state_channel": True,
        "latent_ode_dt": 1.0,
        "use_neuromodulatory_gain": True,
        "use_structural_plasticity": True,
        "structural_update_interval": 1,
        "use_hebbian_plasticity": True,
        "hebbian_eta": 0.01,
        "hebbian_decay": 0.9,
        "use_neuro_symbolic_layer": True,
        "neuro_symbolic_rules": 4,
    }


def test_all_extension_flags_forward_and_cache_decode():
    with _cfg_patch(**_tiny_all_extensions_cfg()):
        model = MertFormer()
        model.eval()

        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, aux, _ = model(x, use_cache=False)
        assert logits.shape == (2, 16, cfg.vocab_size)
        assert torch.isfinite(logits).all()
        assert aux.ndim == 0

        prefill = torch.randint(0, cfg.vocab_size, (1, 12))
        _, _, past = model(prefill, use_cache=True)
        assert past is not None
        nxt = torch.randint(0, cfg.vocab_size, (1, 1))
        logits_next, _, present = model(nxt, past_key_values=past, use_cache=True)
        assert logits_next.shape == (1, 1, cfg.vocab_size)
        assert torch.isfinite(logits_next).all()
        assert present is not None

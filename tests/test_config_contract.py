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


# --- O-2 [2026-07-29]: validate_layer_config must be a validator, not a mutator ---------
#
# The bf16 enforcement used to be the tail of validate_layer_config(), so every one of its
# five external callers silently reassigned cfg.param_dtype as a side effect of asking
# "is this layer config valid?". The concrete victim was
# scripts/offline_4060_demo_train.py, which sets param_dtype=float32 together with
# use_amp=False (deliberate full-fp32 for a stable 4060 demo) and then calls the validator
# 61 lines later -- flipping that choice back to bfloat16 on a bf16-capable card, and doing
# it silently because the notice goes through _cfg_print (gated on TITAN_CONFIG_VERBOSE).

def test_validate_layer_config_does_not_mutate_param_dtype():
    """The validator must leave param_dtype exactly as the caller set it."""
    import torch
    from config.config import MertFormerConfig, validate_layer_config

    conf = MertFormerConfig()
    conf.device = "cuda"                 # the branch that used to trigger the override
    conf.param_dtype = torch.float32     # what offline_4060_demo_train.py asks for

    validate_layer_config(conf)

    assert conf.param_dtype is torch.float32, (
        "validate_layer_config mutated param_dtype; the bf16 enforcement belongs in "
        "enforce_cuda_bf16_param_dtype()"
    )


def test_enforcement_is_still_exposed_and_callable():
    """The behaviour was split out, not deleted -- import-time semantics are unchanged."""
    import torch
    from config.config import MertFormerConfig, enforce_cuda_bf16_param_dtype

    conf = MertFormerConfig()
    conf.device = "cpu"                  # non-CUDA: enforcement must be a no-op
    conf.param_dtype = torch.float32
    enforce_cuda_bf16_param_dtype(conf)
    assert conf.param_dtype is torch.float32


def test_validator_still_raises_on_liquid_moe_overlap():
    """Splitting the side effect must not weaken the actual validation."""
    import pytest as _pytest
    from config.config import MertFormerConfig, validate_layer_config

    conf = MertFormerConfig()
    conf.use_liquid = True
    conf.use_moe = True
    # Force a collision: put a Liquid layer on a layer the MoE schedule also claims.
    conf.liquid_layers_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15, 16, 17]
    conf.moe_every_n_layers = 1
    with _pytest.raises(ValueError, match="overlap"):
        validate_layer_config(conf)


# --- O-1 [2026-07-29]: duplicate-name config fields must agree after overlays -----------
#
# num_layers/num_hidden_layers and num_heads/num_attention_heads are two spellings each of
# one quantity (train/ reads num_layers, layers/mla.py reads num_heads). Nothing enforced
# equality, and every config/model/*.yaml overlay writes BOTH names by hand -- so an
# overlay that set only one would silently build a model whose layer or head count differed
# between components.

def test_alias_mismatch_is_rejected_for_layer_count():
    import pytest as _pytest
    from config.config import MertFormerConfig, _finalize_config

    conf = MertFormerConfig()
    conf.num_layers = 18
    conf.num_hidden_layers = 12          # an overlay that set only one spelling
    with _pytest.raises(ValueError, match="ALIAS MISMATCH"):
        _finalize_config(conf)


def test_alias_mismatch_is_rejected_for_head_count():
    import pytest as _pytest
    from config.config import MertFormerConfig, _finalize_config

    conf = MertFormerConfig()
    conf.num_heads = 16
    conf.num_attention_heads = 8
    with _pytest.raises(ValueError, match="ALIAS MISMATCH"):
        _finalize_config(conf)


def test_live_singleton_and_all_overlays_are_alias_consistent():
    """Zero-false-positive check: the shipped config and every overlay already agree."""
    import glob
    import yaml
    from config.config import cfg, _ALIAS_PAIRS

    for primary, alias in _ALIAS_PAIRS:
        assert int(getattr(cfg, primary)) == int(getattr(cfg, alias)), (
            f"shipped cfg has {primary} != {alias}"
        )

    overlays = sorted(glob.glob("config/model/*.yaml"))
    assert overlays, "no overlays found - test would be vacuous"
    for path in overlays:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for primary, alias in _ALIAS_PAIRS:
            if primary in data or alias in data:
                assert data.get(primary) == data.get(alias), (
                    f"{path}: {primary}={data.get(primary)} but {alias}={data.get(alias)}"
                )


def test_head_dim_convention_warns_but_does_not_raise():
    """head_dim != hidden_size/num_heads is legal: layers/mla.py sizes q_proj/o_proj from
    num_heads * head_dim, so a different inner attention width is a valid design."""
    from config.config import MertFormerConfig, _finalize_config

    conf = MertFormerConfig()
    conf.hidden_size = 2048
    conf.num_heads = 16
    conf.num_attention_heads = 16
    conf.head_dim = 64                   # deliberately not 2048/16 = 128
    _finalize_config(conf)               # must NOT raise
    assert conf.head_dim == 64

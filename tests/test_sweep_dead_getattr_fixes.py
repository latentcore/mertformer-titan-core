"""Repo-wide sweep findings: two more dead-getattr regressions of the same class as
D5 (config/config.py dynamic_param_count) and the ADR-0005 assertion.

[2026-07-11] A full-repo scan of every ``getattr(cfg/conf/self, "X", default)`` and
``hasattr(cfg/conf/self, "X")`` call site (253 hits, 153 unique attribute names) cross-
checked against every real assignment site (``self.X =``, dataclass field
``X: type = ...``, ``setattr(..., "X", ...)``) anywhere in the repo. Five attributes
had ZERO assignment sites. Two were the already-known/fixed D5 and ADR-0005 items.
The other three:

- ``ffn_dropout`` (layers/ffn.py) -- an honest safe-default (``0.0``) for an
  unconfigured dropout rate, explicitly documented in the project's own reference
  doc as a deliberate, reviewed design choice (dropout=0 matches LLaMA/PaLM
  practice). NOT the same disease: no false claim of dynamism, just a standard
  default. Left untouched.
- ``deterministic`` (train/trainer_core.py:seed_all()) -- gates cudnn-deterministic /
  torch.use_deterministic_algorithms mode. Never had a backing field, so
  "Deterministic Mode" could never actually be turned on. Fixed below.
- ``qinn_every_n_layers`` (layers/mertformer_block.py) -- controls QINN placement
  cadence when ``use_qinn=True``. Never had a backing field, so a QINN experiment
  could never actually configure the cadence; it was permanently pinned to 1 (every
  layer). Fixed below.

Both fixes add a real dataclass field with the SAME default the dead getattr always
fell through to, so default behavior is unchanged; only explicit configuration now
actually takes effect.
"""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

import config.config as config_module


def test_deterministic_is_a_real_dataclass_field_defaulting_to_off():
    field_names = {f.name for f in dataclasses.fields(config_module.MertFormerConfig)}
    assert "deterministic" in field_names
    fresh = config_module.MertFormerConfig()
    assert fresh.deterministic is False  # preserves the old always-False fallback


def test_qinn_every_n_layers_is_a_real_dataclass_field_defaulting_to_one():
    field_names = {f.name for f in dataclasses.fields(config_module.MertFormerConfig)}
    assert "qinn_every_n_layers" in field_names
    fresh = config_module.MertFormerConfig()
    assert fresh.qinn_every_n_layers == 1  # preserves the old always-1 fallback


def test_seed_all_actually_activates_deterministic_mode_when_configured(monkeypatch):
    """End-to-end: seed_all() must genuinely flip cudnn state when cfg.deterministic
    is True, and must NOT when it's False -- proving the getattr isn't dead anymore."""
    from train import trainer_core

    prior_det = torch.backends.cudnn.deterministic
    prior_bench = torch.backends.cudnn.benchmark
    prior_algos = torch.are_deterministic_algorithms_enabled()
    try:
        monkeypatch.setattr(trainer_core.cfg, "deterministic", False)
        torch.backends.cudnn.deterministic = False
        trainer_core.seed_all(1234)
        assert torch.backends.cudnn.deterministic is False

        monkeypatch.setattr(trainer_core.cfg, "deterministic", True)
        trainer_core.seed_all(1234)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        # torch.use_deterministic_algorithms() is process-global and NOT undone by
        # resetting cudnn.deterministic/benchmark -- must be explicitly restored so
        # this test doesn't leak deterministic-algorithm warnings into later tests.
        torch.backends.cudnn.deterministic = prior_det
        torch.backends.cudnn.benchmark = prior_bench
        torch.use_deterministic_algorithms(prior_algos, warn_only=True)


def test_qinn_every_n_layers_actually_changes_block_construction(monkeypatch):
    """End-to-end: a MertFormerBlock must genuinely honor a non-default
    qinn_every_n_layers, proving the cadence is no longer permanently pinned to 1."""
    from layers.mertformer_block import MertFormerBlock
    cfg = config_module.cfg

    monkeypatch.setattr(cfg, "use_qinn", True)
    monkeypatch.setattr(cfg, "qinn_every_n_layers", 2)

    block_odd = MertFormerBlock(layer_id=0)   # (0+1) % 2 != 0 -> no QINN
    block_even = MertFormerBlock(layer_id=1)  # (1+1) % 2 == 0 -> QINN present

    if block_even.qinn is None:
        pytest.skip("UnitaryQINN unavailable in this environment")
    assert block_odd.qinn is None
    assert block_even.qinn is not None

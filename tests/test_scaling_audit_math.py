"""Regression test for scaling_audit_math.py's shared-expert + moe-intermediate bugs.

[2026-07-27] `estimate_params()` had two compounding undercounts for every MoE layer:
(1) it reused the dense-FFN `intermediate_size` for MoE experts instead of the real,
larger `moe_intermediate` (layers/moe.py's BitSwiGLU experts are sized off
`moe_intermediate`, not `intermediate_size` -- config/config.py's own
`_estimate_total_params` already made this distinction correctly); (2) it omitted
layers/moe.py's always-instantiated, always-active "shared expert" entirely from both
the total- and active-parameter sums. Together these made the printed "ACTIVE
PARAMETERS" figure diverge sharply from the real architecture (and from
ARCHITECTURE.md's independently-sourced ~1.86B active-param figure). Fixed to mirror
config/config.py's `_estimate_total_params` MoE-sizing + shared-expert accounting.

These tests inject an isolated stub config (monkeypatching the module-level `cfg`
`scaling_audit_math.py` imported at module load) rather than reading the live global
`config.config.cfg` singleton -- that singleton is mutated by other tests elsewhere in
the suite and is not guaranteed pristine by the time this file runs alphabetically.
"""
import re
from types import SimpleNamespace

import scripts.scaling_audit_math as scaling_audit_math


def _stub_cfg(**overrides):
    base = dict(
        model_name="test-stub",
        vocab_size=128256,
        hidden_size=2048,
        num_layers=18,
        intermediate_size=5632,
        moe_intermediate=8192,
        num_experts=8,
        active_experts=2,
        moe_every_n_layers=3,
        num_heads=16,
        head_dim=128,
        use_moe=True,
        batch_size=128,
        max_seq_len=4096,
        max_steps=45000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _run_and_capture(monkeypatch, capsys, cfg_stub):
    monkeypatch.setattr(scaling_audit_math, "cfg", cfg_stub)
    scaling_audit_math.estimate_params()
    out = capsys.readouterr().out
    total_b = float(re.search(r"TOTAL PARAMETERS.*?~([\d.]+)\s*B", out).group(1))
    active_b = float(re.search(r"ACTIVE PARAMETERS.*?~([\d.]+)\s*B", out).group(1))
    return total_b, active_b


def test_canonical_stub_total_matches_measured_within_one_percent(monkeypatch, capsys):
    # Canonical measured total (reports/param_accounting_report.md, FACTS.json):
    # 3,672,982,022 (~3.67B). This script's formula is an analytical approximation,
    # not the measured count, so an exact match isn't expected -- but omitting the
    # shared expert (the pre-fix bug) undercounted by ~300M (~8%), far outside 1%.
    total_b, _ = _run_and_capture(monkeypatch, capsys, _stub_cfg())
    measured_b = 3_672_982_022 / 1e9
    assert abs(total_b - measured_b) / measured_b < 0.01


def test_canonical_stub_active_params_include_shared_expert(monkeypatch, capsys):
    stub = _stub_cfg()
    _, active_b = _run_and_capture(monkeypatch, capsys, stub)

    moe_count = stub.num_layers // stub.moe_every_n_layers
    dense_count = stub.num_layers - moe_count
    embedding = stub.vocab_size * stub.hidden_size
    attn_per_layer = 4 * (stub.hidden_size * (stub.num_heads * stub.head_dim))
    dense_ffn = 3 * stub.hidden_size * stub.intermediate_size
    moe_ffn_correct = 3 * stub.hidden_size * stub.moe_intermediate
    router = stub.hidden_size * stub.num_experts
    shared_correct = moe_ffn_correct + 1  # +1 = the scalar shared_gate param

    active_moe_correct = moe_ffn_correct * stub.active_experts + shared_correct + router
    hand_active_correct = (
        embedding
        + stub.num_layers * attn_per_layer
        + dense_count * dense_ffn
        + moe_count * active_moe_correct
        + stub.num_layers * (2 * stub.hidden_size)
        + stub.hidden_size
    )

    moe_ffn_old_buggy = 3 * stub.hidden_size * stub.intermediate_size  # pre-fix: dense size reused
    active_moe_old_buggy = moe_ffn_old_buggy * stub.active_experts + router  # no shared expert
    hand_active_old_buggy = (
        embedding
        + stub.num_layers * attn_per_layer
        + dense_count * dense_ffn
        + moe_count * active_moe_old_buggy
        + stub.num_layers * (2 * stub.hidden_size)
        + stub.hidden_size
    )

    assert abs(active_b - hand_active_correct / 1e9) < 0.001
    assert hand_active_correct > hand_active_old_buggy  # the fix must raise the active count


def test_moe_layers_bigger_stub_scales_active_params(monkeypatch, capsys):
    small = _stub_cfg(hidden_size=512, intermediate_size=1408, num_layers=9,
                       num_heads=8, head_dim=64, moe_intermediate=2048)
    canonical = _stub_cfg()
    _, small_active = _run_and_capture(monkeypatch, capsys, small)
    _, canonical_active = _run_and_capture(monkeypatch, capsys, canonical)
    assert small_active < canonical_active

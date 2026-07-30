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
        # [2026-07-29] num_kv_heads was MISSING from this stub, which is why the GQA
        # attention bug could hide here: without it the test could only express the
        # (wrong) MHA formula. The canonical config is 16 query heads / 8 kv heads.
        num_kv_heads=8,
        # [2026-07-29] use_liquid / liquid_layers_idx were also missing, which is why the
        # Liquid/CfC mixers being omitted from BOTH parameter sums (~50.35M, 1.37% of the
        # model) was invisible here: with no such field the script's
        # getattr(cfg, "use_liquid", False) resolved to False and counted zero Liquid
        # layers, matching the (also wrong) expectation. Canonical config: layers 4/10/16.
        use_liquid=True,
        liquid_layers_idx=[4, 10, 16],
        liquid_every_n_layers=0,
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
    # 3,672,982,022 (~3.67B).
    #
    # [2026-07-29] This assertion used to pass for the WRONG reason. Two errors were
    # cancelling: the script counted attention with the MHA formula (+75,497,472 over 18
    # layers) while omitting the Liquid/CfC mixers entirely (-50,350,080), plus the
    # LiquidRouter's fluid_mixer/fluid_gate (-147,456) and GQA's q_norm/k_norm (-4,608).
    # Net error landed inside the 1% band, so a 1%-tolerance test could not see either
    # bug. With all four fixed the analytical sum now reproduces the measured count
    # EXACTLY, so this asserts exact equality at the printed 3-decimal precision.
    total_b, _ = _run_and_capture(monkeypatch, capsys, _stub_cfg())
    measured_b = 3_672_982_022 / 1e9
    assert total_b == round(measured_b, 3), (
        f"analytical total {total_b} B must reproduce the measured "
        f"{round(measured_b, 3)} B exactly"
    )


def test_canonical_stub_active_params_include_shared_expert(monkeypatch, capsys):
    stub = _stub_cfg()
    _, active_b = _run_and_capture(monkeypatch, capsys, stub)

    moe_count = stub.num_layers // stub.moe_every_n_layers
    dense_count = stub.num_layers - moe_count
    embedding = stub.vocab_size * stub.hidden_size
    # [2026-07-29] This hand-computation previously hardcoded the MHA formula
    # `4 * hidden * (num_heads * head_dim)` and asserted the script matched it -- so the
    # test LOCKED IN the bug: layers/mla.py is GQA, where k_proj/v_proj are only
    # `num_kv_heads * head_dim` wide. Verified against a live GQA() instance that the
    # expression below equals the summed numel of its q/k/v/o projections exactly.
    # At the canonical 16q/8kv the old formula overcounted by 4,194,304 params per layer
    # (+75,497,472 over 18 layers) and contradicted config._estimate_total_params.
    attn_per_layer = (
        stub.hidden_size * (stub.num_heads * stub.head_dim)          # q_proj
        + 2 * (stub.hidden_size * (stub.num_kv_heads * stub.head_dim))  # k_proj + v_proj
        + (stub.num_heads * stub.head_dim) * stub.hidden_size         # o_proj
        + 2 * stub.head_dim                                           # q_norm + k_norm
    )
    dense_ffn = 3 * stub.hidden_size * stub.intermediate_size
    moe_ffn_correct = 3 * stub.hidden_size * stub.moe_intermediate
    # LiquidRouter, not a bare Linear: main_proj + depthwise Conv1d fluid_mixer
    # (kernel = history_window = 4, bias=False) + fluid_gate.
    router = (
        stub.hidden_size * stub.num_experts
        + stub.hidden_size * 4
        + stub.hidden_size * stub.num_experts
    )
    shared_correct = moe_ffn_correct + 1  # +1 = the scalar shared_gate param
    # Liquid/CfC mixers: 4 hidden x hidden projections + tau_bias + LayerNorm(w, b).
    # Always dense, so fully counted in the ACTIVE sum too.
    liquid_count = len(
        [i for i in stub.liquid_layers_idx if 0 <= int(i) < stub.num_layers]
    ) if getattr(stub, "use_liquid", False) else 0
    liquid_total = liquid_count * (
        4 * stub.hidden_size * stub.hidden_size + stub.hidden_size + 2 * stub.hidden_size
    )

    active_moe_correct = moe_ffn_correct * stub.active_experts + shared_correct + router
    hand_active_correct = (
        embedding
        + stub.num_layers * attn_per_layer
        + dense_count * dense_ffn
        + moe_count * active_moe_correct
        + liquid_total
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

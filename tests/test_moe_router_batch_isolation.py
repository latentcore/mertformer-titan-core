"""Batch-isolation regression test for the MoE LiquidRouter inference state (I.2.5 A2).

During single-token generation the router keeps a stateful ``inference_state`` rolling
buffer (layers/moe.py). ``MertFormer.reset_router_state()`` runs at the start of every
``generate()`` (model/transformers.py) and lays down a fresh zero state shaped
``[batch, hidden, window-1]`` — one independent row per sample. The roadmap flagged
(A2) that the gap was NOT a missing reset but a missing PROOF that the per-sample rows
stay isolated: one sample's tokens must never influence another sample's router logits.
This pins that invariant, so a future change that couples the rows (e.g. a stray
cross-batch reduction) fails here instead of silently corrupting batched generation.

``LiquidRouter`` takes only ``(hidden_size, num_experts)`` and is config-singleton-free,
so these tests are deterministic and order-independent (unlike a full-MoE construction).
"""
import torch

from layers.moe import LiquidRouter


def _fresh_router(hidden: int = 16, experts: int = 8, seed: int = 0) -> LiquidRouter:
    torch.manual_seed(seed)
    router = LiquidRouter(hidden, experts)
    router.eval()  # single-token inference path; no training-only randomness
    return router


def test_router_row_is_independent_of_other_rows():
    hidden, experts, steps = 16, 8, 6
    router = _fresh_router(hidden, experts)
    window = router.history_window - 1

    torch.manual_seed(1)
    stream_a = [torch.randn(1, 1, hidden) for _ in range(steps)]
    stream_b = [torch.randn(1, 1, hidden) for _ in range(steps)]

    # Batched: row 0 = stream A, row 1 = stream B, fed one token at a time.
    router.set_state(torch.zeros(2, hidden, window))
    batched_row0 = []
    with torch.no_grad():
        for a, b in zip(stream_a, stream_b):
            out = router(torch.cat([a, b], dim=0))  # [2, 1, E]
            batched_row0.append(out[0:1].clone())

    # Same router (identical weights), stream A alone.
    router.set_state(torch.zeros(1, hidden, window))
    alone_row0 = []
    with torch.no_grad():
        for a in stream_a:
            alone_row0.append(router(a).clone())  # [1, 1, E]

    for i, (batched, alone) in enumerate(zip(batched_row0, alone_row0)):
        assert torch.allclose(batched, alone, atol=1e-5), (
            f"router row 0 leaked state from row 1 at step {i}: "
            f"max|diff|={(batched - alone).abs().max().item():.3e}"
        )


def test_router_identical_rows_stay_identical():
    # Complementary invariant: two rows fed the SAME stream must stay bit-identical the
    # whole way — no row-index-dependent state.
    hidden, experts, steps = 16, 8, 5
    router = _fresh_router(hidden, experts)
    window = router.history_window - 1
    router.set_state(torch.zeros(2, hidden, window))
    torch.manual_seed(2)
    with torch.no_grad():
        for _ in range(steps):
            tok = torch.randn(1, 1, hidden)
            out = router(torch.cat([tok, tok], dim=0))
            assert torch.allclose(out[0], out[1], atol=1e-6)


def test_router_state_genuinely_persists_across_steps():
    """Discriminating test: proves the rolling buffer actually carries real history
    (not just that batch rows are isolated from each other).

    A prior regression forced ``inference_state.zero_()`` unconditionally at the top of
    ``LiquidRouter.forward()``. Under that regression,
    ``test_router_row_is_independent_of_other_rows`` STILL PASSES — both compared paths
    (batched vs. alone) are equally degraded (both compute with zeroed history every
    step), so they stay equal to each other and the isolation test is blind to the bug.
    This test compares REAL persisted history against a forced-zero-every-step history
    and requires them to diverge once the rolling window has filled — the only way to
    positively confirm ``inference_state`` is doing anything at all.

    ``fluid_gate.weight`` is ``nn.init.zeros_``-initialized by design (III.9: "start with
    no fluid influence, let it learn"), so a freshly constructed router's output is
    history-independent regardless of this bug. Both routers below get identical
    non-zero ``fluid_gate`` weights injected (same seed on both) so the fluid path
    actually contributes and the comparison is meaningful.
    """
    hidden, experts = 16, 8
    window_needed = 3  # history_window - 1

    def _router_with_live_gate(seed: int) -> LiquidRouter:
        router = _fresh_router(hidden, experts, seed=seed)
        torch.manual_seed(1234)
        with torch.no_grad():
            router.fluid_gate.weight.copy_(torch.randn_like(router.fluid_gate.weight))
        return router

    router_real = _router_with_live_gate(seed=7)
    router_real.set_state(torch.zeros(1, hidden, window_needed))
    torch.manual_seed(99)
    stream = [torch.randn(1, 1, hidden) for _ in range(6)]
    with torch.no_grad():
        out_real_history = [router_real(t).clone() for t in stream]

    router_forced_zero = _router_with_live_gate(seed=7)
    with torch.no_grad():
        out_forced_zero = []
        for t in stream:
            router_forced_zero.set_state(torch.zeros(1, hidden, window_needed))  # simulate the regression
            out_forced_zero.append(router_forced_zero(t).clone())

    # Once the window has filled (step >= history_window), real history and
    # forced-zero history must DIFFER. If they're identical, inference_state is not
    # actually being used.
    diverged = any(
        not torch.allclose(a, b, atol=1e-6)
        for a, b in zip(out_real_history[3:], out_forced_zero[3:])
    )
    assert diverged, (
        "router output is IDENTICAL between real history and forced-zero history — "
        "inference_state is not actually being used. Check for an unconditional "
        ".zero_() at the top of forward()."
    )

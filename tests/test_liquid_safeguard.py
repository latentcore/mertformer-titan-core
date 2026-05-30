from utils.liquid_safeguard import update_liquid_spike_state
import torch

from layers.liquid import LiquidMixer
from layers.moe import LiquidRouter


def test_liquid_spike_counter_resets_on_safe_loss():
    counter, frozen, triggered = update_liquid_spike_state(
        loss_value=0.9,
        threshold=5.0,
        counter=2,
        patience=3,
        frozen_until=0,
        global_step=10,
        cooldown_steps=50,
        enabled=True,
    )
    assert counter == 0
    assert frozen == 0
    assert triggered is False


def test_liquid_spike_triggers_freeze_after_patience():
    counter = 0
    frozen = 0
    triggered = False
    for step in range(1, 4):
        counter, frozen, triggered = update_liquid_spike_state(
            loss_value=6.0,
            threshold=5.0,
            counter=counter,
            patience=3,
            frozen_until=frozen,
            global_step=step,
            cooldown_steps=25,
            enabled=True,
        )
    assert triggered is True
    assert counter == 0
    assert frozen == 3 + 25


def test_liquid_spike_disabled_noop():
    counter, frozen, triggered = update_liquid_spike_state(
        loss_value=9.0,
        threshold=5.0,
        counter=1,
        patience=3,
        frozen_until=0,
        global_step=5,
        cooldown_steps=10,
        enabled=False,
    )
    assert (counter, frozen, triggered) == (1, 0, False)


def test_liquid_router_inference_state_casts_to_activation_dtype():
    router = LiquidRouter(hidden_size=8, num_experts=2)
    state = torch.randn(1, 8, router.history_window - 1, dtype=torch.bfloat16)

    router._update_inference_state(state)

    assert router.inference_state.dtype == torch.bfloat16
    assert torch.equal(router.inference_state, state)


def _run_liquid_for_equivalence(model, x):
    x = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    y = model(x)
    loss = (y.float() ** 2).mean()
    loss.backward()
    grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return y.detach(), x.grad.detach().clone(), grads


def test_liquid_train_impls_match_baseline_forward_and_backward():
    torch.manual_seed(1453)
    baseline = LiquidMixer(16, fast_path=False, train_impl="baseline").double()
    baseline.train()
    x = torch.randn(2, 9, 16, dtype=torch.float64)
    y0, xg0, pg0 = _run_liquid_for_equivalence(baseline, x)

    for impl in ("precompute_input", "packed_pair"):
        candidate = LiquidMixer(16, fast_path=False, train_impl=impl).double()
        candidate.load_state_dict(baseline.state_dict())
        candidate.train()
        y1, xg1, pg1 = _run_liquid_for_equivalence(candidate, x)

        assert torch.max(torch.abs(y0 - y1)).item() < 1e-8
        assert torch.max(torch.abs(xg0 - xg1)).item() < 1e-8
        for name, grad0 in pg0.items():
            assert name in pg1
            assert torch.max(torch.abs(grad0 - pg1[name])).item() < 1e-8

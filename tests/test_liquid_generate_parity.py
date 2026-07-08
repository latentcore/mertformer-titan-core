"""
[2026-07-08] Regression tests for the `generate()` Liquid-state parity bug.

THE BUG
-------
`MertFormerBlock.forward()` used to call `self.liquid(x)` with no `h_init` and drop the
mixer's final hidden state. `MertFormer.generate()` threads `past_key_values` (the
attention KV cache) across incremental decode steps but had no equivalent for the
Liquid/CfC recurrent state, so `LiquidMixer.forward(x, h_init=None)` restarted the
recurrence from ZERO on every single decoded token. The Liquid layers still applied their
per-token transform, but carried no temporal context whatsoever during generation — a
stateless no-op, precisely the property they exist to provide. Teacher-forced training
(one forward over the whole sequence) never exercised the bug.

WHAT IS ASSERTED
----------------
1. `LiquidMixer` is an exact recurrence: streaming it chunk-by-chunk with the state
   threaded reproduces the single full-sequence pass.
2. `MertFormerBlock` reproduces its full-sequence output when decoded token-by-token,
   *only* when both the KV cache and the new `liquid_state` are threaded.
3. Negative control: dropping the `liquid_state` (i.e. the old behavior) makes the
   token-by-token path DIVERGE from the full-sequence path. Without this the test above
   could pass vacuously (e.g. if the block silently had no Liquid layer).
4. `MertFormer.generate()` publishes per-layer states and `reset_router_state()` clears
   them, so a fresh generation never inherits the previous sequence's recurrence.
"""

import pytest
import torch

from config.config import cfg
from layers.liquid import LiquidMixer


TOL = 1e-9  # float64 recurrence; chunking must be numerically exact, not just close


@pytest.fixture(scope="function")
def liquid_block_cfg():
    """Tiny config where layer 0 HAS a LiquidMixer and is NOT a MoE layer."""
    saved = {
        k: getattr(cfg, k, None)
        for k in (
            "hidden_size", "intermediate_size", "num_heads", "head_dim", "num_kv_heads",
            "num_layers", "max_seq_len", "device", "use_liquid", "liquid_layers_idx",
            "liquid_every_n_layers", "liquid_fast_path", "liquid_train_impl",
            "use_moe", "moe_every_n_layers", "dropout", "vocab_size", "tie_weights",
            "use_gradient_checkpointing", "num_experts", "num_experts_per_tok",
        )
    }

    cfg.hidden_size = 32
    cfg.intermediate_size = 64
    cfg.num_heads = 4
    cfg.head_dim = 8
    cfg.num_kv_heads = 2
    cfg.num_layers = 2
    cfg.max_seq_len = 64
    cfg.device = "cpu"
    cfg.dropout = 0.0
    cfg.vocab_size = 64
    cfg.tie_weights = True
    cfg.use_gradient_checkpointing = False

    cfg.use_liquid = True
    cfg.liquid_layers_idx = [0]        # layer 0 gets the LiquidMixer
    cfg.liquid_every_n_layers = 0
    cfg.liquid_fast_path = False       # no torch.compile in tests
    cfg.liquid_train_impl = "baseline"

    cfg.use_moe = False                # dense FFN -> deterministic, isolates Liquid
    cfg.moe_every_n_layers = 0

    yield cfg

    for k, v in saved.items():
        if v is not None:
            setattr(cfg, k, v)


def test_liquid_mixer_is_an_exact_chunked_recurrence():
    """Full-sequence pass == streaming the same sequence with h threaded through."""
    torch.manual_seed(1453)
    mixer = LiquidMixer(8, fast_path=False, train_impl="baseline").double().eval()
    x = torch.randn(2, 6, 8, dtype=torch.float64)

    with torch.no_grad():
        full, h_full = mixer(x, h_init=None, return_state=True)

        h = None
        chunks = []
        for t in range(x.size(1)):
            y_t, h = mixer(x[:, t : t + 1, :], h_init=h, return_state=True)
            chunks.append(y_t)
        streamed = torch.cat(chunks, dim=1)

    assert torch.max(torch.abs(full - streamed)).item() < TOL
    assert torch.max(torch.abs(h_full - h)).item() < TOL


def test_block_decode_matches_full_forward_when_liquid_state_is_threaded(liquid_block_cfg):
    """MertFormerBlock: token-by-token decode == full teacher-forced forward."""
    from layers.mertformer_block import MertFormerBlock

    torch.manual_seed(1453)
    block = MertFormerBlock(layer_id=0).double().eval()
    assert block.liquid is not None, "fixture must put a LiquidMixer on layer 0"
    assert not block.is_moe_layer, "fixture must keep layer 0 dense to isolate Liquid"

    x = torch.randn(2, 5, liquid_block_cfg.hidden_size, dtype=torch.float64)

    with torch.no_grad():
        full_out, _, _, _ = block(x, use_cache=False)

        past_kv = None
        liquid_state = None
        steps = []
        for t in range(x.size(1)):
            step_out, _, past_kv, liquid_state = block(
                x[:, t : t + 1, :],
                past_key_value=past_kv,
                use_cache=True,
                liquid_state=liquid_state,
            )
            steps.append(step_out)
        decoded = torch.cat(steps, dim=1)

    assert liquid_state is not None, "block must return its final Liquid hidden state"
    assert torch.max(torch.abs(full_out - decoded)).item() < 1e-8


def test_dropping_liquid_state_reproduces_the_old_divergence(liquid_block_cfg):
    """Negative control: the OLD behavior (h restarts at 0 each step) must NOT match."""
    from layers.mertformer_block import MertFormerBlock

    torch.manual_seed(1453)
    block = MertFormerBlock(layer_id=0).double().eval()
    x = torch.randn(2, 5, liquid_block_cfg.hidden_size, dtype=torch.float64)

    with torch.no_grad():
        full_out, _, _, _ = block(x, use_cache=False)

        past_kv = None
        steps = []
        for t in range(x.size(1)):
            # liquid_state deliberately NOT threaded -> h_init=None -> zeros every step
            step_out, _, past_kv, _ = block(
                x[:, t : t + 1, :], past_key_value=past_kv, use_cache=True
            )
            steps.append(step_out)
        stateless = torch.cat(steps, dim=1)

    # Position 0 is identical by construction (h starts at zero either way); the
    # divergence must show up on the later positions.
    assert torch.max(torch.abs(full_out[:, 1:, :] - stateless[:, 1:, :])).item() > 1e-6


def test_model_publishes_and_resets_liquid_states(liquid_block_cfg):
    """generate() threads states; reset_router_state() drops them."""
    from model.transformers import MertFormer

    torch.manual_seed(1453)
    model = MertFormer().eval()

    input_ids = torch.randint(0, liquid_block_cfg.vocab_size, (1, 3))
    with torch.no_grad():
        model(input_ids, use_cache=True)

    states = model.get_last_liquid_states()
    assert states is not None and len(states) == liquid_block_cfg.num_layers
    assert states[0] is not None, "layer 0 has a LiquidMixer -> must publish a state"
    assert states[0].shape == (1, liquid_block_cfg.hidden_size)
    assert states[1] is None, "layer 1 has no LiquidMixer -> None"

    model.reset_router_state(batch_size=1)
    assert model.get_last_liquid_states() is None

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=3, top_k=1, top_p=1.0)
    assert out.shape == (1, 6)

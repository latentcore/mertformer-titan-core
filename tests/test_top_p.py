"""Contract test for nucleus (top-p) filtering used in
``model/transformers.py`` ``MertFormer.generate()`` (the top-p block, ~lines 394-402).

``generate()`` applies top-p inline with no importable helper, so this test replicates
that exact algorithm and locks its contract on controlled logits: top_p>=1.0 is a no-op,
the argmax token always survives, the far tail is removed, and the kept set's original
probability mass is >= top_p (the nucleus property). If the inline block is ever extracted
into a helper, point the import here at it directly.
"""
import torch


def _apply_top_p(logits: torch.Tensor, top_p) -> torch.Tensor:
    # Verbatim mirror of the generate() top-p block (functional masked_fill form).
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def test_top_p_one_is_noop():
    logits = torch.randn(1, 10)
    assert torch.equal(_apply_top_p(logits.clone(), 1.0), logits)


def test_top_p_argmax_always_survives():
    logits = torch.tensor([[10.0, 1.0, 0.5, 0.1]])
    out = _apply_top_p(logits.clone(), 0.01)
    assert torch.isfinite(out).sum().item() >= 1
    assert out[0, 0].item() == 10.0


def test_top_p_removes_far_tail():
    logits = torch.tensor([[5.0, 4.0, 0.0, -5.0, -10.0]])
    out = _apply_top_p(logits.clone(), 0.9)
    assert out[0, -1].item() == float("-inf")
    assert torch.isfinite(out[0, 0])


def test_top_p_kept_mass_covers_threshold():
    torch.manual_seed(0)
    logits = torch.randn(1, 50)
    top_p = 0.8
    out = _apply_top_p(logits.clone(), top_p)
    probs = torch.softmax(logits, dim=-1)
    kept_mass = probs[torch.isfinite(out)].sum().item()
    assert kept_mass >= top_p - 1e-6

"""Gate 3 (DDP-unfreeze 60s check) regression test.

[2026-07-11] The Gate 3 guard in ``train/train.py`` (fires once, at
``global_step == cfg.liquid_warmup_steps``, when Liquid/tau params are unfrozen) used
to gather a hardcoded ``dummy = torch.tensor([1.0], device=accelerator.device)``
across ranks and check ``gathered.sum().item() != accelerator.num_processes``. Every
rank always contributes exactly ``1.0`` regardless of any real state, so the sum
always equals ``num_processes`` and the check was mathematically incapable of ever
firing -- it verified nothing. It was replaced with a real check: gather one actual
newly-unfrozen tau/liquid parameter's value across ranks and require them to be
identical (they were synced at DDP init and never updated while frozen, so any
mismatch right after unfreeze is real cross-rank divergence).

This cannot be exercised on real multi-GPU/NCCL here (single-device sandbox), per the
task's own instruction: statically confirm (a) ``num_processes == 1`` is a no-op, and
(b) the ``> 1`` branch reads a real parameter, not a constant -- and unit-test the one
piece that genuinely is hardware-independent, the pure tensor-equality comparison
(``_gate3_ranks_synced``), extracted so it doesn't need Accelerate/DDP at all.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train.train")

TRAIN_PY_SOURCE = Path(train.__file__).read_text(encoding="utf-8")


def _gate3_source_block() -> str:
    marker = "# [Gate 3] DDP-unfreeze 60s check"
    start = TRAIN_PY_SOURCE.index(marker)
    end = TRAIN_PY_SOURCE.index("# NOTE: This branch ONLY toggles requires_grad", start)
    return TRAIN_PY_SOURCE[start:end]


def _gate3_code_lines(block: str) -> str:
    """Same block with comment-only lines dropped, so a check for a removed
    anti-pattern isn't tripped up by that anti-pattern being *described* in a
    NOTE comment (as it is, deliberately, right above the real fix)."""
    return "\n".join(
        line for line in block.splitlines() if line.strip() and not line.strip().startswith("#")
    )


# ---------------------------------------------------------------------------
# Real unit test: the pure comparison function.
# ---------------------------------------------------------------------------

def test_gate3_ranks_synced_true_when_all_ranks_match():
    gathered = torch.tensor([0.4231, 0.4231, 0.4231])
    assert train._gate3_ranks_synced(gathered) is True


def test_gate3_ranks_synced_false_when_a_rank_diverges():
    gathered = torch.tensor([0.4231, 0.4231, 9.999])
    assert train._gate3_ranks_synced(gathered) is False


def test_gate3_ranks_synced_single_rank_trivially_true():
    gathered = torch.tensor([0.4231])
    assert train._gate3_ranks_synced(gathered) is True


# ---------------------------------------------------------------------------
# Static verification (no multi-GPU available): the dummy-tensor anti-pattern is
# gone, and the guard now reads a real parameter, gated behind num_processes > 1.
# ---------------------------------------------------------------------------

def test_gate3_no_longer_uses_hardcoded_dummy_tensor():
    code = _gate3_code_lines(_gate3_source_block())
    assert "torch.tensor([1.0]" not in code, (
        "Gate 3 still gathers a hardcoded constant -- gathered.sum() would always "
        "equal num_processes and the divergence check could never fire."
    )
    assert "gathered.sum().item() != accelerator.num_processes" not in code


def test_gate3_reads_a_real_tau_liquid_parameter():
    code = _gate3_code_lines(_gate3_source_block())
    assert "named_parameters()" in code
    assert '"tau" in n or "liquid" in n' in code
    assert "accelerator.gather" in code
    assert "_gate3_ranks_synced" in code


def test_gate3_guard_is_gated_behind_multi_process_check():
    code = _gate3_code_lines(_gate3_source_block())
    # Confirms the whole probe/gather/compare sequence is nested under the
    # num_processes > 1 guard, so a single-process run never touches
    # unwrap_model/gather at all (no-op by construction, not by accident).
    guard_idx = code.index("if accelerator.num_processes > 1:")
    probe_idx = code.index("unwrapped_for_probe")
    assert guard_idx < probe_idx

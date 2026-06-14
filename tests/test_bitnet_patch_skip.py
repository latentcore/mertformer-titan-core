from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from layers.bitlinear import BitLinear
from layers.bitnet_patch import apply_bitnet


class _Demo(nn.Module):
    """Children whose names exercise the bitnet skip-whitelist + the _proj exclusion."""

    def __init__(self) -> None:
        super().__init__()
        # Sensitive params that MUST stay FP (skipped):
        self.gate = nn.Linear(8, 8, bias=False)
        self.router = nn.Linear(8, 8, bias=False)
        self.tau_input_w = nn.Linear(8, 8, bias=False)
        self.shared_expert_gate = nn.Linear(8, 8, bias=False)
        # Output head (skipped by skip_output_head):
        self.lm_head = nn.Linear(8, 8, bias=False)
        # Standard projections that MUST be quantized (converted):
        self.gate_proj = nn.Linear(8, 8, bias=False)  # 'gate' substring, but ends with _proj
        self.up_proj = nn.Linear(8, 8, bias=False)


def test_bitnet_skips_sensitive_but_quantizes_gate_proj():
    m = _Demo()
    apply_bitnet(m, skip_output_head=True, verbose=False)

    # Sensitive layers preserved in full precision.
    assert isinstance(m.gate, nn.Linear) and not isinstance(m.gate, BitLinear)
    assert isinstance(m.router, nn.Linear) and not isinstance(m.router, BitLinear)
    assert isinstance(m.tau_input_w, nn.Linear) and not isinstance(m.tau_input_w, BitLinear)
    assert isinstance(m.shared_expert_gate, nn.Linear) and not isinstance(m.shared_expert_gate, BitLinear)

    # Output head preserved.
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, BitLinear)

    # Standard projections converted to BitLinear — including gate_proj, which the old
    # substring check ("gate" in name) wrongly skipped.
    assert isinstance(m.gate_proj, BitLinear)
    assert isinstance(m.up_proj, BitLinear)

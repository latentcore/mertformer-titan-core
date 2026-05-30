import torch

from layers.bitlinear import BitLinear
from layers.ffn import _ffn_packed_bitlinear
from layers.mla import _mla_packed_kv
from layers.moe import BitSwiGLU, set_moe_pack_enabled


def test_ffn_packed_projection_matches_two_bitlinear_paths():
    torch.manual_seed(1453)
    gate_proj = BitLinear(16, 32, bias=False).double()
    up_proj = BitLinear(16, 32, bias=False).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64)

    gate = gate_proj(x)
    up = up_proj(x)
    packed = _ffn_packed_bitlinear(x, torch.cat([gate_proj.weight, up_proj.weight], dim=0))
    packed_gate, packed_up = packed.chunk(2, dim=-1)

    assert torch.max(torch.abs(gate - packed_gate)).item() < 1e-8
    assert torch.max(torch.abs(up - packed_up)).item() < 1e-8


def test_moe_bitswiglu_packed_path_matches_baseline():
    torch.manual_seed(1453)
    expert = BitSwiGLU(16, 32).double()
    x = torch.randn(11, 16, dtype=torch.float64)

    set_moe_pack_enabled(False)
    baseline = expert(x)
    set_moe_pack_enabled(True)
    packed = expert(x)
    set_moe_pack_enabled(False)

    assert torch.max(torch.abs(baseline - packed)).item() < 1e-8


def test_mla_packed_kv_matches_two_bitlinear_paths():
    torch.manual_seed(1453)
    hidden = 16
    kv_out_dim = 24
    k_proj = BitLinear(hidden, kv_out_dim, bias=False).double()
    v_proj = BitLinear(hidden, kv_out_dim, bias=False).double()
    x = torch.randn(2, 7, hidden, dtype=torch.float64)

    k_base = k_proj(x)
    v_base = v_proj(x)
    k_pack, v_pack = _mla_packed_kv(x, k_proj.weight, v_proj.weight, kv_out_dim)

    assert torch.max(torch.abs(k_base - k_pack)).item() < 1e-8
    assert torch.max(torch.abs(v_base - v_pack)).item() < 1e-8

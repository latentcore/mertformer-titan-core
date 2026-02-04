"""Ternary bit-packing utilities (5 trits -> 1 byte)."""
from __future__ import annotations

from typing import Dict, Tuple
import json
from pathlib import Path

import torch


def _pack_trits(trits: torch.Tensor) -> Tuple[torch.ByteTensor, int]:
    flat = trits.reshape(-1).to(torch.int64)
    pad = (-flat.numel()) % 5
    if pad:
        pad_vals = torch.full((pad,), 1, dtype=flat.dtype, device=flat.device)
        flat = torch.cat([flat, pad_vals], dim=0)
    grouped = flat.view(-1, 5)
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int64, device=flat.device)
    packed = (grouped * powers).sum(dim=1).to(torch.uint8)
    return packed.cpu(), pad


def pack_ternary_tensor(tensor: torch.Tensor) -> Tuple[torch.ByteTensor, Dict]:
    """Pack a tensor with values in {-1,0,1} into bytes."""
    w = tensor.detach().cpu()
    w_q = torch.round(w).clamp(-1, 1).to(torch.int8)
    trits = (w_q + 1).to(torch.int64)  # map {-1,0,1} -> {0,1,2}
    packed, pad = _pack_trits(trits)
    meta = {
        "shape": list(w.shape),
        "pad": pad,
        "dtype": "ternary5in8",
    }
    return packed, meta


def pack_state_dict(state_dict: Dict[str, torch.Tensor], out_bin: Path, out_meta: Path) -> None:
    """Pack all float weights into a single binary with metadata index."""
    offsets = []
    cursor = 0
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    with out_bin.open("wb") as bf:
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                continue
            packed, meta = pack_ternary_tensor(tensor)
            data = packed.numpy().tobytes()
            bf.write(data)
            entry = {
                "name": name,
                "offset": cursor,
                "length": len(data),
                "meta": meta,
            }
            offsets.append(entry)
            cursor += len(data)

    payload = {
        "format": "mertformer_bitpack_v1",
        "entry_count": len(offsets),
        "entries": offsets,
    }
    out_meta.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

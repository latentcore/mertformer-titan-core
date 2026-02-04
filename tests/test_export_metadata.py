from __future__ import annotations

import json
from pathlib import Path
import torch

from mertformer_sdk.utils.bitpack import pack_state_dict


def test_bitpack_metadata(tmp_path: Path):
    state = {
        "layer.weight": torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.5, 0.0]], dtype=torch.float32),
        "layer.bias": torch.tensor([0.0, 1.0], dtype=torch.float32),
    }
    out_bin = tmp_path / "packed.bin"
    out_meta = tmp_path / "packed.json"

    pack_state_dict(state, out_bin, out_meta)
    assert out_bin.exists()
    assert out_meta.exists()

    payload = json.loads(out_meta.read_text(encoding="utf-8"))
    assert payload["format"] == "mertformer_bitpack_v1"
    assert payload["entry_count"] > 0

from __future__ import annotations

from pathlib import Path

import torch

from orchestrator import distillation_manager as dm
from scripts import precompute_logits_topk as phase0_topk


def test_reconstruct_dense_from_topk_scatter_values() -> None:
    item = {
        "indices": torch.tensor([[1, 3], [0, 2]], dtype=torch.int32),
        "values": torch.tensor([[5.0, 1.5], [-2.0, 3.0]], dtype=torch.bfloat16),
    }

    dense = dm._reconstruct_dense_from_topk(item, vocab_size=5)

    assert dense.shape == (2, 5)
    assert dense[0, 1].item() == 5.0
    assert dense[0, 3].item() == 1.5
    assert dense[1, 0].item() == -2.0
    assert dense[1, 2].item() == 3.0
    assert dense[0, 0].item() == -1e4
    assert dense[1, 4].item() == -1e4


def test_precomputed_loader_supports_topk_and_dense_payloads(tmp_path: Path, monkeypatch) -> None:
    stage_name = "stage1"
    monkeypatch.setattr(dm, "_resolve_vocab_size", lambda default=128256: 4)

    payload = {
        "format": "topk_sparse_v1",
        "top_k": 2,
        "vocab_size": 4,
        "logits": [
            {
                "indices": torch.tensor([[1, 3]], dtype=torch.int32),
                "values": torch.tensor([[2.0, -1.0]], dtype=torch.bfloat16),
            },
            torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32),
        ],
    }
    torch.save(payload, tmp_path / f"{stage_name}_train_part_0.pt")

    items = list(dm.PrecomputedLogitsIterable(tmp_path, stage_name, subset="train"))

    assert len(items) == 2
    assert items[0].shape == (1, 4)
    assert items[0][0, 1].item() == 2.0
    assert items[0][0, 3].item() == -1.0
    assert items[0][0, 0].item() == -1e4
    assert torch.equal(items[1], torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32))


def test_phase0_stage_paths_match_repo_layout() -> None:
    assert phase0_topk.STAGE_FILES[4].as_posix().endswith("datasets/stage4_soul/stage4_data.jsonl")
    assert phase0_topk.STAGE_FILES[5].as_posix().endswith("datasets/stage5_tools/stage5_data.jsonl")

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from orchestrator import distillation_manager as dm
from scripts import precompute_logits_topk as phase0_topk
from train import train as train_mod


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
    assert items[0]["format"] == "topk_sparse_v1"
    assert items[0]["vocab_size"] == 4
    assert items[0]["top_k"] == 2
    assert torch.equal(items[0]["indices"], torch.tensor([[1, 3]], dtype=torch.long))
    assert torch.equal(items[0]["values"], torch.tensor([[2.0, -1.0]], dtype=torch.float32))
    assert torch.equal(items[1], torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32))


def test_reconstruct_dense_from_topk_large_payload_requires_override(monkeypatch) -> None:
    monkeypatch.delenv("TITAN_ALLOW_DENSE_TOPK_RECONSTRUCT", raising=False)
    item = {
        "indices": torch.tensor([[1, 3], [0, 2]], dtype=torch.int64),
        "values": torch.tensor([[5.0, 1.5], [-2.0, 3.0]], dtype=torch.float32),
    }

    with pytest.raises(RuntimeError, match="Refusing dense Top-K logits reconstruction"):
        dm._reconstruct_dense_from_topk(item, vocab_size=10_000_000)


def test_sparse_topk_kd_matches_dense_topk_on_toy_vocab() -> None:
    student = torch.tensor(
        [[[0.2, 3.0, -1.0, 1.5], [2.0, -0.5, 0.1, 4.0]]],
        dtype=torch.float32,
    )
    sparse = {
        "format": "topk_sparse_v1",
        "indices": torch.tensor([[[1, 3], [0, 3]]], dtype=torch.long),
        "values": torch.tensor([[[5.0, 2.0], [1.0, 6.0]]], dtype=torch.float32),
        "vocab_size": 4,
        "top_k": 2,
    }
    sparse_loss = train_mod.kd_loss_safe(student, sparse, temp=2.0)
    student_topk = torch.gather(student, dim=-1, index=sparse["indices"])
    dense_topk_loss = F.kl_div(
        F.log_softmax(student_topk / 2.0, dim=-1),
        F.softmax(sparse["values"] / 2.0, dim=-1),
        reduction="none",
    ).sum(dim=-1).mul(4.0).mean()

    assert torch.allclose(sparse_loss, dense_topk_loss, atol=1e-6)


def test_collate_stacks_sparse_topk_teacher_payload() -> None:
    x = torch.tensor([1, 2, 3])
    payload_a = {
        "format": "topk_sparse_v1",
        "indices": torch.tensor([[1, 2], [0, 3], [2, 3]], dtype=torch.long),
        "values": torch.ones(3, 2),
        "vocab_size": 4,
        "top_k": 2,
    }
    payload_b = {
        "format": "topk_sparse_v1",
        "indices": torch.tensor([[0, 1], [1, 2], [0, 3]], dtype=torch.long),
        "values": torch.zeros(3, 2),
        "vocab_size": 4,
        "top_k": 2,
    }

    batch_x, batch_y, teacher = train_mod.collate_fn([(x, x, payload_a), (x, x, payload_b)])

    assert batch_x.shape == (2, 3)
    assert torch.equal(batch_y, batch_x)
    assert teacher["format"] == "topk_sparse_v1"
    assert teacher["indices"].shape == (2, 3, 2)
    assert teacher["values"].shape == (2, 3, 2)


def test_collate_rejects_mixed_dense_and_sparse_teacher_payloads() -> None:
    x = torch.tensor([1, 2])
    sparse = {
        "format": "topk_sparse_v1",
        "indices": torch.tensor([[1, 2], [0, 3]], dtype=torch.long),
        "values": torch.ones(2, 2),
        "vocab_size": 4,
        "top_k": 2,
    }

    with pytest.raises(ValueError, match="Cannot mix dense and sparse"):
        train_mod.collate_fn([(x, x, sparse), (x, x, torch.zeros(2, 4))])


def test_sparse_loader_keeps_canonical_payload_small(tmp_path: Path, monkeypatch) -> None:
    stage_name = "stage1"
    seq_len = 8
    top_k = 2
    vocab_size = 128_256
    monkeypatch.setattr(dm, "_resolve_vocab_size", lambda default=128256: vocab_size)
    payload = {
        "format": "topk_sparse_v1",
        "top_k": top_k,
        "vocab_size": vocab_size,
        "logits": [
            {
                "indices": torch.arange(seq_len * top_k, dtype=torch.long).reshape(seq_len, top_k),
                "values": torch.randn(seq_len, top_k),
            }
        ],
    }
    torch.save(payload, tmp_path / f"{stage_name}_train_part_0.pt")

    item = next(iter(dm.PrecomputedLogitsIterable(tmp_path, stage_name, subset="train")))

    assert item["values"].numel() == seq_len * top_k
    assert item["indices"].numel() == seq_len * top_k
    assert item["vocab_size"] == vocab_size


def test_phase0_stage_paths_match_repo_layout() -> None:
    assert phase0_topk.STAGE_FILES[4].as_posix().endswith("datasets/stage4_soul/stage4_data.jsonl")
    assert phase0_topk.STAGE_FILES[5].as_posix().endswith("datasets/stage5_tools/stage5_data.jsonl")


def test_dense_precompute_requires_explicit_debug_override(tmp_path: Path, monkeypatch) -> None:
    class _Cfg:
        precomputed_logits_path = str(tmp_path)
        device = "cpu"
        require_gated_teacher = False

    manager = dm.DistillationManager(_Cfg(), tokenizer=object())
    manager.teacher_model = object()
    monkeypatch.delenv("TITAN_ALLOW_DENSE_PRECOMPUTE", raising=False)

    with pytest.raises(RuntimeError, match="sparse Top-K logits"):
        manager.precompute_logits([], "stage1")


def test_precomputed_curriculum_stage_entry_clamps_fallback_stage() -> None:
    dataset = object.__new__(train_mod.PrecomputedCurriculumDataset)
    dataset.stage_info = [("fallback", Path("datasets/validation.jsonl"))]

    assert dataset._stage_entry(5) == ("fallback", Path("datasets/validation.jsonl"))


def test_resume_state_keys_raw_lines_not_produced_samples(tmp_path: Path) -> None:
    """B2: resume indexes by RAW LINE INDEX (lines_consumed), back-compat reads done_samples."""
    phase0_topk._save_resume_state(tmp_path, "stage1", lines_consumed=42, sequences_emitted=7)
    state = phase0_topk._load_resume_state(tmp_path, "stage1")
    assert state["lines_consumed"] == 42
    assert state["sequences_emitted"] == 7
    assert phase0_topk._load_done_samples(tmp_path, "stage1") == 42

    import json as _json
    sp = phase0_topk._state_path(tmp_path, "stage2")
    sp.write_text(_json.dumps({"done_samples": 11}), encoding="utf-8")
    assert phase0_topk._load_resume_state(tmp_path, "stage2")["lines_consumed"] == 11


def test_packed_payload_preserves_identity() -> None:
    """B2: the precomputed loader surfaces per-sequence identity + true_len so the
    train side can hard-assert alignment."""
    item = {
        "format": "topk_packed_v1",
        "indices": torch.tensor([[1, 3], [0, 2]], dtype=torch.int32),
        "values": torch.tensor([[5.0, 1.5], [-2.0, 3.0]], dtype=torch.bfloat16),
        "true_len": 2,
        "identity": {"len": 2, "hash": "abc123"},
    }
    payload = dm._as_sparse_topk_payload(item, vocab_size=5)
    assert payload["format"] == "topk_packed_v1"
    assert payload["identity"] == {"len": 2, "hash": "abc123"}
    assert payload["true_len"] == 2
    assert train_mod._is_sparse_topk_payload(payload)

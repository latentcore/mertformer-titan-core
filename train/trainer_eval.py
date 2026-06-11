"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30 V2) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)

Module: Knowledge-distillation losses (dense + sparse Top-K) and
benchmark-metric reading used by validation / saturation gates.
Split out of train/train.py as pure code motion; train/train.py re-exports
every symbol so all historical imports keep working unchanged.
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from train.trainer_data import _is_sparse_topk_payload


# -----------------------------------------------------------------------------
# KD LOSS (Knowledge Distillation) + METRIC READING
# -----------------------------------------------------------------------------


def read_metric_from_json(path: Path, key: str) -> Optional[float]:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get(key)
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _kd_loss_sparse_topk(
    student_logits: torch.Tensor,
    teacher_payload: dict,
    temp: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    indices = teacher_payload["indices"].to(device=student_logits.device, dtype=torch.long)
    values = teacher_payload["values"].to(device=student_logits.device, dtype=torch.float32)
    if indices.shape != values.shape or indices.dim() not in (2, 3):
        raise ValueError(
            "Sparse Top-K teacher payload must use matching [seq, top_k] or [batch, seq, top_k] tensors."
        )
    if tuple(indices.shape[:-1]) != tuple(student_logits.shape[:-1]):
        raise ValueError(
            f"Sparse KD shape mismatch: student={tuple(student_logits.shape)}, "
            f"teacher_indices={tuple(indices.shape)}"
        )
    if indices.numel() and int(indices.max().item()) >= int(student_logits.size(-1)):
        raise ValueError(
            "Sparse teacher indices exceed student vocab dimension; tokenizer/vocab sync is required."
        )

    T = float(temp)
    student_topk = torch.gather(student_logits.float(), dim=-1, index=indices)
    token_kl = F.kl_div(
        F.log_softmax(student_topk / T, dim=-1),
        F.softmax(values / T, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (T * T)
    if mask is not None:
        mask = mask.to(device=token_kl.device, dtype=torch.bool)
        if mask.shape != token_kl.shape:
            raise ValueError(f"KD mask shape mismatch: expected {token_kl.shape}, got {mask.shape}")
        if not bool(mask.any().item()):
            return token_kl.new_zeros(())
        token_kl = token_kl.masked_select(mask)
    return token_kl.mean()


def kd_loss_safe(
    student_logits: torch.Tensor,
    teacher_logits: Any,
    temp: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _is_sparse_topk_payload(teacher_logits):
        return _kd_loss_sparse_topk(student_logits, teacher_logits, temp, mask=mask)

    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    s = student_logits[..., :min_vocab].float()
    t = teacher_logits[..., :min_vocab].float().to(s.device)
    T = float(temp)
    token_kl = F.kl_div(
        F.log_softmax(s / T, dim=-1),
        F.softmax(t / T, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (T * T)
    if mask is not None:
        mask = mask.to(device=token_kl.device, dtype=torch.bool)
        if mask.shape != token_kl.shape:
            raise ValueError(f"KD mask shape mismatch: expected {token_kl.shape}, got {mask.shape}")
        if not bool(mask.any().item()):
            return token_kl.new_zeros(())
        token_kl = token_kl.masked_select(mask)
    return token_kl.mean()

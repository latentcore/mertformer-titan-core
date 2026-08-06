"""Holdout and locked-test evaluation on the packed splits.

Reports raw and legality-masked policy accuracy **separately**. That separation
matters: masked accuracy says "given that we hand the model the legal move set,
how often does it pick Stockfish's move", while raw top-1 legality says "does
the model know the rules on its own". Collapsing them into one headline number
would overstate what the network learned.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..board import VOCAB_SIZE, bin_centers
from ..config import ModelConfig
from ..data.dataset import (
    PackedChessDataset,
    build_hl_gauss_table,
    collate_packed,
    expand_batch_on_device,
    level_to_win_prob,
    make_dataloader,
)
from ..model import ChessFormer


@torch.no_grad()
def evaluate_split(
    model: ChessFormer,
    model_cfg: ModelConfig,
    dataset: PackedChessDataset,
    device: torch.device,
    *,
    batch_size: int = 256,
    max_batches: int = 0,
    num_workers: int = 0,
    use_amp: bool = True,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    started = time.perf_counter()

    hl_gauss = build_hl_gauss_table(model_cfg.num_value_bins).to(device)
    centers = torch.from_numpy(bin_centers(model_cfg.num_value_bins)).to(device)
    loader = make_dataloader(
        dataset, batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=False, drop_last=False,
    )

    n = 0
    masked_top1 = masked_top5 = raw_legal = 0.0
    value_abs_err = value_sq_err = 0.0
    wdl_correct = 0.0
    policy_nll = 0.0
    legal_counts: List[int] = []

    try:
        for i, raw in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if should_stop is not None and should_stop():
                break
            batch = expand_batch_on_device(raw, device, hl_gauss, VOCAB_SIZE)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp and device.type == "cuda"):
                out = model(batch["piece_ids"], batch["meta_ids"])

            logits = out["policy_logits"].float()
            legal = batch["legal_mask"]
            masked = logits.masked_fill(~legal, float("-inf"))
            target = batch["policy_target"].argmax(dim=-1)
            bsz = int(logits.shape[0])

            masked_top1 += float((masked.argmax(dim=-1) == target).sum())
            k = min(5, masked.size(-1))
            top5 = masked.topk(k, dim=-1).indices
            masked_top5 += float((top5 == target.unsqueeze(-1)).any(dim=-1).sum())

            raw_top1 = logits.argmax(dim=-1)
            raw_legal += float(legal.gather(1, raw_top1.unsqueeze(-1)).squeeze(-1).sum())

            log_probs = F.log_softmax(masked, dim=-1)
            log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
            policy_nll += float(-(batch["policy_target"] * log_probs).sum())

            pred_wp = (F.softmax(out["value_logits"].float(), dim=-1) * centers).sum(dim=-1)
            true_wp = level_to_win_prob(batch["value_level"])
            value_abs_err += float((pred_wp - true_wp).abs().sum())
            value_sq_err += float(((pred_wp - true_wp) ** 2).sum())

            if "wdl_logits" in out:
                wdl_correct += float((out["wdl_logits"].argmax(dim=-1) == batch["wdl_target"]).sum())

            legal_counts.append(int(legal.sum()))
            n += bsz
            if progress is not None and (i + 1) % 20 == 0:
                progress({"stage": "holdout", "split": dataset.split, "positions": n})
    finally:
        model.train(was_training)

    if n == 0:
        return {"status": "not_run", "reason": "no batches evaluated", "split": dataset.split}

    return {
        "status": "completed",
        "split": dataset.split,
        "positions_evaluated": n,
        "masked_policy_top1": round(masked_top1 / n, 6),
        "masked_policy_top5": round(masked_top5 / n, 6),
        "raw_top1_is_legal": round(raw_legal / n, 6),
        "policy_cross_entropy": round(policy_nll / n, 6),
        "value_mae_winprob": round(value_abs_err / n, 6),
        "value_rmse_winprob": round(float(np.sqrt(value_sq_err / n)), 6),
        "wdl_accuracy": round(wdl_correct / n, 6) if wdl_correct else None,
        "mean_legal_moves": round(float(np.sum(legal_counts) / n), 3),
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "metric_notes": {
            "masked_policy_top1": "top-1 agreement with the Stockfish best move, given the legal move mask",
            "raw_top1_is_legal": "fraction of positions where the UNMASKED argmax is a legal move",
            "value_mae_winprob": "mean absolute error against the Stockfish-derived win probability",
        },
    }


def evaluate_all_splits(
    model: ChessFormer,
    model_cfg: ModelConfig,
    data_root,
    device: torch.device,
    *,
    batch_size: int = 256,
    max_batches: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    from pathlib import Path

    out: Dict[str, Any] = {"schema": "chessformer_holdout_report_v1"}
    for split in ("val", "test"):
        try:
            dataset = PackedChessDataset(Path(data_root), split)
        except FileNotFoundError:
            out[split] = {"status": "not_run", "reason": f"no {split} split on disk"}
            continue
        out[split] = evaluate_split(
            model, model_cfg, dataset, device,
            batch_size=batch_size, max_batches=max_batches, **kwargs,
        )
    return out

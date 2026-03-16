#!/usr/bin/env python3
"""CfC + MoE tolerance check.

Runs two tiny training passes and compares loss deltas.
Requirement: max relative loss diff <= threshold (default 1%).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

# Ensure repo root is on sys.path when run from automation scripts.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import cfg
from model.transformers import MertFormer


def _patch_cfg(device: str) -> Dict[str, object]:
    keys = [
        "device",
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "vocab_size",
        "use_moe",
        "num_experts",
        "num_experts_per_tok",
        "active_experts",
        "use_liquid",
        "liquid_layers_idx",
        "use_qinn",
        "use_gradient_checkpointing",
        "liquid_fast_path",
        "moe_dispatch_mode",
    ]
    orig = {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}

    cfg.device = device
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_layers = 2
    cfg.num_heads = 4
    cfg.num_kv_heads = 2
    cfg.head_dim = 32
    cfg.vocab_size = 512
    cfg.use_moe = True
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.active_experts = 2
    cfg.use_liquid = True
    cfg.liquid_layers_idx = [0]
    cfg.use_qinn = False
    cfg.use_gradient_checkpointing = False
    return orig


def _restore_cfg(orig: Dict[str, object]) -> None:
    for k, v in orig.items():
        setattr(cfg, k, v)


def _run_variant(
    inputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    *,
    device: str,
    liquid_fast_path: bool,
    moe_dispatch_mode: str,
) -> List[float]:
    cfg.liquid_fast_path = liquid_fast_path
    cfg.moe_dispatch_mode = moe_dispatch_mode

    torch.manual_seed(1234)
    model = MertFormer().to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    losses: List[float] = []
    for step in range(len(inputs)):
        input_ids = inputs[step].to(device)
        target_ids = targets[step].to(device)

        logits, aux_loss, _ = model(input_ids)
        loss = ce(logits.reshape(-1, cfg.vocab_size), target_ids.reshape(-1))
        loss = loss + aux_loss.float()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
    return losses


def _relative_diffs(a: List[float], b: List[float]) -> List[float]:
    diffs = []
    for x, y in zip(a, b):
        denom = max(abs(x), 1e-8)
        diffs.append(abs(x - y) / denom)
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--out", type=str, default="reports/cfc_moe_tolerance_report.json")
    args = parser.parse_args()

    device = "cpu"
    orig = _patch_cfg(device)

    torch.manual_seed(2026)
    inputs = [
        torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
        for _ in range(args.steps)
    ]
    targets = [
        torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
        for _ in range(args.steps)
    ]

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": args.steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "device": device,
        "moe": {},
        "cfc": {},
    }

    try:
        # MoE dispatch tolerance: sequential vs parallel
        base_losses = _run_variant(
            inputs,
            targets,
            device=device,
            liquid_fast_path=False,
            moe_dispatch_mode="sequential",
        )
        par_losses = _run_variant(
            inputs,
            targets,
            device=device,
            liquid_fast_path=False,
            moe_dispatch_mode="parallel",
        )
        moe_diffs = _relative_diffs(base_losses, par_losses)
        report["moe"] = {
            "baseline": "sequential",
            "candidate": "parallel",
            "losses_baseline": base_losses,
            "losses_candidate": par_losses,
            "diffs": moe_diffs,
            "max_diff": max(moe_diffs) if moe_diffs else 0.0,
        }
        report["moe"]["ok"] = report["moe"]["max_diff"] <= args.threshold

        # CfC tolerance: fast path vs standard
        cfc_base = _run_variant(
            inputs,
            targets,
            device=device,
            liquid_fast_path=False,
            moe_dispatch_mode="parallel",
        )
        cfc_fast = _run_variant(
            inputs,
            targets,
            device=device,
            liquid_fast_path=True,
            moe_dispatch_mode="parallel",
        )
        cfc_diffs = _relative_diffs(cfc_base, cfc_fast)
        report["cfc"] = {
            "baseline": "liquid_fast_path=off",
            "candidate": "liquid_fast_path=on",
            "losses_baseline": cfc_base,
            "losses_candidate": cfc_fast,
            "diffs": cfc_diffs,
            "max_diff": max(cfc_diffs) if cfc_diffs else 0.0,
        }
        report["cfc"]["ok"] = report["cfc"]["max_diff"] <= args.threshold

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["ok"] = False
        _restore_cfg(orig)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 1

    report["ok"] = bool(report["moe"].get("ok")) and bool(report["cfc"].get("ok"))

    _restore_cfg(orig)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if report["ok"]:
        print(f"[tolerance] PASS (max_diff_moe={report['moe']['max_diff']:.6f} max_diff_cfc={report['cfc']['max_diff']:.6f})")
        return 0
    print(f"[tolerance] FAIL (max_diff_moe={report['moe']['max_diff']:.6f} max_diff_cfc={report['cfc']['max_diff']:.6f})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

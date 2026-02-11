"""Tiny offline training smoke test.

Runs a short forward/backward/optimizer loop on CPU/MPS and optionally writes a
checkpoint. This is not a benchmark; it is a "can we train at all" sanity test.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg
from model.transformers import MertFormer


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--out-dir", default="checkpoints/smoke")
    parser.add_argument("--cleanup", action="store_true", help="Delete checkpoint after run.")
    args = parser.parse_args()

    device = _pick_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    # Backup global cfg values we patch.
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
    ]
    orig = {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}

    # Tiny config (fast, deterministic-ish).
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

    torch.manual_seed(42)

    start = time.time()
    try:
        model = MertFormer().to(device)
        model.train()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()

        for step in range(args.steps):
            input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
            targets = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)

            logits, aux_loss, _ = model(input_ids)
            loss = ce(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            loss = loss + aux_loss.float()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
                print(f"[smoke] step={step+1}/{args.steps} loss={loss.item():.4f} aux={aux_loss.item():.4f}")

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"smoke_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "meta": {
                    "steps": args.steps,
                    "device": device,
                    "python": os.sys.version.split()[0],
                    "torch": torch.__version__,
                },
                "cfg": {
                    "hidden_size": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "num_kv_heads": cfg.num_kv_heads,
                    "head_dim": cfg.head_dim,
                    "vocab_size": cfg.vocab_size,
                    "use_moe": cfg.use_moe,
                    "use_liquid": cfg.use_liquid,
                },
            },
            ckpt_path,
        )
        print(f"[smoke] checkpoint={ckpt_path}")

        if args.cleanup:
            try:
                ckpt_path.unlink()
                print("[smoke] cleaned up checkpoint")
            except Exception:
                pass

    finally:
        # Restore cfg.
        for k, v in orig.items():
            setattr(cfg, k, v)

    elapsed = time.time() - start
    print(f"[smoke] OK elapsed_sec={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

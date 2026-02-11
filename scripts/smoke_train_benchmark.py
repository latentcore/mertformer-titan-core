"""Smoke benchmark for README metrics (tiny, deterministic-ish, non-claim).

This script is **not** a model-quality benchmark. It exists to produce a stable
"proof of life" metrics table for documentation:
- liquid vs no-liquid (tiny config)
- short training loop on synthetic tokens

Output:
  reports/benchmarks/smoke_train_metrics.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg
from model.transformers import MertFormer


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _avg_tau_bias(model: torch.nn.Module) -> float | None:
    vals = []
    for name, p in model.named_parameters():
        if "tau_bias" in name and p is not None:
            vals.append(F.softplus(p.detach().float()).mean().cpu().item())
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _run_variant(*, use_liquid: bool, device: str, steps: int, seq_len: int, batch_size: int) -> dict:
    # Backup cfg keys we patch.
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

    try:
        # Tiny config (fast).
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
        cfg.use_liquid = bool(use_liquid)
        cfg.liquid_layers_idx = [0] if use_liquid else []
        cfg.use_qinn = False
        cfg.use_gradient_checkpointing = False

        # Deterministic weight init.
        torch.manual_seed(42)

        model = MertFormer().to(device)
        model.train()

        # Deterministic synthetic data (same across variants).
        gen = torch.Generator(device="cpu").manual_seed(123)
        inputs = [torch.randint(0, cfg.vocab_size, (batch_size, seq_len), generator=gen) for _ in range(steps)]
        targets = [torch.randint(0, cfg.vocab_size, (batch_size, seq_len), generator=gen) for _ in range(steps)]

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()

        losses = []
        aux_losses = []

        for step in range(steps):
            input_ids = inputs[step].to(device)
            tgt = targets[step].to(device)

            logits, aux_loss, _ = model(input_ids)
            loss = ce(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
            loss = loss + aux_loss.float()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
            aux_losses.append(float(aux_loss.detach().cpu().item()))

        return {
            "use_liquid": bool(use_liquid),
            "steps": steps,
            "final_loss": losses[-1] if losses else None,
            "avg_loss": float(sum(losses) / len(losses)) if losses else None,
            "final_aux_loss": aux_losses[-1] if aux_losses else None,
            "avg_tau": _avg_tau_bias(model),
        }
    finally:
        for k, v in orig.items():
            setattr(cfg, k, v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "benchmarks" / "smoke_train_metrics.json"))
    args = parser.parse_args()

    device = _pick_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    started = time.time()
    generated_at = _utc_now_iso()

    liquid = _run_variant(use_liquid=True, device=device, steps=args.steps, seq_len=args.seq_len, batch_size=args.batch_size)
    no_liquid = _run_variant(use_liquid=False, device=device, steps=args.steps, seq_len=args.seq_len, batch_size=args.batch_size)

    result = {
        "generated_at_utc": generated_at,
        "note": "Documentation-only smoke benchmark on synthetic tokens. Not a model-quality claim.",
        "env": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "device": device,
        },
        "liquid": liquid,
        "no_liquid": no_liquid,
        "elapsed_sec": round(time.time() - started, 3),
    }

    blob = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["forensic_sha256"] = _sha256_hexdigest(blob)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


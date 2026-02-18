"""Kaggle-ready training stability comparison: MertFormer tiny vs vanilla transformer.

Outputs:
- reports/benchmarks/kaggle_compare_build30.json
- reports/benchmarks/kaggle_compare_build30.csv
- reports/benchmarks/kaggle_compare_build30.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg
from model.transformers import MertFormer


class VanillaTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.tok = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq = input_ids.shape
        if seq > self.max_seq_len:
            raise ValueError(f"sequence {seq} exceeds max_seq_len {self.max_seq_len}")
        x = self.tok(input_ids) + self.pos[:, :seq, :]
        mask = torch.triu(torch.ones(seq, seq, device=input_ids.device, dtype=torch.bool), diagonal=1)
        x = self.blocks(x, mask=mask)
        x = self.norm(x)
        return self.head(x)


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_batch(batch_size: int, seq_len: int, vocab_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += float(torch.sum(g * g).item())
    return math.sqrt(max(total, 0.0))


def _train_variant(
    *,
    name: str,
    model: nn.Module,
    device: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    lr: float,
    aux_loss_enabled: bool,
) -> dict:
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    losses: list[float] = []
    grad_norms: list[float] = []
    nan_steps = 0
    total_tokens = 0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for step in range(steps):
        x, y = _make_batch(batch_size, seq_len, vocab_size, device)
        opt.zero_grad(set_to_none=True)

        if aux_loss_enabled:
            logits, aux_loss, _ = model(x)
            loss = ce(logits.reshape(-1, vocab_size), y.reshape(-1)) + aux_loss.float()
        else:
            logits = model(x)
            loss = ce(logits.reshape(-1, vocab_size), y.reshape(-1))

        if not torch.isfinite(loss):
            nan_steps += 1
            continue

        loss.backward()
        gn = _grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        losses.append(float(loss.detach().cpu().item()))
        grad_norms.append(gn)
        total_tokens += batch_size * seq_len

        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            print(f"[{name}] step={step+1}/{steps} loss={losses[-1]:.4f} grad_norm={gn:.4f}")

    elapsed = time.time() - t0
    tokens_per_sec = (total_tokens / elapsed) if elapsed > 0 else 0.0
    peak_mem_gb = 0.0
    if device == "cuda":
        peak_mem_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3))

    return {
        "name": name,
        "params": _count_params(model),
        "steps": steps,
        "final_loss": losses[-1] if losses else float("inf"),
        "min_loss": min(losses) if losses else float("inf"),
        "avg_loss": (sum(losses) / len(losses)) if losses else float("inf"),
        "avg_grad_norm": (sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0,
        "max_grad_norm": max(grad_norms) if grad_norms else 0.0,
        "nan_steps": nan_steps,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "avg_step_time_sec": (elapsed / max(1, len(losses))),
        "peak_mem_gb": peak_mem_gb,
    }


def _run_mertformer_variant(
    device: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    hidden: int,
    layers: int,
    heads: int,
    lr: float,
    use_moe: bool,
    use_liquid: bool,
) -> dict:
    keys = [
        "device",
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_hidden_layers",
        "num_heads",
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
        "vocab_size",
        "max_seq_len",
        "use_moe",
        "num_experts",
        "num_experts_per_tok",
        "active_experts",
        "moe_every_n_layers",
        "use_liquid",
        "liquid_layers_idx",
        "liquid_every_n_layers",
        "use_qinn",
        "use_gradient_checkpointing",
        "attention_dropout",
        "dropout",
    ]
    backup = {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}

    try:
        cfg.device = device
        cfg.hidden_size = hidden
        cfg.intermediate_size = hidden * 4
        cfg.num_layers = layers
        cfg.num_hidden_layers = layers
        cfg.num_heads = heads
        cfg.num_attention_heads = heads
        cfg.num_kv_heads = max(1, heads // 2)
        cfg.head_dim = hidden // heads
        cfg.vocab_size = vocab_size
        cfg.max_seq_len = seq_len
        cfg.use_moe = bool(use_moe)
        cfg.num_experts = 4
        cfg.num_experts_per_tok = 2
        cfg.active_experts = 2
        cfg.moe_every_n_layers = 2
        cfg.use_liquid = bool(use_liquid)
        cfg.liquid_layers_idx = [i for i in range(layers) if i % 3 == 0]
        cfg.liquid_every_n_layers = 0
        cfg.use_qinn = False
        cfg.use_gradient_checkpointing = False
        cfg.attention_dropout = 0.0
        cfg.dropout = 0.0

        torch.manual_seed(42)
        model = MertFormer()
        return _train_variant(
            name="mertformer",
            model=model,
            device=device,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            lr=lr,
            aux_loss_enabled=True,
        )
    finally:
        for k, v in backup.items():
            setattr(cfg, k, v)


def _run_vanilla_variant(device: str, steps: int, batch_size: int, seq_len: int, vocab_size: int, hidden: int, layers: int, heads: int, lr: float) -> dict:
    torch.manual_seed(42)
    model = VanillaTransformerLM(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        max_seq_len=seq_len,
        dropout=0.0,
    )
    return _train_variant(
        name="vanilla",
        model=model,
        device=device,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        lr=lr,
        aux_loss_enabled=False,
    )


def _write_reports(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "kaggle_compare_build30.json"
    csv_path = out_dir / "kaggle_compare_build30.csv"
    md_path = out_dir / "kaggle_compare_build30.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fields = [
        "name",
        "params",
        "steps",
        "final_loss",
        "min_loss",
        "avg_loss",
        "avg_grad_norm",
        "max_grad_norm",
        "nan_steps",
        "elapsed_sec",
        "tokens_per_sec",
        "avg_step_time_sec",
        "peak_mem_gb",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(payload["mertformer"])
        writer.writerow(payload["vanilla"])

    md = [
        "# Kaggle Build30 Compare",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- device: {payload['device']}",
        f"- quick_mode: {payload['quick_mode']}",
        "",
        "| Variant | Params | Final Loss | Avg Grad Norm | NaN Steps | Tok/s | Avg Step (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| MertFormer | {payload['mertformer']['params']} | {payload['mertformer']['final_loss']:.4f} | {payload['mertformer']['avg_grad_norm']:.4f} | {payload['mertformer']['nan_steps']} | {payload['mertformer']['tokens_per_sec']:.2f} | {payload['mertformer']['avg_step_time_sec']:.4f} |",
        f"| Vanilla | {payload['vanilla']['params']} | {payload['vanilla']['final_loss']:.4f} | {payload['vanilla']['avg_grad_norm']:.4f} | {payload['vanilla']['nan_steps']} | {payload['vanilla']['tokens_per_sec']:.2f} | {payload['vanilla']['avg_step_time_sec']:.4f} |",
        "",
        f"loss_delta(vanilla - mertformer): {payload['loss_delta']:.4f}",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "reports" / "benchmarks"))
    parser.add_argument("--quick", action="store_true", help="2-3 hour proof profile")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=None, help="override hidden size for both variants")
    parser.add_argument("--layers", type=int, default=None, help="override layer count for both variants")
    parser.add_argument("--heads", type=int, default=None, help="override attention heads for both variants")
    parser.add_argument("--mert-hidden", type=int, default=256, help="MertFormer hidden size (default ~30M params)")
    parser.add_argument("--mert-layers", type=int, default=8, help="MertFormer layer count")
    parser.add_argument("--mert-heads", type=int, default=8, help="MertFormer attention heads")
    parser.add_argument("--vanilla-hidden", type=int, default=416, help="Vanilla hidden size (default ~30M params)")
    parser.add_argument("--vanilla-layers", type=int, default=8, help="Vanilla layer count")
    parser.add_argument("--vanilla-heads", type=int, default=8, help="Vanilla attention heads")
    parser.add_argument("--mert-use-moe", type=int, choices=[0, 1], default=1, help="Enable MoE in MertFormer variant")
    parser.add_argument("--mert-use-liquid", type=int, choices=[0, 1], default=1, help="Enable Liquid in MertFormer variant")
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = _pick_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    if args.quick:
        steps = args.steps or 40
        batch_size = args.batch_size or 2
        seq_len = args.seq_len or 64
    else:
        steps = args.steps or 1200
        batch_size = args.batch_size or 8
        seq_len = args.seq_len or 256

    mert_hidden = args.hidden if args.hidden is not None else args.mert_hidden
    mert_layers = args.layers if args.layers is not None else args.mert_layers
    mert_heads = args.heads if args.heads is not None else args.mert_heads
    vanilla_hidden = args.hidden if args.hidden is not None else args.vanilla_hidden
    vanilla_layers = args.layers if args.layers is not None else args.vanilla_layers
    vanilla_heads = args.heads if args.heads is not None else args.vanilla_heads

    mertformer_result = _run_mertformer_variant(
        device=device,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=args.vocab_size,
        hidden=mert_hidden,
        layers=mert_layers,
        heads=mert_heads,
        lr=args.lr,
        use_moe=bool(args.mert_use_moe),
        use_liquid=bool(args.mert_use_liquid),
    )

    vanilla_result = _run_vanilla_variant(
        device=device,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=args.vocab_size,
        hidden=vanilla_hidden,
        layers=vanilla_layers,
        heads=vanilla_heads,
        lr=args.lr,
    )

    target_low = 25_000_000
    target_high = 35_000_000
    mert_band = target_low <= int(mertformer_result["params"]) <= target_high
    vanilla_band = target_low <= int(vanilla_result["params"]) <= target_high

    payload = {
        "schema": "kaggle_compare_build30_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "quick_mode": bool(args.quick),
        "target_param_band": {"low": target_low, "high": target_high},
        "variant_config": {
            "mertformer": {
                "hidden": mert_hidden,
                "layers": mert_layers,
                "heads": mert_heads,
                "use_moe": bool(args.mert_use_moe),
                "use_liquid": bool(args.mert_use_liquid),
            },
            "vanilla": {
                "hidden": vanilla_hidden,
                "layers": vanilla_layers,
                "heads": vanilla_heads,
            },
        },
        "mertformer": mertformer_result,
        "vanilla": vanilla_result,
        "loss_delta": float(vanilla_result["final_loss"] - mertformer_result["final_loss"]),
        "band_check": {
            "mertformer_in_band": mert_band,
            "vanilla_in_band": vanilla_band,
            "both_in_band": bool(mert_band and vanilla_band),
        },
    }

    _write_reports(Path(args.out_dir), payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
$0 LiquidRouter ablation pilot — train a SMALL (~80M) MertFormer twice (Liquid ON vs OFF)
with pure next-token CE (NO 70B teacher, NO KD, NO HF download) and compare loss curves.

Designed to run for free on a Kaggle T4/P100 (or locally on CPU/MPS for a smoke). It builds
MertFormer() directly with a patched config (the train_smoke.py pattern), bypassing the
teacher/KD machinery entirely, so it costs nothing and needs only local data + tokenizer.

Data: real text from datasets/offline_demo/train.jsonl tokenized with the local TR tokenizer
(data/tokenizer/tr); falls back to synthetic random tokens if either is unavailable (the
ablation signal still holds — both variants see identical data).

Usage:
  # Mac/CPU smoke (proves it runs):
  python scripts/run_liquid_ablation.py --steps 3
  # Kaggle free GPU pilot (real signal):
  python scripts/run_liquid_ablation.py --steps 500 --device cuda

Output: reports/ablations/liquid_ablation_results.json + a printed ON-vs-OFF comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg  # noqa: E402
from model.transformers import MertFormer  # noqa: E402

# ~80-85M with the 128k vocab dominating: embed+lm_head ≈ 2 * 128000 * 256.
PILOT_CFG = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_layers": 8,
    "num_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 64,
    "use_moe": True,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "active_experts": 2,
    "moe_every_n_layers": 3,
    "use_qinn": False,
    "use_gradient_checkpointing": False,
}
PATCH_KEYS = list(PILOT_CFG) + ["device", "vocab_size", "max_seq_len", "use_liquid", "liquid_layers_idx"]


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_real_tokens(seq_len: int) -> tuple[torch.Tensor | None, int, str]:
    """Tokenize datasets/offline_demo/train.jsonl with the local TR tokenizer. Returns
    (flat_token_tensor, vocab_size, source) or (None, 0, reason) to trigger the synthetic
    fallback."""
    data_path = PROJECT_ROOT / "datasets" / "offline_demo" / "train.jsonl"
    tok_path = PROJECT_ROOT / "data" / "tokenizer" / "tr"
    if not data_path.exists() or not (tok_path / "tokenizer.json").exists():
        return None, 0, "real data/tokenizer not found"
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True)
        texts = []
        for line in data_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(str(obj.get("text", obj) if isinstance(obj, dict) else obj))
            except json.JSONDecodeError:
                texts.append(line)
        ids: list[int] = []
        for t in texts:
            ids.extend(tok.encode(t))
        if len(ids) < seq_len + 1:
            ids = (ids * (seq_len * 4 // max(1, len(ids)) + 2))  # repeat tiny corpus
        return torch.tensor(ids, dtype=torch.long), len(tok), f"offline_demo+TR ({len(ids)} tokens)"
    except Exception as exc:  # noqa: BLE001
        return None, 0, f"real-data load failed: {exc}"


def _batch(tokens: torch.Tensor | None, vocab: int, bsz: int, seq: int, device: str, step: int):
    """Return (input_ids, targets) for next-token CE. Real tokens window through the corpus
    deterministically by step; synthetic falls back to random."""
    if tokens is None:
        x = torch.randint(0, vocab, (bsz, seq + 1), device=device)
    else:
        usable = tokens.numel() - (seq + 1)
        rows = []
        for b in range(bsz):
            start = ((step * bsz + b) * (seq + 1)) % max(1, usable)
            rows.append(tokens[start:start + seq + 1])
        x = torch.stack(rows).to(device)
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()


def run_variant(use_liquid: bool, *, steps: int, device: str, bsz: int, seq: int,
                tokens: torch.Tensor | None, vocab: int, lr: float) -> dict:
    variant = "liquid_on" if use_liquid else "liquid_off"
    orig = {k: getattr(cfg, k) for k in PATCH_KEYS if hasattr(cfg, k)}
    try:
        for k, v in PILOT_CFG.items():
            setattr(cfg, k, v)
        cfg.device = device
        cfg.vocab_size = vocab
        cfg.max_seq_len = seq
        cfg.use_liquid = use_liquid
        cfg.liquid_layers_idx = [2, 4, 6] if use_liquid else []

        torch.manual_seed(1234)  # identical init across variants for a fair ablation
        model = MertFormer().to(device)
        model.train()
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()

        curve = []
        t0 = time.time()
        for step in range(steps):
            input_ids, targets = _batch(tokens, vocab, bsz, seq, device, step)
            logits, aux_loss, _ = model(input_ids)
            loss = ce(logits.reshape(-1, vocab), targets.reshape(-1)) + aux_loss.float()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            curve.append(round(float(loss.item()), 5))
            if step == 0 or (step + 1) % max(1, steps // 5) == 0:
                print(f"[{variant}] step={step + 1}/{steps} loss={loss.item():.4f} aux={float(aux_loss):.4f}")
        elapsed = round(time.time() - t0, 2)
    finally:
        for k, v in orig.items():
            setattr(cfg, k, v)

    last = curve[-min(10, len(curve)):]
    return {
        "variant": variant,
        "param_count": int(n_params),
        "param_millions": round(n_params / 1e6, 1),
        "steps": steps,
        "elapsed_sec": elapsed,
        "loss_curve": curve,
        "final_loss": curve[-1],
        "mean_last10": round(sum(last) / len(last), 5),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="LiquidRouter ON-vs-OFF $0 ablation pilot.")
    p.add_argument("--steps", type=int, default=500, help="Steps per variant (use 3 for a Mac smoke).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--synthetic", action="store_true", help="Force synthetic data (skip real corpus).")
    p.add_argument("--out", default="reports/ablations/liquid_ablation_results.json")
    args = p.parse_args()

    device = _pick_device(args.device)
    tokens, vocab, source = (None, 32000, "synthetic (forced)") if args.synthetic else _load_real_tokens(args.seq_len)
    if tokens is None and not args.synthetic:
        vocab, source = 32000, f"synthetic fallback ({source})"
    print(f"[pilot] device={device} data={source} vocab={vocab} seq={args.seq_len} bsz={args.batch_size} steps={args.steps}")

    results = {}
    for use_liquid in (True, False):
        results["liquid_on" if use_liquid else "liquid_off"] = run_variant(
            use_liquid, steps=args.steps, device=device, bsz=args.batch_size,
            seq=args.seq_len, tokens=tokens, vocab=vocab, lr=args.lr,
        )

    on, off = results["liquid_on"], results["liquid_off"]
    delta = round(off["mean_last10"] - on["mean_last10"], 5)
    summary = {
        "data_source": source,
        "vocab_size": vocab,
        "device": device,
        "liquid_on_mean_last10": on["mean_last10"],
        "liquid_off_mean_last10": off["mean_last10"],
        "liquid_advantage_(off_minus_on)": delta,
        "verdict": ("liquid HELPS (lower loss)" if delta > 0 else "liquid does NOT help" if delta < 0 else "tie")
        + " — pilot signal only; not a claim until a larger measured run.",
    }
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "variants": results}, indent=2), encoding="utf-8")

    print("\n=== LiquidRouter ablation (pilot signal) ===")
    print(f"  params: ~{on['param_millions']}M | data: {source}")
    print(f"  liquid ON  mean_last10 = {on['mean_last10']}")
    print(f"  liquid OFF mean_last10 = {off['mean_last10']}")
    print(f"  Δ(off-on) = {delta}  → {summary['verdict']}")
    print(f"  written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

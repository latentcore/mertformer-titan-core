#!/usr/bin/env python3
"""
BitNet (use_bitnet) ablation pilot — train a SMALL (~80M) MertFormer twice
(use_bitnet ON vs OFF) with pure next-token CE (NO 70B teacher, NO KD, NO HF download) and
compare loss curves. This is the pending `ablations/bitlinear_off` slot (see
ablations/results.md — "Beklemede: eğitim donanımı gerekir") finally getting a runnable
script, mirroring the exact methodology already used for the Liquid/CfC ablation
(scripts/run_liquid_ablation.py) so the two are directly comparable in style and rigor.

NOTE: this toggles cfg.use_bitnet — every BitLinear projection in the model (layers/bitlinear.py)
falls back to a standard fp32 nn.Linear when OFF (see layers/bitnet_patch.py). use_liquid is
forced OFF in both arms so this isolates BitNet only, the same "isolate one variable" discipline
the Liquid ablation used (it kept MoE ON in both arms rather than mixing confounds).

Designed to run for free on a Kaggle T4/P100 (or locally on CPU/MPS for a smoke). It builds
MertFormer() directly with a patched config (the train_smoke.py pattern), bypassing the
teacher/KD machinery entirely, so it costs nothing and needs only local data + tokenizer.

Data: real text from datasets/offline_demo/train.jsonl tokenized with the local TR tokenizer
(data/tokenizer/tr); falls back to synthetic random tokens if either is unavailable (the
ablation signal still holds — both variants see identical data).

Usage:
  # Mac/CPU smoke (proves it runs):
  python scripts/run_bitlinear_ablation.py --steps 3
  # Kaggle free GPU pilot (real signal):
  python scripts/run_bitlinear_ablation.py --steps 500 --device cuda

This is a $0 PILOT SIGNAL, not the canonical ablation. If the pilot shows a directional effect
worth trusting, follow the same escalation the Liquid ablation went through: a proper multi-seed
run (ABLATION.md's 12-seed methodology) before writing any "measured" claim into results.md.

Output: reports/ablations/bitlinear_ablation_results.json + a printed ON-vs-OFF comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Windows console codepages (cp1254 etc.) crash on Δ/→ in the summary print — same bug class
# already found/fixed in the Turkce_AI_4060 side-project's train.py. Force UTF-8 stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg  # noqa: E402
from model.transformers import MertFormer  # noqa: E402

# ~80-85M with the 128k vocab dominating: embed+lm_head ≈ 2 * 128000 * 256.
# Identical to run_liquid_ablation.py's PILOT_CFG so the two pilots are apples-to-apples.
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
    "use_liquid": False,  # forced OFF in both arms — isolate BitNet only, no confound
    "use_gradient_checkpointing": False,
}
PATCH_KEYS = list(PILOT_CFG) + ["device", "vocab_size", "max_seq_len", "use_bitnet"]


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_texts() -> tuple[list[str] | None, str]:
    """Real text corpus. Prefers the (gitignored, local) offline_demo, else the TRACKED
    datasets/validation.jsonl that ships with the repo — so a fresh Kaggle clone has real text."""
    for rel in ("datasets/offline_demo/train.jsonl", "datasets/validation.jsonl"):
        p = PROJECT_ROOT / rel
        if not p.exists():
            continue
        texts = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(str(obj.get("text", obj) if isinstance(obj, dict) else obj))
            except json.JSONDecodeError:
                texts.append(line)
        if texts:
            return texts, rel
    return None, "no local text corpus"


def _make_tokenizer():
    """Tokenizer chain for the real corpus: local TR tokenizer (Mac) -> a tiny public BPE
    (gpt2, a free ~1MB download that works on Kaggle with internet on) -> char-level (no
    download, fully offline). Returns (encode_fn, vocab_size, name)."""
    tr = PROJECT_ROOT / "data" / "tokenizer" / "tr"
    if (tr / "tokenizer.json").exists():
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(tr), local_files_only=True)
            return (lambda s: tok.encode(s)), len(tok), "TR-local"
        except Exception:  # noqa: BLE001
            pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")  # free, downloads on Kaggle
        return (lambda s: tok.encode(s)), len(tok), "gpt2"
    except Exception:  # noqa: BLE001 - offline / no network: fall back to char-level
        return (lambda s: [ord(c) % 256 for c in s]), 256, "char-level"


def _load_real_tokens(seq_len: int) -> tuple[torch.Tensor | None, int, str]:
    """Flatten the real corpus into a token stream with the best available tokenizer. Returns
    (tokens, vocab_size, source) or (None, 0, reason) to trigger the synthetic fallback."""
    texts, src = _read_texts()
    if texts is None:
        return None, 0, src
    encode, vocab, tname = _make_tokenizer()
    ids: list[int] = []
    for t in texts:
        ids.extend(int(i) % vocab for i in encode(t))
    if len(ids) < seq_len + 1:
        ids = ids * (seq_len * 4 // max(1, len(ids)) + 2)  # repeat a tiny corpus
    return torch.tensor(ids, dtype=torch.long), vocab, f"{src}+{tname} ({len(ids)} tok)"


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


def run_variant(use_bitnet: bool, *, steps: int, device: str, bsz: int, seq: int,
                tokens: torch.Tensor | None, vocab: int, lr: float) -> dict:
    variant = "bitnet_on" if use_bitnet else "bitnet_off"
    orig = {k: getattr(cfg, k) for k in PATCH_KEYS if hasattr(cfg, k)}
    try:
        for k, v in PILOT_CFG.items():
            setattr(cfg, k, v)
        cfg.device = device
        cfg.vocab_size = vocab
        cfg.max_seq_len = seq
        cfg.use_bitnet = use_bitnet

        torch.manual_seed(1234)  # identical init across variants for a fair ablation
        model = MertFormer().to(device)
        model.train()
        n_params = sum(p.numel() for p in model.parameters())
        # trainable/optimizer-state size differs meaningfully between ternary and fp32 —
        # report it, it's the other half of the BitNet pitch (memory, not just quality).
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
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
        "param_bytes_fp32_view": int(param_bytes),  # NOTE: BitNet is fake-quant in training
        "steps": steps,                              # (straight-through), so this does NOT yet
        "elapsed_sec": elapsed,                       # show the real ternary memory win — see
        "loss_curve": curve,                          # printed caveat below.
        "final_loss": curve[-1],
        "mean_last10": round(sum(last) / len(last), 5),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="use_bitnet (BitLinear ternary weights) ON-vs-OFF $0 ablation pilot.")
    p.add_argument("--steps", type=int, default=500, help="Steps per variant (use 3 for a Mac smoke).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--synthetic", action="store_true", help="Force synthetic data (skip real corpus).")
    p.add_argument("--out", default="reports/ablations/bitlinear_ablation_results.json")
    args = p.parse_args()

    device = _pick_device(args.device)
    tokens, vocab, source = (None, 32000, "synthetic (forced)") if args.synthetic else _load_real_tokens(args.seq_len)
    if tokens is None and not args.synthetic:
        vocab, source = 32000, f"synthetic fallback ({source})"
    print(f"[pilot] device={device} data={source} vocab={vocab} seq={args.seq_len} bsz={args.batch_size} steps={args.steps}")

    results = {}
    for use_bitnet in (True, False):
        results["bitnet_on" if use_bitnet else "bitnet_off"] = run_variant(
            use_bitnet, steps=args.steps, device=device, bsz=args.batch_size,
            seq=args.seq_len, tokens=tokens, vocab=vocab, lr=args.lr,
        )

    on, off = results["bitnet_on"], results["bitnet_off"]
    delta = round(off["mean_last10"] - on["mean_last10"], 5)
    summary = {
        "data_source": source,
        "vocab_size": vocab,
        "device": device,
        "bitnet_on_mean_last10": on["mean_last10"],
        "bitnet_off_mean_last10": off["mean_last10"],
        "bitnet_advantage_(off_minus_on)": delta,
        "verdict": ("bitnet costs quality (higher loss ON)" if delta < 0
                    else "bitnet is quality-neutral or better at this tiny scale" if delta >= 0
                    else "tie")
        + " — pilot signal only; not a claim until a larger/multi-seed measured run.",
        "caveat": ("BitLinear here is straight-through fake-quant (fp32 tensors simulating "
                   "ternary weights) — param_bytes_fp32_view is identical for both arms by "
                   "construction. The real memory/latency win only shows up with the actual "
                   "packed int kernel (mertformer_sdk/kernels/*), which this pilot does not "
                   "exercise. This script measures QUALITY only, not the memory claim."),
    }
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "variants": results}, indent=2), encoding="utf-8")

    print("\n=== BitNet (use_bitnet) ablation (pilot signal) ===")
    print(f"  params: ~{on['param_millions']}M | data: {source}")
    print(f"  bitnet ON  mean_last10 = {on['mean_last10']}")
    print(f"  bitnet OFF mean_last10 = {off['mean_last10']}")
    print(f"  Δ(off-on) = {delta}  → {summary['verdict']}")
    print(f"  written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Overfit gate: verify the pipeline can memorize ~1MB of code data.
Safe mode uses a smaller byte budget for local machines.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import List
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg
from model.transformers import MertFormer


@contextmanager
def patched_cfg():
    original = {
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "num_heads": cfg.num_heads,
        "num_attention_heads": getattr(cfg, "num_attention_heads", cfg.num_heads),
        "num_kv_heads": getattr(cfg, "num_kv_heads", cfg.num_heads),
        "head_dim": cfg.head_dim,
        "num_layers": cfg.num_layers,
        "num_hidden_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "max_seq_len": cfg.max_seq_len,
        "use_moe": cfg.use_moe,
        "use_liquid": cfg.use_liquid,
        "use_qinn": cfg.use_qinn,
        "liquid_layers_idx": cfg.liquid_layers_idx,
        "moe_every_n_layers": cfg.moe_every_n_layers,
        "use_gradient_checkpointing": cfg.use_gradient_checkpointing,
    }

    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_experts = 2
    cfg.num_experts_per_tok = 1
    cfg.active_experts = 1
    cfg.num_heads = 4
    cfg.num_attention_heads = 4
    cfg.num_kv_heads = 2
    cfg.head_dim = 32
    cfg.num_layers = 2
    cfg.num_hidden_layers = 2
    cfg.vocab_size = 256
    cfg.max_seq_len = 64
    cfg.use_moe = False
    cfg.use_liquid = False
    cfg.use_qinn = False
    cfg.liquid_layers_idx = []
    cfg.moe_every_n_layers = 0
    cfg.use_gradient_checkpointing = False

    try:
        yield
    finally:
        for key, value in original.items():
            setattr(cfg, key, value)


def load_text_bytes(path: Path, byte_budget: int) -> bytes:
    data = bytearray()
    with path.open("rb") as f:
        while len(data) < byte_budget:
            chunk = f.read(min(1024 * 1024, byte_budget - len(data)))
            if not chunk:
                break
            data.extend(chunk)
    return bytes(data)


def jsonl_to_text(raw: bytes) -> str:
    text_parts: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8", errors="ignore"))
            if isinstance(obj, dict) and "text" in obj:
                text_parts.append(str(obj["text"]))
            else:
                text_parts.append(line.decode("utf-8", errors="ignore"))
        except Exception:
            text_parts.append(line.decode("utf-8", errors="ignore"))
    return "\n".join(text_parts)


def tokenize_to_ids(text: str, vocab_size: int) -> List[int]:
    data = text.encode("utf-8", errors="ignore")
    return [b % vocab_size for b in data]


def build_sequences(tokens: List[int], seq_len: int) -> List[List[int]]:
    if len(tokens) < seq_len + 1:
        if not tokens:
            return []
        reps = (seq_len + 1) // len(tokens) + 1
        padded = (tokens * reps)[: seq_len + 1]
        return [padded]

    sequences = []
    step = max(1, seq_len // 2)
    for i in range(0, max(0, len(tokens) - seq_len - 1), step):
        seq = tokens[i : i + seq_len + 1]
        if len(seq) == seq_len + 1:
            sequences.append(seq)
    return sequences


def run_overfit(dataset_path: Path, byte_budget: int, max_steps: int, target_loss: float) -> None:
    raw = load_text_bytes(dataset_path, byte_budget)
    text = jsonl_to_text(raw)
    tokens = tokenize_to_ids(text, cfg.vocab_size)
    sequences = build_sequences(tokens, cfg.max_seq_len)

    if not sequences:
        raise RuntimeError("Overfit gate failed: no sequences built")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = MertFormer().to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start_loss = None
    final_loss = None

    for step in range(max_steps):
        batch = random.sample(sequences, k=min(4, len(sequences)))
        batch_tensor = torch.tensor(batch, device=device)
        input_ids = batch_tensor[:, :-1]
        labels = batch_tensor[:, 1:]

        logits, aux_loss, _ = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1))
        loss = loss + aux_loss.float() * 0.0

        if start_loss is None:
            start_loss = float(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())
        if step % max(1, max_steps // 5) == 0:
            print(f"Overfit step {step}: loss={final_loss:.4f}")

        if final_loss <= target_loss:
            break

    if start_loss is None or final_loss is None:
        raise RuntimeError("Overfit gate failed: loss not computed")

    improvement = (start_loss - final_loss) / max(start_loss, 1e-6)
    if final_loss > target_loss and improvement < 0.8:
        raise RuntimeError(
            f"Overfit gate failed: start={start_loss:.4f}, final={final_loss:.4f}, improvement={improvement:.2%}"
        )

    print("Overfit gate: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/stage1/stage1_data.jsonl")
    parser.add_argument("--bytes", type=int, default=1_000_000)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--target-loss", type=float, default=1.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset not found: {dataset_path}")

    byte_budget = args.bytes
    max_steps = args.max_steps
    if args.fast:
        byte_budget = min(byte_budget, 200_000)
        max_steps = min(max_steps, 60)

    with patched_cfg():
        run_overfit(dataset_path, byte_budget, max_steps, args.target_loss)


if __name__ == "__main__":
    main()

"""
Checkpoint restore drill to verify save/load integrity.
"""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
import sys

import torch

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
        "active_experts": getattr(cfg, "active_experts", cfg.num_experts_per_tok),
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
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.active_experts = 2
    cfg.num_heads = 4
    cfg.num_attention_heads = 4
    cfg.num_kv_heads = 2
    cfg.head_dim = 32
    cfg.num_layers = 2
    cfg.num_hidden_layers = 2
    cfg.vocab_size = 512
    cfg.max_seq_len = 64
    cfg.use_moe = True
    cfg.use_liquid = True
    cfg.use_qinn = False
    cfg.liquid_layers_idx = [1]
    cfg.moe_every_n_layers = 2
    cfg.use_gradient_checkpointing = False

    try:
        yield
    finally:
        for key, value in original.items():
            setattr(cfg, key, value)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    with patched_cfg():
        model = MertFormer().to(device)
        model.train()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
        _ = model(input_ids)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(input_ids)[0].mean()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = f"{tmpdir}/checkpoint.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": 123,
                },
                ckpt_path,
            )

            model2 = MertFormer().to(device)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            model2.load_state_dict(checkpoint["model"])
            optimizer2.load_state_dict(checkpoint["optimizer"])

            for key, tensor in model.state_dict().items():
                other = model2.state_dict()[key]
                if tensor.shape != other.shape:
                    raise RuntimeError(f"Shape mismatch for {key}")
                if not torch.allclose(tensor, other, atol=1e-6):
                    raise RuntimeError(f"Value mismatch for {key}")

    print("Checkpoint restore drill: PASS")


if __name__ == "__main__":
    main()

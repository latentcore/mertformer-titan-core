#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg, validate_layer_config
from model.transformers import MertFormer


class OfflineDemoTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"
    bos_token = "<s>"
    vocab_size = 259

    def encode(self, text: str, max_length: int) -> list[int]:
        budget = max(2, int(max_length))
        payload = [self.bos_token_id]
        payload.extend((byte + 3) for byte in text.encode("utf-8", errors="ignore")[: budget - 2])
        payload.append(self.eos_token_id)
        return payload[:budget]

    def __call__(self, text: str, truncation: bool = True, max_length: int | None = None, padding: str | None = None, return_tensors: str | None = None):
        limit = int(max_length or 256)
        ids = self.encode(text, limit)
        if padding == "max_length" and len(ids) < limit:
            ids = ids + [self.pad_token_id] * (limit - len(ids))
        tensor = torch.tensor([ids], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": tensor}
        return {"input_ids": tensor.tolist()}


class JsonlTokenDataset(Dataset):
    def __init__(self, path: Path, tokenizer: OfflineDemoTokenizer, max_len: int):
        self.samples: list[torch.Tensor] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = str(obj.get("text", "")).strip()
                if not text:
                    continue
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.samples.append(enc["input_ids"].squeeze(0))
        if not self.samples:
            raise RuntimeError(f"No usable samples found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        return item, item.clone()


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _safe_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return str(value)


def _cfg_snapshot() -> dict[str, Any]:
    return {k: _safe_json(v) for k, v in cfg.__dict__.items()}


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_demo_runtime(tokenizer: OfflineDemoTokenizer) -> None:
    demo_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    demo_steps = int(os.environ.get("TITAN_DEMO_MAX_STEPS", "12"))
    demo_seq_len = int(os.environ.get("TITAN_DEMO_SEQ_LEN", "256"))
    demo_batch = int(os.environ.get("TITAN_DEMO_BATCH_SIZE", "1"))
    demo_lr = float(os.environ.get("TITAN_DEMO_LR", "3e-4"))

    cfg.model_name = os.environ.get("TITAN_DEMO_MODEL_NAME", "MertFormer_Titan_Offline_4060_Demo")
    cfg.version = "v1.0-BUILD30-OFFLINE-4060-DEMO"
    cfg.device = demo_device
    cfg.param_dtype = torch.float32
    cfg.use_amp = False
    cfg.use_torch_compile = False
    cfg.use_gradient_checkpointing = False
    cfg.use_8bit_adam = False
    cfg.use_galore = False
    cfg.use_precomputed_logits = False
    cfg.require_gated_teacher = False
    cfg.distill_alpha = 0.0
    cfg.teacher_model_id = "offline-demo-none"
    cfg.use_tr_tokenizer = True
    cfg.tr_tokenizer_id = "offline_demo_tokenizer"

    cfg.hidden_size = 256
    cfg.intermediate_size = 768
    cfg.num_layers = 2
    cfg.num_hidden_layers = 2
    cfg.num_heads = 2
    cfg.num_attention_heads = 2
    cfg.num_kv_heads = 2
    cfg.head_dim = 128
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = demo_seq_len
    cfg.dropout = 0.0
    cfg.attention_dropout = 0.0

    cfg.use_moe = True
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 1
    cfg.active_experts = 1
    cfg.router_aux_loss_coef = 0.01
    cfg.aux_loss_coef = 0.01
    cfg.moe_every_n_layers = 2
    cfg.use_liquid = True
    cfg.liquid_layers_idx = [0]
    cfg.liquid_every_n_layers = 0
    cfg.liquid_warmup_steps = 0
    cfg.router_jitter = 0.0

    cfg.batch_size = demo_batch
    cfg.micro_batch_size = demo_batch
    cfg.grad_accum_steps = 1
    cfg.learning_rate = demo_lr
    cfg.weight_decay = 0.01
    cfg.warmup_steps = max(1, min(2, demo_steps))
    cfg.max_steps = demo_steps
    cfg.epoch_mode = False
    cfg.token_budget_mode = "fixed_steps"
    cfg.target_tokens_min = 0
    cfg.log_interval = 1
    cfg.save_interval = max(1, demo_steps // 2)
    cfg.val_check_interval = max(1, demo_steps // 3)
    cfg.early_stop_patience = demo_steps + 1
    cfg.grad_clip = 1.0
    cfg.dataloader_num_workers = 0
    cfg.dataloader_prefetch_factor = 1
    cfg.validation_min_samples_warn = 1
    cfg.validation_min_samples_claim = 1
    cfg.output_dir = "./checkpoints/offline_4060_demo"
    cfg.save_dir = "./checkpoints/offline_4060_demo"

    validate_layer_config(cfg)


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model: MertFormer, loader: DataLoader, device: torch.device, pad_id: int, vocab_size: int, max_batches: int = 4) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, (input_ids, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits, _, _ = model(input_ids, use_cache=False)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        val_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_id,
        )
        losses.append(float(val_loss.detach().item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(model: MertFormer, optimizer: torch.optim.Optimizer, step: int, metrics: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "config": _cfg_snapshot(),
    }
    torch.save(state, target)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    root = ROOT
    logs_dir = root / "logs"
    reports_dir = root / "reports"
    data_dir = root / "datasets" / "offline_demo"
    ckpt_dir = root / "checkpoints" / "offline_4060_demo"
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("offline demo corpus missing under datasets/offline_demo")

    tokenizer = OfflineDemoTokenizer()
    configure_demo_runtime(tokenizer)
    seed_all(int(getattr(cfg, "seed", 1453)))

    device = torch.device(cfg.device)
    train_ds = JsonlTokenDataset(train_path, tokenizer, cfg.max_seq_len)
    val_ds = JsonlTokenDataset(val_path, tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.micro_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.micro_batch_size, shuffle=False, num_workers=0)
    train_iter = cycle_loader(train_loader)

    model = MertFormer().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    log_path = logs_dir / "offline_4060_demo.jsonl"
    summary_path = reports_dir / "offline_4060_demo_summary.json"
    latest_ckpt = ckpt_dir / f"{cfg.model_name}_latest.pt"
    best_ckpt = ckpt_dir / f"{cfg.model_name}_best.pt"

    best_val = float("inf")
    started_at = time.time()

    print("============================================================")
    print("💻 MERTFORMER TITAN - OFFLINE RTX 4060 DEMO TRAINING")
    print("============================================================")
    print(f"Device        : {cfg.device}")
    print(f"Steps         : {cfg.max_steps}")
    print(f"Seq Len       : {cfg.max_seq_len}")
    print(f"Batch         : {cfg.micro_batch_size}")
    print(f"Train Samples : {len(train_ds)}")
    print(f"Val Samples   : {len(val_ds)}")
    print(f"Checkpoint Dir: {ckpt_dir}")
    print("Teacher/Net   : disabled")
    print("============================================================")

    for step in range(1, cfg.max_steps + 1):
        input_ids, labels = next(train_iter)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, aux_loss, _ = model(input_ids, use_cache=False)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = F.cross_entropy(
            shift_logits.view(-1, cfg.vocab_size),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )
        aux_value = aux_loss.float() if isinstance(aux_loss, torch.Tensor) else torch.tensor(float(aux_loss), device=device)
        total_loss = loss_ce + (float(getattr(cfg, "router_aux_loss_coef", 0.01)) * aux_value)

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss encountered at step {step}: {float(total_loss.detach().item())}")

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(cfg, "grad_clip", 1.0)))
        optimizer.step()

        payload = {
            "step": step,
            "loss": float(total_loss.detach().item()),
            "loss_ce": float(loss_ce.detach().item()),
            "aux_loss": float(aux_value.detach().item()),
            "grad_norm": float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "elapsed_s": round(time.time() - started_at, 3),
        }

        if step % cfg.val_check_interval == 0 or step == cfg.max_steps:
            val_loss = evaluate(model, val_loader, device, tokenizer.pad_token_id, cfg.vocab_size)
            payload["val_loss"] = float(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, step, payload, best_ckpt)

        if step % cfg.save_interval == 0 or step == cfg.max_steps:
            save_checkpoint(model, optimizer, step, payload, latest_ckpt)

        append_jsonl(log_path, payload)
        print(
            f"step={step:03d} loss={payload['loss']:.4f} ce={payload['loss_ce']:.4f} "
            f"aux={payload['aux_loss']:.4f} grad={payload['grad_norm']:.4f}" +
            (f" val={payload['val_loss']:.4f}" if 'val_loss' in payload else "")
        )

    summary = {
        "status": "completed",
        "mode": "offline_4060_demo",
        "device": cfg.device,
        "steps": cfg.max_steps,
        "seq_len": cfg.max_seq_len,
        "batch_size": cfg.micro_batch_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_val_loss": None if best_val == float("inf") else best_val,
        # Repo-relative paths only — absolute machine paths are forbidden in tracked
        # files (see tests/test_sdk_pilot_cli.py::test_no_desktop_paths_in_tracked_files).
        "latest_checkpoint": latest_ckpt.relative_to(root).as_posix(),
        "best_checkpoint": best_ckpt.relative_to(root).as_posix() if best_ckpt.exists() else None,
        "log_path": log_path.relative_to(root).as_posix(),
        "elapsed_s": round(time.time() - started_at, 3),
        "config": _cfg_snapshot(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Offline demo training completed. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

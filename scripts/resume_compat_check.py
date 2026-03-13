#!/usr/bin/env python3
"""
V2 checkpoint resume compatibility check.
Creates a tiny model, saves a checkpoint, then loads via train/train.py resume helpers.
Writes reports/resume_compat_report.json.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg  # noqa: E402
from model.transformers import MertFormer  # noqa: E402
from train.train import _load_resume_payload, _normalize_state_dict_keys_for_model  # noqa: E402


@contextmanager
def patched_cfg(tmp_save_dir: str, model_name: str):
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
        "model_name": getattr(cfg, "model_name", ""),
        "save_dir": getattr(cfg, "save_dir", ""),
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
    cfg.model_name = model_name
    cfg.save_dir = tmp_save_dir

    try:
        yield
    finally:
        for key, value in original.items():
            setattr(cfg, key, value)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    report_path = PROJECT_ROOT / "reports" / "resume_compat_report.json"
    tmp_rel = "reports/_resume_compat_tmp"
    tmp_dir = PROJECT_ROOT / tmp_rel
    model_name = "mertformer_resume_v2"
    step = 42

    os.environ["TITAN_AUTO_RESUME"] = "1"
    os.environ.pop("TITAN_RESUME_FROM", None)

    report = {
        "schema": "resume_compat_report_v2",
        "generated_at_utc": _utc_now(),
        "status": "FAIL",
        "checkpoint_path": "",
        "resume_step": 0,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "missing_keys_sample": [],
        "unexpected_keys_sample": [],
        "error": "",
    }

    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = tmp_dir / f"{model_name}_latest.pt"

        with patched_cfg(tmp_rel, model_name):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MertFormer().to(device)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "val_loss": 1.234,
            }
            torch.save(state, ckpt_path)

            resume_model = MertFormer().to(device)
            payload = _load_resume_payload(cfg, resume_model, is_main_process=False)
            if payload is None:
                raise RuntimeError("resume_payload_none")

            model_check = MertFormer().to(device)
            normalized = _normalize_state_dict_keys_for_model(state["model"], model_check)
            missing, unexpected = model_check.load_state_dict(normalized, strict=False)

            report.update(
                {
                    "status": "PASS",
                    "checkpoint_path": str(ckpt_path),
                    "resume_step": int(payload.get("step", 0)),
                    "missing_keys": len(missing),
                    "unexpected_keys": len(unexpected),
                    "missing_keys_sample": list(missing)[:10],
                    "unexpected_keys_sample": list(unexpected)[:10],
                }
            )
            if int(payload.get("step", 0)) != step:
                report["status"] = "FAIL"
                report["error"] = f"resume_step_mismatch expected={step} got={payload.get('step', 0)}"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

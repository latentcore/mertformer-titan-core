"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - ONNX EXPORT VALIDATOR
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 27) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD27"
__author__ = "Mert"

import torch
import torch.nn as nn
import os
import sys
import inspect
import warnings

# Add project root to path
sys.path.append(os.getcwd())

from model.transformers import MertFormer
from config.config import cfg

# Wrapper Class (Copied from train.py)
class MertFormerInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        logits, _, _ = self.model(input_ids)
        return logits


def _onnx_export_compat(wrapper, dummy_input, save_path):
    """Export with stable defaults across Torch/ONNX exporter variants."""
    export_kwargs = dict(
        export_params=True,
        opset_version=12,  # Legacy stable opset for project smoke testing.
        do_constant_folding=False,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )

    def _run_export(**kwargs):
        # Keep smoke output clean in CI by suppressing known exporter deprecations.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning, message=".*dynamic_axes.*dynamo=True.*")
            torch.onnx.export(
                wrapper,
                dummy_input,
                save_path,
                **kwargs,
            )

    export_sig = inspect.signature(torch.onnx.export)
    if "dynamo" in export_sig.parameters:
        modern_kwargs = dict(export_kwargs)
        modern_kwargs["dynamo"] = True
        try:
            _run_export(**modern_kwargs)
            return
        except Exception as exc:
            # Fall back to legacy exporter for compatibility in constrained envs.
            print(f"⚠️ Modern ONNX exporter fallback activated: {exc}")

        legacy_kwargs = dict(export_kwargs)
        legacy_kwargs["dynamo"] = False
        _run_export(**legacy_kwargs)
        return

    _run_export(**export_kwargs)

def test_export():
    print("📦 TESTING ONNX EXPORT (Tiny Mode)...")
    
    # Patch Config for Speed
    orig = {
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "num_kv_heads": getattr(cfg, "num_kv_heads", cfg.num_heads),
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "num_experts": cfg.num_experts,
        "active_experts": getattr(cfg, "active_experts", getattr(cfg, "num_experts_per_tok", 1)),
        "device": cfg.device,
        "use_moe": cfg.use_moe,
        "use_liquid": cfg.use_liquid,
    }

    cfg.hidden_size = 64
    cfg.num_layers = 2
    cfg.num_heads = 4
    cfg.num_kv_heads = 2  # Keep KV heads <= Q heads and divisible for GQA
    cfg.head_dim = 16
    cfg.vocab_size = 1000
    cfg.num_experts = 2
    cfg.active_experts = 1
    cfg.device = "cpu" # Keep it simple
    cfg.use_moe = True
    cfg.use_liquid = True
    
    print("   Initializing Model...")
    model = MertFormer()
    model.eval()
    
    wrapper = MertFormerInferenceWrapper(model)
    wrapper.eval()
    
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 32))
    save_path = "test_export.onnx"
    data_path = save_path + ".data"  # Torch may emit external tensor data here.
    
    print("   Exporting...")
    try:
        _onnx_export_compat(wrapper, dummy_input, save_path)
        print(f"✅ ONNX Export Successful: {save_path}")
        
        # Verify File Exists and Size
        size = os.path.getsize(save_path) / 1024
        print(f"   Size: {size:.2f} KB")
        
        # Clean up
        if os.path.exists(data_path):
            os.remove(data_path)
        os.remove(save_path)
        print("   Cleanup OK.")
        assert True
        
    except Exception as e:
        print(f"❌ ONNX Export Failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Best-effort cleanup in case export failed mid-way.
        try:
            if os.path.exists(data_path):
                os.remove(data_path)
        except Exception:
            pass
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass
        for k, v in orig.items():
            setattr(cfg, k, v)

if __name__ == "__main__":
    test_export()
    sys.exit(0)

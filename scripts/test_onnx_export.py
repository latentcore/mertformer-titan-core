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

def test_export():
    print("📦 TESTING ONNX EXPORT (Tiny Mode)...")
    
    # Patch Config for Speed
    cfg.hidden_size = 64
    cfg.num_layers = 2
    cfg.num_heads = 4
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
    
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 32))
    save_path = "test_export.onnx"
    
    print("   Exporting...")
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=12, # [FIX] Legacy Stable Opset
            do_constant_folding=False, # [FIX] Disable folding to avoid graph capture errors
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        print(f"✅ ONNX Export Successful: {save_path}")
        
        # Verify File Exists and Size
        size = os.path.getsize(save_path) / 1024
        print(f"   Size: {size:.2f} KB")
        
        # Clean up
        os.remove(save_path)
        print("   Cleanup OK.")
        return True
        
    except Exception as e:
        print(f"❌ ONNX Export Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_export():
        sys.exit(0)
    else:
        sys.exit(1)

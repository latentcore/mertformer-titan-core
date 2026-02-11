"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - MOBILE EXPORTER
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 27) — Pre-Training
Status : PRE-TRAINING (CLOUD MODE)
==============================================================================

PURPOSE:
This script is designed to run on the CLOUD (A100/H100) after training completions.
It exports the 'MertFormer' PyTorch model to a highly optimized ONNX Graph.

FEATURES:
1. Full Graph Optimization (Constant Folding ON)
2. Dynamic Quantization (INT8) for Mobile/NPU efficiency
3. Numerical Verification (PyTorch vs ONNX output comparison)
4. BitNet Weight Analysis

NOTE: Requires ~32GB RAM for 3B parameter models during optimization.
"""

__version__ = "1.0-BUILD27"
__author__ = "Mert"

import os
import sys
import time
from pathlib import Path
import torch
import numpy as np

# Handle pathing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Dependencies
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("⚠️  WARNING: 'onnx' or 'onnxruntime' not found.")
    print("   -> Since this is a production script, please install them on the Cloud Environment:")
    print("   -> pip install onnx onnxruntime")
    # We continue strictly for PyTorch export part, but verification will fail.
    
from model.transformers import MertFormer
from config.config import cfg

def export_production_model(ckpt_override=None, output_dir=None, bitpack: bool = False):
    # Production-grade export pipeline: load model, validate, export ONNX, and emit metadata for mobile deploy.
    print("\n" + "="*60)
    print("🚀 MERTFORMER TITAN: PRODUCTION MOBILE EXPORT PROTOCOL")
    print(f"   Target Device: Samsung S25 NPU (via ONNX)")
    print("="*60)

    # -------------------------------------------------------------------------
    # 1. SETUP & MODEL LOADING
    # -------------------------------------------------------------------------
    # Target the Build 27 pre-training checkpoint
    ckpt_dir = output_dir or os.path.join(project_root, "checkpoints", "mertformer_titan_prod")
    ckpt_path = os.path.join(ckpt_dir, "MertFormer_Titan_Nano_Final.pt")
    
    # Fallback to simulation checkpoint if production not found (for testing)
    if ckpt_override:
        if str(ckpt_override) == "latest":
            ckpt_path = os.path.join(project_root, cfg.save_dir, f"{cfg.model_name}_latest.pt")
        else:
            ckpt_path = str(ckpt_override)

    if not os.path.exists(ckpt_path):
        print(f"⚠️  Production Checkpoint not found at: {ckpt_path}")
        sim_path = os.path.join(project_root, "checkpoints", "mac_simulation_model.pt")
        if os.path.exists(sim_path):
             print(f"   -> Falling back to Simulation Checkpoint: {sim_path}")
             ckpt_path = sim_path
        else:
             print("❌ CRITICAL: No model checkpoints found!")
             return

    print(f"\n📂 Loading Weights: {ckpt_path}")
    
    # Initialize Model (CPU is preferred for Export stability)
    device = torch.device("cpu")
    model = MertFormer()
    model.to(device)
    model.eval()

    # Load State Dict
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Clean prefix
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("✅ Model loaded successfully into RAM.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # -------------------------------------------------------------------------
    # 1.5 OPTIONAL BITPACK EXPORT (TERNARY 5-IN-8)
    # -------------------------------------------------------------------------
    if bitpack or os.getenv("MERTFORMER_BITPACK", "0") == "1":
        try:
            from mertformer_sdk.utils.bitpack import pack_state_dict
            bitpack_bin = os.path.join(ckpt_dir, "titan_s25_bitpack.bin")
            bitpack_meta = os.path.join(ckpt_dir, "titan_s25_bitpack.json")
            pack_state_dict(model.state_dict(), Path(bitpack_bin), Path(bitpack_meta))
            print("✅ Bitpack export complete.")
        except Exception as e:
            print(f"⚠️  Bitpack export skipped: {e}")

    # -------------------------------------------------------------------------
    # 2. EXPORT PREPARATION (KV CACHE & DYNAMIC AXES)
    # -------------------------------------------------------------------------
    print("\n⚙️  Configuring ONNX Graph Tracing...")
    
    output_fp32 = os.path.join(ckpt_dir, "titan_s25_fp32.onnx")
    output_int8 = os.path.join(ckpt_dir, "titan_s25_int8_quantized.onnx")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dummy Inputs
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long)
    
    # Setup Dynamic Axes & Names
    input_names = ['input_ids']
    output_names = ['logits', 'aux_loss']
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }

    # KV Cache Handling
    past_key_values = []
    num_heads = cfg.num_heads
    head_dim = cfg.head_dim
    
    for i in range(cfg.num_layers):
        # Initial Past: [Batch, Heads, Seq=0, Dim]
        # We trace with seq=0 to support cold-start
        k = torch.zeros(1, num_heads, 0, head_dim)
        v = torch.zeros(1, num_heads, 0, head_dim)
        past_key_values.append((k, v))
        
        # Naming
        k_in, v_in = f'past_key_values_{i}_key', f'past_key_values_{i}_value'
        k_out, v_out = f'present_key_values_{i}_key', f'present_key_values_{i}_value'
        
        input_names.extend([k_in, v_in])
        output_names.extend([k_out, v_out])
        
        dynamic_axes[k_in] = {0: 'batch_size', 2: 'past_sequence_length'}
        dynamic_axes[v_in] = {0: 'batch_size', 2: 'past_sequence_length'}
        dynamic_axes[k_out] = {0: 'batch_size', 2: 'total_sequence_length'}
        dynamic_axes[v_out] = {0: 'batch_size', 2: 'total_sequence_length'}

    # Wrapper to handle Tuple unfolding for ONNX
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, *past_kv_flat):
            # Reconstruct list of tuples
            pkv = []
            for i in range(0, len(past_kv_flat), 2):
                pkv.append((past_kv_flat[i], past_kv_flat[i+1]))
            
            # Forward
            logits, aux, present_pkv = self.model(input_ids, past_key_values=pkv, use_cache=True)
            
            # Flatten output
            flat_present = []
            for k, v in present_pkv:
                flat_present.extend([k, v])
                
            return (logits, aux) + tuple(flat_present)

    wrapper = OnnxWrapper(model)
    
    # Flatten dummy inputs for tracing
    flat_dummy_past = []
    for k, v in past_key_values:
        flat_dummy_past.extend([k, v])
    trace_inputs = (dummy_input,) + tuple(flat_dummy_past)

    # -------------------------------------------------------------------------
    # 3. EXPORT EXECUTION (MASTER FP32)
    # -------------------------------------------------------------------------
    print(f"\n🏗️  Exporting Master Graph (FP32) to: {output_fp32}...")
    print("   -> Optimizations: Constant Folding=TRUE (Assuming High Cloud RAM)")
    
    try:
        start_t = time.time()
        torch.onnx.export(
            wrapper,
            trace_inputs,
            output_fp32,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            export_params=True,
            do_constant_folding=True, # PRODUCTION MODE: ENABLED
            keep_initializers_as_inputs=False, # PRODUCTION MODE: Bake into graph
            verbose=False
        )
        size_gb = os.path.getsize(output_fp32) / (1024**3)
        print(f"✅ Export Success in {time.time()-start_t:.2f}s")
        print(f"📦 Master Graph Size: {size_gb:.2f} GB")
        
    except Exception as e:
        print(f"❌ FP32 Export Failed: {e}")
        return

    # -------------------------------------------------------------------------
    # 4. QUANTIZATION (MOBILE OPTIMIZATION)
    # -------------------------------------------------------------------------
    # Note: Requires onnxruntime
    if 'onnxruntime' in sys.modules:
        print(f"\n📱 Generating Mobile-Optimized INT8 Model...")
        print(f"   Target: {output_int8}")
        
        try:
            quantize_dynamic(
                model_input=output_fp32,
                model_output=output_int8,
                weight_type=QuantType.QInt8 # Standard for Mobile NPU
            )
            q_size_gb = os.path.getsize(output_int8) / (1024**3)
            print(f"✅ Quantization Success!")
            print(f"📦 Mobile Graph Size: {q_size_gb:.2f} GB")
            print(f"📉 Compression Ratio: {size_gb/q_size_gb:.2f}x")
            
        except Exception as e:
            print(f"❌ Quantization Failed: {e}")
    else:
        print("\n⚠️  Skipping Quantization (onnxruntime not installed)")

    # -------------------------------------------------------------------------
    # 4.5 ONNX METADATA (BITPACK HOOK)
    # -------------------------------------------------------------------------
    if bitpack or os.getenv("MERTFORMER_BITPACK", "0") == "1":
        try:
            from mertformer_sdk.utils.onnx_meta import add_bitpack_metadata
            if os.path.exists(output_fp32):
                add_bitpack_metadata(output_fp32)
            if os.path.exists(output_int8):
                add_bitpack_metadata(output_int8)
            print("✅ ONNX metadata updated for bitpack.")
        except Exception as e:
            print(f"⚠️  ONNX metadata update failed: {e}")

    # -------------------------------------------------------------------------
    # 5. VERIFICATION
    # -------------------------------------------------------------------------
    if 'onnxruntime' in sys.modules:
        print(f"\n🔬 Verifying Numerical Integrity...")
        try:
            ort_session = ort.InferenceSession(output_fp32, providers=['CPUExecutionProvider'])
            
            # Prepare inputs
            ort_inputs = {'input_ids': dummy_input.numpy()}
            for i, (k, v) in enumerate(past_key_values):
                ort_inputs[f'past_key_values_{i}_key'] = k.numpy()
                ort_inputs[f'past_key_values_{i}_value'] = v.numpy()
                
            # Run ONNX
            ort_outs = ort_session.run(None, ort_inputs)
            onnx_logits = ort_outs[0]
            
            # Run PyTorch
            with torch.no_grad():
                pt_out = wrapper(dummy_input, *flat_dummy_past)
                pt_logits = pt_out[0].numpy()
                
            # Compare
            diff = np.max(np.abs(onnx_logits - pt_logits))
            print(f"   Mean Difference: {np.mean(np.abs(onnx_logits - pt_logits)):.6f}")
            print(f"   Max Difference : {diff:.6f}")
            
            if diff < 1e-4:
                print("✅ VERIFIED: Model Logic is identical.")
            else:
                print("⚠️  WARNING: Precision drift detected (Expected for BF16->FP32 export).")
                
        except Exception as e:
            print(f"❌ Verification Failed: {e}")

    print("\n" + "="*60)
    print("🏁 EXPORT PROTOCOL COMPLETE")
    print("="*60)

if __name__ == "__main__":
    export_production_model()

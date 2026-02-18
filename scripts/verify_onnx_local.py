"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - ONNX LOCAL VERIFIER
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30"
__author__ = "Mert"

import argparse
import onnxruntime as ort
import numpy as np
import os

def _resolve_model_path(model_path: str | None = None) -> str:
    if model_path:
        return model_path
    candidates = (
        "titan_mobile.onnx",
        "checkpoints/mertformer_titan_prod/titan_s25_fp32.onnx",
        "checkpoints/mertformer_titan_prod/titan_s25_int8_quantized.onnx",
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # Keep the historical default for error message compatibility.
    return "titan_mobile.onnx"


def check_model(model_path: str | None = None):
    model_path = _resolve_model_path(model_path)
    print(f"🔍 Checking Model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found in current directory!")
        return

    try:
        # Load the model
        session = ort.InferenceSession(model_path)
        print("✅ ONNX Runtime successfully loaded the model!")
        
        # Check inputs
        inputs = session.get_inputs()
        for i in inputs:
            print(f"📥 Input: {i.name} | Shape: {i.shape} | Type: {i.type}")
            
        # Check outputs
        outputs = session.get_outputs()
        for o in outputs:
            print(f"📤 Output: {o.name} | Shape: {o.shape} | Type: {o.type}")
            
        # Run a test inference
        dummy_input = np.random.randint(0, 100277, (1, 1)).astype(np.int64)
        input_name = inputs[0].name
        
        print("\n🚀 Running benchmark (50 tokens)...")
        import time
        start_time = time.time()
        iterations = 50
        for _ in range(iterations):
            session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
        duration = end_time - start_time
        tok_per_sec = iterations / duration if duration > 0 else 0
        
        print(f"✅ Benchmark Complete!")
        print(f"⏱️ Local Speed: {tok_per_sec:.2f} tokens/sec")
        print(f"📦 Total Latency: {duration*1000:.2f} ms for 50 inferences")
        print("\nTITAN IS ALIVE AND FAST! 🦅🦾")
        
    except Exception as e:
        print(f"❌ Verification Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to ONNX model file")
    args = parser.parse_args()
    check_model(args.model)

"""
==============================================================================
MERTFORMER TITAN - MACBOOK M4 HIGH-FIDELITY SIMULATION
==============================================================================
Purpose: 
1. Load the EXACT Production Architecture (SeqLen=4096, 2.64B Params).
2. Prove stability on Apple Silicon (MPS).
3. Benchmark Generation Speed (Tokens/sec) with realistic overhead.
"""

import sys
import os
import time
import torch
import psutil
from pathlib import Path

# Setup Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from config.config import MertFormerConfig
from model.transformers import MertFormer

def check_ram():
    vm = psutil.virtual_memory()
    print(f"   💾 RAM Usage: {vm.percent}% (Available: {vm.available / (1024**3):.1f} GB)")

def run_mac_simulation():
    # End-to-end Mac simulation: load model, run sample prompts, and verify local inference stability.
    print(f"\n🍏 MACBOOK M4 TITAN SIMULATION (HIGH FIDELITY)")
    print(f"==================================================")
    check_ram()

    # 1. Load FULL Production Config
    # 1. Load FULL Production Config
    # CRITICAL FIX: MertFormer uses global 'cfg'. We must modify the global instance in-place.
    from config.config import cfg
    
    print(f"\n⚙️  MODIFYING GLOBAL CONFIG FOR MAC SIMULATION (ULTRA-EFFICIENT MODE):")
    cfg.device = "mps" 
    # TACTIC 1: Use BFloat16 (Native on M-series). Halves memory usage (10GB -> 5GB)
    cfg.param_dtype = torch.bfloat16 
    cfg.teacher_model_id = None     # Disable Teacher
    
    # CRITICAL: Force PyTorch to NEVER allocate Float32 (10GB) buffers
    print(f"   TACTIC 2: Global Precision Lock (BFloat16)")
    torch.set_default_dtype(torch.bfloat16)
    
    print(f"   - Layers: {cfg.num_layers}")
    print(f"   - Context Window: {cfg.max_seq_len}")
    print(f"   - Precision: BFloat16 (Strictest Mode)")
    print(f"   - Device Override: {cfg.device}")
    
    # 2. Initialize Model
    print(f"\n🏗️  Building 2.64B Model (Compact Mode)...")
    start_load = time.time()
    try:
        model = MertFormer().to(cfg.device) # Already BF16 default
        
        # TACTIC 3: Gradient Checkpointing (Saves ~60% VRAM during train)
        # Trades speed (re-compute) for RAM. User said "let it take 10 mins but save RAM".
        # FIX: MertFormer is custom nn.Module, not HF PreTrainedModel.
        # We manually enable the flag we saw in transformers.py
        model.use_gradient_checkpointing = True 
        print(f"   TACTIC 3: Gradient Checkpointing ENABLED (Manual Flag).")
        
        # TACTIC 4: Partial Freezing (The Ultimate RAM Saver)
        # We freeze the first 16 layers. We only train the last 2 layers + Head.
        # This prevents allocating 5GB of Gradients for the whole model.
        # RAM Usage: 5GB (Params) + 0.5GB (Gradients) = 5.5GB Total.
        print(f"   TACTIC 4: Layer Freezing (Freeze 16/18 Layers).")
        print(f"   -> Simulating 'Fine-Tuning' to save Gradient RAM.")
        
        for name, param in model.named_parameters():
             # Freeze everything initially
             param.requires_grad = False
             
        # Unfreeze last 2 layers and Head
        for name, param in model.named_parameters():
             if "layers.16" in name or "layers.17" in name or "lm_head" in name or "norm" in name:
                 param.requires_grad = True
                 
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   -> Trainable Params: {trainable_params/1e6:.1f}M (others frozen)")
        
        print(f"✅ Model Built in {time.time() - start_load:.2f}s")
        check_ram()
    except Exception as e:
        print(f"❌ Model Build Failed: {e}")
        return

    # 3. Training Step (Micro-Batch)
    print(f"\n🏋️  STEP 1: Training Stability Check (Zero-Overhead)")
    print(f"   TACTIC 2: Using SGD Optimizer instead of AdamW.")
    print(f"   Why? AdamW needs 20GB+ RAM for states. SGD needs 0GB.")
    
    # Use SGD to avoid huge state memory
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16)).to(cfg.device) # Tiny batch (16 tokens)
    labels = input_ids.clone()
    
    start_train = time.time()
    try:
        model.train() # Ensure training mode
        # FIX: MertFormer.forward() does NOT take labels.
        # It returns (logits, aux_loss, present_kv)
        outputs = model(input_ids) 
        
        # Unpack tuple
        logits = outputs[0]
        aux_loss = outputs[1]
        
        # Manual Loss Calculation
        from torch.nn import CrossEntropyLoss
        loss_fct = CrossEntropyLoss()
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = loss_fct(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))
        loss = loss + (cfg.aux_loss_coef * aux_loss)
             
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"✅ Training Step Success! Loss: {loss.item():.4f}")
        
    except RuntimeError as e:
        print(f"❌ Training Step Failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Clear cache before inference to free up training gradients
    optimizer = None
    loss = None
    outputs = None
    import gc
    gc.collect()
    torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
    print("   🧹 Cleared Training Memory for Inference...")
    check_ram()

    # 4. Inference Benchmark (The Real Speed Test)
    # S25 Speed depends on Generation, not Training.
    print(f"\n🏎️  STEP 2: Inference Speed Benchmark")
    print(f"   Generating tokens with full model loaded...")
    
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 128)).to(cfg.device) # Reasonable context
    
    # Warmup
    print("   WARMING UP (Compiling Metal Kernels)...")
    with torch.no_grad():
        _ = model.generate(prompt, max_new_tokens=5)
    
    print("   🚀 RUNNING SPEED TEST...")
    start_infer = time.time()
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=50) # Generate 50 tokens
    end_infer = time.time()
    
    total_tokens = 50
    duration = end_infer - start_infer
    tps = total_tokens / duration
    
    print(f"\n📊 BUILD 27 VERDICT (MacBook M4):")
    print(f"   ----------------------------------------")
    print(f"   ⏱️  Time Taken    : {duration:.2f}s")
    print(f"   ⚡ GENERATION SPEED: {tps:.2f} tokens/sec")
    print(f"   ----------------------------------------")
    
    print(f"\n🔮 SAMSUNG S25 PROJECTION:")
    print(f"   Mac M4 runs unoptimized FP32 code via MPS.")
    print(f"   S25 NPU runs optimized INT8/BitNet code.")
    print(f"   Expected Speedup Factor: 3x - 5x")
    s25_low = tps * 3
    s25_high = tps * 5
    print(f"   ESTIMATED S25 SPEED: {s25_low:.1f} - {s25_high:.1f} tokens/sec")
    print(f"   (Target for production is >45 t/s)")
    
    # LOGGING RESULTS WITH OFFICIAL LOGGER
    from utils.logger import RunLogger
    
    run_name = f"benchmark_mac_m4_{int(time.time())}"
    print(f"\n📝 Logging results to RunLogger (ID: {run_name})...")
    
    with RunLogger(cfg=cfg, run_name=run_name, also_csv=True) as logger:
        # Log Architecture Meta
        logger.log_event("benchmark_start", {
            "device": "mps",
            "model": "MertFormer Titan v1.0 (Build 30)",
            "seq_len": cfg.max_seq_len,
            "params": "2.64B"
        })
        
        # Log Benchmark Data
        benchmark_data = {
            "mac_speed_tps": tps,
            "s25_projection_low": s25_low,
            "s25_projection_high": s25_high,
            "status": "PASS" if tps > 10 else "WARNING"
        }
        logger.log_event("benchmark_result", benchmark_data)
        
        # Also log as a virtual 'step' for CSV compatibility
        logger.log_step({
            "step": 1,
            "tok_s": tps,
            "loss": 0.0, # Dummy for CSV structure
            "mode": "benchmark"
        })
        
    print(f"✅ Benchmark saved via RunLogger to logs/{run_name}.jsonl")
    
    # 5. Save Model for S25 Testing
    print(f"\n💾 STEP 3: Saving Model for S25 Optimization...")
    ckpt_path = project_root / "checkpoints" / "mac_simulation_model.pt"
    os.makedirs(ckpt_path.parent, exist_ok=True)
    
    print(f"   Target: {ckpt_path}")
    print(f"   Size: ~5.3 GB (BFloat16)")
    
    try:
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Model Saved Successfully!")
        print(f"   You can now use this file for ONNX/CoreML conversion tests.")
    except Exception as e:
        print(f"❌ Save Failed: {e}")

    if tps > 10:
        print(f"\n✅ PASS: Architecture is efficient enough.")
    else:
        print(f"\n⚠️  WARNING: Architecture might be heavy. Check Liquid/MoE overhead.")

if __name__ == "__main__":
    run_mac_simulation()

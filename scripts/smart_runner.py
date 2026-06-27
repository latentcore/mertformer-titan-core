"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - SMART RUNNER ORCHESTRATOR
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert Yünlü"

import os
import time
import subprocess
import threading
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Setup Paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))
from config.config import cfg

def run_command(cmd, desc):
    print(f"🚀 [{desc}] Starting: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ [{desc}] Completed.")
    except subprocess.CalledProcessError as e:
        print(f"❌ [{desc}] Failed with error: {e}")
        sys.exit(1)

def data_pipeline_thread():
    run_command([sys.executable, "scripts/data_pipeline.py"], "DATA PIPELINE")

def distillation_monitor():
    print("👀 DISTILLATION MONITOR: Watching for ready datasets...")
    stages_processed = set()
    total_stages = 5
    
    while len(stages_processed) < total_stages:
        for stage in range(1, total_stages + 1):
            if stage in stages_processed:
                continue
                
            # Check for signal file
            # Path structure: datasets/stageX/stageX_done.signal
            # Note: Directory names in data_pipeline are "stage1", "stage2", "stage3", "stage4_soul", "stage5_tools"
            # We need to map stage number to directory name exactly as in data_pipeline.py
            
            stage_dir_name = f"stage{stage}"
            if stage == 4: stage_dir_name = "stage4_soul"
            if stage == 5: stage_dir_name = "stage5_tools"
            
            signal_path = project_root / "datasets" / stage_dir_name / f"stage{stage}_started.signal"
            
            if signal_path.exists():
                print(f"⚡ DETECTED: Stage {stage} Started! Launching Concurrent Distillation...")
                
                # Run Distillation Manager for this stage
                cmd = [sys.executable, "orchestrator/distillation_manager.py", "--stage", str(stage)]
                run_command(cmd, f"DISTILL STAGE {stage}")
                
                stages_processed.add(stage)
                print(f"📊 Progress: {len(stages_processed)}/{total_stages} stages distilled.")
                
        time.sleep(5) # Poll every 5 seconds
        
    print("🎉 ALL STAGES DISTILLED! Ready for Training.")


def enforce_teacher_access_contract() -> None:
    """
    Hard-fail contract for gated teacher availability.
    """
    if not bool(getattr(cfg, "require_gated_teacher", False)):
        return

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ require_gated_teacher=true but HF_TOKEN is missing.")
        print("   Action: export HF_TOKEN and ensure teacher model access is approved.")
        sys.exit(1)
    try:
        tok = AutoTokenizer.from_pretrained(cfg.teacher_model_id, token=hf_token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    except Exception as exc:
        print(f"❌ Teacher tokenizer access failed for {cfg.teacher_model_id}: {exc}")
        print("   Hard fail policy active (no fallback).")
        sys.exit(1)

def main():
    if os.environ.get("TITAN_OFFLINE", "1") != "0":
        print("❌ TITAN_OFFLINE=1 (offline-first). smart_runner performs dataset downloads and training.")
        print("   Set TITAN_OFFLINE=0 to proceed.")
        sys.exit(2)

    enforce_teacher_access_contract()

    print("================================================================")
    print("🔥 MERTFORMER TITAN - SMART PARALLEL ORCHESTRATOR")
    print("================================================================")
    print("1. Starting Data Pipeline (Background Thread)")
    print("2. Starting Distillation Monitor (Main Thread)")
    print("   -> Will trigger Llama-3-70B Logic as soon as data arrives.")
    print("================================================================")
    
    # 1. Start Data/Network Thread
    t_data = threading.Thread(target=data_pipeline_thread)
    t_data.start()
    
    # 2. Run Distillation (GPU Heavy) in Main Thread (Sequential to each other, parallel to download)
    distillation_monitor()
    
    # 3. Wait for Data Thread (Should be done by now if distillation is done)
    t_data.join()
    
    print("\n" + "="*60)
    print("🚀 LAUNCHING TRAINING (MertFormer Titan)")
    print("="*60)
    
    # 4. Launch Training
    # Ensure config uses precomputed logits
    # Use `python -m accelerate` so the venv interpreter is always used.
    training_cmd = [sys.executable, "-m", "accelerate", "launch", "--main_process_port", "29501", "train/train.py"]
    run_command(training_cmd, "TRAINING")

if __name__ == "__main__":
    main()

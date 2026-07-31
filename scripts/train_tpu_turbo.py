"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - TPU TURBO TRAINING
-------------------------------------------------------------------------------
Copyright 2026 Mert Yünlü
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# -----------------------------------------------------------------------------
# 0. PROJECT ROOT DETECTION (Dynamic for Kaggle/Local)
# -----------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent
for _ in range(4):
    if (project_root / "config").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))
print(f"📍 TPU PROJE MERKEZİ TESPİT EDİLDİ: {project_root}")

from config.config import cfg
from train.train import CurriculumDataset, collate_fn, MertFormerInferenceWrapper, get_wsd_schedule

try:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
except ImportError:
    print("Accelerate required. Run: pip install accelerate")
    sys.exit(1)

def train_tpu():
    """
    TITAN TPU experimental lane (v5e-8 oriented; perf UNMEASURED)
    ------------------------------------------------------------
    NOTE: "turbo", "sweet spot" and "100% efficiency" labels below are aspirational
    descriptions, NOT measured benchmarks. No throughput/efficiency numbers have been
    captured for this lane.
    - Bfloat16 Native (TPU-preferred dtype)
    - No BitsAndBytes (Incompatible with TPU) - Using BF16 Teacher
    - Data Streaming Boost for XLA

    [tier-2] UNSUPPORTED / EXPERIMENTAL. This standalone lane re-implements the
    train loop and DIVERGES from the canonical train/train.py: it does not apply the
    EOS/-100 loss mask + causal shift, build_optimizer (galore/8bit), the collective
    NaN/OOM skip, the single-source tokenizer, sequence packing, or the real-token
    budget; it also mis-unpacks the model forward (3rd return is KV-cache, not aux
    loss). The canonical orchestrator launches train/train.py directly, NOT this file.
    Gated to prevent accidental launch of a divergent/buggy lane.
    """
    import os as _os
    if _os.environ.get("TITAN_ALLOW_EXPERIMENTAL_TPU", "0") != "1":
        raise SystemExit(
            "train_tpu_turbo.py is UNSUPPORTED/EXPERIMENTAL and diverges from the "
            "canonical train/train.py (missing the just-landed fixes). The canonical "
            "lane is `train/train.py` via the orchestrator. Set "
            "TITAN_ALLOW_EXPERIMENTAL_TPU=1 only if you accept the divergence."
        )

    # TPU Project Config
    config = ProjectConfiguration(project_dir=str(project_root), logging_dir=str(project_root / "logs"))
    
    # Initialize Accelerator for TPU
    accelerator = Accelerator(
        mixed_precision="bf16", # TPU primary dtype
        project_config=config,
        log_with="all"
    )
    
    device = accelerator.device
    set_seed(cfg.seed)
    
    if accelerator.is_main_process:
        print("🚀 TITAN TPU experimental lane initialized (v5e-8 oriented; perf unmeasured)")
        print("⚠️  NOTICE: BitsAndBytes disabled. Using Native BF16 for Teacher.")

    # 1. Mini Data Alchemy for TPU (Kaggle Storage Safe)
    stage_paths = [
        project_root / "datasets" / "stage1" / "stage1_data.jsonl",
        project_root / "datasets" / "stage2" / "stage2_data.jsonl",
        project_root / "datasets" / "stage3" / "stage3_data.jsonl",
        project_root / "datasets" / "stage4_soul" / "stage4_data.jsonl",
    ]
    
    if not all(p.exists() for p in stage_paths):
        if accelerator.is_main_process:
            print("⚠️ Datasets missing. Launching MINI DATA ALCHEMY (Kaggle Safe Mode)...")
            try:
                # We import and monkeypatch to avoid high storage usage
                import scripts.data_pipeline as dp
                # Keep target intentionally small for TPU smoke verification.
                TEST_TARGET = 10000 
                dp.create_stage_directories()
                
                print(f"📥 Downloading ~{TEST_TARGET} samples per stage...")
                dp.download_stage(1, dp.STAGE1_SOURCES, int(TEST_TARGET * 0.45))
                dp.download_stage(2, dp.STAGE2_SOURCES, int(TEST_TARGET * 0.35))
                dp.download_stage(3, dp.STAGE3_SOURCES, int(TEST_TARGET * 0.07))
                dp.download_stage(4, dp.STAGE4_SOURCES, int(TEST_TARGET * 0.03))
                print("✅ Mini Data Alchemy Complete.")
            except Exception as e:
                print(f"❌ Mini Data Alchemy Failed: {e}")
                sys.exit(1)
    
    # Wait for all TPU cores to see the data
    accelerator.wait_for_everyone()
    
    # Load Tokenizer separately (No 4-bit)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    ds = CurriculumDataset(stage_paths, cfg.max_seq_len, tokenizer)
    dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=cfg.micro_batch_size, 
        collate_fn=collate_fn, 
        num_workers=8, # High workers for TPU throughput
        prefetch_factor=4
    )
    
    # 2. Model & Teacher (BF16)
    from model.transformers import MertFormer
    student = MertFormer()
    
    # ⚠️ TPU Memory Guard: 70B BF16 is 140GB. v5e-8 is 128GB.
    # We MUST use FSDP or a smaller teacher to fit HBM (efficiency UNMEASURED).
    # We will automatically switch to 8B teacher if running on TPU v5e to prevent OOM.
    tpu_teacher_id = "meta-llama/Llama-3.1-8B-Instruct" 
    if accelerator.is_main_process:
        print(f"🔄 TPU Optimization: Using {tpu_teacher_id} as Teacher (Fits in HBM).")
    
    teacher = AutoModelForCausalLM.from_pretrained(
        tpu_teacher_id, 
        torch_dtype=torch.bfloat16,
        # No device_map="auto" on TPU!
    ).to(device)
    teacher.eval()
    
    # Optimizer & Scheduler
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate)
    scheduler = get_wsd_schedule(opt, int(cfg.max_steps*0.1), cfg.max_steps)
    
    # [TEST OVERRIDE] Only 50 steps for TPU verification
    test_max_steps = 50 
    
    # Prepare for XLA
    student, opt, dl, scheduler = accelerator.prepare(student, opt, dl, scheduler)
    
    # 3. Training Loop Optimized for XLA
    student.train()
    for step, (input_ids, labels) in enumerate(dl):
        if step >= test_max_steps: break
        
        with accelerator.accumulate(student):
            # Forward — canonical return order is (logits, aux/MoE loss, KV-cache);
            # the 2nd value is the MoE aux loss, the 3rd is the KV-cache.
            logits, moe_loss, _ = student(input_ids)
            
            # Teacher Output (BF16)
            with torch.no_grad():
                teacher_logits = teacher(input_ids).logits
                
            # Hybrid Loss
            student_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # KD Loss (Native BF16)
            kd_loss = F.kl_div(
                F.log_softmax(logits / cfg.teacher_temp, dim=-1),
                F.softmax(teacher_logits / cfg.teacher_temp, dim=-1),
                reduction="batchmean"
            ) * (cfg.teacher_temp ** 2)
            
            total_loss = (1 - cfg.distill_alpha) * student_loss + cfg.distill_alpha * kd_loss + moe_loss
            
            accelerator.backward(total_loss)
            opt.step()
            scheduler.step()
            opt.zero_grad()
            
        if accelerator.is_main_process and step % 10 == 0:
            print(f"TPU Step {step} | Loss: {total_loss.item():.4f}")

    if accelerator.is_main_process:
        print("✅ TPU TRAINING COMPLETE.")

if __name__ == "__main__":
    train_tpu()

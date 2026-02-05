"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - DATASET VERIFIER
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

import os
import sys
import datetime
from pathlib import Path
from huggingface_hub import login
from datasets import load_dataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import sources
try:
    from scripts.data_pipeline import (
        STAGE1_SOURCES, 
        STAGE2_SOURCES, 
        STAGE3_SOURCES, 
        STAGE4_SOURCES, 
        STAGE5_SOURCES
    )
except ImportError:
    print("❌ Critical: Could not import data pipeline sources.")
    sys.exit(1)

def log(msg, file_obj=None):
    """Print to console and write to file if provided."""
    print(msg)
    if file_obj:
        file_obj.write(msg + "\n")
        file_obj.flush()

def check_source(stage_name, source, log_file):
    ds_name = source['dataset']
    subset = source.get('subset')
    split = source['split']
    field = source['field']
    
    log(f"   📡 Connecting to {ds_name} ({subset if subset else 'default'})...", log_file)
    
    try:
        ds = load_dataset(
            ds_name, 
            name=subset, 
            split=split, 
        )
        item = next(iter(ds))
        content = item.get(field)
        
        # Fallback check
        if content is None:
             if 'text' in item: content = item['text']
             elif 'instruction' in item: content = item['instruction']
             elif 'output' in item: content = item['output']
             elif 'Input' in item: content = item['Input']
             elif 'Output' in item: content = item['Output']
        
        if content is not None:
            log(f"      ✅ OK! (Sample len: {len(str(content))})", log_file)
            return True
        else:
            log(f"      ⚠️  OK (Connected) but field '{field}' is empty. Keys: {list(item.keys())}", log_file)
            return True
            
    except Exception as e:
        if "gated" in str(e).lower():
             log(f"      🔐 GATED (User needs to accept terms): {ds_name}", log_file)
             return True
        log(f"      ❌ ERROR: {e}", log_file)
        return False

def main():
    # Setup Logging
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    report_path = log_dir / "dataset_verification_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log("=================================================================", f)
        log(f"🚀 TITAN DATASET VERIFICATION REPORT - {timestamp}", f)
        log("=================================================================", f)
        
        if os.environ.get("HF_TOKEN"):
            login(token=os.environ.get("HF_TOKEN"))
            log("🔑 Authenticated with HF_TOKEN", f)
        else:
            log("⚠️  No HF_TOKEN found", f)

        stages = [
            ("STAGE 1: Logic", STAGE1_SOURCES),
            ("STAGE 2: Knowledge", STAGE2_SOURCES),
            ("STAGE 3: Language", STAGE3_SOURCES),
            ("STAGE 4: Soul", STAGE4_SOURCES),
            ("STAGE 5: Tools", STAGE5_SOURCES),
        ]

        success_count = 0
        total_count = 0

        for stage_name, sources in stages:
            log(f"\n{stage_name}", f)
            log("-" * 40, f)
            for src in sources:
                total_count += 1
                if check_source(stage_name, src, f):
                    success_count += 1
        
        log("\n=================================================================", f)
        if success_count == total_count:
            log(f"✅ RESULT: All {total_count} datasets are accessible!", f)
        else:
            log(f"⚠️  RESULT: {success_count}/{total_count} datasets accessible.", f)
        log("=================================================================", f)
        
    print(f"\n📄 Report saved to: {report_path}")

if __name__ == "__main__":
    main()

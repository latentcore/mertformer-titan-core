"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - DATASET VERIFIER
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
import datetime
import argparse
from pathlib import Path
from huggingface_hub import login
from datasets import load_dataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_registry import get_hf_revision

# Import sources
try:
    from scripts.data_pipeline import (
        STAGE1_SOURCES,
        STAGE2_SOURCES,
        STAGE3_SOURCES,
        STAGE4_SOURCES,
        STAGE5_SOURCES,
        # [2026-07-29] Reuse the pipeline's OWN field extractor instead of a local
        # `item.get(field)`. A source's `field` is not always a plain string: it can be
        # a dict (`{"join": ["instruction", "output"]}` for MathInstruct/gsm8k) or a
        # list of candidate keys (`["text", "conversation", "messages"]` for the
        # Stage-4/5 instruction sources). `dict.get()` with an unhashable dict/list key
        # raises TypeError, which the broad `except Exception` below then reported as
        # "error" -- so 7 of the ~11 configured sources were falsely marked unreachable
        # even when they were perfectly fine. _extract_text handles all three shapes.
        _extract_text,
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

def check_source(source, log_file):
    # Returns an honest status string instead of a single True/False pass-flag:
    #   "ok"     -> connected AND the expected content field was actually present
    #   "empty"  -> connected but the field was empty (NOT a real content check)
    #   "gated"  -> dataset is gated; reachability unknown until terms accepted
    #   "error"  -> connection/loading failed
    # Note: only "ok" represents verified accessibility; "empty"/"gated" are
    # surfaced separately so the final summary is not a fake-green gate.
    ds_name = source['dataset']
    subset = source.get('subset')
    split = source['split']
    field = source['field']

    log(f"   📡 Connecting to {ds_name} ({subset if subset else 'default'})...", log_file)

    try:
        revision = get_hf_revision(ds_name)
        if revision:
            log(f"      📌 Revision pinned: {revision}", log_file)
        ds = load_dataset(
            ds_name,
            name=subset,
            split=split,
            revision=revision,
            # [2026-07-29] streaming=True is MANDATORY here, not an optimization.
            # Several configured sources are TB-scale (bigcode/the-stack-dedup,
            # HuggingFaceFW/fineweb-edu, uonlp/CulturaX); without streaming this call
            # tries to download the entire split just to inspect ONE row, filling the
            # disk and hanging for hours before failing. scripts/data_pipeline.py and
            # scripts/eval.py already open the same sources with streaming=True.
            streaming=True,
        )
        item = next(iter(ds))
        # Single-source extractor (see the import note): handles str / list-of-keys /
        # {"join": [...]} field shapes identically to the real pipeline.
        content = _extract_text(item, field) or None

        # Fallback check
        if content is None:
             if 'text' in item: content = item['text']
             elif 'instruction' in item: content = item['instruction']
             elif 'output' in item: content = item['output']
             elif 'Input' in item: content = item['Input']
             elif 'Output' in item: content = item['Output']

        if content is not None:
            log(f"      ✅ OK! (Sample len: {len(str(content))})", log_file)
            return "ok"
        else:
            log(f"      ⚠️  OK (Connected) but field '{field}' is empty. Keys: {list(item.keys())}", log_file)
            return "empty"

    except Exception as e:
        if "gated" in str(e).lower():
             log(f"      🔐 GATED (User needs to accept terms): {ds_name}", log_file)
             return "gated"
        log(f"      ❌ ERROR: {e}", log_file)
        return "error"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--login", action="store_true", help="Explicitly login to Hugging Face via HF_TOKEN")
    args = parser.parse_args()

    # Setup Logging
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    report_path = log_dir / "dataset_verification_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log("=================================================================", f)
        log(f"🚀 TITAN DATASET VERIFICATION REPORT - {timestamp}", f)
        log("=================================================================", f)
        
        if args.login:
            if os.environ.get("HF_TOKEN"):
                login(token=os.environ.get("HF_TOKEN"))
                log("🔑 Authenticated with HF_TOKEN (explicit --login)", f)
            else:
                log("⚠️  --login requested but HF_TOKEN not found (continuing unauthenticated)", f)
        else:
            log("ℹ️  HF login skipped (default offline-first). Use --login to authenticate.", f)

        stages = [
            ("STAGE 1: Logic", STAGE1_SOURCES),
            ("STAGE 2: Knowledge", STAGE2_SOURCES),
            ("STAGE 3: Language", STAGE3_SOURCES),
            ("STAGE 4: Soul", STAGE4_SOURCES),
            ("STAGE 5: Tools", STAGE5_SOURCES),
        ]

        success_count = 0   # only "ok": content actually verified
        empty_count = 0     # connected but field empty (not a real content check)
        gated_count = 0     # gated: reachability unknown until terms accepted
        error_count = 0     # failed to load
        total_count = 0

        for stage_name, sources in stages:
            log(f"\n{stage_name}", f)
            log("-" * 40, f)
            for src in sources:
                total_count += 1
                status = check_source(src, f)
                if status == "ok":
                    success_count += 1
                elif status == "empty":
                    empty_count += 1
                elif status == "gated":
                    gated_count += 1
                else:
                    error_count += 1

        log("\n=================================================================", f)
        # "All accessible" now means every source was content-verified ("ok").
        # Empty/gated are reported separately so this is not a fake-green gate.
        if success_count == total_count:
            log(f"✅ RESULT: All {total_count} datasets are content-verified accessible!", f)
        else:
            log(f"⚠️  RESULT: {success_count}/{total_count} datasets content-verified.", f)
        if empty_count or gated_count or error_count:
            log(
                f"   (empty field: {empty_count}, gated/pending terms: {gated_count}, errors: {error_count})",
                f,
            )
        log("=================================================================", f)
        
    print(f"\n📄 Report saved to: {report_path}")

if __name__ == "__main__":
    main()

"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - DATA PIPELINE
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

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

# Ensure project-local imports work when launched as: python scripts/data_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_registry import get_hf_revision


# =============================================================================
# TR: STAGE CONFİGÜRASYONU / EN: STAGE CONFIGURATION
# =============================================================================
STAGE_DIRS = {
    1: Path("datasets/stage1"),
    2: Path("datasets/stage2"),
    3: Path("datasets/stage3"),
    4: Path("datasets/stage4_soul"),
    5: Path("datasets/stage5_tools")
}

# TR: Stage 1: Saf Lójik (%45 toplam token)
# EN: Stage 1: Pure Logic (45% total tokens)
# TR: Kaynaklar birbirine göre ağırlıklı, toplam örneklere göre normalize edilir.
# EN: Sources weighted relative to each other, normalized by total samples.
STAGE1_SOURCES = [
    {
        "dataset": "bigcode/the-stack-v2",
        "split": "train",
        "field": "content",
        "ratio": 0.28,  # 28% of total
        "filters": ["python", "cpp", "asm"],  # Filter by language
        "min_length": 100,
        "max_length": 50000
    },
    {
        "dataset": "TIGER-Lab/MathInstruct", # [FIX] Nvidia repo removed, switched to TIGER-Lab
        "split": "train",
        "field": "instruction", # Fields: instruction, output
        "ratio": 0.015,  # 1.5% of total
        "filters": None,
        "min_length": 10,
        "max_length": 20000
    },
    {
        "dataset": "openai/gsm8k", 
        "split": "train",
        "subset": "main", # [FIX] Required config name
        "field": "question", 
        "ratio": 0.02, 
        "filters": None,
        "min_length": 10,
        "max_length": 2000
    }
]

# TR: Stage 2: Dünya Bilgisi (%35 toplam token)
# EN: Stage 2: World Knowledge (35% total tokens)
# TR: %35: HuggingFaceFW/fineweb-edu (Sadece Reasoning/Eğitici)
# EN: 35%: HuggingFaceFW/fineweb-edu (Reasoning/Educational only)
STAGE2_SOURCES = [
    {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "field": "text",
        "ratio": 0.35,  # 35% of total
        "filters": None,
        "min_length": 500,
        "max_length": 30000
    }
]

# TR: Stage 3: Kimlik ve Dil (%7 toplam token)
# EN: Stage 3: Identity & Language (7% total tokens)
# TR: %3.5: uonlp/CulturaX (Sadece Türkçe alt kümesi, filtrelenmiş)
# EN: 3.5%: uonlp/CulturaX (Turkish subset only, filtered)
# TR: %3.5: HuggingFaceTB/cosmopedia (Sentetik Yüksek Kalite)
# EN: 3.5%: HuggingFaceTB/cosmopedia (Synthetic High-Quality)
STAGE3_SOURCES = [
    {
        "dataset": "wikimedia/wikipedia", 
        "split": "train",
        "subset": "20231101.tr",
        "field": "text",
        "ratio": 0.015,  # Reduced to 1.5% (High Quality Clean)
        "filters": None,
        "min_length": 100,
        "max_length": 50000
    },
    {
        "dataset": "uonlp/CulturaX", # [NEW] Massive Web Scale Turkish
        "split": "train",
        "subset": "tr", # Turkish Subset
        "field": "text",
        "ratio": 0.040,  # 4% (The bulk of fluency comes from here)
        "filters": None,
        "min_length": 200,
        "max_length": 20000
    },
    {
        "dataset": "HuggingFaceTB/cosmopedia",
        "split": "train",
        "subset": "stories", # [FIX] Selected high-quality subset
        "field": "text",
        "ratio": 0.015,  
        "filters": None,
        "min_length": 400,
        "max_length": 20000
    }
]

# TR: Stage 4: Ruh ve Kimlik (%3 toplam token)
# EN: Stage 4: Soul & Identity (3% total tokens)
# TR: %1.5: OpenAssistant/oasst_top1_2023-08-25 (Yüksek Kalite İnsan Diyaloğu)
# EN: 1.5%: OpenAssistant/oasst_top1_2023-08-25 (High Quality Human Dialogue)
# TR: %1.5: mlabonne/guanaco-llama2-1k (Temiz Talimat Takibi)
# EN: 1.5%: mlabonne/guanaco-llama2-1k (Clean Instruction Following)
STAGE4_SOURCES = [
    {
        "dataset": "OpenAssistant/oasst_top1_2023-08-25",
        "split": "train",
        "field": "text",
        "ratio": 0.010, # 1%
        "filters": None,
        "min_length": 50,
        "max_length": 10000
    },
    {
        "dataset": "mlabonne/guanaco-llama2-1k",
        "split": "train",
        "field": "text",
        "ratio": 0.005, # 0.5%
        "filters": None,
        "min_length": 50,
        "max_length": 10000
    },
    {
        "dataset": "TFLai/Turkish-Alpaca", # [FIX] Replaced broken link with active repo
        "split": "train",
        "field": "output", # Targeting output for safety
        "ratio": 0.010, 
        "filters": None,
        "min_length": 10,
        "max_length": 5000
    },
    {
        "dataset": "turkish-nlp-suite/InstrucTurca", 
        "split": "train",
        "field": "Output", # [FIX] Case-sensitive field name found in verify
        "ratio": 0.010, 
        "filters": None,
        "min_length": 10,
        "max_length": 10000
    }
]

# TR: Stage 5: Araç Kullanımı (%10 toplam token)
# EN: Stage 5: Tool Use (10% total tokens)
# TR: %10: glaiveai/glaive-function-calling-v2 (Fonksiyon Çağırı/Araç Kullanımı)
# EN: 10%: glaiveai/glaive-function-calling-v2 (Function Calling/Tool Use)
STAGE5_SOURCES = [
    {
        "dataset": "glaiveai/glaive-function-calling-v2",
        "split": "train",
        "field": "text",
        "ratio": 0.10, # 10% of total
        "filters": None,
        "min_length": 100,
        "max_length": 15000
    }
]


# =============================================================================
# TR: YARDIMCI FONİKSİYONLAR / EN: HELPER FUNCTIONS
# =============================================================================
def create_stage_directories():
    """TR: Tüm stage dizinlerini oluştur. / EN: Create all stage directories."""
    for stage_dir in STAGE_DIRS.values():
        stage_dir.mkdir(parents=True, exist_ok=True)
    print("✅ Stage directories created")


def filter_by_language(text, language_codes):
    """TR: Basit dil algılama/filtreleme. / EN: Simple language detection/filtering."""
    if language_codes is None:
        return True
    
    # TR: Türkçe için, yaygın Türkçe karakterleri kontrol et
    # EN: For Turkish, check for common Turkish characters
    if "tr" in language_codes:
        turkish_chars = "çğıöşüÇĞIİÖŞÜ"
        if any(c in text for c in turkish_chars):
            return True
    
    # TR: Kod dilleri için, anahtar kelimeleri kontrol et
    # EN: For code languages, check for keywords
    if "python" in language_codes:
        if any(kw in text.lower() for kw in ["def ", "import ", "class ", "print("]):
            return True
    if "cpp" in language_codes:
        if any(kw in text for kw in ["#include", "int main", "std::", "namespace"]):
            return True
    if "asm" in language_codes:
        if any(kw in text.lower() for kw in ["mov ", "add ", "sub ", "call ", "push ", "pop "]):
            return True
    
    return False


def download_stage(stage_num, sources, target_samples_per_source):
    """TR: Verimli kalıcı streaming iterator'lar kullanarak belirli bir stage için veri indir. / EN: Download data for a specific stage using efficient persistent streaming iterators."""
    stage_dir = STAGE_DIRS[stage_num]
    stage_output = stage_dir / f"stage{stage_num}_data.jsonl"
    
    print(f"\n{'='*60}")
    print(f"📦 STAGE {stage_num}: Downloading data...")
    print(f"{'='*60}")
    
    # 1. Initialize File (Overwrite to prevent duplication)
    with open(stage_output, "w", encoding="utf-8") as f:
        pass  # Just clear the file
    # Create Start Signal for Streaming Tailing
    start_signal = stage_dir / f"stage{stage_num}_started.signal"
    with open(start_signal, "w") as f:
        f.write("STARTED")
    print(f"✅ Start Signal created: {start_signal}")

    # 2. Initialize Persistent Iterators (Crucial for Speed)
    # loading dataset once and iterating is O(1), vs reloading every sample O(N)
    iterators = {}
    for i, src in enumerate(sources):
        print(f"   🔌 Connecting to stream: {src['dataset']}...")
        try:
            revision = get_hf_revision(src["dataset"])
            if revision:
                print(f"      📌 Pinned revision: {revision}")
            # [FIX] Pass subset/config name if it exists (Crucial for Wikipedia)
            ds = load_dataset(
                src["dataset"],
                name=src.get("subset", None), # Pass config name (e.g. '20231101.tr')
                split=src["split"],
                revision=revision,
                streaming=True
            ).shuffle(seed=42, buffer_size=10_000) # Use shuffle buffer instead of expensive skip
            iterators[i] = iter(ds)
        except Exception as e:
            print(f"   ❌ Failed to connect {src['dataset']}: {e}")
            iterators[i] = None

    collected = 0
    source_collected = {i: 0 for i in range(len(sources))}
    
    # Calculate samples per source based on ratio
    total_ratio = sum(s["ratio"] for s in sources)
    samples_per_source = {
        i: int(target_samples_per_source * (s["ratio"] / total_ratio))
        for i, s in enumerate(sources)
    }
    
    pbar = tqdm(total=sum(samples_per_source.values()), desc=f"Stage {stage_num}")
    
    consecutive_failures = 0
    
    # [V27.6 IO FIX] Write Buffer Initialization
    write_buffer = []
    BUFFER_SIZE = 10_000 # Batch 10k writes to save SSD life and time
    
    while collected < sum(samples_per_source.values()):
        # Select source based on ratio
        r = random.random() * total_ratio
        cum = 0.0
        source_idx = 0
        for i, src in enumerate(sources):
            cum += src["ratio"]
            if r <= cum:
                source_idx = i
                break
        
        # Skip if this source is already complete or failed to init
        if source_collected[source_idx] >= samples_per_source[source_idx]:
             continue
        if iterators[source_idx] is None:
             # Try other sources if one is dead? For now just skip this iteration
             consecutive_failures += 1
             if consecutive_failures > 1000:
                 print("❌ All sources effectively exhausted or broken.")
                 break
             continue

        source = sources[source_idx]
        src_iter = iterators[source_idx]
        
        try:
            # Efficient O(1) fetch
            sample = next(src_iter)
            
            # Extract text
            text = sample.get(source["field"], "")
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            text = text.strip()
            
            # Validate length
            if not (source["min_length"] <= len(text) <= source["max_length"]):
                continue
            
            # Filter by language if needed
            if source["filters"] and not filter_by_language(text, source["filters"]):
                continue
            
            # [V27.6 IO FIX] Buffer Write Strategy
            write_buffer.append(json.dumps({"text": text}, ensure_ascii=False))
            
            if len(write_buffer) >= BUFFER_SIZE:
                with open(stage_output, "a", encoding="utf-8") as f:
                    f.write("\n".join(write_buffer) + "\n")
                write_buffer = [] # Flush
            
            collected += 1
            source_collected[source_idx] += 1
            pbar.update(1)
            consecutive_failures = 0
            
        except StopIteration:
            print(f"\n⚠️  Source exhausted: {source['dataset']}")
            iterators[source_idx] = None # Mark as dead
        except Exception as e:
            # Occasional network blip
            consecutive_failures += 1
            if consecutive_failures > 50:
                print(f"\n❌ Too many consecutive failures. Aborting stage.")
                break
    
    # [V27.6 IO FIX] Final Flush
    if write_buffer:
        with open(stage_output, "a", encoding="utf-8") as f:
            f.write("\n".join(write_buffer) + "\n")

    pbar.close()
    print(f"✅ Stage {stage_num} complete: {collected} samples collected")
    
    # Print breakdown
    for i, src in enumerate(sources):
        print(f"   - {src['dataset']}: {source_collected[i]} samples")
    
    # Create Signal File for Smart Runner
    signal_file = stage_dir / f"stage{stage_num}_done.signal"
    with open(signal_file, "w") as f:
        f.write("DONE")
    print(f"✅ Signal created: {signal_file}")

    return collected


# =============================================================================
# TR: ANA YÜRÜTME / EN: MAIN EXECUTION
# =============================================================================
def main(target_samples: int = 12_000_000, login_hf: bool = False) -> None:
    # [V27.6 FIX] Reproducibility
    random.seed(42)

    """
    TR: Titan Onyx Storm curriculum öğrenme veri indirme işlemini yürütür.
    EN: Executes Titan Onyx Storm curriculum learning data download.

    İşlem / Process:
    - Stage 1: Pure Logic (45%) - Code + Math
    - Stage 2: World Knowledge (35%) - Educational content
    - Stage 3: Identity & Language (7%) - Turkish + Synthetic
    - Stage 4: Soul (3%) - High-quality dialogue/instruction
    - Stage 5: Tool Use (10%) - Function calling/Tool use
    """
    print("="*60)
    print("🚀 TITAN ONYX STORM - CURRICULUM LEARNING DATA PIPELINE")
    print("="*60)
    
    # Optional HF login (never at import time)
    if login_hf:
        if os.environ.get("HF_TOKEN"):
            print("🔑 Authenticating with Hugging Face (explicit --login)...")
            from huggingface_hub import login as hf_login
            hf_login(token=os.environ.get("HF_TOKEN"))
        else:
            print("⚠️  --login requested but HF_TOKEN not found; continuing unauthenticated.")

    # Create directories
    create_stage_directories()
    
    # Configuration
    # Target sample budget (adjust based on your token budget and compute).
    # Approx tokens ~= samples * avg_tokens_per_sample (e.g., 12M * ~500 ~= ~6B tokens).
    # If your training plan targets higher total tokens, increase TARGET_SAMPLES accordingly.
    TARGET_SAMPLES = int(target_samples)
    
    # Stage 1: Pure Logic (45% = 450,000 samples)
    stage1_target = int(TARGET_SAMPLES * 0.45)
    download_stage(1, STAGE1_SOURCES, stage1_target)
    
    # Stage 2: World Knowledge (35% = 350,000 samples)
    stage2_target = int(TARGET_SAMPLES * 0.35)
    download_stage(2, STAGE2_SOURCES, stage2_target)
    
    # Stage 3: Identity & Language (7% = 70,000 samples)
    stage3_target = int(TARGET_SAMPLES * 0.07)
    download_stage(3, STAGE3_SOURCES, stage3_target)
    
    # Stage 4: Soul (3% = 30,000 samples)
    stage4_target = int(TARGET_SAMPLES * 0.03)
    download_stage(4, STAGE4_SOURCES, stage4_target)

    # Stage 5: Tool Use (10% = 100,000 samples)
    stage5_target = int(TARGET_SAMPLES * 0.10)
    download_stage(5, STAGE5_SOURCES, stage5_target)
    
    print(f"\n{'='*60}")
    print("✅ TITAN DATA PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Stage 1 (Logic):     {STAGE_DIRS[1]}")
    print(f"Stage 2 (Knowledge): {STAGE_DIRS[2]}")
    print(f"Stage 3 (Language):  {STAGE_DIRS[3]}")
    print(f"Stage 4 (Soul):      {STAGE_DIRS[4]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-samples", type=int, default=12_000_000)
    parser.add_argument("--login", action="store_true", help="Explicitly login to Hugging Face via HF_TOKEN")
    args = parser.parse_args()
    main(target_samples=args.target_samples, login_hf=args.login)

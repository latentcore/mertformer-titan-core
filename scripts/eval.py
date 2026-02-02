"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - BENCHMARK SUITE (EVAL)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Licensed under MIT License.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v27.0-FINAL (Locked & Sealed)
Status : PRODUCTION READY (LOCKED)
==============================================================================
"""

__version__ = "27.0-FINAL"
__author__ = "Mert"

import sys
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from model.transformers import MertFormer
from config.config import cfg

def evaluate_gsm8k(model, tokenizer, device, num_samples=100):
    print(f"\n📚 Evaluating GSM8K (Zero-Shot) - {num_samples} samples...")
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    
    correct = 0
    total = 0
    
    iterator = iter(ds)
    
    for _ in tqdm(range(num_samples)):
        try:
            sample = next(iterator)
            question = sample["question"]
            answer = sample["answer"]
            
            # Simple prompt
            prompt = f"Question: {question}\nAnswer:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids, 
                    max_new_tokens=128, 
                    temperature=0.001, # Greedy-ish
                    top_p=1.0, 
                    eos_token_id=tokenizer.eos_token_id
                )
            
            full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_ans = full_output[len(prompt):].strip()
            
            # Very basic extraction (looking for the number at the end)
            # This is a proxy evaluation, rigorous eval requires regex parsing
            if total < 3: # Print first 3
                print(f"\nQ: {question}\nTarget: {answer}\nPred: {generated_ans}\n{'-'*20}")
                
            total += 1
            # Placeholder "correctness" (manual check needed usually, but we assume perfect fit for now)
            # In real script, we'd use regex to find the number matches
        except StopIteration:
            break
            
    print(f"⚠️  GSM8K Eval incomplete (Need Regex/Parser). Running as generation check only.")
    print(f"✅ Generated {total} reasoning traces successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"⚙️  Device: {device}")
    
    # Load Dict
    save_dir = project_root / cfg.save_dir
    ckpt_path = save_dir / f"{cfg.model_name}_latest.pt" if args.ckpt == "latest" else Path(args.ckpt)
    
    print(f"📦 Loading Model: {ckpt_path}")
    model = MertFormer().to(device)
    
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        model.eval()
    else:
        print("❌ Checkpoint not found! Evaluating initialized weights (Random).")
        model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
    
    # Run Evals
    evaluate_gsm8k(model, tokenizer, device, args.samples)

if __name__ == "__main__":
    main()

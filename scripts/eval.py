"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - BENCHMARK SUITE (EVAL)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture (planned target: Samsung S25 NPU — UNVERIFIED)
         NOTE: Bu dosyada NPU/S25 ozel kodu veya olcumu YOKTUR; "Samsung S25 NPU"
         hedefi planlanmis/dogrulanmamis bir iddiadir, olculmus bir sonuc degildir.
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import sys
import re
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from model.transformers import MertFormer
from config.config import cfg

def _extract_gsm8k_number(text):
    """GSM8K cevap dizisinden son sayısal değeri çıkar.

    GSM8K gold cevapları '#### <sayı>' ile biter; üretilen metinde ise
    son geçen sayıyı proxy olarak alırız. Hiçbir sayı yoksa None döner.
    """
    if text is None:
        return None
    marker = text.rsplit("####", 1)
    candidate = marker[1] if len(marker) == 2 else text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", candidate)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def evaluate_gsm8k(model, tokenizer, device, num_samples=100):
    print(f"\n📚 Evaluating GSM8K (Zero-Shot) - {num_samples} samples...")
    from utils.dataset_registry import get_hf_revision
    revision = get_hf_revision("openai/gsm8k")
    if revision:
        print(f"   📌 Pinned revision: {revision}")
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True, revision=revision)
    
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
            
            # Slice by TOKEN count (prompt length), not decoded char length:
            # the tokenizer round-trip may not reproduce the prompt verbatim, so a
            # character-based slice on the decoded string can mis-cut the answer.
            prompt_len = input_ids.shape[1]
            generated_ans = tokenizer.decode(
                generated_ids[0][prompt_len:], skip_special_tokens=True
            ).strip()
            
            # Gerçek doğruluk ölçümü: hem gold cevaptan hem de üretilen
            # metinden son sayıyı çıkarıp karşılaştırıyoruz (proxy exact-match).
            pred_num = _extract_gsm8k_number(generated_ans)
            gold_num = _extract_gsm8k_number(answer)
            is_correct = (pred_num is not None and pred_num == gold_num)

            if total < 3: # Print first 3
                print(f"\nQ: {question}\nTarget: {answer}\nPred: {generated_ans}")
                print(f"(gold={gold_num} | pred={pred_num} | correct={is_correct})\n{'-'*20}")

            total += 1
            if is_correct:
                correct += 1
        except StopIteration:
            break

    accuracy = (correct / total) if total else 0.0
    print(f"✅ Generated {total} reasoning traces successfully.")
    print(f"📊 GSM8K accuracy (numeric proxy exact-match): {correct}/{total} = {accuracy:.4f}")
    print(f"ℹ️  Not: bu metrik son-sayı proxy eşleşmesidir; resmi GSM8K calc-annotation"
          f" değerlendirmesi değildir, geçme-kapısı olarak kullanmayın.")
    return {"correct": correct, "total": total, "accuracy": accuracy}

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
    from utils.tokenizer_resolver import load_tokenizer_from_identity, resolve_tokenizer

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        # TR: Tokenizer'ı checkpoint kimliğinden yükle; yoksa AÇIK hata.
        # EN: Load tokenizer from checkpoint identity; missing -> explicit error
        #     (no silent teacher fallback -> train/eval tokenizer must match).
        tokenizer = load_tokenizer_from_identity(checkpoint.get("tokenizer_id"))
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        model.eval()
    else:
        print("❌ Checkpoint not found! Evaluating initialized weights (Random).")
        tokenizer = resolve_tokenizer(cfg)
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

    # Run Evals
    evaluate_gsm8k(model, tokenizer, device, args.samples)

if __name__ == "__main__":
    main()

"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - CHAT INTERFACE
-------------------------------------------------------------------------------
Copyright 2026 Mert Yunlu
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: sourced from mertformer_sdk.__version__ (single source of truth,
         avoids hand-maintained "BUILD30-V2" fossil / version drift)
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

try:
    from mertformer_sdk import __version__  # single source of truth
except Exception:  # pragma: no cover - SDK optional at import time
    __version__ = "unknown"
__author__ = "Mert Yünlü"

import sys
import torch
import argparse
import traceback
from pathlib import Path

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from model.transformers import MertFormer
from config.config import cfg

def chat():
    parser = argparse.ArgumentParser(description="Chat with MertFormer Titan")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint path or 'latest'")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature (0.0-1.0)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P probability")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens to generate")
    args = parser.parse_args()

    print(f"\n🚀 MERTFORMER TITAN CHAT (Temp: {args.temp})")
    print(f"-------------------------------------------")

    # 1. Load Config & Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"⚙️  Device: {device}")
    model = MertFormer().to(device)
    model.eval()

    # 2. Load Checkpoint
    ckpt_path = None
    save_dir = project_root / cfg.save_dir
    
    if args.ckpt == "latest":
        ckpt_path = save_dir / f"{cfg.model_name}_latest.pt"
    else:
        ckpt_path = Path(args.ckpt)

    from utils.tokenizer_resolver import load_tokenizer_from_identity, resolve_tokenizer

    tokenizer = None
    try:
        if ckpt_path.exists():
            print(f"📦 Loading Checkpoint: {ckpt_path.name}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            # TR: Tokenizer'i checkpoint kimliğinden al (eval/demo ile aynı yol).
            # EN: Take the tokenizer from the checkpoint identity (same path as
            #     eval) so the demo decodes with the training tokenizer.
            tokenizer = load_tokenizer_from_identity(checkpoint.get("tokenizer_id"))
            cfg.vocab_size = len(tokenizer)
            model.resize_token_embeddings(len(tokenizer))
            # Handle both full checkpoint dict and clean state dict
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully.")
        else:
            print(f"⚠️  Checkpoint not found at {ckpt_path}")
            print("⚠️  Initializing with random weights (Untrained Mode)")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        traceback.print_exc()
        return

    # 3. Load Tokenizer (single resolver) when no checkpoint identity was used.
    if tokenizer is None:
        try:
            tokenizer = resolve_tokenizer(cfg)
            cfg.vocab_size = len(tokenizer)
            model.resize_token_embeddings(len(tokenizer))
        except Exception as e:
            print(f"❌ Tokenizer Error: {e}")
            return

    print("\n💬 CHAT READY. Type 'exit' to quit.\n")

    # 4. Chat Loop
    while True:
        try:
            user_input = input("👤 User: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            # Format input (Simple chat template)
            prompt = f"User: {user_input}\nAssistant:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            print("🤖 Model: ", end="", flush=True)

            # Generate
            with torch.no_grad():
                # Simple generation loop with streaming-like print
                generated_ids = input_ids
                past_kv = None
                
                for _ in range(args.max_tokens):
                    # Use cache handling logic similar to generate()
                    if past_kv is not None:
                        curr_input = generated_ids[:, -1:]
                    else:
                        curr_input = generated_ids

                    logits, _, past_kv = model(curr_input, past_key_values=past_kv, use_cache=True)
                    next_token_logits = logits[..., -1, :] / args.temp
                    
                    # Top-P Sampling
                    if args.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > args.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')

                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                    
                    # Decode single token
                    token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                    print(token_text, end="", flush=True)

                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
            
            print("\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    chat()

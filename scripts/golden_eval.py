"""
Golden sample eval for periodic logic checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_prompts(path: Path) -> List[Dict[str, str]]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def dry_run(prompts: List[Dict[str, str]]) -> None:
    categories = {}
    for p in prompts:
        cat = p.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"Golden samples: {len(prompts)}")
    for cat, count in sorted(categories.items()):
        print(f"- {cat}: {count}")


def run_model(prompts: List[Dict[str, str]], ckpt: str) -> None:
    import torch
    from config.config import cfg
    from model.transformers import MertFormer
    from utils.tokenizer_resolver import load_tokenizer_from_identity, resolve_tokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if Path(ckpt).exists():
        checkpoint = torch.load(ckpt, map_location=device)
        # TR: Tokenizer'i checkpoint kimliginden yukle; yoksa ACIK hata (sessiz
        #     teacher fallback YOK). EN: Tokenizer strictly from checkpoint
        #     identity; missing -> explicit error (same pattern as eval/gsm8k.py).
        tokenizer = load_tokenizer_from_identity(checkpoint.get("tokenizer_id"))
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint.get("model", checkpoint))
    else:
        # Smoke-only path (no checkpoint -> no recorded identity).
        print(f"[golden_eval] checkpoint not found ({ckpt}), using random weights (smoke).")
        tokenizer = resolve_tokenizer(cfg)
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    for item in prompts:
        prompt = item.get("prompt", "")
        if not prompt:
            continue
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            # [H6 fix] stop on EOS; decode only the generated tokens.
            output = model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=0.2,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            output[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        print("---")
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="datasets/golden_samples.jsonl")
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise RuntimeError(f"Golden sample file not found: {path}")

    prompts = load_prompts(path)
    if len(prompts) != 50:
        raise RuntimeError(f"Golden sample count must be 50, got {len(prompts)}")

    dry_run(prompts)

    if args.run_model:
        run_model(prompts, args.ckpt)

    print("Golden sample eval: PASS")


if __name__ == "__main__":
    main()

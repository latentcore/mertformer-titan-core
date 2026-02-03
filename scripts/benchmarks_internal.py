"""
Internal truth benchmarking runner for HumanEval and MBPP.
Generates model outputs for offline scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset_safe(name: str, config: str):
    from datasets import load_dataset
    try:
        return load_dataset(name, config, split="test")
    except Exception:
        return load_dataset(name, split="test")


def run_generation(dataset, tokenizer, model, device, out_path: Path, max_new_tokens: int, samples: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            if idx >= samples:
                break
            prompt = row.get("prompt") or row.get("text") or row.get("problem") or ""
            if not prompt:
                continue
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            import torch
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.2)
            completion = tokenizer.decode(output[0], skip_special_tokens=True)
            record = {
                "id": row.get("task_id", idx),
                "prompt": prompt,
                "completion": completion,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
        from config.config import cfg
        from model.transformers import MertFormer
    except Exception as exc:
        raise RuntimeError(f"Missing dependencies for benchmark run: {exc}")

    if not args.run:
        print("Benchmark runner configured. Use --run to execute.")
        return

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = MertFormer().to(device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        candidate = Path(cfg.save_dir) / f"{cfg.model_name}_latest.pt"
        if candidate.exists():
            ckpt_path = candidate

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    else:
        print(f"⚠️  Checkpoint not found: {ckpt_path}. Benchmarks will run on random weights.")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    humaneval = load_dataset_safe("openai_humaneval", "openai_humaneval")
    mbpp = load_dataset_safe("mbpp", "sanitized")

    run_generation(
        humaneval,
        tokenizer,
        model,
        device,
        Path("reports/benchmarks/humaneval_outputs.jsonl"),
        args.max_new_tokens,
        args.samples,
    )
    run_generation(
        mbpp,
        tokenizer,
        model,
        device,
        Path("reports/benchmarks/mbpp_outputs.jsonl"),
        args.max_new_tokens,
        args.samples,
    )

    print("Benchmarks generated.")


if __name__ == "__main__":
    main()

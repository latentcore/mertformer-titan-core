"""
Internal truth benchmarking runner for HumanEval and MBPP.
Generates model outputs for offline scoring.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset_safe(name: str, config: str):
    from datasets import load_dataset
    from utils.dataset_registry import get_hf_revision
    revision = get_hf_revision(name)
    try:
        return load_dataset(name, config, split="test", revision=revision)
    except Exception:
        return load_dataset(name, split="test", revision=revision)


def run_generation(dataset, tokenizer, model, device, out_path: Path, max_new_tokens: int, samples: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    limit = total if samples <= 0 else min(samples, total)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            if idx >= limit:
                break
            prompt = row.get("prompt") or row.get("text") or row.get("problem") or ""
            if not prompt:
                continue
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            import torch
            with torch.no_grad():
                # [H6 fix] stop on EOS; decode only the generated tokens.
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                output[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            record = {
                "id": row.get("task_id", idx),
                "prompt": prompt,
                "completion": completion,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return limit


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--allow-random", action="store_true", help="Run even if checkpoint is missing (random weights).")
    args = parser.parse_args()
    summary_path = Path("reports/benchmarks/internal_smoke_summary.json")

    try:
        import torch
        from transformers import AutoTokenizer
        from config.config import cfg
        from model.transformers import MertFormer
    except Exception as exc:
        raise RuntimeError(f"Missing dependencies for benchmark run: {exc}")

    if not args.run:
        write_summary(
            summary_path,
            {
                "status": "idle",
                "reason": "run flag not set",
            },
        )
        print("Benchmark runner configured. Use --run to execute.")
        return

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        candidate = Path(cfg.save_dir) / f"{cfg.model_name}_latest.pt"
        if candidate.exists():
            ckpt_path = candidate

    from utils.tokenizer_resolver import load_tokenizer_from_identity

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        # TR: Tokenizer'ı checkpoint kimliğinden yükle; yoksa AÇIK hata (sessiz
        #     teacher fallback YOK). EN: Load the tokenizer strictly from the
        #     checkpoint identity; missing -> explicit error. Same pattern as
        #     eval/gsm8k.py so this checkpoint-bound evidence path cannot decode
        #     with a different tokenizer than training used.
        tokenizer = load_tokenizer_from_identity(checkpoint.get("tokenizer_id"))
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint.get("model", checkpoint))
    else:
        if not args.allow_random:
            write_summary(
                summary_path,
                {
                    "status": "not_eligible",
                    "reason": "checkpoint_missing",
                    "checkpoint": str(ckpt_path),
                },
            )
            print(f"NOT ELIGIBLE FOR CLAIM: checkpoint not found: {ckpt_path}")
            print(
                "Reason: benchmark outputs without a trained checkpoint are not valid "
                "for external quality/performance claims."
            )
            return
        print(f"⚠️  Checkpoint not found: {ckpt_path}. Running on random weights (--allow-random).")
        # TR: Yalnizca smoke yolu: ogretmen tokenizer'i, yoksa offline test
        #     tokenizer'i. Bu fallback checkpoint decode yolunda DEGIL.
        # EN: Smoke-only tokenizer: teacher, else the offline test tokenizer.
        #     This fallback never runs on the checkpoint decode path.
        hf_token = os.environ.get("HF_TOKEN")
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id, token=hf_token)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    try:
        humaneval = load_dataset_safe("openai_humaneval", "openai_humaneval")
        mbpp = load_dataset_safe("mbpp", "sanitized")
    except Exception as exc:
        write_summary(
            summary_path,
            {
                "status": "skip",
                "reason": f"dataset_unavailable: {exc}",
            },
        )
        print(f"SKIP: benchmark datasets unavailable ({exc})")
        return

    humaneval_count = run_generation(
        humaneval,
        tokenizer,
        model,
        device,
        Path("reports/benchmarks/humaneval_outputs.jsonl"),
        args.max_new_tokens,
        args.samples,
    )
    mbpp_count = run_generation(
        mbpp,
        tokenizer,
        model,
        device,
        Path("reports/benchmarks/mbpp_outputs.jsonl"),
        args.max_new_tokens,
        args.samples,
    )

    write_summary(
        summary_path,
        {
            "status": "pass",
            "humaneval_outputs": int(humaneval_count),
            "mbpp_outputs": int(mbpp_count),
            "health_pass": bool(humaneval_count > 0 and mbpp_count > 0),
        },
    )
    print(f"Benchmarks generated. HumanEval={humaneval_count}, MBPP={mbpp_count}")


if __name__ == "__main__":
    main()

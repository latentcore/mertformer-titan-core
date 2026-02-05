"""GSM8K evaluator (lightweight, pre-training friendly).

Supports:
- --run: generate model outputs on GSM8K test split
  -> writes reports/benchmarks/gsm8k_outputs.jsonl
- --score-only: compute exact-match accuracy from an existing outputs file
  -> writes reports/benchmarks/gsm8k_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ANSWER_RE = re.compile(r"####\s*([-+]?[\d\.,]+)")
NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return cleaned or None


def _extract_gold(answer: str | None) -> str | None:
    if not answer:
        return None
    match = ANSWER_RE.search(answer)
    if match:
        return _normalize_number(match.group(1))
    return None


def _extract_pred(text: str | None) -> str | None:
    if not text:
        return None
    matches = NUM_RE.findall(text)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def _numbers_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except Exception:
        return a == b


def _load_dataset():
    from datasets import load_dataset
    try:
        return load_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        return load_dataset("openai/gsm8k", split="test")


def _load_model_and_tokenizer(ckpt: str):
    import torch
    from transformers import AutoTokenizer
    from config.config import cfg
    from model.transformers import MertFormer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = MertFormer().to(device)

    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        candidate = Path(cfg.save_dir) / f"{cfg.model_name}_latest.pt"
        if candidate.exists():
            ckpt_path = candidate

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    else:
        print(f"[gsm8k] Checkpoint not found: {ckpt_path}. Using random weights.")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def run_generation(
    out_path: Path,
    max_new_tokens: int,
    samples: int,
    ckpt: str,
) -> int:
    dataset = _load_dataset()
    model, tokenizer, device = _load_model_and_tokenizer(ckpt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    limit = total if samples <= 0 else min(samples, total)

    import torch
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            if idx >= limit:
                break
            question = row.get("question") or row.get("prompt") or row.get("input") or ""
            if not question:
                continue

            input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.2)
            completion = tokenizer.decode(output[0], skip_special_tokens=True)

            gold = _extract_gold(row.get("answer"))
            pred = _extract_pred(completion)
            correct = _numbers_match(pred, gold)

            record = {
                "id": row.get("id", idx),
                "question": question,
                "gold_answer": gold,
                "completion": completion,
                "pred_answer": pred,
                "correct": correct,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    return limit


def score_predictions(pred_path: Path, summary_path: Path) -> dict:
    total = 0
    correct = 0
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            gold = record.get("gold_answer") or _extract_gold(record.get("answer"))
            pred = record.get("pred_answer") or _extract_pred(record.get("completion"))
            is_correct = _numbers_match(pred, gold)
            total += 1
            if is_correct:
                correct += 1

    accuracy = (correct / total) if total else 0.0
    summary = {
        "status": "scored",
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Generate outputs.")
    parser.add_argument("--score-only", action="store_true", help="Score existing outputs.")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--output", type=str, default="reports/benchmarks/gsm8k_outputs.jsonl")
    parser.add_argument("--predictions", type=str, default="")
    args = parser.parse_args()

    out_path = Path(args.output)
    pred_path = Path(args.predictions) if args.predictions else out_path
    summary_path = out_path.parent / "gsm8k_summary.json"

    if args.score_only:
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
        summary = score_predictions(pred_path, summary_path)
        print(f"[gsm8k] Summary written to {summary_path} (accuracy={summary['accuracy']:.4f})")
        return

    if not args.run:
        print("[gsm8k] Use --run to generate outputs or --score-only to score existing outputs.")
        return

    generated = run_generation(out_path, args.max_new_tokens, args.samples, args.ckpt)
    summary = score_predictions(out_path, summary_path)
    print(f"[gsm8k] Generated {generated} examples. Accuracy={summary['accuracy']:.4f}")


if __name__ == "__main__":
    main()

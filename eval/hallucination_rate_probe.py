"""Checkpoint-bound hallucination-rate probe: closed-book factual QA match rate.

[2026-07-12] Part of the post-45K eval-readiness pass (BACKLOG I.7 #78:
"hallucination-rate olcumu"). A small built-in set of closed-book factual
questions with a known ground-truth answer string; the model is prompted
QA-style and its completion is checked for a case-insensitive substring
match against the expected answer (with a couple of accepted synonyms).
Non-match is counted as a hallucination/miss for this probe's purposes
(the miss could also be a formatting mismatch, not necessarily a fabricated
fact -- a real hallucination benchmark like TruthfulQA uses graded human/
model judging; this is a much smaller, offline, string-match proxy).
Reports a hallucination_rate = 1 - match_rate.

Usage:
    python eval/hallucination_rate_probe.py --checkpoint checkpoints/.../latest.pt

Output:
    reports/benchmarks/hallucination_rate_probe_summary.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval._probe_common import (  # noqa: E402
    git_commit,
    load_checkpoint_model,
    measurement_status,
    no_checkpoint_summary,
    resolve_checkpoint_or_none,
    sha256_file,
    utc_now,
    write_summary,
)

SCHEMA = "hallucination_rate_probe_v1"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "benchmarks" / "hallucination_rate_probe_summary.json"
CLAIM_BOUNDARY = (
    "Small built-in closed-book factual-QA string-match proxy (not graded human/"
    "model judging like TruthfulQA). A miss may be a formatting mismatch rather "
    "than a fabricated fact. NOT a full hallucination benchmark, and is not a "
    "capability or 'trained' claim."
)

# (question, accepted-answer-substrings)
QA_PAIRS = [
    ("What is the capital of France?", ("paris",)),
    ("What is the chemical symbol for water?", ("h2o", "h₂o")),
    ("How many continents are there on Earth?", ("seven", "7")),
    ("What planet is known as the Red Planet?", ("mars",)),
    ("Who wrote the play Romeo and Juliet?", ("shakespeare",)),
]


def compute_hallucination_rate(checkpoint: str, max_new_tokens: int, allow_random_weights: bool) -> dict[str, Any]:
    import torch

    model, tokenizer, device = load_checkpoint_model(checkpoint, allow_random_weights)

    per_question = []
    correct = 0
    for question, accepted in QA_PAIRS:
        prompt = f"Q: {question}\nA:"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.7, top_k=50, top_p=0.9)
        completion = tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=True).lower()
        matched = any(ans in completion for ans in accepted)
        if matched:
            correct += 1
        per_question.append({"question": question, "completion": completion.strip(), "matched_expected": matched})

    match_rate = correct / len(QA_PAIRS) if QA_PAIRS else None
    ckpt_path = Path(checkpoint)
    return {
        "schema": SCHEMA,
        "status": measurement_status(checkpoint, allow_random_weights),
        "generated_at_utc": utc_now(),
        "commit": git_commit(),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "max_new_tokens": max_new_tokens,
        "questions": per_question,
        "match_rate": round(match_rate, 4) if match_rate is not None else None,
        "hallucination_rate_proxy": round(1 - match_rate, 4) if match_rate is not None else None,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint-bound closed-book factual-QA hallucination-rate proxy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--allow-random-weights", action="store_true")
    args = parser.parse_args(argv)

    resolved = resolve_checkpoint_or_none(args.checkpoint)
    if resolved is None and not args.allow_random_weights:
        write_summary(no_checkpoint_summary(SCHEMA, args.checkpoint, CLAIM_BOUNDARY), Path(args.out))
        return 0

    summary = compute_hallucination_rate(args.checkpoint, int(args.max_new_tokens), bool(args.allow_random_weights))
    write_summary(summary, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

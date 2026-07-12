"""Checkpoint-bound adversarial prompt-robustness probe.

[2026-07-12] Part of the post-45K eval-readiness pass (BACKLOG I.7 #73/#85,
deduplicated into one probe: "adversarial robustness & auditability" +
"adversarial-robustness testi" were the same request twice).

Methodology: invariance-under-perturbation. For a small fixed prompt set,
generate a baseline completion, then generate again after applying
deterministic, reproducible perturbations (char-swap typos, case-flip,
whitespace-jitter -- no external library, no network) to the SAME prompt.
Similarity between baseline and perturbed completions (token-level Jaccard
overlap) is the robustness signal: a robust model's meaning should not swing
wildly from a typo. This is a standard, cheap invariance-testing proxy, not a
full red-team/jailbreak battery -- labeled as such.

Usage:
    python eval/adversarial_prompt_robustness.py --checkpoint checkpoints/.../latest.pt

Output:
    reports/benchmarks/adversarial_prompt_robustness_summary.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

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

SCHEMA = "adversarial_prompt_robustness_v1"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "benchmarks" / "adversarial_prompt_robustness_summary.json"
CLAIM_BOUNDARY = (
    "Invariance-under-perturbation proxy on a small fixed built-in prompt set "
    "(char-swap / case-flip / whitespace-jitter). This is NOT a red-team, "
    "jailbreak, or full adversarial-robustness battery, and is not a "
    "capability or 'trained' claim."
)

BASE_PROMPTS = [
    "The capital of France is",
    "def add(a, b):\n    return",
    "Once upon a time, there was a",
]


def _perturb(text: str) -> List[str]:
    variants = []
    if len(text) >= 4:
        chars = list(text)
        # deterministic adjacent-char swap near the middle
        i = len(chars) // 2
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        variants.append("".join(chars))
    variants.append(text.swapcase())
    variants.append("  " + text.replace(" ", "  "))
    return variants


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def compute_robustness(checkpoint: str, max_new_tokens: int, allow_random_weights: bool) -> dict[str, Any]:
    import torch

    model, tokenizer, device = load_checkpoint_model(checkpoint, allow_random_weights)

    def complete(prompt: str) -> List[str]:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=1.0, top_k=50, top_p=0.9)
        new_tokens = out[0, ids.size(1):].tolist()
        return tokenizer.convert_ids_to_tokens(new_tokens)

    per_prompt = []
    similarities = []
    for prompt in BASE_PROMPTS:
        baseline_tokens = complete(prompt)
        variant_scores = []
        for variant in _perturb(prompt):
            variant_tokens = complete(variant)
            sim = _jaccard(baseline_tokens, variant_tokens)
            variant_scores.append(round(sim, 4))
            similarities.append(sim)
        per_prompt.append({"prompt": prompt, "variant_similarities": variant_scores})

    mean_similarity = sum(similarities) / len(similarities) if similarities else None
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
        "prompts": per_prompt,
        "mean_completion_similarity": round(mean_similarity, 4) if mean_similarity is not None else None,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint-bound adversarial prompt-robustness probe.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--allow-random-weights", action="store_true")
    args = parser.parse_args(argv)

    resolved = resolve_checkpoint_or_none(args.checkpoint)
    if resolved is None and not args.allow_random_weights:
        write_summary(no_checkpoint_summary(SCHEMA, args.checkpoint, CLAIM_BOUNDARY), Path(args.out))
        return 0

    summary = compute_robustness(args.checkpoint, int(args.max_new_tokens), bool(args.allow_random_weights))
    write_summary(summary, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

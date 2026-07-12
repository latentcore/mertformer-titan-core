#!/usr/bin/env python3
"""
Random-init baseline measurement tool.

[2026-07-12] BACKLOG I.5 #45: "Random-init baseline'i OLC (hardcoded 0.80/0.50'yi
degistir)". eval/generalization_suite.py's baseline_pass_rate=0.80 and
eval/agentic_suite.py's baseline_completion_rate=0.50 are both explicitly
labeled in-code as UNMEASURED placeholders (see their own docstring/comment).
This script actually measures what a genuinely random-init (untrained)
MertFormer scores on both suites, by wrapping model.generate() as the same
Callable[[str], str] "responder" both suites already accept -- no new eval
methodology, just a real value plugged into the same harness.

DELIBERATELY DOES NOT auto-rewrite the 0.80/0.50 constants in
generalization_suite.py / agentic_suite.py: whether "baseline" there means
"random-init floor" or "aspirational gate threshold" is a real semantic
question (the in-code comment reads more like the latter -- "used both as
the reference baseline and as the gate threshold"), and silently changing a
gate-threshold's meaning is a bigger decision than this script should make
unilaterally. This script's job is to make the real number available; a
human decides whether/how to use it.

Usage:
    python scripts/measure_random_init_baseline.py
    MERTFORMER_MODEL_CONFIG=mertformer_small.yaml TITAN_USE_TR_TOKENIZER=1 \\
        python scripts/measure_random_init_baseline.py   # fast CPU smoke

Output:
    reports/benchmarks/random_init_baseline_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval._probe_common import git_commit, utc_now  # noqa: E402
from eval.agentic_suite import evaluate_with_callable as evaluate_agentic  # noqa: E402
from eval.generalization_suite import evaluate_with_callable as evaluate_generalization  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "reports" / "benchmarks" / "random_init_baseline_summary.json"


def _build_responder(max_new_tokens: int):
    import torch

    from config.config import cfg
    from model.transformers import MertFormer
    from utils.tokenizer_resolver import resolve_tokenizer

    tokenizer = resolve_tokenizer(cfg)
    cfg.vocab_size = len(tokenizer)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = MertFormer().to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    def responder(prompt: str) -> str:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=1.0, top_k=50, top_p=0.9)
        return tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=True)

    return responder, device


def measure(max_new_tokens: int) -> dict[str, Any]:
    responder, device = _build_responder(max_new_tokens)

    generalization = evaluate_generalization(responder)
    agentic = evaluate_agentic(responder)

    measured_pass_rate = float(generalization.get("pass_rate", 0.0))
    measured_completion_rate = float(agentic.get("completion_rate", 0.0))

    return {
        "schema": "random_init_baseline_v1",
        "status": "measured_random_init",
        "generated_at_utc": utc_now(),
        "commit": git_commit(),
        "device": str(device),
        "measured_generalization_pass_rate": round(measured_pass_rate, 4),
        "hardcoded_placeholder_in_generalization_suite": 0.80,
        "measured_agentic_completion_rate": round(measured_completion_rate, 4),
        "hardcoded_placeholder_in_agentic_suite": 0.50,
        "note": (
            "These are REAL measured scores from a genuinely random-init (untrained) "
            "MertFormer, produced by the exact same evaluate_with_callable() harness "
            "each suite's own gate_pass logic uses. Whether to replace the 0.80/0.50 "
            "constants with these numbers (as a floor) or keep them as an aspirational "
            "gate threshold is a deliberate follow-up decision, not made by this script."
        ),
        "claim_boundary": (
            "A random-init model's score on these two small toy suites, nothing more. "
            "Not a capability, benchmark, or 'trained' claim."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the real random-init baseline for eval suite gate thresholds.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    summary = measure(int(args.max_new_tokens))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[measure_random_init_baseline] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

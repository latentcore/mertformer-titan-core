"""Checkpoint-bound membership-inference probe: loss-based MIA proxy.

[2026-07-12] Part of the post-45K eval-readiness pass (BACKLOG I.7 #81:
"model-inversion + membership-inference risk analizi").

Methodology: the standard loss-based membership-inference-attack heuristic
(Yeom et al. 2018) -- a model tends to assign lower loss to examples it was
trained on than to genuinely unseen examples, so per-example loss can be used
as an attack signal (threshold at the median loss; "attack accuracy" is how
well that threshold separates the two groups).

IMPORTANT SCOPE NOTE: this repo does not ship the actual 45K training corpus
(it is fetched/precomputed at train time, not committed), so this probe
CANNOT compare against real train-set members. It compares two DETERMINISTIC
HALVES of the same held-out validation corpus as a synthetic stand-in
(first half vs. second half, split by line index) purely to exercise the
attack-accuracy methodology end-to-end. Once real train-set samples exist
(post-45K), swap --member-corpus to point at an actual training-data sample
for a real reading. Until then this reports a null-hypothesis sanity check
(attack accuracy should be ~50% on two halves of the SAME unseen corpus,
since neither half was in training) -- an unexpectedly high accuracy here
would itself indicate a probe bug, not a real leak.

Usage:
    python eval/membership_inference_probe.py --checkpoint checkpoints/.../latest.pt

Output:
    reports/benchmarks/membership_inference_probe_summary.json
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

SCHEMA = "membership_inference_probe_v1"
DEFAULT_CORPUS = PROJECT_ROOT / "datasets" / "validation.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "benchmarks" / "membership_inference_probe_summary.json"
CLAIM_BOUNDARY = (
    "Loss-based membership-inference-attack proxy (Yeom et al. 2018). Without "
    "--member-corpus pointing at a real training-data sample, this compares two "
    "halves of the SAME held-out corpus as a methodology sanity check (expected "
    "attack accuracy ~50%), NOT a real membership-leak measurement. Not a "
    "capability or 'trained' claim."
)


def _sequence_losses(model, tokenizer, device, rows: List[str], max_seq_len: int) -> List[float]:
    import torch
    import torch.nn.functional as F

    losses = []
    with torch.no_grad():
        for text in rows:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids.to(device)
            if ids.size(1) < 2:
                continue
            logits, _, _ = model(ids, use_cache=False)
            shift_logits = logits[:, :-1, :].float()
            shift_labels = ids[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            losses.append(float(loss.item()))
    return losses


def _attack_accuracy(member_losses: List[float], nonmember_losses: List[float]) -> float:
    """Threshold at the pooled median; predict 'member' if loss < threshold."""
    pooled = sorted(member_losses + nonmember_losses)
    if not pooled:
        return 0.5
    threshold = pooled[len(pooled) // 2]
    correct = sum(1 for l in member_losses if l < threshold)
    correct += sum(1 for l in nonmember_losses if l >= threshold)
    total = len(member_losses) + len(nonmember_losses)
    return correct / total if total else 0.5


def compute_membership_inference(
    checkpoint: str,
    corpus: Path,
    member_corpus: Path,
    max_rows: int,
    allow_random_weights: bool,
) -> dict[str, Any]:
    from config.config import cfg

    model, tokenizer, device = load_checkpoint_model(checkpoint, allow_random_weights)
    max_seq_len = int(cfg.max_seq_len)

    def _texts(path: Path) -> List[str]:
        import json

        from train.packing import extract_row_text

        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = extract_row_text(obj) if isinstance(obj, dict) else None
                if text:
                    out.append(text)
        return out

    synthetic_split = member_corpus == corpus
    all_texts = _texts(corpus)
    if synthetic_split:
        half = len(all_texts) // 2
        member_texts = all_texts[:half][:max_rows]
        nonmember_texts = all_texts[half:][:max_rows]
    else:
        member_texts = _texts(member_corpus)[:max_rows]
        nonmember_texts = all_texts[:max_rows]

    member_losses = _sequence_losses(model, tokenizer, device, member_texts, max_seq_len)
    nonmember_losses = _sequence_losses(model, tokenizer, device, nonmember_texts, max_seq_len)
    accuracy = _attack_accuracy(member_losses, nonmember_losses)

    ckpt_path = Path(checkpoint)
    return {
        "schema": SCHEMA,
        "status": measurement_status(checkpoint, allow_random_weights),
        "synthetic_split": synthetic_split,
        "generated_at_utc": utc_now(),
        "commit": git_commit(),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "member_count": len(member_losses),
        "nonmember_count": len(nonmember_losses),
        "mean_member_loss": round(sum(member_losses) / len(member_losses), 4) if member_losses else None,
        "mean_nonmember_loss": round(sum(nonmember_losses) / len(nonmember_losses), 4) if nonmember_losses else None,
        "attack_accuracy": round(accuracy, 4),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint-bound loss-based membership-inference probe.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument(
        "--member-corpus",
        default=str(DEFAULT_CORPUS),
        help="Real training-data sample once available; defaults to --corpus (synthetic split).",
    )
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--allow-random-weights", action="store_true")
    args = parser.parse_args(argv)

    resolved = resolve_checkpoint_or_none(args.checkpoint)
    if resolved is None and not args.allow_random_weights:
        write_summary(no_checkpoint_summary(SCHEMA, args.checkpoint, CLAIM_BOUNDARY), Path(args.out))
        return 0

    summary = compute_membership_inference(
        checkpoint=args.checkpoint,
        corpus=Path(args.corpus),
        member_corpus=Path(args.member_corpus),
        max_rows=int(args.max_rows),
        allow_random_weights=bool(args.allow_random_weights),
    )
    write_summary(summary, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

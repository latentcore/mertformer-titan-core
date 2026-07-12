"""Checkpoint-bound calibration probe: next-token Expected Calibration Error (ECE).

[2026-07-12] Part of the post-45K eval-readiness pass (BACKLOG I.7 #71:
"calibrated uncertainty / abstention"). Standard ECE (Guo et al. 2017):
bucket the model's own max-softmax confidence on each real next-token
prediction into M equal-width bins, then

    ECE = sum_b (|bin_b| / N) * |accuracy(bin_b) - avg_confidence(bin_b)|

A perfectly calibrated model has confidence == accuracy in every bin (ECE=0).
This is a real, standard, offline-computable metric -- but it is a SMALL,
single-corpus proxy, not a full calibration/abstention battery. Reuses the
repo's deterministic packer (train/packing.iter_packed_sequences), the same
one eval/held_out_ppl.py uses, so token boundaries are identical.

Usage:
    python eval/calibration_ece.py --checkpoint checkpoints/.../latest.pt
    python eval/calibration_ece.py --checkpoint checkpoints/does_not_exist.pt   # -> SKIPPED

Output:
    reports/benchmarks/calibration_ece_summary.json
"""
from __future__ import annotations

import argparse
import json
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

SCHEMA = "calibration_ece_v1"
DEFAULT_CORPUS = PROJECT_ROOT / "datasets" / "validation.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "benchmarks" / "calibration_ece_summary.json"
CLAIM_BOUNDARY = (
    "Next-token Expected Calibration Error (Guo et al. 2017) on a fixed held-out "
    "corpus with the repo's deterministic packer. This is a small single-corpus "
    "calibration proxy, NOT a full uncertainty/abstention evaluation battery, and "
    "is not a capability or 'trained' claim."
)


def compute_ece(
    checkpoint: str,
    corpus: Path,
    max_sequences: int,
    num_bins: int,
    allow_random_weights: bool,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from config.config import cfg
    from eval.held_out_ppl import _corpus_rows
    from train.packing import iter_packed_sequences

    model, tokenizer, device = load_checkpoint_model(checkpoint, allow_random_weights)
    max_seq_len = int(cfg.max_seq_len)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

    bin_conf_sum = [0.0] * num_bins
    bin_correct = [0] * num_bins
    bin_count = [0] * num_bins
    total = 0
    seq_count = 0

    with torch.no_grad():
        for seq in iter_packed_sequences(_corpus_rows(corpus), tokenizer, max_seq_len, eos_id, pad_id):
            if max_sequences > 0 and seq_count >= max_sequences:
                break
            true_len = int(seq["true_len"])
            if true_len < 2:
                continue
            input_ids = torch.tensor([seq["input_ids"]], dtype=torch.long, device=device)
            logits, _, _ = model(input_ids, use_cache=False)
            probs = F.softmax(logits[:, :-1, :].float(), dim=-1)
            labels = input_ids[:, 1:]
            positions = torch.arange(1, input_ids.size(1), device=device)
            mask = (positions < true_len).reshape(-1)

            probs_flat = probs.reshape(-1, probs.size(-1))[mask]
            labels_flat = labels.reshape(-1)[mask]
            if probs_flat.numel() == 0:
                seq_count += 1
                continue

            confidence, prediction = probs_flat.max(dim=-1)
            correct = (prediction == labels_flat).float()

            for c, ok in zip(confidence.tolist(), correct.tolist()):
                b = min(int(c * num_bins), num_bins - 1)
                bin_conf_sum[b] += c
                bin_correct[b] += int(ok)
                bin_count[b] += 1
                total += 1
            seq_count += 1

    if total == 0:
        raise RuntimeError("Calibration ECE: zero scorable tokens -- corpus empty or too short.")

    ece = 0.0
    bins_report = []
    for b in range(num_bins):
        n = bin_count[b]
        if n == 0:
            bins_report.append({"bin": b, "count": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = bin_conf_sum[b] / n
        acc = bin_correct[b] / n
        ece += (n / total) * abs(acc - avg_conf)
        bins_report.append({"bin": b, "count": n, "avg_confidence": round(avg_conf, 4), "accuracy": round(acc, 4)})

    ckpt_path = Path(checkpoint)
    return {
        "schema": SCHEMA,
        "status": measurement_status(checkpoint, allow_random_weights),
        "generated_at_utc": utc_now(),
        "commit": git_commit(),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "corpus": str(corpus.relative_to(PROJECT_ROOT)) if corpus.is_relative_to(PROJECT_ROOT) else str(corpus),
        "sequences": seq_count,
        "total_tokens_scored": total,
        "num_bins": num_bins,
        "ece": round(ece, 6),
        "bins": bins_report,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint-bound next-token calibration (ECE) probe.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--allow-random-weights", action="store_true")
    args = parser.parse_args(argv)

    resolved = resolve_checkpoint_or_none(args.checkpoint)
    if resolved is None and not args.allow_random_weights:
        summary = no_checkpoint_summary(SCHEMA, args.checkpoint, CLAIM_BOUNDARY)
        write_summary(summary, Path(args.out))
        return 0

    summary = compute_ece(
        checkpoint=args.checkpoint,
        corpus=Path(args.corpus),
        max_sequences=int(args.max_sequences),
        num_bins=int(args.num_bins),
        allow_random_weights=bool(args.allow_random_weights),
    )
    write_summary(summary, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

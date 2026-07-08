"""Held-out perplexity harness (checkpoint-bound).

[2026-07-08] Built from scratch for the pre-45K stabilization pass. BACKLOG.md's
2026-07-02 run-feedback lists a held-out perplexity harness as a hard prerequisite for
the 45K run: without it there is no checkpoint-bound quality signal at all, only the
training loss — and a training loss can fall while the model memorizes (the Kaggle smoke
reached ppl 1.19 by memorizing 2000 steps of the same data).

Discipline (mirrors eval/gsm8k.py and scripts/golden_score.py):
  * checkpoint-mandatory: the tokenizer is loaded from the checkpoint's recorded identity.
    A missing checkpoint is an ERROR, never a silent fallback to the teacher tokenizer.
    `--allow-random-weights` is the single explicit, opt-in escape hatch for smoke checks.
  * deterministic: fixed corpus + fixed seed + the repo's ONE shared packer
    (train/packing.iter_packed_sequences), the same pure function that keeps teacher
    logits and student tokens byte-aligned. No RNG, no batch-boundary effects.
  * honest labeling: the summary records `status`, the commit, the corpus/checkpoint
    hashes and an explicit claim boundary. A perplexity number is NOT a capability claim.

Usage:
    python eval/held_out_ppl.py --ckpt checkpoints/.../mertformer_titan_prod_final.pt
    python eval/held_out_ppl.py --ckpt none --allow-random-weights --max-sequences 2

Output:
    reports/benchmarks/held_out_ppl_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CORPUS = PROJECT_ROOT / "datasets" / "validation.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "benchmarks" / "held_out_ppl_summary.json"
SCHEMA = "held_out_ppl_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except OSError:
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _corpus_rows(path: Path) -> Iterator[Tuple[int, str]]:
    """Yield (raw_line_index, text) exactly as the training packer expects."""
    from train.packing import extract_row_text

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            text = extract_row_text(obj)
            if text:
                yield idx, text


def compute_held_out_ppl(
    ckpt: str,
    corpus: Path,
    max_sequences: int,
    seed: int,
    allow_random_weights: bool,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from config.config import cfg
    from train.packing import iter_packed_sequences

    # Reuse eval/gsm8k.py's loader verbatim: ONE checkpoint/tokenizer-identity discipline
    # for the whole eval/ package (no silent teacher-tokenizer fallback).
    from eval.gsm8k import _load_model_and_tokenizer

    if not corpus.exists():
        raise FileNotFoundError(f"Held-out corpus not found: {corpus}")

    torch.manual_seed(int(seed))
    model, tokenizer, device = _load_model_and_tokenizer(
        ckpt, allow_random_weights=allow_random_weights
    )
    model.eval()

    max_seq_len = int(cfg.max_seq_len)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

    nll_sum = 0.0
    token_count = 0
    seq_count = 0

    with torch.no_grad():
        for seq in iter_packed_sequences(
            _corpus_rows(corpus), tokenizer, max_seq_len, eos_id, pad_id
        ):
            if max_sequences > 0 and seq_count >= max_sequences:
                break

            true_len = int(seq["true_len"])
            if true_len < 2:
                continue  # nothing to predict

            input_ids = torch.tensor([seq["input_ids"]], dtype=torch.long, device=device)
            logits, _, _ = model(input_ids, use_cache=False)

            # Next-token prediction: target position t is predicted from position t-1.
            # Only real (non-pad) targets count, i.e. t in [1, true_len - 1].
            shift_logits = logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            positions = torch.arange(1, input_ids.size(1), device=device)
            mask = positions < true_len

            token_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            )
            masked = token_nll[mask.reshape(-1)]
            nll_sum += float(masked.sum().item())
            token_count += int(masked.numel())
            seq_count += 1

    if token_count == 0:
        raise RuntimeError(
            "Held-out ppl: zero scorable tokens — corpus empty or every sequence too short."
        )

    mean_nll = nll_sum / token_count
    # exp() of a large mean NLL overflows to inf for an untrained model; report it honestly
    # rather than crashing or clamping to a prettier number.
    try:
        ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    except OverflowError:
        ppl = float("inf")

    ckpt_path = Path(ckpt)
    return {
        "schema": SCHEMA,
        "status": "random_init_smoke" if allow_random_weights and not ckpt_path.exists() else "measured",
        "generated_at_utc": _utc_now(),
        "commit": _git_commit(),
        "device": str(device),
        "seed": int(seed),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256(ckpt_path),
        "corpus": str(corpus.relative_to(PROJECT_ROOT)) if corpus.is_relative_to(PROJECT_ROOT) else str(corpus),
        "corpus_sha256": _sha256(corpus),
        "tokenizer_id": getattr(tokenizer, "name_or_path", None),
        "vocab_size": int(len(tokenizer)),
        "max_seq_len": max_seq_len,
        "sequences": int(seq_count),
        "total_tokens": int(token_count),
        "nll_sum": round(nll_sum, 6),
        "mean_nll": round(mean_nll, 6),
        "ppl": ppl if ppl == float("inf") else round(ppl, 6),
        "claim_boundary": (
            "Held-out next-token perplexity on a fixed corpus with the repo's deterministic "
            "packer. This is NOT a benchmark, capability, or 'trained' claim; it is one "
            "checkpoint-bound number and is only meaningful next to the checkpoint SHA above."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint-bound held-out perplexity.")
    parser.add_argument("--ckpt", required=True, help="Path to a training checkpoint (.pt).")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Held-out JSONL corpus.")
    parser.add_argument("--max-sequences", type=int, default=64, help="0 = whole corpus.")
    parser.add_argument("--seed", type=int, default=1453)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--allow-random-weights",
        action="store_true",
        help="Explicit opt-in: score an untrained model (smoke check only, never a claim).",
    )
    args = parser.parse_args(argv)

    summary = compute_held_out_ppl(
        ckpt=args.ckpt,
        corpus=Path(args.corpus),
        max_sequences=int(args.max_sequences),
        seed=int(args.seed),
        allow_random_weights=bool(args.allow_random_weights),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\n[held_out_ppl] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

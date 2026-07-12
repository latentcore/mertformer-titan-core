"""Shared plumbing for the checkpoint-bound eval/*_probe.py harnesses added
2026-07-12 (calibration, adversarial robustness, bias/fairness, toxicity,
hallucination-rate, membership-inference).

Discipline (same as eval/held_out_ppl.py, eval/gsm8k.py):
  * --checkpoint is required, but a MISSING checkpoint is a graceful SKIPPED
    status (reason_code=NO_CHECKPOINT), not a crash -- these probes exist to
    be wired into automation *before* a real checkpoint exists, and must not
    break a pipeline that calls them speculatively.
  * When a checkpoint IS present, real methodology runs against it -- these
    are small, offline, honestly-scoped proxies (documented per-probe), never
    a claim of frontier-lab-battery coverage.
  * Every summary carries a claim_boundary string so a bare number can never
    be quoted without its scope.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def no_checkpoint_summary(schema: str, checkpoint: str, claim_boundary: str) -> dict:
    return {
        "schema": schema,
        "status": "SKIPPED",
        "reason_code": "NO_CHECKPOINT",
        "generated_at_utc": utc_now(),
        "commit": git_commit(),
        "checkpoint": str(checkpoint),
        "message": f"Checkpoint not found at {checkpoint!r}; probe not run. This is expected pre-45K.",
        "claim_boundary": claim_boundary,
    }


def write_summary(summary: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[{summary.get('schema', 'probe')}] wrote {out_path}")


def measurement_status(checkpoint: str, allow_random_weights: bool) -> str:
    """Mirrors eval.held_out_ppl.compute_held_out_ppl's honest status labeling:
    a --allow-random-weights run against a genuinely missing checkpoint is
    "random_init_smoke", never "measured" -- a random-init number must never
    be mistaken for a real measurement in a saved summary."""
    return "random_init_smoke" if allow_random_weights and resolve_checkpoint_or_none(checkpoint) is None else "measured"


def resolve_checkpoint_or_none(ckpt: str) -> Optional[Path]:
    """Mirrors eval.gsm8k._resolve_checkpoint_path's resolution (explicit path,
    else cfg.save_dir/<model_name>_latest.pt), returning None if truly absent --
    used to decide the graceful NO_CHECKPOINT skip before any model load is attempted."""
    from config.config import cfg

    ckpt_path = Path(ckpt)
    if ckpt_path.exists():
        return ckpt_path
    candidate = Path(cfg.save_dir) / f"{cfg.model_name}_latest.pt"
    if candidate.exists():
        return candidate
    return None


def load_checkpoint_model(ckpt: str, allow_random_weights: bool = False):
    """Thin wrapper over eval.gsm8k._load_model_and_tokenizer -- the repo's one
    checkpoint/tokenizer-identity loader, reused rather than re-implemented."""
    from eval.gsm8k import _load_model_and_tokenizer

    return _load_model_and_tokenizer(ckpt, allow_random_weights=allow_random_weights)

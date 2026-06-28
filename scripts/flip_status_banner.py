#!/usr/bin/env python3
"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - STATUS BANNER AUDIT (report-only)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================

REPORT-ONLY auditor for the source-file status banner. It lists every tracked
file carrying `PRE-TRAINING (UNVERIFIED)` and (read-only) previews the
evidence it *would* need before a post-training flip. It HAS NO WRITE PATH.

Why no writer here (incident-driven, deliberate):
- A pre-built auto-writer is unsafe in a claim-disciplined repo: a naive
  "checkpoint file + eval summary exists" gate is satisfied by a stray demo/
  smoke checkpoint plus a stub `summary.json` ("status: ready" with zero
  counts), and would stamp a FALSE "TRAINED" claim across ~50 files.
- The real flip is a CLAIM and must be a deliberate post-run action, designed
  and tested against a *real* 45K checkpoint + a *real* eval (non-zero metric),
  with explicit human confirmation. See BACKLOG.md "evidence-gated banner flip".

So this tool only *reports*; flipping is done later, by hand or by a writer
built once the genuine artifacts exist to gate against.
"""
from __future__ import annotations

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert Yünlü"

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
PRE_BANNER = "PRE-TRAINING (UNVERIFIED)"


def _git_tracked_text_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    paths = []
    for line in out.splitlines():
        line = line.strip()
        if line.endswith((".py", ".md", ".yaml", ".yml", ".txt", ".json")):
            paths.append(ROOT / line)
    return paths


def _carriers() -> list[Path]:
    found = []
    for p in _git_tracked_text_files():
        try:
            if p.is_file() and PRE_BANNER in p.read_text(encoding="utf-8", errors="ignore"):
                found.append(p)
        except Exception:
            continue
    return found


def _real_eval_metric() -> Optional[str]:
    """Read-only: does a NON-stub eval summary exist (a real, non-zero metric)?"""
    p = ROOT / "reports" / "benchmarks" / "gsm8k_summary.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    total = int(data.get("total", 0) or 0)
    if total <= 0:
        return None  # stub / zero-count summary is NOT evidence
    return f"gsm8k accuracy={data.get('accuracy')} over {total}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Report-only status-banner auditor (no write path).")
    ap.add_argument("--list", action="store_true", help="List every file carrying the pre-training banner.")
    args = ap.parse_args()

    carriers = _carriers()
    print(f"[audit] tracked files carrying '{PRE_BANNER}': {len(carriers)}")
    if args.list:
        for p in carriers:
            print(f"   - {p.relative_to(ROOT)}")

    metric = _real_eval_metric()
    if metric is None:
        print("[audit] post-training flip: NOT ELIGIBLE — no real eval metric yet "
              "(need a non-zero gsm8k_summary.json from the post-45K run).")
    else:
        print(f"[audit] a real eval metric exists ({metric}). A deliberate, "
              "human-confirmed flip can be designed now — this tool still will not write.")
    print("[audit] report-only: this script never modifies files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

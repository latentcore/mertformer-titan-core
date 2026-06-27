#!/usr/bin/env python3
"""
FACTS single-source drift gate.

Reads reports/FACTS.json (the canonical numbers) and asserts that:
  1. No CURRENT-TRUTH markdown surface contains a stale measured-param id
     (the historical 3,698,246,156 / 3698246156 must only live in dated snapshots).
  2. The canonical measured-param id appears in README.md and README_TR.md.

Dated / archived historical surfaces are intentionally excluded so they can keep
their as-of numbers (a Feb-2026 audit legitimately recorded the older count).

Exit 0 = consistent; exit 1 = drift detected (prints offenders). Wired into
scripts/verify_all.sh so a stale number can never silently re-enter current docs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACTS_PATH = ROOT / "reports" / "FACTS.json"

# Path substrings that mark a file as dated/archived historical (excluded from the gate).
HISTORICAL_MARKERS = (
    "/snapshots/",
    "codex_deep_audit",
    "before_final",
    "ocean_pre45k",
    "_partial_evidence",
    "reports/FACTS.json",  # the canonical file lists the stale ids on purpose
)


def _tracked_markdown() -> list[Path]:
    out = subprocess.check_output(
        ["git", "ls-files", "*.md", "*.markdown"], cwd=ROOT, encoding="utf-8"
    )
    return [ROOT / line for line in out.splitlines() if line.strip()]


def _is_historical(rel: str) -> bool:
    return any(marker in rel for marker in HISTORICAL_MARKERS)


def main() -> int:
    facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    canonical = str(facts["measured_runtime_params"])  # "3,672,982,022"
    stale_ids = [s for s in facts["stale_measured_params_historical_only"] if any(c.isdigit() for c in s) and "B" not in s]

    offenders: list[str] = []
    for path in _tracked_markdown():
        rel = path.relative_to(ROOT).as_posix()
        if _is_historical(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            # Don't let an unreadable md file silently bypass the stale-id gate:
            # warn loudly so a skipped surface is visible, then keep degrading.
            print(f"WARN: skip unreadable markdown {rel}: {exc}", file=sys.stderr)
            continue
        for stale in stale_ids:
            if stale in text:
                offenders.append(f"{rel}: contains stale measured-param id {stale!r}")

    # Canonical number must be present where it is claimed authoritative.
    for required in ("README.md", "README_TR.md"):
        if canonical not in (ROOT / required).read_text(encoding="utf-8"):
            offenders.append(f"{required}: missing canonical measured-param id {canonical!r}")

    if offenders:
        print("FACTS consistency: FAIL")
        for o in offenders:
            print(f"  - {o}")
        print(f"\nCanonical measured-param id is {canonical!r} (reports/FACTS.json).")
        print("Stale ids belong only in dated historical snapshots.")
        return 1

    print(f"FACTS consistency: OK (canonical measured-param id {canonical!r}; no stale ids in current-truth md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

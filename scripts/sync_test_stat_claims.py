#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STAT_RE = re.compile(r"\d+\s+passed,\s*\d+\s+skipped")

TARGETS = [
    # FACTS.json is the single-source-of-truth; bind its test_stat to the live pytest count
    # too (it was missing here, which let FACTS.json drift to 369 while README went to 370).
    ROOT / "reports" / "FACTS.json",
    ROOT / "README.md",
    ROOT / "README_TR.md",
    ROOT / "reports" / "release_snapshot.md",
    ROOT / "reports" / "release_snapshot_TR.md",
    ROOT / "reports" / "one_command_full_sop_summary.md",
    # Outreach brief is a living surface that quotes the "latest recorded closure
    # context" test stat; auto-sync it so it never drifts behind the suite again.
    ROOT / "reports" / "outreach" / "mertformer_titan_executive_brief_2026-05-22.md",
    # Present-tense GO/verification ledgers. These self-bill as the "final consistency
    # ledger" / "GO Status" / "clean-room verification", so they must always carry the
    # LIVE suite count, never a stale one. (Dated audits like codex_deep_audit_* keep
    # their as-of counts and are intentionally NOT synced here.)
    ROOT / "reports" / "go_status_matrix.md",
    ROOT / "reports" / "go_status_matrix_TR.md",
    ROOT / "reports" / "final_sync_matrix.md",
    ROOT / "reports" / "final_sync_matrix_TR.md",
    ROOT / "reports" / "cleanroom_verification.md",
    ROOT / "reports" / "cleanroom_verification_TR.md",
]


def sync_file(path: Path, stat: str) -> bool:
    text = path.read_text(encoding="utf-8")
    updated = STAT_RE.sub(stat, text)
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize current pytest pass/skipped claim markers.")
    parser.add_argument("--test-stat", default=os.environ.get("MERTFORMER_EXPECTED_TEST_STAT", ""))
    args = parser.parse_args()
    stat = args.test_stat.strip()
    if not re.fullmatch(r"\d+\s+passed,\s*\d+\s+skipped", stat):
        raise SystemExit(f"invalid or missing --test-stat: {stat!r}")

    changed = []
    missing = []
    for path in TARGETS:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        if sync_file(path, stat):
            changed.append(str(path.relative_to(ROOT)))

    print(
        {
            "status": "ok",
            "test_stat": stat,
            "changed": changed,
            "missing": missing,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

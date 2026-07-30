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
    # (The outreach executive brief moved to private/commercial/ — no longer a tracked
    #  reviewer surface, so it is not synced here.)
    # Present-tense GO/verification ledgers. These self-bill as the "final consistency
    # ledger" / "GO Status" / "clean-room verification", so they must always carry the
    # LIVE suite count, never a stale one. (Dated audits like codex_deep_audit_* keep
    # their as-of counts and are intentionally NOT synced here.)
    ROOT / "reports" / "go_status_matrix.md",
    ROOT / "reports" / "go_status_matrix_TR.md",
    ROOT / "reports" / "final_sync_matrix.md",
    ROOT / "reports" / "final_sync_matrix_TR.md",
    ROOT / "reports" / "cleanroom_verification.md",
    # Root closure scaffold (STATUS.md is the "reviewer's first-read canonical status
    # surface" per the Master Protocol's own §6.1) -- found missing here 2026-07-18,
    # which let it silently drift to a stale count (503) while README/FACTS tracked
    # the live one (511). These self-bill as current-truth, never a dated snapshot.
    ROOT / "STATUS.md",
    ROOT / "STATUS_TR.md",
    ROOT / "TRUTH_MATRIX.md",
    ROOT / "TRUTH_MATRIX_TR.md",
    ROOT / "REPRODUCE.md",
    ROOT / "REPRODUCE_TR.md",
    ROOT / "reports" / "cleanroom_verification_TR.md",
    # Live "how to run this" guide -- states the expected live count as a fact for a new
    # reader, not a dated snapshot. Found stale at 503 while FACTS.json/README were at 511
    # (2026-07-18); it isn't a bilingual pair (no QUICKSTART_CPU_TR.md exists).
    ROOT / "docs" / "QUICKSTART_CPU.md",
]


# Tracked files that legitimately carry a `N passed, M skipped` string WITHOUT being
# synced. Two distinct reasons, both deliberate:
#
#   * DATED HISTORICAL RECORDS -- append-only ledgers and snapshots where each entry
#     states the count as of ITS OWN date. BACKLOG/CHANGELOG/DECISIONS each carry a whole
#     series (370, 388, 412, ... ) precisely because they are a history; rewriting them to
#     the live number would destroy that. ADRs are immutable by policy.
#   * TEST FIXTURES -- synthetic counts inside tests. Syncing these would rewrite the very
#     inputs the tests assert on.
#
# [2026-07-30] This list exists so the pair (TARGETS, HISTORICAL_ALLOWLIST) is EXHAUSTIVE
# over the tracked tree. The failure this closes has already happened twice, as this file's
# own comments record: reports/FACTS.json was missing from TARGETS and drifted to 369 while
# README moved to 370; STATUS.md was missing and drifted to 503 while README/FACTS were at
# 511. Both times a current-truth surface was silently excluded because the list is
# hand-maintained and nothing checked it. tests/test_test_stat_claim_coverage.py now fails
# when a tracked file carries the pattern and appears in NEITHER list, so a new surface
# must be classified on purpose instead of drifting by default.
HISTORICAL_ALLOWLIST = frozenset({
    # Append-only ledgers: one entry per dated pass, each with its own as-of count.
    "BACKLOG.md",
    "BACKLOG_TR.md",
    "CHANGELOG.md",
    "CHANGELOG_TR.md",
    "DECISIONS.md",
    "DECISIONS_TR.md",
    # Immutable architecture decision records.
    "adr/ADR-0004-blocker-fix-pass-core-override.md",
    "adr/ADR-0005-parallel-precompute-orchestration.md",
    # Dated audits and snapshots -- they document a moment, not the present.
    "reports/codex_deep_audit_EN.md",
    "reports/codex_deep_audit_TR.md",
    "reports/current_delta_addendum_2026_05_15.md",
    "reports/tokenizer_parity_fix.md",
    "reports/snapshots/2026-02-24/evidence_snapshot_2026-02-24.json",
    "reports/snapshots/2026-02-24/mertformer_master_decision_report_TR_2026-02-24.md",
    # Superseded documents kept verbatim as history.
    "archive/audits/codex_deep_audit_DE.md",
    "archive/documents/README_before_final_simplification.md",
    "archive/documents/README_TR_before_final_simplification.md",
    "archive/readme_full_2026-06-16.md",
    "archive/readme_TR_full_2026-06-16.md",
    # Test fixtures: synthetic counts that the tests themselves assert against.
    "tests/test_check_doc_claim_consistency.py",
    "tests/test_sdk_pilot_cli.py",
    # The coverage guard itself -- its regex test feeds sample summary lines through
    # STAT_RE. It flagged itself the moment it was staged, which is the guard working.
    "tests/test_test_stat_claim_coverage.py",
})


def tracked_files_with_stat_claims() -> dict[str, list[str]]:
    """Every tracked file carrying a `N passed, M skipped` string, mapped to its values.

    Used by the coverage guard (see HISTORICAL_ALLOWLIST). Reads via `git ls-files` so it
    sees exactly what ships, and skips binary/undecodable files rather than guessing.
    """
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        check=True, capture_output=True, text=True,
    )
    found: dict[str, list[str]] = {}
    for rel in proc.stdout.split("\n"):
        rel = rel.strip()
        if not rel:
            continue
        path = ROOT / rel
        if path.suffix.lower() not in {".md", ".json", ".txt", ".py", ".sh", ".yaml", ".yml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        matches = STAT_RE.findall(text)
        if matches:
            found[rel] = sorted(set(matches))
    return found


def unclassified_stat_claim_files() -> list[str]:
    """Tracked files carrying a test-stat claim that are in neither list."""
    target_rels = {str(p.relative_to(ROOT)) for p in TARGETS}
    return sorted(
        rel for rel in tracked_files_with_stat_claims()
        if rel not in target_rels and rel not in HISTORICAL_ALLOWLIST
    )


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

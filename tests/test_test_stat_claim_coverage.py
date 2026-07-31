"""Every tracked test-stat claim must be classified on purpose (2026-07-30).

``scripts/sync_test_stat_claims.py`` rewrites the live ``N passed, M skipped`` figure into
a hand-maintained ``TARGETS`` list of current-truth surfaces. Files that legitimately keep
an old figure -- append-only ledgers, dated audits, immutable ADRs, test fixtures -- are
deliberately excluded.

The hazard is the hand-maintained part, and it has already bitten twice, as that script's
own comments record: ``reports/FACTS.json`` was missing from TARGETS and drifted to 369
while README moved to 370; ``STATUS.md`` was missing and drifted to 503 while README and
FACTS were at 511. Both times a reviewer-facing surface silently kept a stale number
because nothing checked the list for completeness.

These tests make (TARGETS, HISTORICAL_ALLOWLIST) exhaustive over the tracked tree, so a new
file carrying the pattern must be classified deliberately rather than drifting by default.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.sync_test_stat_claims as SYNC  # noqa: E402


def test_every_tracked_stat_claim_is_classified():
    """The core guard: no tracked file may carry the pattern unclassified."""
    unclassified = SYNC.unclassified_stat_claim_files()
    assert not unclassified, (
        "These tracked files carry a 'N passed, M skipped' claim but appear in neither "
        "TARGETS (synced to the live count) nor HISTORICAL_ALLOWLIST (deliberately frozen) "
        "in scripts/sync_test_stat_claims.py. Add each one to whichever is correct -- if it "
        f"states current truth it MUST be synced or it will silently go stale: {unclassified}"
    )


def test_the_two_lists_do_not_overlap():
    """A file cannot be both synced and deliberately frozen."""
    target_rels = {p.relative_to(SYNC.ROOT).as_posix() for p in SYNC.TARGETS}
    overlap = sorted(target_rels & set(SYNC.HISTORICAL_ALLOWLIST))
    assert not overlap, f"listed as both synced and historical: {overlap}"


def test_all_sync_targets_exist_and_carry_a_claim():
    """A dead target silently does nothing; catch it here instead."""
    found = SYNC.tracked_files_with_stat_claims()
    dead = []
    for path in SYNC.TARGETS:
        rel = path.relative_to(SYNC.ROOT).as_posix()
        if not path.exists():
            dead.append(f"{rel} (missing)")
        elif rel not in found:
            dead.append(f"{rel} (no stat claim to sync)")
    assert not dead, f"sync targets that do nothing: {dead}"


def test_all_historical_allowlist_entries_exist():
    """Stale allowlist entries hide the fact that a surface was renamed or removed."""
    missing = [rel for rel in SYNC.HISTORICAL_ALLOWLIST
               if not (SYNC.ROOT / rel).exists()]
    assert not missing, f"HISTORICAL_ALLOWLIST references files that no longer exist: {missing}"


def test_synced_surfaces_agree_with_each_other():
    """All current-truth surfaces must state the SAME count.

    They are rewritten together by the ladder, so a disagreement means one of them was
    edited by hand, or the ladder has not been run since a change -- exactly the drift this
    machinery exists to prevent.
    """
    found = SYNC.tracked_files_with_stat_claims()
    claims: dict[tuple[str, str], list[str]] = {}
    for path in SYNC.TARGETS:
        rel = path.relative_to(SYNC.ROOT).as_posix()
        for value in found.get(rel, []):
            claims.setdefault(value, []).append(rel)
    assert len(claims) <= 1, (
        "current-truth surfaces disagree on the test count; run "
        f"`bash scripts/verify_all.sh` to re-propagate: {claims}"
    )


def test_historical_ledgers_are_not_synced():
    """Guard the OTHER direction: rewriting a dated ledger would destroy its history.

    BACKLOG/CHANGELOG/DECISIONS each carry a whole series of counts, one per dated entry.
    If one of them ever ends up in TARGETS, every historical figure in it collapses to the
    live one and the record of how the suite grew is gone.
    """
    target_rels = {p.relative_to(SYNC.ROOT).as_posix() for p in SYNC.TARGETS}
    ledgers = {"BACKLOG.md", "BACKLOG_TR.md", "CHANGELOG.md", "CHANGELOG_TR.md",
               "DECISIONS.md", "DECISIONS_TR.md"}
    wrongly_synced = sorted(ledgers & target_rels)
    assert not wrongly_synced, (
        f"append-only ledgers must never be synced: {wrongly_synced}"
    )

    # And confirm they really do carry multiple distinct values, i.e. they are histories.
    found = SYNC.tracked_files_with_stat_claims()
    for ledger in sorted(ledgers):
        if ledger in found:
            assert len(found[ledger]) >= 2, (
                f"{ledger} carries only one stat value; if it is no longer a history, "
                f"reconsider whether it belongs in HISTORICAL_ALLOWLIST"
            )


def test_stat_regex_matches_pytest_summary_format():
    """The regex must match what pytest actually prints, including varied spacing."""
    for line in ("717 passed, 4 skipped", "717 passed,  4 skipped", "1 passed, 0 skipped"):
        assert re.fullmatch(r"\d+\s+passed,\s*\d+\s+skipped", line), line
        assert SYNC.STAT_RE.search(f"suite: {line} in 60s")

from __future__ import annotations

from scripts import check_doc_claim_consistency as claim_consistency


def test_accepted_test_stats_allows_known_platform_variants() -> None:
    assert claim_consistency.accepted_test_stats("158 passed, 4 skipped") == {
        "157 passed, 5 skipped",
        "158 passed, 4 skipped",
        "159 passed, 3 skipped",
    }
    assert claim_consistency.accepted_test_stats("159 passed, 3 skipped") == {
        "158 passed, 4 skipped",
        "159 passed, 3 skipped",
        "160 passed, 2 skipped",
    }


def test_accepted_test_stats_does_not_allow_unrelated_stale_counts() -> None:
    assert "138 passed, 4 skipped" not in claim_consistency.accepted_test_stats("158 passed, 4 skipped")
    assert claim_consistency.parse_test_stat("158 passed, 4 skipped") == (158, 4)
    assert claim_consistency.parse_test_stat("not-a-stat") is None


def test_check_readiness_surface_catches_missing_and_stale_needles() -> None:
    errors: list[str] = []
    claim_consistency.check_readiness_surface(
        "demo.md",
        "TRAIN_ALLOWED READY_OFFLINE_CLEAN",
        ["READY_REMOTE_BOOTSTRAP"],
        ["READY_OFFLINE_CLEAN"],
        errors,
    )
    assert any("missing readiness truth" in item for item in errors)
    assert any("stale readiness truth" in item for item in errors)

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_closure_governance_pack as pack


def test_current_source_docs_include_new_generated_surfaces_and_adr_entries() -> None:
    paths = {entry["path"] for entry in pack.current_source_docs()}
    assert "reports/final_master_plan_freeze.md" in paths
    assert "reports/known_limits_v1.md" in paths
    assert "reports/repo_closure_scorecard.md" in paths
    assert "adr/ADR-0001-source-of-truth-and-claim-boundary.md" in paths


def test_known_limits_calls_out_post_run_boundary() -> None:
    md = pack.build_known_limits(
        {
            "final_status": "TRAIN_ALLOWED",
            "blockers": ["online_teacher:MISSING_HF_TOKEN"],
        }
    )
    assert "Repo-side training readiness is currently `TRAIN_ALLOWED`" in md
    assert "claim-grade benchmark outputs tied to trained checkpoints" in md
    assert "`3000+ Elo`, `20 ms/move`, `10000x speedup`" in md


def test_repo_closure_scorecard_has_twenty_four_targets() -> None:
    payload = pack.build_repo_closure_scorecard_payload()
    ids = {item["item_id"] for item in payload["items"]}
    assert payload["schema"] == "repo_closure_scorecard_v1"
    assert payload["target_count"] == 24
    assert "master_plan_freeze" in ids
    assert "known_limits_doc" in ids
    assert "test_verification_matrix" in ids


def test_adr_index_lists_current_records() -> None:
    md = pack.build_adr_index()
    assert "ADR Index" in md
    assert "ADR-0001-source-of-truth-and-claim-boundary.md" in md
    assert "ADR-0002-change-control-and-closure-governance.md" in md

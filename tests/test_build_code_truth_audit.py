from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_code_truth_audit as audit


def test_build_payload_includes_kernel_maturity_and_groups() -> None:
    payload = audit.build_payload(marker_limit=20)
    assert payload["schema"] == "code_truth_delta_audit_v1"
    assert "no-touch" in payload["surface_groups"]
    assert "AGENTS.md" in payload["surface_groups"]["no-touch"]

    by_path = {entry["path"]: entry for entry in payload["technical_surfaces"]}
    assert by_path["mertformer_sdk/kernels/cpp/bitnet_cpu.cpp"]["maturity"] == "reference_safe"
    assert by_path["mertformer_sdk/kernels/metal/engine.py"]["maturity"] == "tested_fallback"
    assert by_path["scripts/chess_5080_onefile.py"]["evidence_complete"] is True


def test_build_payload_uses_four_column_evidence_model() -> None:
    payload = audit.build_payload(marker_limit=10)
    first = payload["technical_surfaces"][0]
    assert set(first["evidence"]) == {"code_path", "canonical_command", "verification", "artifact"}
    assert payload["evidence_contract"]["required_columns"] == audit.EVIDENCE_COLUMNS


def test_markdown_mentions_done_rule_and_crosswalk() -> None:
    payload = audit.build_payload(marker_limit=5)
    md = audit.build_markdown(payload)
    assert "A closure-critical item is only done" in md
    assert "Doc-to-Code Crosswalk" in md
    assert "`README.md`" in md

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_start_gate_module():
    script_path = ROOT / "scripts" / "start_gate.py"
    spec = importlib.util.spec_from_file_location("start_gate_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_operator_decision_green_path_includes_rent_and_transfer_message():
    module = _load_start_gate_module()
    payload = module.build_operator_decision(
        True,
        True,
        {
            "recommended_path": "offline_clean",
            "decision_reason_code": "READY_OFFLINE_CLEAN",
            "blockers": ["online_teacher:MISSING_HF_TOKEN"],
        },
        {
            "verify_all": {"ok": True},
        },
    )
    assert payload["next_action"] == "ALLOCATE_TARGET_MACHINE_AND_START"
    assert payload["required_transfer_files"]
    assert "Allocate or rent the target training machine" in payload["operator_message"]


def test_build_operator_decision_blocked_path_says_do_not_rent_yet():
    module = _load_start_gate_module()
    payload = module.build_operator_decision(
        False,
        False,
        {
            "recommended_path": None,
            "decision_reason_code": "offline_clean:MISSING_STAGE_JSONL",
            "blockers": ["offline_clean:MISSING_STAGE_JSONL"],
        },
        {
            "verify_all": {"ok": False},
        },
    )
    assert payload["next_action"] == "DO_NOT_RENT_YET_FIX_REPO_BLOCKERS"
    assert "Do not rent or allocate the expensive machine yet" in payload["operator_message"]
    assert payload["required_transfer_files"] == []


def test_operator_decision_markdown_lists_transfer_files():
    module = _load_start_gate_module()
    payload = {
        "next_action": "ALLOCATE_TARGET_MACHINE_AND_START",
        "train_allowed": True,
        "structural_ok": True,
        "recommended_path": "offline_clean",
        "decision_reason_code": "READY_OFFLINE_CLEAN",
        "operator_message": "test message",
        "blockers": [],
        "required_transfer_files": ["zero_touch_start.sh", "reports/train_readiness_decision.json"],
    }
    md = module.build_operator_decision_md(payload)
    assert "Start Gate Operator Decision" in md
    assert "`zero_touch_start.sh`" in md
    assert "`reports/train_readiness_decision.json`" in md

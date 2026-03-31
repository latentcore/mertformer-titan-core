from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    script_path = ROOT / "scripts" / "build_max_closure_handoff.py"
    spec = importlib.util.spec_from_file_location("build_max_closure_handoff_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_succeeds_without_desktop_directory(tmp_path: Path, monkeypatch):
    module = _load_module()
    reports_dir = tmp_path / "reports"
    desktop_dir = tmp_path / "missing-desktop"

    monkeypatch.setattr(module, "REPORTS", reports_dir)
    monkeypatch.setattr(module, "FREEZE_JSON", reports_dir / "final_freeze_manifest.json")
    monkeypatch.setattr(module, "FREEZE_MD", reports_dir / "final_freeze_manifest.md")
    monkeypatch.setattr(module, "COMMANDS_MD", reports_dir / "final_commands.md")
    monkeypatch.setattr(module, "HANDOFF_MD", reports_dir / "repo_external_handoff.md")
    monkeypatch.setattr(module, "DESKTOP", desktop_dir)
    monkeypatch.setattr(module, "DESKTOP_HANDOFF", desktop_dir / "MertFormer_Build30_Max_Closure_Handoff.md")
    monkeypatch.delenv("TITAN_DESKTOP_HANDOFF_MODE", raising=False)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "master_closure_matrix.json").write_text(
        json.dumps({"summary": {"total_items": 4, "this_pass": 3, "phase_2": 1, "external": 0, "rejected_with_reason": 0}}),
        encoding="utf-8",
    )
    (reports_dir / "train_readiness_decision.json").write_text(
        json.dumps(
            {
                "final_status": "TRAIN_ALLOWED",
                "decision_reason_code": "READY_OFFLINE_CLEAN",
                "recommended_path": "offline_clean",
            }
        ),
        encoding="utf-8",
    )

    rc = module.main()

    assert rc == 0
    assert module.HANDOFF_MD.exists()
    assert not module.DESKTOP_HANDOFF.exists()

    handoff = module.HANDOFF_MD.read_text(encoding="utf-8")
    assert "desktop_copy_status: `skipped`" in handoff
    assert "desktop_copy_path: `<DESKTOP_PATH>/MertFormer_Build30_Max_Closure_Handoff.md`" in handoff
    assert "desktop_copy_reason: `desktop directory unavailable`" in handoff

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script_path = ROOT / "scripts" / "build_chess_training_readiness_report.py"
    spec = importlib.util.spec_from_file_location("build_chess_training_readiness_report_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_payload_marks_ready_when_required_surfaces_are_green(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "apps" / "chess_gui").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)

    for rel in [
        "scripts/chess_5080_onefile.py",
        "scripts/export_chess_5080_share.py",
        "scripts/build_chess_5080_windows_delivery.py",
        "apps/chess_gui/play_mertformer_chess_web.py",
    ]:
        (root / rel).write_text("# ok\n", encoding="utf-8")

    (root / "reports" / "chess_gui_onefile_sync_report.json").write_text(
        json.dumps({"status": "canonical_fallback_ready"}),
        encoding="utf-8",
    )
    (root / "reports" / "chess_teaching_contract_report.json").write_text(
        json.dumps({"summary": {"all_green": True}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "REPORT_JSON", root / "reports" / "out.json")
    monkeypatch.setattr(module, "REPORT_MD", root / "reports" / "out.md")

    payload = module.build_payload()
    assert payload["final_status"] == "READY_FOR_TRAINING"
    assert payload["summary"]["all_required_green"] is True


def test_main_returns_nonzero_when_required_surface_missing(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "apps" / "chess_gui").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    (root / "scripts" / "chess_5080_onefile.py").write_text("# ok\n", encoding="utf-8")
    (root / "reports" / "chess_gui_onefile_sync_report.json").write_text(
        json.dumps({"status": "drift_detected"}),
        encoding="utf-8",
    )
    (root / "reports" / "chess_teaching_contract_report.json").write_text(
        json.dumps({"summary": {"all_green": False}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "REPORT_JSON", root / "reports" / "out.json")
    monkeypatch.setattr(module, "REPORT_MD", root / "reports" / "out.md")

    rc = module.main()
    assert rc == 1

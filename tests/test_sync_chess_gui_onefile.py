from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script_path = ROOT / "scripts" / "sync_chess_gui_onefile.py"
    spec = importlib.util.spec_from_file_location("sync_chess_gui_onefile_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sync_onefile_copies_canonical_into_gui_dir(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    canonical = tmp_path / "repo" / "scripts" / "chess_5080_onefile.py"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("print('canonical')\n", encoding="utf-8")

    gui_dir = tmp_path / "gui"
    gui_dir.mkdir(parents=True, exist_ok=True)
    target = gui_dir / "chess_5080_onefile.py"
    target.write_text("print('stale')\n", encoding="utf-8")

    monkeypatch.setattr(module, "CANONICAL_ONEFILE", canonical)
    monkeypatch.setattr(module, "REPORT_JSON", tmp_path / "reports" / "sync.json")
    monkeypatch.setattr(module, "REPORT_MD", tmp_path / "reports" / "sync.md")

    report = module.sync_onefile(gui_dir, check_only=False)
    assert report["status"] == "synced"
    assert report["copied"] is True
    assert target.read_text(encoding="utf-8") == "print('canonical')\n"


def test_main_check_only_detects_drift(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    canonical = tmp_path / "repo" / "scripts" / "chess_5080_onefile.py"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("print('canonical')\n", encoding="utf-8")

    gui_dir = tmp_path / "gui"
    gui_dir.mkdir(parents=True, exist_ok=True)
    (gui_dir / "chess_5080_onefile.py").write_text("print('stale')\n", encoding="utf-8")

    monkeypatch.setattr(module, "CANONICAL_ONEFILE", canonical)
    monkeypatch.setattr(module, "REPORT_JSON", tmp_path / "reports" / "sync.json")
    monkeypatch.setattr(module, "REPORT_MD", tmp_path / "reports" / "sync.md")
    monkeypatch.setattr(sys, "argv", ["sync_chess_gui_onefile.py", "--gui-dir", str(gui_dir), "--check-only"])

    rc = module.main()
    report = json.loads(module.REPORT_JSON.read_text(encoding="utf-8"))
    assert rc == 1
    assert report["status"] == "drift_detected"
    assert report["copied"] is False


def test_check_only_accepts_repo_gui_canonical_fallback_when_local_copy_missing(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    canonical = tmp_path / "repo" / "scripts" / "chess_5080_onefile.py"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("print('canonical')\n", encoding="utf-8")

    gui_dir = tmp_path / "repo" / "apps" / "chess_gui"
    gui_dir.mkdir(parents=True, exist_ok=True)
    (gui_dir / "play_mertformer_chess_web.py").write_text("print('gui')\n", encoding="utf-8")

    monkeypatch.setattr(module, "CANONICAL_ONEFILE", canonical)
    monkeypatch.setattr(module, "DEFAULT_GUI_DIR", gui_dir)
    monkeypatch.setattr(module, "REPORT_JSON", tmp_path / "reports" / "sync.json")
    monkeypatch.setattr(module, "REPORT_MD", tmp_path / "reports" / "sync.md")
    monkeypatch.setattr(sys, "argv", ["sync_chess_gui_onefile.py", "--gui-dir", str(gui_dir), "--check-only"])

    rc = module.main()
    report = json.loads(module.REPORT_JSON.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["status"] == "canonical_fallback_ready"
    assert report["copied"] is False
    assert report["local_copy_present"] is False

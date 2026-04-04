from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.export_chess_5080_share as export_share


def test_build_obfuscated_wrapper_contains_share_mode_flag() -> None:
    wrapper = export_share.build_obfuscated_wrapper("print('ok')\n", "demo.py")
    assert "MERTFORMER_CHESS_SHARE_MODE" in wrapper
    assert "marshal.loads" in wrapper


def test_export_main_creates_bundle(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("print('hello')\n", encoding="utf-8")
    monkeypatch.setattr(export_share, "SOURCE", source)
    monkeypatch.setattr(export_share, "DESKTOP", tmp_path)
    target = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", ["export", "--out-dir", str(target)])
    export_share.main()
    manifest = json.loads((target / "share_manifest.json").read_text(encoding="utf-8"))
    assert (target / "mertformer_chess_5080_share.py").exists()
    assert (target / "source.py").exists()
    assert manifest["files"]

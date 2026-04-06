from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.export_chess_5080_share as export_share


def test_render_build_scripts_keep_builder_entrypoints_and_runtime_contract() -> None:
    bat = export_share.render_build_bat('builder.py')
    ps1 = export_share.render_build_ps1('builder.py')
    assert 'builder.py' in bat
    assert 'builder.py' in ps1
    assert 'MERTFORMER_CHESS_ARCHIVE_PASSWORD' not in bat
    assert 'MERTFORMER_CHESS_ARCHIVE_PASSWORD' not in ps1


def test_export_main_creates_bundle(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / 'source.py'
    source.write_text("print('hello')\n", encoding='utf-8')
    builder = tmp_path / 'build.py'
    builder.write_text("print('build')\n", encoding='utf-8')
    monkeypatch.setattr(export_share, 'SOURCE', source)
    monkeypatch.setattr(export_share, 'WINDOWS_BUILDER', builder)
    monkeypatch.setattr(export_share, 'DESKTOP', tmp_path)
    target = tmp_path / 'bundle'
    monkeypatch.setattr(sys, 'argv', ['export', '--out-dir', str(target)])
    export_share.main()
    manifest = json.loads((target / 'delivery_manifest.json').read_text(encoding='utf-8'))
    assert (target / 'source.py').exists()
    assert (target / 'build.py').exists()
    assert (target / 'build_windows_delivery.bat').exists()
    assert (target / 'build_windows_delivery.ps1').exists()
    assert (target / 'RUN_FINAL_BUILD.ps1').exists()
    assert (target / 'RUN_FINAL_BUILD.bat').exists()
    assert 'RUN_FINAL_BUILD.ps1' in (target / 'README_BUILD.md').read_text(encoding='utf-8')
    assert '--mode arena' in (target / 'README_BUILD.md').read_text(encoding='utf-8')
    assert 'does not embed `MERTFORMER_CHESS_ARCHIVE_PASSWORD` into the compiled launcher' in (target / 'README_BUILD.md').read_text(encoding='utf-8')
    assert 'logs/run_log.jsonl' in (target / 'README_BUILD.md').read_text(encoding='utf-8')
    assert manifest['files']
    assert manifest['contract']['recommended_entrypoint'] == 'RUN_FINAL_BUILD.ps1'
    assert manifest['contract']['observability']['main_run_log'] == 'logs/run_log.jsonl'

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_chess_5080_windows_delivery as builder


def test_render_launcher_embeds_required_runtime_flags() -> None:
    launcher = builder.render_launcher('secret-pass')
    assert 'MERTFORMER_CHESS_ENCRYPT_OUTPUT' in launcher
    assert 'MERTFORMER_CHESS_SINGLE_OUTPUT' in launcher
    assert 'secret-pass' in launcher


def test_build_nuitka_command_targets_onefile(tmp_path: Path) -> None:
    launcher = tmp_path / 'launcher.py'
    command = builder.build_nuitka_command(launcher, tmp_path, 'demo.exe')
    assert '--onefile' in command
    assert '--windows-console-mode=disable' in command
    assert any(part.endswith('demo.exe') for part in command)

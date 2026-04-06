from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_chess_5080_windows_delivery as builder


def test_render_launcher_embeds_required_runtime_flags() -> None:
    launcher = builder.render_launcher()
    assert 'MERTFORMER_CHESS_ENCRYPT_OUTPUT' in launcher
    assert 'MERTFORMER_CHESS_SINGLE_OUTPUT' in launcher
    assert 'MERTFORMER_CHESS_ARCHIVE_PASSWORD' not in launcher


def test_build_nuitka_command_targets_onefile(tmp_path: Path) -> None:
    launcher = tmp_path / 'launcher.py'
    command = builder.build_nuitka_command(launcher, tmp_path, 'demo.exe')
    assert '--onefile' in command
    assert '--python-flag=no_docstrings' in command
    assert '--python-flag=no_asserts' in command
    assert '--file-reference-choice=runtime' in command
    assert '--windows-console-mode=disable' in command
    assert any(part.endswith('demo.exe') for part in command)


def test_torch_install_is_acceptable_requires_cu128_runtime() -> None:
    assert builder.torch_install_is_acceptable({'found': True, 'version': '2.6.1+cu128', 'cuda': '12.8'}) is True
    assert builder.torch_install_is_acceptable({'found': True, 'version': '2.6.1', 'cuda': ''}) is False
    assert builder.torch_install_is_acceptable({'found': True, 'version': '2.5.1+cu128', 'cuda': '12.8'}) is False

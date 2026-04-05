#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'scripts' / 'chess_5080_onefile.py'
WINDOWS_BUILDER = ROOT / 'scripts' / 'build_chess_5080_windows_delivery.py'
DESKTOP = Path.home() / 'Desktop'
PREFIX = 'MertFormer_Chess_5080_Delivery'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def render_build_bat(builder_name: str) -> str:
    return textwrap.dedent(
        f'''\
        @echo off
        setlocal
        set ROOT=%~dp0
        if "%MERTFORMER_CHESS_ARCHIVE_PASSWORD%"=="" (
          echo [delivery] ERROR: set MERTFORMER_CHESS_ARCHIVE_PASSWORD before building.
          exit /b 2
        )
        if not exist "%ROOT%\\.delivery-build-venv\\Scripts\\python.exe" (
          py -3 -m venv "%ROOT%\\.delivery-build-venv" || exit /b 1
        )
        call "%ROOT%\\.delivery-build-venv\\Scripts\\python.exe" -m pip install --upgrade pip || exit /b 1
        call "%ROOT%\\.delivery-build-venv\\Scripts\\python.exe" "%ROOT%\\{builder_name}" --workspace "%ROOT%" %*
        '''
    )


def render_build_ps1(builder_name: str) -> str:
    return textwrap.dedent(
        f'''\
        $ErrorActionPreference = 'Stop'
        $root = Split-Path -Parent $MyInvocation.MyCommand.Path
        if (-not $env:MERTFORMER_CHESS_ARCHIVE_PASSWORD) {{
            throw 'Set MERTFORMER_CHESS_ARCHIVE_PASSWORD before building.'
        }}
        $venvPython = Join-Path $root '.delivery-build-venv\\Scripts\\python.exe'
        if (-not (Test-Path $venvPython)) {{
            py -3 -m venv (Join-Path $root '.delivery-build-venv')
        }}
        & $venvPython -m pip install --upgrade pip
        & $venvPython (Join-Path $root '{builder_name}') --workspace $root @args
        '''
    )


def render_readme(source_name: str, builder_name: str) -> str:
    return textwrap.dedent(
        f'''\
        # RTX 5080 Windows Delivery Build Workspace

        This bundle is an internal build workspace. The public-facing artifact is the final compiled Windows executable, not these source files.

        Included files:
        - `{source_name}`: canonical readable source-of-truth chess onefile
        - `{builder_name}`: Windows build orchestrator
        - `build_windows_delivery.bat`: simplest Windows entrypoint
        - `build_windows_delivery.ps1`: PowerShell entrypoint
        - `delivery_manifest.json`: source and builder hashes

        Windows build contract:
        1. Copy this folder to a Windows 10/11 machine.
        2. Set `MERTFORMER_CHESS_ARCHIVE_PASSWORD` in that Windows session.
        3. Run `build_windows_delivery.bat`.
        4. The builder will create an external delivery folder that contains only the final `.exe`.

        Safety and trust model:
        - The repo copy stays open and auditable.
        - The external build uses a compiled standalone executable for practical hardening.
        - No anti-forensic self-delete behavior is used.
        - The final runtime artifact is designed to emit an encrypted archive when the build-time password is embedded.
        '''
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Export Windows RTX 5080 chess delivery build workspace')
    parser.add_argument('--out-dir', help='Optional destination directory')
    args = parser.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f'Source script missing: {SOURCE}')
    if not WINDOWS_BUILDER.exists():
        raise SystemExit(f'Windows builder missing: {WINDOWS_BUILDER}')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bundle_dir = Path(args.out_dir).expanduser() if args.out_dir else (DESKTOP / f'{PREFIX}_{stamp}')
    bundle_dir.mkdir(parents=True, exist_ok=True)

    source_copy = bundle_dir / SOURCE.name
    builder_copy = bundle_dir / WINDOWS_BUILDER.name
    build_bat = bundle_dir / 'build_windows_delivery.bat'
    build_ps1 = bundle_dir / 'build_windows_delivery.ps1'
    readme = bundle_dir / 'README_BUILD.md'
    manifest = bundle_dir / 'delivery_manifest.json'
    zip_path = DESKTOP / f'{PREFIX}_{stamp}.zip'
    sha_path = DESKTOP / f'{PREFIX}_{stamp}.zip.sha256'

    shutil.copy2(SOURCE, source_copy)
    shutil.copy2(WINDOWS_BUILDER, builder_copy)
    build_bat.write_text(render_build_bat(builder_copy.name), encoding='utf-8')
    build_ps1.write_text(render_build_ps1(builder_copy.name), encoding='utf-8')
    readme.write_text(render_readme(source_copy.name, builder_copy.name), encoding='utf-8')

    file_rows = []
    for path in (source_copy, builder_copy, build_bat, build_ps1, readme):
        file_rows.append(
            {
                'path': path.name,
                'sha256': sha256_file(path),
                'size_bytes': path.stat().st_size,
            }
        )
    manifest_payload = {
        'generated_at_local': stamp,
        'bundle_dir': str(bundle_dir),
        'files': file_rows,
        'contract': {
            'final_external_artifact': 'single Windows executable',
            'password_storage': 'builder requires MERTFORMER_CHESS_ARCHIVE_PASSWORD and does not persist it to tracked files',
            'repo_source_of_truth': str(SOURCE),
        },
    }
    manifest.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.iterdir()):
            if path.is_dir():
                continue
            zf.write(path, arcname=path.name)
    sha_path.write_text(f'{sha256_file(zip_path)}  {zip_path.name}\n', encoding='utf-8')

    print(
        json.dumps(
            {
                'bundle_dir': str(bundle_dir),
                'zip_path': str(zip_path),
                'sha256_path': str(sha_path),
                'source': str(SOURCE),
                'builder': str(WINDOWS_BUILDER),
            },
            indent=2,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

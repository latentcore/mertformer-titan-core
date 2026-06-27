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
        $venvPython = Join-Path $root '.delivery-build-venv\\Scripts\\python.exe'
        if (-not (Test-Path $venvPython)) {{
            py -3 -m venv (Join-Path $root '.delivery-build-venv')
        }}
        & $venvPython -m pip install --upgrade pip
        & $venvPython (Join-Path $root '{builder_name}') --workspace $root @args
        '''
    )


def render_run_final_ps1() -> str:
    return textwrap.dedent(
        """\
        $ErrorActionPreference = 'Stop'
        $ProgressPreference = 'Continue'

        $deliveryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
        Set-Location $deliveryRoot

        $required = @(
            'chess_5080_onefile.py',
            'build_chess_5080_windows_delivery.py',
            'build_windows_delivery.ps1',
            'README_BUILD.md',
            'delivery_manifest.json'
        )

        foreach ($item in $required) {
            if (-not (Test-Path (Join-Path $deliveryRoot $item))) {
                throw "Eksik dosya: $item"
            }
        }

        Write-Host ''
        Write-Host '=== MertFormer Chess 5080 Windows Delivery Build ===' -ForegroundColor Cyan
        Write-Host "Workspace: $deliveryRoot" -ForegroundColor DarkGray
        Write-Host 'Build basliyor...' -ForegroundColor Green
        Write-Host ''

        & (Join-Path $deliveryRoot 'build_windows_delivery.ps1')

        $exePath = Join-Path $deliveryRoot 'external_delivery\\mertformer_chess_5080.exe'
        if (-not (Test-Path $exePath)) {
            throw 'EXE uretilmedi. internal_build altindaki build report dosyasini kontrol et.'
        }

        $exeInfo = Get-Item $exePath
        $exeSizeMb = [math]::Round($exeInfo.Length / 1MB, 2)
        $exeHash = (Get-FileHash $exePath -Algorithm SHA256).Hash

        Write-Host ''
        Write-Host '=== BUILD TAMAMLANDI ===' -ForegroundColor Green
        Write-Host "EXE   : $exePath" -ForegroundColor White
        Write-Host "BOYUT : $exeSizeMb MB" -ForegroundColor White
        Write-Host "SHA256: $exeHash" -ForegroundColor Yellow
        Write-Host ''
        Write-Host 'Windows operatora gidecek final dosya budur:' -ForegroundColor Cyan
        Write-Host "  $exePath" -ForegroundColor White
        Write-Host ''
        Write-Host 'Notlar:' -ForegroundColor Magenta
        Write-Host '- Bu build compiled Windows executable uretir; kaynak .py dosyasi final teslim değildir.'
        Write-Host '- Derlenen EXE pratikte kaynak koda göre daha korumalıdır, ama tersine mühendislik teorik olarak tamamen imkansiz değildir.'
        Write-Host '- Arena modu source/runtime yuzeyinde desteklenir; anlamli oyun icin egitilmis checkpoint kullan.'
        Write-Host '- Final EXE runtime artefactlarini kendi yaninda olusturdugu runtime/ kokeni altinda toplar; Desktop spam hedeflenmez.'
        Write-Host '- Stockfish benchmark gerekiyorsa runtime stockfish binarysini indirip cachelemeyi dener; manuel stockfish.exe koymak zorunlu degildir.'
        Write-Host '- Runtime sonucu sifreli archive uretecekse EXE yi calistirmadan once MERTFORMER_CHESS_ARCHIVE_PASSWORD ortam degiskenini hedef makinede ayarla.'
        """
    )


def render_run_final_bat() -> str:
    return textwrap.dedent(
        """\
        @echo off
        setlocal
        set ROOT=%~dp0
        powershell -ExecutionPolicy Bypass -File "%ROOT%RUN_FINAL_BUILD.ps1"
        """
    )


def render_readme(source_name: str, builder_name: str, bundle_name: str) -> str:
    return textwrap.dedent(
        f'''\
        # RTX 5080 Windows Delivery Build Workspace

        This bundle is the Windows-side build input. The public-facing artifact is the final compiled Windows executable, not these source files.

        ## Included Files
        - `{source_name}`: canonical readable source-of-truth chess onefile
        - `{builder_name}`: Windows build orchestrator
        - `build_windows_delivery.bat`: minimal Windows entrypoint
        - `build_windows_delivery.ps1`: PowerShell entrypoint used by the orchestrator
        - `RUN_FINAL_BUILD.ps1`: guided PowerShell wrapper with checks, final EXE verification, and SHA256 display
        - `RUN_FINAL_BUILD.bat`: one-click BAT wrapper that launches the guided PowerShell wrapper
        - `delivery_manifest.json`: bundle file hashes

        ## Correct Build Model
        - Transport/share artifact: this folder zipped as `{bundle_name}.zip`
        - Windows build input: the extracted folder
        - Final public artifact: `external_delivery/mertformer_chess_5080.exe`

        ## Recommended Windows Build Steps
        1. Copy or extract this folder onto a Windows 10/11 machine.
        2. Open PowerShell in this folder.
        3. Run `RUN_FINAL_BUILD.ps1`.

        Exact command:

        ```powershell
        cd "$env:USERPROFILE\\Desktop\\{bundle_name}"
        .\\RUN_FINAL_BUILD.ps1
        ```

        ## What The Build Does
        - creates `.delivery-build-venv`
        - installs/updates build dependencies
        - installs CUDA 12.8 Windows `torch` if needed
        - compiles a standalone Windows EXE with Nuitka
        - writes the final EXE under `external_delivery/`
        - writes a machine-readable build report under `internal_build/`

        ## Internet Usage (Clean First Build)
        For a clean Windows build with no existing builder venv and no cached wheels, the dependency payload currently requested by the build is approximately (figures below are rough, version-dependent estimates - not measured guarantees - and will drift as pinned package versions change):

        - base dependencies: ~22 MB
        - CUDA 12.8 Windows `torch` wheel: ~2.6 GB
        - combined dependency payload: ~2.7 GB

        Practical note: small extra overhead beyond this can still happen because of pip metadata traffic and any helper downloads triggered by the compiler toolchain. Treat these numbers as an approximate package-download floor for a clean first build, not a universal guaranteed total on every machine.

        ## Security / Trust Model
        - The repo copy stays open and auditable.
        - The external build produces a compiled Windows EXE, which is materially harder to inspect than a raw `.py` file.
        - This is practical protection, not a mathematical impossibility proof against reverse engineering.
        - Optional Authenticode signing support already exists through the builder if you provide `signtool` plus signing certificate environment variables.

        ## Runtime Contract
        - The final runtime artifact is designed to emit a single encrypted result archive.
        - The runtime archive password must be supplied on the target machine via `MERTFORMER_CHESS_ARCHIVE_PASSWORD` before launching the final EXE when encrypted output is required.
        - The Windows builder does not embed `MERTFORMER_CHESS_ARCHIVE_PASSWORD` into the compiled launcher.
        - The runtime writes the main structured log to `logs/run_log.jsonl`.
        - The runtime writes operator-facing logging reports to `reports/logging_contract.json` and `reports/observability_report.json`.
        - Fatal failures are expected to appear in both `logs/run_log.jsonl` and a Desktop-side `*_FAILED_*.json` artifact.
        - The compiled EXE runs with the `delivery_windows_oneclick` runtime profile.
        - Runtime artifacts are expected under `runtime/` next to the EXE instead of spraying new run folders directly onto Desktop.
        - The runtime attempts to auto-fetch/cache Stockfish under `runtime/stockfish/` when benchmark surfaces need it.
        - The chess script now includes an interactive human-vs-AI CLI through `--mode arena`; meaningful play expects a trained checkpoint via `--resume-from`.
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
    run_final_ps1 = bundle_dir / 'RUN_FINAL_BUILD.ps1'
    run_final_bat = bundle_dir / 'RUN_FINAL_BUILD.bat'
    readme = bundle_dir / 'README_BUILD.md'
    manifest = bundle_dir / 'delivery_manifest.json'
    bundle_basename = bundle_dir.name if args.out_dir else f'{PREFIX}_{stamp}'
    zip_path = DESKTOP / f'{bundle_basename}.zip'
    sha_path = DESKTOP / f'{bundle_basename}.zip.sha256'

    shutil.copy2(SOURCE, source_copy)
    shutil.copy2(WINDOWS_BUILDER, builder_copy)
    build_bat.write_text(render_build_bat(builder_copy.name), encoding='utf-8')
    build_ps1.write_text(render_build_ps1(builder_copy.name), encoding='utf-8')
    run_final_ps1.write_text(render_run_final_ps1(), encoding='utf-8')
    run_final_bat.write_text(render_run_final_bat(), encoding='utf-8')
    readme.write_text(render_readme(source_copy.name, builder_copy.name, bundle_basename), encoding='utf-8')

    file_rows = []
    for path in (source_copy, builder_copy, build_bat, build_ps1, run_final_ps1, run_final_bat, readme):
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
            'password_storage': 'runtime password is provided via MERTFORMER_CHESS_ARCHIVE_PASSWORD on the target machine and is not embedded into the compiled launcher',
            'repo_source_of_truth': str(SOURCE),
            'recommended_entrypoint': 'RUN_FINAL_BUILD.ps1',
            'runtime_profile': 'delivery_windows_oneclick',
            'runtime_root': 'runtime/',
            'stockfish_contract': 'runtime auto-fetch/cache under runtime/stockfish when benchmark surfaces need a local engine',
            'observability': {
                'main_run_log': 'logs/run_log.jsonl',
                'logging_contract_report': 'reports/logging_contract.json',
                'observability_report': 'reports/observability_report.json',
                'failure_artifact': 'desktop FAILED json + fatal_exception event in run_log.jsonl',
            },
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

#!/usr/bin/env python3
from __future__ import annotations
import base64
import datetime as dt
import hashlib
import json
import os
import stat
import tempfile
import textwrap
import zlib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / 'scripts' / 'mertformer_5080_final_onefile.py'
PACKAGES = REPO_ROOT / 'packages'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def main() -> int:
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    PACKAGES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mertformer_5080_delivery_') as tmp:
        staging = Path(tmp) / f'MertFormer_5080_Final_Delivery_{stamp}'
        staging.mkdir(parents=True, exist_ok=True)
        payload = base64.b85encode(zlib.compress(SOURCE.read_bytes(), 9)).decode('ascii')
        protected = staging / 'mertformer_5080_final_onefile_protected.py'
        _write(protected, """#!/usr/bin/env python3
# Practical protection launcher. It is obfuscated/packed, not impossible to reverse engineer.
import base64, zlib
PAYLOAD = %r
code = zlib.decompress(base64.b85decode(PAYLOAD.encode('ascii')))
ns = {'__name__': '__main__', '__file__': '<mertformer_5080_final_onefile_protected>'}
exec(compile(code, '<mertformer_5080_final_onefile_protected>', 'exec'), ns)
""" % payload)
        os.chmod(protected, os.stat(protected).st_mode | stat.S_IXUSR)
        entry_template = """#!/usr/bin/env python3
# Practical protected EXE entry. Args are fixed so the receiver can double-click a single EXE.
import base64, sys, zlib
sys.argv = [sys.argv[0], "--mode", "run", "--profile", {profile!r}]
PAYLOAD = {payload!r}
code = zlib.decompress(base64.b85decode(PAYLOAD.encode("ascii")))
ns = {{"__name__": "__main__", "__file__": "<mertformer_5080_{profile}_exe>"}}
exec(compile(code, "<mertformer_5080_{profile}_exe>", "exec"), ns)
"""
        _write(staging / 'mertformer_5080_exe_safe.py', entry_template.format(profile='safe_5080', payload=payload))
        _write(staging / 'mertformer_5080_exe_challenge.py', entry_template.format(profile='challenge_5080', payload=payload))
        _write(staging / 'run_5080_default.bat', '@echo off\r\nset MERTFORMER_SELF_BOOTSTRAP=1\r\npython mertformer_5080_final_onefile_protected.py --mode run --profile safe_5080\r\npause\r\n')
        _write(staging / 'run_5080_challenge.bat', '@echo off\r\nset MERTFORMER_SELF_BOOTSTRAP=1\r\npython mertformer_5080_final_onefile_protected.py --mode run --profile challenge_5080\r\npause\r\n')
        _write(staging / 'run_smoke_test.bat', '@echo off\r\nset MERTFORMER_SELF_BOOTSTRAP=1\r\npython mertformer_5080_final_onefile_protected.py --mode smoke --profile smoke --device auto --no-chat\r\npause\r\n')
        _write(staging / 'run_5080_default.ps1', '$env:MERTFORMER_SELF_BOOTSTRAP="1"\npython .\\mertformer_5080_final_onefile_protected.py --mode run --profile safe_5080\n')
        _write(staging / 'run_5080_challenge.ps1', '$env:MERTFORMER_SELF_BOOTSTRAP="1"\npython .\\mertformer_5080_final_onefile_protected.py --mode run --profile challenge_5080\n')
        _write(staging / 'build_windows_exe.ps1', textwrap.dedent('''
            $ErrorActionPreference = "Stop"
            Write-Host "[MertFormer] Building a practical single EXE. This can take a long time because PyTorch/CUDA files are large."
            if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
                throw "Python was not found. Install Python 3.11+ for your user, then rerun this script."
            }
            python -m venv .venv_build
            $PY = ".\\.venv_build\\Scripts\\python.exe"
            & $PY -m pip install --upgrade pip wheel setuptools
            & $PY -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
            & $PY -m pip install --upgrade -r .\\requirements_5080.txt
            $env:MERTFORMER_SELF_BOOTSTRAP = "0"
            $collect = @(
                "--collect-all", "torch",
                "--collect-all", "numpy",
                "--collect-all", "datasets",
                "--collect-all", "tokenizers",
                "--collect-all", "safetensors",
                "--collect-all", "psutil",
                "--collect-all", "cryptography",
                "--hidden-import", "torch",
                "--hidden-import", "datasets",
                "--hidden-import", "tokenizers",
                "--hidden-import", "safetensors",
                "--hidden-import", "psutil",
                "--hidden-import", "cryptography"
            )
            $targets = @(
                @{ Name = "MertFormer5080Safe"; Entry = ".\\mertformer_5080_exe_safe.py" },
                @{ Name = "MertFormer5080Challenge"; Entry = ".\\mertformer_5080_exe_challenge.py" }
            )
            foreach ($target in $targets) {
                Write-Host "[MertFormer] Building $($target.Name).exe"
                & $PY -m PyInstaller --clean --noconfirm --onefile --name $target.Name @collect $target.Entry
                $EXE = ".\\dist\\$($target.Name).exe"
                if (-not (Test-Path $EXE)) { throw "EXE build failed: $EXE was not created." }
                $hash = (Get-FileHash $EXE -Algorithm SHA256).Hash.ToLower()
                "$hash  $($target.Name).exe" | Out-File -Encoding ascii "$EXE.sha256"
                Write-Host "[MertFormer] EXE built at $EXE"
                Write-Host "[MertFormer] SHA256: $hash"
            }
            Write-Host "[MertFormer] For friend-only delivery, send exactly one EXE:"
            Write-Host "  Safe first run:      dist\\MertFormer5080Safe.exe"
            Write-Host "  Overnight challenge: dist\\MertFormer5080Challenge.exe"
        '''))
        _write(staging / 'requirements_5080.txt', textwrap.dedent('''
            numpy
            datasets
            tokenizers
            safetensors
            psutil
            cryptography
            pyinstaller
        '''))
        _write(staging / 'README_FRIEND_TR.md', textwrap.dedent('''
            # MertFormer 5080 Friend Run

            1. Windows bilgisayarda NVIDIA driver güncel olsun.
            2. EXE üretmek için PowerShell'de:
               `powershell -ExecutionPolicy Bypass -File .\\build_windows_exe.ps1`
            3. Oluşan dosyalar:
               - `dist\\MertFormer5080Safe.exe`: güvenli varsayılan profil.
               - `dist\\MertFormer5080Challenge.exe`: daha büyük/uzun gece koşusu.
            4. Arkadaşa sadece bir exe verilecekse, uzun eğitim için `MertFormer5080Challenge.exe` ver.
            5. Sadece Python ile denemek istersen önce `run_smoke_test.bat`, sonra `run_5080_default.bat` kullan.
            6. Daha agresif uzun koşu Python yolunda istenirse `run_5080_challenge.bat` ayrı opsiyonel yoldur.
            7. Çıktılar varsayılan olarak script/exe klasöründeki run/evidence klasörlerine yazılır.

            Not: Bu paket pratik korumalıdır, imkansız reverse-engineering garantisi yoktur.
            Kalite iddiası smoke ile değil benchmark ile açılır.
            `safe_5080` ana önerilen teslim profilidir; `challenge_5080` deneysel-güç odaklı ikinci yoldur.
            Tek EXE hedefi pratik olarak desteklenir, ancak NVIDIA driver ve Windows güvenlik izinleri sistemde hazır olmalıdır.
        '''))
        manifest = {
            'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'source_sha256': sha(SOURCE),
            'protected_py_sha256': sha(protected),
            'recommended_profile': 'safe_5080',
            'optional_profile': 'challenge_5080',
            'truth_boundary': 'No Gemma-2B superiority claim without measured eval.',
            'files': sorted(p.name for p in staging.iterdir() if p.is_file()),
        }
        _write(staging / 'delivery_manifest.json', json.dumps(manifest, indent=2))
        zip_path = PACKAGES / f'{staging.name}.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(staging.rglob('*')):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(staging)))
        _write(zip_path.with_suffix(zip_path.suffix + '.sha256'), sha(zip_path) + '  ' + zip_path.name + '\n')
        # 'ok' is bound to a real post-write check (zip exists and is non-empty),
        # not unconditionally True. Defensive: zip write above would normally raise
        # on failure, but verify the artifact rather than assume success.
        ok = zip_path.exists() and zip_path.stat().st_size > 0
        print(json.dumps({'ok': ok, 'delivery_zip': str(zip_path), 'sha256': sha(zip_path)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

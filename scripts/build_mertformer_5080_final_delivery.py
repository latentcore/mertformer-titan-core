#!/usr/bin/env python3
from __future__ import annotations
import base64
import datetime as dt
import hashlib
import json
import os
import shutil
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
        _write(staging / 'run_5080_default.bat', '@echo off\r\nset MERTFORMER_SELF_BOOTSTRAP=1\r\npython mertformer_5080_final_onefile_protected.py --mode run --profile safe_5080\r\npause\r\n')
        _write(staging / 'run_5080_challenge.bat', '@echo off\r\nset MERTFORMER_SELF_BOOTSTRAP=1\r\npython mertformer_5080_final_onefile_protected.py --mode run --profile challenge_5080\r\npause\r\n')
        _write(staging / 'run_smoke_test.bat', '@echo off\r\nset MERTFORMER_SELF_BOOTSTRAP=1\r\npython mertformer_5080_final_onefile_protected.py --mode smoke --profile smoke --device auto --no-chat\r\npause\r\n')
        _write(staging / 'run_5080_default.ps1', '$env:MERTFORMER_SELF_BOOTSTRAP="1"\npython .\\mertformer_5080_final_onefile_protected.py --mode run --profile safe_5080\n')
        _write(staging / 'run_5080_challenge.ps1', '$env:MERTFORMER_SELF_BOOTSTRAP="1"\npython .\\mertformer_5080_final_onefile_protected.py --mode run --profile challenge_5080\n')
        _write(staging / 'build_windows_exe.ps1', textwrap.dedent('''
            $ErrorActionPreference = "Stop"
            python -m pip install --upgrade pyinstaller
            pyinstaller --onefile --name MertFormer5080Final .\\mertformer_5080_final_onefile_protected.py
            Write-Host "EXE built at dist\\MertFormer5080Final.exe"
        '''))
        _write(staging / 'requirements_5080.txt', textwrap.dedent('''
            --index-url https://download.pytorch.org/whl/cu128
            torch
            numpy
            datasets
            tokenizers
            safetensors
            psutil
            cryptography
        '''))
        _write(staging / 'README_FRIEND_TR.md', textwrap.dedent('''
            # MertFormer 5080 Friend Run

            1. Windows bilgisayarda NVIDIA driver güncel olsun.
            2. `run_smoke_test.bat` ile önce küçük test çalıştır.
            3. Sorun yoksa önce `run_5080_default.bat` ile güvenli profili çalıştır.
            4. Daha agresif uzun koşu istenirse `run_5080_challenge.bat` ayrı opsiyonel yoldur.
            5. Çıktılar varsayılan olarak script klasöründeki run/evidence klasörlerine yazılır.

            Not: Bu paket pratik korumalıdır, imkansız reverse-engineering garantisi yoktur.
            Kalite iddiası smoke ile değil benchmark ile açılır.
            `safe_5080` ana önerilen teslim profilidir; `challenge_5080` deneysel-güç odaklı ikinci yoldur.
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
        print(json.dumps({'ok': True, 'delivery_zip': str(zip_path), 'sha256': sha(zip_path)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

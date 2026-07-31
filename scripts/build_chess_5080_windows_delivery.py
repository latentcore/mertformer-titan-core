#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PASSWORD_ENV = 'MERTFORMER_CHESS_ARCHIVE_PASSWORD'
DEFAULT_TORCH_INDEX_ENV = 'MERTFORMER_CHESS_TORCH_INDEX_URL'
DEFAULT_OUTPUT_EXE = 'mertformer_chess_5080.exe'
SOURCE_NAME = 'chess_5080_onefile.py'


class BuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def require_windows() -> None:
    if os.name != 'nt':
        raise BuildError('This builder must run on Windows because the final artifact is a Windows executable.')


def ensure_python_package(package: str) -> bool:
    code = (
        'import importlib.util,sys; '
        f'sys.exit(0 if importlib.util.find_spec({package!r}) is not None else 1)'
    )
    return subprocess.run([sys.executable, '-c', code], check=False).returncode == 0


def parse_major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", version)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def inspect_torch_install() -> Dict[str, Any]:
    code = textwrap.dedent(
        """
        import json
        try:
            import torch
            payload = {
                "found": True,
                "version": getattr(torch, "__version__", ""),
                "cuda": getattr(getattr(torch, "version", None), "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False,
            }
        except Exception as exc:
            payload = {
                "found": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(payload))
        """
    ).strip()
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return {"found": False, "error": result.stderr.strip() or result.stdout.strip() or "torch_probe_failed"}
    try:
        return json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {"found": False, "error": "torch_probe_invalid_json", "raw": result.stdout.strip()}


def torch_install_is_acceptable(report: Dict[str, Any]) -> bool:
    if not bool(report.get("found", False)):
        return False
    version = str(report.get("version", "")).split("+", 1)[0]
    cuda_version = str(report.get("cuda") or "")
    return parse_major_minor(version) >= (2, 6) and cuda_version == "12.8"


def ensure_build_dependencies() -> None:
    base_requirements = [
        'pip>=24',
        'setuptools>=70',
        'wheel>=0.45',
        'nuitka>=2.7,<3',
        'ordered-set>=4.1,<5',
        'numpy>=1.24,<3',
        'zstandard>=0.21,<1',
        'python-chess>=1.999,<2',
        'psutil>=5.9,<8',
        'pyzipper>=0.3.6,<1',
    ]
    run([sys.executable, '-m', 'pip', 'install', '--upgrade', *base_requirements])
    torch_report = inspect_torch_install()
    if not torch_install_is_acceptable(torch_report):
        index_url = os.environ.get(DEFAULT_TORCH_INDEX_ENV, 'https://download.pytorch.org/whl/cu128')
        run([sys.executable, '-m', 'pip', 'install', '--upgrade', '--index-url', index_url, 'torch>=2.6,<3'])


def render_launcher() -> str:
    return textwrap.dedent(
        '''\
        from __future__ import annotations
        import os
        import sys
        from pathlib import Path

        os.environ.setdefault('MERTFORMER_CHESS_ALLOW_INSTALL', '0')
        os.environ.setdefault('MERTFORMER_CHESS_SHARE_MODE', '0')
        os.environ.setdefault('MERTFORMER_CHESS_SELF_DELETE', '0')
        os.environ.setdefault('MERTFORMER_CHESS_ENCRYPT_OUTPUT', '1')
        os.environ.setdefault('MERTFORMER_CHESS_ENCRYPTION_REQUIRED', '1')
        os.environ.setdefault('MERTFORMER_CHESS_SINGLE_OUTPUT', '1')
        os.environ.setdefault('MERTFORMER_CHESS_CLEANUP_AFTER_BUNDLE', '1')
        os.environ.setdefault('MERTFORMER_CHESS_PROFILE', 'delivery_windows_oneclick')

        runtime_root = Path(sys.argv[0]).resolve().parent / 'runtime'
        os.environ.setdefault('MERTFORMER_CHESS_ARTIFACT_ROOT', str(runtime_root))
        os.environ.setdefault('MERTFORMER_CHESS_CACHE_ROOT', str(runtime_root / 'cache'))
        os.environ.setdefault('MERTFORMER_CHESS_STOCKFISH_CACHE_ROOT', str(runtime_root / 'stockfish'))
        os.environ.setdefault('MERTFORMER_CHESS_STOCKFISH_AUTO_FETCH', '1')

        from chess_5080_onefile import main

        raise SystemExit(main())
        '''
    )


def build_nuitka_command(launcher_path: Path, output_dir: Path, output_filename: str) -> List[str]:
    return [
        sys.executable,
        '-m',
        'nuitka',
        '--onefile',
        '--assume-yes-for-downloads',
        '--follow-imports',
        '--include-module=pyzipper',
        '--python-flag=no_docstrings',
        '--python-flag=no_asserts',
        '--file-reference-choice=runtime',
        '--windows-console-mode=disable',
        f'--output-dir={output_dir}',
        f'--output-filename={output_filename}',
        '--remove-output',
        str(launcher_path),
    ]


def detect_signtool() -> Optional[str]:
    explicit = os.environ.get('MERTFORMER_CHESS_SIGNTOOL_PATH', '').strip()
    if explicit and Path(explicit).exists():
        return explicit
    return shutil.which('signtool.exe') or shutil.which('signtool')


def try_sign_executable(exe_path: Path) -> Dict[str, Any]:
    pfx_path = os.environ.get('MERTFORMER_CHESS_SIGN_CERT_PFX', '').strip()
    pfx_password = os.environ.get('MERTFORMER_CHESS_SIGN_CERT_PASSWORD', '')
    timestamp_url = os.environ.get('MERTFORMER_CHESS_SIGN_TIMESTAMP_URL', 'http://timestamp.digicert.com').strip()
    signtool = detect_signtool()
    if not signtool or not pfx_path:
        return {'status': 'skipped', 'reason': 'credentials_or_signtool_missing'}
    if not Path(pfx_path).exists():
        return {'status': 'skipped', 'reason': 'pfx_missing'}
    cmd = [
        signtool,
        'sign',
        '/fd',
        'SHA256',
        '/f',
        pfx_path,
        '/p',
        pfx_password,
        '/tr',
        timestamp_url,
        '/td',
        'SHA256',
        str(exe_path),
    ]
    try:
        run(cmd)
        return {'status': 'signed', 'signtool': signtool, 'timestamp_url': timestamp_url}
    except subprocess.CalledProcessError as exc:
        return {'status': 'failed', 'signtool': signtool, 'error': str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description='Build hardened Windows RTX 5080 chess executable')
    parser.add_argument('--workspace', default='.', help='Workspace containing chess_5080_onefile.py')
    parser.add_argument('--output-name', default=DEFAULT_OUTPUT_EXE, help='Final executable name')
    args = parser.parse_args()

    require_windows()

    workspace = Path(args.workspace).expanduser().resolve()
    source_path = workspace / SOURCE_NAME
    if not source_path.exists():
        raise SystemExit(f'Missing source script in workspace: {source_path}')

    ensure_build_dependencies()

    internal_dir = workspace / 'internal_build'
    external_dir = workspace / 'external_delivery'
    internal_dir.mkdir(parents=True, exist_ok=True)
    if external_dir.exists():
        shutil.rmtree(external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with tempfile.TemporaryDirectory(prefix='mertformer_chess_delivery_') as tmp_raw:
        tmp_dir = Path(tmp_raw)
        source_copy = tmp_dir / SOURCE_NAME
        launcher_path = tmp_dir / 'delivery_launcher.py'
        shutil.copy2(source_path, source_copy)
        launcher_path.write_text(render_launcher(), encoding='utf-8')

        nuitka_output = internal_dir / f'nuitka_build_{stamp}'
        nuitka_output.mkdir(parents=True, exist_ok=True)
        build_cmd = build_nuitka_command(launcher_path, nuitka_output, args.output_name)
        run(build_cmd, cwd=tmp_dir)

        expected_exe = nuitka_output / args.output_name
        if not expected_exe.exists():
            candidates = sorted(nuitka_output.rglob('*.exe'))
            if not candidates:
                raise BuildError('Nuitka finished without producing a Windows executable.')
            expected_exe = candidates[0]

        final_exe = external_dir / args.output_name
        shutil.copy2(expected_exe, final_exe)

    sign_report = try_sign_executable(final_exe)
    build_report = {
        'built_at_local': stamp,
        'workspace': str(workspace),
        'source_path': str(source_path),
        'output_executable': str(final_exe),
        'output_sha256': sha256_file(final_exe),
        'nuitka_version': subprocess.check_output([sys.executable, '-m', 'nuitka', '--version'], text=True, encoding="utf-8", errors="replace").strip(),
        'python': sys.version,
        'torch_runtime': inspect_torch_install(),
        'signing': sign_report,
        'external_delivery_contents': [path.name for path in sorted(external_dir.iterdir())],
        'password_env': DEFAULT_PASSWORD_ENV,
        'notes': {
            'final_external_artifact': 'single Windows executable',
            'source_repo_remains_open': True,
            'runtime_output_contract': 'single encrypted archive from the compiled executable',
            'runtime_root_contract': 'runtime artifacts stay under <exe_dir>/runtime and the executable auto-fetches/cache-manages Stockfish when needed',
            'runtime_password_contract': f'set {DEFAULT_PASSWORD_ENV} before running the final executable when encrypted output is required',
            'password_embedded_in_launcher': False,
            'observability_contract': {
                'main_run_log': 'logs/run_log.jsonl',
                'logging_contract_report': 'reports/logging_contract.json',
                'observability_report': 'reports/observability_report.json',
                'failure_artifact': 'desktop FAILED json + fatal_exception event in run_log.jsonl',
            },
        },
    }
    report_path = internal_dir / f'build_report_{stamp}.json'
    report_path.write_text(json.dumps(build_report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(json.dumps(build_report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Regression test for a real import-shadow bug found 2026-07-12 while building
scripts/measure_random_init_baseline.py.

eval/ had no __init__.py, making it a PEP 420 namespace package. scripts/eval.py
(an unrelated benchmark-suite CLI) sits alongside it. Per PEP 420, a namespace
package loses to a REGULAR module of the same name found anywhere else on
sys.path -- so running any scripts/*.py file (which puts scripts/ on sys.path)
silently rebound `eval` to scripts/eval.py, breaking every
`from eval.<submodule> import ...` in the whole eval/ package with a confusing
"'eval' is not a package" error. Fixed by adding eval/__init__.py, making it a
real package (which always wins immediately, regardless of sys.path order or
what else is named "eval" elsewhere on the path).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_eval_package_has_an_init_file() -> None:
    assert (ROOT / "eval" / "__init__.py").exists(), (
        "eval/__init__.py must exist -- without it, eval/ is a namespace package "
        "that loses to scripts/eval.py (see this file's module docstring)."
    )


def test_importing_eval_submodule_from_a_scripts_subprocess_does_not_shadow() -> None:
    """Reproduces the exact failure mode: run a fresh subprocess whose sys.path[0]
    is scripts/ (as if `python3 scripts/whatever.py` were invoked), then import
    an eval/ submodule the way scripts/measure_random_init_baseline.py does."""
    probe = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from eval._probe_common import utc_now\n"
        "print('IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(ROOT / "scripts"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "IMPORT_OK" in result.stdout

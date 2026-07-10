"""Circular-import backstop for the eagerly-imported ``orchestrator`` package (I.7.4).

``orchestrator/__init__.py`` eagerly imports most of its submodules, and ``train.py``
imports from ``orchestrator`` at startup — so a new unguarded cross-import there would
break the training entrypoint. This imports the package (triggering the eager set) and
then every submodule in a FRESH subprocess, so a circular dependency surfaces here as a
hard import error instead of only at launch. A true cycle raises ``ImportError: cannot
import name ... (most likely due to a circular import)`` on first import, which the
subprocess exit code reports.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "orchestrator"


def _submodules() -> list[str]:
    return sorted(
        f"orchestrator.{p.stem}" for p in PKG.glob("*.py") if p.stem != "__init__"
    )


def test_orchestrator_package_imports():
    result = subprocess.run(
        [sys.executable, "-c", "import orchestrator"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, f"`import orchestrator` failed:\n{result.stderr}"


def test_every_orchestrator_submodule_imports():
    mods = _submodules()
    assert mods, "no orchestrator submodules discovered — test wiring is wrong"
    code = (
        "import importlib, sys\n"
        "fails = []\n"
        f"for m in {mods!r}:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as exc:\n"
        "        fails.append(f'{m}: {type(exc).__name__}: {exc}')\n"
        "sys.stdout.write('\\n'.join(fails))\n"
        "sys.exit(1 if fails else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "orchestrator submodule import failure(s):\n" + result.stdout + result.stderr
    )

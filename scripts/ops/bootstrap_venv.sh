#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a deterministic local virtualenv for Titan.
# Default: Python 3.11 + dev extras (lint/test). Demo extras (pygame) are opt-in.
#
# Usage:
#   bash scripts/bootstrap_venv.sh
#   bash scripts/bootstrap_venv.sh --demo
#   PYTHON_BIN=python3.11 VENV_DIR=.titan-venv bash scripts/bootstrap_venv.sh --demo

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.titan-venv}"

INSTALL_DEMO=0
for arg in "$@"; do
  case "$arg" in
    --demo) INSTALL_DEMO=1 ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[bootstrap] Using: $PYTHON_BIN"
echo "[bootstrap] Venv:  $VENV_DIR"

"$PYTHON_BIN" -m venv "$VENV_DIR"

"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

if [ "$INSTALL_DEMO" -eq 1 ]; then
  "$VENV_DIR/bin/python" -m pip install -e ".[dev,demo]"
else
  "$VENV_DIR/bin/python" -m pip install -e ".[dev]"
fi

echo "[bootstrap] Done."

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY="${TITAN_PYTHON:-}"
if [[ -n "$PY" ]]; then
  if [[ ! -x "$PY" ]] && ! command -v "$PY" >/dev/null 2>&1; then
    echo "[zero-touch] TITAN_PYTHON is set but not executable/in PATH: $PY" >&2
    exit 2
  fi
else
  if [[ ! -x ".titan-venv/bin/python" ]]; then
    if [[ -f "scripts/bootstrap_venv.sh" ]]; then
      echo "[zero-touch] bootstrapping .titan-venv ..."
      bash scripts/bootstrap_venv.sh
    fi
  fi

  if [[ -x ".titan-venv/bin/python" ]]; then
    PY=".titan-venv/bin/python"
  else
    PY="python3"
  fi
fi

exec "$PY" scripts/final_orchestrator.py "$@"

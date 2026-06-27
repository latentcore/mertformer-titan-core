#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-/tmp/nihai_cleanroom}"
PYTHON_BIN="${TITAN_CLEANROOM_PYTHON:-python3.11}"

echo "🧪 Clean-room verify starting..."
echo "   source: ${ROOT_DIR}"
echo "   workdir: ${WORKDIR}"
echo "   python: ${PYTHON_BIN}"

rm -rf "${WORKDIR}"
git clone "${ROOT_DIR}" "${WORKDIR}" >/tmp/cleanroom_clone.log 2>&1

cd "${WORKDIR}"
"${PYTHON_BIN}" -m venv .cleanroom-venv
VENV_PY="${WORKDIR}/.cleanroom-venv/bin/python"

"${VENV_PY}" -m pip install -U pip >/tmp/cleanroom_pip_u.log 2>&1
"${VENV_PY}" -m pip install -r requirements.txt >/tmp/cleanroom_pip_req.log 2>&1
"${VENV_PY}" -m pip install -e . >/tmp/cleanroom_pip_edit.log 2>&1
"${VENV_PY}" -m pip install ruff >/tmp/cleanroom_pip_ruff.log 2>&1

"${VENV_PY}" -m pytest -q >/tmp/cleanroom_pytest.log 2>&1
"${VENV_PY}" -m ruff check . >/tmp/cleanroom_ruff.log 2>&1
TITAN_PYTHON="${VENV_PY}" TITAN_OFFLINE=1 bash scripts/verify_all.sh >/tmp/cleanroom_verify_all.log 2>&1

echo "✅ Clean-room verify completed."
echo "   pytest: $(tail -n 1 /tmp/cleanroom_pytest.log)"
echo "   ruff:   $(tail -n 1 /tmp/cleanroom_ruff.log)"
echo "   verify: $(tail -n 1 /tmp/cleanroom_verify_all.log)"

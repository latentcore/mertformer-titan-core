# Clean-Room Verification (Build 30)

## Purpose
Validate that a fresh clone can reproduce the core quality gates on an isolated path.

## Environment
- Source repo: `.`
- Clean clone path: `/tmp/nihai_cleanroom_b27`
- Python: `3.11.14`
- Venv: `.cleanroom-venv`
- Commit under test: `a07a8c1`

## Commands Executed
```bash
SOURCE_REPO="<repo-url-or-local-path>"
git clone "$SOURCE_REPO" /tmp/nihai_cleanroom_b27
cd /tmp/nihai_cleanroom_b27
python3.11 -m venv .cleanroom-venv
.cleanroom-venv/bin/python -m pip install -U pip
.cleanroom-venv/bin/python -m pip install -r requirements.txt
.cleanroom-venv/bin/python -m pip install -e .
.cleanroom-venv/bin/python -m pytest -q
.cleanroom-venv/bin/python -m pip install ruff
.cleanroom-venv/bin/python -m ruff check .
TITAN_PYTHON="$PWD/.cleanroom-venv/bin/python" TITAN_OFFLINE=1 bash scripts/verify_all.sh
```

## Results
- `pytest`: `626 passed, 5 skipped`
- `ruff`: `All checks passed!`
- `verify_all`: `[verify] OK`

## Notes
- Clean-room run was executed in a fresh local clone and separate virtual environment.
- This report is build-time evidence for A17 (clean-room verification gate).

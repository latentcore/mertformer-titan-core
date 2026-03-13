#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${TITAN_PYTHON:-}"
if [[ -z "$PY_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.titan-venv/bin/python" ]]; then
    PY_BIN="$ROOT_DIR/.titan-venv/bin/python"
  else
    PY_BIN="python3"
  fi
fi

ART_DIR="artifacts"
ZIP_PATH="$ART_DIR/mertformer_release.zip"
SHA_PATH="$ART_DIR/mertformer_release.zip.sha256"
REPORT_A="reports/artifacts_zip_denylist_audit.json"
REPORT_B="reports/zip_audit_artifacts.json"

mkdir -p "$ART_DIR" "$ROOT_DIR/reports"

rm -f "$ZIP_PATH" "$SHA_PATH"
zip -r "$ZIP_PATH" . \
  -x ".git/*" "*/.git/*" "*.pyc" "*__pycache__*" \
     ".titan-venv/*" ".lint-venv/*" ".venv/*" ".idea/*" \
     ".pytest_cache/*" ".ruff_cache/*" ".mypy_cache/*" \
     ".env" ".env.*" "logs/*" "reports/.one_command_full_sop_raw.*" \
     "artifacts/mertformer_release.zip" "artifacts/mertformer_release.zip.sha256"

shasum -a 256 "$ZIP_PATH" > "$SHA_PATH"

"$PY_BIN" "$ROOT_DIR/scripts/zip_denylist_audit.py" --zip "$ZIP_PATH" \
  | tee "$REPORT_A" > "$REPORT_B"

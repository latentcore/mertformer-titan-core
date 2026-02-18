#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$ROOT_DIR/packages"
REPORT_EN="$ROOT_DIR/reports/release_snapshot.md"
REPORT_TR="$ROOT_DIR/reports/release_snapshot_TR.md"

REL_ZIP="$PKG_DIR/MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip"
LOCKED_AGE="$PKG_DIR/MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age"

mkdir -p "$PKG_DIR"

rm -f "$REL_ZIP" "$LOCKED_AGE"

(
  cd "$ROOT_DIR"
  zip -rq "$REL_ZIP" . \
    -x '.git/*' \
       '.titan-venv/*' '.titan-venv.bak_*/*' '.lint-venv/*' '.venv/*' \
       '__pycache__/*' '*.pyc' '.pytest_cache/*' '.ruff_cache/*' '.mypy_cache/*' \
       'logs/*' '.DS_Store' 'packages/*' '.env' 'checkpoints/*' 'datasets/*' 'data/*'
)

recipient=""
if [[ -n "${AGE_RECIPIENT_FILE:-}" && -f "${AGE_RECIPIENT_FILE:-}" ]]; then
  # Resolve recipient from file. Supports direct recipient text or age secret key file.
  if grep -Eq '^age1[0-9a-z]+$' "$AGE_RECIPIENT_FILE"; then
    recipient="$(grep -E '^age1[0-9a-z]+$' "$AGE_RECIPIENT_FILE" | head -n1)"
  else
    recipient="$(age-keygen -y "$AGE_RECIPIENT_FILE" 2>/dev/null || true)"
  fi

  if [[ -z "$recipient" ]]; then
    recipient="$(grep -Eo 'age1[0-9a-z]+' "$AGE_RECIPIENT_FILE" | head -n1 || true)"
  fi

  if [[ -z "$recipient" ]]; then
    echo "ERROR: Could not resolve AGE recipient from AGE_RECIPIENT_FILE" >&2
    exit 2
  fi

  age -r "$recipient" -o "$LOCKED_AGE" "$REL_ZIP"
else
  echo "WARN: AGE_RECIPIENT_FILE missing; skipping locked secure .age artifact." >&2
fi

rel_sha="$(shasum -a 256 "$REL_ZIP" | awk '{print $1}')"
locked_sha=""
if [[ -s "$LOCKED_AGE" ]]; then
  locked_sha="$(shasum -a 256 "$LOCKED_AGE" | awk '{print $1}')"
fi

cat <<EOF
release_zip=$REL_ZIP
release_sha256=$rel_sha
locked_age=$LOCKED_AGE
locked_sha256=$locked_sha
EOF

# Best-effort update of release snapshot references.
if [[ -f "$REPORT_EN" ]]; then
  sed -i '' \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Release\.zip|MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip|g" \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Locked\.secure\.age|MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age|g" \
    "$REPORT_EN"
fi

if [[ -f "$REPORT_TR" ]]; then
  sed -i '' \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Release\.zip|MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip|g" \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Locked\.secure\.age|MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age|g" \
    "$REPORT_TR"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$ROOT_DIR/packages"
REPORT_EN="$ROOT_DIR/reports/release_snapshot.md"
REPORT_TR="$ROOT_DIR/reports/release_snapshot_TR.md"

REL_ZIP="$PKG_DIR/MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip"
HAMDI_AGE="$PKG_DIR/MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age"
LOCKED_AGE="$PKG_DIR/MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age"

if [[ -z "${AGE_RECIPIENT_FILE:-}" || ! -f "${AGE_RECIPIENT_FILE:-}" ]]; then
  echo "ERROR: AGE_RECIPIENT_FILE missing or not a file" >&2
  exit 2
fi

if [[ -z "${HAMDI_PASSPHRASE_FILE:-}" || ! -f "${HAMDI_PASSPHRASE_FILE:-}" ]]; then
  echo "ERROR: HAMDI_PASSPHRASE_FILE missing or not a file" >&2
  exit 2
fi

mkdir -p "$PKG_DIR"

# Resolve recipient from file. Supports direct recipient text or age secret key file.
recipient=""
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

hamdi_passphrase="$(head -n1 "$HAMDI_PASSPHRASE_FILE" | tr -d '\r')"
if [[ -z "$hamdi_passphrase" ]]; then
  echo "ERROR: HAMDI passphrase is empty" >&2
  exit 2
fi

if ! command -v expect >/dev/null 2>&1; then
  echo "ERROR: expect is required for non-interactive passphrase age generation" >&2
  exit 2
fi

rm -f "$REL_ZIP" "$HAMDI_AGE" "$LOCKED_AGE"

(
  cd "$ROOT_DIR"
  zip -rq "$REL_ZIP" . \
    -x '.git/*' \
       '.titan-venv/*' '.titan-venv.bak_*/*' '.lint-venv/*' '.venv/*' \
       '__pycache__/*' '*.pyc' '.pytest_cache/*' '.ruff_cache/*' '.mypy_cache/*' \
       'logs/*' '.DS_Store' 'packages/*' '.env' 'checkpoints/*' 'datasets/*' 'data/*'
)

AGE_PASS="$hamdi_passphrase" expect -f - "$REL_ZIP" "$HAMDI_AGE" <<'EXPECT'
set timeout -1
set in_file [lindex $argv 0]
set out_file [lindex $argv 1]
set pass $env(AGE_PASS)
log_user 0

spawn age -p -o $out_file $in_file
expect {
    -re "Enter passphrase.*" {}
    timeout { exit 3 }
    eof { exit 4 }
}
send -- "$pass\r"
expect {
    -re "Confirm passphrase.*" {}
    timeout { exit 5 }
    eof { exit 6 }
}
send -- "$pass\r"
expect eof
catch wait result
set code [lindex $result 3]
exit $code
EXPECT

if [[ ! -s "$HAMDI_AGE" ]]; then
  echo "ERROR: Hamdi passphrase age artifact was not created." >&2
  exit 2
fi

age -r "$recipient" -o "$LOCKED_AGE" "$REL_ZIP"

rel_sha="$(shasum -a 256 "$REL_ZIP" | awk '{print $1}')"
hamdi_sha="$(shasum -a 256 "$HAMDI_AGE" | awk '{print $1}')"
locked_sha="$(shasum -a 256 "$LOCKED_AGE" | awk '{print $1}')"

cat <<EOF
release_zip=$REL_ZIP
release_sha256=$rel_sha
hamdi_age=$HAMDI_AGE
hamdi_sha256=$hamdi_sha
locked_age=$LOCKED_AGE
locked_sha256=$locked_sha
EOF

# Best-effort update of release snapshot references.
if [[ -f "$REPORT_EN" ]]; then
  sed -i '' \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Release\.zip|MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip|g" \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Hamdi_Package_Release\.zip|MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age|g" \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Hamdi_Package_Release\.passphrase\.age|MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age|g" \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Locked\.secure\.age|MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age|g" \
    "$REPORT_EN"
fi

if [[ -f "$REPORT_TR" ]]; then
  sed -i '' \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Release\.zip|MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip|g" \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Hamdi_Package_Release\.zip|MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age|g" \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Hamdi_Package_Release\.passphrase\.age|MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age|g" \
    -e "s|MertFormer_Titan_OnyxStorm_v1.0_B[0-9][0-9]_Locked\.secure\.age|MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age|g" \
    "$REPORT_TR"
fi

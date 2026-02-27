#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$ROOT_DIR/packages"
REPORT_EN="$ROOT_DIR/reports/release_snapshot.md"
REPORT_TR="$ROOT_DIR/reports/release_snapshot_TR.md"
PY="$ROOT_DIR/.titan-venv/bin/python"

REL_ZIP="$PKG_DIR/MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip"
LOCKED_AGE="$PKG_DIR/MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age"

mkdir -p "$PKG_DIR"

if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

"$PY" "$ROOT_DIR/scripts/check_tokenizer_sync.py"

rm -f "$REL_ZIP" "$LOCKED_AGE"

(
  cd "$ROOT_DIR"
  zip -rq "$REL_ZIP" . \
    -x '.git/*' \
       '.titan-venv/*' '.titan-venv.bak_*/*' '.lint-venv/*' '.venv/*' \
       '__pycache__/*' '*/__pycache__/*' '*.pyc' '.pytest_cache/*' '*/.pytest_cache/*' '.ruff_cache/*' '*/.ruff_cache/*' '.mypy_cache/*' '*/.mypy_cache/*' \
       'logs/*' '.DS_Store' 'packages/*' '.env' 'checkpoints/*' \
       'datasets/stage*/*' 'datasets/logits/*' 'data/*' 'artifacts/*'
)

"$PY" "$ROOT_DIR/scripts/zip_denylist_audit.py" --zip "$REL_ZIP"

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
lock_status="skipped (expected: AGE_RECIPIENT_FILE missing)"
if [[ -s "$LOCKED_AGE" ]]; then
  locked_sha="$(shasum -a 256 "$LOCKED_AGE" | awk '{print $1}')"
  lock_status="generated"
fi

cat <<EOF
release_zip=$REL_ZIP
release_sha256=$rel_sha
locked_age=$LOCKED_AGE
locked_sha256=$locked_sha
EOF

# Snapshot updates (cross-platform, deterministic)
"$PY" - "$REPORT_EN" "$REPORT_TR" "$rel_sha" "$locked_sha" "$lock_status" <<'PY'
import re
import sys
from pathlib import Path

report_en, report_tr, rel_sha, locked_sha, lock_status = sys.argv[1:]


def update_release_section(path: Path, is_tr: bool) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    if is_tr:
        h1 = "## Release Artefaktları (Desktop)"
        h2 = "## Bilinen Gate / Blokerler"
        sha_label = "SHA-256"
        status_line = f"- Locked artefakt durumu: `{lock_status}`"
        locked_sha_line = (
            f"  - `{locked_sha}` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`)"
            if locked_sha
            else "  - `SKIPPED` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`)"
        )
    else:
        h1 = "## Release Artifacts (Desktop)"
        h2 = "## Known Gates / Blockers"
        sha_label = "SHA-256"
        status_line = f"- Locked artifact status: `{lock_status}`"
        locked_sha_line = (
            f"  - `{locked_sha}` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`)"
            if locked_sha
            else "  - `SKIPPED` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`)"
        )

    block = (
        f"{h1}\n\n"
        "- `MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip`\n"
        "- `MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`\n"
        f"{status_line}\n"
        f"- {sha_label}:\n"
        f"  - `{rel_sha}` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip`)\n"
        f"{locked_sha_line}\n"
    )

    pattern = re.compile(re.escape(h1) + r".*?" + re.escape(h2), re.S)
    replacement = block + "\n" + h2
    if pattern.search(text):
        text = pattern.sub(replacement, text, count=1)
    path.write_text(text, encoding="utf-8")


update_release_section(Path(report_en), is_tr=False)
update_release_section(Path(report_tr), is_tr=True)
PY

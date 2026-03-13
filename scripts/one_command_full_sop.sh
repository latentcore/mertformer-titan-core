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

LOG_PATH="$ROOT_DIR/reports/one_command_full_sop.log"
SUMMARY_PATH="$ROOT_DIR/reports/one_command_full_sop_summary.md"
RAW_LOG="$(mktemp "$ROOT_DIR/reports/.one_command_full_sop_raw.XXXXXX.log")"

mkdir -p "$ROOT_DIR/reports" "$ROOT_DIR/packages" "$ROOT_DIR/artifacts"

REL_ZIP="packages/MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip"

run_step() {
  local name="$1"; shift
  echo "[run] step=$name"
  "$@"
}

sanitize_file() {
  local path="$1"
  "$PY_BIN" - "$path" "$ROOT_DIR" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
root = str(Path(sys.argv[2]).resolve())
if not p.exists():
    raise SystemExit(0)
text = p.read_text(encoding="utf-8", errors="replace")
text = text.replace(root, "<REPO_ROOT>")
text = re.sub(r"/Users/[^/\s]+/Desktop/[^\s\"']+", "<DESKTOP_PATH>", text)
p.write_text(text, encoding="utf-8")
PY
}

cleanup_tmp() {
  rm -f "$RAW_LOG" 2>/dev/null || true
}
trap cleanup_tmp EXIT

# Ensure legacy/stale SOP log cannot break path-guard tests before verify_all.
if [[ -f "$LOG_PATH" ]]; then
  sanitize_file "$LOG_PATH"
fi

{
  echo "[run] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  run_step "verify_all" env TITAN_OFFLINE=1 TITAN_WANDB=0 TITAN_PYTHON="$PY_BIN" bash scripts/verify_all.sh
  run_step "cfc_moe_tolerance_check" "$PY_BIN" scripts/cfc_moe_tolerance_check.py --out reports/cfc_moe_tolerance_report.json
  run_step "md_quality_all" "$PY_BIN" scripts/md_quality_gate.py --root . --scope all --out reports/md_lint_report.json
  run_step "linkcheck_all" "$PY_BIN" scripts/linkcheck_gate.py --root . --scope all --out reports/linkcheck_report.json
  run_step "docs_inventory" "$PY_BIN" scripts/docs_inventory.py
  run_step "sync_manifest" "$PY_BIN" scripts/sync_manifest.py --root . --manifest reports/release_manifest.json --structure docs/PROJECT_STRUCTURE.md --matrix reports/file_sync_matrix.json --sync-report reports/project_structure_sync_report.json --policy-report reports/policy_sync_report.json
  run_step "dealroom_sync" "$PY_BIN" scripts/dealroom_sync.py
  run_step "unicode_path_guard" "$PY_BIN" scripts/unicode_path_guard.py --root . --out reports/unicode_path_guard_report.json --fail-on-hit
  run_step "duplicate_zip_guard" "$PY_BIN" scripts/duplicate_zip_guard.py --root packages --root artifacts --out reports/duplicate_zip_guard_report.json
  run_step "intermediate_cache_cleanup" "$PY_BIN" scripts/run_and_clean_pycache.py --root . --include-tool-caches --full-clean --include-venv-caches -- bash -lc true
  run_step "clean_runtime_artifacts_check" bash scripts/clean_runtime_artifacts.sh --check
  run_step "release_build30" bash scripts/release_build30.sh
  run_step "artifact_release_zip" bash scripts/build_artifacts_release_zip.sh
  if [[ "${SOP_PLOT_TRAINING_LOG:-0}" == "1" ]]; then
    LOG_PATH_CAND="${SOP_TRAINING_LOG:-}"
    if [[ -z "$LOG_PATH_CAND" ]]; then
      LOG_PATH_CAND=$(ls -t "$ROOT_DIR"/logs/*.jsonl 2>/dev/null | head -n 1 || true)
    fi
    if [[ -n "$LOG_PATH_CAND" && -f "$LOG_PATH_CAND" ]]; then
      run_step "plot_training_log" "$PY_BIN" scripts/plot_training_log.py "$LOG_PATH_CAND" --out reports/training_dashboard.png
      echo "training_dashboard=reports/training_dashboard.png"
    else
      echo "plot_training_log: skipped (no jsonl log found)"
    fi
  fi
  run_step "zip_denylist_audit zip=$REL_ZIP" "$PY_BIN" scripts/zip_denylist_audit.py --zip "$REL_ZIP"
  run_step "secret_scan" "$PY_BIN" scripts/secret_scan.py

  echo "[run] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$RAW_LOG"

"$PY_BIN" - "$RAW_LOG" "$LOG_PATH" "$SUMMARY_PATH" "$ROOT_DIR" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path


def _sanitize(text: str, root: str) -> str:
    out = text.replace(root, "<REPO_ROOT>")
    out = re.sub(r"/Users/[^/\s]+/Desktop/[^\s\"']+", "<DESKTOP_PATH>", out)
    return out


raw_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
root = str(Path(sys.argv[4]).resolve())

raw = raw_path.read_text(encoding="utf-8", errors="replace")
clean = _sanitize(raw, root)
log_path.write_text(clean, encoding="utf-8")

start_utc = ""
end_utc = ""
pytest_line = ""
md_quality_line = ""
linkcheck_line = ""
unicode_ok = False
duplicate_ok = False
runtime_clean_ok = False
zip_ok = False
secret_ok = False
dashboard_path = ""
tolerance_line = ""
tolerance_ok = False
release_sha = ""
locked_sha = ""

for line in clean.splitlines():
    if line.startswith("[run] start_utc="):
        start_utc = line.split("=", 1)[1].strip()
    elif line.startswith("[run] end_utc="):
        end_utc = line.split("=", 1)[1].strip()
    elif re.search(r"\d+ passed, \d+ skipped", line):
        pytest_line = line.strip()
    elif line.startswith("md_quality: "):
        md_quality_line = line.strip()
    elif line.startswith("linkcheck: "):
        linkcheck_line = line.strip()
    elif line.startswith("OK: unicode path guard"):
        unicode_ok = True
    elif line.startswith("OK: duplicate zip guard"):
        duplicate_ok = True
    elif line.startswith("OK: runtime artifacts and caches are clean"):
        runtime_clean_ok = True
    elif line.startswith("OK: no secret patterns detected in tracked files."):
        secret_ok = True
    elif line.startswith("[tolerance]"):
        tolerance_line = line.strip()
        tolerance_ok = "PASS" in line
    elif line.startswith("training_dashboard="):
        dashboard_path = line.split("=", 1)[1].strip()
    elif line.startswith("release_sha256="):
        release_sha = line.split("=", 1)[1].strip()
    elif line.startswith("locked_sha256="):
        locked_sha = line.split("=", 1)[1].strip()
    elif '"ok": true' in line and "zip_path" not in line:
        # zip denylist emits {"ok": true} in a small JSON block.
        zip_ok = True

locked_generated = "yes" if locked_sha else "no"
zip_status = "PASS" if zip_ok else "FAIL"

summary = "\n".join(
    [
        "# One-Command Full SOP Summary",
        "",
        f"- start_utc: {start_utc}",
        f"- end_utc: {end_utc}",
        f"- pytest: {pytest_line or 'not_found'}",
        f"- md_quality_all: {md_quality_line or 'not_found'}",
        f"- linkcheck_all: {linkcheck_line or 'not_found'}",
        f"- unicode_path_guard: {'PASS' if unicode_ok else 'FAIL'}",
        f"- duplicate_zip_guard: {'PASS' if duplicate_ok else 'FAIL'}",
        f"- clean_runtime_artifacts_check: {'PASS' if runtime_clean_ok else 'FAIL'}",
        f"- cfc_moe_tolerance_check: {tolerance_line or 'not_found'}",
        f"- zip_denylist_audit: {zip_status}",
        f"- secret_scan: {'PASS' if secret_ok else 'FAIL'}",
        f"- training_dashboard: {dashboard_path or 'not_generated'}",
        f"- release_zip_sha256: `{release_sha}`",
        f"- locked_age_sha256: `{locked_sha}`",
        f"- locked_age_generated: {locked_generated}",
        "",
        "## Raw Log",
        "- `reports/one_command_full_sop.log`",
        "",
    ]
)
summary_path.write_text(summary, encoding="utf-8")
PY

echo "[run] SOP artifacts refreshed:"
echo " - reports/one_command_full_sop.log"
echo " - reports/one_command_full_sop_summary.md"

SOP_AUTO_COMMIT="${SOP_AUTO_COMMIT:-0}"
SOP_AUTO_PUSH="${SOP_AUTO_PUSH:-0}"
SOP_COMMIT_MSG="${SOP_COMMIT_MSG:-chore: refresh SOP validation artifacts (pass)}"

if git rev-parse --is-inside-work-tree &>/dev/null; then
  if [[ "$SOP_AUTO_COMMIT" == "1" ]]; then
    git add reports packages artifacts || true
    if ! git diff --cached --quiet; then
      git commit -m "$SOP_COMMIT_MSG"
      if [[ "$SOP_AUTO_PUSH" == "1" ]]; then
        git push origin "$(git branch --show-current)"
      fi
    fi
  fi
fi

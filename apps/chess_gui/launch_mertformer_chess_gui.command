#!/bin/zsh
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$APP_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_DIR/.titan-venv/bin/python"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/gui_server_latest.log"
mkdir -p "$LOG_DIR"
cd "$APP_DIR"
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching MertFormer Chess GUI"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing canonical chess onefile from repo"
  "$PYTHON_BIN" "$REPO_DIR/scripts/sync_chess_gui_onefile.py" --gui-dir "$APP_DIR"
  "$PYTHON_BIN" -u "$APP_DIR/play_mertformer_chess_web.py" --port 8765
} 2>&1 | tee -a "$LOG_FILE"

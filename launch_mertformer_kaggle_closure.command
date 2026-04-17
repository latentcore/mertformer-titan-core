#!/bin/zsh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$REPO_DIR/.titan-venv/bin/python"
MODE="${TITAN_KAGGLE_MODE:-train-end}"
PROFILE="${TITAN_KAGGLE_PROFILE:-auto}"
ASSERT_SECONDS="${TITAN_KAGGLE_ASSERT_SECONDS:-10800}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[kaggle-oneclick] python3 not found." >&2
    exit 127
  fi
fi

cd "$REPO_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching canonical Kaggle closure lane"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] mode=$MODE profile=$PROFILE"

bash "$REPO_DIR/scripts/macos_keepawake.sh" --assert-seconds "$ASSERT_SECONDS" -- \
  bash "$REPO_DIR/zero_touch_start.sh" --kaggle-onefile --mode "$MODE" --profile "$PROFILE"

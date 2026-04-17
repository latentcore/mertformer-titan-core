#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/macos_keepawake.sh [--assert-seconds 10800] -- <command> [args...]

Examples:
  bash scripts/macos_keepawake.sh -- python3 scripts/kaggle_onefile_closure_build30.py --mode verify
  bash scripts/macos_keepawake.sh --assert-seconds 10800 -- bash zero_touch_start.sh --kaggle-onefile --mode train-end

Notes:
  - Uses process-scoped `caffeinate`; it does not mutate global power settings.
  - `--assert-seconds` keeps the system awake for a bounded warm-up window before the command starts.
EOF
}

ASSERT_SECONDS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --assert-seconds)
      ASSERT_SECONDS="${2:-0}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

if ! command -v caffeinate >/dev/null 2>&1; then
  echo "[keepawake] caffeinate not found on this system." >&2
  exit 127
fi

if [[ "${ASSERT_SECONDS}" != "0" ]]; then
  caffeinate -dimsu -t "${ASSERT_SECONDS}" &
  ASSERT_PID=$!
else
  ASSERT_PID=""
fi

"$@" &
CMD_PID=$!

cleanup() {
  if [[ -n "${ASSERT_PID}" ]]; then
    kill "${ASSERT_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

caffeinate -dimsu -w "${CMD_PID}" &
CAFFEINATE_PID=$!

wait "${CMD_PID}"
STATUS=$?

kill "${CAFFEINATE_PID}" >/dev/null 2>&1 || true
exit "${STATUS}"

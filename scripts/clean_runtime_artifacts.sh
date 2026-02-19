#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="clean"
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
fi

ROOT_JSONL=()
while IFS= read -r f; do
  ROOT_JSONL+=("$f")
done < <(find "$ROOT_DIR" -maxdepth 1 -type f -name 'kaggle_onefile_build30_*.jsonl' | sort)
EXTERNAL_ONEFILE="/Users/mertyunlu/Desktop/kaggle_onefile_demo_build30.py"

tracked_count=0
for f in "${ROOT_JSONL[@]}"; do
  rel="${f#$ROOT_DIR/}"
  if git ls-files --error-unmatch "$rel" >/dev/null 2>&1; then
    tracked_count=$((tracked_count + 1))
  fi
done

if [[ "$tracked_count" -gt 0 ]]; then
  echo "FAIL: tracked artifact matched cleanup pattern; aborting for safety." >&2
  exit 2
fi

missing=0
if [[ "${#ROOT_JSONL[@]}" -gt 0 ]]; then
  echo "Found runtime artifacts (${#ROOT_JSONL[@]}):"
  printf ' - %s\n' "${ROOT_JSONL[@]}"
  missing=1
fi

if [[ -f "$EXTERNAL_ONEFILE" ]]; then
  echo "Found external onefile drift source: $EXTERNAL_ONEFILE"
  missing=1
fi

if [[ "$MODE" == "check" ]]; then
  if [[ "$missing" -eq 0 ]]; then
    echo "OK: runtime artifacts are clean"
    exit 0
  fi
  echo "FAIL: cleanup required" >&2
  exit 1
fi

for f in "${ROOT_JSONL[@]}"; do
  rm -f "$f"
done

if [[ -f "$EXTERNAL_ONEFILE" ]]; then
  rm -f "$EXTERNAL_ONEFILE"
fi

echo "DONE: runtime artifacts cleanup completed"

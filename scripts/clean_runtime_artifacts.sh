#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="clean"
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
fi

EXTERNAL_ONEFILE="${HOME}/Desktop/kaggle_onefile_demo_build30.py"

RUNTIME_FILES=()
RUNTIME_DIRS=()

while IFS= read -r f; do
  RUNTIME_FILES+=("$f")
done < <(find "$ROOT_DIR" -maxdepth 1 -type f -name 'kaggle_onefile_build30_*.jsonl' | sort)

while IFS= read -r d; do
  RUNTIME_DIRS+=("$d")
done < <(
  find "$ROOT_DIR" -type d \
    \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' \) \
    -not -path "$ROOT_DIR/.git/*" \
    -not -path "$ROOT_DIR/.titan-venv/*" \
    -not -path "$ROOT_DIR/.venv/*" \
    -not -path "$ROOT_DIR/venv/*" \
    | sort
)

while IFS= read -r f; do
  RUNTIME_FILES+=("$f")
done < <(
  find "$ROOT_DIR" -type f \
    \( -name '*.pyc' -o -name '*.pyo' \) \
    -not -path "$ROOT_DIR/.git/*" \
    -not -path "$ROOT_DIR/.titan-venv/*" \
    -not -path "$ROOT_DIR/.venv/*" \
    -not -path "$ROOT_DIR/venv/*" \
    | sort
)

for d in "$ROOT_DIR/temp_preflight_data" "$ROOT_DIR/temp_preflight_logits" "$ROOT_DIR/.cache"; do
  if [[ -d "$d" ]]; then
    RUNTIME_DIRS+=("$d")
  fi
done

for f in "$ROOT_DIR/.coverage" "$ROOT_DIR/coverage.xml"; do
  if [[ -f "$f" ]]; then
    RUNTIME_FILES+=("$f")
  fi
done

tracked_count=0
if [[ -n "${RUNTIME_FILES[*]-}" ]]; then
  for f in "${RUNTIME_FILES[@]}"; do
    rel="${f#$ROOT_DIR/}"
    if git ls-files --error-unmatch "$rel" >/dev/null 2>&1; then
      echo "FAIL: tracked file matched cleanup target: $rel" >&2
      tracked_count=$((tracked_count + 1))
    fi
  done
fi
if [[ -n "${RUNTIME_DIRS[*]-}" ]]; then
  for d in "${RUNTIME_DIRS[@]}"; do
    rel="${d#$ROOT_DIR/}"
    if [[ -n "$(git ls-files "$rel" 2>/dev/null)" ]]; then
      echo "FAIL: tracked content detected under cleanup directory: $rel" >&2
      tracked_count=$((tracked_count + 1))
    fi
  done
fi
if [[ "$tracked_count" -gt 0 ]]; then
  exit 2
fi

dirty=0
if [[ -n "${RUNTIME_FILES[*]-}" ]]; then
  echo "Found runtime/cache files (${#RUNTIME_FILES[@]}):"
  printf ' - %s\n' "${RUNTIME_FILES[@]}"
  dirty=1
fi
if [[ -n "${RUNTIME_DIRS[*]-}" ]]; then
  echo "Found runtime/cache directories (${#RUNTIME_DIRS[@]}):"
  printf ' - %s\n' "${RUNTIME_DIRS[@]}"
  dirty=1
fi
if [[ -f "$EXTERNAL_ONEFILE" ]]; then
  echo "Found external onefile drift source: $EXTERNAL_ONEFILE"
  dirty=1
fi

if [[ "$MODE" == "check" ]]; then
  if [[ "$dirty" -eq 0 ]]; then
    echo "OK: runtime artifacts and caches are clean"
    exit 0
  fi
  echo "FAIL: cleanup required" >&2
  exit 1
fi

if [[ -n "${RUNTIME_FILES[*]-}" ]]; then
  for f in "${RUNTIME_FILES[@]}"; do
    rm -f "$f"
  done
fi
if [[ -n "${RUNTIME_DIRS[*]-}" ]]; then
  for d in "${RUNTIME_DIRS[@]}"; do
    rm -rf "$d"
  done
fi
if [[ -f "$EXTERNAL_ONEFILE" ]]; then
  rm -f "$EXTERNAL_ONEFILE"
fi

echo "DONE: runtime artifacts and caches cleanup completed"

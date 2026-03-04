#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-v1.0.0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p reports

cat > reports/release_closure_note.md <<EOF
# Release Closure Note

- Tag: $TAG
- Rule: no additional commits on closed release tag.
- Hotfix policy: new semver tag required.
EOF

if git rev-parse "$TAG" >/dev/null 2>&1; then
  exists=true
else
  exists=false
fi

cat > reports/release_closure_lock_report.json <<EOF
{
  "tag": "$TAG",
  "tag_exists": $exists,
  "closure_lock": true
}
EOF

echo "release closure lock prepared for $TAG"

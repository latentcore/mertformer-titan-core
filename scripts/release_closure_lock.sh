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

# closure_lock must reflect a real check, not a hardcoded "true":
# the tag must exist AND HEAD must point at the tag commit
# (i.e. no additional commits on the closed release tag, per the rule above).
closure_lock=false
if [ "$exists" = "true" ]; then
  tag_commit="$(git rev-list -n 1 "$TAG" 2>/dev/null || true)"
  head_commit="$(git rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$tag_commit" ] && [ "$tag_commit" = "$head_commit" ]; then
    closure_lock=true
  fi
fi

cat > reports/release_closure_lock_report.json <<EOF
{
  "tag": "$TAG",
  "tag_exists": $exists,
  "closure_lock": $closure_lock
}
EOF

echo "release closure lock prepared for $TAG"

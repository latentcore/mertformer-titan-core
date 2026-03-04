#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OWNER_REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
OWNER="${OWNER_REPO%%/*}"
REPO="${OWNER_REPO##*/}"

mkdir -p reports

PAYLOAD_FILE="$(mktemp)"
cat >"$PAYLOAD_FILE" <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["verify"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": false
}
JSON

set +e
gh api \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/${OWNER}/${REPO}/branches/main/protection" \
  --input "$PAYLOAD_FILE" >/tmp/gh_policy.out 2>/tmp/gh_policy.err
rc=$?
set -e

rm -f "$PAYLOAD_FILE"

if [[ $rc -eq 0 ]]; then
  status="applied"
else
  status="skipped_or_failed"
fi

cat > reports/github_policy_report.json <<JSON
{
  "status": "${status}",
  "repo": "${OWNER_REPO}",
  "return_code": ${rc}
}
JSON

echo "github policy: ${status}"
if [[ $rc -ne 0 ]]; then
  cat /tmp/gh_policy.err >&2 || true
fi

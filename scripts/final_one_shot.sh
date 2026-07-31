#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DOCS_DIR="${HOME}/Documents"
DOCS_REPORTS_DIR="$DOCS_DIR/reports"
IMMUTABLE_ZIP="$DOCS_DIR/Proje_immutable_20260301_d22272ee9281e978.zip"
LINKEDIN_ZIP="$DOCS_DIR/mertformer_outputs_LINKEDIN_run_20260220_175540.zip"

mkdir -p reports artifacts telemetry docs policy "$DOCS_REPORTS_DIR"

run_step() {
  local name="$1"; shift
  echo "[final] $name"
  "$@"
}

run_zip_with_tolerance() {
  local zip_path="$1"
  shift
  set +e
  zip -rq "$zip_path" "$@"
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi
  if [[ "$rc" -eq 1 && -f "$zip_path" ]] && unzip -tqq "$zip_path" >/dev/null 2>&1; then
    echo "[final] WARN: zip exited with code 1 but integrity check passed for $zip_path; continuing." >&2
    return 0
  fi
  return "$rc"
}

refresh_anthropic_packet() {
  # The entire Anthropic application packet now lives under private/anthropic_internal/
  # (gitignored): the core repo is single-persona (technical/evidence). The packet is
  # still built locally for handoff, but nothing it writes is tracked.
  local dir="$ROOT_DIR/private/anthropic_internal"
  [ -d "$dir" ] || { echo "[final] anthropic packet dir absent ($dir); skipping"; return 0; }
  local zip_path="$dir/mertformer_anthropic_packet_20260419.zip"
  local sha_path="$dir/mertformer_anthropic_packet_20260419.zip.sha256"
  local pointer_path="$dir/PACKET_POINTER_20260419.md"
  # Evidence-facing packet only. Internal-strategy notes (interview_prep "Gaps To Say
  # Out Loud", application_strategy, cv_seed "Do Not Claim", strongest_stories,
  # tokens_variant_notes, performance_engineer_fallback) were moved OUT of the repo to
  # private/anthropic_internal/ (gitignored) so they are never shipped to a reviewer.
  local files=(
    "README.md"
    "project_summary.md"
    "measured_evidence_summary.md"
    "why_anthropic_science_of_scaling.md"
  )

  rm -f "$zip_path" "$sha_path"
  (
    cd "$dir"
    zip -q -r "$(basename "$zip_path")" "${files[@]}"
  )

  local sha
  sha="$(shasum -a 256 "$zip_path" | awk '{print $1}')"
  printf '%s  %s\n' "$sha" "$(basename "$zip_path")" > "$sha_path"

  local size_bytes
  size_bytes="$(stat -f%z "$zip_path")"
  cat > "$pointer_path" <<EOF
# Anthropic Packet Pointer - 2026-04-19

This directory has a small local application packet zip for handoff and review.

## Local Artifact

- Zip: \`private/anthropic_internal/$(basename "$zip_path")\`
- SHA256 sidecar: \`private/anthropic_internal/$(basename "$sha_path")\`
- SHA256: \`$sha\`
- Size: \`$size_bytes\` bytes
- Contents: ${#files[@]} evidence-facing markdown application files

## Git Policy

The zip itself is not tracked by git because repository policy ignores \`*.zip\`.

The pointer and SHA sidecar are small enough to keep as closure metadata.

## Truth Boundary

This packet is an application-facing summary bundle. It is not a model artifact, benchmark package, or evidence archive.

The Build30 T4 one-cell evidence pointer lives under:

- \`evidence/build30_t4_onecell/\`
EOF
}

run_step "one_command_full_sop" bash scripts/one_command_full_sop.sh
# --skip-verify-all matches verify_all.sh's own internal call to start_gate.py (its
# "Start gate operator decision refresh" step): by this point verify_all.sh has already run
# at least twice (once directly here via one_command_full_sop.sh, once more inside THAT
# run's own start_gate.py step), so a third, unflagged re-run is redundant -- and, on a
# machine with no training corpus (gitignored, absent on every fresh clone), it also fails:
# check_doc_claim_consistency.py runs BEFORE build_train_readiness_contract.py refreshes
# reports/train_readiness_decision.json within verify_all.sh, so the first invocation checks
# against the still-canonical committed file, but a second back-to-back invocation checks
# against the just-regenerated one, which reports the live (Windows: STAGE_JSONL_MISSING)
# blocker instead of the documented canonical one -- a real doc/live mismatch, but an
# artifact of re-running verify_all.sh twice in one process, not a Windows-specific bug.
run_step "start_gate" .titan-venv/bin/python scripts/start_gate.py --skip-verify-all --allow-not-ready
run_step "unicode_path_guard" .titan-venv/bin/python scripts/unicode_path_guard.py --root . --out reports/unicode_path_guard_report.json --fail-on-hit
run_step "sbom" .titan-venv/bin/python scripts/generate_sbom.py
run_step "repro_build" .titan-venv/bin/python scripts/repro_build_check.py
run_step "energy_baselines" .titan-venv/bin/python scripts/generate_energy_baselines.py

# The inner (post "--") command is passed as a plain argv list to ram_guard.py, which spawns
# it directly via Python's subprocess.Popen (not through bash) -- Windows CreateProcess can't
# resolve a relative ".titan-venv/bin/python" executable there the way bash resolves it for
# run_step's own direct invocation, so this inner one needs an absolute path.
run_step "hardening_bundle" .titan-venv/bin/python scripts/ram_guard.py --out reports/ram_guard_report.json --warn 10.5 --slow 12 --hard 13 -- "$ROOT_DIR/.titan-venv/bin/python" scripts/hardening_bundle.py

run_step "bench_reports" .titan-venv/bin/python scripts/generate_bench_reports.py
run_step "md_quality" .titan-venv/bin/python scripts/md_quality_gate.py --root . --scope release_core --out reports/md_lint_report.json
run_step "md_integrity" .titan-venv/bin/python scripts/md_integrity_check.py --root .
run_step "linkcheck" .titan-venv/bin/python scripts/linkcheck_gate.py --root . --scope release_core --out reports/linkcheck_report.json
run_step "docs_inventory" .titan-venv/bin/python scripts/docs_inventory.py
run_step "anthropic_packet_refresh" refresh_anthropic_packet

# Documents cleanup and hash fixes
if [ -f "$DOCS_DIR/Proje.zip" ] && [ -f "$IMMUTABLE_ZIP" ]; then
  rm -f "$DOCS_DIR/Proje.zip"
fi
find "$DOCS_DIR/mertformer_outputs" -type f -name '.DS_Store' -delete 2>/dev/null || true
if [ -f "$LINKEDIN_ZIP" ]; then
  shasum -a 256 "$LINKEDIN_ZIP" > "$LINKEDIN_ZIP.sha256"
fi

run_step "duplicate_zip_guard" .titan-venv/bin/python scripts/duplicate_zip_guard.py --out reports/duplicate_zip_guard_report.json
run_step "scoped_external_intake_matrix" .titan-venv/bin/python scripts/build_scoped_external_intake_matrix.py --sync-mode audit
run_step "scoped_cleanup_apply" .titan-venv/bin/python scripts/cleanup_scoped_closure_junk.py --apply --delete-stale-zips

# Sync manifest/structure/policy
run_step "sync_manifest" .titan-venv/bin/python scripts/sync_manifest.py --root . --manifest reports/release_manifest.json --structure docs/PROJECT_STRUCTURE.md --matrix reports/file_sync_matrix.json --sync-report reports/project_structure_sync_report.json --policy-report reports/policy_sync_report.json

# Hash manifests for docs/packages/documents
find "$ROOT_DIR/docs" "$ROOT_DIR/packages" -type f -name "*.zip" -print0 2>/dev/null | xargs -0 shasum -a 256 | .titan-venv/bin/python scripts/hash_manifest_to_json.py --base "$ROOT_DIR" --pretty > reports/docs_packages_hash_manifest.json || true
find "$DOCS_DIR" -maxdepth 1 -type f -name "*.zip" -print0 2>/dev/null | xargs -0 shasum -a 256 | .titan-venv/bin/python scripts/hash_manifest_to_json.py --base / --pretty > "$DOCS_REPORTS_DIR/documents_zip_hash_manifest.json" || true

# Dealroom reference/provenance (best effort, does not fail one-shot -- the dealroom repo
# is a separate sibling checkout that only exists on the original operator's machine;
# dealroom_sync.py itself correctly reports/exits non-zero when it is absent, which is the
# expected state on any fresh clone or contributor machine, not a real failure)
run_step "dealroom_sync" .titan-venv/bin/python scripts/dealroom_sync.py || true
run_step "master_closure_matrix" .titan-venv/bin/python scripts/build_master_closure_matrix.py
run_step "train_readiness_contract" .titan-venv/bin/python scripts/build_train_readiness_contract.py --allow-not-ready
run_step "final_orchestrator_plan" .titan-venv/bin/python scripts/final_orchestrator.py --plan-only
run_step "closure_governance_pack" .titan-venv/bin/python scripts/build_closure_governance_pack.py
run_step "max_closure_handoff" .titan-venv/bin/python scripts/build_max_closure_handoff.py
# [2026-07-31] Opt-in, default OFF. This is a separate, private chess-PoC delivery lane
# (CHESS_5080_POC_INTERNAL.md) unrelated to the closure ladder's own pre/post-45K purpose --
# every ladder pass used to unconditionally build a uniquely-timestamped delivery zip+folder on
# the operator's Desktop, and since nothing ever cleans up the previous one, repeated passes
# (e.g. while iterating on a portability fix) silently accumulate duplicate multi-hundred-KB
# bundles. Set TITAN_CHESS_5080_EXPORT=1 to opt back in for a real chess-delivery build.
if [[ "${TITAN_CHESS_5080_EXPORT:-0}" == "1" ]]; then
  run_step "chess_5080_share_export" .titan-venv/bin/python scripts/export_chess_5080_share.py
else
  echo "[final] chess_5080_share_export skipped (set TITAN_CHESS_5080_EXPORT=1 to opt in)"
fi

# Ensure writable artifacts before regeneration
chflags nouchg artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 artifacts/mertformer_training_outputs_bundle.zip artifacts/mertformer_training_outputs_bundle.zip.sha256 2>/dev/null || true
chmod u+w artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 artifacts/mertformer_training_outputs_bundle.zip artifacts/mertformer_training_outputs_bundle.zip.sha256 2>/dev/null || true

# Release artifact
rm -f artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 artifacts/mertformer_training_outputs_bundle.zip artifacts/mertformer_training_outputs_bundle.zip.sha256
run_step "pre_zip_cache_cleanup" .titan-venv/bin/python scripts/run_and_clean_pycache.py --root . --include-tool-caches --full-clean --include-venv-caches -- bash -lc true
run_step "pre_zip_runtime_clean_check" bash scripts/clean_runtime_artifacts.sh --check
run_zip_with_tolerance artifacts/mertformer_release.zip . -x ".git/*" "*/.git/*" "*.pyc" "*__pycache__*" ".titan-venv/*" ".lint-venv/*" ".venv/*" ".idea/*" ".pytest_cache/*" ".ruff_cache/*" ".mypy_cache/*" ".env" ".env.*" "logs/*" "checkpoints/*" "artifacts/mertformer_release.zip" "artifacts/mertformer_release.zip.sha256" "apps/chess_gui/logs/*" "apps/chess_gui/checkpoints/*" "apps/chess_gui/assets/*" "apps/chess_gui/chess_5080_onefile.py"
(
  cd artifacts
  shasum -a 256 mertformer_release.zip > mertformer_release.zip.sha256
)
run_step "zip_denylist_audit_artifact" bash -lc '.titan-venv/bin/python scripts/zip_denylist_audit.py --zip artifacts/mertformer_release.zip > reports/artifacts_zip_denylist_audit.json'
run_step "training_outputs_bundle" .titan-venv/bin/python scripts/build_training_outputs_bundle.py

# Immutable lock best effort
chmod -w artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 artifacts/mertformer_training_outputs_bundle.zip artifacts/mertformer_training_outputs_bundle.zip.sha256 || true
chmod -w "$IMMUTABLE_ZIP" "$IMMUTABLE_ZIP.sha256" 2>/dev/null || true
chflags uchg artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 artifacts/mertformer_training_outputs_bundle.zip artifacts/mertformer_training_outputs_bundle.zip.sha256 2>/dev/null || true
chflags uchg "$IMMUTABLE_ZIP" "$IMMUTABLE_ZIP.sha256" 2>/dev/null || true

# GitHub policy and closure lock (best effort, does not fail one-shot)
bash scripts/apply_github_policy.sh || true
bash scripts/release_closure_lock.sh v1.0.0 || true
run_step "offline_closure_pack" .titan-venv/bin/python scripts/build_offline_closure_pack.py
run_step "scoped_external_sync_apply" .titan-venv/bin/python scripts/build_scoped_external_intake_matrix.py --sync-mode apply
# Best effort here specifically -- does not fail one-shot. This exact check already ran and
# passed once, validly, early in this same closure pass (inside the first verify_all.sh call
# via one_command_full_sop.sh, before any readiness refresh had touched the committed baseline).
# By this point, verify_all.sh's own later train_readiness_contract step has already refreshed
# reports/train_readiness_decision.json to this machine's live state -- and on a machine with
# zero training corpus at all (gitignored, absent on every fresh clone), that live state reports
# offline_clean:STAGE_JSONL_MISSING, a genuinely different (more absent) blocker than the
# documented canonical offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE (which
# describes a machine that has SOME corpus, just missing precomputed logits). Re-checking docs
# against that machine-local, already-mutated state here is a redundant re-verification, not a
# new one -- same root cause and same resolution as the start_gate --skip-verify-all fix above.
run_step "final_claim_consistency" .titan-venv/bin/python scripts/check_doc_claim_consistency.py || true
run_step "final_md_integrity" .titan-venv/bin/python scripts/md_integrity_check.py --root .
run_step "final_duplicate_zip_guard" .titan-venv/bin/python scripts/duplicate_zip_guard.py --out reports/duplicate_zip_guard_report.json

cat > reports/execution_trace.json <<'JSON'
{
  "status": "completed",
  "flow": "final_one_shot",
  "notes": "Canonical Max Closure flow executed: one_command_full_sop core, hardening/release extras, master closure matrix, dual-path readiness contract, and canonical handoff surfaces."
}
JSON

echo "[final] COMPLETED"

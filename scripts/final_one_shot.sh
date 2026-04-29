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
  local dir="$ROOT_DIR/applications/anthropic"
  local zip_path="$dir/mertformer_anthropic_packet_20260419.zip"
  local sha_path="$dir/mertformer_anthropic_packet_20260419.zip.sha256"
  local pointer_path="$dir/PACKET_POINTER_20260419.md"
  local files=(
    "README.md"
    "application_strategy.md"
    "project_summary.md"
    "measured_evidence_summary.md"
    "why_anthropic_science_of_scaling.md"
    "science_of_scaling_cv_seed.md"
    "tokens_variant_notes.md"
    "strongest_stories.md"
    "interview_prep.md"
    "performance_engineer_fallback.md"
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

- Zip: \`applications/anthropic/$(basename "$zip_path")\`
- SHA256 sidecar: \`applications/anthropic/$(basename "$sha_path")\`
- SHA256: \`$sha\`
- Size: \`$size_bytes\` bytes
- Contents: 10 markdown application files

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
run_step "start_gate" .titan-venv/bin/python scripts/start_gate.py
run_step "unicode_path_guard" .titan-venv/bin/python scripts/unicode_path_guard.py --root . --out reports/unicode_path_guard_report.json --fail-on-hit
run_step "sbom" .titan-venv/bin/python scripts/generate_sbom.py
run_step "repro_build" .titan-venv/bin/python scripts/repro_build_check.py
run_step "energy_baselines" .titan-venv/bin/python scripts/generate_energy_baselines.py

run_step "hardening_bundle" .titan-venv/bin/python scripts/ram_guard.py --out reports/ram_guard_report.json --warn 10.5 --slow 12 --hard 13 -- .titan-venv/bin/python scripts/hardening_bundle.py

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

# Dealroom reference/provenance
run_step "dealroom_sync" .titan-venv/bin/python scripts/dealroom_sync.py
run_step "master_closure_matrix" .titan-venv/bin/python scripts/build_master_closure_matrix.py
run_step "train_readiness_contract" .titan-venv/bin/python scripts/build_train_readiness_contract.py --allow-not-ready
run_step "final_orchestrator_plan" .titan-venv/bin/python scripts/final_orchestrator.py --plan-only
run_step "closure_governance_pack" .titan-venv/bin/python scripts/build_closure_governance_pack.py
run_step "max_closure_handoff" .titan-venv/bin/python scripts/build_max_closure_handoff.py
run_step "chess_5080_share_export" .titan-venv/bin/python scripts/export_chess_5080_share.py

# Ensure writable artifacts before regeneration
chflags nouchg artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 artifacts/mertformer_training_outputs_bundle.zip artifacts/mertformer_training_outputs_bundle.zip.sha256 2>/dev/null || true
chmod u+w artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 artifacts/mertformer_training_outputs_bundle.zip artifacts/mertformer_training_outputs_bundle.zip.sha256 2>/dev/null || true

# Release artifact
rm -f artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 artifacts/mertformer_training_outputs_bundle.zip artifacts/mertformer_training_outputs_bundle.zip.sha256
run_step "pre_zip_cache_cleanup" .titan-venv/bin/python scripts/run_and_clean_pycache.py --root . --include-tool-caches --full-clean --include-venv-caches -- bash -lc true
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
run_step "final_claim_consistency" .titan-venv/bin/python scripts/check_doc_claim_consistency.py
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

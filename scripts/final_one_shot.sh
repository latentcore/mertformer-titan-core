#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p reports artifacts telemetry docs policy /Users/mertyunlu/Documents/reports

run_step() {
  local name="$1"; shift
  echo "[final] $name"
  "$@"
}

run_step "start_gate" .titan-venv/bin/python scripts/start_gate.py
run_step "unicode_path_guard" .titan-venv/bin/python scripts/unicode_path_guard.py --root . --out reports/unicode_path_guard_report.json --fail-on-hit
run_step "sbom" .titan-venv/bin/python scripts/generate_sbom.py
run_step "repro_build" .titan-venv/bin/python scripts/repro_build_check.py
run_step "energy_baselines" .titan-venv/bin/python scripts/generate_energy_baselines.py

run_step "hardening_bundle" .titan-venv/bin/python scripts/ram_guard.py --out reports/ram_guard_report.json --warn 10.5 --slow 12 --hard 13 -- .titan-venv/bin/python scripts/hardening_bundle.py

run_step "bench_reports" .titan-venv/bin/python scripts/generate_bench_reports.py
run_step "md_quality" .titan-venv/bin/python scripts/md_quality_gate.py --root . --scope release_core --out reports/md_lint_report.json
run_step "linkcheck" .titan-venv/bin/python scripts/linkcheck_gate.py --root . --scope release_core --out reports/linkcheck_report.json
run_step "docs_inventory" .titan-venv/bin/python scripts/docs_inventory.py

# Documents cleanup and hash fixes
if [ -f /Users/mertyunlu/Documents/Proje.zip ] && [ -f /Users/mertyunlu/Documents/Proje_immutable_20260301_d22272ee9281e978.zip ]; then
  rm -f /Users/mertyunlu/Documents/Proje.zip
fi
find /Users/mertyunlu/Documents/mertformer_outputs -type f -name '.DS_Store' -delete 2>/dev/null || true
if [ -f /Users/mertyunlu/Documents/mertformer_outputs_LINKEDIN_run_20260220_175540.zip ]; then
  shasum -a 256 /Users/mertyunlu/Documents/mertformer_outputs_LINKEDIN_run_20260220_175540.zip > /Users/mertyunlu/Documents/mertformer_outputs_LINKEDIN_run_20260220_175540.zip.sha256
fi

run_step "duplicate_zip_guard" .titan-venv/bin/python scripts/duplicate_zip_guard.py --out reports/duplicate_zip_guard_report.json

# Sync manifest/structure/policy
run_step "sync_manifest" .titan-venv/bin/python scripts/sync_manifest.py --root . --manifest reports/release_manifest.json --structure docs/PROJECT_STRUCTURE.md --matrix reports/file_sync_matrix.json --sync-report reports/project_structure_sync_report.json --policy-report reports/policy_sync_report.json

# Hash manifests for docs/packages/documents
find "$ROOT_DIR/docs" "$ROOT_DIR/packages" -type f -name "*.zip" -print0 2>/dev/null | xargs -0 shasum -a 256 | .titan-venv/bin/python scripts/hash_manifest_to_json.py --base "$ROOT_DIR" --pretty > reports/docs_packages_hash_manifest.json || true
find /Users/mertyunlu/Documents -maxdepth 1 -type f -name "*.zip" -print0 2>/dev/null | xargs -0 shasum -a 256 | .titan-venv/bin/python scripts/hash_manifest_to_json.py --base / --pretty > /Users/mertyunlu/Documents/reports/documents_zip_hash_manifest.json || true

# Dealroom reference/provenance
run_step "dealroom_sync" .titan-venv/bin/python scripts/dealroom_sync.py

# Demo bundle
run_step "demo_bundle" .titan-venv/bin/python scripts/generate_demo_bundle.py

# Release artifact
zip -r artifacts/mertformer_release.zip . -x ".git/*" "*/.git/*" "*.pyc" "*__pycache__*" ".titan-venv/*" ".lint-venv/*" ".venv/*" ".idea/*" ".pytest_cache/*" ".ruff_cache/*" ".mypy_cache/*" "artifacts/mertformer_release.zip" "artifacts/mertformer_release.zip.sha256"
shasum -a 256 artifacts/mertformer_release.zip > artifacts/mertformer_release.zip.sha256

# Immutable lock best effort
chmod -w artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 reports/demo_checksum.sha256 || true
chmod -w /Users/mertyunlu/Documents/Proje_immutable_20260301_d22272ee9281e978.zip /Users/mertyunlu/Documents/Proje_immutable_20260301_d22272ee9281e978.zip.sha256 2>/dev/null || true
chflags uchg artifacts/mertformer_release.zip artifacts/mertformer_release.zip.sha256 reports/demo_checksum.sha256 2>/dev/null || true
chflags uchg /Users/mertyunlu/Documents/Proje_immutable_20260301_d22272ee9281e978.zip /Users/mertyunlu/Documents/Proje_immutable_20260301_d22272ee9281e978.zip.sha256 2>/dev/null || true

# GitHub policy and closure lock (best effort, does not fail one-shot)
bash scripts/apply_github_policy.sh || true
bash scripts/release_closure_lock.sh v1.0.0 || true

cat > reports/execution_trace.json <<'EOF'
{
  "status": "completed",
  "flow": "final_one_shot",
  "notes": "All closure phases executed with fail-fast for critical gates and best-effort for external governance APIs."
}
EOF

echo "[final] COMPLETED"

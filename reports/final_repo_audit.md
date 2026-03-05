# Final Repo Audit

Generated UTC: 2026-03-04T01:49:00Z

## Scope
- One-shot closure workflow executed via `scripts/final_one_shot.sh`.
- Main repo release branch: `codex/release-v1-final`.
- Dealroom repo release branch: `codex/release-v1-final-dealroom`.

## Gates
- `scripts/secret_scan.py`: pass
- `scripts/check_57_matrix.py`: pass (`all_green=true`, no in-scope pending)
- `scripts/verify_all.sh`: pass
- Unicode path guard: pass
- Duplicate zip guard: pass
- Manifest sync guard: pass

## Security / Hardening Outputs
- `reports/sbom.cdx.json`
- `reports/static_analysis_report.json`
- `reports/sanitizer_report.json`
- `reports/kernel_fuzz_report.json`
- `reports/determinism_report.json`
- `reports/differential_backend_report.json`
- `reports/license_gate_report.json`

## Release Artifacts
- `artifacts/mertformer_release.zip`
- `artifacts/mertformer_release.zip.sha256`
- `reports/release_manifest.json`

## Provenance
- Main signed commit/tag: `v1.0.0`
- Main PR merges: `#6`, `#7`, `#8`, `#9`, `#10`, `#11`, `#12`
- Dealroom signed commit/tag: `v1.0.0-dealroom`
- Dealroom PR merge: `#2`
- Ownership bundle: `reports/ownership_proof_bundle.json`

## Post-Closure CI Hotfix (2026-03-04)
- Root cause: CI `verify_all` stage re-bootstrapped `.titan-venv` when `TITAN_PYTHON` was set to a command name, triggering duplicate dependency install and disk exhaustion.
- Fix 1: `scripts/verify_all.sh` now accepts both executable paths and PATH-resolvable command names for `TITAN_PYTHON`.
- Fix 2: `.github/workflows/ci.yml` sets `TITAN_PYTHON=python3` in `Verify all` to reuse the existing setup-python environment.

## Notes
- GitHub branch-protection API update returned HTTP 403 on account plan limit; see `reports/github_policy_report.json`.
- Real 2.64B training was not started in this closure pass.

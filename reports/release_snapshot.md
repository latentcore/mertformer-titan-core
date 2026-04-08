# Release Snapshot (Review-Ready)

This file is a human-readable snapshot of the repository at a review point-in-time.

## Snapshot

- Snapshot freshness: see `reports/one_command_full_sop_summary.md` for the latest closure run window.
- Current Git SHA (local): run `git rev-parse --short HEAD` in this working tree.
- Baseline Python: 3.11 (see `repro/python.md`)
- Default mode: offline-first (`TITAN_OFFLINE=1`)

## Verified (Run)

Runbook:

```bash
bash scripts/bootstrap_venv.sh
bash scripts/verify_all.sh
```

Expected outputs:
- Secret scan: PASS
- Pytest: PASS (`203 passed, 3 skipped`)
- Preflight (offline): PASS
- Operator gate (safe, offline): PASS

Additional spot checks:

```bash
TITAN_OFFLINE=1 bash run.sh --test
.titan-venv/bin/python scripts/train_smoke.py --cleanup
```

## Key Docs

- Engineering audit: `reports/codex_deep_audit_EN.md`, `reports/codex_deep_audit_DE.md`, `reports/codex_deep_audit_TR.md`
- Turkish counterparts for audits are pointer files: `reports/codex_deep_audit_EN_TR.md`, `reports/codex_deep_audit_DE_TR.md` (canonical TR content: `reports/codex_deep_audit_TR.md`)
- Repo-side closure scorecard: `reports/repo_closure_scorecard.md`, `reports/repo_closure_scorecard.json`
- Closure freeze and known limits: `reports/final_master_plan_freeze.md`, `reports/known_limits_v1.md`
- Maintenance, quality, and verification contract: `reports/support_maintenance_policy.md`, `reports/quality_gate_matrix.md`, `reports/test_verification_matrix.md`
- ADR chain: `reports/adr_index.md`, `adr/ADR-0001-source-of-truth-and-claim-boundary.md`
- Verified vs Target matrix: `reports/verified_matrix.md`, `reports/verified_matrix_TR.md`
- External review checklist: `reports/review_checklist.md`, `reports/review_checklist_TR.md`
- Final sync matrix: `reports/final_sync_matrix.md`, `reports/final_sync_matrix_TR.md`
- Efficiency convergence analysis: `reports/efficiency_convergence_analysis.md`, `reports/efficiency_convergence_analysis_TR.md`
- Usage guide: `USAGE_GUIDE.md`, `USAGE_GUIDE_TR.md`
- Demo proof clip: `assets/snake_demo_proof.mp4`
- Clean-room verification: `reports/cleanroom_verification.md`
- Dataset compliance: `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`, `datasets/inventory*`

## Release Artifacts (Desktop)

- `MertFormer_Titan_OnyxStorm_v2.0_B30_Release.zip`
- `MertFormer_Titan_OnyxStorm_v2.0_B30_Locked.secure.age`
- Locked artifact status: `skipped (expected: AGE_RECIPIENT_FILE missing)`
- SHA-256:
  - `39d2272b1f8e2a25bc0cb33518e3896ef4c7a03703819477b590e6d30bc51585` (`MertFormer_Titan_OnyxStorm_v2.0_B30_Release.zip`)
  - `SKIPPED` (`MertFormer_Titan_OnyxStorm_v2.0_B30_Locked.secure.age`)

## Known Gates / Blockers

- Dataset compliance gate:
  - 🟡 Licenses verified for core training datasets; optional/demo entries in `datasets/LICENSES*.md` remain `TBD` until explicitly enabled
  - ✅ Snapshot registry recorded in `datasets/hashes.json` (pinned revisions + manifest fingerprints)
- Remaining work is operational and post-training:
  - Run production training on target hardware
  - Generate benchmark reports from the produced checkpoints

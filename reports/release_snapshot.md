# Release Snapshot (Review-Ready)

This file is a human-readable snapshot of the repository at a review point-in-time.

## Snapshot

- Date (local): 2026-03-04
- Base Git SHA (short): `6f41827`
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
- Pytest: PASS (`111 passed, 3 skipped`)
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
- Verified vs Target matrix: `reports/verified_matrix.md`, `reports/verified_matrix_TR.md`
- External review checklist: `reports/review_checklist.md`, `reports/review_checklist_TR.md`
- Final sync matrix: `reports/final_sync_matrix.md`, `reports/final_sync_matrix_TR.md`
- Efficiency convergence analysis: `reports/efficiency_convergence_analysis.md`, `reports/efficiency_convergence_analysis_TR.md`
- Usage guide: `USAGE_GUIDE.md`, `USAGE_GUIDE_TR.md`
- Demo proof clip: `assets/snake_demo_proof.mp4`
- Clean-room verification: `reports/cleanroom_verification.md`
- Dataset compliance: `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`, `datasets/inventory*`

## Release Artifacts (Desktop)

- `MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip`
- `MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`
- Locked artifact status: `skipped (expected: AGE_RECIPIENT_FILE missing)`
- SHA-256:
  - `aa02cf628b66b63c1e678aa27a46c8a5908ad94f3e689c828f280287a2573619` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip`)
  - `SKIPPED` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`)

## Known Gates / Blockers

- Dataset compliance gate:
  - ✅ Licenses verified (no `TBD` in `datasets/LICENSES*.md`)
  - ✅ Snapshot registry recorded in `datasets/hashes.json` (pinned revisions + manifest fingerprints)
- Remaining work is operational and post-training:
  - Run production training on target hardware
  - Generate benchmark reports from the produced checkpoints

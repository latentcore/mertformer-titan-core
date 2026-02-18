# Release Snapshot (Review-Ready)

This file is a human-readable snapshot of the repository at a review point-in-time.

## Snapshot

- Date (local): 2026-02-18
- Base Git SHA (short): `git rev-parse --short HEAD`
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
- Pytest: PASS (`58 passed, 3 skipped`)
- Preflight (offline): PASS
- Operator gate (safe, offline): PASS

Additional spot checks:

```bash
TITAN_OFFLINE=1 bash run.sh --test
.titan-venv/bin/python scripts/train_smoke.py --cleanup
```

## Key Docs

- Engineering audit: `reports/codex_deep_audit_EN.md`, `reports/codex_deep_audit_DE.md`, `reports/codex_deep_audit_TR.md`
- Turkish counterparts for audits: `reports/codex_deep_audit_EN_TR.md`, `reports/codex_deep_audit_DE_TR.md`
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
- `MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age`
- `MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`
- SHA-256:
  - `3efc60d316d7caccdd65786c74a667cf74ef3687cd8d8be7e19e4088e6c3880e` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip`)
  - `2fc620ae2a122626992e302a313111d8ab1781b1acaa7b7613e9fbf5a88183bd` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age`)
  - `9ec43d6bf41c251346e7337068f097b2d8be4481bebfe652813108c091db01a2` (`MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`)

## Known Gates / Blockers

- Dataset compliance gate:
  - ✅ Licenses verified (no `TBD` in `datasets/LICENSES*.md`)
  - ✅ Snapshot registry recorded in `datasets/hashes.json` (pinned revisions + manifest fingerprints)
- Remaining work is operational and post-training:
  - Run production training on target hardware
  - Generate benchmark reports from the produced checkpoints

# Release Snapshot (Review-Ready)

This file is a human-readable snapshot of the repository at a review point-in-time.

## Snapshot

- Date (local): 2026-02-06
- Base Git SHA (short): 63e47a2
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
- Pytest: PASS (`21 passed, 4 skipped`)
- Preflight (offline): PASS
- Operator gate (safe, offline): PASS

Additional spot checks:

```bash
TITAN_OFFLINE=1 bash run.sh --test
.titan-venv/bin/python scripts/train_smoke.py --cleanup
```

## Key Docs

- Engineering audit: `reports/codex_deep_audit_DE.md`, `reports/codex_deep_audit_TR.md`
- Verified vs Target matrix: `reports/verified_matrix.md`, `reports/verified_matrix_TR.md`
- External review checklist: `reports/review_checklist.md`, `reports/review_checklist_TR.md`
- Dataset compliance: `datasets/SOURCES*.md`, `datasets/LICENSES*.md`, `datasets/hashes.json`, `datasets/inventory*`

## Known Gates / Blockers

- Production training is blocked until:
  - All training datasets have verified licenses (no `TBD` in `datasets/LICENSES*.md`)
  - Snapshot metadata + hashes are recorded in `datasets/hashes.json`

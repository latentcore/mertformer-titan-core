# Final Sync Matrix (Build 30)

This file is the final consistency ledger for documentation, version labels, and verification runs.

## 1) EN/TR Markdown Pairing

Rule: every tracked `X.md` has `X_TR.md` pair where applicable.

| Check | Result |
| --- | --- |
| Missing pairs | 0 |
| Orphan `_TR.md` files | 1 (`reports/codex_deep_audit_TR.md`, intentional canonical TR audit) |
| Deep audit pointer policy | EN_TR + DE_TR are pointer files to `reports/codex_deep_audit_TR.md` |
| Added in this final pass | `reports/pilot_readiness_kit_TR.md`, `reports/pilot_offer_packages_TR.md`, `reports/sales_funnel_90d_TR.md`, `reports/drone_sitl_demo_TR.md`, `reports/cleanroom_verification_TR.md`, `reports/go_status_matrix_TR.md` |

## 2) Build Label Consistency

| Scope | Result |
| --- | --- |
| User-visible version references | Build 30 aligned |
| Legacy references in active user-facing scripts | normalized |
| Lineage references | retained only where clearly marked roadmap/history |

## 3) Verification Runs (Release Candidate)

| Command | Status |
| --- | --- |
| `.titan-venv/bin/python -m pytest -q` | PASS (`108 passed, 4 skipped`) |
| `bash scripts/verify_all.sh` | PASS |
| `TITAN_OFFLINE=1 bash run.sh --test` | PASS |
| `.titan-venv/bin/python scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl` | PASS |

## 4) Demo Proof

| Artifact | Status |
| --- | --- |
| `assets/snake_demo_proof.mp4` | generated |
| `snake_demo.py` auto-restart + telemetry | verified |
| README links | updated |

## 5) Packaging Gates

| Gate | Status |
| --- | --- |
| Clean zip excludes venv/cache/log/.env | PASS |
| `.age` package gate | PASS or SKIPPED (expected when `AGE_RECIPIENT_FILE` is missing) |
| SHA-256 created for all 2 packages | PASS |

## 6) Release Identifiers

- Final git SHA (main): `git rev-parse --short HEAD`
- Release zip: `MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip`
- Locked secure package: `MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age`
- SHA-256 registry: see `reports/release_snapshot.md` (Release Artifacts section).

## 7) Portable Training Readiness (Build30)

| Check | Result |
| --- | --- |
| `run.sh --train-ready` strict mode | Added (`strict_online_training_readiness`) |
| `run.sh` profile contract | Added (`TITAN_PROFILE=stable|max_arch`) |
| Teacher hard-fail policy | Enabled (`require_gated_teacher=true`) |
| Curriculum single source ratios | Enabled (`config.curriculum_stage_ratios`) |
| Golden assertion scorer | Added (`scripts/golden_score.py`) |
| Readiness manifest/report outputs | Added (`reports/training_readiness_manifest.json`, `logs/preflight/train_ready_status.json`) |
